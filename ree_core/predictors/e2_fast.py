"""
E2 Fast Predictor Implementation

E2 predicts immediate observations and short-horizon state, generating
candidate trajectories for E3 selection. It operates on the affordance/action
latent (z_A) derived from the shared sensory latent (z_S).

Key responsibilities:
- Generate N candidate trajectories via forward rollouts
- Support multiple generation strategies (MPC, CEM, beam search)
- Predict short-horizon consequences of action sequences
- Provide inputs to E3 for trajectory evaluation

E2 is the "fast" predictor operating at shorter timescales than E1.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E2Config
from ree_core.latent.stack import LatentState


@dataclass
class Trajectory:
    """A candidate trajectory through latent space.

    Attributes:
        states: List of latent states along the trajectory
        actions: Action sequence that generated this trajectory
        predicted_observations: Predicted observations at each step
        total_length: Number of steps in the trajectory
    """
    states: List[torch.Tensor]      # List of [batch, latent_dim] tensors
    actions: torch.Tensor           # [batch, horizon, action_dim]
    predicted_observations: Optional[List[torch.Tensor]] = None
    harm_predictions: Optional[torch.Tensor] = None  # [batch, horizon]

    @property
    def total_length(self) -> int:
        return len(self.states)

    def get_final_state(self) -> torch.Tensor:
        """Get the final state of the trajectory."""
        return self.states[-1]

    def get_state_sequence(self) -> torch.Tensor:
        """Stack all states into a tensor [batch, horizon, latent_dim]."""
        return torch.stack(self.states, dim=1)


class E2FastPredictor(nn.Module):
    """
    E2 Fast Predictor for short-horizon trajectory generation.

    Generates candidate sensorimotor futures by rolling the world model
    forward from the current latent state. These candidates are then
    evaluated by E3 for selection.

    Architecture:
    - Transition model: predicts next latent state given current state and action
    - Observation model: predicts observations from latent state
    - Harm model: predicts potential harm/degradation signals
    """

    def __init__(self, config: Optional[E2Config] = None):
        super().__init__()
        self.config = config or E2Config()

        # Transition model: z_{t+1} = f(z_t, a_t)
        self.transition = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )

        # Observation predictor: o_t = g(z_t)
        # Predicts what observations would result from a latent state
        self.observation_predictor = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )

        # Harm predictor: h_t = h(z_t, a_t)
        # Predicts potential harm/degradation from state-action pairs
        self.harm_predictor = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid()  # Harm in [0, 1]
        )

        # Action encoder for embedding discrete or continuous actions
        self.action_encoder = nn.Linear(self.config.action_dim, self.config.action_dim)

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the next latent state given current state and action.

        Args:
            current_state: Current latent state [batch, latent_dim]
            action: Action to take [batch, action_dim]

        Returns:
            Predicted next latent state [batch, latent_dim]
        """
        # Encode action
        action_embed = self.action_encoder(action)

        # Concatenate state and action
        state_action = torch.cat([current_state, action_embed], dim=-1)

        # Predict next state (residual connection)
        delta = self.transition(state_action)
        next_state = current_state + delta

        return next_state

    def predict_observation(self, state: torch.Tensor) -> torch.Tensor:
        """Predict observation from latent state."""
        return self.observation_predictor(state)

    def predict_harm(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict potential harm from state-action pair.

        This is used for ethical cost estimation during trajectory scoring.

        Args:
            state: Latent state [batch, latent_dim]
            action: Action [batch, action_dim]

        Returns:
            Predicted harm level [batch, 1] in [0, 1]
        """
        state_action = torch.cat([state, action], dim=-1)
        return self.harm_predictor(state_action)

    def rollout(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor
    ) -> Trajectory:
        """
        Roll out a trajectory given an action sequence.

        Args:
            initial_state: Starting latent state [batch, latent_dim]
            action_sequence: Sequence of actions [batch, horizon, action_dim]

        Returns:
            Trajectory containing states, actions, and predictions
        """
        batch_size = initial_state.shape[0]
        horizon = action_sequence.shape[1]

        states = [initial_state]
        observations = []
        harm_predictions = []

        current_state = initial_state

        for t in range(horizon):
            action = action_sequence[:, t, :]

            # Predict harm for this state-action
            harm = self.predict_harm(current_state, action)
            harm_predictions.append(harm)

            # Predict observation at current state
            obs = self.predict_observation(current_state)
            observations.append(obs)

            # Predict next state
            next_state = self.predict_next_state(current_state, action)
            states.append(next_state)
            current_state = next_state

        return Trajectory(
            states=states,
            actions=action_sequence,
            predicted_observations=observations,
            harm_predictions=torch.cat(harm_predictions, dim=-1)  # [batch, horizon]
        )

    def generate_random_actions(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate random action sequences for exploration."""
        return torch.randn(batch_size, horizon, self.config.action_dim, device=device)

    def generate_candidates_random(
        self,
        initial_state: torch.Tensor,
        num_candidates: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> List[Trajectory]:
        """
        Generate candidate trajectories using random shooting.

        Simple MPC-style generation: sample random action sequences
        and roll them out through the model.

        Args:
            initial_state: Starting latent state [batch, latent_dim]
            num_candidates: Number of candidates to generate
            horizon: Rollout horizon

        Returns:
            List of Trajectory objects
        """
        num_candidates = num_candidates or self.config.num_candidates
        horizon = horizon or self.config.rollout_horizon
        device = initial_state.device
        batch_size = initial_state.shape[0]

        candidates = []

        for _ in range(num_candidates):
            # Generate random action sequence
            actions = self.generate_random_actions(batch_size, horizon, device)

            # Roll out trajectory
            trajectory = self.rollout(initial_state, actions)
            candidates.append(trajectory)

        return candidates

    def generate_candidates_cem(
        self,
        initial_state: torch.Tensor,
        num_candidates: Optional[int] = None,
        horizon: Optional[int] = None,
        num_iterations: int = 3,
        elite_fraction: float = 0.2
    ) -> List[Trajectory]:
        """
        Generate candidate trajectories using Cross-Entropy Method.

        CEM iteratively refines action distributions by:
        1. Sample action sequences from current distribution
        2. Evaluate trajectories
        3. Fit new distribution to elite samples
        4. Repeat

        Args:
            initial_state: Starting latent state
            num_candidates: Number of candidates per iteration
            horizon: Rollout horizon
            num_iterations: Number of CEM iterations
            elite_fraction: Fraction of top samples to use for fitting

        Returns:
            List of Trajectory objects (final iteration)
        """
        num_candidates = num_candidates or self.config.num_candidates
        horizon = horizon or self.config.rollout_horizon
        device = initial_state.device
        batch_size = initial_state.shape[0]

        # Initialize action distribution (Gaussian)
        action_mean = torch.zeros(batch_size, horizon, self.config.action_dim, device=device)
        action_std = torch.ones(batch_size, horizon, self.config.action_dim, device=device)

        num_elite = max(1, int(num_candidates * elite_fraction))

        for iteration in range(num_iterations):
            # Sample action sequences
            all_actions = []
            all_trajectories = []

            for _ in range(num_candidates):
                # Sample from current distribution
                noise = torch.randn_like(action_mean)
                actions = action_mean + action_std * noise
                all_actions.append(actions)

                # Roll out trajectory
                trajectory = self.rollout(initial_state, actions)
                all_trajectories.append(trajectory)

            # Score trajectories (simple: minimize predicted harm)
            scores = []
            for traj in all_trajectories:
                # Lower harm = better score (negate for sorting)
                score = -traj.harm_predictions.sum(dim=-1).mean()
                scores.append(score)

            scores = torch.tensor(scores, device=device)

            # Select elite samples
            elite_indices = torch.argsort(scores, descending=True)[:num_elite]

            # Fit new distribution to elite samples
            elite_actions = torch.stack([all_actions[i] for i in elite_indices])
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0) + 1e-6  # Prevent collapse

        # Return final candidates
        return all_trajectories

    def generate_candidates(
        self,
        initial_state: torch.Tensor,
        method: str = "random",
        **kwargs
    ) -> List[Trajectory]:
        """
        Generate candidate trajectories using specified method.

        Args:
            initial_state: Starting latent state
            method: Generation method ("random" or "cem")
            **kwargs: Additional arguments for the generation method

        Returns:
            List of Trajectory objects
        """
        if method == "random":
            return self.generate_candidates_random(initial_state, **kwargs)
        elif method == "cem":
            return self.generate_candidates_cem(initial_state, **kwargs)
        else:
            raise ValueError(f"Unknown generation method: {method}")

    def forward(
        self,
        initial_state: torch.Tensor,
        num_candidates: Optional[int] = None
    ) -> List[Trajectory]:
        """
        Forward pass: generate candidate trajectories from initial state.

        This is the main interface for E3 to obtain trajectory candidates.
        """
        return self.generate_candidates(initial_state, method="random", num_candidates=num_candidates)
