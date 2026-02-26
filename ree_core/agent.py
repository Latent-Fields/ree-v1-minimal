"""
REE Agent Implementation

The main REE agent that integrates all architectural components:
- Latent Stack (L-space) for multi-timescale state representation
- E1 Deep Predictor for long-horizon context
- E2 Fast Predictor for trajectory generation
- E3 Trajectory Selector for ethical action selection
- Residue Field for persistent moral cost

The agent implements the canonical REE loop:
1. SENSE - Receive observations and harm signals
2. UPDATE - Update latent state across all depths
3. GENERATE - Generate candidate trajectories via E2
4. SCORE - Score trajectories with reality, ethics, and residue
5. SELECT - E3 selects trajectory under precision control
6. ACT - Execute next action from selected trajectory
7. RESIDUE - Update residue field if harm occurred
8. OFFLINE - Periodically perform offline integration (sleep)
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import REEConfig
from ree_core.latent.stack import LatentStack, LatentState
from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.trajectory.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField


@dataclass
class AgentState:
    """Complete state of the REE agent at a timestep."""
    latent_state: LatentState
    precision: float
    step: int
    harm_accumulated: float
    is_committed: bool


class REEAgent(nn.Module):
    """
    Reflective-Ethical Engine Agent.

    An agent that acts under uncertainty while retaining ethical
    continuity over time. Implements all REE architectural invariants:

    1. Ethical cost is PERSISTENT, not resettable
    2. Harm contributes to ethical cost via MIRROR MODELLING
    3. Moral residue CANNOT BE ERASED, only integrated
    4. Language cannot override embodied harm sensing
    5. Precision is ROUTED and DEPTH-SPECIFIC, not global
    6. Offline integration is REQUIRED for long-term viability

    Usage:
        agent = REEAgent.from_config(observation_dim=100, action_dim=4)
        obs = env.reset()
        for step in range(1000):
            action = agent.act(obs)
            obs, harm, done, info = env.step(action)
            agent.update_residue(harm)
            if done:
                break
    """

    def __init__(self, config: REEConfig):
        """
        Initialize the REE agent.

        Args:
            config: Complete REE configuration
        """
        super().__init__()
        self.config = config

        # Core components
        self.latent_stack = LatentStack(config.latent)
        self.e1 = E1DeepPredictor(config.e1)
        self.e2 = E2FastPredictor(config.e2)
        self.residue_field = ResidueField(config.residue)
        self.e3 = E3TrajectorySelector(config.e3, self.residue_field)

        # Observation encoder (maps raw obs to latent input)
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.latent.observation_dim, config.latent.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent.latent_dim, config.latent.observation_dim)
        )

        # Action decoder (maps latent action to environment action)
        self.action_decoder = nn.Linear(config.e2.action_dim, config.e2.action_dim)

        # State tracking
        self._current_latent: Optional[LatentState] = None
        self._step_count = 0
        self._experience_buffer: List[torch.Tensor] = []
        self._harm_this_episode = 0.0

        # Set device
        self.device = torch.device(config.device)

    @classmethod
    def from_config(
        cls,
        observation_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        **kwargs
    ) -> "REEAgent":
        """
        Create an REE agent from basic specifications.

        Args:
            observation_dim: Dimension of observations
            action_dim: Number of actions
            latent_dim: Dimension of latent space
            **kwargs: Additional config overrides

        Returns:
            Configured REEAgent instance
        """
        config = REEConfig.from_dims(observation_dim, action_dim, latent_dim)

        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return cls(config)

    def reset(self) -> None:
        """
        Reset agent state for a new episode.

        Note: This does NOT reset the residue field (invariant!).
        Only latent state and tracking variables are reset.
        """
        # Reset latent state
        self._current_latent = self.latent_stack.init_state(batch_size=1, device=self.device)

        # Reset E1 hidden state
        self.e1.reset_hidden_state()

        # Reset tracking (but NOT residue)
        self._step_count = 0
        self._harm_this_episode = 0.0

        # Keep experience buffer for cross-episode learning

    def sense(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Process raw observation (SENSE step of REE loop).

        Args:
            observation: Raw observation from environment

        Returns:
            Encoded observation ready for latent stack
        """
        # Ensure proper shape and device
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        observation = observation.to(self.device).float()

        # Encode observation
        encoded = self.obs_encoder(observation)

        return encoded

    def update_latent(self, encoded_obs: torch.Tensor) -> LatentState:
        """
        Update latent state (UPDATE step of REE loop).

        Args:
            encoded_obs: Encoded observation

        Returns:
            Updated latent state
        """
        # Update latent stack with new observation
        new_latent = self.latent_stack.encode(encoded_obs, self._current_latent)

        # Store in experience buffer for offline learning
        self._experience_buffer.append(new_latent.z_gamma.detach().clone())
        if len(self._experience_buffer) > 1000:
            self._experience_buffer = self._experience_buffer[-1000:]

        self._current_latent = new_latent
        return new_latent

    def generate_trajectories(
        self,
        latent_state: LatentState,
        num_candidates: Optional[int] = None
    ) -> List:
        """
        Generate candidate trajectories (GENERATE step of REE loop).

        Uses E2 fast predictor to roll out possible futures.

        Args:
            latent_state: Current latent state
            num_candidates: Number of trajectories to generate

        Returns:
            List of candidate Trajectory objects
        """
        # Get affordance latent (z_beta) for trajectory generation
        z_beta = self.latent_stack.get_affordance_latent(latent_state)

        # Get E1 prior for conditioning
        _, prior = self.e1(z_beta)

        # Generate candidates using E2
        candidates = self.e2.generate_candidates(
            z_beta,
            method="random",
            num_candidates=num_candidates
        )

        return candidates

    def select_action(
        self,
        candidates: List,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Select action from candidates (SCORE + SELECT steps of REE loop).

        E3 scores trajectories and selects one under precision control.

        Args:
            candidates: List of candidate trajectories
            temperature: Selection temperature

        Returns:
            Selected action tensor
        """
        # E3 selects best trajectory
        result = self.e3.select(candidates, temperature)

        return result.selected_action

    def act(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Complete REE action selection loop.

        Combines: SENSE -> UPDATE -> GENERATE -> SCORE -> SELECT -> ACT

        Args:
            observation: Raw observation from environment
            temperature: Selection temperature for exploration

        Returns:
            Action to execute
        """
        # 1. SENSE
        encoded_obs = self.sense(observation)

        # 2. UPDATE
        latent_state = self.update_latent(encoded_obs)

        # 3. GENERATE
        candidates = self.generate_trajectories(latent_state)

        # 4 & 5. SCORE + SELECT
        action = self.select_action(candidates, temperature)

        # 6. ACT (return action for environment)
        self._step_count += 1

        return action

    def update_residue(self, harm_signal: float) -> Dict[str, Any]:
        """
        Update residue field after action (RESIDUE step of REE loop).

        If harm occurred, accumulate residue at current latent state.
        This implements the REE invariant that residue cannot be erased.

        Args:
            harm_signal: Harm signal from environment (negative = harm)

        Returns:
            Dictionary of update metrics
        """
        metrics = {}

        if harm_signal < 0:  # Harm occurred
            harm_magnitude = abs(harm_signal)
            self._harm_this_episode += harm_magnitude

            # Get current latent state
            if self._current_latent is not None:
                # Accumulate residue at current location
                z_gamma = self._current_latent.z_gamma
                residue_metrics = self.residue_field.accumulate(z_gamma, harm_magnitude)
                metrics.update({f"residue_{k}": v for k, v in residue_metrics.items()})

            # Update E3 with harm information
            if self._current_latent is not None:
                e3_metrics = self.e3.post_action_update(
                    self._current_latent.z_gamma,
                    harm_occurred=True
                )
                metrics.update({f"e3_{k}": v for k, v in e3_metrics.items()})

        metrics["harm_signal"] = harm_signal
        metrics["harm_this_episode"] = self._harm_this_episode

        return metrics

    def compute_prediction_loss(self) -> torch.Tensor:
        """
        Return a differentiable E1 world-model prediction loss for training.

        Samples a random contiguous sequence from the experience buffer,
        runs E1's LSTM predictor, and returns the MSE loss between predictions
        and actual observed states.  Gradients flow through E1's transition_rnn,
        output_proj, and prior_generator weights.

        The inference hidden state is saved and restored so that online acting
        is not disturbed by the training replay.

        Returns:
            Scalar loss tensor with grad_fn attached.  Returns a zero-loss
            tensor (still differentiable via E1 params) if the buffer is too
            short to form a sequence.
        """
        # Differentiable zero anchored to E1 params (safe fallback)
        zero_loss = next(self.e1.parameters()).sum() * 0.0

        if len(self._experience_buffer) < 2:
            return zero_loss

        buf_len = len(self._experience_buffer)
        horizon = self.e1.config.prediction_horizon
        # Need at least 2 entries: initial state + 1 target
        max_start = max(1, buf_len - 1)
        start_idx = int(torch.randint(0, max_start, (1,)).item())
        end_idx = min(start_idx + horizon + 1, buf_len)

        if end_idx - start_idx < 2:
            return zero_loss

        # Buffer items have shape [1, dim] (batch dim = 1).  Squeeze to [dim],
        # stack into [seq_len, dim], then add batch dim → [1, seq_len, dim].
        sequence = torch.stack(
            [x.squeeze(0) for x in self._experience_buffer[start_idx:end_idx]]
        ).unsqueeze(0)  # [1, seq_len, dim]

        # Temporarily reset hidden state for a clean replay pass
        saved_hidden = self.e1._hidden_state
        self.e1.reset_hidden_state()

        initial_state = sequence[:, 0, :]          # [1, dim] detached — fine as input
        horizon_len = sequence.shape[1] - 1
        predictions = self.e1.predict_long_horizon(initial_state, horizon=horizon_len)
        targets = sequence[:, 1:, :]               # [1, horizon_len, dim] detached

        loss = F.mse_loss(predictions[:, :targets.shape[1], :], targets)

        # Restore inference hidden state (predict_long_horizon stored detached end-state)
        self.e1._hidden_state = saved_hidden

        return loss

    def act_with_log_prob(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0
    ):
        """
        Action selection with policy-gradient bookkeeping.

        Identical to act() but also returns the log-probability of the
        selected trajectory so the trainer can compute a REINFORCE loss.

        Args:
            observation: Raw observation from environment
            temperature: Selection temperature

        Returns:
            (action, log_prob) — log_prob is a scalar tensor connected to
            the computation graph through E3's scorer weights.
        """
        encoded_obs = self.sense(observation)
        latent_state = self.update_latent(encoded_obs)
        candidates = self.generate_trajectories(latent_state)
        result = self.e3.select(candidates, temperature)
        self._step_count += 1
        return result.selected_action, result.log_prob

    def offline_integration(self) -> Dict[str, float]:
        """
        Perform offline integration (OFFLINE/SLEEP step of REE loop).

        This periodic process:
        - Improves the world model via replay
        - Integrates and contextualizes residue
        - Recalibrates precision and option space

        Should be called periodically (e.g., every N steps).

        Returns:
            Dictionary of integration metrics
        """
        metrics = {}

        # E1 integrates experience
        if len(self._experience_buffer) > 10:
            e1_metrics = self.e1.integrate_experience(self._experience_buffer)
            metrics.update({f"e1_{k}": v for k, v in e1_metrics.items()})

        # Residue field integration (contextualization)
        residue_metrics = self.residue_field.integrate()
        metrics.update({f"residue_{k}": v for k, v in residue_metrics.items()})

        return metrics

    def should_integrate(self) -> bool:
        """Check if it's time for offline integration."""
        return self._step_count % self.config.offline_integration_frequency == 0

    def get_state(self) -> AgentState:
        """Get complete agent state for monitoring."""
        return AgentState(
            latent_state=self._current_latent,
            precision=self.e3.current_precision,
            step=self._step_count,
            harm_accumulated=self._harm_this_episode,
            is_committed=self.e3._committed_trajectory is not None
        )

    def get_residue_statistics(self) -> Dict[str, torch.Tensor]:
        """Get residue field statistics for monitoring."""
        return self.residue_field.get_statistics()

    def forward(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Forward pass: select action from observation."""
        return self.act(observation, temperature)


# Convenience function for running a complete episode
def run_episode(
    agent: REEAgent,
    env,
    max_steps: int = 1000,
    render: bool = False
) -> Dict[str, Any]:
    """
    Run a complete episode with the REE agent.

    Args:
        agent: REE agent instance
        env: Environment instance
        max_steps: Maximum steps per episode
        render: Whether to render each step

    Returns:
        Episode statistics
    """
    agent.reset()
    obs = env.reset()

    total_reward = 0.0
    total_harm = 0.0
    steps = 0

    for step in range(max_steps):
        # Agent selects action
        action = agent.act(obs)

        # Environment step
        obs, harm_signal, done, info = env.step(action)

        # Update residue
        agent.update_residue(harm_signal)

        # Track statistics
        if harm_signal < 0:
            total_harm += abs(harm_signal)
        else:
            total_reward += harm_signal

        # Periodic offline integration
        if agent.should_integrate():
            agent.offline_integration()

        steps += 1

        if render:
            print(env.render())
            print(f"Step {step}, Harm: {harm_signal:.3f}")
            print()

        if done:
            break

    # Final statistics
    residue_stats = agent.get_residue_statistics()

    return {
        "steps": steps,
        "total_reward": total_reward,
        "total_harm": total_harm,
        "final_health": info.get("health", 0),
        "final_energy": info.get("energy", 0),
        "total_residue": residue_stats["total_residue"].item(),
        "num_harm_events": residue_stats["num_harm_events"].item()
    }
