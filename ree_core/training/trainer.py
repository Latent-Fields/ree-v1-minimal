"""
REE Trainer

Implements the training loop for REEAgent in GridWorld.

Architecture:
  Two separate optimizers with different learning rates reflect the
  two-timescale structure of REE:
    - E1 world model (slow, low LR): trains on prediction loss via replay
    - E3 policy (fast, higher LR): trains on REINFORCE with harm-as-signal

Both components keep their own gradient updates separate so that world-model
updates do not directly interfere with trajectory-selection updates.

REINFORCE formulation:
  After each episode, the policy gradient is:
    policy_loss = -mean(log_prob_t * G)    for t in episode
  where G = -total_harm (harm is negative reward; minimising harm = maximising return).

E1 world model formulation:
  After each episode, sample a random sequence from the experience buffer
  and compute MSE between E1's predictions and the observed states.
  Gradients flow through the LSTM transition model.

Both losses use gradient clipping to prevent instability.
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from ree_core.agent import REEAgent
from ree_core.environment.grid_world import GridWorld


class REETrainer:
    """
    Trainer for REEAgent in GridWorld.

    Usage:
        env = GridWorld(size=10, num_hazards=4)
        agent = REEAgent(config=REEConfig.from_dims(env.observation_dim, env.action_dim))
        trainer = REETrainer(agent, env)
        history = trainer.train(num_episodes=200)
    """

    def __init__(
        self,
        agent: REEAgent,
        env: GridWorld,
        e1_lr: float = 1e-4,
        policy_lr: float = 1e-3,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            agent: REEAgent to train.
            env: GridWorld environment.
            e1_lr: Learning rate for E1 world model (slow timescale).
            policy_lr: Learning rate for E3 trajectory selector (fast timescale).
            max_grad_norm: Gradient clipping max norm.
        """
        self.agent = agent
        self.env = env
        self.max_grad_norm = max_grad_norm

        # E1 shared basis: world model + latent encoder + obs encoder
        e1_params = (
            list(agent.e1.parameters())
            + list(agent.latent_stack.parameters())
            + list(agent.obs_encoder.parameters())
        )
        # E3 policy: trajectory scoring (reality_scorer + ethical_scorer)
        policy_params = list(agent.e3.parameters())

        self.e1_optimizer = torch.optim.Adam(e1_params, lr=e1_lr)
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=policy_lr)

    def run_episode(
        self,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run one episode, collecting log_probs for policy gradient.

        Args:
            max_steps: Maximum steps per episode.
            seed: Optional seed for reproducibility.

        Returns:
            Dict with 'log_probs', 'total_harm', 'steps'.
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.agent.reset()
        obs = self.env.reset()

        log_probs: List[torch.Tensor] = []
        total_harm = 0.0
        steps = 0

        for _ in range(max_steps):
            action, log_prob = self.agent.act_with_log_prob(obs)

            if log_prob is not None:
                log_probs.append(log_prob)

            obs, harm_signal, done, info = self.env.step(action)
            self.agent.update_residue(harm_signal)

            if harm_signal < 0:
                total_harm += abs(harm_signal)

            steps += 1
            if done:
                break

        return {
            "log_probs": log_probs,
            "total_harm": total_harm,
            "steps": steps,
        }

    def update(self, episode_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute and apply gradients after one episode.

        Args:
            episode_result: Output of run_episode().

        Returns:
            Dict with 'policy_loss' and/or 'e1_loss'.
        """
        log_probs = episode_result["log_probs"]
        total_harm = episode_result["total_harm"]
        metrics: Dict[str, float] = {}

        # --- E3 policy gradient (REINFORCE) ---
        # G = -total_harm: harm is negative reward; agent should learn to avoid it.
        # policy_loss = -mean(log_prob_t * G)  [negative because we minimise loss]
        if log_probs:
            episode_return = float(-total_harm)
            log_probs_tensor = torch.stack(log_probs)  # [T]
            policy_loss = -(log_probs_tensor * episode_return).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.policy_optimizer.param_groups for p in group["params"]],
                self.max_grad_norm,
            )
            self.policy_optimizer.step()
            metrics["policy_loss"] = policy_loss.item()

        # --- E1 world model prediction loss ---
        e1_loss = self.agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            self.e1_optimizer.zero_grad()
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.e1_optimizer.param_groups for p in group["params"]],
                self.max_grad_norm,
            )
            self.e1_optimizer.step()
            metrics["e1_loss"] = e1_loss.item()

        return metrics

    def train(
        self,
        num_episodes: int = 200,
        max_steps: int = 100,
        seed: Optional[int] = None,
        verbose: bool = True,
        log_interval: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Run the full training loop.

        Args:
            num_episodes: Total episodes to train for.
            max_steps: Maximum steps per episode.
            seed: Base random seed (each episode gets seed + ep * 1000).
            verbose: Print progress.
            log_interval: Print every N episodes.

        Returns:
            List of per-episode metric dicts.
        """
        history: List[Dict[str, Any]] = []

        for ep in range(num_episodes):
            ep_seed = (seed + ep * 1000) if seed is not None else None
            episode_result = self.run_episode(max_steps=max_steps, seed=ep_seed)
            update_metrics = self.update(episode_result)

            record: Dict[str, Any] = {
                "episode": ep,
                "total_harm": episode_result["total_harm"],
                "steps": episode_result["steps"],
                **update_metrics,
            }
            history.append(record)

            if verbose and (ep % log_interval == 0 or ep == num_episodes - 1):
                e1_str = (
                    f"  e1={update_metrics['e1_loss']:.4f}"
                    if "e1_loss" in update_metrics
                    else ""
                )
                pol_str = (
                    f"  pol={update_metrics['policy_loss']:.4f}"
                    if "policy_loss" in update_metrics
                    else ""
                )
                print(
                    f"  Ep {ep:4d}/{num_episodes}"
                    f"  harm={episode_result['total_harm']:.3f}"
                    f"{e1_str}{pol_str}"
                )

        return history
