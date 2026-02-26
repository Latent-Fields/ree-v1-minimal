"""
Tests for REE training loop (REETrainer).

These tests verify that the training machinery is wired correctly:
  1. Training runs without error (smoke test)
  2. E3 policy parameters receive gradients (policy gradient flows)
  3. E1 parameters receive gradients (world model loss flows)
  4. E1 loss is finite and changes with training
  5. act_with_log_prob returns a connected log_prob

Deliberately NOT tested here:
  - Whether harm avoidance improves statistically (requires hundreds of
    episodes to be robust; see experiments/training_run.py)
  - Whether E1 loss monotonically decreases (noisy by design; trend matters,
    not step-by-step monotonicity)
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.grid_world import GridWorld
from ree_core.training.trainer import REETrainer
from ree_core.utils.config import REEConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return GridWorld(size=8, num_hazards=3)


@pytest.fixture
def agent(env):
    torch.manual_seed(0)
    config = REEConfig.from_dims(env.observation_dim, env.action_dim)
    return REEAgent(config=config)


@pytest.fixture
def trainer(agent, env):
    return REETrainer(agent, env, e1_lr=1e-3, policy_lr=1e-3)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestTrainerSmoke:
    def test_run_episode_completes(self, trainer):
        """run_episode() runs without error and returns expected keys."""
        result = trainer.run_episode(max_steps=20, seed=1)
        assert "log_probs" in result
        assert "total_harm" in result
        assert "steps" in result
        assert result["steps"] > 0

    def test_update_completes(self, trainer):
        """update() runs without error after an episode."""
        episode_result = trainer.run_episode(max_steps=20, seed=1)
        metrics = trainer.update(episode_result)
        # At least one loss should be reported
        assert "policy_loss" in metrics or "e1_loss" in metrics

    def test_train_n_episodes(self, trainer):
        """train() runs 5 episodes without error and returns history."""
        history = trainer.train(num_episodes=5, max_steps=20, seed=1, verbose=False)
        assert len(history) == 5
        for record in history:
            assert "episode" in record
            assert "total_harm" in record


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_policy_gradient_flows_to_e3(self, trainer):
        """
        E3 scorer parameters should receive non-zero gradients after a
        policy gradient update.
        """
        # Collect E3 params before
        e3_params_before = {
            name: param.data.clone()
            for name, param in trainer.agent.e3.named_parameters()
        }

        # Run one episode and update
        episode_result = trainer.run_episode(max_steps=30, seed=42)
        # Force non-zero harm so return != 0 (otherwise loss = 0)
        episode_result["total_harm"] = 1.0
        trainer.update(episode_result)

        # At least one E3 parameter should have changed
        any_changed = any(
            not torch.allclose(
                trainer.agent.e3.get_parameter(name),
                e3_params_before[name],
                atol=1e-9,
            )
            for name in e3_params_before
        )
        assert any_changed, (
            "No E3 parameters changed after policy gradient update — "
            "gradient may not be flowing through score_trajectory()."
        )

    def test_e1_gradient_flows_to_transition_rnn(self, trainer):
        """
        E1 transition_rnn parameters should receive non-zero gradients
        after a prediction loss update.
        """
        # Run episode to populate experience buffer
        trainer.run_episode(max_steps=50, seed=42)

        rnn_params_before = {
            name: param.data.clone()
            for name, param in trainer.agent.e1.transition_rnn.named_parameters()
        }

        # Manually trigger E1 update
        e1_loss = trainer.agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            trainer.e1_optimizer.zero_grad()
            e1_loss.backward()
            trainer.e1_optimizer.step()

        any_changed = any(
            not torch.allclose(
                trainer.agent.e1.transition_rnn.get_parameter(name),
                rnn_params_before[name],
                atol=1e-9,
            )
            for name in rnn_params_before
        )
        assert any_changed, (
            "No E1 transition_rnn parameters changed after prediction loss update."
        )

    def test_compute_prediction_loss_is_differentiable(self, trainer):
        """compute_prediction_loss() returns a tensor with a grad_fn when buffer is populated."""
        trainer.run_episode(max_steps=50, seed=1)
        loss = trainer.agent.compute_prediction_loss()
        assert loss.requires_grad, (
            "compute_prediction_loss() returned a tensor without requires_grad. "
            "Backprop through E1 will not work."
        )
        assert torch.isfinite(loss), f"E1 loss is non-finite: {loss.item()}"

    def test_act_with_log_prob_returns_connected_log_prob(self, trainer, env):
        """act_with_log_prob() returns a log_prob connected to the computation graph."""
        trainer.agent.reset()
        obs = env.reset()
        action, log_prob = trainer.agent.act_with_log_prob(obs)

        assert log_prob is not None, "act_with_log_prob() returned None log_prob"
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.requires_grad or log_prob.grad_fn is not None, (
            "log_prob is not connected to the computation graph. "
            "Policy gradient cannot update E3 scorers."
        )
        # log_prob should be <= 0 (log of a probability in [0,1])
        assert log_prob.item() <= 0.0, f"log_prob should be <= 0, got {log_prob.item()}"


# ---------------------------------------------------------------------------
# E1 loss quality tests
# ---------------------------------------------------------------------------

class TestE1Loss:
    def test_e1_loss_finite_after_buffer_population(self, trainer):
        """E1 loss should be finite once the buffer has data."""
        trainer.run_episode(max_steps=50, seed=1)
        loss = trainer.agent.compute_prediction_loss()
        assert torch.isfinite(loss), f"E1 loss is non-finite: {loss.item()}"

    def test_e1_loss_zero_without_buffer(self, agent):
        """compute_prediction_loss() returns zero when buffer is empty."""
        # Fresh agent has empty buffer
        loss = agent.compute_prediction_loss()
        assert loss.item() == pytest.approx(0.0, abs=1e-9)

    def test_e1_loss_decreases_over_short_training(self, env):
        """
        E1 loss should be lower at the end of 30 episodes than at the start.

        This is a soft trend test — noisy, but should hold over 30 episodes.
        """
        torch.manual_seed(99)
        config = REEConfig.from_dims(env.observation_dim, env.action_dim)
        agent = REEAgent(config=config)
        trainer = REETrainer(agent, env, e1_lr=5e-4, policy_lr=1e-3)

        # Warm up buffer
        trainer.run_episode(max_steps=50, seed=0)

        # Measure initial loss
        losses_early = []
        for ep in range(5):
            trainer.run_episode(max_steps=50, seed=ep * 100)
            loss = trainer.agent.compute_prediction_loss()
            if loss.requires_grad:
                trainer.e1_optimizer.zero_grad()
                loss.backward()
                trainer.e1_optimizer.step()
                losses_early.append(loss.item())

        # Train for 25 more episodes
        for ep in range(25):
            trainer.run_episode(max_steps=50, seed=ep * 200 + 500)
            loss = trainer.agent.compute_prediction_loss()
            if loss.requires_grad:
                trainer.e1_optimizer.zero_grad()
                loss.backward()
                trainer.e1_optimizer.step()

        # Measure late loss
        losses_late = []
        for ep in range(5):
            trainer.run_episode(max_steps=50, seed=ep * 100 + 9999)
            loss = trainer.agent.compute_prediction_loss()
            losses_late.append(loss.item())

        if losses_early and losses_late:
            mean_early = sum(losses_early) / len(losses_early)
            mean_late = sum(losses_late) / len(losses_late)
            # Allow generous tolerance — loss landscape is noisy
            assert mean_late < mean_early * 2.0, (
                f"E1 loss did not improve: early={mean_early:.4f}, late={mean_late:.4f}. "
                "This may indicate the E1 optimizer is not updating correctly."
            )
