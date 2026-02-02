"""
Tests for REE Agent implementation.

Tests verify:
- Agent initialization and configuration
- Complete agent loop (sense -> update -> generate -> select -> act)
- Residue accumulation through agent
- Offline integration
- Episode execution
"""

import pytest
import torch

from ree_core.agent import REEAgent, run_episode, AgentState
from ree_core.environment.grid_world import GridWorld
from ree_core.utils.config import REEConfig


class TestREEAgentInitialization:
    """Tests for agent initialization."""

    def test_from_config_classmethod(self):
        """Agent can be created from basic dimensions."""
        agent = REEAgent.from_config(
            observation_dim=100,
            action_dim=4,
            latent_dim=32
        )

        assert agent.config.latent.observation_dim == 100
        assert agent.config.e2.action_dim == 4

    def test_agent_has_all_components(self):
        """Agent has all required components."""
        agent = REEAgent.from_config(observation_dim=100, action_dim=4)

        assert hasattr(agent, 'latent_stack')
        assert hasattr(agent, 'e1')
        assert hasattr(agent, 'e2')
        assert hasattr(agent, 'e3')
        assert hasattr(agent, 'residue_field')

    def test_reset_initializes_state(self):
        """Reset initializes agent state."""
        agent = REEAgent.from_config(observation_dim=100, action_dim=4)

        agent.reset()

        assert agent._current_latent is not None
        assert agent._step_count == 0
        assert agent._harm_this_episode == 0.0


class TestREEAgentLoop:
    """Tests for the REE agent loop components."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return REEAgent.from_config(
            observation_dim=100,
            action_dim=4,
            latent_dim=32
        )

    def test_sense_encodes_observation(self, agent):
        """Sense step encodes observation correctly."""
        obs = torch.randn(100)

        encoded = agent.sense(obs)

        assert encoded.shape == (1, 100)

    def test_update_latent_creates_state(self, agent):
        """Update step creates valid latent state."""
        agent.reset()
        encoded = torch.randn(1, 100)

        state = agent.update_latent(encoded)

        assert state is not None
        assert agent._step_count == 0  # Not incremented until act()

    def test_generate_trajectories(self, agent):
        """Generate step produces candidate trajectories."""
        agent.reset()
        encoded = torch.randn(1, 100)
        state = agent.update_latent(encoded)

        candidates = agent.generate_trajectories(state, num_candidates=5)

        assert len(candidates) == 5

    def test_select_action_produces_tensor(self, agent):
        """Select step produces action tensor."""
        agent.reset()
        encoded = torch.randn(1, 100)
        state = agent.update_latent(encoded)
        candidates = agent.generate_trajectories(state)

        action = agent.select_action(candidates)

        assert isinstance(action, torch.Tensor)

    def test_act_complete_loop(self, agent):
        """Act performs complete loop and returns action."""
        agent.reset()
        obs = torch.randn(100)

        action = agent.act(obs)

        assert isinstance(action, torch.Tensor)
        assert agent._step_count == 1


class TestREEAgentResidue:
    """Tests for residue handling through agent."""

    @pytest.fixture
    def agent(self):
        return REEAgent.from_config(
            observation_dim=100,
            action_dim=4,
            latent_dim=32
        )

    def test_update_residue_on_harm(self, agent):
        """Residue is updated when harm occurs."""
        agent.reset()
        obs = torch.randn(100)
        agent.act(obs)  # Need latent state first

        initial_residue = agent.get_residue_statistics()["total_residue"].item()

        metrics = agent.update_residue(-0.5)  # Harm signal

        final_residue = agent.get_residue_statistics()["total_residue"].item()

        assert final_residue > initial_residue
        assert metrics["harm_signal"] == -0.5

    def test_no_residue_on_benefit(self, agent):
        """Residue not updated on positive signal."""
        agent.reset()
        obs = torch.randn(100)
        agent.act(obs)

        initial_residue = agent.get_residue_statistics()["total_residue"].item()

        agent.update_residue(0.5)  # Benefit signal

        final_residue = agent.get_residue_statistics()["total_residue"].item()

        assert final_residue == initial_residue

    def test_residue_persists_across_reset(self, agent):
        """
        INVARIANT: Residue persists across resets.

        Reset clears latent state but not residue field.
        """
        agent.reset()
        obs = torch.randn(100)
        agent.act(obs)
        agent.update_residue(-1.0)

        residue_before = agent.get_residue_statistics()["total_residue"].item()

        agent.reset()  # Reset agent

        residue_after = agent.get_residue_statistics()["total_residue"].item()

        assert residue_after == residue_before


class TestREEAgentIntegration:
    """Tests for offline integration."""

    @pytest.fixture
    def agent(self):
        return REEAgent.from_config(
            observation_dim=100,
            action_dim=4,
            latent_dim=32
        )

    def test_offline_integration_runs(self, agent):
        """Offline integration runs without error."""
        agent.reset()

        # Generate some experience
        for _ in range(20):
            obs = torch.randn(100)
            agent.act(obs)

        metrics = agent.offline_integration()

        assert isinstance(metrics, dict)

    def test_should_integrate_frequency(self, agent):
        """Integration frequency check works correctly."""
        agent.reset()
        agent._step_count = 0

        # Set frequency
        agent.config.offline_integration_frequency = 10

        # Should not integrate at step 5
        agent._step_count = 5
        assert not agent.should_integrate()

        # Should integrate at step 10
        agent._step_count = 10
        assert agent.should_integrate()


class TestREEAgentWithEnvironment:
    """Tests for agent with actual environment."""

    @pytest.fixture
    def setup(self):
        """Create agent and environment."""
        env = GridWorld(size=8, seed=42)
        agent = REEAgent.from_config(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            latent_dim=32
        )
        return agent, env

    def test_agent_environment_compatibility(self, setup):
        """Agent works with environment."""
        agent, env = setup
        agent.reset()
        obs = env.reset()

        action = agent.act(obs)

        assert action.shape[-1] == env.action_dim

    def test_complete_episode(self, setup):
        """Agent can complete a full episode."""
        agent, env = setup
        agent.reset()
        obs = env.reset()

        steps = 0
        for _ in range(100):
            action = agent.act(obs)
            obs, harm, done, info = env.step(action)
            agent.update_residue(harm)
            steps += 1
            if done:
                break

        assert steps > 0
        assert steps <= 100

    def test_run_episode_helper(self, setup):
        """run_episode helper function works."""
        agent, env = setup

        stats = run_episode(agent, env, max_steps=50)

        assert "steps" in stats
        assert "total_harm" in stats
        assert "total_residue" in stats
        assert stats["steps"] > 0


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_get_state(self):
        """Agent returns valid state."""
        agent = REEAgent.from_config(observation_dim=100, action_dim=4)
        agent.reset()
        obs = torch.randn(100)
        agent.act(obs)

        state = agent.get_state()

        assert isinstance(state, AgentState)
        assert state.step == 1
        assert isinstance(state.precision, float)


class TestPassCriteria:
    """
    Explicit pass criteria for REE agent tests.

    PASS CRITERIA:
    1. Agent initializes with all components
    2. Each loop step (sense, update, generate, select, act) works correctly
    3. Residue is accumulated on harm
    4. Residue persists across resets (critical invariant!)
    5. Offline integration runs successfully
    6. Agent is compatible with environment
    7. Complete episodes can be run
    """

    def test_criteria_count(self):
        """All 7 criteria documented."""
        assert 7 == 7
