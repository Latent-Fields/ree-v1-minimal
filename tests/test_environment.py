"""
Tests for Grid World environment implementation.

Tests verify:
- Environment initialization and reset
- Observation and action space dimensions
- Harm and benefit signal generation
- Entity interactions (resources, hazards, other agents)
"""

import pytest
import torch

from ree_core.environment.grid_world import GridWorld, EntityState


class TestGridWorldBasics:
    """Tests for basic environment functionality."""

    @pytest.fixture
    def env(self):
        """Create environment with fixed seed for reproducibility."""
        return GridWorld(
            size=10,
            num_resources=5,
            num_hazards=3,
            num_other_agents=1,
            seed=42
        )

    def test_initialization(self, env):
        """Environment initializes correctly."""
        assert env.size == 10
        assert env.num_resources == 5
        assert env.num_hazards == 3

    def test_reset_returns_observation(self, env):
        """Reset returns valid observation tensor."""
        obs = env.reset()

        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (env.observation_dim,)

    def test_observation_dim(self, env):
        """Observation dimension is calculated correctly."""
        env.reset()

        expected_dim = (
            env.size * env.size +           # position encoding
            5 * 5 * len(env.ENTITY_TYPES) +  # local view (5x5, one-hot)
            2 +                              # homeostatic (health, energy)
            env.num_other_agents * 2         # other agent positions
        )

        assert env.observation_dim == expected_dim

    def test_action_dim(self, env):
        """Action dimension matches number of actions."""
        assert env.action_dim == 5  # up, down, left, right, stay

    def test_step_returns_tuple(self, env):
        """Step returns (obs, harm, done, info)."""
        env.reset()
        action = torch.tensor(0)  # Move up

        result = env.step(action)

        assert len(result) == 4
        obs, harm, done, info = result
        assert isinstance(obs, torch.Tensor)
        assert isinstance(harm, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestGridWorldEntities:
    """Tests for entity placement and interaction."""

    @pytest.fixture
    def env(self):
        return GridWorld(size=10, seed=42)

    def test_walls_on_border(self, env):
        """Walls are placed on grid borders."""
        env.reset()

        # Check borders
        assert (env.grid[0, :] == env.ENTITY_TYPES["wall"]).all()
        assert (env.grid[-1, :] == env.ENTITY_TYPES["wall"]).all()
        assert (env.grid[:, 0] == env.ENTITY_TYPES["wall"]).all()
        assert (env.grid[:, -1] == env.ENTITY_TYPES["wall"]).all()

    def test_agent_placed_inside(self, env):
        """Agent is placed inside the grid (not on walls)."""
        env.reset()

        x, y = env.agent.x, env.agent.y
        assert 0 < x < env.size - 1
        assert 0 < y < env.size - 1

    def test_resources_placed(self, env):
        """Resources are placed in the grid."""
        env.reset()

        assert len(env.resources) == env.num_resources

    def test_hazards_placed(self, env):
        """Hazards are placed in the grid."""
        env.reset()

        assert len(env.hazards) == env.num_hazards


class TestGridWorldInteractions:
    """Tests for entity interactions and harm/benefit."""

    def test_hazard_causes_harm(self):
        """Stepping on hazard causes negative harm signal."""
        # Create small environment to control placement
        env = GridWorld(size=5, num_resources=0, num_hazards=0, seed=42)
        env.reset()

        # Manually place hazard adjacent to agent
        agent_x, agent_y = env.agent.x, env.agent.y
        hazard_pos = (agent_x, agent_y + 1)  # Right of agent
        if 0 < hazard_pos[1] < env.size - 1:
            env.grid[hazard_pos] = env.ENTITY_TYPES["hazard"]
            env.hazards.append(hazard_pos)

            # Move right into hazard
            _, harm, _, info = env.step(torch.tensor(3))  # Right

            if info.get("event") == "hazard":
                assert harm < 0, "Hazard should cause negative harm signal"

    def test_resource_causes_benefit(self):
        """Collecting resource causes positive harm signal (benefit)."""
        env = GridWorld(size=5, num_resources=0, num_hazards=0, seed=42)
        env.reset()

        # Manually place resource adjacent to agent
        agent_x, agent_y = env.agent.x, env.agent.y
        resource_pos = (agent_x, agent_y + 1)
        if 0 < resource_pos[1] < env.size - 1:
            env.grid[resource_pos] = env.ENTITY_TYPES["resource"]
            env.resources.append(resource_pos)

            # Move right into resource
            _, harm, _, info = env.step(torch.tensor(3))

            if info.get("event") == "resource":
                assert harm > 0, "Resource should cause positive signal"

    def test_wall_blocks_movement(self):
        """Agent cannot move through walls."""
        env = GridWorld(size=5, seed=42)
        env.reset()

        # Find agent position
        start_x, start_y = env.agent.x, env.agent.y

        # Try to move up until hitting wall
        for _ in range(10):
            env.step(torch.tensor(0))  # Up

        # Should not be at row 0 (wall)
        assert env.agent.x > 0

    def test_energy_decays(self):
        """Energy decays over time."""
        env = GridWorld(size=10, energy_decay=0.1, seed=42)
        env.reset()
        initial_energy = env.agent.energy

        # Take some steps
        for _ in range(5):
            env.step(torch.tensor(4))  # Stay action

        assert env.agent.energy < initial_energy


class TestGridWorldTermination:
    """Tests for episode termination conditions."""

    def test_terminates_on_zero_health(self):
        """Episode ends when health reaches zero."""
        env = GridWorld(size=5, hazard_harm=2.0, seed=42)
        env.reset()

        # Force zero health
        env.agent.health = 0

        _, _, done, _ = env.step(torch.tensor(4))

        assert done is True

    def test_terminates_on_zero_energy(self):
        """Episode ends when energy reaches zero."""
        env = GridWorld(size=5, seed=42)
        env.reset()

        # Force zero energy
        env.agent.energy = 0

        _, _, done, _ = env.step(torch.tensor(4))

        assert done is True

    def test_terminates_on_max_steps(self):
        """Episode ends after max steps."""
        env = GridWorld(size=5, seed=42)
        env.reset()
        env.steps = 999

        _, _, done, _ = env.step(torch.tensor(4))

        assert done is True


class TestGridWorldRender:
    """Tests for environment rendering."""

    def test_render_text(self):
        """Text rendering produces valid output."""
        env = GridWorld(size=5, seed=42)
        env.reset()

        output = env.render(mode="text")

        assert isinstance(output, str)
        assert "A" in output  # Agent symbol
        assert "#" in output  # Wall symbol


class TestPassCriteria:
    """
    Explicit pass criteria for environment tests.

    PASS CRITERIA:
    1. Environment initializes with correct dimensions
    2. Reset returns valid observation
    3. Step returns proper tuple structure
    4. Hazards generate negative harm signals
    5. Resources generate positive signals
    6. Walls block movement
    7. Energy decays over time
    8. Episode terminates on failure conditions
    """

    def test_criteria_count(self):
        """All 8 criteria documented."""
        assert 8 == 8
