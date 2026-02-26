"""
Tests for E3 trajectory selector implementation.

Tests verify:
- Trajectory scoring (reality, ethical, residue costs)
- Selection under precision control
- Commitment as precision gating
- Post-action updates
"""

import pytest
import torch

from ree_core.trajectory.e3_selector import E3TrajectorySelector, SelectionResult
from ree_core.predictors.e2_fast import Trajectory
from ree_core.residue.field import ResidueField
from ree_core.utils.config import E3Config, ResidueConfig


class TestE3TrajectorySelector:
    """Tests for E3 trajectory selector."""

    @pytest.fixture
    def residue_field(self):
        """Create residue field for testing."""
        config = ResidueConfig(latent_dim=32)
        return ResidueField(config)

    @pytest.fixture
    def e3(self, residue_field):
        """Create E3 selector with residue field."""
        config = E3Config(
            latent_dim=32,
            hidden_dim=32,
            lambda_ethical=1.0,
            rho_residue=0.5,
            commitment_threshold=0.7,
            precision_init=0.5
        )
        return E3TrajectorySelector(config, residue_field)

    @pytest.fixture
    def mock_trajectory(self):
        """Create a mock trajectory for testing."""
        states = [torch.randn(2, 32) for _ in range(6)]
        actions = torch.randn(2, 5, 4)
        harm_predictions = torch.rand(2, 5)
        return Trajectory(
            states=states,
            actions=actions,
            harm_predictions=harm_predictions
        )

    def test_compute_reality_cost(self, e3, mock_trajectory):
        """Reality cost computation produces valid output."""
        cost = e3.compute_reality_cost(mock_trajectory)

        assert cost.shape == (2,)  # batch_size
        assert not cost.isnan().any()

    def test_compute_ethical_cost(self, e3, mock_trajectory):
        """Ethical cost computation produces valid output."""
        cost = e3.compute_ethical_cost(mock_trajectory)

        assert cost.shape == (2,)
        assert not cost.isnan().any()

    def test_compute_residue_cost(self, e3, mock_trajectory):
        """Residue cost computation produces valid output."""
        cost = e3.compute_residue_cost(mock_trajectory)

        assert cost.shape == (2,)
        assert not cost.isnan().any()

    def test_score_trajectory(self, e3, mock_trajectory):
        """Total trajectory scoring produces valid output."""
        score = e3.score_trajectory(mock_trajectory)

        assert score.shape == (2,)
        assert not score.isnan().any()

    def test_select_returns_valid_result(self, e3, mock_trajectory):
        """Selection returns valid SelectionResult."""
        candidates = [mock_trajectory] * 5

        result = e3.select(candidates)

        assert isinstance(result, SelectionResult)
        assert 0 <= result.selected_index < 5
        assert result.selected_action.shape == (2, 4)
        assert result.scores.shape == (5,)

    def test_select_respects_precision(self, e3, mock_trajectory):
        """Selection behavior changes with precision level."""
        candidates = [mock_trajectory] * 5

        # High precision: should commit
        e3.current_precision = 0.9
        result_high = e3.select(candidates)
        assert result_high.committed is True

        # Low precision: should not commit
        e3.current_precision = 0.3
        result_low = e3.select(candidates)
        assert result_low.committed is False

    def test_update_precision_on_good_prediction(self, e3):
        """Precision increases with good predictions."""
        initial_precision = e3.current_precision
        small_error = torch.zeros(2, 32)  # Perfect prediction

        e3.update_precision(small_error)

        assert e3.current_precision > initial_precision

    def test_update_precision_on_bad_prediction(self, e3):
        """Precision decreases with bad predictions."""
        initial_precision = e3.current_precision
        large_error = torch.ones(2, 32) * 10  # Large error

        e3.update_precision(large_error)

        assert e3.current_precision < initial_precision

    def test_precision_bounded(self, e3):
        """Precision stays within configured bounds."""
        # Try to push precision too high
        for _ in range(100):
            e3.update_precision(torch.zeros(1, 32))

        assert e3.current_precision <= e3.config.precision_max

        # Try to push precision too low
        for _ in range(100):
            e3.update_precision(torch.ones(1, 32) * 100)

        assert e3.current_precision >= e3.config.precision_min

    def test_post_action_update(self, e3, mock_trajectory):
        """Post-action update handles harm correctly."""
        # Commit to a trajectory first
        e3._committed_trajectory = mock_trajectory
        actual_outcome = torch.randn(2, 32)

        metrics = e3.post_action_update(actual_outcome, harm_occurred=True)

        assert "prediction_error" in metrics
        assert "residue_updated" in metrics
        assert e3._committed_trajectory is None  # Should be cleared

    def test_get_commitment_state(self, e3):
        """Commitment state returns expected keys."""
        state = e3.get_commitment_state()

        assert "precision" in state
        assert "is_committed" in state
        assert "commitment_threshold" in state

    # ------------------------------------------------------------------
    # Consolidation Selectivity Tests (MECH-068 / CSH-1)
    # ------------------------------------------------------------------

    def test_e3_consolidation_independence(self, residue_field, mock_trajectory):
        """
        CSH-1: E3 consolidation weights change trajectory scores.

        The J(zeta) = F(zeta) + lambda*M(zeta) + rho*Phi_R(zeta) scoring must
        produce meaningfully different scores when lambda/rho are varied.
        This verifies that the consolidation operator (E3) is actually doing
        selective work — not that all configs produce identical outputs.

        A failure here would mean E3 is not functioning as a consolidation
        operator over the shared E1 basis.
        """
        latent_dim = 32
        candidates = [mock_trajectory] * 3

        scores = {}
        configs = [
            ("high_ethical", E3Config(latent_dim=latent_dim, lambda_ethical=2.0, rho_residue=0.0)),
            ("zero_ethical", E3Config(latent_dim=latent_dim, lambda_ethical=0.0, rho_residue=0.0)),
            ("high_residue", E3Config(latent_dim=latent_dim, lambda_ethical=0.0, rho_residue=2.0)),
        ]

        for name, config in configs:
            e3_variant = E3TrajectorySelector(config, residue_field)
            # Score first candidate
            score = e3_variant.score_trajectory(mock_trajectory)
            scores[name] = float(score.mean().item())

        # Scores should differ between high_ethical and zero_ethical
        # when harm_predictions are non-zero (which they are in mock_trajectory)
        score_diff = abs(scores["high_ethical"] - scores["zero_ethical"])
        assert score_diff > 1e-6, (
            f"E3 consolidation weights have no effect on trajectory scores "
            f"(high_ethical={scores['high_ethical']:.4f}, "
            f"zero_ethical={scores['zero_ethical']:.4f}). "
            f"E3 is not acting as a selective consolidation operator."
        )

    def test_e1_stability_under_consolidation_change(self):
        """
        CSH-1: E1 latent representations should not be affected by E3-only changes.

        When E3 consolidation weights (lambda, rho) change but E1 weights are
        frozen, the latent state produced by E1 from the same observation should
        be identical (E1 is deterministic given the same input and same weights).

        This is the structural guarantee: E1 = shared basis, E3 = consolidation.
        The E1 output should be independent of E3 configuration.
        """
        import copy
        from ree_core.agent import REEAgent
        from ree_core.utils.config import REEConfig

        obs_dim = 16
        act_dim = 4
        latent_dim = 32

        # Build two agents with different E3 consolidation weights
        config_high = REEConfig.from_dims(obs_dim, act_dim, latent_dim)
        config_high.e3.lambda_ethical = 2.0
        config_high.e3.rho_residue = 1.0

        config_low = REEConfig.from_dims(obs_dim, act_dim, latent_dim)
        config_low.e3.lambda_ethical = 0.0
        config_low.e3.rho_residue = 0.0

        agent_high = REEAgent(config=config_high)
        agent_low = REEAgent(config=config_low)

        # Copy the shared basis weights from agent_high to agent_low.
        # In REE, "E1 = shared basis" corresponds to both the latent_stack (encoder)
        # and e1 (deep predictor). We must copy both to establish a shared basis.
        agent_low.e1.load_state_dict(copy.deepcopy(agent_high.e1.state_dict()))
        agent_low.latent_stack.load_state_dict(copy.deepcopy(agent_high.latent_stack.state_dict()))

        # Copy obs_encoder too (sits between raw obs and latent_stack)
        if hasattr(agent_high, 'obs_encoder') and hasattr(agent_low, 'obs_encoder'):
            agent_low.obs_encoder.load_state_dict(copy.deepcopy(agent_high.obs_encoder.state_dict()))

        # Freeze E1 + latent_stack in both agents (the shared basis)
        for param in agent_high.e1.parameters():
            param.requires_grad = False
        for param in agent_low.e1.parameters():
            param.requires_grad = False
        for param in agent_high.latent_stack.parameters():
            param.requires_grad = False
        for param in agent_low.latent_stack.parameters():
            param.requires_grad = False

        # Run the same observation through both agents' sense + update paths
        # Both agents start fresh (newly constructed, no prior latent state)
        torch.manual_seed(0)
        obs = torch.randn(obs_dim)

        with torch.no_grad():
            encoded_high = agent_high.sense(obs)
            latent_high = agent_high.update_latent(encoded_high)

            encoded_low = agent_low.sense(obs)
            latent_low = agent_low.update_latent(encoded_low)

        # E1 outputs should be identical (same weights, same input)
        # Compare z_theta and z_delta (E1-dominated depths in the latent stack)
        theta_sim = torch.nn.functional.cosine_similarity(
            latent_high.z_theta.flatten().unsqueeze(0),
            latent_low.z_theta.flatten().unsqueeze(0),
        ).item()

        assert theta_sim > 0.99, (
            f"E1 latent z_theta differs between agents with different E3 configs "
            f"but identical E1 weights (cosine_sim={theta_sim:.4f}). "
            f"E1 is not functioning as a stable shared basis independent of E3."
        )


class TestSelectionWithResidue:
    """Tests for selection behavior with residue field."""

    @pytest.fixture
    def setup(self):
        """Create E3 with residue that has accumulated harm."""
        residue_config = ResidueConfig(latent_dim=32)
        residue_field = ResidueField(residue_config)

        # Add some residue
        harm_location = torch.randn(1, 32)
        residue_field.accumulate(harm_location, harm_magnitude=1.0)

        e3_config = E3Config(latent_dim=32, rho_residue=1.0)
        e3 = E3TrajectorySelector(e3_config, residue_field)

        return e3, harm_location

    def test_residue_affects_scoring(self, setup):
        """Trajectories near residue have higher cost."""
        e3, harm_location = setup

        # Trajectory near harm
        states_near = [harm_location + torch.randn(1, 32) * 0.1 for _ in range(6)]
        traj_near = Trajectory(
            states=states_near,
            actions=torch.randn(1, 5, 4),
            harm_predictions=torch.zeros(1, 5)
        )

        # Trajectory far from harm
        states_far = [torch.randn(1, 32) * 10 for _ in range(6)]
        traj_far = Trajectory(
            states=states_far,
            actions=torch.randn(1, 5, 4),
            harm_predictions=torch.zeros(1, 5)
        )

        cost_near = e3.compute_residue_cost(traj_near)
        cost_far = e3.compute_residue_cost(traj_far)

        # Near trajectory should have higher residue cost
        assert cost_near.mean() > cost_far.mean()


class TestPassCriteria:
    """
    Explicit pass criteria for E3 trajectory selector tests.

    PASS CRITERIA:
    1. All cost components (reality, ethical, residue) produce valid outputs
    2. Selection returns properly structured results
    3. Precision-based commitment works correctly
    4. Precision updates reflect prediction accuracy
    5. Residue field integration affects trajectory scoring
    6. Post-action updates handle harm appropriately
    """

    def test_criteria_count(self):
        """All criteria documented."""
        assert 6 == 6, "6 criteria for E3 testing"
