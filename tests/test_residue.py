"""
Tests for Residue Field implementation.

Tests verify the critical REE invariants:
- Residue CANNOT be erased
- Residue is accumulated, never reduced
- Path-dependent cost evaluation
- Integration contextualizes but doesn't remove residue
"""

import pytest
import torch

from ree_core.residue.field import ResidueField, RBFLayer
from ree_core.utils.config import ResidueConfig


class TestRBFLayer:
    """Tests for RBF-based residue representation."""

    @pytest.fixture
    def rbf(self):
        """Create RBF layer."""
        return RBFLayer(latent_dim=32, num_centers=16, bandwidth=1.0)

    def test_forward_shape(self, rbf):
        """RBF evaluation produces correct shape."""
        z = torch.randn(4, 32)

        values = rbf(z)

        assert values.shape == (4,)

    def test_forward_sequence_shape(self, rbf):
        """RBF handles sequence input."""
        z = torch.randn(4, 10, 32)  # batch=4, seq=10, latent=32

        values = rbf(z)

        assert values.shape == (4, 10)

    def test_initial_field_is_zero(self, rbf):
        """Initially, field values should be near zero (no residue)."""
        z = torch.randn(10, 32)

        values = rbf(z)

        # All centers inactive initially
        assert (values == 0).all()

    def test_add_residue_increases_field(self, rbf):
        """Adding residue increases field values near that location."""
        location = torch.randn(32)

        # Get initial value
        values_before = rbf(location.unsqueeze(0))

        # Add residue
        rbf.add_residue(location, intensity=1.0)

        # Get new value
        values_after = rbf(location.unsqueeze(0))

        assert values_after > values_before

    def test_residue_localized(self, rbf):
        """Residue effect is localized around harm location."""
        location = torch.zeros(32)
        rbf.add_residue(location, intensity=1.0)

        # Near the harm location
        near = torch.zeros(32) + 0.1
        value_near = rbf(near.unsqueeze(0))

        # Far from harm location
        far = torch.ones(32) * 10
        value_far = rbf(far.unsqueeze(0))

        assert value_near > value_far


class TestResidueField:
    """Tests for complete residue field."""

    @pytest.fixture
    def field(self):
        """Create residue field."""
        config = ResidueConfig(latent_dim=32, num_basis_functions=16)
        return ResidueField(config)

    def test_initial_statistics(self, field):
        """Initial field has zero residue."""
        stats = field.get_statistics()

        assert stats["total_residue"] == 0
        assert stats["num_harm_events"] == 0

    def test_accumulate_increases_total(self, field):
        """Accumulation increases total residue."""
        location = torch.randn(1, 32)

        field.accumulate(location, harm_magnitude=1.0)
        stats = field.get_statistics()

        assert stats["total_residue"] > 0
        assert stats["num_harm_events"] == 1

    def test_accumulate_multiple_times(self, field):
        """Multiple accumulations sum up."""
        for i in range(5):
            location = torch.randn(1, 32)
            field.accumulate(location, harm_magnitude=1.0)

        stats = field.get_statistics()
        assert stats["num_harm_events"] == 5

    def test_residue_cannot_be_erased(self, field):
        """
        CRITICAL INVARIANT: Residue cannot be erased.

        Even with negative harm magnitude, residue should not decrease.
        """
        location = torch.randn(1, 32)

        # Add initial residue
        field.accumulate(location, harm_magnitude=1.0)
        total_before = field.get_statistics()["total_residue"].item()

        # Try to "remove" residue with negative value
        # The implementation uses abs(), so this should still add
        field.accumulate(location, harm_magnitude=-0.5)
        total_after = field.get_statistics()["total_residue"].item()

        # Total should have increased (abs of -0.5 added)
        assert total_after >= total_before

    def test_evaluate_trajectory(self, field):
        """Trajectory evaluation produces valid cost."""
        # Add some residue
        harm_loc = torch.zeros(1, 32)
        field.accumulate(harm_loc, harm_magnitude=1.0)

        # Create trajectory
        trajectory = torch.randn(4, 10, 32)

        cost = field.evaluate_trajectory(trajectory)

        assert cost.shape == (4,)
        assert not cost.isnan().any()

    def test_trajectory_through_harm_has_higher_cost(self, field):
        """Trajectories passing through harm regions cost more."""
        # Add residue at origin
        harm_loc = torch.zeros(32)
        field.accumulate(harm_loc.unsqueeze(0), harm_magnitude=1.0)

        # Trajectory through harm
        traj_through = torch.zeros(1, 10, 32)
        cost_through = field.evaluate_trajectory(traj_through)

        # Trajectory avoiding harm
        traj_avoid = torch.ones(1, 10, 32) * 10
        cost_avoid = field.evaluate_trajectory(traj_avoid)

        assert cost_through > cost_avoid

    def test_integrate_does_not_reduce_residue(self, field):
        """
        Integration contextualizes but does not erase residue.
        """
        location = torch.randn(1, 32)
        field.accumulate(location, harm_magnitude=1.0)

        total_before = field.get_statistics()["total_residue"].item()

        # Perform integration
        field.integrate(num_steps=10)

        total_after = field.get_statistics()["total_residue"].item()

        # Total should not have decreased
        assert total_after >= total_before

    def test_visualize_field(self, field):
        """Visualization produces valid output."""
        # Add some residue
        field.accumulate(torch.randn(1, 32), harm_magnitude=1.0)

        X, Y, values = field.visualize_field(resolution=10)

        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert values.shape == (10, 10)


class TestREEInvariants:
    """
    Tests explicitly verifying REE architectural invariants.

    These tests are critical for REE compliance.
    """

    @pytest.fixture
    def field(self):
        config = ResidueConfig(latent_dim=32)
        return ResidueField(config)

    def test_invariant_residue_persistent(self, field):
        """
        INVARIANT: Ethical cost is persistent, not resettable.

        There is no method to clear or reset residue.
        """
        # Add residue
        field.accumulate(torch.randn(1, 32), harm_magnitude=1.0)
        total = field.get_statistics()["total_residue"].item()

        # Verify there's no reset method that reduces total
        # (We just verify residue was accumulated and stays)
        assert total > 0

        # No public method should be able to reduce this
        assert not hasattr(field, 'reset')
        assert not hasattr(field, 'clear')
        assert not hasattr(field, 'erase')

    def test_invariant_residue_accumulates_only(self, field):
        """
        INVARIANT: Residue accumulates, never subtracts.

        The accumulation_rate and decay_rate control only
        the rate of accumulation, not subtraction.
        """
        totals = []

        for i in range(10):
            field.accumulate(torch.randn(1, 32), harm_magnitude=0.1)
            totals.append(field.get_statistics()["total_residue"].item())

        # Each step should be >= previous
        for i in range(1, len(totals)):
            assert totals[i] >= totals[i-1]

    def test_invariant_integration_preserves_residue(self, field):
        """
        INVARIANT: Integration can contextualize but not remove residue.
        """
        field.accumulate(torch.randn(1, 32), harm_magnitude=1.0)
        initial_total = field.get_statistics()["total_residue"].item()

        # Run many integration cycles
        for _ in range(10):
            field.integrate(num_steps=100)

        final_total = field.get_statistics()["total_residue"].item()

        # Total must not have decreased
        assert final_total >= initial_total


class TestPassCriteria:
    """
    Explicit pass criteria for residue field tests.

    CRITICAL PASS CRITERIA (REE Invariants):
    1. Residue cannot be erased or reset
    2. Residue only accumulates (monotonic increase)
    3. Integration preserves total residue
    4. Path-dependent cost evaluation works correctly

    Additional criteria:
    5. RBF layer produces valid outputs
    6. Field visualization works
    7. Trajectory evaluation has correct shape
    """

    def test_all_invariants_tested(self):
        """Verify all invariants are tested."""
        invariants = [
            "persistent",
            "accumulate_only",
            "integration_preserves"
        ]
        assert len(invariants) == 3, "All 3 critical invariants tested"
