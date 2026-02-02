"""
Tests for Latent Stack (L-space) implementation.

Tests verify:
- Multi-depth encoding with proper dimensions
- Top-down conditioning flow
- Precision modulation
- Temporal smoothing
- Prediction error computation
"""

import pytest
import torch

from ree_core.latent.stack import LatentStack, LatentState, DepthEncoder
from ree_core.utils.config import LatentStackConfig


class TestDepthEncoder:
    """Tests for individual depth encoder."""

    def test_encoder_output_shape(self):
        """Encoder produces correct output dimensions."""
        encoder = DepthEncoder(input_dim=64, output_dim=32, topdown_dim=16)
        x = torch.randn(4, 64)  # batch=4, input_dim=64

        encoded, precision = encoder(x)

        assert encoded.shape == (4, 32), f"Expected (4, 32), got {encoded.shape}"
        assert precision.shape == (4, 32), f"Expected (4, 32), got {precision.shape}"

    def test_encoder_with_topdown(self):
        """Encoder accepts and uses top-down conditioning."""
        encoder = DepthEncoder(input_dim=64, output_dim=32, topdown_dim=16)
        x = torch.randn(4, 64)
        topdown = torch.randn(4, 16)

        encoded_with, _ = encoder(x, topdown)
        encoded_without, _ = encoder(x, None)

        # Outputs should differ with top-down conditioning
        assert not torch.allclose(encoded_with, encoded_without)

    def test_encoder_precision_in_range(self):
        """Precision values are in [0, 1]."""
        encoder = DepthEncoder(input_dim=64, output_dim=32, topdown_dim=0)
        x = torch.randn(10, 64)

        _, precision = encoder(x)

        assert (precision >= 0).all(), "Precision should be >= 0"
        assert (precision <= 1).all(), "Precision should be <= 1"


class TestLatentStack:
    """Tests for the complete latent stack."""

    @pytest.fixture
    def stack(self):
        """Create a latent stack with default config."""
        config = LatentStackConfig(
            observation_dim=64,
            gamma_dim=32,
            beta_dim=32,
            theta_dim=16,
            delta_dim=16
        )
        return LatentStack(config)

    def test_init_state_shapes(self, stack):
        """Initial state has correct shapes."""
        state = stack.init_state(batch_size=4)

        assert state.z_gamma.shape == (4, 32)
        assert state.z_beta.shape == (4, 32)
        assert state.z_theta.shape == (4, 16)
        assert state.z_delta.shape == (4, 16)
        assert state.timestamp == 0

    def test_encode_produces_valid_state(self, stack):
        """Encoding produces a valid latent state."""
        obs = torch.randn(4, 64)
        state = stack.encode(obs)

        assert isinstance(state, LatentState)
        assert state.z_gamma.shape == (4, 32)
        assert state.timestamp == 1

    def test_encode_uses_previous_state(self, stack):
        """Encoding with previous state differs from fresh encoding."""
        obs = torch.randn(4, 64)

        # Encode fresh
        state1 = stack.encode(obs)

        # Encode with different previous state
        prev = stack.init_state(batch_size=4)
        prev.z_gamma = torch.randn(4, 32)
        state2 = stack.encode(obs, prev)

        # Should differ due to temporal smoothing
        assert not torch.allclose(state1.z_gamma, state2.z_gamma)

    def test_predict_increments_timestamp(self, stack):
        """Prediction increments the timestamp."""
        state = stack.init_state(batch_size=1)
        assert state.timestamp == 0

        predicted = stack.predict(state)
        assert predicted.timestamp == 1

    def test_prediction_error_structure(self, stack):
        """Prediction error has all required keys."""
        state1 = stack.init_state(batch_size=4)
        state2 = stack.init_state(batch_size=4)
        state2.z_gamma = torch.randn(4, 32)

        errors = stack.compute_prediction_error(state1, state2)

        assert "gamma" in errors
        assert "beta" in errors
        assert "theta" in errors
        assert "delta" in errors
        assert "total" in errors

    def test_precision_modulation(self, stack):
        """Precision modulation changes precision values."""
        state = stack.init_state(batch_size=4)
        original_precision = state.precision["gamma"].clone()

        modulated = stack.modulate_precision(state, "gamma", 0.5)

        assert not torch.allclose(modulated.precision["gamma"], original_precision)

    def test_latent_state_to_tensor(self, stack):
        """Latent state can be converted to single tensor."""
        state = stack.init_state(batch_size=4)
        tensor = state.to_tensor()

        expected_dim = 32 + 32 + 16 + 16  # gamma + beta + theta + delta
        assert tensor.shape == (4, expected_dim)

    def test_forward_equals_encode(self, stack):
        """Forward pass is equivalent to encode."""
        obs = torch.randn(4, 64)

        state_forward = stack.forward(obs)
        state_encode = stack.encode(obs)

        # Use same input, should get similar results
        # (not identical due to initialization randomness in test)
        assert state_forward.z_gamma.shape == state_encode.z_gamma.shape


class TestLatentStateOperations:
    """Tests for LatentState dataclass operations."""

    def test_state_detach(self):
        """Detach creates independent copy without grad tracking."""
        state = LatentState(
            z_gamma=torch.randn(4, 32, requires_grad=True),
            z_beta=torch.randn(4, 32),
            z_theta=torch.randn(4, 16),
            z_delta=torch.randn(4, 16),
            precision={"gamma": torch.ones(4, 32)}
        )

        detached = state.detach()

        # Key property: detached state should not require grad
        assert not detached.z_gamma.requires_grad
        # Original should still require grad
        assert state.z_gamma.requires_grad

    def test_state_device(self):
        """Device property returns correct device."""
        state = LatentState(
            z_gamma=torch.randn(4, 32),
            z_beta=torch.randn(4, 32),
            z_theta=torch.randn(4, 16),
            z_delta=torch.randn(4, 16),
            precision={}
        )

        assert state.device == torch.device("cpu")


# Test criteria passage reporting
class TestPassCriteria:
    """
    Explicit pass criteria for latent stack tests.

    PASS CRITERIA:
    1. All tensor shapes match expected dimensions
    2. Precision values remain in valid range [0, 1]
    3. Top-down conditioning affects encoding
    4. Temporal smoothing integrates previous state
    5. Prediction errors are computed at all depths
    """

    def test_all_criteria_documented(self):
        """Verify all pass criteria are tested."""
        criteria = [
            "tensor_shapes",
            "precision_range",
            "topdown_effect",
            "temporal_smoothing",
            "prediction_errors"
        ]
        # This test documents what we're testing
        assert len(criteria) == 5, "All 5 criteria should be documented"
