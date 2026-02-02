"""
Configuration classes for REE components.

Provides dataclass-based configuration with sensible defaults
for all REE architectural components.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LatentStackConfig:
    """Configuration for the multi-timescale latent stack (L-space).

    The latent stack represents temporally displaced prediction depths:
    - gamma (γ): Shared sensory binding / feature conjunction
    - beta (β): Affordance and immediate action-set maintenance
    - theta (θ): Sequence context, temporal ordering
    - delta (δ): Regime, motivational set, long-horizon context
    """
    observation_dim: int = 64
    latent_dim: int = 64

    # Dimensions for each depth level
    gamma_dim: int = 64   # Sensory binding
    beta_dim: int = 64    # Affordance/action
    theta_dim: int = 32   # Sequence context
    delta_dim: int = 32   # Regime/motivation

    # Top-down conditioning dimensions
    topdown_dim: int = 16

    # Activation function
    activation: str = "relu"


@dataclass
class E1Config:
    """Configuration for E1 Deep Predictor.

    E1 handles long-horizon latent trajectories and context,
    operating at slower timescales than E2.
    """
    latent_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    prediction_horizon: int = 20  # Steps into future
    learning_rate: float = 1e-4


@dataclass
class E2Config:
    """Configuration for E2 Fast Predictor.

    E2 predicts immediate observations and short-horizon state,
    generating candidate trajectories for E3 selection.
    """
    latent_dim: int = 64
    action_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 2
    rollout_horizon: int = 10  # Steps for trajectory rollout
    num_candidates: int = 32  # Number of trajectory candidates
    learning_rate: float = 3e-4


@dataclass
class E3Config:
    """Configuration for E3 Trajectory Selector.

    E3 evaluates candidate trajectories and selects one by minimizing:
    J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)
    """
    latent_dim: int = 64
    hidden_dim: int = 64

    # Scoring weights
    lambda_ethical: float = 1.0   # Weight for ethical cost M
    rho_residue: float = 0.5      # Weight for residue field Φ_R

    # Precision control
    commitment_threshold: float = 0.7  # Precision threshold for commitment
    precision_init: float = 0.5
    precision_max: float = 1.0
    precision_min: float = 0.1


@dataclass
class ResidueConfig:
    """Configuration for the Residue Field φ(z).

    Residue is stored as persistent curvature over latent space,
    making ethical cost path-dependent and supporting moral continuity.
    """
    latent_dim: int = 64
    hidden_dim: int = 64

    # Residue accumulation
    accumulation_rate: float = 0.1  # How fast residue accumulates
    decay_rate: float = 0.0         # 0 = no decay (invariant: residue cannot be erased)

    # Field representation
    num_basis_functions: int = 32   # For RBF representation
    kernel_bandwidth: float = 1.0

    # Integration (offline processing)
    integration_rate: float = 0.01  # Rate of contextualization


@dataclass
class EnvironmentConfig:
    """Configuration for the Grid World environment."""
    size: int = 10
    num_resources: int = 5
    num_hazards: int = 3
    num_other_agents: int = 1

    # Reward/harm signals
    resource_benefit: float = 1.0
    hazard_harm: float = -1.0
    collision_harm: float = -0.5

    # Other agent properties
    other_agent_coupling: float = 0.5  # Mirror modelling weight


@dataclass
class REEConfig:
    """Master configuration for the complete REE agent.

    Bundles all component configurations with coordination settings.
    """
    # Component configs
    latent: LatentStackConfig = field(default_factory=LatentStackConfig)
    e1: E1Config = field(default_factory=E1Config)
    e2: E2Config = field(default_factory=E2Config)
    e3: E3Config = field(default_factory=E3Config)
    residue: ResidueConfig = field(default_factory=ResidueConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Global settings
    device: str = "cpu"
    seed: Optional[int] = None

    # Agent loop settings
    offline_integration_frequency: int = 100  # Steps between "sleep" cycles

    @classmethod
    def from_dims(
        cls,
        observation_dim: int,
        action_dim: int,
        latent_dim: int = 64
    ) -> "REEConfig":
        """Create config from basic dimension specifications."""
        config = cls()
        config.latent.observation_dim = observation_dim
        config.latent.latent_dim = latent_dim
        config.latent.gamma_dim = latent_dim
        config.latent.beta_dim = latent_dim
        config.e1.latent_dim = latent_dim
        config.e2.latent_dim = latent_dim
        config.e2.action_dim = action_dim
        config.e3.latent_dim = latent_dim
        config.residue.latent_dim = latent_dim
        return config
