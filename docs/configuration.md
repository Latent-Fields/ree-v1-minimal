# Configuration Guide

Detailed guide to configuring REE-v1-Minimal agents and components.

## Table of Contents

- [Overview](#overview)
- [Quick Configuration](#quick-configuration)
- [Configuration Classes](#configuration-classes)
- [Latent Stack Configuration](#latent-stack-configuration)
- [Predictor Configuration](#predictor-configuration)
- [Residue Field Configuration](#residue-field-configuration)
- [Trajectory Scoring](#trajectory-scoring)
- [Advanced Configuration](#advanced-configuration)

## Overview

REE uses a hierarchical configuration system with separate configs for each component. All configs are Python dataclasses with sensible defaults.

## Quick Configuration

### Minimal Setup

```python
from ree_core import REEAgent

# Use defaults - just specify dimensions
agent = REEAgent.from_config(
    observation_dim=252,
    action_dim=5,
    latent_dim=64
)
```

### Custom Configuration

```python
from ree_core.utils.config import REEConfig

# Create custom config
config = REEConfig.from_dims(
    observation_dim=252,
    action_dim=5,
    latent_dim=128  # Larger latent space
)

# Override specific settings
config.offline_integration_frequency = 100
config.residue.accumulation_rate = 2.0
config.e3.lambda_ethical = 0.8

# Create agent with custom config
agent = REEAgent(config)
```

## Configuration Classes

### REEConfig

Main configuration container for the entire agent.

```python
@dataclass
class REEConfig:
    latent: LatentStackConfig
    e1: E1Config
    e2: E2Config
    e3: E3Config
    residue: ResidueConfig
    offline_integration_frequency: int = 100
    device: str = "cpu"
```

**Parameters:**

- `latent` (LatentStackConfig): Latent stack configuration
- `e1` (E1Config): E1 deep predictor configuration
- `e2` (E2Config): E2 fast predictor configuration
- `e3` (E3Config): E3 trajectory selector configuration
- `residue` (ResidueConfig): Residue field configuration
- `offline_integration_frequency` (int): Steps between offline integration (default: 100)
- `device` (str): PyTorch device ("cpu" or "cuda")

**Factory Method:**

```python
config = REEConfig.from_dims(
    observation_dim: int,
    action_dim: int,
    latent_dim: int = 64
)
```

Creates a complete configuration with reasonable defaults based on the specified dimensions.

## Latent Stack Configuration

### LatentStackConfig

Configuration for the multi-timescale latent representation.

```python
@dataclass
class LatentStackConfig:
    observation_dim: int
    latent_dim: int
    gamma_dim: int  # Default: latent_dim
    beta_dim: int   # Default: latent_dim // 2
    theta_dim: int  # Default: latent_dim // 4
    delta_dim: int  # Default: latent_dim // 4
    topdown_dim: int  # Default: latent_dim // 2
    activation: str = "relu"
```

**Parameters:**

- `observation_dim` (int): Dimension of environment observations
- `latent_dim` (int): Base latent dimension (other dims derived from this)
- `gamma_dim` (int): Sensory binding depth dimension
- `beta_dim` (int): Affordance depth dimension
- `theta_dim` (int): Sequence context depth dimension
- `delta_dim` (int): Regime/motivation depth dimension
- `topdown_dim` (int): Dimension of top-down conditioning signals
- `activation` (str): Activation function ("relu" or "tanh")

**Typical Values:**

```python
# Small agent (fast, less memory)
config = LatentStackConfig(
    observation_dim=252,
    latent_dim=32,
    gamma_dim=32,
    beta_dim=16,
    theta_dim=8,
    delta_dim=8
)

# Medium agent (default)
config = LatentStackConfig(
    observation_dim=252,
    latent_dim=64,
    gamma_dim=64,
    beta_dim=32,
    theta_dim=16,
    delta_dim=16
)

# Large agent (more capacity, slower)
config = LatentStackConfig(
    observation_dim=252,
    latent_dim=128,
    gamma_dim=128,
    beta_dim=64,
    theta_dim=32,
    delta_dim=32
)
```

## Predictor Configuration

### E1Config

Configuration for the long-horizon deep predictor.

```python
@dataclass
class E1Config:
    latent_dim: int
    hidden_dim: int  # Default: latent_dim * 2
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
```

**Parameters:**

- `latent_dim` (int): Input/output dimension
- `hidden_dim` (int): RNN hidden dimension
- `num_layers` (int): Number of RNN layers
- `dropout` (float): Dropout rate
- `learning_rate` (float): Learning rate for offline integration

**Example:**

```python
from ree_core.utils.config import E1Config

e1_config = E1Config(
    latent_dim=64,
    hidden_dim=128,
    num_layers=3,
    dropout=0.2
)
```

### E2Config

Configuration for the fast trajectory generator.

```python
@dataclass
class E2Config:
    latent_dim: int
    action_dim: int
    horizon: int = 5
    num_candidates: int = 10
    hidden_dim: int  # Default: latent_dim
```

**Parameters:**

- `latent_dim` (int): Latent state dimension
- `action_dim` (int): Number of actions
- `horizon` (int): Trajectory rollout horizon (steps)
- `num_candidates` (int): Number of candidate trajectories to generate
- `hidden_dim` (int): Hidden layer dimension

**Typical Values:**

```python
# Fast, fewer candidates
e2_config = E2Config(
    latent_dim=64,
    action_dim=5,
    horizon=3,
    num_candidates=5
)

# Default
e2_config = E2Config(
    latent_dim=64,
    action_dim=5,
    horizon=5,
    num_candidates=10
)

# Thorough, more candidates
e2_config = E2Config(
    latent_dim=64,
    action_dim=5,
    horizon=8,
    num_candidates=20
)
```

### E3Config

Configuration for trajectory selection and scoring.

```python
@dataclass
class E3Config:
    latent_dim: int
    lambda_ethical: float = 0.5
    rho_residue: float = 1.0
    base_precision: float = 1.0
    commitment_threshold: float = 0.8
```

**Parameters:**

- `latent_dim` (int): Latent state dimension
- `lambda_ethical` (float): Weight for ethical cost term M(ζ)
- `rho_residue` (float): Weight for residue field term Φ_R(ζ)
- `base_precision` (float): Base precision/confidence level
- `commitment_threshold` (float): Threshold for trajectory commitment

**Scoring Function:**

The trajectory score is:
```
J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)
```

Where:
- F(ζ): Reality constraint (always weight 1.0)
- M(ζ): Ethical cost (weight `lambda_ethical`)
- Φ_R(ζ): Residue field cost (weight `rho_residue`)

**Tuning the Weights:**

```python
# Ethics-focused: avoid harm even at cost of efficiency
e3_config = E3Config(
    lambda_ethical=1.0,  # High ethical weight
    rho_residue=2.0      # High residue avoidance
)

# Balanced (default)
e3_config = E3Config(
    lambda_ethical=0.5,
    rho_residue=1.0
)

# Efficiency-focused: less ethical constraint
e3_config = E3Config(
    lambda_ethical=0.2,  # Lower ethical weight
    rho_residue=0.5      # Lower residue avoidance
)
```

**Note:** Even with low weights, the agent still accumulates residue and tracks harm (invariant).

## Residue Field Configuration

### ResidueConfig

Configuration for the persistent residue field.

```python
@dataclass
class ResidueConfig:
    latent_dim: int
    num_basis_functions: int = 100
    kernel_bandwidth: float = 1.0
    accumulation_rate: float = 1.0
    hidden_dim: int  # Default: latent_dim
```

**Parameters:**

- `latent_dim` (int): Latent space dimension
- `num_basis_functions` (int): Number of RBF centers for residue storage
- `kernel_bandwidth` (float): RBF kernel bandwidth (larger = more spread)
- `accumulation_rate` (float): Multiplier for residue accumulation
- `hidden_dim` (int): Neural field hidden dimension

**Tuning Guidelines:**

```python
# Sparse memory: fewer centers, broader spread
residue_config = ResidueConfig(
    latent_dim=64,
    num_basis_functions=50,
    kernel_bandwidth=2.0,
    accumulation_rate=1.0
)

# Dense memory: more centers, tighter spread
residue_config = ResidueConfig(
    latent_dim=64,
    num_basis_functions=200,
    kernel_bandwidth=0.5,
    accumulation_rate=1.0
)

# Stronger residue effect
residue_config = ResidueConfig(
    latent_dim=64,
    num_basis_functions=100,
    kernel_bandwidth=1.0,
    accumulation_rate=2.0  # Double the residue accumulation
)
```

**Performance Considerations:**

- More basis functions → more memory, slower evaluation
- Smaller bandwidth → more precise localization, less generalization
- Larger bandwidth → broader influence, more generalization

## Trajectory Scoring

### Understanding the Scoring Function

Each candidate trajectory ζ is scored with:

```
J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)
```

Lower scores are better (minimize cost).

**Component Breakdown:**

1. **F(ζ) - Reality Constraint**
   - Always computed, weight = 1.0
   - Measures: predictive coherence, physical viability
   - Penalizes implausible trajectories

2. **M(ζ) - Ethical Cost**
   - Weight controlled by `lambda_ethical`
   - Measures: predicted harm to self and others
   - Uses mirror modelling and homeostatic predictions

3. **Φ_R(ζ) - Residue Field**
   - Weight controlled by `rho_residue`
   - Path integral of residue field along trajectory
   - Makes agent avoid previously harmful regions

### Balancing the Terms

```python
# Example: Prioritize avoiding residue over new ethical concerns
config = REEConfig.from_dims(252, 5, 64)
config.e3.lambda_ethical = 0.3  # New harm is less weighted
config.e3.rho_residue = 2.0     # Past harm is highly weighted

# Example: Prioritize immediate ethics over past
config.e3.lambda_ethical = 1.0  # New harm is highly weighted
config.e3.rho_residue = 0.3     # Past harm is less weighted
```

## Advanced Configuration

### Device Configuration

```python
# Use GPU if available
config = REEConfig.from_dims(252, 5, 64)
config.device = "cuda" if torch.cuda.is_available() else "cpu"

agent = REEAgent(config)
```

### Per-Component Override

```python
# Start with defaults
config = REEConfig.from_dims(252, 5, 64)

# Customize specific components
config.latent.activation = "tanh"
config.e1.num_layers = 3
config.e2.horizon = 8
config.e2.num_candidates = 15
config.residue.num_basis_functions = 200

# Use custom config
agent = REEAgent(config)
```

### Environment-Specific Tuning

```python
# For simple environments: smaller, faster
simple_config = REEConfig.from_dims(
    observation_dim=50,
    action_dim=4,
    latent_dim=32
)
simple_config.e2.horizon = 3
simple_config.e2.num_candidates = 5

# For complex environments: larger, more thorough
complex_config = REEConfig.from_dims(
    observation_dim=500,
    action_dim=10,
    latent_dim=128
)
complex_config.e2.horizon = 8
complex_config.e2.num_candidates = 20
complex_config.residue.num_basis_functions = 200
```

### Offline Integration Frequency

Controls how often the agent performs sleep-like offline processing:

```python
# Frequent integration (more compute, better learning)
config.offline_integration_frequency = 50

# Default
config.offline_integration_frequency = 100

# Rare integration (less compute, slower learning)
config.offline_integration_frequency = 500
```

## Configuration Examples

### Example 1: Minimal Agent

```python
from ree_core.utils.config import REEConfig
from ree_core import REEAgent

config = REEConfig.from_dims(
    observation_dim=100,
    action_dim=4,
    latent_dim=32
)
config.e2.num_candidates = 5
config.e2.horizon = 3

agent = REEAgent(config)
```

### Example 2: Ethics-Focused Agent

```python
config = REEConfig.from_dims(252, 5, 64)

# Emphasize ethical considerations
config.e3.lambda_ethical = 1.5
config.e3.rho_residue = 2.0

# More thorough trajectory search
config.e2.num_candidates = 20
config.e2.horizon = 8

# Dense residue memory
config.residue.num_basis_functions = 200
config.residue.accumulation_rate = 1.5

agent = REEAgent(config)
```

### Example 3: Fast Agent

```python
config = REEConfig.from_dims(252, 5, 32)

# Minimize computation
config.e2.num_candidates = 5
config.e2.horizon = 3
config.residue.num_basis_functions = 50

# Less frequent integration
config.offline_integration_frequency = 200

agent = REEAgent(config)
```

## Configuration Validation

The configuration system validates parameters at creation time:

```python
try:
    config = REEConfig.from_dims(
        observation_dim=-1,  # Invalid!
        action_dim=5,
        latent_dim=64
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Further Reading

- [API Reference](api-reference.md) - Full API documentation
- [Architecture Guide](architecture.md) - Understanding the components
- [Examples](../examples/) - Configuration examples in context
