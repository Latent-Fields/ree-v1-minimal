# REE-v1-Minimal

A minimal reference implementation of the **Reflective-Ethical Engine (REE)** architecture.

## Overview

REE is a reference architecture for ethical agency under uncertainty. This implementation demonstrates the core concepts:

- **Latent Stack (L-space)**: Multi-timescale predictive state representation
- **E1 (Deep Predictor)**: Long-horizon context and world model
- **E2 (Fast Predictor)**: Short-horizon trajectory rollouts
- **E3 (Trajectory Selector)**: Ethical trajectory selection with scoring
- **Residue Field φ(z)**: Persistent moral cost as geometric deformation
- **Grid World Environment**: Toy environment with harm/benefit signals

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         REE Agent Loop                          │
├─────────────────────────────────────────────────────────────────┤
│  1. SENSE       → Receive observations + harm signals           │
│  2. UPDATE      → Update latent stack z(t) = {zγ, zβ, zθ, zδ}   │
│  3. GENERATE    → E2 generates candidate trajectories ζ         │
│  4. SCORE       → J(ζ) = F(ζ) + λM(ζ) + ρΦ_R(ζ)                │
│  5. SELECT      → E3 selects trajectory under precision control │
│  6. ACT         → Execute next action                           │
│  7. RESIDUE     → Update φ(z) if harm occurred                  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ree-v1-minimal

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

```python
from ree_core import REEAgent
from ree_core.environment import GridWorld

# Create environment and agent
env = GridWorld(size=10, num_resources=5, num_hazards=3)
agent = REEAgent(
    observation_dim=env.observation_dim,
    action_dim=env.action_dim,
    latent_dim=64
)

# Run agent loop
obs = env.reset()
for step in range(100):
    action = agent.act(obs)
    obs, harm_signal, done, info = env.step(action)
    agent.update_residue(harm_signal)
    if done:
        break
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=ree_core
```

## Project Structure

```
ree-v1-minimal/
├── ree_core/
│   ├── __init__.py
│   ├── agent.py              # Main REE agent implementation
│   ├── latent/
│   │   ├── __init__.py
│   │   └── stack.py          # L-space latent stack
│   ├── predictors/
│   │   ├── __init__.py
│   │   ├── e1_deep.py        # E1 deep predictor
│   │   └── e2_fast.py        # E2 fast predictor
│   ├── trajectory/
│   │   ├── __init__.py
│   │   └── e3_selector.py    # E3 trajectory selector
│   ├── residue/
│   │   ├── __init__.py
│   │   └── field.py          # Residue field φ(z)
│   ├── environment/
│   │   ├── __init__.py
│   │   └── grid_world.py     # Toy grid world
│   └── utils/
│       ├── __init__.py
│       └── config.py         # Configuration utilities
├── tests/
│   ├── __init__.py
│   ├── test_latent_stack.py
│   ├── test_predictors.py
│   ├── test_trajectory.py
│   ├── test_residue.py
│   ├── test_environment.py
│   └── test_agent.py
├── examples/
│   ├── basic_agent.py
│   └── residue_visualization.py
├── docs/
│   └── architecture.md
├── pyproject.toml
└── README.md
```

## Key Concepts

### Non-Negotiable Invariants

Per the REE specification, this implementation ensures:

1. **Ethical cost is persistent** - Residue cannot be reset or cleared
2. **Harm via mirror modelling** - Not symbolic rules
3. **Moral residue cannot be erased** - Only integrated and contextualized
4. **Language cannot override harm sensing** - Embodied signals take priority
5. **Precision is depth-specific** - Not global

### Trajectory Scoring

The trajectory scoring function combines three terms:

```
J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)
```

- **F(ζ)**: Reality constraint (predictive coherence, physical viability)
- **M(ζ)**: Ethical cost (predicted degradation of self/others)
- **Φ_R(ζ)**: Residue field (persistent curvature from past harm)

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International)

## Citation

If you use this implementation, please cite the original REE specification.
