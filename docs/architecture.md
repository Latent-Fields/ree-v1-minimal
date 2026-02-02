# REE-v1-Minimal Architecture

This document provides a comprehensive overview of the Reflective-Ethical Engine (REE) architecture and its implementation in this minimal reference codebase.

## Table of Contents

- [Overview](#overview)
- [Core Principles](#core-principles)
- [System Architecture](#system-architecture)
- [Components](#components)
- [The REE Loop](#the-ree-loop)
- [Ethical Invariants](#ethical-invariants)
- [Design Rationale](#design-rationale)

## Overview

REE is a reference architecture for ethical agency under uncertainty. Unlike traditional reinforcement learning approaches that optimize a single objective, REE maintains **ethical continuity** through persistent moral residue while balancing multiple constraints:

- **Reality constraints** (F): Physical viability and predictive coherence
- **Ethical costs** (M): Predicted degradation of self and others
- **Residue field** (Φ): Path-dependent accumulation of past harm

The architecture is built on a multi-timescale latent representation (L-space) that separates perceptual binding from higher-level planning and motivation.

## Core Principles

### 1. Multi-Timescale Representation

REE uses a hierarchical latent stack operating at different timescales:

```
z_γ (gamma) - Sensory binding (fastest, ~10-100ms)
    ↓
z_β (beta) - Affordances and immediate actions (~100ms-1s)
    ↓
z_θ (theta) - Sequence context and narrative (~1-10s)
    ↓
z_δ (delta) - Regime and motivational state (slowest, ~10s+)
```

Each level:
- Receives bottom-up input from the level below
- Receives top-down priors from the level above
- Maintains depth-specific precision (attention/confidence)
- Operates on its own timescale

### 2. Predictive Processing

The architecture is fundamentally predictive:

- **E1 (Deep Predictor)**: Long-horizon world model operating over θ and δ
- **E2 (Fast Predictor)**: Short-horizon trajectory generator operating over β
- **Precision-weighted prediction errors**: Drive learning and attention

### 3. Ethical Path-Dependence

Residue field φ(z) makes ethical cost **path-dependent**:
- Not a simple scalar penalty
- Geometric deformation of latent space
- Makes trajectories through previously harmful regions more costly
- **Cannot be erased** (architectural invariant)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT                              │
│  (GridWorld or external environment)                        │
│                                                             │
│  Outputs: observations, harm signals, rewards               │
└─────────────────┬───────────────────────────────────────────┘
                  │ observation
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                    REE AGENT                                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. SENSE                                            │  │
│  │     • Observation encoder                            │  │
│  │     • Multi-modal fusion                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. UPDATE (Latent Stack)                            │  │
│  │     • Bottom-up encoding: obs → z_γ → z_β → z_θ → z_δ│  │
│  │     • Top-down conditioning: z_δ → z_θ → z_β → z_γ   │  │
│  │     • Precision modulation at each depth             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. GENERATE (E2 Fast Predictor)                     │  │
│  │     • Roll out candidate trajectories from z_β       │  │
│  │     • Sample action space                            │  │
│  │     • Generate N candidate futures                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  4. SCORE (E3 Trajectory Selector)                   │  │
│  │     • For each trajectory ζ:                         │  │
│  │       J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)              │  │
│  │       - F(ζ): Reality constraint                     │  │
│  │       - M(ζ): Ethical cost (harm prediction)        │  │
│  │       - Φ_R(ζ): Residue field cost (path integral)  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  5. SELECT (E3 Trajectory Selector)                  │  │
│  │     • Select trajectory with lowest total cost       │  │
│  │     • Apply precision-based stochasticity            │  │
│  │     • Extract next action                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  6. ACT                                              │  │
│  │     • Execute selected action in environment         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  7. RESIDUE UPDATE                                   │  │
│  │     • If harm occurred: φ(z_γ) += harm_magnitude    │  │
│  │     • Update RBF centers at harm location            │  │
│  │     • Accumulate total residue (never subtract!)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  8. OFFLINE INTEGRATION (periodic)                   │  │
│  │     • E1 integrates experience (replay)              │  │
│  │     • Residue field contextualization                │  │
│  │     • Precision recalibration                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Latent Stack (L-space)

**File**: `ree_core/latent/stack.py`

The latent stack represents the agent's multi-timescale internal state:

- **z_γ (Gamma)**: Shared sensory binding
  - Fuses multi-modal evidence
  - Perceptual precision weighting
  - Cannot be directly overwritten by higher levels (corrigibility)
  
- **z_β (Beta)**: Affordance and immediate action space
  - What actions are currently available
  - Short-term action consequences
  - Used by E2 for trajectory rollouts

- **z_θ (Theta)**: Sequence context
  - Short-horizon narrative
  - Temporal ordering
  - Event sequences

- **z_δ (Delta)**: Regime and motivation
  - Long-horizon context
  - Motivational state
  - Default-mode attractors

**Key Methods**:
- `encode()`: Bottom-up + top-down encoding of observations
- `predict()`: Predict next state for trajectory rollouts
- `compute_prediction_error()`: Precision-weighted errors
- `modulate_precision()`: Attention via gain modulation

### E1 Deep Predictor

**File**: `ree_core/predictors/e1_deep.py`

Long-horizon world model that:
- Predicts future latent states at θ and δ depths
- Provides priors for trajectory generation
- Integrates experience during offline processing
- Learns through prediction error minimization

Uses recurrent architecture (LSTM/GRU) to maintain temporal context.

### E2 Fast Predictor

**File**: `ree_core/predictors/e2_fast.py`

Short-horizon trajectory generator that:
- Operates over z_β (affordance space)
- Rolls out candidate trajectories (typically 3-10 steps)
- Samples from action space
- Fast enough for online decision making

Generates trajectories by:
1. Sampling actions from current affordance space
2. Predicting next states using learned dynamics
3. Repeating for short horizon

### E3 Trajectory Selector

**File**: `ree_core/trajectory/e3_selector.py`

Ethical trajectory selection that:
- Scores each candidate trajectory
- Combines reality, ethics, and residue costs
- Selects action under precision control
- Implements commitment dynamics

**Scoring Function**:
```python
J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)
```

Where:
- **F(ζ)**: Reality constraint (coherence, physical viability)
- **M(ζ)**: Ethical cost (predicted harm to self/others)
- **Φ_R(ζ)**: Residue accumulation along path
- **λ, ρ**: Weighting coefficients

### Residue Field φ(z)

**File**: `ree_core/residue/field.py`

Persistent moral cost representation:

**Implementation**:
- RBF (Radial Basis Function) layer for explicit "scars"
- Neural network for learned field shape
- Combined evaluation: `φ(z) = RBF(z) + NN(z)`

**Key Properties**:
- **Persistence**: Residue cannot be erased (invariant)
- **Path-dependence**: Cost depends on trajectory through space
- **Accumulation**: Harm only adds, never subtracts
- **Integration**: Offline processing contextualizes but doesn't remove

**Methods**:
- `accumulate()`: Add residue when harm occurs
- `evaluate()`: Get field value at a point
- `evaluate_trajectory()`: Integrate field along path
- `integrate()`: Offline contextualization

### Environment (GridWorld)

**File**: `ree_core/environment/grid_world.py`

2D test environment providing:
- Multi-modal observations (position, local view, homeostatic state)
- Harm signals (hazards, collisions)
- Benefit signals (resources)
- Other agents for mirror modelling

**Observations include**:
- Agent position (one-hot)
- Local 5×5 view (multi-channel)
- Health and energy (homeostatic)
- Other agent positions

**Actions**: Up, Down, Left, Right, Stay

## The REE Loop

Each timestep executes the canonical REE loop:

1. **SENSE**: Encode raw observations
2. **UPDATE**: Update latent stack (bottom-up + top-down)
3. **GENERATE**: E2 generates candidate trajectories
4. **SCORE**: E3 scores trajectories with F + λM + ρΦ
5. **SELECT**: Choose trajectory under precision control
6. **ACT**: Execute next action
7. **RESIDUE**: Update φ(z) if harm occurred
8. **OFFLINE**: Periodic integration (every N steps)

## Ethical Invariants

REE enforces several non-negotiable invariants:

### 1. Residue Cannot Be Erased

```python
# CORRECT: Adding residue
residue_field.accumulate(location, harm_magnitude)

# INCORRECT: This is not possible in REE
# residue_field.clear()  # Does not exist!
# residue_field.subtract(...)  # Does not exist!
```

### 2. Harm via Mirror Modelling

Harm is detected through:
- Homeostatic signals (health, energy)
- Mirror modelling of other agents
- Prediction errors in self-models

NOT through:
- Symbolic rules
- Language commands
- External labels

### 3. Language Cannot Override Harm

```python
# Harm signal from embodied sensors takes priority
harm_signal = env.step(action)  # From environment
# Language cannot override this signal
# even if language says "this is not harmful"
```

### 4. Precision is Depth-Specific

```python
# Attention modulates precision at specific depths
latent = latent_stack.modulate_precision(
    latent, 
    depth="beta",  # Only affects β layer
    gain=1.5
)
# NOT a global attention mechanism
```

### 5. Perceptual Corrigibility

Higher levels cannot directly overwrite sensory state z_γ:

```python
# z_gamma updated only by prediction errors from observations
# NOT by semantic override from higher levels
```

## Design Rationale

### Why Multi-Timescale?

Different processes operate on different timescales:
- Perception: milliseconds
- Action selection: hundreds of milliseconds  
- Planning: seconds
- Motivation: tens of seconds to minutes

A single-timescale representation cannot efficiently handle all of these.

### Why Geometric Residue?

A scalar penalty can be optimized away or traded off against reward. A **geometric field** φ(z):
- Makes cost path-dependent (trajectory matters)
- Cannot be "optimized away"
- Supports long-term ethical continuity
- Allows contextualization without erasure

### Why Predictive Processing?

Prediction errors drive:
- Learning (minimize surprise)
- Attention (precision modulation)
- Model updates (improve predictions)

This creates a unified framework where perception, action, and learning emerge from minimizing prediction error.

### Why Offline Integration?

Online processing focuses on immediate action selection. Offline processing (sleep-like):
- Integrates experience across episodes
- Improves world models
- Contextualizes residue
- Recalibrates precision

This mirrors biological sleep's role in memory consolidation and model refinement.

## Implementation Notes

### Computational Efficiency

- E2 is fast (forward rollouts only)
- E1 is slow but runs offline
- Latent stack is small (typically 64-256 dims total)
- Residue field uses RBFs for sparse representation

### Extensibility

The architecture supports:
- Different environments (replace GridWorld)
- Different predictors (replace E1/E2 implementations)
- Different residue representations (replace RBF layer)
- Different trajectory scorers (modify E3)

Core invariants must be maintained regardless of implementation.

### Training vs Inference

In this minimal implementation:
- Components are randomly initialized
- No end-to-end training currently
- Can add training loops while respecting invariants

For production:
- E1/E2 trained via prediction error minimization
- E3 trained via trajectory outcome prediction
- Residue field never trained to decrease residue

## Further Reading

- [API Reference](api-reference.md) - Detailed API documentation
- [Getting Started](getting-started.md) - Quick start guide
- [Configuration](configuration.md) - Configuration options
- [Examples](../examples/) - Code examples
