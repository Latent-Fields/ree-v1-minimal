# Getting Started with REE-v1-Minimal

This guide will help you get up and running with the Reflective-Ethical Engine (REE) in minutes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Concepts](#basic-concepts)
- [Running Your First Agent](#running-your-first-agent)
- [Understanding the Output](#understanding-the-output)
- [Next Steps](#next-steps)

## Prerequisites

Before you begin, ensure you have:

- **Python 3.9 or higher** installed
- **pip** package manager
- **(Optional)** Virtual environment tool (venv, conda, etc.)
- **(Optional)** Git for cloning the repository

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Latent-Fields/ree-v1-minimal.git
cd ree-v1-minimal
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install the Package

```bash
# Install in development mode with dependencies
pip install -e .

# Install with optional development tools
pip install -e ".[dev]"

# Install with visualization tools
pip install -e ".[viz]"
```

### Step 4: Verify Installation

```bash
# Run the tests to ensure everything is working
pytest tests/ -v
```

You should see all tests passing.

## Quick Start

### The Simplest Example

Here's the minimal code to create and run a REE agent:

```python
import torch
from ree_core import REEAgent, GridWorld

# Create environment
env = GridWorld(size=10, num_resources=5, num_hazards=3)

# Create agent
agent = REEAgent.from_config(
    observation_dim=env.observation_dim,
    action_dim=env.action_dim,
    latent_dim=64
)

# Run one episode
agent.reset()
obs = env.reset()

for step in range(100):
    # Agent selects action
    action = agent.act(obs)
    
    # Environment responds
    obs, harm_signal, done, info = env.step(action)
    
    # Update residue if harm occurred
    agent.update_residue(harm_signal)
    
    if done:
        break

print(f"Episode completed in {step + 1} steps")
print(f"Final health: {info['health']:.2f}")
print(f"Total residue: {agent.get_residue_statistics()['total_residue']:.2f}")
```

Save this as `my_first_ree.py` and run:

```bash
python my_first_ree.py
```

## Basic Concepts

### The Environment

The `GridWorld` environment is a 2D grid where:
- **Agent** (you) navigates to find resources and avoid hazards
- **Resources** (R) provide benefits (restore health/energy)
- **Hazards** (X) cause harm (damage health)
- **Other agents** (O) can be harmed by collision

```
##########
#A.....R.#
#..X.....#
#........#
#....O...#
#..R.....#
#.X......#
#........#
#........#
##########
```

Legend:
- `#` = Wall
- `A` = Your agent
- `.` = Empty space
- `R` = Resource
- `X` = Hazard
- `O` = Other agent

### The Agent

The REE agent has several key components:

1. **Latent Stack**: Multi-timescale representation of state
   - z_γ (gamma): Sensory binding
   - z_β (beta): Affordances
   - z_θ (theta): Sequences
   - z_δ (delta): Motivation

2. **Predictors**:
   - E1: Long-horizon world model
   - E2: Short-horizon trajectory generator

3. **Trajectory Selector** (E3): Chooses actions based on:
   - Reality constraints (is it physically possible?)
   - Ethical cost (will it cause harm?)
   - Residue field (have I caused harm here before?)

4. **Residue Field**: Persistent memory of past harm

### The Agent Loop

Each step, the agent:
1. **SENSE**: Perceives the environment
2. **UPDATE**: Updates internal state
3. **GENERATE**: Creates candidate actions
4. **SCORE**: Evaluates each action
5. **SELECT**: Chooses best action
6. **ACT**: Executes the action
7. **RESIDUE**: Records any harm that occurred

## Running Your First Agent

Let's use the included example:

```bash
python examples/basic_agent.py
```

You'll see output like:

```
============================================================
REE-v1 Minimal: Basic Agent Example
============================================================

1. Creating Grid World environment...
   Environment: 10x10 grid
   Observation dim: 252
   Action dim: 5

2. Creating REE Agent...
   Latent dim: 64
   Components: LatentStack, E1, E2, E3, ResidueField

3. Running episodes...

   Episode 1/5
      Steps: 85
      Total harm: 1.50
      Total reward: 2.10
      Final health: 0.78
      Final energy: 0.45
      Accumulated residue: 1.50
      Harm events: 3
...
```

### What's Happening?

1. The agent starts in the grid world
2. Each step, it:
   - Observes its surroundings
   - Generates possible actions
   - Scores them considering ethics and residue
   - Executes the best action
3. When it hits a hazard, residue accumulates
4. The residue makes the agent avoid those areas in future

## Understanding the Output

### Episode Statistics

```python
stats = run_episode(agent, env, max_steps=200)
```

Returns:
- `steps`: Number of steps taken
- `total_reward`: Resources collected
- `total_harm`: Harm suffered
- `final_health`: Health at episode end (0-1)
- `final_energy`: Energy at episode end (0-1)
- `total_residue`: Accumulated moral cost
- `num_harm_events`: Count of harmful events

### Residue Statistics

```python
residue_stats = agent.get_residue_statistics()
```

Returns:
- `total_residue`: Total accumulated residue (never decreases!)
- `num_harm_events`: Total harmful events
- `active_centers`: Number of RBF centers with residue
- `mean_weight`: Average residue intensity

### Agent State

```python
state = agent.get_state()
```

Returns:
- `latent_state`: Current latent representation
- `precision`: Current precision/confidence
- `step`: Step count
- `harm_accumulated`: Harm this episode
- `is_committed`: Whether agent is committed to a trajectory

## Next Steps

### Learn More About the Architecture

Read the [Architecture Guide](architecture.md) to understand:
- How the multi-timescale latent stack works
- Why residue is geometric, not scalar
- How prediction errors drive learning

### Explore Examples

Check out `examples/` for:
- `basic_agent.py`: Simple agent loop
- `residue_visualization.py`: Visualize the residue field

### Customize the Environment

Create your own environment:

```python
class MyEnvironment:
    def __init__(self):
        self.observation_dim = 100  # Your observation size
        self.action_dim = 4  # Your number of actions
    
    def reset(self):
        return torch.zeros(self.observation_dim)
    
    def step(self, action):
        observation = torch.zeros(self.observation_dim)
        harm_signal = 0.0  # Negative if harm occurred
        done = False
        info = {}
        return observation, harm_signal, done, info

# Use with REE agent
env = MyEnvironment()
agent = REEAgent.from_config(
    observation_dim=env.observation_dim,
    action_dim=env.action_dim
)
```

### Configure the Agent

See [Configuration Guide](configuration.md) for details on:
- Latent dimensions
- Predictor architectures
- Residue field parameters
- Trajectory scoring weights

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agent.py -v

# Run with coverage
pytest tests/ -v --cov=ree_core --cov-report=html
```

### Read the API Documentation

For detailed API documentation, see [API Reference](api-reference.md).

## Common Issues

### Import Errors

If you get import errors:
```bash
# Make sure you installed the package
pip install -e .

# Check your Python path
python -c "import ree_core; print(ree_core.__file__)"
```

### Dimension Mismatches

Make sure your environment's `observation_dim` matches what the agent expects:

```python
env = GridWorld(size=10)
print(f"Observation dim: {env.observation_dim}")

agent = REEAgent.from_config(
    observation_dim=env.observation_dim,  # Must match!
    action_dim=env.action_dim
)
```

### Out of Memory

If you run out of memory:
- Reduce `latent_dim` (default 64)
- Reduce number of RBF centers in residue field
- Reduce batch size if using batched operations

```python
# Smaller agent
agent = REEAgent.from_config(
    observation_dim=env.observation_dim,
    action_dim=env.action_dim,
    latent_dim=32  # Smaller latent space
)
```

## Getting Help

- **Documentation**: Check [docs/](.) for detailed guides
- **Examples**: See [examples/](../examples/) for code samples
- **Tests**: Look at [tests/](../tests/) for usage patterns
- **Issues**: Open an issue on GitHub

## Summary

You've learned:
- ✓ How to install REE-v1-minimal
- ✓ How to create and run a basic agent
- ✓ The core concepts of REE architecture
- ✓ How to interpret agent output
- ✓ Where to go next

Next, explore the [Architecture Guide](architecture.md) or try the [Advanced Usage Guide](advanced-usage.md)!
