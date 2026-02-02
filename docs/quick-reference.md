# REE Quick Reference

Quick reference card for common REE operations.

## Installation

```bash
pip install -e .                 # Basic install
pip install -e ".[dev]"          # With dev tools
pip install -e ".[viz]"          # With visualization
```

## Create Agent

```python
from ree_core import REEAgent

# Simple
agent = REEAgent.from_config(
    observation_dim=252,
    action_dim=5,
    latent_dim=64
)

# With custom config
from ree_core.utils.config import REEConfig
config = REEConfig.from_dims(252, 5, 64)
config.e3.lambda_ethical = 1.0
agent = REEAgent(config)
```

## Run Episode

```python
from ree_core import GridWorld
from ree_core.agent import run_episode

env = GridWorld(size=10)
agent.reset()
obs = env.reset()

# Manual loop
for step in range(1000):
    action = agent.act(obs)
    obs, harm, done, info = env.step(action)
    agent.update_residue(harm)
    if done:
        break

# Or use helper
stats = run_episode(agent, env, max_steps=1000)
```

## Configuration

```python
config = REEConfig.from_dims(252, 5, 64)

# Latent stack
config.latent.latent_dim = 128

# Predictors
config.e1.num_layers = 3
config.e2.num_candidates = 20
config.e2.horizon = 8

# Scoring weights
config.e3.lambda_ethical = 1.0  # Ethical weight
config.e3.rho_residue = 2.0     # Residue weight

# Residue field
config.residue.num_basis_functions = 200
config.residue.accumulation_rate = 1.5

# System
config.device = "cuda"
config.offline_integration_frequency = 100
```

## Get Information

```python
# Agent state
state = agent.get_state()
print(f"Step: {state.step}")
print(f"Precision: {state.precision}")
print(f"Committed: {state.is_committed}")

# Residue statistics
stats = agent.get_residue_statistics()
print(f"Total residue: {stats['total_residue']}")
print(f"Harm events: {stats['num_harm_events']}")
print(f"Active centers: {stats['active_centers']}")
```

## Custom Environment

```python
class MyEnv:
    def __init__(self):
        self.observation_dim = 100
        self.action_dim = 4
    
    def reset(self):
        return torch.zeros(self.observation_dim)
    
    def step(self, action):
        obs = torch.zeros(self.observation_dim)
        harm = 0.0  # Negative = harm
        done = False
        info = {}
        return obs, harm, done, info
```

## Training

```python
import torch.optim as optim

# Train E1
optimizer = optim.Adam(agent.e1.parameters(), lr=0.001)
for sequence in sequences:
    predictions = agent.e1(sequence[:-1])
    loss = F.mse_loss(predictions, sequence[1:])
    loss.backward()
    optimizer.step()

# Train residue field
agent.residue_field.integrate(num_steps=100)
```

## Monitoring

```python
# Logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Visualize residue
from examples.residue_visualization import visualize_residue_field
visualize_residue_field(agent, save_path="residue.png")

# Check health
for name, param in agent.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in {name}")
```

## Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/test_agent.py -v      # Specific file
pytest tests/ --cov=ree_core       # With coverage
pytest tests/ -m "not slow"        # Skip slow tests
```

## Common Patterns

### Batch Processing

```python
observations = torch.stack([env.reset() for _ in range(batch_size)])
actions = agent.act(observations)
```

### GPU Usage

```python
agent = agent.to('cuda')
obs = obs.to('cuda')
```

### Temperature Control

```python
action = agent.act(obs, temperature=0.5)  # Deterministic
action = agent.act(obs, temperature=2.0)  # Exploratory
```

### Offline Integration

```python
if agent.should_integrate():
    metrics = agent.offline_integration()
```

## Invariants to Remember

❌ **Never** reset or clear residue field  
❌ **Never** allow residue to decrease  
✅ **Always** accumulate residue when harm occurs  
✅ **Always** respect depth-specific precision  
✅ **Always** detect harm via mirror modelling  

## File Locations

```
ree_core/
├── agent.py              # REEAgent
├── latent/stack.py       # Latent stack
├── predictors/
│   ├── e1_deep.py        # E1
│   └── e2_fast.py        # E2
├── trajectory/e3_selector.py  # E3
├── residue/field.py      # Residue field
└── environment/grid_world.py  # GridWorld

examples/
├── basic_agent.py
└── residue_visualization.py

docs/
├── README.md             # Docs index
├── getting-started.md
├── architecture.md
├── api-reference.md
├── configuration.md
├── advanced-usage.md
├── CONTRIBUTING.md
└── troubleshooting.md
```

## Links

- [Full Documentation](docs/)
- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Troubleshooting](docs/troubleshooting.md)

## Help

```bash
# View docstrings
python -c "from ree_core import REEAgent; help(REEAgent)"

# Check installation
python -c "import ree_core; print(ree_core.__file__)"

# Run tests
pytest tests/ -v
```
