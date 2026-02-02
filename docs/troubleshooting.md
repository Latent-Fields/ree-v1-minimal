# Troubleshooting Guide

Common issues and solutions when working with REE-v1-Minimal.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Import Errors](#import-errors)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Training Problems](#training-problems)
- [Environment Integration](#environment-integration)
- [Testing Issues](#testing-issues)

## Installation Issues

### Problem: pip install fails with dependency conflicts

**Symptoms:**
```
ERROR: Cannot install ree-v1-minimal because these package versions have conflicting dependencies.
```

**Solution:**
```bash
# Use a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate

# Install with specific versions
pip install torch>=2.0.0
pip install -e .
```

### Problem: PyTorch not found or wrong version

**Symptoms:**
```python
ImportError: No module named 'torch'
```

**Solution:**
```bash
# Install PyTorch for your system
# CPU only:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# With CUDA (check your CUDA version):
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Tests fail after installation

**Solution:**
```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Run tests to verify
pytest tests/ -v
```

## Import Errors

### Problem: Cannot import ree_core

**Symptoms:**
```python
>>> from ree_core import REEAgent
ModuleNotFoundError: No module named 'ree_core'
```

**Solution:**
```bash
# Make sure package is installed
pip install -e .

# Check installation
python -c "import ree_core; print(ree_core.__file__)"

# If still failing, check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/ree-v1-minimal"
```

### Problem: Circular import errors

**Symptoms:**
```
ImportError: cannot import name 'REEAgent' from partially initialized module 'ree_core'
```

**Solution:**
- This usually indicates you're running Python from the wrong directory
```bash
# Don't run from the package directory
cd /path/to/ree-v1-minimal
python -c "from ree_core import REEAgent"  # Works

cd /path/to/ree-v1-minimal/ree_core
python -c "from ree_core import REEAgent"  # Fails!

# Solution: Run from project root
cd /path/to/ree-v1-minimal
```

## Runtime Errors

### Problem: Dimension mismatch errors

**Symptoms:**
```python
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x252 and 100x64)
```

**Solution:**
```python
# Check environment observation dimension
env = GridWorld(size=10)
print(f"Observation dim: {env.observation_dim}")  # e.g., 252

# Create agent with matching dimension
agent = REEAgent.from_config(
    observation_dim=env.observation_dim,  # Must match!
    action_dim=env.action_dim,
    latent_dim=64
)
```

### Problem: NaN values in outputs

**Symptoms:**
```python
# Agent produces NaN actions
action = agent.act(obs)
# tensor([nan, nan, nan, nan, nan])
```

**Solutions:**

1. Check input observations:
```python
# Make sure observations are valid
assert not torch.isnan(obs).any(), "NaN in observations!"
assert torch.isfinite(obs).all(), "Inf in observations!"
```

2. Check for numerical instability:
```python
# Use smaller learning rates
config = REEConfig.from_dims(252, 5, 64)
config.e1.learning_rate = 0.0001  # Smaller

# Use gradient clipping in training
torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
```

3. Initialize with smaller weights:
```python
# After creating agent
for param in agent.parameters():
    if param.dim() > 1:
        torch.nn.init.xavier_uniform_(param, gain=0.1)
```

### Problem: Out of memory errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce latent dimensions:
```python
agent = REEAgent.from_config(
    observation_dim=252,
    action_dim=5,
    latent_dim=32  # Smaller (was 64)
)
```

2. Reduce trajectory candidates:
```python
config = REEConfig.from_dims(252, 5, 64)
config.e2.num_candidates = 5  # Fewer (was 10)
config.e2.horizon = 3  # Shorter (was 5)
```

3. Reduce residue field capacity:
```python
config.residue.num_basis_functions = 50  # Fewer (was 100)
```

4. Use CPU instead of GPU for small models:
```python
config.device = "cpu"
```

### Problem: Agent always selects same action

**Symptoms:**
```python
# Agent keeps selecting action 0
for _ in range(100):
    action = agent.act(obs)
    print(action.argmax())  # Always 0
```

**Solutions:**

1. Increase temperature for more exploration:
```python
action = agent.act(obs, temperature=2.0)  # More random
```

2. Check if action decoder is functioning:
```python
# Inspect action probabilities
state = agent.get_state()
print(f"Precision: {state.precision}")  # Should not be 0

# Check if trajectories are diverse
candidates = agent.generate_trajectories(agent._current_latent)
print(f"Generated {len(candidates)} candidates")
```

3. Ensure E2 is generating diverse trajectories:
```python
config = REEConfig.from_dims(252, 5, 64)
config.e2.num_candidates = 20  # More candidates
```

## Performance Issues

### Problem: Agent is too slow

**Solutions:**

1. Reduce trajectory rollout depth:
```python
config.e2.horizon = 3  # Shorter rollouts
config.e2.num_candidates = 5  # Fewer candidates
```

2. Reduce latent dimensions:
```python
config = REEConfig.from_dims(252, 5, 32)  # Smaller
```

3. Use GPU:
```python
config.device = "cuda"
agent = REEAgent(config).to("cuda")
```

4. Profile to find bottlenecks:
```python
import time

# Time each component
start = time.time()
encoded = agent.sense(obs)
print(f"Sense: {time.time() - start:.4f}s")

start = time.time()
latent = agent.update_latent(encoded)
print(f"Update: {time.time() - start:.4f}s")

start = time.time()
candidates = agent.generate_trajectories(latent)
print(f"Generate: {time.time() - start:.4f}s")
```

### Problem: Offline integration takes too long

**Solution:**
```python
# Reduce integration frequency
config.offline_integration_frequency = 500  # Less frequent

# Or skip it for testing
# Just don't call agent.offline_integration()
```

## Training Problems

### Problem: Loss is not decreasing

**Solutions:**

1. Check learning rate:
```python
# Try different learning rates
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower
# or
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher
```

2. Check for gradient flow:
```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"No gradient: {name}")
```

3. Verify data:
```python
# Check training data
print(f"Input range: [{inputs.min():.2f}, {inputs.max():.2f}]")
print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
assert not torch.isnan(inputs).any()
assert not torch.isnan(targets).any()
```

### Problem: Residue field doesn't learn patterns

**Solution:**
```python
# Increase number of integration steps
metrics = agent.residue_field.integrate(
    num_steps=100,  # More steps (was 10)
    learning_rate=0.01
)

# Check if there's enough harm history
print(f"Harm history size: {len(agent.residue_field._harm_history)}")
# Should have at least 10-20 samples
```

## Environment Integration

### Problem: Custom environment causes crashes

**Checklist:**

1. Check return types:
```python
def step(self, action):
    # Must return exactly these types:
    observation = torch.Tensor(...)  # torch.Tensor
    harm_signal = 0.0  # float
    done = False  # bool
    info = {}  # dict
    return observation, harm_signal, done, info
```

2. Check dimensions:
```python
# Observation must match declared dimension
assert observation.shape[-1] == self.observation_dim
```

3. Check action handling:
```python
# Handle different action formats
if isinstance(action, torch.Tensor):
    if action.dim() > 0:
        action = action.argmax().item()
    else:
        action = action.item()
action = int(action) % self.action_dim
```

### Problem: Gym environment integration fails

**Solution:**
```python
class GymWrapper:
    def __init__(self, env_name):
        import gym
        self.env = gym.make(env_name)
        
        # Handle different observation spaces
        if hasattr(self.env.observation_space, 'shape'):
            self.observation_dim = self.env.observation_space.shape[0]
        else:
            self.observation_dim = self.env.observation_space.n
        
        # Handle different action spaces
        if hasattr(self.env.action_space, 'n'):
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
    
    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(np.array(obs)).float()
    
    def step(self, action):
        # Convert action
        if isinstance(action, torch.Tensor):
            action = action.argmax().item()
        
        obs, reward, done, info = self.env.step(action)
        
        # Convert observation
        obs = torch.from_numpy(np.array(obs)).float()
        
        # Convert reward to harm signal
        # REE expects: negative = harm, positive = benefit
        # If your env uses standard RL rewards (higher = better):
        #   harm_signal = -reward  # Invert
        # If your env already uses harm convention:
        harm_signal = reward  # Direct (assumes reward is already in harm format)
        
        return obs, harm_signal, done, info
```

## Testing Issues

### Problem: Tests fail on CI but pass locally

**Common causes:**

1. **Random seed not set:**
```python
# Set seed at start of test
def test_something():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # ... rest of test
```

2. **Device differences:**
```python
# Force CPU in tests
@pytest.fixture
def agent():
    config = REEConfig.from_dims(100, 4, 32)
    config.device = "cpu"  # Force CPU
    return REEAgent(config)
```

3. **Float precision:**
```python
# Use approximate comparisons
assert torch.allclose(output, expected, atol=1e-5)
# Instead of:
# assert output == expected
```

### Problem: Tests are too slow

**Solutions:**

1. Mark slow tests:
```python
@pytest.mark.slow
def test_long_training():
    # ... slow test ...
    pass

# Run fast tests only:
# pytest -v -m "not slow"
```

2. Reduce test complexity:
```python
# Use smaller agents in tests
agent = REEAgent.from_config(
    observation_dim=50,  # Smaller
    action_dim=4,
    latent_dim=16  # Smaller
)

# Run fewer steps
for _ in range(10):  # Not 1000
    ...
```

## Common Error Messages

### "Expected tensor to have dtype torch.float32"

**Solution:**
```python
# Convert observations to float
obs = obs.float()

# Or in environment:
def reset(self):
    return torch.tensor(self.state).float()  # Add .float()
```

### "IndexError: Dimension out of range"

**Solution:**
```python
# Check tensor dimensions
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor dims: {tensor.dim()}")

# Add dimensions if needed
if tensor.dim() == 1:
    tensor = tensor.unsqueeze(0)  # Add batch dimension
```

### "RuntimeError: Input and hidden must have same dtype"

**Solution:**
```python
# Ensure consistent dtypes
input_tensor = input_tensor.float()
hidden_state = hidden_state.float()
```

## Getting More Help

If you're still stuck:

1. **Check the examples**: [examples/](../examples/)
2. **Read the docs**: [docs/](.)
3. **Search issues**: Check GitHub issues
4. **Ask for help**: Open a GitHub issue with:
   - Minimal code to reproduce
   - Full error message
   - Python version, PyTorch version
   - What you've already tried

## Debugging Tips

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Python debugger

```python
import pdb

# Add breakpoint
pdb.set_trace()

# Or use ipdb for better interface
import ipdb
ipdb.set_trace()
```

### Check agent health

```python
# Utility to check for common issues
def diagnose_agent(agent):
    print("Agent Diagnostics:")
    print("=" * 50)
    
    # Check parameters
    has_nan = False
    for name, param in agent.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️  NaN in {name}")
            has_nan = True
    
    if not has_nan:
        print("✓ No NaN in parameters")
    
    # Check residue
    stats = agent.get_residue_statistics()
    print(f"✓ Total residue: {stats['total_residue']:.3f}")
    print(f"✓ Harm events: {stats['num_harm_events']}")
    
    # Check state
    if agent._current_latent is not None:
        print(f"✓ Latent state initialized")
    else:
        print("⚠️  No latent state (call reset()?)")
    
    print("=" * 50)

# Use it
diagnose_agent(agent)
```
