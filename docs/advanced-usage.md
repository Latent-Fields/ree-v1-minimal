# Advanced Usage Guide

Advanced patterns and techniques for using REE-v1-Minimal.

## Table of Contents

- [Custom Environments](#custom-environments)
- [Training Components](#training-components)
- [Extending the Architecture](#extending-the-architecture)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Performance Optimization](#performance-optimization)
- [Multi-Agent Scenarios](#multi-agent-scenarios)
- [Integration with Other Frameworks](#integration-with-other-frameworks)

## Custom Environments

### Creating a Custom Environment

To use REE with your own environment, implement these methods:

```python
import torch
from typing import Tuple, Dict, Any

class MyCustomEnvironment:
    """Custom environment for REE agent."""
    
    def __init__(self):
        # Define observation and action spaces
        self.observation_dim = 128
        self.action_dim = 6
        
        # Initialize environment state
        self.state = None
    
    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        self.state = self._init_state()
        return self._get_observation()
    
    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute action and return (obs, harm_signal, done, info)."""
        # Update state based on action
        self._update_state(action)
        
        # Compute observation
        observation = self._get_observation()
        
        # Compute harm signal (negative = harm, positive = benefit)
        harm_signal = self._compute_harm()
        
        # Check if episode is done
        done = self._is_terminal()
        
        # Additional info
        info = {
            "step": self.step_count,
            "state": self.state,
        }
        
        return observation, harm_signal, done, info
    
    def _get_observation(self) -> torch.Tensor:
        """Convert state to observation tensor."""
        return torch.zeros(self.observation_dim)
    
    def _compute_harm(self) -> float:
        """Compute harm signal from current state."""
        # Negative values = harm occurred
        # Positive values = benefit occurred
        # Zero = neutral
        return 0.0
```

### Example: Continuous Control Environment

```python
import torch
import numpy as np

class ContinuousReachingTask:
    """Robotic arm reaching task with obstacle avoidance."""
    
    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.observation_dim = dimension * 4  # pos, vel, target, obstacle
        self.action_dim = dimension  # joint torques
        
        self.position = None
        self.velocity = None
        self.target = None
        self.obstacle = None
    
    def reset(self) -> torch.Tensor:
        self.position = torch.zeros(self.dimension)
        self.velocity = torch.zeros(self.dimension)
        self.target = torch.rand(self.dimension) * 2 - 1
        self.obstacle = torch.rand(self.dimension) * 2 - 1
        return self._get_observation()
    
    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        # Simple physics
        dt = 0.01
        self.velocity += action * dt
        self.position += self.velocity * dt
        
        # Compute harm (distance to obstacle)
        obstacle_dist = torch.norm(self.position - self.obstacle)
        harm_signal = 0.0
        
        if obstacle_dist < 0.1:  # Collision
            harm_signal = -1.0
        elif obstacle_dist < 0.3:  # Near miss
            harm_signal = -0.1 * (0.3 - obstacle_dist) / 0.2
        
        # Compute reward (reaching target)
        target_dist = torch.norm(self.position - self.target)
        if target_dist < 0.1:
            harm_signal += 1.0  # Benefit for reaching target
        
        done = target_dist < 0.1 or obstacle_dist < 0.05
        
        return self._get_observation(), harm_signal, done, {}
    
    def _get_observation(self) -> torch.Tensor:
        return torch.cat([
            self.position,
            self.velocity,
            self.target,
            self.obstacle
        ])
```

### Using Custom Environments

```python
from ree_core import REEAgent

# Create custom environment
env = ContinuousReachingTask(dimension=3)

# Create agent
agent = REEAgent.from_config(
    observation_dim=env.observation_dim,
    action_dim=env.action_dim,
    latent_dim=64
)

# Run episode
agent.reset()
obs = env.reset()

for step in range(1000):
    action = agent.act(obs)
    obs, harm_signal, done, info = env.step(action)
    agent.update_residue(harm_signal)
    
    if done:
        break
```

## Training Components

### Training E1 Deep Predictor

E1 learns by minimizing prediction error on latent state sequences:

```python
import torch.optim as optim

def train_e1(agent, experience_buffer, num_epochs=10):
    """Train E1 on collected experience."""
    optimizer = optim.Adam(agent.e1.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Sample sequences from experience
        sequences = sample_sequences(experience_buffer, seq_len=20, batch_size=32)
        
        total_loss = 0
        for sequence in sequences:
            optimizer.zero_grad()
            
            # Predict next states
            predictions = []
            agent.e1.reset_hidden_state()
            
            for t in range(len(sequence) - 1):
                pred, _ = agent.e1(sequence[t])
                predictions.append(pred)
            
            # Compute prediction error
            predictions = torch.stack(predictions)
            targets = sequence[1:]
            loss = torch.nn.functional.mse_loss(predictions, targets)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(sequences):.4f}")

def sample_sequences(buffer, seq_len, batch_size):
    """Sample random sequences from experience buffer."""
    sequences = []
    for _ in range(batch_size):
        start_idx = np.random.randint(0, len(buffer) - seq_len)
        seq = torch.stack(buffer[start_idx:start_idx + seq_len])
        sequences.append(seq)
    return sequences
```

### Training E2 Fast Predictor

E2 learns dynamics for short-horizon trajectory prediction:

```python
def train_e2(agent, transitions, num_epochs=10):
    """Train E2 on state transitions."""
    optimizer = optim.Adam(agent.e2.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Sample transitions
        batch = sample_transitions(transitions, batch_size=64)
        
        optimizer.zero_grad()
        
        # Predict next state given current state and action
        z_current = batch['z_beta']
        actions = batch['actions']
        z_next_true = batch['z_beta_next']
        
        # E2 forward pass
        z_next_pred = agent.e2.predict_next_state(z_current, actions)
        
        # Loss: prediction error
        loss = torch.nn.functional.mse_loss(z_next_pred, z_next_true)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, E2 Loss: {loss.item():.4f}")
```

### Training Residue Field (Contextualization)

The residue field can be trained to better represent the residue geometry:

```python
def train_residue_field(agent, num_steps=100):
    """Train neural component of residue field."""
    optimizer = optim.Adam(agent.residue_field.neural_field.parameters(), lr=0.001)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Sample from harm history
        if len(agent.residue_field._harm_history) < 10:
            continue
        
        harm_locations = torch.stack(
            agent.residue_field._harm_history[-100:]
        )
        
        # Add noise for generalization
        noise = torch.randn_like(harm_locations) * 0.5
        sample_points = harm_locations + noise
        
        # Target: RBF field values
        with torch.no_grad():
            targets = agent.residue_field.rbf_field(sample_points)
        
        # Predict with neural field
        predictions = agent.residue_field.neural_field(sample_points).squeeze(-1)
        
        # Loss
        loss = torch.nn.functional.mse_loss(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}, Residue Field Loss: {loss.item():.4f}")
```

### Full Training Loop

```python
def train_agent(agent, env, num_episodes=100):
    """Complete training loop for REE agent."""
    
    experience_buffer = []
    transitions = []
    
    for episode in range(num_episodes):
        agent.reset()
        obs = env.reset()
        
        episode_transitions = []
        
        for step in range(1000):
            # Get current latent state
            z_current = agent._current_latent
            
            # Act
            action = agent.act(obs)
            
            # Environment step
            obs_next, harm_signal, done, info = env.step(action)
            
            # Update residue
            agent.update_residue(harm_signal)
            
            # Store transition
            z_next = agent._current_latent
            transitions.append({
                'z_beta': z_current.z_beta,
                'actions': action,
                'z_beta_next': z_next.z_beta,
                'harm': harm_signal
            })
            
            # Store for E1
            experience_buffer.append(z_current.z_gamma)
            
            obs = obs_next
            
            if done:
                break
        
        # Periodic training
        if episode % 10 == 0 and len(experience_buffer) > 100:
            print(f"\nTraining after episode {episode}...")
            
            # Train E1
            train_e1(agent, experience_buffer, num_epochs=5)
            
            # Train E2
            train_e2(agent, transitions, num_epochs=5)
            
            # Train residue field
            train_residue_field(agent, num_steps=50)
            
            print("Training complete.\n")
```

## Extending the Architecture

### Custom Latent Encoder

Replace the default observation encoder with a custom one:

```python
import torch.nn as nn

class ConvolutionalEncoder(nn.Module):
    """Custom encoder for image observations."""
    
    def __init__(self, input_channels, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
    
    def forward(self, x):
        return self.conv(x)

# Replace agent's encoder
agent = REEAgent.from_config(observation_dim=84*84*4, action_dim=4)
agent.obs_encoder = ConvolutionalEncoder(
    input_channels=4,
    latent_dim=agent.config.latent.observation_dim
)
```

### Custom Trajectory Scorer

Implement a custom scoring function for E3:

```python
class CustomE3Scorer(nn.Module):
    """Custom trajectory scorer with additional terms."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Add custom networks for scoring
    
    def score_trajectory(self, trajectory, latent_state):
        """Custom scoring function."""
        # Standard REE terms
        F = self.compute_reality_cost(trajectory)
        M = self.compute_ethical_cost(trajectory)
        Phi = self.evaluate_residue(trajectory)
        
        # Add custom term
        C = self.compute_custom_cost(trajectory, latent_state)
        
        # Combined score
        total_cost = F + self.config.lambda_ethical * M + \
                    self.config.rho_residue * Phi + \
                    self.custom_weight * C
        
        return total_cost
```

## Monitoring and Debugging

### Detailed Logging

```python
import logging
from ree_core.agent import run_episode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("REE")

def run_episode_with_logging(agent, env, max_steps=1000):
    """Run episode with detailed logging."""
    agent.reset()
    obs = env.reset()
    
    for step in range(max_steps):
        # Log latent state
        state = agent.get_state()
        logger.debug(f"Step {step}:")
        logger.debug(f"  Precision: {state.precision:.3f}")
        logger.debug(f"  Committed: {state.is_committed}")
        
        # Act
        action = agent.act(obs)
        logger.debug(f"  Action: {action.argmax().item()}")
        
        # Step
        obs, harm_signal, done, info = env.step(action)
        
        # Log harm
        if harm_signal < 0:
            logger.warning(f"  HARM: {harm_signal:.3f}")
        elif harm_signal > 0:
            logger.info(f"  BENEFIT: {harm_signal:.3f}")
        
        # Update residue
        metrics = agent.update_residue(harm_signal)
        if 'residue_total_residue' in metrics:
            logger.debug(f"  Total residue: {metrics['residue_total_residue']:.3f}")
        
        if done:
            break
    
    return step + 1
```

### Visualization Tools

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_residue_field(agent, save_path="residue_field.png"):
    """Visualize the 2D slice of residue field."""
    X, Y, values = agent.residue_field.visualize_field(
        z_range=(-3, 3),
        resolution=100,
        slice_dims=(0, 1)
    )
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), values.numpy(), levels=20, cmap='Reds')
    plt.colorbar(label='Residue Intensity')
    plt.xlabel('Latent Dimension 0')
    plt.ylabel('Latent Dimension 1')
    plt.title('Residue Field φ(z)')
    plt.savefig(save_path)
    plt.close()

def plot_trajectory_scores(scores_history):
    """Plot evolution of trajectory scores over time."""
    plt.figure(figsize=(12, 6))
    
    for key in ['F', 'M', 'Phi', 'total']:
        values = [s[key] for s in scores_history]
        plt.plot(values, label=key)
    
    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.title('Trajectory Scoring Components Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory_scores.png')
    plt.close()
```

### Debugging Utilities

```python
def check_agent_health(agent):
    """Check agent for common issues."""
    issues = []
    
    # Check for NaN values
    for name, param in agent.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"NaN values in {name}")
    
    # Check residue field
    stats = agent.get_residue_statistics()
    if stats['total_residue'] < 0:
        issues.append("Negative residue (invariant violation!)")
    
    # Check latent state
    if agent._current_latent is not None:
        if torch.isnan(agent._current_latent.z_gamma).any():
            issues.append("NaN in latent state")
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Agent health check passed")
    
    return len(issues) == 0
```

## Performance Optimization

### Batch Processing

```python
def batch_agent_rollout(agent, envs, max_steps=1000):
    """Run multiple environments in parallel."""
    batch_size = len(envs)
    
    # Reset all
    observations = torch.stack([env.reset() for env in envs])
    agent.reset()
    
    done_mask = torch.zeros(batch_size, dtype=torch.bool)
    
    for step in range(max_steps):
        if done_mask.all():
            break
        
        # Batch action selection
        actions = agent.act(observations)
        
        # Step environments
        for i, env in enumerate(envs):
            if not done_mask[i]:
                obs, harm, done, info = env.step(actions[i])
                observations[i] = obs
                agent.update_residue(harm)
                done_mask[i] = done
```

### GPU Acceleration

```python
# Move agent to GPU
agent = REEAgent.from_config(
    observation_dim=252,
    action_dim=5,
    latent_dim=64
)
agent = agent.to('cuda')

# Ensure observations are on GPU
obs = env.reset().to('cuda')
action = agent.act(obs)
```

### Caching and Memoization

```python
from functools import lru_cache

class OptimizedREEAgent(REEAgent):
    """REE agent with caching for repeated computations."""
    
    @lru_cache(maxsize=100)
    def _cached_residue_eval(self, z_tuple):
        """Cache residue evaluations."""
        z = torch.tensor(z_tuple)
        return self.residue_field.evaluate(z)
```

## Multi-Agent Scenarios

### Multiple REE Agents

```python
def multi_agent_scenario(num_agents=3):
    """Run multiple REE agents in shared environment."""
    env = GridWorld(size=15, num_resources=10, num_hazards=5)
    
    # Create multiple agents
    agents = [
        REEAgent.from_config(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            latent_dim=64
        )
        for _ in range(num_agents)
    ]
    
    # Run episode
    observations = [env.reset() for _ in range(num_agents)]
    
    for step in range(1000):
        # Each agent acts
        for i, agent in enumerate(agents):
            action = agent.act(observations[i])
            obs, harm, done, info = env.step(action)
            agent.update_residue(harm)
            observations[i] = obs
            
            if done:
                break
```

## Integration with Other Frameworks

### OpenAI Gym Integration

```python
import gym

class GymWrapper:
    """Wrap Gym environment for REE."""
    
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
    
    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(obs).float()
    
    def step(self, action):
        action_idx = action.argmax().item()
        obs, reward, done, info = self.env.step(action_idx)
        
        # Convert reward to harm signal
        # Negative reward = harm
        harm_signal = reward
        
        return torch.from_numpy(obs).float(), harm_signal, done, info

# Use with REE
env = GymWrapper('CartPole-v1')
agent = REEAgent.from_config(
    observation_dim=env.observation_dim,
    action_dim=env.action_dim
)
```

## Further Reading

- [Architecture](architecture.md) - Architectural details
- [API Reference](api-reference.md) - Complete API
- [Examples](../examples/) - More code examples
