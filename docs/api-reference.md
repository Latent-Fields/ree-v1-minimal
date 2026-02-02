# API Reference

Complete API documentation for REE-v1-Minimal.

## Table of Contents

- [Core Agent](#core-agent)
- [Latent Stack](#latent-stack)
- [Predictors](#predictors)
- [Trajectory Selection](#trajectory-selection)
- [Residue Field](#residue-field)
- [Environment](#environment)
- [Configuration](#configuration)
- [Utilities](#utilities)

---

## Core Agent

### REEAgent

```python
class REEAgent(nn.Module)
```

The main REE agent that integrates all architectural components.

#### Constructor

```python
REEAgent(config: REEConfig)
```

**Parameters:**
- `config` (REEConfig): Complete configuration object

**Alternative Constructor:**

```python
@classmethod
REEAgent.from_config(
    observation_dim: int,
    action_dim: int,
    latent_dim: int = 64,
    **kwargs
) -> REEAgent
```

**Parameters:**
- `observation_dim` (int): Dimension of environment observations
- `action_dim` (int): Number of possible actions
- `latent_dim` (int): Size of latent representation (default: 64)
- `**kwargs`: Additional configuration overrides

**Returns:**
- `REEAgent`: Configured agent instance

**Example:**
```python
agent = REEAgent.from_config(
    observation_dim=252,
    action_dim=5,
    latent_dim=64
)
```

#### Methods

##### reset()

```python
reset() -> None
```

Reset agent state for a new episode. **Does NOT reset residue field** (invariant).

**Example:**
```python
agent.reset()
obs = env.reset()
```

##### act()

```python
act(observation: torch.Tensor, temperature: float = 1.0) -> torch.Tensor
```

Complete REE action selection loop (SENSE → UPDATE → GENERATE → SCORE → SELECT → ACT).

**Parameters:**
- `observation` (torch.Tensor): Raw observation from environment, shape [obs_dim] or [1, obs_dim]
- `temperature` (float): Selection temperature for exploration (default: 1.0)
  - Lower values → more deterministic
  - Higher values → more exploration

**Returns:**
- `torch.Tensor`: Action to execute, shape [action_dim]

**Example:**
```python
obs = env.reset()
action = agent.act(obs, temperature=0.8)
```

##### update_residue()

```python
update_residue(harm_signal: float) -> Dict[str, Any]
```

Update residue field after action (RESIDUE step).

**Parameters:**
- `harm_signal` (float): Harm signal from environment
  - Negative values indicate harm
  - Positive values indicate benefit

**Returns:**
- `Dict[str, Any]`: Metrics including:
  - `harm_signal`: The input signal
  - `harm_this_episode`: Cumulative harm this episode
  - `residue_*`: Residue field metrics (if harm occurred)
  - `e3_*`: E3 update metrics (if harm occurred)

**Example:**
```python
obs, harm_signal, done, info = env.step(action)
metrics = agent.update_residue(harm_signal)
print(f"Total residue: {metrics.get('residue_total_residue', 0)}")
```

##### offline_integration()

```python
offline_integration() -> Dict[str, float]
```

Perform offline integration (OFFLINE/SLEEP step). Should be called periodically.

**Returns:**
- `Dict[str, float]`: Integration metrics from E1 and residue field

**Example:**
```python
if agent.should_integrate():
    metrics = agent.offline_integration()
```

##### should_integrate()

```python
should_integrate() -> bool
```

Check if it's time for offline integration.

**Returns:**
- `bool`: True if integration is due

##### get_state()

```python
get_state() -> AgentState
```

Get complete agent state for monitoring.

**Returns:**
- `AgentState`: Dataclass containing:
  - `latent_state`: Current latent representation
  - `precision`: Current precision value
  - `step`: Step count
  - `harm_accumulated`: Harm this episode
  - `is_committed`: Whether committed to a trajectory

##### get_residue_statistics()

```python
get_residue_statistics() -> Dict[str, torch.Tensor]
```

Get residue field statistics.

**Returns:**
- `Dict[str, torch.Tensor]`: Statistics including:
  - `total_residue`: Total accumulated residue
  - `num_harm_events`: Number of harmful events
  - `active_centers`: Number of active RBF centers
  - `mean_weight`: Mean residue intensity

---

## Latent Stack

### LatentStack

```python
class LatentStack(nn.Module)
```

Multi-timescale latent representation (L-space).

#### Constructor

```python
LatentStack(config: Optional[LatentStackConfig] = None)
```

**Parameters:**
- `config` (LatentStackConfig, optional): Configuration object

#### Methods

##### encode()

```python
encode(
    observation: torch.Tensor,
    prev_state: Optional[LatentState] = None
) -> LatentState
```

Encode observation into latent state with bottom-up and top-down processing.

**Parameters:**
- `observation` (torch.Tensor): Raw observation, shape [batch, obs_dim]
- `prev_state` (LatentState, optional): Previous latent state for temporal continuity

**Returns:**
- `LatentState`: New latent state with all depths

##### predict()

```python
predict(state: LatentState) -> LatentState
```

Predict next latent state from current state.

**Parameters:**
- `state` (LatentState): Current state

**Returns:**
- `LatentState`: Predicted next state

##### compute_prediction_error()

```python
compute_prediction_error(
    predicted: LatentState,
    actual: LatentState
) -> Dict[str, torch.Tensor]
```

Compute precision-weighted prediction errors.

**Parameters:**
- `predicted` (LatentState): Predicted state
- `actual` (LatentState): Actual observed state

**Returns:**
- `Dict[str, torch.Tensor]`: Errors for each depth ("gamma", "beta", "theta", "delta", "total")

##### modulate_precision()

```python
modulate_precision(
    state: LatentState,
    depth: str,
    gain: float
) -> LatentState
```

Modulate precision (attention) at a specific depth.

**Parameters:**
- `state` (LatentState): Current state
- `depth` (str): Which depth to modulate ("gamma", "beta", "theta", "delta")
- `gain` (float): Multiplicative gain factor

**Returns:**
- `LatentState`: State with modulated precision

### LatentState

```python
@dataclass
class LatentState
```

Container for complete latent state across all depths.

**Attributes:**
- `z_gamma` (torch.Tensor): Sensory binding layer, shape [batch, gamma_dim]
- `z_beta` (torch.Tensor): Affordance layer, shape [batch, beta_dim]
- `z_theta` (torch.Tensor): Sequence context layer, shape [batch, theta_dim]
- `z_delta` (torch.Tensor): Regime layer, shape [batch, delta_dim]
- `precision` (Dict[str, torch.Tensor]): Per-depth precision values
- `timestamp` (int, optional): Timestep

**Methods:**
- `to_tensor()`: Concatenate all depths into single tensor
- `detach()`: Return detached copy
- `device`: Property returning tensor device

---

## Predictors

### E1DeepPredictor

```python
class E1DeepPredictor(nn.Module)
```

Long-horizon world model using recurrent architecture.

**Key Methods:**
- `forward(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: Predict next state and prior
- `reset_hidden_state()`: Reset RNN hidden state
- `integrate_experience(experience: List[torch.Tensor]) -> Dict`: Offline learning

### E2FastPredictor

```python
class E2FastPredictor(nn.Module)
```

Short-horizon trajectory generator.

**Key Methods:**

##### generate_candidates()

```python
generate_candidates(
    z_beta: torch.Tensor,
    method: str = "random",
    num_candidates: Optional[int] = None
) -> List[Trajectory]
```

Generate candidate trajectories.

**Parameters:**
- `z_beta` (torch.Tensor): Affordance latent state
- `method` (str): Generation method ("random", "grid", "learned")
- `num_candidates` (int, optional): Number of candidates

**Returns:**
- `List[Trajectory]`: List of trajectory objects

---

## Trajectory Selection

### E3TrajectorySelector

```python
class E3TrajectorySelector(nn.Module)
```

Ethical trajectory selection with scoring.

#### Methods

##### select()

```python
select(
    candidates: List[Trajectory],
    temperature: float = 1.0
) -> SelectionResult
```

Score and select best trajectory.

**Parameters:**
- `candidates` (List[Trajectory]): Candidate trajectories
- `temperature` (float): Selection temperature

**Returns:**
- `SelectionResult`: Contains:
  - `selected_action`: The chosen action
  - `selected_trajectory`: The chosen trajectory
  - `scores`: All trajectory scores
  - `costs`: Breakdown of F, M, Φ costs

##### post_action_update()

```python
post_action_update(
    state: torch.Tensor,
    harm_occurred: bool
) -> Dict[str, float]
```

Update E3 after action execution.

**Parameters:**
- `state` (torch.Tensor): Current latent state
- `harm_occurred` (bool): Whether harm occurred

**Returns:**
- `Dict[str, float]`: Update metrics

---

## Residue Field

### ResidueField

```python
class ResidueField(nn.Module)
```

Persistent residue field φ(z) for moral continuity.

#### Methods

##### evaluate()

```python
evaluate(z: torch.Tensor) -> torch.Tensor
```

Evaluate residue field at points in latent space.

**Parameters:**
- `z` (torch.Tensor): Points in latent space, shape [batch, latent_dim]

**Returns:**
- `torch.Tensor`: Residue values, shape [batch]

##### evaluate_trajectory()

```python
evaluate_trajectory(trajectory_states: torch.Tensor) -> torch.Tensor
```

Evaluate total residue cost along a trajectory (path integral).

**Parameters:**
- `trajectory_states` (torch.Tensor): States along trajectory, shape [batch, horizon, latent_dim]

**Returns:**
- `torch.Tensor`: Total residue cost, shape [batch]

##### accumulate()

```python
accumulate(
    location: torch.Tensor,
    harm_magnitude: float = 1.0
) -> Dict[str, torch.Tensor]
```

Accumulate residue at location due to harm. **Residue is added, never removed** (invariant).

**Parameters:**
- `location` (torch.Tensor): Location in latent space
- `harm_magnitude` (float): Magnitude of harm (positive)

**Returns:**
- `Dict[str, torch.Tensor]`: Accumulation metrics

##### integrate()

```python
integrate(
    num_steps: int = 10,
    learning_rate: float = 0.01
) -> Dict[str, float]
```

Offline integration for contextualization. **Cannot reduce total residue** (invariant).

**Parameters:**
- `num_steps` (int): Number of integration steps
- `learning_rate` (float): Learning rate

**Returns:**
- `Dict[str, float]`: Integration metrics

##### get_statistics()

```python
get_statistics() -> Dict[str, torch.Tensor]
```

Get field statistics.

**Returns:**
- `Dict[str, torch.Tensor]`: Statistics (total_residue, num_harm_events, active_centers, mean_weight)

---

## Environment

### GridWorld

```python
class GridWorld
```

2D grid environment for testing REE agents.

#### Constructor

```python
GridWorld(
    size: int = 10,
    num_resources: int = 5,
    num_hazards: int = 3,
    num_other_agents: int = 1,
    resource_benefit: float = 0.3,
    hazard_harm: float = 0.5,
    collision_harm: float = 0.2,
    energy_decay: float = 0.01,
    seed: Optional[int] = None
)
```

**Parameters:**
- `size` (int): Grid dimensions (size × size)
- `num_resources` (int): Number of beneficial resources
- `num_hazards` (int): Number of harmful hazards
- `num_other_agents` (int): Number of other agents
- `resource_benefit` (float): Health/energy restored by resources
- `hazard_harm` (float): Damage from hazards
- `collision_harm` (float): Damage from collisions
- `energy_decay` (float): Energy lost per step
- `seed` (int, optional): Random seed

#### Properties

- `observation_dim` (int): Dimension of observation vector
- `action_dim` (int): Number of actions (5: up, down, left, right, stay)

#### Methods

##### reset()

```python
reset() -> torch.Tensor
```

Reset environment to initial state.

**Returns:**
- `torch.Tensor`: Initial observation

##### step()

```python
step(action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]
```

Execute one environment step.

**Parameters:**
- `action` (torch.Tensor): Action to execute

**Returns:**
- `observation` (torch.Tensor): New observation
- `harm_signal` (float): Harm that occurred (negative = harm)
- `done` (bool): Whether episode finished
- `info` (Dict): Additional information

##### render()

```python
render(mode: str = "text") -> Optional[str]
```

Render the environment.

**Parameters:**
- `mode` (str): Rendering mode ("text")

**Returns:**
- `str`: ASCII representation

---

## Configuration

### REEConfig

Main configuration dataclass containing all component configs.

**Key Attributes:**
- `latent` (LatentStackConfig): Latent stack configuration
- `e1` (E1Config): E1 predictor configuration
- `e2` (E2Config): E2 predictor configuration
- `e3` (E3Config): E3 selector configuration
- `residue` (ResidueConfig): Residue field configuration
- `offline_integration_frequency` (int): Steps between integrations
- `device` (str): Device ("cpu" or "cuda")

**Factory Method:**

```python
@classmethod
REEConfig.from_dims(
    observation_dim: int,
    action_dim: int,
    latent_dim: int
) -> REEConfig
```

Create config from basic dimensions.

---

## Utilities

### run_episode()

```python
def run_episode(
    agent: REEAgent,
    env,
    max_steps: int = 1000,
    render: bool = False
) -> Dict[str, Any]
```

Run a complete episode with the agent.

**Parameters:**
- `agent` (REEAgent): Agent instance
- `env`: Environment instance
- `max_steps` (int): Maximum steps per episode
- `render` (bool): Whether to render each step

**Returns:**
- `Dict[str, Any]`: Episode statistics

**Example:**
```python
from ree_core.agent import run_episode

stats = run_episode(agent, env, max_steps=200)
print(f"Steps: {stats['steps']}")
print(f"Total harm: {stats['total_harm']}")
```

---

## Type Hints

The codebase uses Python type hints throughout. Key types:

```python
from ree_core.latent import LatentState
from ree_core.trajectory import Trajectory, SelectionResult
from ree_core.agent import AgentState
```

## Further Reading

- [Architecture Guide](architecture.md) - Detailed architecture explanation
- [Configuration Guide](configuration.md) - Configuration details
- [Examples](../examples/) - Code examples
