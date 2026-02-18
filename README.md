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

## Documentation

📚 **[Complete Documentation](docs/)** - Comprehensive guides and references

### Quick Links

- **[Quick Reference](docs/quick-reference.md)** ⚡ - Cheat sheet for common operations
- **[Getting Started](docs/getting-started.md)** - Quick setup and first steps
- **[Architecture Guide](docs/architecture.md)** - Detailed architectural overview
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Configuration](docs/configuration.md)** - Configuration options and tuning
- **[Advanced Usage](docs/advanced-usage.md)** - Advanced patterns and techniques
- **[Contributing](docs/CONTRIBUTING.md)** - How to contribute to the project
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

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

## Experiment Pack v1 Emission

Run the experiment harness and emit contract-compliant artifacts:

```bash
python experiments/run.py \
  --suite baseline_explicit_cost \
  --seed 7 \
  --max-steps 200 \
  --claim-id MECH-056 \
  --claim-id Q-011 \
  --output-root evidence/experiments
```

Output layout:

```text
<output_root>/<experiment_type>/runs/<run_id>/
  manifest.json
  metrics.json
  summary.md
  jepa_adapter_signals.v1.json   # required for JEPA-backed harness runs
  traces/    # optional
  media/     # optional
```

Notes:
- `--output-root` overrides output location.
- If `--output-root` is omitted, `REE_EXPERIMENT_OUTPUT_ROOT` is used.
- If neither is set, output defaults to `runs/`.
- `run_id` is deterministic from `timestamp_utc + suite + seed` unless `--run-id` is provided.
- claim linkage can be overridden with `--claim-id`, `--evidence-class`, and `--evidence-direction`.

Field guarantees:
- `manifest.json` uses schema version `experiment_pack/v1`.
- `metrics.json` uses schema version `experiment_pack_metrics/v1`.
- `metrics.values` contains numeric values only, keyed by stable snake_case metric IDs.
- `manifest.status` is `PASS` or `FAIL`; known failures are surfaced in `failure_signatures`.
- `manifest.json` includes `claim_ids_tested`, `evidence_class`, `evidence_direction`, `producer_capabilities`, and `environment`.
- `environment` includes: `env_id`, `env_version`, `dynamics_hash`, `reward_hash`, `observation_hash`, `config_hash`, `tier`.
- JEPA-backed harness runs (`runner.name = ree-v1-minimal-harness`) include:
  - `manifest.artifacts.adapter_signals_path = "jepa_adapter_signals.v1.json"`
  - `jepa_adapter_signals.v1.json` with required keys:
    - `schema_version`, `experiment_type`, `run_id`
    - `adapter.{name,version}`
    - `stream_presence`
    - `pe_latent_fields` (includes `mean` and `p95`)
    - `uncertainty_estimator`
    - `signal_metrics` with required metric keys/ranges

MECH-056 extension:
- when `claim_ids_tested` includes `MECH-056`, metrics include:
  - `trajectory_commit_channel_usage_count`
  - `perceptual_sampling_channel_usage_count`
  - `structural_consolidation_channel_usage_count`
  - `precommit_semantic_overwrite_events`
  - `structural_bias_magnitude`
  - `structural_bias_rate`
  - `environment_shortcut_leakage_events`
  - `environment_unobservable_critical_state_rate`
  - `environment_controllability_score`
  - `environment_transition_consistency_rate`
- summary includes channel escalation order and trigger rationale for non-primary channel activations.

Example output pack:
- `/Users/dgolden/Documents/GitHub/ree-v1-minimal/examples/experiment_pack_example/claim_probe_mech_056/runs/2026-02-13T090000Z_example/`

CI drift guard:

```bash
EXPERIMENT_PACK_ROOT=tests/fixtures/experiment_pack_v1 python scripts/validate_experiment_packs.py
```

Cross-repo contract lockstep:
- lock file: `contracts/ree_assembly_contract_lock.v1.json`
- vendored contract schemas:
  - `contracts/ree_assembly/schemas/v1/manifest.schema.json`
  - `contracts/ree_assembly/schemas/v1/metrics.schema.json`
  - `contracts/ree_assembly/schemas/v1/jepa_adapter_signals.v1.json`
- CI fails when either:
  - lock-file hashes do not match vendored schemas
  - emitted packs fail required manifest/metrics/adapter checks

Contract update procedure:
1. pick REE_assembly source commit to sync.
2. copy schemas from that commit into `contracts/ree_assembly/schemas/v1/`.
3. compute sha256 and update `contracts/ree_assembly_contract_lock.v1.json`.
4. refresh fixture/example packs if required fields changed.
5. run `EXPERIMENT_PACK_ROOT=tests/fixtures/experiment_pack_v1 python scripts/validate_experiment_packs.py`.

Weekly producer handoff report (parity/backstop lane):
- source template: `/Users/dgolden/Documents/GitHub/REE_assembly/evidence/planning/WEEKLY_HANDOFF_TEMPLATE.md`
- current-cycle report path: `evidence/planning/weekly_handoff/latest.md`
- generate from latest bridging qualification runs:

```bash
python3 scripts/generate_weekly_handoff.py \
  --run-root runs/bridging_qualification_056_058_059_060 \
  --output evidence/planning/weekly_handoff/latest.md
```

- validate required sections/columns:

```bash
python3 scripts/validate_weekly_handoff.py --input evidence/planning/weekly_handoff/latest.md
```

- deterministic generation behavior:
  - rows are sorted by `experiment_type` then `run_id`.
  - `generated_utc` defaults to max observed run timestamp unless overridden.
  - parity note compares claim-summary directionality against latest `ree-v2` handoff when available; otherwise emits `N/A`.
  - Local Compute Options Watch defaults to `N/A` for parity/backstop unless explicitly overridden.

Ingestion compatibility check (from `REE_assembly` checkout):

```bash
python3 evidence/experiments/scripts/build_experiment_indexes.py --root /path/to/your/output_root
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

## Learn More

- 📖 **[Documentation](docs/)** - Complete guides and API reference
- 🎯 **[Examples](examples/)** - Working code examples
- 🧪 **[Tests](tests/)** - Test suite and usage patterns
- 🤝 **[Contributing](docs/CONTRIBUTING.md)** - How to contribute

## Key Features

✅ **Multi-timescale Latent Representation**: Hierarchical state spanning perception to motivation  
✅ **Ethical Path-Dependence**: Geometric residue field makes moral cost trajectory-dependent  
✅ **Architectural Invariants**: Residue cannot be erased, ensuring ethical continuity  
✅ **Predictive Processing**: Unified framework based on precision-weighted prediction errors  
✅ **Extensible Design**: Easy to integrate with custom environments and components  


## License

Apache License 2.0 (see `LICENSE`).

## Citation

- Cite this repository using `CITATION.cff`.
- For canonical architectural attribution, cite Daniel Golden's REE specification in `https://github.com/Latent-Fields/REE_assembly/` (also captured as the preferred citation in `CITATION.cff`).
