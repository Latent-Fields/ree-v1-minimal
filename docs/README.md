# REE-v1-Minimal Documentation

Welcome to the comprehensive documentation for the Reflective-Ethical Engine (REE) minimal reference implementation.

## 📖 Documentation Structure

### Getting Started

- **[Getting Started Guide](getting-started.md)** - Installation, setup, and your first agent
  - Prerequisites and installation
  - Quick start example
  - Basic concepts explained
  - Understanding the output
  - Next steps and resources

### Core Documentation

- **[Architecture Guide](architecture.md)** - Deep dive into REE architecture
  - System overview and core principles
  - Multi-timescale latent representation
  - Component descriptions (E1, E2, E3, Residue Field)
  - The canonical REE loop
  - Ethical invariants
  - Design rationale

- **[API Reference](api-reference.md)** - Complete API documentation
  - REEAgent class and methods
  - Latent Stack (LatentState, LatentStack)
  - Predictors (E1, E2)
  - Trajectory Selection (E3)
  - Residue Field
  - Environment (GridWorld)
  - Configuration classes
  - Utility functions

- **[Configuration Guide](configuration.md)** - Configuring REE components
  - Quick configuration
  - Configuration class hierarchy
  - Latent stack configuration
  - Predictor configuration (E1, E2, E3)
  - Residue field configuration
  - Trajectory scoring weights
  - Advanced configuration patterns
  - Environment-specific tuning

### Advanced Topics

- **[Advanced Usage Guide](advanced-usage.md)** - Advanced patterns and techniques
  - Creating custom environments
  - Training components (E1, E2, residue field)
  - Extending the architecture
  - Monitoring and debugging
  - Performance optimization
  - Multi-agent scenarios
  - Integration with other frameworks (Gym, etc.)

### Contributing and Support

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to REE
  - Development setup
  - Coding standards (black, isort, mypy)
  - Testing guidelines
  - Architectural invariants (critical!)
  - Documentation requirements
  - Pull request process

- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
  - Installation issues
  - Import errors
  - Runtime errors (NaN, dimension mismatch, OOM)
  - Performance issues
  - Training problems
  - Environment integration
  - Testing issues
  - Debugging tips

## 🎯 Quick Links by Topic

### For New Users

1. Start here: [Getting Started](getting-started.md)
2. Run your first agent: [Quick Start](getting-started.md#quick-start)
3. Understand the concepts: [Basic Concepts](getting-started.md#basic-concepts)

### For Developers

1. Read: [Architecture Guide](architecture.md)
2. Explore: [API Reference](api-reference.md)
3. Configure: [Configuration Guide](configuration.md)
4. Contribute: [Contributing Guide](CONTRIBUTING.md)

### For Researchers

1. Understand invariants: [Ethical Invariants](architecture.md#ethical-invariants)
2. Learn the theory: [Design Rationale](architecture.md#design-rationale)
3. Extend the system: [Advanced Usage](advanced-usage.md#extending-the-architecture)
4. Integrate your work: [Custom Environments](advanced-usage.md#custom-environments)

## 🔑 Key Concepts

### The REE Loop

Every timestep, the agent executes:

1. **SENSE** - Encode observations
2. **UPDATE** - Update multi-timescale latent state
3. **GENERATE** - Generate candidate trajectories
4. **SCORE** - Score with F (reality) + λM (ethics) + ρΦ (residue)
5. **SELECT** - Choose best trajectory
6. **ACT** - Execute action
7. **RESIDUE** - Accumulate moral cost if harm occurred
8. **OFFLINE** - Periodic integration and learning

### Architectural Invariants

⚠️ **Critical:** These invariants are non-negotiable:

1. **Residue cannot be erased** - Only accumulated and contextualized
2. **Harm via mirror modelling** - Not symbolic rules
3. **Language cannot override embodied harm signals**
4. **Precision is depth-specific** - Not global attention
5. **Perceptual corrigibility** - Higher levels cannot overwrite sensory state

See [Ethical Invariants](architecture.md#ethical-invariants) for details.

### Multi-Timescale Representation

The latent stack operates at different timescales:

- **z_γ (gamma)** - Sensory binding (~10-100ms)
- **z_β (beta)** - Affordances (~100ms-1s)
- **z_θ (theta)** - Sequences (~1-10s)
- **z_δ (delta)** - Motivation (~10s+)

### Trajectory Scoring

Each trajectory ζ is scored by:

```
J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)
```

Where:
- **F(ζ)** - Reality constraint (physical viability)
- **M(ζ)** - Ethical cost (predicted harm)
- **Φ_R(ζ)** - Residue field (path-dependent moral cost)

## 📚 Documentation by Use Case

### "I want to get started quickly"

→ [Getting Started Guide](getting-started.md)

### "I want to understand how it works"

→ [Architecture Guide](architecture.md)

### "I want to use it with my environment"

→ [Custom Environments](advanced-usage.md#custom-environments)

### "I want to train the components"

→ [Training Components](advanced-usage.md#training-components)

### "I want to configure the agent"

→ [Configuration Guide](configuration.md)

### "Something's not working"

→ [Troubleshooting Guide](troubleshooting.md)

### "I want to contribute"

→ [Contributing Guide](CONTRIBUTING.md)

### "I need API details"

→ [API Reference](api-reference.md)

## 🎓 Learning Path

### Beginner Path

1. **Install**: Follow [Getting Started](getting-started.md#installation)
2. **Run**: Try the [Quick Start](getting-started.md#quick-start)
3. **Explore**: Run [examples/basic_agent.py](../examples/basic_agent.py)
4. **Understand**: Read [Basic Concepts](getting-started.md#basic-concepts)

### Intermediate Path

1. **Architecture**: Read [Architecture Guide](architecture.md)
2. **Configure**: Try different [configurations](configuration.md)
3. **Customize**: Create a [custom environment](advanced-usage.md#custom-environments)
4. **Monitor**: Use [monitoring tools](advanced-usage.md#monitoring-and-debugging)

### Advanced Path

1. **Train**: Implement [component training](advanced-usage.md#training-components)
2. **Extend**: Build [custom components](advanced-usage.md#extending-the-architecture)
3. **Optimize**: Apply [performance optimizations](advanced-usage.md#performance-optimization)
4. **Contribute**: Follow [contribution guidelines](CONTRIBUTING.md)

## 🔍 Finding Information

### By Component

- **REEAgent**: [API Reference - Core Agent](api-reference.md#core-agent)
- **Latent Stack**: [API Reference - Latent Stack](api-reference.md#latent-stack)
- **E1 Predictor**: [API Reference - E1](api-reference.md#e1deeppredictor)
- **E2 Predictor**: [API Reference - E2](api-reference.md#e2fastpredictor)
- **E3 Selector**: [API Reference - E3](api-reference.md#e3trajectoryselector)
- **Residue Field**: [API Reference - Residue](api-reference.md#residue-field)
- **GridWorld**: [API Reference - Environment](api-reference.md#environment)

### By Task

- **Installation**: [Getting Started - Installation](getting-started.md#installation)
- **Configuration**: [Configuration Guide](configuration.md)
- **Training**: [Advanced Usage - Training](advanced-usage.md#training-components)
- **Debugging**: [Troubleshooting](troubleshooting.md)
- **Testing**: [Contributing - Testing](CONTRIBUTING.md#testing-guidelines)

## 🎯 Code Examples

See the [examples/](../examples/) directory for working code:

- `basic_agent.py` - Simple agent in GridWorld
- `residue_visualization.py` - Visualize the residue field

## 🧪 Tests

See the [tests/](../tests/) directory for comprehensive tests:

- `test_agent.py` - Agent functionality
- `test_latent_stack.py` - Latent stack operations
- `test_predictors.py` - E1 and E2 predictors
- `test_trajectory.py` - E3 trajectory selection
- `test_residue.py` - Residue field (including invariants!)
- `test_environment.py` - GridWorld environment

## 📝 Additional Resources

### External Links

- [Project Repository](https://github.com/Latent-Fields/ree-v1-minimal)
- [License (CC BY 4.0)](../LICENSE)

### Internal Files

- [README](../README.md) - Project overview
- [pyproject.toml](../pyproject.toml) - Package configuration
- [.gitignore](../.gitignore) - Git ignore rules

## 💡 Tips

- **Use the search**: Most documentation formats support search (Ctrl+F)
- **Follow links**: Documentation is heavily cross-referenced
- **Try examples**: Learn by running and modifying examples
- **Read tests**: Tests show real usage patterns
- **Ask questions**: Open GitHub issues for questions

## 🆘 Getting Help

1. **Check this documentation** - Most questions are answered here
2. **Try troubleshooting** - See [Troubleshooting Guide](troubleshooting.md)
3. **Search issues** - Someone may have asked before
4. **Ask the community** - Open a GitHub issue or discussion

## 📄 Documentation Standards

This documentation follows these principles:

- **Clear examples** - Every concept has code examples
- **Cross-referenced** - Easy navigation between related topics
- **Practical focus** - Emphasis on how-to, not just theory
- **Progressive disclosure** - Start simple, go deeper as needed
- **Maintained** - Updated with code changes

## 🔄 Updates

This documentation is kept in sync with the codebase. If you find:

- Outdated information
- Broken links
- Unclear explanations
- Missing topics

Please open an issue or submit a pull request!

---

**Happy coding with REE! 🚀**
