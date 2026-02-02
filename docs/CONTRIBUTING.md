# Contributing to REE-v1-Minimal

Thank you for your interest in contributing to the Reflective-Ethical Engine (REE) project!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Architectural Invariants](#architectural-invariants)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project is committed to providing a welcoming and respectful environment for all contributors. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Code samples** demonstrating the issue
- **Environment details** (Python version, OS, dependencies)

**Example Bug Report:**

```markdown
**Title:** ResidueField.accumulate raises error with batch inputs

**Description:**
When calling accumulate() with batched latent states, the function crashes.

**To Reproduce:**
```python
from ree_core.residue import ResidueField
import torch

field = ResidueField()
z_batch = torch.randn(10, 64)  # Batch of 10
field.accumulate(z_batch, harm_magnitude=1.0)  # Crashes here
```

**Expected:** Should handle batch or average over batch
**Actual:** TypeError: ...

**Environment:**
- Python 3.10
- PyTorch 2.0.1
- REE-v1-minimal main branch
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear use case** for the enhancement
- **Detailed description** of the proposed functionality
- **Examples** of how it would be used
- **Compatibility** with REE architectural invariants

### Contributing Code

We welcome contributions including:

- Bug fixes
- New features (that respect REE invariants)
- Performance improvements
- Documentation improvements
- Additional examples
- Test coverage improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR-USERNAME/ree-v1-minimal.git
cd ree-v1-minimal
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Development Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev,viz]"
```

This installs:
- Core dependencies (torch, numpy, etc.)
- Development tools (pytest, black, isort, mypy)
- Visualization tools (matplotlib, seaborn)

### 4. Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/ -v
```

### 5. Create a Branch

```bash
# Create a branch for your work
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## Coding Standards

### Python Style

We follow PEP 8 with these tools:

#### Black (Code Formatting)

```bash
# Format all code
black ree_core/ tests/ examples/

# Check without making changes
black --check ree_core/
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311", "py312"]
```

#### isort (Import Sorting)

```bash
# Sort imports
isort ree_core/ tests/ examples/

# Check without making changes
isort --check ree_core/
```

#### mypy (Type Checking)

```bash
# Run type checker
mypy ree_core/
```

### Code Quality Checklist

Before submitting, ensure:

- ✓ Code is formatted with `black`
- ✓ Imports are sorted with `isort`
- ✓ Type hints are used for function signatures
- ✓ Docstrings follow Google/NumPy style
- ✓ No unused imports or variables
- ✓ Tests pass: `pytest tests/ -v`
- ✓ Type checking passes: `mypy ree_core/`

### Docstring Style

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Longer description if needed. Can span multiple lines.
    Explain the purpose and behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative

    Example:
        >>> example_function(5, "test")
        True
    """
    pass
```

### Type Hints

Use type hints throughout:

```python
from typing import Optional, List, Dict, Tuple
import torch

def process_latent(
    z: torch.Tensor,
    history: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Process latent state with optional history."""
    ...
```

## Testing Guidelines

### Writing Tests

Tests are in the `tests/` directory. Use pytest:

```python
# tests/test_new_feature.py
import pytest
import torch
from ree_core import REEAgent

def test_agent_creation():
    """Test that agent can be created with default config."""
    agent = REEAgent.from_config(
        observation_dim=100,
        action_dim=4,
        latent_dim=32
    )
    assert agent is not None
    assert agent.config.latent.latent_dim == 32

def test_agent_reset():
    """Test agent reset functionality."""
    agent = REEAgent.from_config(100, 4, 32)
    agent.reset()
    
    state = agent.get_state()
    assert state.step == 0
    assert state.harm_accumulated == 0.0

def test_residue_persistence():
    """Test that residue persists across resets (invariant)."""
    agent = REEAgent.from_config(100, 4, 32)
    agent.reset()
    
    # Cause harm
    z = torch.randn(1, 32)
    agent.residue_field.accumulate(z, harm_magnitude=1.0)
    
    residue_before = agent.get_residue_statistics()['total_residue']
    
    # Reset agent
    agent.reset()
    
    residue_after = agent.get_residue_statistics()['total_residue']
    
    # Residue should persist (invariant!)
    assert residue_before == residue_after

@pytest.mark.parametrize("latent_dim", [32, 64, 128])
def test_different_latent_dims(latent_dim):
    """Test agent with different latent dimensions."""
    agent = REEAgent.from_config(100, 4, latent_dim)
    assert agent.config.latent.latent_dim == latent_dim
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agent.py -v

# Run specific test
pytest tests/test_agent.py::test_agent_creation -v

# Run with coverage
pytest tests/ -v --cov=ree_core --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

### Test Coverage

Aim for >80% test coverage:

```bash
pytest tests/ --cov=ree_core --cov-report=term-missing
```

### Testing Checklist

- ✓ All new code has tests
- ✓ Tests pass locally: `pytest tests/ -v`
- ✓ Edge cases are covered
- ✓ Invariants are tested (especially residue persistence)
- ✓ Tests are fast (< 1 second each, unless marked slow)

## Architectural Invariants

When contributing, you **MUST** respect these non-negotiable invariants:

### 1. Residue Cannot Be Erased

```python
# ✓ CORRECT
residue_field.accumulate(location, harm_magnitude)

# ✗ INCORRECT - violates invariant
# residue_field.reset()  # Don't add this!
# residue_field.clear()  # Don't add this!
# residue_field.subtract(...)  # Don't add this!
```

### 2. Residue Only Increases

```python
# The total_residue buffer should only increase
assert new_total_residue >= old_total_residue
```

### 3. Agent Reset Does Not Reset Residue

```python
# When implementing any reset functionality:
def reset(self):
    self._current_latent = self.latent_stack.init_state(...)
    self._step_count = 0
    # ... reset other state ...
    
    # DO NOT reset residue field!
    # self.residue_field.reset()  # NO!
```

### 4. Harm Detection via Mirror Modelling

Harm should come from:
- Homeostatic signals (health, energy)
- Mirror models of other agents
- Prediction errors in self-models

NOT from:
- Symbolic rules
- Language overrides
- External labels (unless they reflect actual harm)

### 5. Precision is Depth-Specific

```python
# ✓ CORRECT - depth-specific precision
latent_stack.modulate_precision(state, depth="beta", gain=1.5)

# ✗ INCORRECT - global precision
# agent.global_precision = 0.8  # Don't add this!
```

### Invariant Testing

Always add tests for invariants:

```python
def test_residue_cannot_decrease():
    """Test that residue never decreases (invariant)."""
    agent = REEAgent.from_config(100, 4, 32)
    
    for _ in range(100):
        # Store current residue
        stats = agent.get_residue_statistics()
        old_residue = stats['total_residue'].item()
        
        # Accumulate more residue
        z = torch.randn(1, 32)
        agent.residue_field.accumulate(z, harm_magnitude=0.5)
        
        # Check it increased
        new_residue = agent.get_residue_statistics()['total_residue'].item()
        assert new_residue >= old_residue, "Residue decreased (invariant violated!)"
```

## Documentation

### Code Documentation

All public APIs should have:
- Clear docstrings
- Type hints
- Usage examples in docstrings

### Documentation Files

When adding features, update:
- `docs/api-reference.md` - API documentation
- `docs/architecture.md` - If architecture changes
- `README.md` - If it affects quick start
- `CHANGELOG.md` - Document changes

### Documentation Style

- Use clear, concise language
- Provide code examples
- Link to related documentation
- Explain *why*, not just *what*

## Pull Request Process

### Before Submitting

1. **Test your changes**:
   ```bash
   pytest tests/ -v
   ```

2. **Format your code**:
   ```bash
   black ree_core/ tests/ examples/
   isort ree_core/ tests/ examples/
   ```

3. **Type check**:
   ```bash
   mypy ree_core/
   ```

4. **Update documentation** if needed

5. **Add tests** for new functionality

### Pull Request Template

Use this template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Test coverage maintained/improved

## Invariants
- [ ] Residue persistence respected
- [ ] No residue erasure or reset
- [ ] Depth-specific precision maintained
- [ ] Harm detection via mirror modelling

## Documentation
- [ ] Docstrings updated
- [ ] API reference updated (if needed)
- [ ] README updated (if needed)
- [ ] Examples added/updated (if needed)

## Checklist
- [ ] Code formatted with black
- [ ] Imports sorted with isort
- [ ] Type hints added
- [ ] Tests pass
- [ ] Documentation updated
```

### Review Process

1. Submit pull request
2. Automated tests run (CI)
3. Code review by maintainers
4. Address feedback
5. Approval and merge

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add support for batch residue accumulation"
git commit -m "Fix NaN handling in latent stack encoder"
git commit -m "Update API docs for E3 trajectory selector"

# Less good
git commit -m "Fix bug"
git commit -m "Update"
git commit -m "WIP"
```

## Development Workflow

```bash
# 1. Create branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit code ...

# 3. Test
pytest tests/ -v

# 4. Format
black ree_core/
isort ree_core/

# 5. Commit
git add .
git commit -m "Add my feature"

# 6. Push
git push origin feature/my-feature

# 7. Create pull request on GitHub
```

## Getting Help

- **Documentation**: Check [docs/](.) first
- **Examples**: See [examples/](../examples/)
- **Issues**: Search existing issues
- **Discussions**: Start a GitHub discussion
- **Questions**: Open an issue with "question" label

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

Thank you for contributing to REE!
