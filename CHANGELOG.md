# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite including:
  - **Getting Started Guide** (`docs/getting-started.md`) - Installation, quick start, and basic concepts
  - **Architecture Guide** (`docs/architecture.md`) - Detailed architectural overview with 14KB of content
  - **API Reference** (`docs/api-reference.md`) - Complete API documentation for all components
  - **Configuration Guide** (`docs/configuration.md`) - Configuration options and tuning guidelines
  - **Advanced Usage Guide** (`docs/advanced-usage.md`) - Advanced patterns, custom environments, and training
  - **Contributing Guide** (`docs/CONTRIBUTING.md`) - Development setup, coding standards, and PR process
  - **Troubleshooting Guide** (`docs/troubleshooting.md`) - Common issues and solutions
  - **Documentation Index** (`docs/README.md`) - Comprehensive navigation and quick reference
- Updated main README with documentation links and key features section
- Over 4,100 lines of comprehensive documentation
- Code examples in all documentation sections
- Cross-referenced documentation for easy navigation

### Documentation Highlights
- Complete API reference for REEAgent, LatentStack, Predictors (E1, E2, E3), and Residue Field
- Detailed explanation of ethical invariants and why they matter
- Step-by-step guides for custom environments and component training
- Performance optimization strategies
- Multi-agent scenario examples
- Integration guides for Gym and other frameworks
- Comprehensive troubleshooting section with common errors and solutions
- Development workflow and testing guidelines for contributors

## [0.1.0] - Initial Release

### Added
- Core REE agent implementation
- Multi-timescale latent stack (L-space)
- E1 deep predictor for long-horizon world modeling
- E2 fast predictor for trajectory generation
- E3 trajectory selector with ethical scoring
- Residue field with RBF-based persistent memory
- GridWorld test environment
- Basic examples (basic_agent.py, residue_visualization.py)
- Comprehensive test suite
- PyPI package configuration
