# Contributing to Bull Machine

Thank you for your interest in contributing to Bull Machine! This document provides guidelines and instructions for contributing to this institutional-grade algorithmic trading framework.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Questions and Support](#questions-and-support)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful, professional, and collaborative in all interactions.

## Ways to Contribute

### 🐛 Bug Reports

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS, dependencies)
- Relevant logs or error messages

### 💡 Feature Requests

We welcome feature suggestions! Please include:
- Clear description of the proposed feature
- Use case and motivation
- Potential implementation approach (if applicable)
- Impact on existing functionality

### 📝 Documentation

Documentation improvements are always appreciated:
- Fix typos or clarify existing docs
- Add examples or tutorials
- Document undocumented features
- Improve README or docstrings

### 🔧 Code Contributions

Code contributions should:
- Fix bugs or add features
- Include tests
- Follow code style guidelines
- Update documentation as needed

## Development Setup

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip or conda package manager
- Git

### Installation

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Bull-machine-.git
   cd Bull-machine-
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt

   # Development dependencies
   pip install -e ".[dev]"
   ```

4. **Verify installation**
   ```bash
   pytest tests/unit/  # Run unit tests
   ```

### Project Structure

```
Bull-machine-/
├── engine/           # Core trading engine
│   ├── archetypes/   # Trading pattern detectors
│   ├── context/      # Regime detection
│   ├── optimization/ # Parameter optimization
│   └── portfolio/    # Position sizing & allocation
├── configs/          # Configuration files
├── tests/            # Test suite
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
├── bin/              # Development scripts (not committed)
└── docs/             # Documentation
```

## Code Style Guidelines

We use automated tools to ensure consistent code quality:

### Linting and Formatting

- **ruff**: Fast Python linter (replaces flake8, isort)
- **black**: Code formatter (120 char line length)
- **mypy**: Static type checker

### Running Code Quality Tools

```bash
# Format code with black
black engine/ tests/

# Lint with ruff
ruff check engine/ tests/

# Type check with mypy
mypy engine/
```

### Style Rules

1. **Line Length**: 120 characters (pragmatic limit)
2. **Type Hints**: Required for all function signatures
3. **Docstrings**: Use Google-style docstrings for public APIs
4. **Imports**: Sorted by ruff (stdlib → third-party → local)
5. **Naming Conventions**:
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private members: `_leading_underscore`

### Example

```python
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from engine.context import RegimeDetector


def calculate_position_size(
    fusion_score: float,
    adx_value: float,
    max_position: float = 1.0,
) -> Tuple[float, str]:
    """Calculate adaptive position size based on confluence and trend strength.

    Args:
        fusion_score: Multi-domain confluence score (0.0-1.0)
        adx_value: ADX trend strength indicator
        max_position: Maximum position size cap

    Returns:
        Tuple of (position_size, sizing_method)
    """
    if fusion_score < 0.6:
        return 0.0, "rejected_low_confluence"

    base_size = fusion_score * max_position
    adx_multiplier = min(adx_value / 25.0, 1.5)

    return min(base_size * adx_multiplier, max_position), "adaptive"
```

## Testing Requirements

All code contributions must include tests:

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions/classes in isolation
   - Fast execution (<1s per test)
   - Mock external dependencies
   - Required for all new features

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - May use real data or fixtures
   - Slower execution acceptable
   - Required for system-level changes

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_regime_allocator.py

# Run with coverage
pytest --cov=engine tests/

# Run fast tests only (exclude slow marker)
pytest -m "not slow"
```

### Writing Tests

```python
import pytest
from engine.portfolio import RegimeAllocator


def test_regime_allocator_soft_gating():
    """Test soft gating prevents hard zeros with 1% exploration floor."""
    allocator = RegimeAllocator(exploration_floor=0.01)

    # Even with zero regime score, should get exploration allocation
    weights = allocator.allocate(
        archetype_scores={"S1": 0.95},
        regime_scores={"S1": 0.0},  # Hard mismatch
    )

    assert weights["S1"] >= 0.01  # Exploration floor
    assert weights["S1"] < 0.95   # Should be penalized
```

### Coverage Requirements

- New features: ≥90% coverage
- Bug fixes: Add regression test
- Critical paths: 100% coverage (entry/exit logic, position sizing)

## Commit Message Conventions

We use **Conventional Commits** format for clear, semantic commit history:

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code refactoring (no functional changes)
- `perf`: Performance improvements
- `chore`: Maintenance tasks (deps, configs)
- `ci`: CI/CD changes

### Scopes

- `archetypes`: Trading pattern detectors
- `optimization`: Parameter optimization
- `portfolio`: Position sizing & allocation
- `regime`: Regime detection
- `temporal`: Temporal intelligence features
- `tests`: Test suite

### Examples

```bash
# Feature
git commit -m "feat(archetypes): add long squeeze detector for overleveraged longs"

# Bug fix
git commit -m "fix(regime): use detector direction instead of config default

Fixes critical bug where S5 ignored short direction from detector.
Restores 68 short signals that were incorrectly filtered."

# Documentation
git commit -m "docs: add CONTRIBUTING.md with development guidelines"

# Chore
git commit -m "chore: update .gitignore to allow docs diagrams"
```

### Co-Authorship

For AI-assisted commits, include co-author:

```
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow code style guidelines
   - Add tests
   - Update documentation

3. **Run quality checks**
   ```bash
   black engine/ tests/
   ruff check engine/ tests/
   mypy engine/
   pytest
   ```

4. **Commit with conventional commits**
   ```bash
   git commit -m "feat(scope): description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

### Submitting the PR

1. **Open a pull request** to `main` branch
2. **Fill out the PR template** with:
   - Summary of changes
   - Motivation and context
   - Testing performed
   - Screenshots (if UI changes)
   - Breaking changes (if any)

3. **Wait for CI checks** to pass:
   - All tests passing
   - Code quality checks (ruff, mypy)
   - Coverage requirements met

4. **Address review feedback**
   - Respond to comments
   - Make requested changes
   - Push updates (don't force-push)

### PR Review Process

- Maintainers will review within 3-5 business days
- At least 1 approval required for merge
- CI checks must pass
- No merge conflicts

### After Merge

- Delete your feature branch
- Pull latest main: `git pull upstream main`
- Your contribution will be included in the next release

## Branch Naming Conventions

Use descriptive branch names with prefixes:

- `feature/` - New features (e.g., `feature/gann-cycles`)
- `fix/` - Bug fixes (e.g., `fix/s5-direction-bug`)
- `docs/` - Documentation (e.g., `docs/add-contributing`)
- `refactor/` - Code refactoring (e.g., `refactor/logic-v2-adapter`)
- `test/` - Test additions (e.g., `test/soft-gating-integration`)
- `chore/` - Maintenance (e.g., `chore/update-dependencies`)

## Questions and Support

### Getting Help

- **Documentation**: Check `docs/` directory and README
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions

### Contact

- **Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Issues
- **Security Issues**: Email team@bullmachine.ai (private disclosure)

---

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- CHANGELOG.md

Thank you for contributing to Bull Machine! 🚀
