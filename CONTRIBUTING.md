# Contributing to MemPack

Thank you for your interest in contributing to MemPack! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mempack/mempack.git
   cd mempack
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**
   ```bash
   make test
   ```

5. **Run linting**
   ```bash
   make lint
   ```

## Code Style

MemPack follows these coding standards:

- **Python 3.10+** with type hints
- **Black** for code formatting
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking
- **Google-style docstrings**

### Pre-commit Hooks

Install pre-commit hooks to automatically format and lint your code:

```bash
pip install pre-commit
pre-commit install
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_types.py

# Run with coverage
make test-cov
```

### Test Structure

- `tests/unit/` - Unit tests for individual modules
- `tests/e2e/` - End-to-end integration tests

### Writing Tests

- Write tests for new features
- Maintain test coverage above 85%
- Use descriptive test names
- Include both positive and negative test cases

## Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run the test suite** to ensure nothing is broken
6. **Submit a pull request** with a clear description

### Pull Request Guidelines

- Use clear, descriptive commit messages
- Keep PRs focused on a single feature or bugfix
- Include tests for new functionality
- Update documentation for user-facing changes
- Ensure all CI checks pass

## Bug Reports

When reporting bugs, please include:

- MemPack version
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

## Feature Requests

For feature requests, please:

- Check existing issues first
- Provide a clear use case
- Explain the expected behavior
- Consider implementation complexity

## Documentation

- Update docstrings for new functions/classes
- Add examples for new features
- Update README.md for user-facing changes
- Keep API documentation current

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Publish to PyPI

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. By participating, you agree to uphold this code.

## Questions?

Feel free to open an issue for questions or discussions about contributing to MemPack.
