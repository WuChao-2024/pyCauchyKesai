# Contributing to pyCauchyKesai

Thank you for your interest in contributing to pyCauchyKesai! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Linux aarch64 (Horizon RDK series development boards) for hardware testing
- Python >= 3.10
- CMake >= 3.18
- C++17 compiler (gcc/g++)
- Git

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone git@github.com:WuChao-2024/pyCauchyKesai.git
cd pyCauchyKesai
```

2. Install development dependencies:
```bash
pip install pytest pytest-cov ruff mypy
```

3. Build the package:
```bash
cd Nash
pip wheel .
pip install pycauchykesai-*.whl
```

## Code Style

### Python Code

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use `ruff` for linting:
```bash
ruff check Nash/pyCauchyKesai/
```

### C++ Code

- Follow Google C++ Style Guide with modifications:
  - Use 4 spaces for indentation (not 2)
  - Maximum line length: 120 characters
  - Use `snake_case` for variables and functions
  - Use `PascalCase` for classes
- Use `clang-format` for formatting (config provided in `.clang-format`)

### Commit Messages

- Use conventional commit format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions/changes
  - `chore:` for maintenance tasks

Example:
```
feat: add support for dynamic batch size

- Implement batch size detection in constructor
- Update validation logic to handle variable batch
- Add tests for batch size edge cases
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyCauchyKesai --cov-report=html

# Run specific test file
pytest tests/test_python_wrapper.py
```

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names: `test_<functionality>_<scenario>`
- Mock hardware dependencies using fixtures in `conftest.py`
- Aim for >80% code coverage for Python code

### Hardware Tests

Hardware-dependent tests should be placed in a separate `tests_hardware/` directory and run manually on target devices.

## Pull Request Process

1. **Fork and Branch**
   - Fork the repository
   - Create a feature branch: `git checkout -b feat/your-feature-name`

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Locally**
   - Run all tests: `pytest`
   - Check code style: `ruff check`
   - Verify type hints: `mypy Nash/pyCauchyKesai/`

4. **Commit and Push**
   - Commit with conventional commit messages
   - Push to your fork: `git push origin feat/your-feature-name`

5. **Create Pull Request**
   - Open a PR against the `main` branch
   - Fill out the PR template
   - Link related issues
   - Wait for CI checks to pass

6. **Code Review**
   - Address reviewer feedback
   - Keep the PR focused and atomic
   - Squash commits if requested

## Areas for Contribution

### High Priority

- **Performance optimization**: Profile and optimize hot paths
- **Documentation**: Improve API docs, add more examples
- **Testing**: Increase test coverage, add edge case tests
- **Error handling**: Improve error messages and recovery

### Medium Priority

- **Type stubs**: Enhance `.pyi` files with more detailed types
- **Examples**: Add real-world usage examples
- **Benchmarking**: Create comprehensive benchmark suite
- **CI/CD**: Improve GitHub Actions workflows

### Low Priority

- **Code cleanup**: Remove commented code, improve naming
- **Logging**: Add structured logging support
- **Profiling tools**: Add built-in profiling utilities

## Architecture Guidelines

### C++ Layer

- Keep C++ code minimal and focused on hardware interaction
- Use RAII for resource management
- Release GIL during blocking operations
- Use `std::atomic` for thread-safe state

### Python Layer

- Keep Python wrapper thin and focused on usability
- Provide both dict and formatted output for Agent-friendly APIs
- Use type hints for all public APIs
- Handle edge cases gracefully

### Thread Safety

- All public APIs must be thread-safe
- Use atomic operations for shared state
- Document thread-safety guarantees

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Keep Chinese and English docs in sync

## Questions?

- Open an issue for questions
- Join discussions in existing issues
- Contact maintainers: WuChao (D-Robotics)

## License

By contributing, you agree that your contributions will be licensed under the GNU AGPL v3 License.
