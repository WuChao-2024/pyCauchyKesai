# pyCauchyKesai Tests

This directory contains the test suite for pyCauchyKesai.

## Test Structure

- `conftest.py` - Pytest configuration and fixtures, including C++ module mocking
- `test_python_wrapper.py` - Tests for Python wrapper layer (ModelSummary, BenchmarkResult, color helpers)
- `test_integration.py` - Integration tests for CauchyKesai class with mocked backend

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
pytest
```

### Run with coverage report

```bash
pytest --cov=pyCauchyKesai --cov-report=html --cov-report=term
```

### Run specific test file

```bash
pytest tests/test_python_wrapper.py
pytest tests/test_integration.py
```

### Run specific test class or function

```bash
pytest tests/test_python_wrapper.py::TestColorHelper
pytest tests/test_integration.py::TestCauchyKesaiConstruction::test_basic_construction
```

## Test Philosophy

Since pyCauchyKesai requires BPU hardware to run, the test suite uses mocking to validate:

1. **Python wrapper logic** - TTY detection, dict subclasses, formatting
2. **API contracts** - Method signatures, return types, exception handling
3. **Integration** - Python-C++ boundary behavior

The C++ extension module is mocked in `conftest.py`, allowing tests to run on any platform without hardware dependencies.

## Hardware Tests

For testing on actual BPU hardware, create a separate `tests_hardware/` directory with real model files and integration tests. These should be run manually on target devices.
