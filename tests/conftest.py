"""
Pytest configuration and fixtures for pyCauchyKesai tests.
"""
import pytest
import sys
from unittest.mock import MagicMock, Mock
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def mock_cpp_module():
    """
    Mock the C++ extension module for testing without hardware.

    This allows testing Python wrapper logic without requiring actual BPU hardware.
    """
    # Create mock C++ module
    mock_module = MagicMock()

    # Mock CauchyKesai class from C++ extension
    mock_cpp_class = Mock()

    # Mock instance methods
    def mock_init(self, model_path, n_task=1, model_cnt_select=0):
        self._model_path = model_path
        self._n_task = n_task
        self._model_cnt_select = model_cnt_select
        self._busy_tasks = [False] * n_task
        return None

    def mock_s(self):
        return {
            'model_path': '/mock/model.hbm',
            'model_names': [{'index': 0, 'name': 'mock_model', 'selected': True}],
            'n_task': 4,
            'memory_mb': 10.5,
            'inputs': [
                {'index': 0, 'name': 'input0', 'dtype': 'uint8', 'shape': [1, 224, 224, 3]}
            ],
            'outputs': [
                {'index': 0, 'name': 'output0', 'dtype': 'float32', 'shape': [1, 1000]}
            ]
        }

    def mock_t(self):
        return {
            'time_us': 1234.56,
            'time_ms': 1.23456,
            'time_s': 0.00123456,
            'time_min': 0.0000205760
        }

    def mock_is_busy(self, task_id=0):
        if task_id < 0 or task_id >= self._n_task:
            raise IndexError(f"task_id out of range: got {task_id}, valid range [0, {self._n_task-1}]")
        return self._busy_tasks[task_id]

    mock_cpp_class.__init__ = mock_init
    mock_cpp_class.s = mock_s
    mock_cpp_class.t = mock_t
    mock_cpp_class.is_busy = mock_is_busy

    mock_module.CauchyKesai = mock_cpp_class
    mock_module.__version__ = "0.0.9_AI_Native"
    mock_module.__author__ = "Cauchy - WuChao in D-Robotics"
    mock_module.__date__ = "2025-05-30"

    # Inject mock into sys.modules before importing
    sys.modules['pyCauchyKesai.pyCauchyKesai'] = mock_module

    yield mock_module

    # Cleanup
    if 'pyCauchyKesai.pyCauchyKesai' in sys.modules:
        del sys.modules['pyCauchyKesai.pyCauchyKesai']


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a temporary mock model file."""
    model_file = tmp_path / "test_model.hbm"
    model_file.write_bytes(b"mock model data")
    return str(model_file)


@pytest.fixture
def sample_input_data():
    """Generate sample input data for testing."""
    return [
        np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    ]


@pytest.fixture
def sample_output_data():
    """Generate sample output data for testing."""
    return [
        np.random.randn(1, 1000).astype(np.float32)
    ]
