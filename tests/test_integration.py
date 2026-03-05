"""
Integration tests for CauchyKesai class (Python wrapper + mocked C++ backend).

These tests validate the integration between Python wrapper and C++ extension,
using mocked C++ backend from conftest.py.
"""
import pytest
import numpy as np


class TestCauchyKesaiConstruction:
    """Test CauchyKesai object construction and initialization."""

    def test_basic_construction(self, mock_model_path):
        """Test basic model construction with default parameters."""
        from pyCauchyKesai import CauchyKesai

        # This will use mocked C++ backend
        model = CauchyKesai(mock_model_path)
        assert model is not None

    def test_construction_with_n_task(self, mock_model_path):
        """Test construction with custom n_task parameter."""
        from pyCauchyKesai import CauchyKesai

        model = CauchyKesai(mock_model_path, n_task=8)
        assert model is not None

    def test_construction_with_all_params(self, mock_model_path):
        """Test construction with all parameters."""
        from pyCauchyKesai import CauchyKesai

        model = CauchyKesai(mock_model_path, n_task=4, model_cnt_select=0)
        assert model is not None


class TestCauchyKesaiSummary:
    """Test model summary functionality."""

    @pytest.fixture
    def model(self, mock_model_path):
        from pyCauchyKesai import CauchyKesai
        return CauchyKesai(mock_model_path, n_task=4)

    def test_s_returns_model_summary(self, model):
        """Test that s() returns ModelSummary instance."""
        from pyCauchyKesai import ModelSummary

        summary = model.s()
        assert isinstance(summary, ModelSummary)
        assert isinstance(summary, dict)

    def test_s_contains_required_keys(self, model):
        """Test that summary contains all required keys."""
        summary = model.s()

        required_keys = ['model_path', 'model_names', 'n_task', 'memory_mb', 'inputs', 'outputs']
        for key in required_keys:
            assert key in summary

    def test_s_model_names_structure(self, model):
        """Test model_names list structure."""
        summary = model.s()

        assert isinstance(summary['model_names'], list)
        assert len(summary['model_names']) > 0

        for entry in summary['model_names']:
            assert 'index' in entry
            assert 'name' in entry
            assert 'selected' in entry
            assert isinstance(entry['selected'], bool)

    def test_s_inputs_structure(self, model):
        """Test inputs list structure."""
        summary = model.s()

        assert isinstance(summary['inputs'], list)
        for inp in summary['inputs']:
            assert 'index' in inp
            assert 'name' in inp
            assert 'dtype' in inp
            assert 'shape' in inp
            assert isinstance(inp['shape'], list)

    def test_s_outputs_structure(self, model):
        """Test outputs list structure."""
        summary = model.s()

        assert isinstance(summary['outputs'], list)
        for out in summary['outputs']:
            assert 'index' in out
            assert 'name' in out
            assert 'dtype' in out
            assert 'shape' in out
            assert isinstance(out['shape'], list)


class TestCauchyKesaiBenchmark:
    """Test benchmark functionality."""

    @pytest.fixture
    def model(self, mock_model_path):
        from pyCauchyKesai import CauchyKesai
        return CauchyKesai(mock_model_path)

    def test_t_returns_benchmark_result(self, model):
        """Test that t() returns BenchmarkResult instance."""
        from pyCauchyKesai import BenchmarkResult

        result = model.t()
        assert isinstance(result, BenchmarkResult)
        assert isinstance(result, dict)

    def test_t_contains_timing_keys(self, model):
        """Test that benchmark result contains all timing keys."""
        result = model.t()

        required_keys = ['time_us', 'time_ms', 'time_s', 'time_min']
        for key in required_keys:
            assert key in result
            assert isinstance(result[key], (int, float))
            assert result[key] >= 0

    def test_t_timing_consistency(self, model):
        """Test that timing values are consistent across units."""
        result = model.t()

        # Check rough consistency (allowing for floating point errors)
        assert result['time_ms'] == pytest.approx(result['time_us'] / 1000, rel=1e-5)
        assert result['time_s'] == pytest.approx(result['time_ms'] / 1000, rel=1e-5)
        assert result['time_min'] == pytest.approx(result['time_s'] / 60, rel=1e-5)


class TestCauchyKesaiRepr:
    """Test __repr__ functionality."""

    @pytest.fixture
    def model(self, mock_model_path):
        from pyCauchyKesai import CauchyKesai
        return CauchyKesai(mock_model_path, n_task=4)

    def test_repr_format(self, model):
        """Test that __repr__ returns expected format."""
        repr_str = repr(model)

        assert repr_str.startswith("CauchyKesai(")
        assert repr_str.endswith(")")
        assert "model=" in repr_str
        assert "inputs=" in repr_str
        assert "outputs=" in repr_str
        assert "n_task=" in repr_str

    def test_repr_contains_model_name(self, model):
        """Test that repr contains model name."""
        repr_str = repr(model)
        assert "mock_model" in repr_str

    def test_repr_shows_task_count(self, model):
        """Test that repr shows correct task count."""
        repr_str = repr(model)
        assert "n_task=4" in repr_str


class TestCauchyKesaiIsBusy:
    """Test is_busy() task status query."""

    @pytest.fixture
    def model(self, mock_model_path):
        from pyCauchyKesai import CauchyKesai
        return CauchyKesai(mock_model_path, n_task=4)

    def test_is_busy_default_task(self, model):
        """Test is_busy with default task_id."""
        result = model.is_busy()
        assert isinstance(result, bool)

    def test_is_busy_specific_task(self, model):
        """Test is_busy with specific task_id."""
        for task_id in range(4):
            result = model.is_busy(task_id)
            assert isinstance(result, bool)

    def test_is_busy_invalid_task_negative(self, model):
        """Test is_busy raises IndexError for negative task_id."""
        with pytest.raises(IndexError, match="task_id out of range"):
            model.is_busy(-1)

    def test_is_busy_invalid_task_too_large(self, model):
        """Test is_busy raises IndexError for task_id >= n_task."""
        with pytest.raises(IndexError, match="task_id out of range"):
            model.is_busy(4)  # n_task=4, so valid range is [0,3]

    def test_is_busy_boundary_values(self, model):
        """Test is_busy at boundary values."""
        # Should work
        model.is_busy(0)
        model.is_busy(3)

        # Should fail
        with pytest.raises(IndexError):
            model.is_busy(4)


class TestModuleMetadata:
    """Test module-level metadata."""

    def test_version_exists(self):
        import pyCauchyKesai
        assert hasattr(pyCauchyKesai, '__version__')
        assert isinstance(pyCauchyKesai.__version__, str)

    def test_author_exists(self):
        import pyCauchyKesai
        assert hasattr(pyCauchyKesai, '__author__')
        assert isinstance(pyCauchyKesai.__author__, str)

    def test_date_exists(self):
        import pyCauchyKesai
        assert hasattr(pyCauchyKesai, '__date__')
        assert isinstance(pyCauchyKesai.__date__, str)
