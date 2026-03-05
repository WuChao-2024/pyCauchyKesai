"""
Tests for pyCauchyKesai Python wrapper layer.

These tests validate the Python-side logic (ModelSummary, BenchmarkResult, _color, etc.)
without requiring BPU hardware. The C++ extension is mocked in conftest.py.
"""
import sys
import io
import pytest


class TestColorHelper:
    """Test the _color() TTY detection helper."""

    def test_color_with_tty(self, monkeypatch):
        """When stderr is a TTY, ANSI codes should be present."""
        from pyCauchyKesai import _color

        mock_stderr = io.StringIO()
        mock_stderr.isatty = lambda: True
        monkeypatch.setattr(sys, 'stderr', mock_stderr)

        result = _color("hello")
        assert "\033[1;31m" in result
        assert "\033[0m" in result
        assert "hello" in result

    def test_color_without_tty(self, monkeypatch):
        """When stderr is not a TTY, output should be plain text."""
        from pyCauchyKesai import _color

        mock_stderr = io.StringIO()
        mock_stderr.isatty = lambda: False
        monkeypatch.setattr(sys, 'stderr', mock_stderr)

        result = _color("hello")
        assert "\033[" not in result
        assert result == "hello"

    def test_color_no_isatty(self, monkeypatch):
        """When stderr has no isatty method, output should be plain text."""
        from pyCauchyKesai import _color

        mock_stderr = object()  # no isatty attribute
        monkeypatch.setattr(sys, 'stderr', mock_stderr)

        result = _color("hello")
        assert "\033[" not in result
        assert result == "hello"


class TestModelSummary:
    """Test ModelSummary dict subclass."""

    @pytest.fixture
    def summary_data(self):
        return {
            'model_path': '/opt/model/yolov8.hbm',
            'model_names': [
                {'index': 0, 'name': 'yolov8n_640x640_nv12', 'selected': True},
                {'index': 1, 'name': 'yolov8s_640x640_nv12', 'selected': False},
            ],
            'n_task': 4,
            'memory_mb': 25.3,
            'inputs': [
                {'index': 0, 'name': 'images_y', 'dtype': 'uint8', 'shape': [1, 640, 640, 1]},
                {'index': 1, 'name': 'images_uv', 'dtype': 'uint8', 'shape': [1, 320, 320, 2]},
            ],
            'outputs': [
                {'index': 0, 'name': 'output0', 'dtype': 'float32', 'shape': [1, 80, 80, 80]},
            ]
        }

    def test_dict_access(self, summary_data):
        from pyCauchyKesai import ModelSummary
        s = ModelSummary(summary_data)

        assert s['model_path'] == '/opt/model/yolov8.hbm'
        assert s['n_task'] == 4
        assert s['memory_mb'] == 25.3
        assert len(s['inputs']) == 2
        assert len(s['outputs']) == 1
        assert s['inputs'][0]['name'] == 'images_y'
        assert s['inputs'][0]['dtype'] == 'uint8'
        assert s['inputs'][0]['shape'] == [1, 640, 640, 1]

    def test_repr_contains_key_info(self, summary_data, monkeypatch):
        from pyCauchyKesai import ModelSummary

        # Force non-TTY to get clean output
        mock_stderr = io.StringIO()
        mock_stderr.isatty = lambda: False
        monkeypatch.setattr(sys, 'stderr', mock_stderr)

        s = ModelSummary(summary_data)
        text = repr(s)

        assert "Model File:" in text
        assert "/opt/model/yolov8.hbm" in text
        assert "Model Names:" in text
        assert "yolov8n_640x640_nv12" in text
        assert "[*Select]" in text
        assert "Task N:" in text
        assert "4" in text
        assert "images_y" in text
        assert "uint8" in text
        assert "640" in text
        assert "output0" in text
        assert "float32" in text

    def test_str_equals_repr(self, summary_data):
        from pyCauchyKesai import ModelSummary
        s = ModelSummary(summary_data)
        assert str(s) == repr(s)

    def test_is_dict_subclass(self, summary_data):
        from pyCauchyKesai import ModelSummary
        s = ModelSummary(summary_data)
        assert isinstance(s, dict)


class TestBenchmarkResult:
    """Test BenchmarkResult dict subclass."""

    @pytest.fixture
    def bench_data(self):
        return {
            'time_us': 5432.1,
            'time_ms': 5.4321,
            'time_s': 0.0054321,
            'time_min': 9.0535e-05,
        }

    def test_dict_access(self, bench_data):
        from pyCauchyKesai import BenchmarkResult
        b = BenchmarkResult(bench_data)

        assert b['time_us'] == 5432.1
        assert b['time_ms'] == 5.4321
        assert b['time_s'] == 0.0054321
        assert b['time_min'] == pytest.approx(9.0535e-05)

    def test_repr_contains_timing(self, bench_data, monkeypatch):
        from pyCauchyKesai import BenchmarkResult

        mock_stderr = io.StringIO()
        mock_stderr.isatty = lambda: False
        monkeypatch.setattr(sys, 'stderr', mock_stderr)

        b = BenchmarkResult(bench_data)
        text = repr(b)

        assert "Inference Info:" in text
        assert "microseconds" in text
        assert "milliseconds" in text
        assert "seconds" in text
        assert "minutes" in text
        assert "\u03bcs" in text  # μs

    def test_str_equals_repr(self, bench_data):
        from pyCauchyKesai import BenchmarkResult
        b = BenchmarkResult(bench_data)
        assert str(b) == repr(b)

    def test_is_dict_subclass(self, bench_data):
        from pyCauchyKesai import BenchmarkResult
        b = BenchmarkResult(bench_data)
        assert isinstance(b, dict)
