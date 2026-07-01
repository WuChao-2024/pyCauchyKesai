"""Pytest config for golden characterization tests (M4).

Self-contained: registers the `golden` marker and skips the whole module
when GOLDEN_DATA_DIR has no synced data. Numerical characterization is done
by run_characterization.py; these tests are a light smoke gate (model loads +
produces finite output), with NO numerical thresholds (pure characterization).
"""
import os, sys
import pytest

TD = os.path.dirname(os.path.abspath(__file__))
if TD not in sys.path:
    sys.path.insert(0, TD)
import harness as H


def pytest_configure(config):
    config.addinivalue_line("markers", "golden: golden HBM characterization (needs synced data + BPU)")


@pytest.fixture(scope="session")
def golden_data_ok():
    if not H.data_present():
        pytest.skip(f"no golden data under GOLDEN_DATA_DIR={H.GOLDEN_DATA_DIR}")
    return True
