"""tests/dimensions 专用 conftest: 把本目录加到 sys.path(供 import _harness/_paths),
并提供 session 级 BPU 空闲 fixture。"""
import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _harness as H   # noqa: E402  (触发 src/golden path 注入)


@pytest.fixture(scope="session")
def bpu_free():
    """BPU 空闲探测, 整 session 一次。被占用则 skip 所有需硬件用例。"""
    if not H.probe_bpu_free():
        pytest.skip("BPU 被占用(可能 robotrea 服务), 跳过需硬件用例")
    return True


@pytest.fixture(scope="session")
def acc():
    """session 级报告累加器: acc[csv_name] = {fields, rows}。"""
    return {}


@pytest.fixture(scope="session", autouse=True)
def _report_writer(acc):
    """session 结束时把累加的各维度 CSV 落盘。"""
    yield
    try:
        H.flush_reports(acc)
    except Exception as e:  # noqa: BLE001
        print(f"[dimensions] flush_reports 失败: {e}")


def pytest_collection_modifyitems(config, items):
    """给带 dimN marker 的用例自动补 bpu marker(都需要 BPU), 便于 -m bpu 统一过滤。"""
    for item in items:
        kms = item.keywords
        if any(f"dim{n}" in kms for n in range(1, 12)):
            if "bpu" not in kms:
                item.add_marker(pytest.mark.bpu)
