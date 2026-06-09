"""pytest 配置和夹具 — pyCauchyKesai"""

import os
import sys
import pytest
import numpy as np


# ============================================================================
# 平台检测
# ============================================================================
def detect_platform():
    """运行时检测当前平台: Nash-e, Nash-m, Nash-p"""
    try:
        from pyCauchyKesai import IONArray
        return "Nash"  # IONArray 存在 = Nash 系列
    except ImportError:
        return "unknown"


PLATFORM = detect_platform()


def pytest_configure(config):
    config.addinivalue_line("markers", "bpu: 需要 BPU 硬件")
    config.addinivalue_line("markers", "nash: Nash 系列 (Nash-e/Nash-m/Nash-p) 平台")
    config.addinivalue_line("markers", "benchmark: 性能基准")
    config.addinivalue_line("markers", "slow: 耗时测试")


def pytest_collection_modifyitems(config, items):
    for item in items:
        markers = item.keywords
        if "nash" in markers and PLATFORM not in ("Nash",):
            item.add_marker(pytest.mark.skip(reason="需要 Nash (Nash-e/Nash-m/Nash-p) 平台"))


# ============================================================================
# 查找测试模型
# ============================================================================
def find_test_model():
    path = os.environ.get("TEST_MODEL_PATH", "")
    if path and os.path.exists(path):
        return path
    search = [
        "/root/OELLM_Runtime/assets",
    ]
    for base in search:
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".hbm"):
                    return os.path.join(root, f)
    return None
