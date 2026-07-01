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
    # 多维度接口一致性测试套件的正交 marker（可叠加，如 @dim4 @dim5）
    for spec in (
        "dim1: D1 算子/结构一致性（刻画）",
        "dim2: D2 张量形态（rank/shape）",
        "dim3: D3 IO 元数（单/多 输入输出）",
        "dim4: D4 推理路径 7 路 cross-path 一致性",
        "dim5: D5 内存/对齐（零拷贝 + padded/strided）",
        "dim6: D6 量化/反量化（SCALE 路径）",
        "dim7: D7 核数等价（core1 vs core2）",
        "dim8: D8 并发语义（n_task/交错/超时）",
        "dim9: D9 错误路径鲁棒性",
        "dim10: D10 确定性（同输入逐元素相等）",
        "dim11: D11 API 元数据契约",
    ):
        config.addinivalue_line("markers", spec)


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
