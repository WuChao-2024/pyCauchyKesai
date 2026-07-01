"""版本号单源回归测试。

pyCauchyKesai.__version__ 必须与包元数据（pyproject.toml 写进 wheel METADATA 的版本）一致，
防止 C++ 侧 / 文档里再次出现与 pyproject 不同步的版本字面量。
纯元数据检查，无需 BPU / 模型。
"""
from importlib.metadata import version

import pyCauchyKesai


def test_version_matches_metadata():
    assert pyCauchyKesai.__version__ == version("pycauchykesai")


def test_version_not_placeholder():
    # 已正确安装（非源码树裸跑），版本不应是兜底占位
    assert pyCauchyKesai.__version__ != "0.0.0+unknown"
