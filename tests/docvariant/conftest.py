"""tests/docvariant conftest — 针对 cn_CauchyKesai_Auto.md / cn_CauchyKesai_with_IONArray.md
文档细节的测试,数据由 tests/generate/gen_doc_variants.py 在云端产出的变异 hbm 驱动。

数据布局(DOC_DATA_DIR,由 sync 脚本从云端 ~//openexplore_test_cases/docvariants 拉回):
  DOC_DATA_DIR/manifest_doc.json          gen_doc_variants 的 manifest(含 compile_status)
  DOC_DATA_DIR/hbm/<name>/<name>.hbm      各变异编译规范的 hbm
  DOC_DATA_DIR/golden/<name>/{data.npz,meta.json}   golden 真值 + meta

测试用 pyCauchyKesai 已装 wheel(v1.4.0),板端 pyCauchyKesai conda env 跑。
BPU 被占用时整组 skip,不伪造结果。
"""
import os
import sys
import json
import pytest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DOC_DATA_DIR = os.environ.get(
    "DOC_DATA_DIR", "/root/ssd/OELLM_Runtime/docvariant_data")
MANIFEST = os.path.join(DOC_DATA_DIR, "manifest_doc.json")


def _load_entries():
    """读 manifest,只保留编译成功的 entry。"""
    if not os.path.exists(MANIFEST):
        return []
    try:
        d = json.load(open(MANIFEST))
    except Exception:
        return []
    return [e for e in d.get("manifest", [])
            if e.get("compile_status") == "ok" and e.get("hbm")]


ENTRIES = _load_entries()


def pytest_configure(config):
    config.addinivalue_line("markers", "docvar: docvariant 文档细节测试(需变异 hbm + BPU)")


def hbm_path(entry):
    return os.path.join(DOC_DATA_DIR, entry["hbm"])


def by_name(name):
    """按 name 取单个 entry。"""
    for e in ENTRIES:
        if e["name"] == name:
            return e
    return None


def select(core=None, image=None, name_in=None):
    """按 yaml 特征筛选 entry。
    core: 编译核数;image: True=rgb/nv12 图像输入,False=featuremap;name_in: name 集合。
    """
    out = []
    for e in ENTRIES:
        y = e.get("yaml", {})
        if core is not None and y.get("core", 1) != core:
            continue
        if image is not None and ("input" in y) != image:
            continue
        if name_in is not None and e["name"] not in name_in:
            continue
        out.append(e)
    return out


# ── fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def bpu_free():
    """BPU 空闲探测,整 session 一次。"""
    if not ENTRIES:
        pytest.skip(f"无 docvariant 数据({MANIFEST});先跑 gen_doc_variants + sync")
    from pyCauchyKesai import CauchyKesai
    try:
        CauchyKesai(hbm_path(ENTRIES[0]), n_task=1).summary()
    except Exception as e:
        pytest.skip(f"BPU 被占(可能 robotrea 服务): {e}")
    return True


@pytest.fixture
def factory(bpu_free):
    """entry → CauchyKesai 工厂(多核模型自动设调度核)。"""
    from pyCauchyKesai import CauchyKesai

    def make(entry, n_task=1):
        m = CauchyKesai(hbm_path(entry), n_task=n_task)
        if entry["yaml"].get("core", 1) > 1:
            m.set_scheduling_params(list(range(entry["yaml"]["core"])))
        return m

    return make


def make_inputs(model):
    """按 model.summary() 的实际 inputs dtype/shape 构造随机输入。
    nv12/rgb 经 OE 前处理后 dtype 可能是 uint8(y+uv),按 summary 实际值适配。
    """
    info = model.summary()
    ins = []
    for i in info["inputs"]:
        shape = tuple(i["shape"])
        dt = i["dtype"]
        if dt in ("uint8", "uint16"):
            ins.append(np.random.randint(0, 256, shape).astype(np.dtype(dt)))
        elif dt in ("int8",):
            ins.append(np.random.randint(-128, 128, shape).astype(np.int8))
        elif dt in ("int16", "uint16"):
            ins.append(np.random.randint(0, 100, shape).astype(np.dtype(dt)))
        elif dt == "float16":
            ins.append(np.random.randn(*shape).astype(np.float16))
        else:
            ins.append(np.random.randn(*shape).astype(np.float32))
    return ins


def all_entries_ids():
    return [e["name"] for e in ENTRIES] or ["no-data"]
