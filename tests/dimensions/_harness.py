"""多维度接口一致性测试套件的共享 harness。

复用 golden/harness.py(数据驱动: manifest/meta/npz), 只补:
  - load_model: 加载后自动按 core_num 调 set_scheduling_params(修复 core2 多核 bug)
  - 取数保留原 dtype(不写死 float32/float64, 为 D2/D6 留路)
  - BPU 空闲探测 / 按 family/precision/core_num/variant 选模型
  - per-dimension CSV 记录助手
"""
import os
import sys
import csv
import numpy as np

# 定位 golden harness（pyCauchyKesai 用已安装包，不在 path 注入 src/——避免 src/ 无编译 .so 时遮蔽已装包）
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))   # pyCauchyKesai/
_GOLDEN = os.path.join(_REPO, "tests", "golden")
if _GOLDEN not in sys.path:
    sys.path.insert(0, _GOLDEN)

from pyCauchyKesai import CauchyKesai            # noqa: E402
import harness as GH                              # noqa: E402  golden harness

GOLDEN_DATA_DIR = GH.GOLDEN_DATA_DIR
REPORT_DIR = os.path.join(_HERE, "report")
os.makedirs(REPORT_DIR, exist_ok=True)


# ── 模型加载 ──────────────────────────────────────────────────────────────
def load_model(entry, n_task=1):
    """加载 HBM, 多核模型自动设调度核(否则 hbUCPSubmitTask 拒 CORE_ANY)。"""
    m = CauchyKesai(GH.hbm_path(entry["cid"]), n_task=n_task)
    if entry.get("core_num", 1) > 1:
        m.set_scheduling_params(list(range(entry["core_num"])))
    return m


def probe_bpu_free():
    """加载第一个 HBM 跑 summary() 探活; 失败说明 BPU 被占。"""
    try:
        entries = GH.manifest_entries()
        if not entries:
            return False
        CauchyKesai(GH.hbm_path(entries[0]["cid"]), n_task=1).summary()
        return True
    except Exception:
        return False


# ── 模型选择 ──────────────────────────────────────────────────────────────
def select_entries(family=None, precision=None, core_num=None, variant=None,
                   multi_io=False, single_io=False):
    """按维度筛选 manifest entry。
    multi_io=True: 只要多输入或多输出模型(mixed_io)
    single_io=True: 只要单输入单输出模型
    """
    out = []
    for e in GH.manifest_entries():
        if family is not None and e["family"] != family:
            continue
        if precision is not None and e["precision"] != precision:
            continue
        if core_num is not None and e["core_num"] != core_num:
            continue
        if variant is not None and e["variant"] != variant:
            continue
        n_in = len(e["inputs"])
        n_out = len(e["outputs"])
        if multi_io and not (n_in > 1 or n_out > 1):
            continue
        if single_io and (n_in != 1 or n_out != 1):
            continue
        out.append(e)
    return out


# ── 取数(保留原 dtype) ────────────────────────────────────────────────────
def sample_inputs_raw(entry, sample_idx=0):
    """与 GH.sample_inputs 等价但不强制 float32(保留 meta 声明的 dtype)。"""
    meta = GH.load_meta(entry["model_id"])
    npz = GH.load_golden_npz(entry["model_id"])
    out = []
    for i in meta["inputs"]:
        key = f"in_{sample_idx:02d}_{i['name']}"
        out.append(np.ascontiguousarray(npz[key]))
    return out


def sample_outputs_raw(entry, sample_idx=0):
    meta = GH.load_meta(entry["model_id"])
    npz = GH.load_golden_npz(entry["model_id"])
    out = []
    for i in meta["outputs"]:
        key = f"out_{sample_idx:02d}_{i['name']}"
        out.append(np.ascontiguousarray(npz[key]))
    return out


# ── per-dimension 报告助手 ────────────────────────────────────────────────
def record_csv(name, rows, fieldnames):
    """写 report/<name>.csv。"""
    path = os.path.join(REPORT_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ── session 级报告累加器(conftest 提供 acc dict) ─────────────────────────
def collect(acc, csv_name, row):
    """累加一行到 acc[csv_name](session 级 dict), session 结束统一落盘。"""
    slot = acc.setdefault(csv_name, {"fields": list(row.keys()), "rows": []})
    slot["rows"].append(row)


def flush_reports(acc):
    """session 结束时把 acc 里各 CSV 写盘。"""
    written = []
    for name, slot in acc.items():
        written.append(record_csv(f"{name}.csv", slot["rows"], slot["fields"]))
    return written


def is_padded(io_meta):
    """通过 alignedByteSize > natural nbytes 推断是否 padded(S600 对齐填充)。
    is_padded_layout/stride 未作为 Python 属性暴露, 用 summary 的 alignedByteSize 近似。
    """
    nat = int(np.prod(io_meta["shape"])) * np.dtype(io_meta["dtype"]).itemsize
    return io_meta.get("alignedByteSize", nat) > nat


# ── 路径一致性检查(D1/D2/D3 共用, 复用 _paths) ────────────────────────────
def path_consistency(entry):
    """对 entry 跑 sync/zerocopy/async 三路, 返回一致性结果 dict。
    覆盖核心接口契约: 三路输出应一致(bit-identical), sync 输出有限且 shape 匹配 meta。
    """
    import _paths as P   # 延迟 import 避免循环
    inputs = sample_inputs_raw(entry, 0)
    meta_out_shapes = [tuple(o["shape"]) for o in GH.load_meta(entry["model_id"])["outputs"]]
    ref = P.run_sync(load_model(entry), inputs)
    sync_finite = all(np.all(np.isfinite(o)) for o in ref)
    sync_shape_ok = [tuple(o.shape) for o in ref] == meta_out_shapes
    zc = P.run_zerocopy(load_model(entry), inputs)
    asc = P.run_async(load_model(entry), inputs)
    zc_max = max((P.assert_identical(r, c, strict=True, label=f"zerocopy{i}")
                  for i, (r, c) in enumerate(zip(ref, zc))), default=0.0)
    asc_max = max((P.assert_identical(r, c, strict=True, label=f"async{i}")
                   for i, (r, c) in enumerate(zip(ref, asc))), default=0.0)
    return {
        "sync_finite": bool(sync_finite),
        "sync_shape_ok": bool(sync_shape_ok),
        "n_in": len(inputs), "n_out": len(ref),
        "in_shapes": str([list(i.shape) for i in inputs]),
        "out_shapes": str([list(o.shape) for o in ref]),
        "zerocopy_maxdiff": float(zc_max),
        "async_maxdiff": float(asc_max),
        "all_identical": bool(sync_finite and sync_shape_ok and zc_max == 0.0 and asc_max == 0.0),
    }


def parse_model_id(model_id):
    """解析 build_all 的 model_id: {backbone}_{rank}d_{variant}_{arity}。
    返回 dict(backbone, rank, variant, arity)。rank='stack'(老族)/'1d'..'5d'。"""
    import re
    toks = model_id.split("_")
    arity = ""
    if re.fullmatch(r"\di\do", toks[-1]):
        arity = toks[-1]; toks = toks[:-1]
    rank = ""
    for i, t in enumerate(toks):
        if re.fullmatch(r"\dd", t):
            rank = t; backbone = "_".join(toks[:i]); variant = "_".join(toks[i + 1:])
            return {"backbone": backbone, "rank": rank, "variant": variant, "arity": arity}
    # 老族(conv_stack_large 等): backbone=前段, rank=stack
    return {"backbone": "_".join(toks[:-1]) if len(toks) > 1 else model_id,
            "rank": "stack", "variant": toks[-1], "arity": arity or "1i1o"}


def select_matrix(core=1, backbone=None, rank=None, variant=None, arity=None):
    """从 manifest 按 parse_model_id 的字段筛选(core 固定, 默认 1)。"""
    def _match(val, want):
        if want is None:
            return True
        vals = [want] if isinstance(want, str) else list(want)
        return val in vals
    out = []
    for e in GH.manifest_entries():
        if e.get("core_num") != core:
            continue
        p = parse_model_id(e["model_id"])
        if _match(p["backbone"], backbone) and _match(p["rank"], rank) \
                and _match(p["variant"], variant) and _match(p["arity"], arity):
            out.append(e)
    return out


def one_per(entries, key):
    """按 parse_model_id 的 key(backbone/rank/variant/arity) 去重, 每组取第一个。"""
    seen, out = set(), []
    for e in entries:
        k = parse_model_id(e["model_id"]).get(key)
        if k in seen:
            continue
        seen.add(k); out.append(e)
    return out
