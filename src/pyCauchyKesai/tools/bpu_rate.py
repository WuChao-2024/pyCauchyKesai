#!/usr/bin/env python3
"""BPU 占用率查询工具。

直接读取 /sys/devices/system/bpu/bpu*/ratio (hrut_somstatus 的底层数据源)，
自适应任意核心数硬件，返回结构化 dict。

Usage:
    from pyCauchyKesai.tools.bpu_rate import get_bpu_status

    status = get_bpu_status()
    # {
    #     "num_cores": 4,
    #     "core_ids": [0, 1, 2, 3],
    #     "ratios": {0: 0, 1: 5, 2: 0, 3: 0},
    #     "idle_cores": [0, 2, 3],
    #     "busy_cores": [1],
    #     "all_idle": False,
    # }
"""

import glob
import time
from typing import Dict

BPU_SYSFS = "/sys/devices/system/bpu"


def _read_ratios() -> Dict[int, int]:
    """读取所有 BPU 核心的当前 ratio 值。"""
    ratios: Dict[int, int] = {}
    for path in sorted(glob.glob(f"{BPU_SYSFS}/bpu[0-9]*/ratio")):
        # 从路径中提取核心编号: .../bpu0/ratio → 0
        core_id = int(path.rsplit("/", 2)[-2].replace("bpu", ""))
        with open(path) as f:
            ratios[core_id] = int(f.read().strip())
    return ratios


def get_bpu_status(n_samples: int = 1, interval_sec: float = 0.0) -> dict:
    """读取 BPU 状态，返回结构化 dict。

    Parameters
    ----------
    n_samples : int
        采样次数（默认 1）。>1 时返回最后一次采样的瞬时值。
    interval_sec : float
        采样间隔（秒），仅 n_samples > 1 时生效。

    Returns
    -------
    dict
        num_cores   : int           — BPU 核心总数
        core_ids    : list[int]     — 核心索引 [0, 1, ...]
        ratios      : dict[int,int] — 每核占用率 (0-100), 0=空闲
        idle_cores  : list[int]     — 空闲核心
        busy_cores  : list[int]     — 占用核心
        all_idle    : bool          — 是否全部空闲
    """
    ratios: Dict[int, int] = {}
    for i in range(n_samples):
        ratios = _read_ratios()
        if i < n_samples - 1:
            time.sleep(interval_sec)

    if not ratios:
        raise RuntimeError(
            f"未找到 BPU 设备 (检查 {BPU_SYSFS}/bpu*/ratio 是否存在)"
        )

    core_ids = sorted(ratios.keys())
    idle_cores = [c for c in core_ids if ratios[c] == 0]
    busy_cores = [c for c in core_ids if ratios[c] > 0]

    return {
        "num_cores": len(core_ids),
        "core_ids": core_ids,
        "ratios": ratios,
        "idle_cores": idle_cores,
        "busy_cores": busy_cores,
        "all_idle": len(busy_cores) == 0,
    }


def get_core_count() -> int:
    """返回当前机器 BPU 核心总数。"""
    return get_bpu_status()["num_cores"]


def print_summary():
    """打印人类可读的 BPU 状态摘要。"""
    s = get_bpu_status()
    print(f"BPU cores: {s['num_cores']}  (ids: {s['core_ids']})")
    print(f"  idle: {s['idle_cores']}  busy: {s['busy_cores'] or '(none)'}")
    print(f"  ratios: {s['ratios']}")


# ---- CLI ----
if __name__ == "__main__":
    import sys
    if "--json" in sys.argv:
        import json
        print(json.dumps(get_bpu_status(), indent=2, ensure_ascii=False))
    else:
        print_summary()
