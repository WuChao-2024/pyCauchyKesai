"""
IONMemory 基本功能测试: 分配/释放、numpy 往返、flush、capsule 保活、non-owning 包装、偏移算术
"""

import numpy as np
from pyCauchyKesai import IONMemory
import time, sys


def now():
    return time.perf_counter() * 1000


def bar(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ═══════════════════════════════════════════════════════════
# 测试 1: 分配/释放 + 基本属性
# ═══════════════════════════════════════════════════════════

def test_01_alloc():
    bar("1. 分配/释放: 基本属性校验")
    m = IONMemory(1024)
    assert m.size == 1024, f"size 应 == 1024, 实际 {m.size}"
    assert m.is_allocated is True, "应已分配"
    print(f"  size = {m.size}")
    print(f"  is_allocated = {m.is_allocated}")
    print("  OK")
    del m


# ═══════════════════════════════════════════════════════════
# 测试 2: numpy uint8 往返
# ═══════════════════════════════════════════════════════════

def test_02_numpy_roundtrip():
    bar("2. numpy uint8 写入/读回")
    m = IONMemory(1024)
    arr = m.numpy()
    assert arr.dtype == np.uint8, f"dtype 应为 uint8, 实际 {arr.dtype}"
    assert arr.shape == (1024,), f"shape 应为 (1024,), 实际 {arr.shape}"

    data = np.arange(1024, dtype=np.uint8)
    arr[:] = data

    # 重新获取 numpy 比对
    arr2 = m.numpy()
    assert np.array_equal(arr2, data), "写入数据与读回不一致"
    print(f"  dtype={arr.dtype}, shape={arr.shape}")
    print(f"  写入 np.arange(1024, dtype=uint8), 读回 OK")
    print("  OK")
    del m


# ═══════════════════════════════════════════════════════════
# 测试 3: flush 往返
# ═══════════════════════════════════════════════════════════

def test_03_flush():
    bar("3. flush_clean + flush_invalidate 往返")
    m = IONMemory(1024, cached=True)
    data = np.arange(256, dtype=np.uint8).repeat(4)  # 1024 bytes
    m.numpy()[:] = data

    m.flush_clean()
    m.flush_invalidate()

    result = m.numpy().copy()
    assert np.array_equal(result, data), "flush 往返后数据不一致"
    print(f"  cached: 写入 -> flush_clean -> flush_invalidate -> 读回 OK")
    del m

    # uncached: flush 是 no-op，但也不应 crash
    m2 = IONMemory(1024, cached=False)
    m2.numpy()[:] = data
    m2.flush_clean()
    m2.flush_invalidate()
    result2 = m2.numpy().copy()
    assert np.array_equal(result2, data), "uncached flush 往返后数据不一致"
    print(f"  uncached: 写入 -> flush_clean -> flush_invalidate -> 读回 OK")
    print("  OK")
    del m2


# ═══════════════════════════════════════════════════════════
# 测试 4: capsule 保活 — 改 numpy 视图反映到 IONMemory
# ═══════════════════════════════════════════════════════════

def test_04_capsule_keepalive():
    bar("4. capsule 保活: 修改 numpy 视图后重新获取验证一致")
    m = IONMemory(1024)
    arr = m.numpy()

    # 写入可识别 pattern
    arr[:] = np.arange(1024, dtype=np.uint8)

    # 修改 numpy 视图的某些字节
    arr[0] = 0xFF
    arr[512] = 0xFE

    # 删除这个 numpy 视图，重新获取
    del arr

    arr2 = m.numpy()
    assert arr2[0] == 0xFF, f"capsule 保活失败: arr2[0] == {arr2[0]}, 期望 0xFF"
    assert arr2[512] == 0xFE, f"capsule 保活失败: arr2[512] == {arr2[512]}, 期望 0xFE"
    assert arr2[1] == 1, f"原始数据被破坏: arr2[1] == {arr2[1]}, 期望 1"
    print(f"  修改 arr[0]=0xFF, arr[512]=0xFE, 删 arr, 重新 numpy 验证一致 OK")
    print("  OK")
    del m


# ═══════════════════════════════════════════════════════════
# 注: 原 test_05 (non_owning_view) 已随 v1.2 移除该公开 API 删除。
#     IONMemory 的共享语义由 IONArray.from_memory（test_ion_array.py test_12）覆盖。
# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    TESTS = [
        ("1.分配/释放",       test_01_alloc),
        ("2.numpy往返",     test_02_numpy_roundtrip),
        ("3.flush往返",        test_03_flush),
        ("4.capsule保活",      test_04_capsule_keepalive),
    ]

    results = []
    for label, fn in TESTS:
        try:
            fn()
            results.append((label, "OK"))
        except Exception as e:
            results.append((label, f"FAIL -- {e}"))

    print(f"\n{'='*70}")
    print("  结果汇总")
    print(f"{'='*70}")
    for label, status in results:
        print(f"  {label:20s} {status}")
    n_ok = sum(1 for _, s in results if s == "OK")
    print(f"  {'─'*40}")
    print(f"  {n_ok}/{len(results)} 通过")
    sys.exit(0 if n_ok == len(results) else 1)
