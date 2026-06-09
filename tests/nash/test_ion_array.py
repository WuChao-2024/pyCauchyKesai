"""
IONArray 全排列测试: 每个 shape × 每个操作, numpy vs ION(cached) vs ION(uncached) 全对比
"""

import numpy as np
from pyCauchyKesai import IONArray
import time, sys, unicodedata

# ═══════════════════════════════════════════════════════════
# 测试矩阵定义
# ═══════════════════════════════════════════════════════════

DT = np.dtype('float32')

SHAPES = [
    ("1D-1K",         [1000]),
    ("2D-32x64",      [32, 64]),
    ("3D-1x64x128",   [1, 64, 128]),
    ("CNN-224",       [1, 3, 224, 224]),
    ("DET-8400",      [8400, 84]),
    ("LLM-cache",      [2,8,4096,2048]),
]

# ═══════════════════════════════════════════════════════════
# 工具函数 (极简)
# ═══════════════════════════════════════════════════════════

def now():
    return time.perf_counter() * 1000

def hsz(b):
    if b < 1e6: return f"{b/1024:.0f}KB"
    return f"{b/1e6:.1f}MB"

# CJK 对齐用的显示宽度
def _w(s):
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(s))

def _pad(s, w):
    return str(s) + ' ' * max(0, w - _w(s))

def table(headers, rows):
    """打印对齐表格，无竖线，仅表头下划线"""
    all_rows = [list(headers)] + [[str(c) for c in r] for r in rows]
    widths = [max(_w(r[i]) for r in all_rows) for i in range(len(headers))]
    def line(r):
        return "  ".join(_pad(c, widths[i]) for i, c in enumerate(r))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in all_rows[1:]:
        print(line(r))

def bar(title):
    print(f"\n{'='*80}\n  {title}\n{'='*80}")

# ═══════════════════════════════════════════════════════════
# 测试 1: 构造 — 创建耗时 + 基本属性
# ═══════════════════════════════════════════════════════════

def test_01_create():
    bar("1. 创建: numpy vs ION(cached) vs ION(uncached)")
    rows = []
    for name, shape in SHAPES:
        nbytes = int(np.prod(shape)) * DT.itemsize

        t0 = now()
        np_arr = np.empty(shape, dtype=DT)
        tn = now() - t0

        t0 = now()
        ic = IONArray(DT, shape, cached=True)
        tc = now() - t0

        t0 = now()
        iu = IONArray(DT, shape, cached=False)
        tu = now() - t0

        # 属性检查
        assert ic.is_cached and not iu.is_cached
        assert ic.phy_addr > 0 and iu.phy_addr > 0
        assert ic.phy_addr != iu.phy_addr
        assert ic.as_array().shape == tuple(shape)
        assert ic.as_array().dtype == DT

        rows.append([name, str(shape), hsz(nbytes),
                     f"{tn:.3f}", f"{tc:.3f}", f"{tu:.3f}",
                     "OK"])

        del np_arr, ic, iu

    table(["名称","shape","大小","numpy(ms)","IONc(ms)","IONu(ms)","状态"], rows)

# ═══════════════════════════════════════════════════════════
# 测试 2: 写入随机数据 + 读回验证
# ═══════════════════════════════════════════════════════════

def test_02_write_read():
    bar("2. 写入随机数据+读回: numpy vs ION(cached) vs ION(uncached)")
    rows = []
    for name, shape in SHAPES:
        data = np.random.randn(*shape).astype(DT)

        # ── numpy (参考基准) ──
        np_arr = np.empty(shape, dtype=DT)
        t0 = now(); np_arr[:] = data; tw_np = now() - t0
        t0 = now(); np_out = np_arr.copy(); tr_np = now() - t0

        # ── ION cached ──
        ic = IONArray(DT, shape, cached=True)
        t0 = now(); ic.as_array()[:] = data; tw_c = now() - t0
        t0 = now(); c_out = ic.as_array().copy(); tr_c = now() - t0
        ok_c = "OK" if np.allclose(data, c_out, atol=1e-5) else "FAIL"
        del ic, c_out

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        t0 = now(); iu.as_array()[:] = data; tw_u = now() - t0
        t0 = now(); u_out = iu.as_array().copy(); tr_u = now() - t0
        ok_u = "OK" if np.allclose(data, u_out, atol=1e-5) else "FAIL"
        del iu, u_out

        assert ok_c == "OK" and ok_u == "OK"

        rows.append([name, hsz(data.nbytes),
                     f"{tw_np:.3f}", f"{tw_c:.3f}", f"{tw_u:.3f}",
                     f"{tr_np:.3f}", f"{tr_c:.3f}", f"{tr_u:.3f}",
                     ok_c, ok_u])

        del data, np_arr, np_out

    table(["名称","大小",
           "写np","写IONc","写IONu",
           "读np","读IONc","读IONu",
           "c精度","u精度"], rows)

# ═══════════════════════════════════════════════════════════
# 测试 3: flush + invalidate 往返
# ═══════════════════════════════════════════════════════════

def test_03_flush():
    bar("3. flush+invalidate 往返: cached vs uncached")
    rows = []
    for name, shape in SHAPES:
        data = np.random.randn(*shape).astype(DT)

        # ── ION cached: flush/inv 有效 ──
        ic = IONArray(DT, shape, cached=True)
        ic.as_array()[:] = data
        t0 = now(); ic.flush(); tf_c = now() - t0
        t0 = now(); ic.invalidate(); ti_c = now() - t0
        t0 = now(); c_out = ic.as_array().copy(); tr_c = now() - t0
        ok_c = "OK" if np.allclose(data, c_out, atol=1e-5) else "FAIL"
        del ic, c_out

        # ── ION uncached: flush/inv 是 no-op ──
        iu = IONArray(DT, shape, cached=False)
        iu.as_array()[:] = data
        t0 = now(); iu.flush(); tf_u = now() - t0
        t0 = now(); iu.invalidate(); ti_u = now() - t0
        t0 = now(); u_out = iu.as_array().copy(); tr_u = now() - t0
        ok_u = "OK" if np.allclose(data, u_out, atol=1e-5) else "FAIL"
        del iu, u_out

        assert ok_c == "OK" and ok_u == "OK"

        rows.append([name, hsz(data.nbytes),
                     f"{tf_c:.4f}", f"{ti_c:.4f}", f"{tr_c:.3f}",
                     f"{tf_u:.4f}", f"{ti_u:.4f}", f"{tr_u:.3f}",
                     ok_c, ok_u])

        del data

    table(["名称","大小",
           "Cflush","Cinv","C读",
           "Uflush","Uinv","U读",
           "c精度","u精度"], rows)

# ═══════════════════════════════════════════════════════════
# 测试 4: 切片/索引赋值
# ═══════════════════════════════════════════════════════════

def test_04_slice():
    bar("4. 切片索引赋值: numpy vs ION(cached) vs ION(uncached)")
    rows = []
    for name, shape in SHAPES:
        nbytes = int(np.prod(shape)) * DT.itemsize

        # ── numpy ──
        np_arr = np.zeros(shape, dtype=DT)
        np_arr[..., 0] = 42.0                          # 首元素沿最后一维
        if np_arr.ndim >= 2:
            np_arr[0, ...] = -1.0                      # 第一行/片

        # ── ION cached ──
        ic = IONArray(DT, shape, cached=True)
        ic.as_array()[:] = 0
        ic.as_array()[..., 0] = 42.0
        if ic.as_array().ndim >= 2:
            ic.as_array()[0, ...] = -1.0
        ok_c = "OK" if np.allclose(np_arr, ic.as_array(), atol=1e-5) else "FAIL"
        del ic

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        iu.as_array()[:] = 0
        iu.as_array()[..., 0] = 42.0
        if iu.as_array().ndim >= 2:
            iu.as_array()[0, ...] = -1.0
        ok_u = "OK" if np.allclose(np_arr, iu.as_array(), atol=1e-5) else "FAIL"
        del iu

        assert ok_c == "OK" and ok_u == "OK"

        rows.append([name, str(shape), hsz(nbytes), ok_c, ok_u])
        del np_arr

    table(["名称","shape","大小","c精度","u精度"], rows)

# ═══════════════════════════════════════════════════════════
# 测试 5: 算术运算 (不修改原数组)
# ═══════════════════════════════════════════════════════════

def test_05_arithmetic():
    bar("5. 算术运算: 二元/三元/比较/聚合/数学 — numpy vs ION(cached) vs ION(uncached)")

    summary = []
    for name, shape in SHAPES:
        nbytes = int(np.prod(shape)) * DT.itemsize

        # 准备 3 组随机数据
        a = np.random.randn(*shape).astype(DT) * 2 + 1
        b = np.random.randn(*shape).astype(DT) * 0.5 + 0.1
        c = np.random.randn(*shape).astype(DT) * 0.1

        # ── numpy 参考 ──
        na, nb, nc = a.copy(), b.copy(), c.copy()

        # ── ION cached (3 个独立数组) ──
        ca = IONArray(DT, shape, cached=True); ca.as_array()[:] = a
        cb = IONArray(DT, shape, cached=True); cb.as_array()[:] = b
        cc = IONArray(DT, shape, cached=True); cc.as_array()[:] = c

        # ── ION uncached (3 个独立数组) ──
        ua = IONArray(DT, shape, cached=False); ua.as_array()[:] = a
        ub = IONArray(DT, shape, cached=False); ub.as_array()[:] = b
        uc = IONArray(DT, shape, cached=False); uc.as_array()[:] = c

        ok, fail = 0, []

        # ─── 二元逐元素运算 ───
        for label, fn in [
            ("a+b",  lambda x, y: x + y),
            ("a-b",  lambda x, y: x - y),
            ("a*b",  lambda x, y: x * y),
            ("a/b",  lambda x, y: x / (y + 0.01)),
            ("a**2", lambda x, y: x ** 2),
        ]:
            ref = fn(na, nb)
            rc  = fn(ca.as_array(), cb.as_array())
            ru  = fn(ua.as_array(), ub.as_array())
            if np.allclose(ref, rc, atol=1e-4): ok += 1
            else: fail.append(f"c.{label}")
            if np.allclose(ref, ru, atol=1e-4): ok += 1
            else: fail.append(f"u.{label}")

        # ─── 三元融合 ───
        for label, expr in [
            ("a*b+c",   na * nb + nc),
            ("(a+b)*c", (na + nb) * nc),
        ]:
            ref = expr
            if label == "a*b+c":
                rc = ca.as_array() * cb.as_array() + cc.as_array()
                ru = ua.as_array() * ub.as_array() + uc.as_array()
            else:
                rc = (ca.as_array() + cb.as_array()) * cc.as_array()
                ru = (ua.as_array() + ub.as_array()) * uc.as_array()
            if np.allclose(ref, rc, atol=1e-4): ok += 1
            else: fail.append(f"c.{label}")
            if np.allclose(ref, ru, atol=1e-4): ok += 1
            else: fail.append(f"u.{label}")

        # ─── 比较运算 ───
        ref_gt = na > nb
        if np.array_equal(ref_gt, ca.as_array() > cb.as_array()): ok += 1
        else: fail.append("c.a>b")
        if np.array_equal(ref_gt, ua.as_array() > ub.as_array()): ok += 1
        else: fail.append("u.a>b")

        # ─── 聚合 ───
        for label, fn in [
            ("sum",  np.sum),   ("mean", np.mean),
            ("max",  np.max),   ("min",  np.min),
            ("std",  np.std),   ("var",  np.var),
        ]:
            ref = fn(na)
            if np.allclose(ref, fn(ca.as_array()), atol=1e-4): ok += 1
            else: fail.append(f"c.{label}")
            if np.allclose(ref, fn(ua.as_array()), atol=1e-4): ok += 1
            else: fail.append(f"u.{label}")

        # ─── 数学函数 ───
        for label, fn in [
            ("sqrt", lambda x: np.sqrt(np.abs(x))),
            ("exp",  lambda x: np.exp(np.clip(x, -5, 5))),
            ("abs",  lambda x: np.abs(x)),
            ("sin",  lambda x: np.sin(x)),
        ]:
            ref = fn(na)
            if np.allclose(ref, fn(ca.as_array()), atol=1e-4): ok += 1
            else: fail.append(f"c.{label}")
            if np.allclose(ref, fn(ua.as_array()), atol=1e-4): ok += 1
            else: fail.append(f"u.{label}")

        # ─── 原数组完整性 ───
        if np.allclose(a, ca.as_array(), atol=1e-5): ok += 1
        else: fail.append("c.intact")
        if np.allclose(a, ua.as_array(), atol=1e-5): ok += 1
        else: fail.append("u.intact")

        total = ok + len(fail)
        status = "OK" if len(fail) == 0 else f"FAIL:{','.join(fail)}"
        summary.append([name, hsz(nbytes), str(ok), str(total), status])

        assert len(fail) == 0, f"{name}: {fail}"

        del ca, cb, cc, ua, ub, uc
        del a, b, c, na, nb, nc

    table(["名称","大小","通过","总数","状态"], summary)

# ═══════════════════════════════════════════════════════════
# 测试 6: 写入带宽 (MB/s)
# ═══════════════════════════════════════════════════════════

def test_06_bandwidth():
    bar("6. 写入带宽: ION(cached) vs ION(uncached) — 10轮平均")
    rows = []
    for name, shape in SHAPES:
        data = np.random.randn(*shape).astype(DT)
        mb = data.nbytes / 1e6

        # ── ION cached ──
        ic = IONArray(DT, shape, cached=True)
        ct = []
        for _ in range(10):
            t0 = now(); ic.as_array()[:] = data; ct.append(now() - t0)
        bw_c = mb / (np.mean(ct) / 1000)
        del ic

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        ut = []
        for _ in range(10):
            t0 = now(); iu.as_array()[:] = data; ut.append(now() - t0)
        bw_u = mb / (np.mean(ut) / 1000)
        del iu

        ratio = bw_u / max(bw_c, 0.001)
        rows.append([name, hsz(data.nbytes),
                     f"{bw_c:.0f}", f"{bw_u:.0f}",
                     f"{ratio:.1f}x"])

        del data

    table(["名称","大小","C(MB/s)","U(MB/s)","U/C"], rows)

# ═══════════════════════════════════════════════════════════
# 测试 7: phy_addr 唯一性
# ═══════════════════════════════════════════════════════════

def test_07_phy_addr():
    bar("7. phy_addr 唯一性: 10 cached + 10 uncached")
    ions = []
    for i in range(10):
        ions.append(IONArray(DT, [1000], cached=True))
        ions.append(IONArray(DT, [1000], cached=False))
    addrs = [ion.phy_addr for ion in ions]
    unique = len(set(addrs))
    assert unique == 20, f"期望 20 个唯一地址, 实际 {unique}"
    print(f"  20 个 IONArray → {unique} 个唯一 phy_addr → OK")

# ═══════════════════════════════════════════════════════════
# 测试 8: 广播赋值
# ═══════════════════════════════════════════════════════════

def test_08_broadcast():
    bar("8. 广播赋值: numpy vs ION(cached) vs ION(uncached)")
    rows = []
    for name, shape in SHAPES:
        if len(shape) < 2:
            continue  # 1D 跳过广播测试
        nbytes = int(np.prod(shape)) * DT.itemsize
        nd = len(shape)

        # 广播行: 沿最后一维, 固定前面所有维为 0
        row_idx = tuple([0] * (nd - 1) + [slice(None)])  # (0, 0, ..., :)
        row_val = np.arange(shape[-1], dtype=DT)

        # ── numpy ──
        np_arr = np.zeros(shape, dtype=DT)
        t0 = now(); np_arr[:] = 5.0; ts_np = now() - t0
        t0 = now(); np_arr[row_idx] = row_val; tr_np = now() - t0

        # ── ION cached ──
        ic = IONArray(DT, shape, cached=True)
        ic.as_array()[:] = 0
        t0 = now(); ic.as_array()[:] = 5.0; ts_c = now() - t0
        t0 = now(); ic.as_array()[row_idx] = row_val; tr_c = now() - t0
        ok_c = "OK" if np.allclose(np_arr, ic.as_array(), atol=1e-5) else "FAIL"
        del ic

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        iu.as_array()[:] = 0
        t0 = now(); iu.as_array()[:] = 5.0; ts_u = now() - t0
        t0 = now(); iu.as_array()[row_idx] = row_val; tr_u = now() - t0
        ok_u = "OK" if np.allclose(np_arr, iu.as_array(), atol=1e-5) else "FAIL"
        del iu

        assert ok_c == "OK" and ok_u == "OK"

        rows.append([name, str(shape), hsz(nbytes),
                     f"{ts_np:.3f}", f"{ts_c:.3f}", f"{ts_u:.3f}",
                     f"{tr_np:.3f}", f"{tr_c:.3f}", f"{tr_u:.3f}",
                     ok_c, ok_u])

        del np_arr

    table(["名称","shape","大小",
           "标np","标IONc","标IONu",
           "行np","行IONc","行IONu",
           "c","u"], rows)

# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    TESTS = [
        ("1.创建",            test_01_create),
        ("2.写入+读回",       test_02_write_read),
        ("3.flush+inv往返",   test_03_flush),
        ("4.切片索引",        test_04_slice),
        ("5.算术运算",        test_05_arithmetic),
        ("6.写入带宽",        test_06_bandwidth),
        ("7.phy_addr唯一性",  test_07_phy_addr),
        ("8.广播赋值",        test_08_broadcast),
    ]

    results = []
    for label, fn in TESTS:
        try:
            fn()
            results.append((label, "OK"))
        except Exception as e:
            results.append((label, f"FAIL — {e}"))

    print(f"\n{'='*80}")
    print("  结果汇总")
    print(f"{'='*80}")
    for label, status in results:
        print(f"  {label:20s} {status}")
    n_ok = sum(1 for _, s in results if s == "OK")
    print(f"  {'─'*40}")
    print(f"  {n_ok}/{len(results)} 通过")
    sys.exit(0 if n_ok == len(results) else 1)
