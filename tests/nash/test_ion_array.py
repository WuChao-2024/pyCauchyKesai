"""
IONArray 全排列测试: 每个 shape × 每个操作, numpy vs ION(cached) vs ION(uncached) 全对比
"""

import numpy as np
from pyCauchyKesai import IONArray, IONArrayDesc
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

        # 属性检查（phy_addr/is_cached 已下沉内部；用 is_allocated + 数据独立性验证）
        assert ic.is_allocated and iu.is_allocated
        ic.numpy()[:] = 1.0
        assert iu.numpy().flat[0] != 1.0, "cached/uncached 应是独立内存"
        assert ic.numpy().shape == tuple(shape)
        assert ic.numpy().dtype == DT

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
        t0 = now(); ic.numpy()[:] = data; tw_c = now() - t0
        t0 = now(); c_out = ic.numpy().copy(); tr_c = now() - t0
        ok_c = "OK" if np.allclose(data, c_out, atol=1e-5) else "FAIL"
        del ic, c_out

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        t0 = now(); iu.numpy()[:] = data; tw_u = now() - t0
        t0 = now(); u_out = iu.numpy().copy(); tr_u = now() - t0
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
        ic.numpy()[:] = data
        t0 = now(); ic.flush_clean(); tf_c = now() - t0
        t0 = now(); ic.flush_invalidate(); ti_c = now() - t0
        t0 = now(); c_out = ic.numpy().copy(); tr_c = now() - t0
        ok_c = "OK" if np.allclose(data, c_out, atol=1e-5) else "FAIL"
        del ic, c_out

        # ── ION uncached: flush/inv 是 no-op ──
        iu = IONArray(DT, shape, cached=False)
        iu.numpy()[:] = data
        t0 = now(); iu.flush_clean(); tf_u = now() - t0
        t0 = now(); iu.flush_invalidate(); ti_u = now() - t0
        t0 = now(); u_out = iu.numpy().copy(); tr_u = now() - t0
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
        ic.numpy()[:] = 0
        ic.numpy()[..., 0] = 42.0
        if ic.numpy().ndim >= 2:
            ic.numpy()[0, ...] = -1.0
        ok_c = "OK" if np.allclose(np_arr, ic.numpy(), atol=1e-5) else "FAIL"
        del ic

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        iu.numpy()[:] = 0
        iu.numpy()[..., 0] = 42.0
        if iu.numpy().ndim >= 2:
            iu.numpy()[0, ...] = -1.0
        ok_u = "OK" if np.allclose(np_arr, iu.numpy(), atol=1e-5) else "FAIL"
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
        ca = IONArray(DT, shape, cached=True); ca.numpy()[:] = a
        cb = IONArray(DT, shape, cached=True); cb.numpy()[:] = b
        cc = IONArray(DT, shape, cached=True); cc.numpy()[:] = c

        # ── ION uncached (3 个独立数组) ──
        ua = IONArray(DT, shape, cached=False); ua.numpy()[:] = a
        ub = IONArray(DT, shape, cached=False); ub.numpy()[:] = b
        uc = IONArray(DT, shape, cached=False); uc.numpy()[:] = c

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
            rc  = fn(ca.numpy(), cb.numpy())
            ru  = fn(ua.numpy(), ub.numpy())
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
                rc = ca.numpy() * cb.numpy() + cc.numpy()
                ru = ua.numpy() * ub.numpy() + uc.numpy()
            else:
                rc = (ca.numpy() + cb.numpy()) * cc.numpy()
                ru = (ua.numpy() + ub.numpy()) * uc.numpy()
            if np.allclose(ref, rc, atol=1e-4): ok += 1
            else: fail.append(f"c.{label}")
            if np.allclose(ref, ru, atol=1e-4): ok += 1
            else: fail.append(f"u.{label}")

        # ─── 比较运算 ───
        ref_gt = na > nb
        if np.array_equal(ref_gt, ca.numpy() > cb.numpy()): ok += 1
        else: fail.append("c.a>b")
        if np.array_equal(ref_gt, ua.numpy() > ub.numpy()): ok += 1
        else: fail.append("u.a>b")

        # ─── 聚合 ───
        for label, fn in [
            ("sum",  np.sum),   ("mean", np.mean),
            ("max",  np.max),   ("min",  np.min),
            ("std",  np.std),   ("var",  np.var),
        ]:
            ref = fn(na)
            if np.allclose(ref, fn(ca.numpy()), atol=1e-4): ok += 1
            else: fail.append(f"c.{label}")
            if np.allclose(ref, fn(ua.numpy()), atol=1e-4): ok += 1
            else: fail.append(f"u.{label}")

        # ─── 数学函数 ───
        for label, fn in [
            ("sqrt", lambda x: np.sqrt(np.abs(x))),
            ("exp",  lambda x: np.exp(np.clip(x, -5, 5))),
            ("abs",  lambda x: np.abs(x)),
            ("sin",  lambda x: np.sin(x)),
        ]:
            ref = fn(na)
            if np.allclose(ref, fn(ca.numpy()), atol=1e-4): ok += 1
            else: fail.append(f"c.{label}")
            if np.allclose(ref, fn(ua.numpy()), atol=1e-4): ok += 1
            else: fail.append(f"u.{label}")

        # ─── 原数组完整性 ───
        if np.allclose(a, ca.numpy(), atol=1e-5): ok += 1
        else: fail.append("c.intact")
        if np.allclose(a, ua.numpy(), atol=1e-5): ok += 1
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
            t0 = now(); ic.numpy()[:] = data; ct.append(now() - t0)
        bw_c = mb / (np.mean(ct) / 1000)
        del ic

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        ut = []
        for _ in range(10):
            t0 = now(); iu.numpy()[:] = data; ut.append(now() - t0)
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

def test_07_memory_independence():
    bar("7. 内存独立性: 20 个 IONArray 互不干扰（替代原 phy_addr 唯一性）")
    ions = []
    for i in range(10):
        ions.append(IONArray(DT, [1000], cached=True))
        ions.append(IONArray(DT, [1000], cached=False))
    # phy_addr 已下沉内部；用数据隔离验证每个 IONArray 是独立内存
    for idx, ion in enumerate(ions):
        ion.numpy()[:] = idx  # 各写不同值
    ok = all(int(ions[i].numpy().flat[0]) == i for i in range(len(ions)))
    assert ok, "IONArray 间存在内存共享（写入互相污染）"
    print(f"  20 个 IONArray 各写各的值 → 读回一致 → 内存独立 OK")

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
        ic.numpy()[:] = 0
        t0 = now(); ic.numpy()[:] = 5.0; ts_c = now() - t0
        t0 = now(); ic.numpy()[row_idx] = row_val; tr_c = now() - t0
        ok_c = "OK" if np.allclose(np_arr, ic.numpy(), atol=1e-5) else "FAIL"
        del ic

        # ── ION uncached ──
        iu = IONArray(DT, shape, cached=False)
        iu.numpy()[:] = 0
        t0 = now(); iu.numpy()[:] = 5.0; ts_u = now() - t0
        t0 = now(); iu.numpy()[row_idx] = row_val; tr_u = now() - t0
        ok_u = "OK" if np.allclose(np_arr, iu.numpy(), atol=1e-5) else "FAIL"
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
# 测试 9: from_numpy — 布局感知写入（natural 路径 + 校验）
# ═══════════════════════════════════════════════════════════

def test_09_from_numpy():
    bar("9. from_numpy 布局感知写入: natural 回读 + dtype/shape/size 校验")
    rows = []
    for name, shape in SHAPES:
        data = np.random.randn(*shape).astype(DT)

        # cached + uncached 回读一致性
        ic = IONArray(DT, shape, cached=True);  ic.from_numpy(data)
        iu = IONArray(DT, shape, cached=False); iu.from_numpy(data)
        ok_c = "OK" if np.allclose(data, ic.numpy(), atol=1e-5) else "FAIL"
        ok_u = "OK" if np.allclose(data, iu.numpy(), atol=1e-5) else "FAIL"
        # numpy strides 必须是 natural（bare 无 padding）
        nat_strides = tuple(
            int(np.prod(shape[i+1:])) * DT.itemsize for i in range(len(shape))
        )
        stride_ok = ic.numpy().strides == nat_strides
        del ic, iu
        assert ok_c == "OK" and ok_u == "OK" and stride_ok, f"{name}: c={ok_c} u={ok_u} stride_ok={stride_ok}"
        rows.append([name, str(shape), ok_c, ok_u, "OK" if stride_ok else "FAIL"])

    # ── 校验：dtype / shape / 超大 三类都应抛 ──
    ion = IONArray(DT, [4, 8], cached=True)
    def expect_throw(label, arr):
        try:
            ion.from_numpy(arr); return f"{label}:未抛(错!)"
        except Exception: return f"{label}:抛✓"
    v1 = expect_throw("dtype", np.zeros((4,8), dtype=np.int32))
    v2 = expect_throw("shape", np.zeros((4,9), dtype=DT))
    v3 = expect_throw("size",  np.zeros((100,100), dtype=DT))
    print(f"  校验: {v1}  {v2}  {v3}")
    assert "未抛" not in v1+v2+v3

    table(["名称","shape","c回读","u回读","natural stride"], rows)

# ═══════════════════════════════════════════════════════════
# 测试 10: is_padded_layout — bare IONArray 必为 False（natural）
#
# 注：padded=True 的情形（strided/对齐布局）需从模型 IONArray(m.input_descs[i])
# 构建，依赖 BPU 加载 .hbm。当前环境 BPU 被 robotrea 服务占用（iova 失败），
# 故 padded=True 的 runtime 验证留待硬件可用时补充。
# 代码层保证：from_numpy/numpy/fill_properties 三者共用同一 stride_，故
#   写-stride == 读-stride == BPU-stride，A1（导出口 natural vs BPU aligned）
#   的 mismatch 已从结构上消除。
# ═══════════════════════════════════════════════════════════

def test_10_natural_layout():
    bar("10. natural 布局: bare IONArray 的 numpy strides 必连续（替代原 is_padded_layout）")
    rows = []
    for name, shape in SHAPES:
        ion = IONArray(DT, shape, cached=True)
        # is_padded_layout 已下沉内部；bare IONArray 的 numpy 必是 natural 连续 strides
        nat_strides = tuple(int(np.prod(shape[i+1:])) * DT.itemsize for i in range(len(shape)))
        is_natural = ion.numpy().strides == nat_strides
        assert is_natural, f"{name}: bare 应为 natural strides, 实际 {ion.numpy().strides}"
        rows.append([name, str(shape), "natural" if is_natural else "FAIL"])
        del ion
    table(["名称","shape","layout"], rows)

# ═══════════════════════════════════════════════════════════
# 测试 11: __array__ 协议 — np.asarray(ion)/np.sum(ion) 直通 + 零拷贝
#
# 注：padded/strided 路径的 __array__ 保真（strided 视图）需从模型
# IONArray(m.input_descs[i]) 构建，依赖 BPU 加载 .hbm；当前环境 BPU 被占用（iova 失败），
# 故仅验 bare+natural 路径。__array__ 复用 numpy()，布局正确性同 numpy。
# ═══════════════════════════════════════════════════════════

def test_11_array_protocol():
    bar("11. __array__ 协议: np.asarray(ion)/np.sum(ion) 直通 + 零拷贝")
    rows = []
    for name, shape in SHAPES:
        data = np.random.randn(*shape).astype(DT)
        ion = IONArray(DT, shape, cached=True)
        ion.numpy()[:] = data

        # 1) np.asarray 返回 ndarray，shape/dtype/strides 与 numpy 完全一致
        a = np.asarray(ion)
        shape_ok = a.shape == tuple(shape)
        dtype_ok = a.dtype == DT
        strides_ok = a.strides == ion.numpy().strides

        # 2) 数据保真
        data_ok = np.array_equal(a, ion.numpy())

        # 3) 零拷贝铁证：np.asarray(ion) 与 ion.numpy() 共享同一物理内存
        shares = np.shares_memory(a, ion.numpy())
        a.flat[0] = 999.0
        sync_ok = ion.numpy().flat[0] == 999.0   # 改 a 后 ion 同步 → 同一块内存
        a.flat[0] = float(data.flat[0])              # 还原

        # 4) numpy 函数直通（无需显式 .numpy()）
        sum_ok = np.isclose(np.sum(ion), ion.numpy().sum(), atol=1e-4)
        sqrt_ok = np.allclose(np.sqrt(np.abs(ion)),
                              np.sqrt(np.abs(ion.numpy())), atol=1e-4)

        ok = all([shape_ok, dtype_ok, strides_ok, data_ok,
                  shares, sync_ok, sum_ok, sqrt_ok])
        rows.append([name, str(shape),
                     "OK" if data_ok else "FAIL",
                     "OK" if shares else "FAIL",
                     "OK" if sync_ok else "FAIL",
                     "OK" if sum_ok else "FAIL"])
        assert ok, f"{name}: shape={shape_ok} dtype={dtype_ok} strides={strides_ok} " \
                   f"data={data_ok} shares={shares} sync={sync_ok} sum={sum_ok} sqrt={sqrt_ok}"
        del ion

    table(["名称","shape","数据保真","共享内存","改写同步","np.sum直通"], rows)

    # 5) dtype 转换：f16→f32 应触发拷贝（脱离原物理内存）
    ion16 = IONArray(np.dtype('float16'), [4, 8], cached=True)
    ion16.numpy()[:] = np.arange(32, dtype=np.float16).reshape(4, 8)
    a32 = np.asarray(ion16, dtype=np.float32)
    cast_ok = a32.dtype == np.float32 and np.allclose(
        a32, ion16.numpy().astype(np.float32), atol=1e-3)
    a32.flat[0] = -1.0
    indep_ok = ion16.numpy().flat[0] != -1.0    # 拷贝独立，改 a32 不影响 ion16
    print(f"  dtype转换 f16→f32: {'OK' if cast_ok else 'FAIL'}, 拷贝独立: {'OK' if indep_ok else 'FAIL'}")
    assert cast_ok and indep_ok

# ═══════════════════════════════════════════════════════════
# 测试 12: 共享 IONMemory — from_memory 工厂 + 偏移隔离
# ═══════════════════════════════════════════════════════════

def test_12_shared_ion_memory():
    bar("12. 共享 IONMemory: from_memory + 偏移隔离")
    from pyCauchyKesai import IONMemory

    DT32 = np.dtype('float32')
    shape = [1024]  # 4096 bytes each

    # 创建 IONMemory: 8192 bytes (放两个 1024-float32 区块)
    m = IONMemory(8192, cached=True)
    assert m.is_allocated and m.size == 8192

    # 模板 desc（无内存；from_memory 继承其 tensor 性质）
    tpl = IONArrayDesc(DT32, shape)

    # 从同一 IONMemory 创建两个偏移视图
    a = IONArray.from_memory(m, 0, tpl)
    b = IONArray.from_memory(m, 4096, tpl)

    # 1) 共享验证: a.memory 是同一个 IONMemory 对象
    assert a.memory is b.memory, "a and b must share the same IONMemory"
    assert a.memory is m, "a.memory must be the original IONMemory"

    # 2) 偏移隔离（替代原 phy_addr 偏移断言）: 写 a 不影响 b，证明偏移正确
    a_arr = np.arange(1024, dtype=DT32)
    b_arr = np.arange(2000, 3024, dtype=DT32)  # 不同数据，1024 元素
    a.from_numpy(a_arr)
    b.from_numpy(b_arr)

    a_read = a.numpy().copy()
    b_read = b.numpy().copy()
    assert np.array_equal(a_read, a_arr), "a data mismatch after write"
    assert np.array_equal(b_read, b_arr), "b data mismatch (contaminated by a!)"

    # 3) 基本属性（is_cached/mem_size 下沉内部；用 is_allocated + desc.nbytes）
    assert a.is_allocated and b.is_allocated
    assert a.desc.nbytes() == 4096 and b.desc.nbytes() == 4096

    print("  共享验证: a.memory is 同一对象 OK")
    print("  偏移隔离: 写 a 不影响 b（偏移正确）OK")
    print("  全部通过")

# ═══════════════════════════════════════════════════════════
# 测试 13: from_memory 工厂 — 性质继承 + sys_mem 偏移
# ═══════════════════════════════════════════════════════════

def test_13_from_memory_factory():
    bar("13. from_memory 工厂: 性质继承 + sys_mem 偏移")
    from pyCauchyKesai import IONMemory

    DT = np.dtype('uint8')
    shape = [4, 8]  # 32 bytes

    # 构造模板 desc（无内存；from_memory 继承其 tensor 性质）
    tpl = IONArrayDesc(DT, shape)
    # from_memory 会继承这些性质

    m = IONMemory(128, cached=True)
    # ION 内存未清零：用 sentinel 填充，以便验证偏移写入不污染其他区域
    m.numpy()[:] = 0xAA

    ion = IONArray.from_memory(m, 64, tpl)

    # 性质继承验证（is_cached 下沉内部；dtype/shape 在 desc 上）
    assert ion.desc.dtype == DT, f"dtype mismatch: {ion.desc.dtype} vs {DT}"
    assert tuple(ion.desc.shape) == tuple(shape), f"shape mismatch: {ion.desc.shape} vs {shape}"
    assert ion.is_allocated
    assert ion.desc.nbytes() == 32  # 4×8×1

    # 偏移正确性（替代原 phy_addr 偏移 + mem_size 断言，验证更强）:
    # offset=64 写入 32 字节，offset 0-63 不污染，offset 64-95 为写入数据
    data = np.arange(32, dtype=DT).reshape(4, 8)
    ion.from_numpy(data)
    readback = ion.numpy().copy()
    assert np.array_equal(readback, data), "data readback mismatch"
    raw = m.numpy()
    assert np.all(raw[0:64] == 0xAA), "offset 0-63 不应被 offset=64 的写入污染"
    assert np.all(raw[96:128] == 0xAA), "offset 96-127 不应被污染"
    assert np.array_equal(raw[64:96], data.flatten()), "offset 64-95 应为写入数据"

    print("  性质继承: dtype/shape OK")
    print("  偏移正确: offset=64 写入读回一致，前后 sentinel 边界未污染 OK")
    print("  from_numpy + numpy 往返 OK")

# ═══════════════════════════════════════════════════════════
# 测试 14: move 后安全 — use-after-move 不 crash
# ═══════════════════════════════════════════════════════════

def test_14_deferred_and_move_patterns():
    bar("14. defer + clone + from_memory 生命周期模式")
    from pyCauchyKesai import IONMemory

    DT = np.dtype('float32')

    # ── 14a: deferred IONArray (no memory until allocate) ──
    ion_def = IONArray(DT, [100], cached=True, defer=True)
    assert not ion_def.is_allocated, "deferred IONArray should not be allocated"
    # flush/invalidate on unallocated → no crash
    ion_def.flush_clean()
    ion_def.flush_invalidate()
    # allocate
    ion_def.allocate()
    assert ion_def.is_allocated
    data = np.arange(100, dtype=DT)
    ion_def.from_numpy(data)
    assert np.array_equal(ion_def.numpy().copy(), data)
    print("  14a: defer→allocate→from_numpy→numpy OK")

    # ── 14b: 独立分配不共享内存（替代原 clone 测试，clone 已下沉内部） ──
    src = IONArray(DT, [50], cached=True)
    src.from_numpy(np.ones(50, dtype=DT))

    # 独立构造的 IONArray 不应共享 src 的 IONMemory
    indep = IONArray(DT, [50], cached=True)
    assert indep.memory is not src.memory, "独立构造应分配独立 IONMemory"
    # 写 indep 不影响 src（数据隔离）
    indep.numpy()[:] = 42.0
    assert src.numpy().flat[0] == 1.0, "写独立 IONArray 不应影响 src"
    print("  14b: 独立 IONArray 不共享 src 内存 + 数据隔离 OK")

    # ── 14c: from_memory + 共享 IONMemory + 生命周期 ──
    m = IONMemory(4096, cached=True)
    tpl = IONArrayDesc(DT, [512])
    a = IONArray.from_memory(m, 0, tpl)
    b = IONArray.from_memory(m, 2048, tpl)
    # a/b 共享同一 IONMemory
    assert a.memory is b.memory
    # IONMemory 通过 shared_ptr 保活，即使 m 变量被覆盖
    m = None  # noqa: 让 Python GC m
    # a/b 仍然有效（shared_ptr 持有 IONMemory）
    assert a.is_allocated
    assert b.is_allocated
    a_data = np.arange(512, dtype=DT)
    a.from_numpy(a_data)
    assert np.array_equal(a.numpy().copy(), a_data)
    print("  14c: from_memory + IONMemory 生命周期保活（m=None 后仍有效） OK")

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
        ("7.内存独立性",       test_07_memory_independence),
        ("8.广播赋值",        test_08_broadcast),
        ("9.from_numpy布局感知", test_09_from_numpy),
        ("10.natural_layout",  test_10_natural_layout),
        ("11.__array__协议",   test_11_array_protocol),
        ("12.共享IONMemory",   test_12_shared_ion_memory),
        ("13.from_memory工厂", test_13_from_memory_factory),
        ("14.生命周期模式",   test_14_deferred_and_move_patterns),
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
