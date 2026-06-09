"""
IONArray 前处理 Demo: YOLO & ResNet
从 cv2.imread 到 IONArray 模型输入，逐步分析内存拷贝

运行: cd examples && python ion_preprocess_demo.py
"""

import numpy as np
import cv2
import time
import sys
from pyCauchyKesai import IONArray

# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def now():
    return time.perf_counter() * 1000

def is_view(arr):
    """判断数组是否是视图 (共享内存)"""
    base = arr.base
    return base is not None

def mem_location(arr):
    """判断数组数据在 ION 还是 CPU"""
    # 顺着 base 链找到最底层
    a = arr
    while a.base is not None:
        a = a.base
    # 如果最底层是 as_array() 返回的 ndarray，它的 base 是 IONArray 的 capsule
    # 简单判断: 如果 arr 拥有自己的 data (base is None)，就是独立 CPU 内存
    if arr.__array_interface__['data'][0] == a.__array_interface__['data'][0]:
        return "视图(共享)"
    return "独立(新分配)"

def check_equal(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)

# ═══════════════════════════════════════════════════════════
# YOLO 前处理
#   输入: uint8 BGR HWC (任意分辨率)
#   输出: float32 RGB NCHW [1, 3, 640, 640], 像素值 0~1
# ═══════════════════════════════════════════════════════════

def yolo_standard(img, target=640):
    """标准 CPU 前处理: 全部在 CPU 内存"""
    steps = []

    t0 = now(); resized = cv2.resize(img, (target, target)); steps.append(("resize", now()-t0, "CPU新分配", resized.nbytes))
    t0 = now(); rgb = resized[..., ::-1];                    steps.append(("BGR→RGB", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); f32 = rgb.astype(np.float32);                steps.append(("uint8→float32", now()-t0, "CPU新分配", f32.nbytes))
    t0 = now(); f32 /= 255.0;                                steps.append(("/255", now()-t0, "CPU原地", 0))
    t0 = now(); chw = f32.transpose(2, 0, 1);               steps.append(("HWC→CHW", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); nchw = chw[np.newaxis];                      steps.append(("加batch维", now()-t0, "视图(零拷贝)", 0))

    return nchw, steps

def yolo_ion(img, target=640):
    """ION 前处理: 类型转换后直接写 ION, normalize 原地"""
    steps = []

    t0 = now(); resized = cv2.resize(img, (target, target)); steps.append(("resize", now()-t0, "CPU新分配", resized.nbytes))
    t0 = now(); rgb = resized[..., ::-1];                    steps.append(("BGR→RGB", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); chw = rgb.transpose(2, 0, 1);               steps.append(("HWC→CHW", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); nchw_view = chw[np.newaxis];                 steps.append(("加batch维", now()-t0, "视图(零拷贝)", 0))

    # float32 转换 + 写入 ION (一次拷贝)
    ion = IONArray(np.dtype('float32'), [1, 3, target, target])
    a = ion.as_array()  # 拿到 numpy 视图引用
    t0 = now(); a[:] = nchw_view.astype(np.float32);        steps.append(("→float32+写ION", now()-t0, "CPU新分配→ION", ion.mem_size))

    # normalize 在 ION 内存原地操作
    t0 = now(); a /= 255.0;                                 steps.append(("/255", now()-t0, "ION原地", 0))
    t0 = now(); ion.flush();                      steps.append(("flush", now()-t0, "cache操作", 0))

    return ion, steps

def yolo_ion_resize_direct(img, target=640):
    """ION 激进模式: cv2.resize 直接写入 uint8 IONArray"""
    steps = []

    # resize 直接写入 ION
    ion_u8 = IONArray(np.dtype('uint8'), [target, target, 3])
    t0 = now(); cv2.resize(img, (target, target), dst=ion_u8.as_array()); steps.append(("resize→ION", now()-t0, "直写ION", ion_u8.mem_size))

    # BGR→RGB + HWC→CHW + 加batch维 (全是视图)
    t0 = now(); rgb = ion_u8.as_array()[..., ::-1];             steps.append(("BGR→RGB", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); chw = rgb.transpose(2, 0, 1);               steps.append(("HWC→CHW", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); nchw = chw[np.newaxis];                     steps.append(("加batch维", now()-t0, "视图(零拷贝)", 0))

    # uint8→float32 (不可避免) + 写入 ION
    ion_f32 = IONArray(np.dtype('float32'), [1, 3, target, target])
    a = ion_f32.as_array()
    t0 = now(); a[:] = nchw.astype(np.float32);              steps.append(("uint8→f32+写ION", now()-t0, "CPU新分配→ION", ion_f32.mem_size))

    # normalize 原地
    t0 = now(); a /= 255.0;                                  steps.append(("/255", now()-t0, "ION原地", 0))
    t0 = now(); ion_f32.flush();                   steps.append(("flush", now()-t0, "cache操作", 0))

    return ion_f32, steps

# ═══════════════════════════════════════════════════════════
# ResNet 前处理
#   输入: uint8 BGR HWC (任意分辨率)
#   输出: float32 RGB NCHW [1, 3, 224, 224], ImageNet normalize
# ═══════════════════════════════════════════════════════════

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def resnet_standard(img):
    """标准 CPU 前处理"""
    steps = []

    t0 = now(); resized = cv2.resize(img, (224, 224)); steps.append(("resize", now()-t0, "CPU新分配", resized.nbytes))
    t0 = now(); rgb = resized[..., ::-1];              steps.append(("BGR→RGB", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); f32 = rgb.astype(np.float32);          steps.append(("uint8→float32", now()-t0, "CPU新分配", f32.nbytes))
    t0 = now(); f32 /= 255.0;                          steps.append(("/255", now()-t0, "CPU原地", 0))
    t0 = now(); chw = f32.transpose(2, 0, 1);         steps.append(("HWC→CHW", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); nchw = chw[np.newaxis];                steps.append(("加batch维", now()-t0, "视图(零拷贝)", 0))
    # ImageNet normalize
    t0 = now()
    for c in range(3):
        nchw[0, c] = (nchw[0, c] - MEAN[c]) / STD[c]
    steps.append(("mean/std", now()-t0, "CPU原地", 0))

    return nchw, steps

def resnet_ion(img):
    """ION 前处理: normalize 全部原地"""
    steps = []

    t0 = now(); resized = cv2.resize(img, (224, 224)); steps.append(("resize", now()-t0, "CPU新分配", resized.nbytes))
    t0 = now(); rgb = resized[..., ::-1];              steps.append(("BGR→RGB", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); chw = rgb.transpose(2, 0, 1);         steps.append(("HWC→CHW", now()-t0, "视图(零拷贝)", 0))
    t0 = now(); nchw = chw[np.newaxis];                steps.append(("加batch维", now()-t0, "视图(零拷贝)", 0))

    ion = IONArray(np.dtype('float32'), [1, 3, 224, 224])
    a = ion.as_array()  # 拿到 numpy 视图引用
    t0 = now(); a[:] = nchw.astype(np.float32);            steps.append(("→float32+写ION", now()-t0, "CPU新分配→ION", ion.mem_size))

    # 全部 ION 原地
    t0 = now(); a /= 255.0;                                steps.append(("/255", now()-t0, "ION原地", 0))
    t0 = now()
    for c in range(3):
        a[0, c] -= MEAN[c]
        a[0, c] /= STD[c]
    steps.append(("mean/std", now()-t0, "ION原地", 0))
    t0 = now(); ion.flush();                     steps.append(("flush", now()-t0, "cache操作", 0))

    return ion, steps

# ═══════════════════════════════════════════════════════════
# 打印与对比
# ═══════════════════════════════════════════════════════════

import unicodedata

def _w(s):
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(s))

def _pad(s, w):
    return str(s) + ' ' * max(0, w - _w(s))

def table(headers, rows):
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

def print_steps(steps):
    rows = []
    copy_count = 0
    for name, ms, mem_type, nbytes in steps:
        is_copy = "新分配" in mem_type
        if is_copy:
            copy_count += 1
        rows.append([name, f"{ms:.3f}", mem_type, f"{nbytes/1e6:.2f}MB" if nbytes > 0 else "—"])
    table(["步骤","耗时ms","内存类型","数据量"], rows)
    return copy_count

# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 生成测试图像 (不用外部文件)
    img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    print(f"测试图像: {img.shape} {img.dtype} — {img.nbytes/1e6:.1f}MB")

    # ─── YOLO ───
    bar("YOLO 前处理: 标准 CPU 全流程")
    yolo_cpu, steps = yolo_standard(img)
    copies = print_steps(steps)
    print(f"  → 拷贝次数: {copies}  输出: {yolo_cpu.shape} {yolo_cpu.dtype}")

    bar("YOLO 前处理: ION 优化 (normalize 原地)")
    yolo_ion_arr, steps = yolo_ion(img)
    copies = print_steps(steps)
    print(f"  → 拷贝次数: {copies}  输出: IONArray phy_addr={hex(yolo_ion_arr.phy_addr)}")

    bar("YOLO 前处理: ION 激进 (resize 直写 ION)")
    yolo_ion_aggressive, steps = yolo_ion_resize_direct(img)
    copies = print_steps(steps)
    print(f"  → 拷贝次数: {copies}  输出: IONArray phy_addr={hex(yolo_ion_aggressive.phy_addr)}")

    # 精度对比
    yolo_ion_data = yolo_ion_arr.as_array().copy()
    yolo_ion_agg_data = yolo_ion_aggressive.as_array().copy()
    ok1 = check_equal(yolo_cpu, yolo_ion_data)
    ok2 = check_equal(yolo_cpu, yolo_ion_agg_data)
    print(f"\n  CPU vs ION优化:      {'OK' if ok1 else 'FAIL'}")
    print(f"  CPU vs ION激进:      {'OK' if ok2 else 'FAIL'}")

    del yolo_ion_arr, yolo_ion_aggressive, yolo_ion_data, yolo_ion_agg_data

    # ─── ResNet ───
    bar("ResNet 前处理: 标准 CPU 全流程")
    rn_cpu, steps = resnet_standard(img)
    copies = print_steps(steps)
    print(f"  → 拷贝次数: {copies}  输出: {rn_cpu.shape} {rn_cpu.dtype}")

    bar("ResNet 前处理: ION 优化 (normalize 原地)")
    rn_ion_arr, steps = resnet_ion(img)
    copies = print_steps(steps)
    print(f"  → 拷贝次数: {copies}  输出: IONArray phy_addr={hex(rn_ion_arr.phy_addr)}")

    # 精度对比
    rn_ion_data = rn_ion_arr.as_array().copy()
    ok3 = check_equal(rn_cpu, rn_ion_data)
    print(f"\n  CPU vs ION:          {'OK' if ok3 else 'FAIL'}")

    del rn_ion_arr, rn_ion_data

    # ─── 总结 ───
    bar("零拷贝分析总结")
    table(["步骤", "能否零拷贝", "原因", "替代方案"], [
        ["cv2.imread",    "不可", "OpenCV 控制，分配 CPU 内存", "—"],
        ["cv2.resize",    "可直写ION", "dst= 参数可指定输出数组", "IONArray(uint8) 作为 dst"],
        ["BGR→RGB",       "视图",   "[...,::-1] 是 numpy view",  "—"],
        ["uint8→float32", "不可",   "元素大小变化 (1B→4B)",       "astype 必然新分配"],
        ["/255 归一化",   "可ION原地", "/= 操作直接改 ION 内存",  "ion.as_array() /= 255.0"],
        ["HWC→CHW",       "视图",   "transpose 是 numpy view",    "—"],
        ["加 batch 维",   "视图",   "np.newaxis 是 numpy view",   "—"],
        ["mean/std",      "可ION原地", "-= /= 操作直接改 ION 内存", "ion.as_array()[0,c] -= mean"],
        ["flush",         "仅cached", "无 cache 则 no-op",        "uncached 免 flush"],
    ])
    print()
    print("  不可避免的拷贝: imread(1) + astype(1) = 2 次")
    print("  resize 直写 ION 可省 1 次，总数降到 2 次")
    print("  其余全部可零拷贝: 视图操作 + ION 原地操作")

    sys.exit(0 if ok1 and ok2 and ok3 else 1)
