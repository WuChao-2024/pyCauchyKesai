"""IONArray 大内存分配和带宽测试。
使用方法: python test.py [GB]
  python test.py       # 默认 1 GB
  python test.py 8     # 分配 8 GB
"""
from pyCauchyKesai import IONArray
import numpy as np
import sys
import time


def main():
    # 从命令行参数读取目标内存大小，默认 1 GB
    target_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    bytes_per_elem = 4                 # float32
    target_bytes = int(target_gb * 1024 ** 3)
    n_elements = target_bytes // bytes_per_elem
    actual_gb = n_elements * bytes_per_elem / 1024 ** 3

    print(f"目标: {target_gb:.1f} GB 连续ION内存")
    print(f"元素数: {n_elements:,} 个 float32  ({actual_gb:.2f} GB)")

    print("正在分配 IONArray...", flush=True)
    t0 = time.time()
    arr = IONArray(np.dtype('float32'), [n_elements])
    t1 = time.time()

    print(f"分配完成, 耗时: {t1 - t0:.3f}s")
    print(f"phys_addr: {hex(arr.phy_addr)}")
    print(f"mem_size:  {arr.mem_size:,} bytes  ({arr.mem_size / 1024**3:.2f} GB)")

    # 初始化
    val = np.float32(2.71828)
    print(f"初始化: arr[:] = {val}", flush=True)
    arr.numpy()[:] = val

    # 原地逐元素乘法
    factor = np.float32(3.14159)
    print(f"原地乘法: arr *= {factor}", flush=True)
    t2 = time.time()
    arr.numpy() *= factor
    t3 = time.time()
    dt = t3 - t2

    print(f"乘法完成, 耗时: {dt:.4f}s")
    if dt > 0:
        bw = n_elements * bytes_per_elem * 2 / dt / 1024**3
        print(f"内存带宽: {bw:.2f} GB/s  (读 + 写)")

    # 验证
    expected = val * factor
    print(f"\n验证: arr[0]  = {arr.numpy()[0]}  (期望 {expected})")
    print(f"       arr[1]  = {arr.numpy()[1]}")
    print(f"       arr[-1] = {arr.numpy()[-1]}")
    print("完成!")


if __name__ == "__main__":
    main()
