#!/bin/bash
# ============================================================================
# build_wheels.sh — 构建 pyCauchyKesai wheel
#
# 唯一安装方式: 在目标 Python 环境中构建 whl → pip install
#
# BPU 对齐值（32/64）由运行时 hbUCPGetSocName 检测，S100/S100P/S600 三板用同一条命令，
# 无需任何平台参数。
#
# 用法:
#   ./scripts/build_wheels.sh
#
# 移植性: wheel 能跑在哪些板上取决于构建机的 glibc/工具链——
#   S100/S100P 构建 → 三板通用；S600 构建 → 仅 S600。详见 docs/cn_install.md。
# ============================================================================
set -euo pipefail

DIST_DIR="dist"

echo "============================================"
echo "  Building pyCauchyKesai wheel"
echo "  Python:   $(python3 -c 'import sys; print(sys.version)')"
echo "============================================"

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

python3 -m pip wheel . \
    --wheel-dir="$DIST_DIR" \
    --no-deps \
    --verbose

echo ""
echo "============================================"
echo "  Build complete. Wheel:"
ls -lh "$DIST_DIR"/*.whl 2>/dev/null || { echo "  ERROR: no wheel found!"; exit 1; }
echo ""
echo "  Install with:"
echo "    pip install $DIST_DIR/pycauchykesai-*.whl"
echo "============================================"
