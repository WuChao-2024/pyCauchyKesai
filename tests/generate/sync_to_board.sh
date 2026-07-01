#!/usr/bin/env bash
# 闭环 step 2（板端执行）：把云端产物拉回板端 golden_hbm_matrix/。
# 只拉测试需要的：manifest.json + 扁平化 .hbm + golden/{data.npz,meta.json}
# （丢掉 onnx/configs/calib/log 等大文件）。
#
# 用法: ./sync_to_board.sh
# 环境变量: REMOTE / RDIR / BOARD_DATA_DIR（均可覆盖默认）
set -euo pipefail

REMOTE="${REMOTE:-chao.wu@120.48.157.2}"
RDIR="${RDIR:-openexplore_test_cases}"
DST="${BOARD_DATA_DIR:-/root/ssd/OELLM_Runtime/golden_hbm_matrix}"
mkdir -p "$DST/hbm" "$DST/golden"

echo "[1/3] manifest.json"
scp "$REMOTE:~/$RDIR/manifest.json" "$DST/manifest.json"

echo "[2/3] hbm（只 .hbm，扁平化 hbm/<cid>/<cid>.hbm → hbm/<cid>.hbm）"
rsync -aqzm --include='*/' --include='*.hbm' --exclude='*' \
  "$REMOTE:~/$RDIR/hbm/" "$DST/hbm/"
( cd "$DST/hbm" && \
  find . -mindepth 2 -name '*.hbm' -exec mv {} . \; 2>/dev/null; \
  find . -mindepth 1 -type d -empty -delete 2>/dev/null || true )

echo "[3/3] golden（只 data.npz + meta.json）"
rsync -aqzm --include='*/' --include='data.npz' --include='meta.json' --exclude='*' \
  "$REMOTE:~/$RDIR/golden/" "$DST/golden/"

echo ""
echo "✓ 同步完成: $(ls "$DST"/hbm/*.hbm 2>/dev/null | wc -l) hbm," \
     "$(find "$DST"/golden -name meta.json | wc -l) golden dirs → $DST"
echo "  下一步（跑接口一致性测试）:"
echo "    cd /root/ssd/OELLM_Runtime/CauchyKesai_ProMax/pyCauchyKesai"
echo "    conda activate robotrea_python_runtime"
echo "    GOLDEN_DATA_DIR=$DST PYTHONPATH=src pytest \\"
echo "      tests/dimensions/d1_ops_structure tests/dimensions/d2_tensor_shape -v"
