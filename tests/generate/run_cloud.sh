#!/usr/bin/env bash
# 闭环 step 1（板端执行）：把 tests/generate/ 同步到云端，在 docker 里跑 build_matrix.py
# 生成 torch→onnx→golden→calib→hbm→manifest 全套产物（留在云端 $RDIR 下）。
#
# 用法:
#   ./run_cloud.sh                      # 全量生成(可断点续跑)
#   ./run_cloud.sh --dry-run            # 只在云端枚举矩阵+计数, 不编译(快, 不需 GPU)
#   ./run_cloud.sh --skip-compile       # 只生成 onnx/golden/calib, 不跑 hb_compile
#   ./run_cloud.sh --limit 4            # 只处理前 4 个组合(调试)
#   ./run_cloud.sh --no-resume          # 强制重新生成(忽略缓存)
#
# 环境变量(可覆盖默认):
#   REMOTE=chao.wu@120.48.157.2  RDIR=openexplore_test_cases
#   IMG=registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
set -euo pipefail

REMOTE="${REMOTE:-chao.wu@120.48.157.2}"
RDIR="${RDIR:-openexplore_test_cases}"
IMG="${IMG:-registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0}"
GEN_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/2] 同步生成代码 → $REMOTE:~/$RDIR/generate/"
rsync -az --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'report' \
  "$GEN_DIR"/ "$REMOTE:~/$RDIR/generate/"

echo "[2/2] 云端 docker 跑 build_matrix.py（--gpus all 必需：校准阶段用 GPU，否则 SIGSEGV）"
ssh "$REMOTE" "docker run --rm --gpus all --entrypoint '' \
  -v ~/$RDIR:/work \
  -e OETC_WORK=/work \
  $IMG python3 /work/generate/build_matrix.py $*"

echo ""
echo "✓ 生成完成，产物在云端 ~/$RDIR/{onnx,golden,calib,configs,hbm,manifest.json}。"
echo "  下一步：./sync_to_board.sh   （把产物拉回板端）"
