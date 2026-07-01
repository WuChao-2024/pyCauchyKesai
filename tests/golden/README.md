# Golden HBM 闭环刻画测试库

参数量极小、计算量受控的 BPU 测试模型矩阵：torch 浮点真值 → ONNX → `hb_compile` int16 → HBM → 板端 pyCauchyKesai 推理 → 与 golden 对比出量化误差报告。

**语义：纯刻画（characterization），不做 pass/fail 数值阈值。** 数值结果落 `report/`，结构性问题（shape 不匹配、BPU 被占）如实暴露。

## 架构

| 环节 | 在哪 | 产物 |
|---|---|---|
| torch 建模 + golden + onnx | 远程 x86 docker | `golden/<id>/{meta.json,data.npz}`, `onnx/<id>.onnx`, `calib/<id>/*.npy` |
| `hb_compile` int16/int8 矩阵 | 远程 x86 docker (`--gpus all` 必需) | `hbm/<cid>/<cid>.hbm` + `_advice.csv` + `manifest.json` |
| 板端闭环刻画 | 本机 arm64 (pyCauchyKesai) | `report/{per_case.csv,summary.json,by_axis.csv}` |

矩阵：4 族（conv_stack / matmul_stack / transformer_block / mixed_io）× 3 shape × {int8,int16} × {core1,core2} = **48 个 HBM**。

## 数据布局（板端）

`GOLDEN_DATA_DIR`（默认 `/root/ssd/OELLM_Runtime/golden_hbm`）：

```
manifest.json
hbm/<cid>.hbm                 # 扁平化
golden/<model_id>/data.npz    # numpy-only（板端无 h5py）
golden/<model_id>/meta.json
```

## 从远程同步（板端执行）

```bash
REMOTE=chao.wu@120.48.157.2
RDIR=~/openexplore_test_cases
GDD=/root/ssd/OELLM_Runtime/golden_hbm
mkdir -p $GDD/hbm $GDD/golden
# manifest
scp $REMOTE:$RDIR/manifest.json $GDD/
# 扁平化 hbm（只要 .hbm，丢掉 onnx/bc 等大文件）
rsync -avzm --include='*/' --include='*.hbm' --exclude='*' \
  $REMOTE:$RDIR/hbm/ $GDD/hbm/
# golden npz + meta（只要这两个）
rsync -avzm --include='*/' --include='data.npz' --include='meta.json' --exclude='*' \
  $REMOTE:$RDIR/golden/ $GDD/golden/
# 拍平 hbm/<cid>/<cid>.hbm -> hbm/<cid>.hbm
( cd $GDD/hbm && find . -mindepth 2 -name '*.hbm' -exec mv {} . \; && find . -mindepth 1 -type d -empty -delete )
```

## 运行刻画

```bash
# 闭环刻画报告（BPU 需空闲；被 robotrea 占用会明确报错退出，不假装通过）
PYTHONPATH=<repo>/src python3 tests/golden/run_characterization.py

# 看报告
cat tests/golden/report/by_axis.csv     # int8 vs int16 各族精度对比
cat tests/golden/report/summary.json    # 每个模型的 cos/maxabs/latency/int16_ratio
```

## pytest 烟雾测试（轻壳）

```bash
PYTHONPATH=src pytest tests/golden/ -m golden -q
# 无数据 → 全 skip；有数据 → 每个 HBM 加载+跑 sample0+断言输出有限（无数值阈值）
```

## 构建侧（远程）命令备忘

```bash
ssh $REMOTE 'docker run --rm --gpus all -v $HOME/openexplore_test_cases:/work \
  <image> /work/tools/run_py.sh /work/tools/gen_golden.py'        # M1
ssh $REMOTE 'docker run --rm -v $HOME/openexplore_test_cases:/work \
  <image> /work/tools/run_py.sh /work/tools/gen_configs.py'       # M2 生成 yaml
ssh $REMOTE 'docker run --rm --gpus all -v $HOME/openexplore_test_cases:/work \
  <image> /work/tools/run_py.sh /work/tools/compile.py --all'     # M2 编译（--gpus all 必需！）
ssh $REMOTE 'docker run --rm -v $HOME/openexplore_test_cases:/work \
  <image> /work/tools/run_py.sh /work/tools/collect_manifest.py'  # 生成 manifest
ssh $REMOTE 'docker run --rm -v $HOME/openexplore_test_cases:/work \
  <image> /work/tools/run_py.sh /work/tools/convert_npy.py'       # HDF5->npz（板端读）
```

镜像：`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`
关键坑：镜像 `ENTRYPOINT=[/bin/bash]`，所以跑 python 要用 `run_py.sh` 包装；真实校准**必须** `--gpus all` 否则 calibration 阶段 SIGSEGV。
