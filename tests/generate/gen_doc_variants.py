#!/usr/bin/env python3
"""gen_doc_variants.py — 针对 pyCauchyKesai 两份 API 文档的可测细节,
生成一组变异的 OE 编译工作目录,并依次编译产出不同编译规范的 HBM。

设计依据(文档细节 → 变异规范):
  - 核数(core1/core2/core4): set_scheduling_params 核掩码严格性 / 构造默认掩码 / core 等价
  - 精度(int8/int16): 量化刻画、quantize 路径
  - nv12 输入(y+uv uint8): 文档「快速开始」m([y,uv])、input_source pyramid、uint8 dtype
  - rgb 输入(uint8): quantize 写侧量化、BPU dtype 对照
  - 多输入/多输出/mixed_io: inputs/outputs 元数、check_input/output 多 idx、绑定顺序
  - rank(1D/3D): shape/ndim 变体、IONArrayDesc.ndim
  - max_time_per_fc: 抢占优先级 254/255(运行时配 UNSET_HB_DNN_USER_DEFINED_L2M_SIZES)
  - model_output_type=int8: 尝试触发 SCALE 输出(quantiType=1),验证平台可达性
  - chain_a/chain_b: A 输出 shape == B 输入 shape,多模型零拷贝链式

纯云端 docker 内环境(torch/onnx/hb_compile)产出。每个 spec 独立 try/except,
失败不阻塞,可达性记入 manifest 的 compile_status。

用法(云端 docker 内):
  python3 /work/generate/gen_doc_variants.py --skip-compile   # 只生成 onnx/calib/yaml
  python3 /work/generate/gen_doc_variants.py                  # 生成 + 逐个 hb_compile
  python3 /work/generate/gen_doc_variants.py --only conv_core1
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_matrix as BM          # 复用 torch 模型构造 / 常量
from build_matrix import build_module, _names_for_arity

WORK = os.environ.get("OETC_WORK", "/work") + "/docvariants"
SEED = 20260630
N_CALIB = getattr(BM, "N_CALIB", 10)
N_GOLDEN = getattr(BM, "N_GOLDEN", 10)
OPSET = getattr(BM, "OPSET", 14)


# ============================================================================
# DOC_SPECS: 每条 = 一个变异编译工作目录
#   name     产物/cid 前缀
#   bb       backbone(build_module)
#   arity    1i1o / 2i1o / 1i2o / 2i2o
#   shape    单输入逻辑 shape(多输入时复制)
#   yaml     编译规范变异字段:
#              core(1/2/4), precision(int8/int16),
#              input=(train,rt,layout) 触发 rgb/nv12 前处理(mean/scale),
#              max_time_per_fc(int us), model_output_type(int8)
#   doc      覆盖的文档细节(写入 manifest 供测试侧引用)
# ============================================================================
DOC_SPECS = [
    # 核数变异 —— set_scheduling_params 严格性 / core 等价 / 构造默认掩码
    dict(name="conv_core1", bb="conv2d", arity="1i1o", shape=[1, 4, 16, 16],
         yaml=dict(core=1), doc="基线单核;覆盖构造/summary/__call__/task_slot/异步/IONArray 等通用细节"),
    dict(name="conv_core2", bb="conv2d", arity="1i1o", shape=[1, 4, 16, 16],
         yaml=dict(core=2), doc="多核:set_scheduling_params 核掩码恰好N核/构造默认[0,1]/core1==core2 等价"),
    dict(name="conv_core4", bb="conv2d", arity="1i1o", shape=[1, 4, 16, 16],
         yaml=dict(core=4), doc="核数=4 边界:set_scheduling_params 需恰好4核"),

    # 精度变异 —— 量化刻画
    dict(name="conv_int8_c1", bb="conv2d", arity="1i1o", shape=[1, 4, 16, 16],
         yaml=dict(core=1, precision="int8"), doc="int8 量化:量化刻画/精度对比"),

    # nv12 输入 —— 文档「快速开始」m([y,uv]) uint8 双输入
    dict(name="conv_nv12_c1", bb="conv2d", arity="1i1o", shape=[1, 3, 16, 16],
         yaml=dict(core=1, input=("rgb", "nv12", "NCHW")), doc="nv12 输入:input_type_rt=nv12,runtime 暴露 y+uv uint8 双输入"),

    # rgb 输入 —— uint8, quantize 写侧量化
    dict(name="conv_rgb_c1", bb="conv2d", arity="1i1o", shape=[1, 3, 16, 16],
         yaml=dict(core=1, input=("rgb", "rgb", "NCHW")), doc="rgb uint8 输入:quantize 写侧量化路径/BPU dtype 对照"),

    # IO 元数 —— inputs/outputs 多元数 / check 多 idx / 绑定顺序
    dict(name="concat_2i1o", bb="concat_split", arity="2i1o", shape=[1, 16],
         yaml=dict(core=1), doc="多输入:inputs[slot] 多 idx 绑定/check_input 多 idx/输入顺序"),
    dict(name="concat_1i2o", bb="concat_split", arity="1i2o", shape=[1, 16],
         yaml=dict(core=1), doc="多输出:outputs[slot] 多 idx/输出顺序/wait 返回多 ndarray"),
    dict(name="concat_2i2o", bb="concat_split", arity="2i2o", shape=[1, 16],
         yaml=dict(core=1), doc="mixed_io:多输入多输出交叉绑定/check_input+check_output"),

    # rank 变异 —— shape/ndim
    dict(name="matmul_1d", bb="matmul", arity="1i1o", shape=[16],
         yaml=dict(core=1), doc="1D 张量:IONArrayDesc.ndim/shape 变体"),
    dict(name="matmul_3d", bb="matmul", arity="1i1o", shape=[1, 16, 32],
         yaml=dict(core=1), doc="3D 张量:rank 变体"),

    # 抢占 —— 编译时 max_time_per_fc(运行时 priority 254/255 + UNSET L2 Cache)
    #   OE 约束:max_time_per_fc 只能为 0 或 [1000, 4294967295](us),设 1000
    dict(name="conv_preempt_c1", bb="conv2d", arity="1i1o", shape=[1, 4, 16, 16],
         yaml=dict(core=1, max_time_per_fc=1000), doc="抢占:编译时 max_time_per_fc,使 function-call 可被抢占"),

    # SCALE 输出 —— 尝试 model_output_type=int8 触发 quantiType=SCALE(平台可达性待验证)
    dict(name="matmul_scaleout_c1", bb="matmul", arity="1i1o", shape=[2, 8, 32],
         yaml=dict(core=1, model_output_type="int8"), doc="SCALE 输出尝试:model_output_type=int8,验证 S600 是否仍插 Dequantize"),

    # 多模型零拷贝链 —— A.out shape == B.in shape
    dict(name="chain_a", bb="matmul", arity="1i1o", shape=[1, 32],
         yaml=dict(core=1), doc="零拷贝链式上游:A.outputs[0] 喂 B.inputs[0](同物理内存)"),
    dict(name="chain_b", bb="matmul", arity="1i1o", shape=[1, 32],
         yaml=dict(core=1), doc="零拷贝链式下游:check_input(A.outputs[0]) / from_memory 共享"),
]


# ============================================================================
# 生成 onnx + calib + golden + meta(复用 build_module)
# ============================================================================
def gen_spec(spec):
    import numpy as np
    import torch
    name = spec["name"]
    bb, arity, shape = spec["bb"], spec["arity"], spec["shape"]
    in_names = _names_for_arity(arity)
    in_shapes = [tuple(shape)] * len(in_names)

    seed = SEED + sum(ord(c) for c in name)
    torch.manual_seed(seed)
    module, out_names = build_module(bb, arity, in_shapes)
    module.eval()

    # 推 output shape
    with torch.no_grad():
        dums = [torch.zeros(*s) for s in in_shapes]
        o = module(*dums)
    out_list = o if isinstance(o, (list, tuple)) else [o]
    out_shapes = [tuple(t.shape) for t in out_list]

    # calib 数据:统一 float32(OE 图像输入 rgb/nv12 的校准也用 float32 featuremap,
    # 见云端 rgb_identity_fixed 先例 —— input_type_train=rgb 但 calib float32;
    # runtime 才接 uint8 图像,故 summary 的 dtype 以板端为准)
    g = torch.Generator().manual_seed(seed + 7777)
    samples = []
    for _ in range(N_CALIB):
        samp = []
        for s in in_shapes:
            arr = (torch.randn(*s, generator=g) * 0.5).numpy().astype(np.float32)
            samp.append(arr)
        samples.append(samp)

    # golden: float 前向(图像用其 float 视图喂 onnx)
    golden_pairs = []
    with torch.no_grad():
        for i in range(N_GOLDEN):
            tins = [torch.from_numpy(samples[i][j].astype(np.float32)) for j in range(len(in_names))]
            o = module(*tins)
            ol = o if isinstance(o, (list, tuple)) else [o]
            golden_pairs.append((samples[i], [t.numpy().astype(np.float32) for t in ol]))

    # 导出 onnx(图输入恒 float32;rgb/nv12 前处理由 OE runtime 做)
    onnx_dir = f"{WORK}/onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = f"{onnx_dir}/{name}.onnx"
    with torch.no_grad():
        dums = [torch.zeros(*s) for s in in_shapes]
        torch.onnx.export(module, tuple(dums) if len(in_names) > 1 else dums[0],
                          onnx_path, opset_version=OPSET,
                          input_names=in_names, output_names=out_names)

    # calib 落盘(单输入平铺;多输入 per-input 子目录)
    calib_root = f"{WORK}/calib/{name}"
    if len(in_names) == 1:
        os.makedirs(calib_root, exist_ok=True)
        for i, samp in enumerate(samples):
            np.save(f"{calib_root}/sample_{i:02d}.npy", np.ascontiguousarray(samp[0]))
        calib_dirs = [calib_root]
    else:
        calib_dirs = []
        for j, nm in enumerate(in_names):
            d = f"{calib_root}/{nm}"
            os.makedirs(d, exist_ok=True)
            calib_dirs.append(d)
            for i, samp in enumerate(samples):
                np.save(f"{d}/sample_{i:02d}.npy", np.ascontiguousarray(samp[j]))

    # golden npz + meta
    golden_dir = f"{WORK}/golden/{name}"
    os.makedirs(golden_dir, exist_ok=True)
    pack = {}
    for i, (ins, outs) in enumerate(golden_pairs):
        for j, nm in enumerate(in_names):
            pack[f"in_{i:02d}_{nm}"] = np.ascontiguousarray(ins[j])
        for j, nm in enumerate(out_names):
            pack[f"out_{i:02d}_{nm}"] = np.ascontiguousarray(outs[j])
    np.savez(f"{golden_dir}/data.npz", **pack)

    in_dtype = "float32"  # calib/golden 用 float32;runtime dtype 以板端 summary 为准
    meta = dict(
        name=name, backbone=bb, arity=arity, yaml=spec["yaml"], doc=spec["doc"],
        inputs=[{"name": n, "shape": list(s), "dtype": in_dtype} for n, s in zip(in_names, in_shapes)],
        outputs=[{"name": n, "shape": list(s), "dtype": "float32"} for n, s in zip(out_names, out_shapes)],
        in_names=in_names, out_names=out_names,
        calib_dirs=calib_dirs,
        onnx=onnx_path, n_golden=N_GOLDEN, n_calib=N_CALIB,
    )
    json.dump(meta, open(f"{golden_dir}/meta.json", "w"), indent=2)
    return meta


# ============================================================================
# 变异 YAML 生成(支持 rgb/nv12 前处理 / core / precision / max_time_per_fc / model_output_type)
# ============================================================================
def write_yaml_variant(name, meta, yover):
    in_names = meta["in_names"]
    calib_dirs = meta["calib_dirs"]
    n = len(in_names)
    core = yover.get("core", 1)
    prec = yover.get("precision", "int16")
    inp = yover.get("input")        # (train, rt, layout)

    if inp:
        itt, irt, layout = inp
    else:
        itt = irt = "featuremap"
        layout = "NHWC"

    if n > 1:
        name_field = "; ".join(in_names)
        shape_field = "; ".join("x".join(str(x) for x in i["shape"]) for i in meta["inputs"])
        cal_field = "; ".join(calib_dirs)
        itt_field = "; ".join([itt] * n)
        irt_field = "; ".join([irt] * n)
        layout_field = "; ".join([layout] * n)
    else:
        name_field = in_names[0]
        shape_field = "x".join(str(x) for x in meta["inputs"][0]["shape"])
        cal_field = calib_dirs[0]
        itt_field = itt
        irt_field = irt
        layout_field = layout

    # rgb/nv12 前处理参数(2 空格,与 input_name 同级,同属 input_parameters)
    preprocess = ""
    if inp:
        preprocess += "  mean_value: 0 0 0\n"
        preprocess += "  scale_value: 0.00390625 0.00390625 0.00390625\n"
        preprocess += "  input_space_and_range: regular\n"

    # 量化配置
    quant = f"      all_node_type: {prec}"
    if "model_output_type" in yover:
        quant += f"\n      model_output_type: {yover['model_output_type']}"

    # 编译器参数
    compiler = (
        f"  core_num: {core}\n"
        f"  optimize_level: O2\n"
        f"  compile_mode: latency\n"
        f"  debug: false"
    )
    if "max_time_per_fc" in yover:
        compiler += f"\n  max_time_per_fc: {yover['max_time_per_fc']}"

    y = f"""# auto-generated by gen_doc_variants.py — {name} ({spec_doc(yover)})
model_parameters:
  march: nash-p
  onnx_model: {meta['onnx']}
  output_model_file_prefix: {name}
  working_dir: {WORK}/hbm/{name}
input_parameters:
  input_name: '{name_field}'
  input_type_train: {itt_field}
  input_type_rt: {irt_field}
  input_layout_train: {layout_field}
  input_shape: '{shape_field}'
{preprocess}calibration_parameters:
  cal_data_dir: '{cal_field}'
  quant_config:
    model_config:
{quant}
compiler_parameters:
{compiler}
"""
    os.makedirs(f"{WORK}/configs", exist_ok=True)
    p = f"{WORK}/configs/{name}.yaml"
    open(p, "w").write(y)
    return p


def spec_doc(yover):
    tags = []
    if "input" in yover:
        tags.append(yover["input"][1])
    tags.append(f"core{yover.get('core', 1)}")
    tags.append(yover.get("precision", "int16"))
    if "max_time_per_fc" in yover:
        tags.append("preempt")
    if "model_output_type" in yover:
        tags.append("scale_out")
    return "/".join(tags)


# ============================================================================
# 编译单个 spec(hb_compile)
# ============================================================================
def compile_spec(name):
    import shutil
    wd = f"{WORK}/hbm/{name}"
    os.makedirs(wd, exist_ok=True)
    yaml_path = f"{WORK}/configs/{name}.yaml"
    try:
        r = subprocess.run(["hb_compile", "-c", yaml_path],
                           capture_output=True, text=True, timeout=900)
        hbm = f"{wd}/{name}.hbm"
        if not os.path.exists(hbm):
            tail = (r.stderr + r.stdout)[-1200:]
            return False, tail.replace("\n", " | ")[:900]
        return True, None
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT 900s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ============================================================================
# main
# ============================================================================
def main():
    global subprocess
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-compile", action="store_true", help="只生成 onnx/calib/yaml,不编译")
    ap.add_argument("--only", default=None, help="只处理指定 name(逗号分隔)")
    ap.add_argument("--gen-only", action="store_true", help="别名:等同 --skip-compile")
    args = ap.parse_args()

    specs = DOC_SPECS
    if args.only:
        want = set(s.strip() for s in args.only.split(","))
        specs = [s for s in DOC_SPECS if s["name"] in want]

    os.makedirs(WORK, exist_ok=True)
    manifest = []
    errors = []

    for i, spec in enumerate(specs):
        name = spec["name"]
        entry = {"name": name, "doc": spec["doc"], "yaml": spec["yaml"],
                 "compile_status": "pending"}
        try:
            meta = gen_spec(spec)
            entry.update({k: meta[k] for k in
                          ("inputs", "outputs", "in_names", "out_names", "n_golden")})
            yaml_path = write_yaml_variant(name, meta, spec["yaml"])
            entry["yaml_path"] = yaml_path.replace(WORK, "/work")
            print(f"[{i+1}/{len(specs)}] {name}: GEN ok (onnx+calib+yaml)")
        except Exception as e:
            import traceback
            entry["compile_status"] = f"GEN-FAIL: {type(e).__name__}: {e}"
            errors.append({"name": name, "stage": "gen", "error": str(e)})
            print(f"[{i+1}/{len(specs)}] {name}: GEN-FAIL {type(e).__name__}: {e}")
            traceback.print_exc()
            manifest.append(entry)
            continue

        if args.skip_compile or args.gen_only:
            entry["compile_status"] = "gen-only"
            manifest.append(entry)
            continue

        ok, err = compile_spec(name)
        entry["compile_status"] = "ok" if ok else f"COMPILE-FAIL: {err}"
        if ok:
            entry["hbm"] = f"hbm/{name}/{name}.hbm"
            print(f"[{i+1}/{len(specs)}] {name}: COMPILE ok → {entry['hbm']}")
        else:
            errors.append({"name": name, "stage": "compile", "error": err})
            print(f"[{i+1}/{len(specs)}] {name}: COMPILE-FAIL {err[:200]}")
        manifest.append(entry)

    json.dump({"manifest": manifest, "errors": errors, "work": WORK.replace("/work", "")},
              open(f"{WORK}/manifest_doc.json", "w"), indent=2, ensure_ascii=False)
    nok = sum(1 for e in manifest if e["compile_status"] == "ok")
    print(f"\n=== done: {nok}/{len(specs)} compiled, {len(errors)} errors ===")
    print(f"manifest → {WORK}/manifest_doc.json")


if __name__ == "__main__":
    main()
