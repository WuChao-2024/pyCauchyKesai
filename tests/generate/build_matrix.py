#!/usr/bin/env python3
"""pyCauchyKesai 一致性测试库 — 云端一键构建（gen + onnx + golden + calib + hb_compile + manifest）。

在云端 docker（torch/onnx/hb_compile 全在）里跑：
    docker run --gpus all --entrypoint "" -v ~/openexplore_test_cases:/work <img> python3 /work/build_all.py

  --dry-run : 只枚举交叉积 + 打印 shape + 计数，不 import torch、不编译（本地可跑）
  --limit N : 只处理前 N 个组合（调试用）
  --skip-compile : 只生成 onnx/golden/calib，不跑 hb_compile

数据（一份输入，两种落盘）：
  calib/<id>/sample_XX.npy                 单输入（平铺）
  calib/<id>/<input_name>/sample_XX.npy    多输入（每输入一目录）
  golden/<id>/data.npz                     前10组 in+out 对（前10与 calib 前10 byte-identical）
  onnx/<id>.onnx                           静态 shape, opset 14
  configs/<cid>.yaml / hbm/<cid>/...       编译产物
  manifest.json (成功) + compile_errors.json (失败, 不掩盖)

全 int16（all_node_type:int16），debug:false，march:nash-p。
"""
import os, sys, json, argparse, subprocess, traceback

WORK = os.environ.get("OETC_WORK", "/work")
SEED = 12345
N_GOLDEN = 10
N_CALIB = 50
OPSET = 14
PRECISION = "int16"

# ---------------- 轴 spec ----------------
BACKBONES = ['conv2d', 'conv3d', 'depthwise_conv2d', 'matmul', 'transformer',
             'elementwise', 'softmax', 'layernorm', 'reshape_transpose', 'concat_split']
APPLICABLE_NDIM = {
    'conv2d': ['4d'], 'conv3d': ['5d'], 'depthwise_conv2d': ['4d'],
    'matmul': ['2d', '3d'], 'transformer': ['3d'],
    'elementwise': ['2d', '3d', '4d'],
    'softmax': ['1d', '2d', '3d'], 'layernorm': ['2d', '3d'],
    'reshape_transpose': ['3d', '4d'],
    'concat_split': ['1d', '2d', '3d'],
}
SHAPE_VARIANTS = ['aligned', 'odd', 'bigc_smallspatial', 'smallc_bigspatial',
                  'tiny', 'large_batch', 'boundary_reduce', 'nonpow2_channel']
ARITY_FOR_BACKBONE = {
    'conv2d': ['1i1o'], 'conv3d': ['1i1o'], 'depthwise_conv2d': ['1i1o'],
    'matmul': ['1i1o'], 'transformer': ['1i1o'],
    'softmax': ['1i1o'], 'layernorm': ['1i1o'], 'reshape_transpose': ['1i1o'],
    'elementwise': ['2i1o', '3i1o'],
    'concat_split': ['2i1o', '3i1o', '1i2o', '2i2o'],
}
CORES = [1, 2]

# ---------------- shape gen ----------------
BASE = {'1d': dict(D=16), '2d': dict(B=4, C=32), '3d': dict(B=2, T=8, C=32),
        '4d': dict(N=1, C=4, H=16, W=16), '5d': dict(N=1, C=4, D=4, H=8, W=8)}
ORDER = {'1d': ['D'], '2d': ['B', 'C'], '3d': ['B', 'T', 'C'],
         '4d': ['N', 'C', 'H', 'W'], '5d': ['N', 'C', 'D', 'H', 'W']}
SPATIAL = {'H', 'W', 'D'}; SEQ = {'T', 'D'}; BATCH = {'B', 'N'}; CH = {'C'}


def make_shape(ndim, variant, backbone):
    """Return concrete input shape tuple, or None if variant N/A for this ndim."""
    d = dict(BASE[ndim]); roles = set(ORDER[ndim])
    if variant == 'aligned':
        pass
    elif variant == 'odd':
        t = roles & (SPATIAL | SEQ)
        if not t:
            return None
        for r in t:
            d[r] = 17
    elif variant == 'bigc_smallspatial':
        if not (roles & CH) or not (roles & (SPATIAL | SEQ)):
            return None
        d['C'] = 64
        for r in roles & (SPATIAL | SEQ):
            d[r] = 3
    elif variant == 'smallc_bigspatial':
        if not (roles & CH) or not (roles & (SPATIAL | SEQ)):
            return None
        d['C'] = 2
        for r in roles & SPATIAL:
            d[r] = 48
        for r in roles & SEQ:
            d[r] = 16
    elif variant == 'tiny':
        for r in roles:
            d[r] = 1 if r in BATCH else 2
    elif variant == 'large_batch':
        bt = roles & BATCH
        if not bt:
            return None
        for r in bt:
            d[r] = 32
    elif variant == 'boundary_reduce':
        if backbone not in ('softmax', 'layernorm'):
            return None
        last = ORDER[ndim][-1]
        d[last] = 8192
        for r in roles:
            if r == last:
                continue
            d[r] = 1 if r in BATCH else 2
    elif variant == 'nonpow2_channel':
        if not (roles & CH):
            return None
        d['C'] = 5
    return tuple(d[r] for r in ORDER[ndim])


def enumerate_combos():
    """Yield (backbone, ndim, variant, arity, in_shape, model_id)."""
    out = []
    for bb in BACKBONES:
        for ndim in APPLICABLE_NDIM[bb]:
            for arity in ARITY_FOR_BACKBONE[bb]:
                for v in SHAPE_VARIANTS:
                    sh = make_shape(ndim, v, bb)
                    if sh is None:
                        continue
                    mid = f"{bb}_{ndim}_{v}_{arity}"
                    out.append((bb, ndim, v, arity, sh, mid))
    return out


# ---------------- torch 模型构建（lazy import）----------------
def _pick_heads(C):
    for h in (8, 4, 2, 1):
        if C % h == 0:
            return h
    return 1


def build_module(backbone, arity, in_shapes):
    """in_shapes: list of tuples. Return (nn.Module, out_names)."""
    import torch
    import torch.nn as nn

    class Seq(nn.Module):
        def __init__(self, *layers):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    sh0 = in_shapes[0]

    if backbone == 'conv2d':
        N, Cin, H, W = sh0
        cout = max(4, min(Cin * 2, 16))
        return Seq(nn.Conv2d(Cin, cout, 3, padding=1), nn.ReLU(),
                   nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(),
                   nn.Conv2d(cout, cout, 3, padding=1)), ['output']

    if backbone == 'conv3d':
        N, Cin, D, H, W = sh0
        cout = max(4, min(Cin * 2, 8))
        return Seq(nn.Conv3d(Cin, cout, 3, padding=1), nn.ReLU(),
                   nn.Conv3d(cout, cout, 3, padding=1), nn.ReLU(),
                   nn.Conv3d(cout, cout, 3, padding=1)), ['output']

    if backbone == 'depthwise_conv2d':
        N, C, H, W = sh0
        return Seq(nn.Conv2d(C, C, 3, padding=1, groups=C), nn.ReLU(),
                   nn.Conv2d(C, C, 3, padding=1, groups=C), nn.ReLU(),
                   nn.Conv2d(C, C, 3, padding=1, groups=C)), ['output']

    if backbone == 'matmul':
        C = sh0[-1]
        return Seq(nn.Linear(C, C), nn.GELU(), nn.Linear(C, C), nn.GELU(),
                   nn.Linear(C, C)), ['output']

    if backbone == 'transformer':
        C = sh0[-1]
        h = _pick_heads(C)

        class Blk(nn.Module):
            def __init__(s):
                super().__init__()
                s.ln1 = nn.LayerNorm(C)
                s.qkv = nn.Linear(C, 3 * C)
                s.proj = nn.Linear(C, C)
                s.ln2 = nn.LayerNorm(C)
                s.fc1 = nn.Linear(C, C * 2)
                s.fc2 = nn.Linear(C * 2, C)
                s.h = h
                s.dk = C // h
                s.C = C

            def forward(s, x):
                B, T, _ = x.shape
                y = s.ln1(x)
                qkv = s.qkv(y).reshape(B, T, 3, s.h, s.dk)
                q, k, v = qkv.unbind(dim=2)
                q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) / (s.dk ** 0.5)
                att = att.softmax(dim=-1)
                o = (att @ v).transpose(1, 2).reshape(B, T, s.C)
                x = x + s.proj(o)
                y = s.ln2(x)
                return x + s.fc2(nn.functional.gelu(s.fc1(y)))

        class TF(nn.Module):
            def __init__(s):
                super().__init__()
                s.b1 = Blk(); s.b2 = Blk()

            def forward(s, x):
                return s.b2(s.b1(x))

        return TF(), ['output']

    if backbone == 'elementwise':
        n = int(arity[0])

        class EW(nn.Module):
            def __init__(s):
                super().__init__()
                C = sh0[-1]
                s.post = nn.Linear(C, C)

            def forward(s, *xs):
                out = xs[0]
                for t in xs[1:]:
                    out = out + t
                return s.post(out)

        return EW(), ['output']

    if backbone == 'softmax':
        return Seq(nn.Softmax(dim=-1)), ['output']

    if backbone == 'layernorm':
        C = sh0[-1]

        class LN(nn.Module):
            def __init__(s):
                super().__init__()
                s.ln = nn.LayerNorm(C)

            def forward(s, x):
                return s.ln(x)

        return LN(), ['output']

    if backbone == 'reshape_transpose':
        if len(sh0) == 3:
            B, T, C = sh0

            class RT3(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.fc = nn.Linear(T, T)

                def forward(s, x):
                    y = x.transpose(1, 2)  # [B,C,T]
                    y = s.fc(y)
                    return y.transpose(1, 2)  # back [B,T,C]

            return RT3(), ['output']
        else:
            N, C, H, W = sh0

            class RT4(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.fc = nn.Linear(C, C)

                def forward(s, x):
                    y = x.permute(0, 2, 3, 1)  # NHWC
                    y = s.fc(y)
                    return y.permute(0, 3, 1, 2)  # NCHW

            return RT4(), ['output']

    if backbone == 'concat_split':
        C = sh0[-1]
        n_in = int(arity[0]); n_out = int(arity.split('i')[1][0])
        if arity in ('2i1o', '3i1o'):
            class Cat(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.fc = nn.Linear(C * n_in, C)

                def forward(s, *xs):
                    import torch
                    y = torch.cat(xs, dim=-1)
                    return s.fc(y)

            return Cat(), ['output']
        if arity == '1i2o':
            class Split(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.fc = nn.Linear(C, C)
                    s.a = C // 2
                    s.b = C - s.a

                def forward(s, x):
                    import torch
                    y = s.fc(x)
                    return y[..., :s.a], y[..., s.a:]

            return Split(), ['output_a', 'output_b']
        if arity == '2i2o':
            class Br(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.fa = nn.Linear(C, C); s.fb = nn.Linear(C, C)

                def forward(s, a, b):
                    return s.fa(a), s.fb(b)

            return Br(), ['output_a', 'output_b']

    raise ValueError(f"unhandled {backbone}/{arity}")


# ---------------- 数据生成 ----------------
def _names_for_arity(arity):
    n_in = int(arity[0])
    if n_in == 1:
        return ['input']
    return [f'input_{c}' for c in 'abc'[:n_in]]


def gen_one(combo, args):
    """Build onnx + golden npz + calib npy for one (backbone,ndim,variant,arity)."""
    import numpy as np
    import torch
    bb, ndim, variant, arity, in_shape, mid = combo
    in_names = _names_for_arity(arity)
    in_shapes = [in_shape] * len(in_names)
    out_dir = f"{WORK}/onnx"; os.makedirs(out_dir, exist_ok=True)
    onnx_path = f"{out_dir}/{mid}.onnx"
    golden_dir = f"{WORK}/golden/{mid}"; os.makedirs(golden_dir, exist_ok=True)
    npz_path = f"{golden_dir}/data.npz"
    calib_root = f"{WORK}/calib/{mid}"

    resumable = (not args.no_resume) and os.path.exists(onnx_path) and os.path.exists(npz_path)
    if resumable:
        return {"model_id": mid, "skipped": True}

    # build module
    seed = SEED + sum(ord(c) for c in mid)
    torch.manual_seed(seed)
    module, out_names = build_module(bb, arity, in_shapes)
    module.eval()

    # derive output shapes via a dummy forward
    with torch.no_grad():
        dums = [torch.zeros(*s) for s in in_shapes]
        o = module(*dums)
    out_list = o if isinstance(o, (list, tuple)) else [o]
    out_shapes = [tuple(t.shape) for t in out_list]

    # generate inputs (deterministic per model)
    g = torch.Generator().manual_seed(seed + 7777)
    samples = []  # each: list[np.float32]
    for _ in range(N_CALIB):
        samp = []
        for s in in_shapes:
            arr = (torch.randn(*s, generator=g) * 0.5).numpy().astype(np.float32)
            samp.append(arr)
        samples.append(samp)

    # golden: forward first N_GOLDEN, capture outputs
    golden_pairs = []  # (inputs, outputs)
    with torch.no_grad():
        for i in range(N_GOLDEN):
            tins = [torch.from_numpy(samples[i][j]) for j in range(len(in_names))]
            o = module(*tins)
            ol = o if isinstance(o, (list, tuple)) else [o]
            golden_pairs.append((samples[i], [t.numpy().astype(np.float32) for t in ol]))

    # save calib npy (single: flat; multi: per-input subdir)
    if len(in_names) == 1:
        os.makedirs(calib_root, exist_ok=True)
        for i, samp in enumerate(samples):
            np.save(f"{calib_root}/sample_{i:02d}.npy", np.ascontiguousarray(samp[0]))
        calib_dirs = [calib_root]
    else:
        calib_dirs = []
        for j, nm in enumerate(in_names):
            d = f"{calib_root}/{nm}"; os.makedirs(d, exist_ok=True); calib_dirs.append(d)
            for i, samp in enumerate(samples):
                np.save(f"{d}/sample_{i:02d}.npy", np.ascontiguousarray(samp[j]))

    # save golden npz (first 10 in+out). 前N_GOLDEN与calib前N_GOLDEN byte-identical.
    pack = {}
    for i, (ins, outs) in enumerate(golden_pairs):
        for j, nm in enumerate(in_names):
            pack[f"in_{i:02d}_{nm}"] = np.ascontiguousarray(ins[j])
        for j, nm in enumerate(out_names):
            pack[f"out_{i:02d}_{nm}"] = np.ascontiguousarray(outs[j])
    np.savez(npz_path, **pack)

    # export onnx
    with torch.no_grad():
        dums = [torch.zeros(*s) for s in in_shapes]
        torch.onnx.export(module, tuple(dums) if len(in_names) > 1 else dums[0],
                          onnx_path, opset_version=OPSET,
                          input_names=in_names, output_names=out_names)
    import onnx
    onnx.checker.check_model(onnx.onnx_pb if False else onnx.load(onnx_path))

    meta = {
        "model_id": mid, "backbone": bb, "ndim": ndim, "shape_variant": variant,
        "arity": arity, "inputs": [{"name": n, "shape": list(s), "dtype": "float32"}
                                   for n, s in zip(in_names, in_shapes)],
        "outputs": [{"name": n, "shape": list(s), "dtype": "float32"}
                    for n, s in zip(out_names, out_shapes)],
        "n_golden": N_GOLDEN, "n_calib": N_CALIB, "onnx_opset": OPSET,
        "weight_seed": seed, "input_seed": seed + 7777,
        "forced_fallback_predict": (bb == 'conv3d'),
        "coverage_tags": [bb, ndim, arity, variant] + (['multi_input'] if len(in_names) > 1 else []),
        "calib_dirs": [d.replace(WORK, '/work') for d in calib_dirs],
        "in_names": in_names, "out_names": out_names,
    }
    json.dump(meta, open(f"{golden_dir}/meta.json", "w"), indent=2)
    n_params = sum(p.numel() for p in module.parameters())
    meta["n_params"] = int(n_params)
    json.dump(meta, open(f"{golden_dir}/meta.json", "w"), indent=2)
    return meta


# ---------------- 编译 ----------------
def _find_artifact(cid, suffix):
    """Find <cid>{suffix} anywhere under hbm/<cid>/ (hb_compile may nest in subdir)."""
    root = f"{WORK}/hbm/{cid}"
    if not os.path.isdir(root):
        return None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == f"{cid}{suffix}":
                return os.path.join(dirpath, f)
    # loose match
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(suffix) and cid in f:
                return os.path.join(dirpath, f)
    return None


def parse_quant(cid):
    """int16_node_ratio / fallbacks / bpu_nodes from _advice.csv.
    dtype 字段是 si16/si8/f32（不是 int16）。计算节点 = BPU 节点排除边界 quantize。
    int16_ratio = InputDataType==si16 的计算节点占比。"""
    import csv
    advice_csv = _find_artifact(cid, "_advice.csv")
    if not advice_csv:
        return {"int16_node_ratio": None, "fallbacks": None, "bpu_nodes": None}
    try:
        rows = list(csv.DictReader(open(advice_csv)))
    except Exception:
        return {"int16_node_ratio": None, "fallbacks": None, "bpu_nodes": None}
    bpu = [r for r in rows if str(r.get('Device', '')).lower().startswith('bpu')]
    compute = [r for r in bpu if 'quantize' not in str(r.get('OpType', '')).lower()]
    if not compute:
        return {"int16_node_ratio": None, "fallbacks": 0, "bpu_nodes": len(bpu)}
    si16 = [r for r in compute if 'si16' in str(r.get('InputDataType', '')).lower()]
    fb = [r for r in compute if 'si16' not in str(r.get('InputDataType', '')).lower()]
    return {"int16_node_ratio": round(len(si16) / len(compute), 4),
            "fallbacks": len(fb), "bpu_nodes": len(compute)}


def write_yaml(cid, meta, core):
    in_names = meta["in_names"]; calib_dirs = meta["calib_dirs"]
    n = len(in_names)
    # 多输入时所有 per-input 字段都要 N 个值('; '分隔)；单输入用单值
    if n > 1:
        name_field = "; ".join(in_names)
        shape_field = "; ".join("x".join(str(x) for x in i["shape"]) for i in meta["inputs"])
        cal_field = "; ".join(calib_dirs)
        type_train = "; ".join(["featuremap"] * n)
        type_rt = "; ".join(["featuremap"] * n)
        layout = "; ".join(["NHWC"] * n)
    else:
        name_field = in_names[0]
        shape_field = "x".join(str(x) for x in meta["inputs"][0]["shape"])
        cal_field = calib_dirs[0]
        type_train = "featuremap"
        type_rt = "featuremap"
        layout = "NHWC"
    y = f"""# auto-generated
model_parameters:
  march: nash-p
  onnx_model: {WORK}/onnx/{meta['model_id']}.onnx
  output_model_file_prefix: {cid}
  working_dir: {WORK}/hbm/{cid}
input_parameters:
  input_name: '{name_field}'
  input_type_train: {type_train}
  input_type_rt: {type_rt}
  input_layout_train: {layout}
  input_shape: '{shape_field}'
calibration_parameters:
  cal_data_dir: '{cal_field}'
  quant_config:
    model_config:
      all_node_type: {PRECISION}
compiler_parameters:
  core_num: {core}
  optimize_level: O2
  compile_mode: latency
  debug: false
"""
    os.makedirs(f"{WORK}/configs", exist_ok=True)
    p = f"{WORK}/configs/{cid}.yaml"
    open(p, "w").write(y)
    return p


def compile_one(cid, meta, core):
    """Run hb_compile, return (ok, info_or_error)."""
    import shutil
    yaml_path = write_yaml(cid, meta, core)
    wd = f"{WORK}/hbm/{cid}"
    os.makedirs(wd, exist_ok=True)
    try:
        r = subprocess.run(["hb_compile", "-c", yaml_path],
                           capture_output=True, text=True, timeout=900)
        hbm = f"{wd}/{cid}.hbm"
        if not os.path.exists(hbm):
            # 兜底：hb_compile 可能把产物放到 working_dir/<prefix>/ 下
            alt = f"{wd}/{cid}/{cid}.hbm"
            if os.path.exists(alt):
                hbm = alt
            else:
                tail = (r.stderr + r.stdout)[-1500:]
                return False, {"cid": cid, "compile_error": tail[-1200:]}
        q = parse_quant(cid)
        return True, q
    except subprocess.TimeoutExpired:
        return False, {"cid": cid, "compile_error": "TIMEOUT 900s"}
    except Exception as e:
        return False, {"cid": cid, "compile_error": f"{type(e).__name__}: {e}"}


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    combos = enumerate_combos()
    if args.limit:
        combos = combos[:args.limit]

    if args.dry_run:
        print(f"{'backbone':18} {'ndim':4} {'variant':18} {'arity':6} {'in_shape':22} model_id")
        print("=" * 110)
        for bb, ndim, v, ar, sh, mid in combos:
            print(f"{bb:18} {ndim:4} {v:18} {ar:6} {str(list(sh)):22} {mid}")
        print(f"\n=== {len(combos)} ONNX specs × {len(CORES)} cores = {len(combos)*len(CORES)} HBM targets ===")
        return

    os.makedirs(WORK, exist_ok=True)
    # pass 1: gen onnx + golden + calib
    metas = {}
    for i, combo in enumerate(combos):
        bb, ndim, v, ar, sh, mid = combo
        try:
            m = gen_one(combo, args)
            if m.get("skipped"):
                print(f"[{i+1}/{len(combos)}] {mid} (cached)")
            else:
                print(f"[{i+1}/{len(combos)}] {mid} onnx+golden+calib OK (params={m.get('n_params')})")
            # reload meta (cached case)
            meta_path = f"{WORK}/golden/{mid}/meta.json"
            if os.path.exists(meta_path):
                metas[mid] = json.load(open(meta_path))
        except Exception as e:
            print(f"[{i+1}/{len(combos)}] {mid} GEN-FAIL {type(e).__name__}: {e}")
            traceback.print_exc()

    if args.skip_compile:
        print(f"\n--skip-compile: generated {len(metas)} specs. done.")
        return

    # pass 2: compile × core
    manifest, errors = [], []
    targets = [(mid, core) for mid in metas for core in CORES]
    for i, (mid, core) in enumerate(targets):
        cid = f"{mid}__{PRECISION}__core{core}"
        meta = metas[mid]
        # resumable
        if (not args.no_resume) and os.path.exists(f"{WORK}/hbm/{cid}/{cid}.hbm"):
            q = parse_quant(cid)
            entry = _entry(cid, meta, core, q)
            manifest.append(entry)
            print(f"[{i+1}/{len(targets)}] {cid} (cached)")
            continue
        ok, info = compile_one(cid, meta, core)
        if ok:
            entry = _entry(cid, meta, core, info)
            manifest.append(entry)
            print(f"[{i+1}/{len(targets)}] {cid} OK i16={info['int16_node_ratio']} fb={info['fallbacks']}")
        else:
            errors.append(info)
            print(f"[{i+1}/{len(targets)}] {cid} FAIL: {str(info.get('compile_error',''))[:80]}")

    json.dump(manifest, open(f"{WORK}/manifest.json", "w"), indent=2)
    json.dump(errors, open(f"{WORK}/compile_errors.json", "w"), indent=2)
    print(f"\n=== manifest {len(manifest)} ok / {len(errors)} failed ===")


def _entry(cid, meta, core, q):
    return {
        "cid": cid, "model_id": meta["model_id"], "backbone": meta["backbone"],
        "ndim": meta["ndim"], "shape_variant": meta["shape_variant"], "arity": meta["arity"],
        "core_num": core, "precision": PRECISION,
        "inputs": meta["inputs"], "outputs": meta["outputs"],
        "int16_node_ratio": q.get("int16_node_ratio"), "fallbacks": q.get("fallbacks"),
        "bpu_nodes": q.get("bpu_nodes"), "forced_fallback": meta.get("forced_fallback_predict", False),
        "coverage_tags": meta.get("coverage_tags", []), "n_params": meta.get("n_params"),
        "hbm_path": f"hbm/{cid}/{cid}.hbm".replace("/work/", ""),
        "golden_meta": f"golden/{meta['model_id']}/meta.json".replace("/work/", ""),
        "n_golden": N_GOLDEN,
    }


if __name__ == "__main__":
    main()
