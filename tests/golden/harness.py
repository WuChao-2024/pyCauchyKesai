"""Board-side golden verify harness: metrics + discovery + loaders.

Deps: numpy + pyCauchyKesai (the conda env robotrea_python_runtime). No h5py —
golden data is read from data.npz produced by remote convert_npy.py.

Layout under GOLDEN_DATA_DIR (default /root/ssd/OELLM_Runtime/golden_hbm):
  manifest.json
  hbm/<cid>.hbm                  (flattened from remote hbm/<cid>/<cid>.hbm)
  golden/<model_id>/data.npz
  golden/<model_id>/meta.json
"""
import os, json, math
import numpy as np

GOLDEN_DATA_DIR = os.environ.get(
    "GOLDEN_DATA_DIR", "/root/ssd/OELLM_Runtime/golden_hbm")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def manifest_entries():
    """Return list of manifest entries, or [] if data dir absent."""
    mp = f"{GOLDEN_DATA_DIR}/manifest.json"
    if not os.path.exists(mp):
        return []
    return json.load(open(mp))


def data_present():
    """True iff GOLDEN_DATA_DIR has a manifest + at least one hbm."""
    if not os.path.isdir(GOLDEN_DATA_DIR):
        return False
    if not os.path.exists(f"{GOLDEN_DATA_DIR}/manifest.json"):
        return False
    return len(manifest_entries()) > 0


def hbm_path(cid):
    return f"{GOLDEN_DATA_DIR}/hbm/{cid}.hbm"


def golden_npz_path(model_id):
    return f"{GOLDEN_DATA_DIR}/golden/{model_id}/data.npz"


# ---------------------------------------------------------------------------
# metrics (lifted & generalized from robotrea_python_runtime/compare.py)
# ---------------------------------------------------------------------------
def compute_metrics(bpu, gold):
    """Element-wise metrics between two equal-shaped arrays."""
    bpu = np.asarray(bpu).astype(np.float64).ravel()
    gold = np.asarray(gold).astype(np.float64).ravel()
    res = {"shape_bpu": int(bpu.size), "shape_gold": int(gold.size)}
    if bpu.shape != gold.shape:
        res["error"] = "shape_mismatch"
        return res
    diff = bpu - gold
    absd = np.abs(diff)
    scale = np.maximum(np.abs(gold), 1e-8)
    mse = float(np.mean(diff ** 2))
    cos = float(np.dot(bpu, gold) / (np.linalg.norm(bpu) * np.linalg.norm(gold) + 1e-12))
    res.update({
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(absd)),
        "max_abs_error": float(np.max(absd)),
        "rel_error": float(np.mean(absd / scale)),
        "cosine_similarity": cos,
        "agreement_1e-2": float(np.mean(absd < 1e-2)),
        "agreement_1e-3": float(np.mean(absd < 1e-3)),
        "agreement_1e-4": float(np.mean(absd < 1e-4)),
        "agreement_1e-5": float(np.mean(absd < 1e-5)),
        "gold_abs_max": float(np.max(np.abs(gold))),
    })
    return res


# ---------------------------------------------------------------------------
# golden loaders
# ---------------------------------------------------------------------------
def load_golden_npz(model_id):
    path = golden_npz_path(model_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no golden npz for {model_id}: {path}")
    return np.load(path)


def golden_input_names(meta):
    return [i["name"] for i in meta["inputs"]]


def golden_output_names(meta):
    return [o["name"] for o in meta["outputs"]]


def sample_inputs(npz, meta, idx):
    """Ordered list of np.float32 input arrays for golden sample idx."""
    out = []
    for nm in golden_input_names(meta):
        key = f"in_{idx:02d}_{nm}"
        out.append(np.ascontiguousarray(npz[key].astype(np.float32)))
    return out


def sample_outputs(npz, meta, idx):
    """Ordered list of np.float64 golden output arrays for sample idx."""
    return [np.asarray(npz[f"out_{idx:02d}_{nm}"]).astype(np.float64)
            for nm in golden_output_names(meta)]


def load_meta(model_id):
    return json.load(open(f"{GOLDEN_DATA_DIR}/golden/{model_id}/meta.json"))
