#!/usr/bin/env python3
"""M3: board-side closed-loop characterization.

For every manifest entry: load HBM via pyCauchyKesai, feed the golden inputs,
compare BPU output vs torch golden (compute_metrics), benchmark latency.
Emits report/{per_case.csv, summary.json, by_axis.csv}.

Semantics: CHARACTERIZATION ONLY (no pass/fail thresholds). Numerical results
are recorded for analysis; structural problems (shape mismatch, BPU busy) are
surfaced, never hidden.

Run: PYTHONPATH=<repo>/src python3 tests/golden/run_characterization.py
"""
import os, sys, csv, json, math, time

# locate repo src so `from pyCauchyKesai import CauchyKesai` resolves
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = f"{REPO}/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
TD = os.path.dirname(os.path.abspath(__file__))
if TD not in sys.path:
    sys.path.insert(0, TD)

import numpy as np
from pyCauchyKesai import CauchyKesai
import harness as H

REPORT_DIR = os.environ.get("GOLDEN_REPORT_DIR", f"{TD}/report")
os.makedirs(REPORT_DIR, exist_ok=True)


def order_inputs_for_model(model, meta, sample_inputs_in_meta_order):
    """Feed inputs in hbm tensor order. Assert shape match per index.

    hb_compile preserves onnx input order, so meta order == hbm order; we assert
    shapes agree and reorder only if names visibly mismatch the summary.
    """
    info = model.summary()
    sum_shapes = [tuple(i["shape"]) for i in info["inputs"]]
    meta_shapes = [tuple(i["shape"]) for i in meta["inputs"]]
    if sum_shapes == meta_shapes:
        return sample_inputs_in_meta_order
    # shape mismatch -> try name alignment via summary
    sum_names = [i["name"] for i in info["inputs"]]
    meta_names = [i["name"] for i in meta["inputs"]]
    if sum_names == meta_names and sum_shapes == meta_shapes:
        return sample_inputs_in_meta_order
    raise RuntimeError(
        f"input shape/name mismatch: summary={sum_shapes}/{sum_names} "
        f"meta={meta_shapes}/{meta_names}")


def run_one(entry, meta):
    cid = entry["cid"]
    hpath = H.hbm_path(cid)
    if not os.path.exists(hpath):
        return None, f"hbm missing: {hpath}"
    npz = H.load_golden_npz(entry["model_id"])
    in_names = H.golden_input_names(meta)
    out_names = H.golden_output_names(meta)
    n_golden = meta["n_golden"]

    model = CauchyKesai(hpath, n_task=1)
    # Multi-core models (S600 core_num=2) REQUIRE explicit core assignment at
    # SubmitTask time; CORE_ANY is rejected with "backend must be specified to
    # specific cores". Single-core models accept CORE_ANY (the default).
    core_num = entry.get("core_num", 1)
    if core_num > 1:
        model.set_scheduling_params(list(range(core_num)))
    rows = []
    per_sample_cos = []
    per_sample_maxabs = []
    for idx in range(n_golden):
        inputs = H.sample_inputs(npz, meta, idx)
        inputs = order_inputs_for_model(model, meta, inputs)
        outs = model(inputs)               # list[np.ndarray], float32
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        gold = H.sample_outputs(npz, meta, idx)
        per_out = [H.compute_metrics(o, g) for o, g in zip(outs, gold)]
        # structural check
        if any("error" in m for m in per_out):
            return None, f"sample {idx}: {per_out}"
        cos_vals = [m["cosine_similarity"] for m in per_out]
        maxabs_vals = [m["max_abs_error"] for m in per_out]
        mse_vals = [m["mse"] for m in per_out]
        agree_vals = [m["agreement_1e-3"] for m in per_out]
        cos_mean = float(np.mean(cos_vals))
        per_sample_cos.append(cos_mean)
        per_sample_maxabs.append(float(np.max(maxabs_vals)))
        row = {
            "cid": cid, "family": entry["family"], "variant": entry["variant"],
            "precision": entry["precision"], "core_num": entry["core_num"],
            "int16_node_ratio": entry.get("int16_node_ratio"),
            "sample": idx,
            "cos_mean": cos_mean,
            "max_abs_max": float(np.max(maxabs_vals)),
            "mse_mean": float(np.mean(mse_vals)),
            "agree_1e-3_mean": float(np.mean(agree_vals)),
            "gold_abs_max": float(np.max([m["gold_abs_max"] for m in per_out])),
        }
        for nm, m in zip(out_names, per_out):
            row[f"cos_{nm}"] = m["cosine_similarity"]
            row[f"maxabs_{nm}"] = m["max_abs_error"]
        rows.append(row)

    # latency (benchmark delegates to start+wait, v1.2 API)
    try:
        tinfo = model.benchmark(timeout_ms=60000)
        latency_ms = tinfo.get("time_ms")
    except Exception as e:
        latency_ms = None
    for r in rows:
        r["latency_ms"] = latency_ms

    summary = {
        "cid": cid, "family": entry["family"], "variant": entry["variant"],
        "precision": entry["precision"], "core_num": entry["core_num"],
        "int16_node_ratio": entry.get("int16_node_ratio"),
        "n_outputs": len(out_names),
        "cos_mean_over_samples": float(np.mean(per_sample_cos)),
        "cos_min_over_samples": float(np.min(per_sample_cos)),
        "max_abs_over_samples": float(np.max(per_sample_maxabs)),
        "latency_ms": latency_ms,
    }
    return {"rows": rows, "summary": summary}, None


def write_by_axis(all_rows):
    """int8-vs-int16 contrast per family (avg over variant/core/sample)."""
    from collections import defaultdict
    bucket = defaultdict(list)  # (family, precision) -> [cos_mean]
    mbucket = defaultdict(list)  # (family, precision) -> [max_abs_max]
    for r in all_rows:
        bucket[(r["family"], r["precision"])].append(r["cos_mean"])
        mbucket[(r["family"], r["precision"])].append(r["max_abs_max"])
    out = f"{REPORT_DIR}/by_axis.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "precision", "n", "cos_mean", "max_abs_mean"])
        for k in sorted(bucket):
            cos = bucket[k]
            w.writerow([k[0], k[1], len(cos), round(float(np.mean(cos)), 6),
                        round(float(np.mean(mbucket[k])), 6)])
    return out


def main():
    if not H.data_present():
        print(f"[ERR] GOLDEN_DATA_DIR={H.GOLDEN_DATA_DIR} has no manifest/hbm. "
              f"Sync from remote first (see README).")
        sys.exit(2)
    entries = H.manifest_entries()
    print(f"characterizing {len(entries)} models from {H.GOLDEN_DATA_DIR}")

    all_rows, all_summary = [], []
    failures = []
    # BPU liveness probe: load the first hbm; abort cleanly if device held.
    try:
        first = entries[0]
        _ = CauchyKesai(H.hbm_path(first["cid"]), n_task=1).summary()
    except Exception as e:
        print(f"[ERR] cannot load BPU model {first['cid']}: {e}\n"
              f"      Likely the BPU is held by another service (robotrea). "
              f"Stop it and retry. Not faking results.")
        sys.exit(3)

    for i, entry in enumerate(entries):
        meta = H.load_meta(entry["model_id"])
        t0 = time.time()
        try:
            res, err = run_one(entry, meta)
        except Exception as e:
            res, err = None, f"EXC: {type(e).__name__}: {e}"
        dt = time.time() - t0
        if err:
            failures.append((entry["cid"], err))
            print(f"  [{i+1:2d}/{len(entries)}] FAIL {entry['cid']:42s} {err}")
        else:
            all_rows.extend(res["rows"])
            all_summary.append(res["summary"])
            s = res["summary"]
            print(f"  [{i+1:2d}/{len(entries)}] ok   {entry['cid']:42s} "
                  f"cos={s['cos_mean_over_samples']:.5f} "
                  f"maxabs={s['max_abs_over_samples']:.4g} "
                  f"lat={s['latency_ms']}ms int16={s['int16_node_ratio']} ({dt:.1f}s)")

    # write reports
    if all_rows:
        cols = sorted({k for r in all_rows for k in r})
        with open(f"{REPORT_DIR}/per_case.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
    json.dump({"summaries": all_summary, "failures": failures},
              open(f"{REPORT_DIR}/summary.json", "w"), indent=2)
    ax = write_by_axis(all_rows)

    print(f"\n=== report ===")
    print(f"  per_case.csv : {len(all_rows)} sample rows")
    print(f"  summary.json : {len(all_summary)} models ok, {len(failures)} failures")
    print(f"  by_axis.csv  : {ax}")
    if failures:
        print(f"\n  FAILURES:")
        for cid, err in failures:
            print(f"    {cid}: {err}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
