"""M4: light smoke gate per golden model — loads, runs golden sample 0,
asserts output count + finiteness. NO numerical thresholds (characterization
lives in run_characterization.py). Marked `golden`; skips when no data synced.
"""
import os, sys
import numpy as np
import pytest

TD = os.path.dirname(os.path.abspath(__file__))
if TD not in sys.path:
    sys.path.insert(0, TD)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(TD)))
SRC = f"{REPO}/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pyCauchyKesai import CauchyKesai
import harness as H


def _entries():
    return H.manifest_entries() if H.data_present() else []


@pytest.mark.golden
@pytest.mark.parametrize("entry", _entries(), ids=[e["cid"] for e in _entries()] or ["no-data"])
def test_model_runs_finite(entry, golden_data_ok):
    cid = entry["cid"]
    hpath = H.hbm_path(cid)
    assert os.path.exists(hpath), f"hbm missing: {hpath}"

    meta = H.load_meta(entry["model_id"])
    npz = H.load_golden_npz(entry["model_id"])
    n_out = len(meta["outputs"])

    model = CauchyKesai(hpath, n_task=1)
    # Multi-core (S600 core_num=2) models require explicit core assignment at
    # SubmitTask; CORE_ANY is rejected. See run_characterization.py for details.
    if entry.get("core_num", 1) > 1:
        model.set_scheduling_params(list(range(entry["core_num"])))
    inputs = H.sample_inputs(npz, meta, 0)
    outs = model(inputs)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]

    # structural gate only: right count + finite. No accuracy threshold.
    assert len(outs) == n_out, f"{cid}: got {len(outs)} outputs, expected {n_out}"
    for o in outs:
        arr = np.asarray(o)
        assert np.all(np.isfinite(arr)), f"{cid}: non-finite output"
        # shape sanity vs meta (per index)
    print(f"\n  [{cid}] ok: {n_out} outputs finite")
