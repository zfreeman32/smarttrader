"""Retrain R0 follow-up: does the frozen dataset config, built on the FULL source history (file start .. end bar),
reproduce the frozen source_history.parquet? Compares the last ``tail`` rows before each end bar for every feature
selected by roster models built from this config (C1 tolerance, NaN == NaN). Equal -> 8,000-bar mismatches are
genuine long memory; unequal -> the frozen tape came from a different build path (reference drift).
Usage: python scripts/audit_frozen_full_history.py <dataset_dir> <ES|EURUSD> <end_iso> [<end_iso> ...] [start_pos=0] [tail=300]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.config import FeatureBuilderConfig  # noqa: E402
from scripts.audit_feature_causality import changed, raw_build  # noqa: E402
from scripts.section4_parity_harness import FrozenReference, dataset_metadata_for, roster_by_id, training_frame  # noqa: E402


def main() -> int:
    dataset_dir, stream = sys.argv[1], sys.argv[2]
    ends = [a for a in sys.argv[3:] if "=" not in a]
    opts = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    start_pos, tail = int(opts.get("start_pos", 0)), int(opts.get("tail", 300))
    config = FeatureBuilderConfig.from_dict(json.loads((ROOT / dataset_dir / "dataset.metadata.json").read_text(encoding="utf-8"))["config"])
    source = training_frame(stream)
    last = max(int(source.timestamp.searchsorted(pd.Timestamp(e))) for e in ends)
    bars = source.iloc[start_pos:last + 1].rename(columns={"timestamp": "datetime"}).reset_index(drop=True)
    if stream == "EURUSD":
        bars = bars.drop(columns=["contract_symbol"])
    started = time.perf_counter()
    full, _ = raw_build(config, bars)
    seconds = round(time.perf_counter() - started, 1)
    frozen, owners = {}, {}
    for model in roster_by_id().values():
        reference = FrozenReference(model)
        if dataset_metadata_for(reference.contract)[0].parent.resolve() != (ROOT / dataset_dir).resolve():
            continue
        names = [n for n in reference.summary["selected_feature_names"] if n not in frozen]
        hist = pd.read_parquet(reference.prepared / "source_history.parquet", columns=["datetime", *names])
        hist.index = pd.to_datetime(hist.pop("datetime"), utc=True)
        for name in names:
            frozen[name], owners[name] = hist[name], model["queue_id"]
    result = {}
    for end in ends:
        rows = full.index[full.index <= pd.Timestamp(end)][-tail:]
        for name, series in frozen.items():
            if name not in full.columns:
                result.setdefault(name, {})[end] = "missing_in_build"
                continue
            f_raw, b_raw = series.reindex(rows), full.loc[rows, name]
            f = pd.to_numeric(f_raw, errors="coerce").to_numpy(float)
            b = pd.to_numeric(b_raw, errors="coerce").to_numpy(float)
            with np.errstate(invalid="ignore"):
                close = np.isfinite(f) & np.isfinite(b) & (np.abs(b - f) <= 1e-6 + 1e-6 * np.abs(f))
            bad = changed(b_raw, f_raw) & ~close
            if bad.any():
                result.setdefault(name, {})[end] = dict(cells=int(bad.sum()), first=str(rows[int(np.argmax(bad))]))
    out = ROOT / "artifacts/feature_causality_audit" / f"{Path(dataset_dir).name}_fullhistory_start{start_pos}.json"
    out.write_text(json.dumps(dict(dataset=dataset_dir, stream=stream, ends=ends, start_pos=start_pos, bars=len(bars),
                                   build_seconds=seconds, features_compared=len(frozen), mismatched=result), indent=1), encoding="utf-8")
    print(f"{Path(dataset_dir).name} full-history ({len(bars)} bars, {seconds}s): compared {len(frozen)}, mismatched {len(result)}: "
          + ", ".join(f"{k}({owners.get(k)})" for k in sorted(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
