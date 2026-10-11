"""Causal-prefix and bounded-memory audit of a frozen dataset config's feature producers (retrain step R0 / guard).

For one dataset config and one end bar T of the pinned training source it builds raw (pre-fill) producer output.
Causal prefix (fixed start S, varying end): C = bars S..T+append, and ``cuts`` prefixes P_k = bars S..T_k with
T_k = T - k*step. A column is non-causal if any of the last ``depth`` rows of any P_k differs from C (exact; same code,
same start, so only appended bars can change them). Lookahead depth = the farthest row from T_k that changed.
Bounded memory: B = the last ``short`` bars up to T (the app's runtime window) vs A_j = the last ``longs[j]`` bars up
to T (several start points, since a structure older than ``short`` bars is only active for some starts); a column is
memory-unbounded if any of the last ``tail`` rows of any A_j differs from B beyond C1 (1e-6 + 1e-6*|ref|).
Frozen contract test: for features selected by roster models built from this dataset config, B is also compared with
those models' frozen full-history ``source_history.parquet`` on the same rows (C1, NaN == NaN): ``frozen_cells``.

Usage: python scripts/audit_feature_causality.py <dataset_dir> <ES|EURUSD> <end_iso> [short=8000 long=16000
       append=30 tail=300 key=value config overrides...]
Writes artifacts/feature_causality_audit/<dataset>_<end>.json (summaries only) and prints a short table.
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

from features.builder import FeatureDatasetBuilder  # noqa: E402
from features.config import FeatureBuilderConfig  # noqa: E402
from scripts.section4_parity_harness import training_frame  # noqa: E402

OUT = ROOT / "artifacts/feature_causality_audit"
INVENTORY = ROOT / "docs/audits/section4_feature_inventory_20261002.json"


def raw_build(config: FeatureBuilderConfig, bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    captured: dict = {}

    def observer(frame, producers):
        captured["frame"] = frame.copy()
        captured["producers"] = dict(producers)

    FeatureDatasetBuilder(config).build(bars.copy(), pre_fill_observer=observer)
    frame = captured["frame"]
    time_column = next(c for c in ("datetime", "timestamp") if c in frame.columns)
    frame.index = pd.to_datetime(frame[time_column], utc=True)
    return frame, captured["producers"]


def changed(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Exact inequality: missing == missing, numbers compared numerically, anything else as text."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    a_na, b_na = a.isna().to_numpy(), b.isna().to_numpy()
    x = pd.to_numeric(a, errors="coerce").to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(float)
    numeric = ~np.isnan(x) & ~np.isnan(y)
    text_equal = a.astype(str).to_numpy() == b.astype(str).to_numpy()
    equal = np.where(a_na | b_na, a_na & b_na, np.where(numeric, x == y, text_equal))
    return ~equal


def main() -> int:
    dataset_dir, stream, end = sys.argv[1], sys.argv[2], sys.argv[3]
    opts = dict(a.split("=", 1) for a in sys.argv[4:])
    longs = [int(v) for v in opts.pop("longs", "8500,9500,11000,16000,24000").split(",")]
    short, append, tail, cuts, step, depth = (int(opts.pop(k, d)) for k, d in (
        ("short", 8000), ("append", 30), ("tail", 300), ("cuts", 12), ("step", 23), ("depth", 40)))
    long_ = max(longs)
    meta = json.loads((ROOT / dataset_dir / "dataset.metadata.json").read_text(encoding="utf-8"))
    raw_config = dict(meta["config"])
    raw_config.update({k: (None if v == "None" else v) for k, v in opts.items()})
    config = FeatureBuilderConfig.from_dict(raw_config)
    source = training_frame(stream)
    pos = int(source.timestamp.searchsorted(pd.Timestamp(end)))
    if source.timestamp.iloc[pos] != pd.Timestamp(end) or pos - long_ < 0 or pos + append >= len(source):
        raise SystemExit("end bar not usable for the requested windows")
    bars = source.rename(columns={"timestamp": "datetime"})
    if stream == "EURUSD":
        bars = bars.drop(columns=["contract_symbol"])
    timings, frames = {}, {}
    start = pos - short + 1
    if pos - max(longs) + 1 < 0:
        raise SystemExit("end bar has too little history for the longest window")
    windows = {"C": (start, pos + 1 + append), "B": (start, pos + 1)}
    windows.update({f"A{n}": (pos - n + 1, pos + 1) for n in longs})
    for k in range(1, cuts + 1):
        windows[f"P{k}"] = (start, pos + 1 - k * step)
    for name, (lo, hi) in windows.items():
        started = time.perf_counter()
        frames[name], producers = raw_build(config, bars.iloc[lo:hi].reset_index(drop=True))
        timings[name] = round(time.perf_counter() - started, 1)
    selected = {}
    if INVENTORY.exists():
        for entry in json.loads(INVENTORY.read_text(encoding="utf-8"))["models"]:
            for feature in entry["features"]:
                selected.setdefault(feature["feature"], []).append(entry["queue_id"])
    prefixes = ["B", *(f"P{k}" for k in range(1, cuts + 1))]
    rows_tail = frames["B"].index[-tail:]
    frozen = {}
    v2 = int(raw_config.get("feature_version", 1) or 1) >= 2
    try:
        from scripts.section4_parity_harness import FrozenReference, dataset_metadata_for, roster_by_id
        for model in ([] if v2 else roster_by_id().values()):  # the frozen tapes are v1; v2 has no frozen tape yet
            reference = FrozenReference(model)
            if dataset_metadata_for(reference.contract)[0].parent.resolve() != (ROOT / dataset_dir).resolve():
                continue
            path = reference.prepared / "source_history.parquet"
            names = [n for n in reference.summary["selected_feature_names"] if n not in frozen]
            hist = pd.read_parquet(path, columns=["datetime", *names])
            hist.index = pd.to_datetime(hist.pop("datetime"), utc=True)
            hist = hist.reindex(rows_tail)
            for name in names:
                frozen[name] = hist[name]
    except Exception as exc:  # the frozen test is additive evidence; record why it is missing
        frozen = {"__error__": repr(exc)}
    result = {}
    for column, producer in producers.items():
        if not all(column in frames[n] for n in frames):
            continue
        nc_cells, look = 0, 0
        for name in prefixes:
            rows = frames[name].index[-depth:]
            diff = changed(frames[name].loc[rows, column], frames["C"].loc[rows, column])
            if diff.any():
                nc_cells += int(diff.sum())
                look = max(look, int(depth - np.argmax(diff)))
        mem_cells, mem_end, max_abs = 0, False, None
        b_raw = frames["B"].loc[rows_tail, column]
        b = pd.to_numeric(b_raw, errors="coerce").to_numpy(float)
        for n in longs:
            a_raw = frames[f"A{n}"].loc[rows_tail, column]
            a = pd.to_numeric(a_raw, errors="coerce").to_numpy(float)
            exact = changed(b_raw, a_raw)
            with np.errstate(invalid="ignore"):
                close = np.abs(b - a) <= 1e-6 + 1e-6 * np.abs(a)
            numeric = np.isfinite(a) & np.isfinite(b)
            mem = exact & ~(numeric & close)
            mem_cells = max(mem_cells, int(mem.sum()))
            mem_end = mem_end or bool(mem[-1])
            if (numeric & mem).any():
                diff = float(np.max(np.abs(b - a)[numeric & mem]))
                max_abs = diff if max_abs is None else max(max_abs, diff)
        frozen_cells = None
        if column in frozen and not isinstance(frozen[column], str):
            f_raw = frozen[column]
            f = pd.to_numeric(f_raw, errors="coerce").to_numpy(float)
            exact = changed(b_raw, f_raw)
            with np.errstate(invalid="ignore"):
                close = np.isfinite(f) & np.isfinite(b) & (np.abs(b - f) <= 1e-6 + 1e-6 * np.abs(f))
            frozen_cells = int((exact & ~close).sum())
        result[column] = dict(producer=producer, frozen_cells=frozen_cells, noncausal_cells=nc_cells, lookahead_rows_from_end=look,
                              memory_cells=mem_cells, memory_endpoint=mem_end, memory_max_abs=max_abs,
                              selected_by=sorted(selected.get(column, [])))
    summary = dict(dataset=dataset_dir, stream=stream, end=end, short=short, long=long_, append=append, tail=tail,
                   cuts=cuts, step=step, depth=depth, longs=longs,
                   overrides=opts, timings_seconds=timings, columns=len(result),
                   noncausal=sorted(c for c, r in result.items() if r["noncausal_cells"]),
                   memory_unbounded=sorted(c for c, r in result.items() if r["memory_cells"]),
                   frozen_mismatch=sorted(c for c, r in result.items() if r["frozen_cells"]),
                   frozen_error=frozen.get("__error__") if isinstance(frozen.get("__error__"), str) else None,
                   features=result)
    if v2:  # R1 acceptance: columns eligible as v2 predictors must be causal and window-invariant
        from features.causal_contract import v2_predictor_excluded

        for column, item in result.items():
            item["v2_excluded"] = v2_predictor_excluded(column)
        summary["eligible_noncausal"] = sorted(c for c in summary["noncausal"] if not result[c]["v2_excluded"])
        summary["eligible_memory_unbounded"] = sorted(c for c in summary["memory_unbounded"] if not result[c]["v2_excluded"])
        summary["excluded_columns"] = sum(1 for r in result.values() if r["v2_excluded"])
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{k}{v}" for k, v in sorted(opts.items())) or "frozen"
    path = OUT / f"{Path(dataset_dir).name}_{pd.Timestamp(end).strftime('%Y%m%dT%H%M')}_{tag}.json"
    path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    sel_nc = [c for c in summary["noncausal"] if result[c]["selected_by"]]
    sel_mem = [c for c in summary["memory_unbounded"] if result[c]["selected_by"]]
    if v2:
        print(f"{Path(dataset_dir).name} {end} v2: columns {len(result)} (excluded {summary['excluded_columns']}), "
              f"eligible non-causal {len(summary['eligible_noncausal'])}: {summary['eligible_noncausal'][:6]}, "
              f"eligible memory-unbounded {len(summary['eligible_memory_unbounded'])}: {summary['eligible_memory_unbounded'][:6]}, "
              f"timings B {timings.get('B')}s -> {path.relative_to(ROOT)}")
        return 0
    print(f"{Path(dataset_dir).name} {end}: columns {len(result)}, non-causal {len(summary['noncausal'])} "
          f"(selected {len(sel_nc)}), memory-unbounded {len(summary['memory_unbounded'])} (selected {len(sel_mem)}), "
          f"timings {timings} -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
