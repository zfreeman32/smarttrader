"""Which strategy_signals strategies carry unbounded memory? (retrain R1, v2 performance split)

Builds the ``strategy_signals`` feature set alone (v2 strategy ids, unblocked) on windows of ``short`` and each of
``longs`` bars ending at T, and compares the last ``tail`` rows per column (C1 tolerance, NaN == NaN). A strategy
whose columns differ carries memory beyond ``short`` bars and must run per weekly block in v2; all others are
window-invariant and run once on the whole frame.
Usage: python scripts/audit_strategy_memory.py <dataset_dir> <ES|EURUSD> <end_iso> [short=8000 longs=16000,24000 tail=300]
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.causal_contract import v2_strategy_ids  # noqa: E402
from features.config import FeatureBuilderConfig  # noqa: E402
from features.io import standardize_market_frame  # noqa: E402
from features.registry import FEATURE_REGISTRY  # noqa: E402
from features.strategy_registry import STRATEGY_REGISTRY, slugify_strategy_name  # noqa: E402
from scripts.audit_feature_causality import changed  # noqa: E402
from scripts.section4_parity_harness import training_frame  # noqa: E402


def main() -> int:
    dataset_dir, stream, end = sys.argv[1:4]
    opts = dict(a.split("=", 1) for a in sys.argv[4:])
    short, tail = int(opts.get("short", 8000)), int(opts.get("tail", 300))
    longs = [int(v) for v in opts.get("longs", "16000,24000").split(",")]
    raw = json.loads((ROOT / dataset_dir / "dataset.metadata.json").read_text(encoding="utf-8"))["config"]
    config = FeatureBuilderConfig.from_dict(raw)
    ids = v2_strategy_ids(config.strategy_ids)
    config = dataclasses.replace(config, strategy_ids=ids)
    source = training_frame(stream)
    pos = int(source.timestamp.searchsorted(pd.Timestamp(end)))
    bars = source.rename(columns={"timestamp": "datetime"})
    if stream == "EURUSD":
        bars = bars.drop(columns=["contract_symbol"])
    frames, timings = {}, {}
    for n in [short, *longs]:
        window = standardize_market_frame(bars.iloc[pos - n + 1:pos + 1].reset_index(drop=True),
                                          source_timezone=config.source_timezone, canonical_timezone=config.canonical_timezone)
        started = time.perf_counter()
        built = FEATURE_REGISTRY.build("strategy_signals", window, config)
        timings[n] = round(time.perf_counter() - started, 1)
        built.index = pd.to_datetime(window["datetime"], utc=True)
        frames[n] = built
        print(f"window {n}: {timings[n]} s, {built.shape[1]} columns", flush=True)
    rows = frames[short].index[-tail:]
    slug_to_id = {slugify_strategy_name(STRATEGY_REGISTRY.resolve(i)): i for i in ids}
    memory_columns = {}
    for column in frames[short].columns:
        b_raw = frames[short].loc[rows, column]
        b = pd.to_numeric(b_raw, errors="coerce").to_numpy(float)
        worst = 0
        for n in longs:
            if column not in frames[n].columns:
                worst = tail
                continue
            a_raw = frames[n].loc[rows, column]
            a = pd.to_numeric(a_raw, errors="coerce").to_numpy(float)
            with np.errstate(invalid="ignore"):
                close = np.isfinite(a) & np.isfinite(b) & (np.abs(b - a) <= 1e-6 + 1e-6 * np.abs(a))
            worst = max(worst, int((changed(b_raw, a_raw) & ~close).sum()))
        if worst:
            memory_columns[column] = worst
    memory_ids = set()
    for column in memory_columns:
        body = column[len("strategy__"):] if column.startswith("strategy__") else column
        slug = max((s for s in slug_to_id if body.startswith(s + "__")), key=len, default=None)
        memory_ids.add(slug_to_id.get(slug, f"UNMAPPED:{column}"))
    out = dict(dataset=dataset_dir, stream=stream, end=end, short=short, longs=longs, tail=tail, timings=timings,
               strategies=len(ids), memory_columns=memory_columns, memory_strategy_ids=sorted(memory_ids))
    path = ROOT / "artifacts/feature_causality_audit" / f"strategy_memory_{Path(dataset_dir).name}_{pd.Timestamp(end).strftime('%Y%m%dT%H%M')}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"STRATEGY MEMORY: {len(memory_ids)} of {len(ids)} strategies ({len(memory_columns)} columns): {sorted(memory_ids)} -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
