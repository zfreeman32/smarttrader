"""Retrain R2a: full-history feature tape under feature contract v2 (features/causal_contract.py), one build path.

For one B2 group it rebuilds every feature column from the pinned training source (the same bars the live app and
the section 4 audits use) with the frozen v1 dataset config plus ``feature_version=2``, ``bounded_memory_warmup_bars``
and in-process strategies (``strategy_timeout_seconds=None``). It writes a new features root in the layout
``scripts/rebuild_b2_prepared.py`` consumes (``features.csv.gz`` with a UTC ``datetime`` column in training-source row
order, plus ``features.metadata.json``). ``feature_columns`` lists only v2-eligible predictors, so positional, identity
and look-ahead columns can never be selected.

Acceptance (``--check-ends``): for each end bar, a fresh 8,000-bar v2 build must equal the tape on the last 300 rows
for every eligible feature (C1: 1e-6 + 1e-6*|ref|, NaN == NaN). That makes the full-history tape and the app's window
the same function. The previous features root is read only for its config and row count and is never modified.

Usage: python scripts/build_b2_v2_features.py --group ict|frvp_july|frvp_setups|ote --base-features-root <old>
       --output-root artifacts/<new> [--check-ends ISO ISO ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder  # noqa: E402
from features.causal_contract import V2_DEFAULT_WARMUP_BARS, strategy_one_hot_prefix, v2_predictor_excluded  # noqa: E402
from features.config import FeatureBuilderConfig  # noqa: E402
from scripts.audit_feature_causality import changed  # noqa: E402
from scripts.section4_parity_harness import TRAINING_SOURCES, training_frame  # noqa: E402

STREAM = {"ict": "ES", "frvp_july": "ES", "frvp_setups": "ES", "ote": "EURUSD"}
CODE = ("features/causal_contract.py", "features/builder.py", "features/transforms.py", "features/config.py",
        "frvp/feature_sets/frvp_context.py", "features/feature_sets/strategy_signals.py",
        "features/strategies/high_volume_points_causal_v2.py", "features/strategies/ichimoku_strategy_causal_v2.py",
        "features/strategies/tesla_stock_boom_causal_v2.py", "scripts/build_b2_v2_features.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def v2_config(base_metadata: dict) -> FeatureBuilderConfig:
    raw = dict(base_metadata["config"])
    raw.update(feature_version=2, bounded_memory_warmup_bars=V2_DEFAULT_WARMUP_BARS, strategy_timeout_seconds=None)
    return FeatureBuilderConfig.from_dict(raw)


def source_bars(stream: str) -> pd.DataFrame:
    bars = training_frame(stream).rename(columns={"timestamp": "datetime"})
    return bars.drop(columns=["contract_symbol"]) if stream == "EURUSD" else bars


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=sorted(STREAM), required=True)
    ap.add_argument("--base-features-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--check-ends", nargs="*", default=[])
    args = ap.parse_args()
    output = args.output_root.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        ap.error("Use a new directory under artifacts.")
    base_meta = json.loads((args.base_features_root / "features.metadata.json").read_text(encoding="utf-8"))
    stream = STREAM[args.group]
    config = v2_config(base_meta)
    bars = source_bars(stream)
    print(f"{args.group}: building v2 features on {len(bars)} bars", flush=True)
    started = time.perf_counter()
    frame, build_meta = FeatureDatasetBuilder(config).build(bars)
    seconds = round(time.perf_counter() - started, 1)
    print(f"{args.group}: built {frame.shape} in {seconds} s", flush=True)
    if len(frame) != int(base_meta["rows"]):
        raise SystemExit(f"row count {len(frame)} != base tape {base_meta['rows']} (label alignment would break)")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    generated = list(build_meta.get("feature_columns") or [])
    eligible = [c for c in generated if not v2_predictor_excluded(c)]
    excluded = [c for c in generated if v2_predictor_excluded(c)]
    # Freeze the tape's strategy one-hot vocabulary so window builds (acceptance, app) emit the same columns.
    one_hots = sorted(c for c in generated if strategy_one_hot_prefix(c) and frame[c].isin([0, 1]).all())
    if one_hots:
        config = dataclasses.replace(config, v2_strategy_one_hot_columns=one_hots)

    checks = {}
    for end in args.check_ends:
        pos = int(bars.datetime.searchsorted(pd.Timestamp(end)))
        window, _ = FeatureDatasetBuilder(config).build(bars.iloc[pos - 8000 + 1:pos + 1].reset_index(drop=True))
        window.index = pd.to_datetime(window["datetime"], utc=True)
        tape = frame.set_index("datetime")
        rows = window.index[-300:]
        bad = {}
        for column in eligible:
            if column not in window.columns:
                bad[column] = "missing_in_window"
                continue
            a_raw, b_raw = tape.loc[rows, column], window.loc[rows, column]
            a = pd.to_numeric(a_raw, errors="coerce").to_numpy(float)
            b = pd.to_numeric(b_raw, errors="coerce").to_numpy(float)
            with np.errstate(invalid="ignore"):
                close = np.isfinite(a) & np.isfinite(b) & (np.abs(b - a) <= 1e-6 + 1e-6 * np.abs(a))
            cells = int((changed(a_raw, b_raw) & ~close).sum())
            if cells:
                bad[column] = cells
        checks[end] = dict(columns=len(eligible), mismatched=len(bad), detail=dict(list(bad.items())[:50]))
        print(f"{args.group}: acceptance {end}: {len(bad)} of {len(eligible)} eligible columns differ"
              + (f" e.g. {list(bad)[:6]}" if bad else ""), flush=True)

    output.mkdir(parents=True)
    keep = ["datetime", *[c for c in ("open", "high", "low", "close", "volume", "contract_symbol") if c in frame.columns],
            *[c for c in generated if c in frame.columns]]
    path = output / "features.csv.gz"
    frame.loc[:, list(dict.fromkeys(keep))].to_csv(path, index=False, compression="gzip")
    source = ROOT / TRAINING_SOURCES[stream]
    metadata = dict(build_meta)
    metadata.update(
        config=config.to_dict(), rows=int(len(frame)), feature_columns=eligible, feature_column_count=len(eligible),
        source_path=str(source.relative_to(ROOT)), source_metadata_file=None,
        b2_v2=dict(version="b2-features-v2", feature_contract="features/causal_contract.py", group=args.group,
                   base_features_root=str(args.base_features_root), base_metadata_sha256=sha(args.base_features_root / "features.metadata.json"),
                   source_sha256=sha(source), features_sha256=sha(path), build_seconds=seconds,
                   v2_excluded_columns=excluded, acceptance=checks,
                   code_sha256={p: sha(ROOT / p) for p in CODE}, trading_model_training_run=False,
                   created_at_utc=datetime.now(timezone.utc).isoformat()))
    (output / "features.metadata.json").write_text(json.dumps(metadata, indent=1, default=str), encoding="utf-8")
    failed = [e for e, c in checks.items() if c["mismatched"]]
    print(f"{args.group}: wrote {output.relative_to(ROOT)} ({len(eligible)} eligible, {len(excluded)} excluded); "
          f"acceptance {'PASS' if not failed else 'FAIL at ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
