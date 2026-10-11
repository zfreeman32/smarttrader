"""Versioned B2 FRVP labels: preserve saved setup candidates and historical rules."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.labeling import frvp_labeling_engine as f
from frvp.feature_sets.frvp_context import _prepare_frvp_context
from frvp.pipelines.es_primary_phase04 import _default_phase03_labeling_params
from model_testing.artifact_lineage_audit import digest


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def compare(old, new):
    result = {}
    for c in old.columns.intersection(new.columns):
        if pd.api.types.is_numeric_dtype(old[c]) or pd.api.types.is_bool_dtype(old[c]):
            different = ~np.isclose(pd.to_numeric(old[c]).to_numpy(float), pd.to_numeric(new[c]).to_numpy(float), rtol=1e-6, atol=1e-6, equal_nan=True)
        else:
            different = old[c].fillna("").astype(str).to_numpy() != new[c].fillna("").astype(str).to_numpy()
        if different.any():
            result[c] = int(different.sum())
    return result


def replay(market, path, continuous, features, params):
    print("Building HTF context", params.completed_htf_publication, flush=True)
    atr, _, structural, sw30, sw1h, _, _ = f._detect_htf_swings(path, params)
    events = f._build_event_candidates(market, features, structural.reindex(market.index))
    match30 = f.compute_htf_swing_match(events, sw30, params.htf_confluence_window_minutes)
    match1h = f.compute_htf_swing_match(events, sw1h, params.htf_confluence_window_minutes * 2)
    f.annotate_swing_context(events, sw30, sw1h, match30, match1h, base_bar_minutes=5)
    if params.completed_htf_publication:
        f.annotate_confirmed_htf_spacing(events, sw30, sw1h, base_bar_minutes=5)
    f.trend_scan_swings(path, events, params)
    prices = market[["open", "high", "low", "close"]]
    f._evaluate_event_barriers(prices, structural, events, params)
    f._score_event_entries(prices, structural, events)
    f._compute_event_quality(events, params)
    events, diag = f._apply_exclusion_masks(market.reset_index(drop=True), events, continuous, 5, params)
    concurrency = f._apply_event_concurrency_and_weights(market.index, events, params)
    labels = f._materialize_frvp_targets(market.reset_index(drop=True), events, concurrency)
    labels["atr"] = atr.reindex(labels.index)
    labels["structural_atr"] = structural.reindex(labels.index)
    labels["warmup_mask"] = False
    labels.iloc[:params.warmup_bars, labels.columns.get_loc("warmup_mask")] = True
    labels.index.name = "datetime"
    return labels, f.frvp_events_to_frame(events), diag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        ap.error("Use a new version directory under artifacts.")
    output.mkdir(parents=True)
    raw_path = ROOT / "data/futures_data/ES-5m-tagged.csv"
    raw = pd.read_csv(raw_path)
    print("Preparing unchanged continuous price path", flush=True)
    context = _prepare_frvp_context(raw, f._frvp_only_config(instrument="es"), instrument="es")
    market = context.market.copy()
    market["timestamp"] = pd.to_datetime(market.timestamp, utc=True)
    market = market.set_index("timestamp", drop=False)
    path = context.continuous.path_bars.bars.copy()
    path.index = pd.to_datetime(path.pop("timestamp"), utc=True)
    path = path[["open", "high", "low", "close", "volume"]]
    params = replace(_default_phase03_labeling_params(instrument="es"), completed_htf_publication=False)
    summary = {"version": output.name, "raw_sha256": digest(raw_path), "historical_params": asdict(params),
               "repairs": ["completed HTF publication and full buckets", "confirmed-only HTF setup gate", "helper predictors excluded downstream"],
               "batches": {}, "training_run": False}
    dump(output / "manifest.json", summary)
    for name in ("frvp_es_primary_refresh_20260701", "frvp_es_primary_setup_targets_20260829"):
        batch = output / name
        batch.mkdir()
        old_root = ROOT / "artifacts" / name / "phase03"
        saved_events = pd.read_csv(old_root / "es_primary_frvp_events.csv")
        old_labels = pd.read_csv(old_root / "es_primary_frvp_labels.csv").set_index("datetime")
        old_labels.index = pd.to_datetime(old_labels.index, utc=True)
        assert old_labels.index.equals(market.index)
        features = pd.DataFrame(0., index=range(len(market)), columns=["frvp_setup_type", "frvp_setup_side", "frvp_setup_confidence_rule"])
        ids = saved_events.swing_index.to_numpy(int)
        assert len(set(ids)) == len(ids)
        assert (market.index[ids] == pd.to_datetime(saved_events.swing_time, utc=True)).all()
        features.loc[ids] = saved_events[["setup_type", "setup_side", "setup_confidence"]].to_numpy()
        print("Legacy replay", name, flush=True)
        legacy, legacy_events, legacy_diag = replay(market, path, context.continuous, features, params)
        legacy_diff = compare(old_labels, legacy)
        dump(batch / "legacy_reproduction.json", legacy_diff)
        # Historical versions may contain fewer target columns; compare common fields.
        if legacy_diff:
            summary["batches"][name] = {"status": "legacy_mismatch_requires_review", "differences": legacy_diff}
            dump(output / "manifest.json", summary)
            raise RuntimeError(f"Historical labeling did not reproduce: {legacy_diff}")
        del legacy, legacy_events
        gc.collect()
        print("Corrected replay", name, flush=True)
        labels, events, diag = replay(market, path, context.continuous, features, replace(params, completed_htf_publication=True))
        differences = compare(old_labels, labels)
        labels.reset_index().to_csv(batch / "labels.csv.gz", index=False)
        events.to_csv(batch / "events.csv.gz", index=False)
        dump(batch / "comparison.json", differences)
        dump(batch / "diagnostics.json", diag)
        dump(batch / "labeling_config.json", asdict(replace(params, completed_htf_publication=True)))
        summary["batches"][name] = {"status": "labels_rebuilt", "rows": len(labels), "events": len(events),
                                   "changed_columns": differences, "source_labels_sha256": digest(old_root / "es_primary_frvp_labels.csv"),
                                   "source_events_sha256": digest(old_root / "es_primary_frvp_events.csv"),
                                   "labels_sha256": digest(batch / "labels.csv.gz"), "events_sha256": digest(batch / "events.csv.gz")}
        dump(output / "manifest.json", summary)
        del labels, events, features, old_labels
        gc.collect()
    print("Completed FRVP label, quality, exclusion and weight rebuilds", flush=True)


if __name__ == "__main__":
    main()
