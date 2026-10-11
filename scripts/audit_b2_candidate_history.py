"""Audit retained candidates against raw, consecutive source-bar runtime history.

This is an offline diagnostic. It never installs a model or clears collection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from scripts import run_b2_retraining as old
from scripts.review_b2_first_pass import replay_predictions
from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import file_sha256
from preprocessing.feature_selection import encode_candidate_features, build_fill_values, apply_fill_values
from ote_live.models.loaders import (LoadedRuntimeModel, validate_source_history_runtime,
    _load_backend_model, _resolve_runtime_amp_usage)
from ote_live.models.runners import RuntimeModelRunner

IDS = (2, 1, 11, 17, 7)


def dump_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_frame(manifest, model_id, dataset):
    job = manifest["models"][str(model_id)]
    # Most prepared roots sit beside their source dataset.  The FRVP
    # recency-weighted view is different: its summary points back to the
    # pinned July source tape.  Always resolve the declared source parent so
    # a derived prepared view cannot silently become the source history.
    prepared_summary = ROOT / Path(job["old_prepared_dir"]).parent / "summary.json"
    summary = json.loads(prepared_summary.read_text())
    declared_source = Path(summary.get("source_dataset", summary["input_file"]))
    path = declared_source if declared_source.is_absolute() else ROOT / declared_source
    path = path.resolve()
    relative = str(path.relative_to(ROOT))
    expected = manifest["parents"][relative]
    old.require_hash(path, expected)
    frame = pd.read_parquet(path, columns=["source_row_idx", "datetime", *dataset.feature_names])
    np.testing.assert_array_equal(frame.source_row_idx, np.arange(len(frame)))
    timestamps = pd.to_datetime(frame.datetime, utc=True)
    if not timestamps.is_monotonic_increasing or not timestamps.is_unique:
        raise ValueError("Source timestamps must be unique and chronological")
    encoded, encoding = encode_candidate_features(frame[dataset.feature_names], dataset.feature_names)
    if encoding["encoded_columns"]:
        raise ValueError("Selected categorical predictors require a separately frozen encoder")
    fit_ids = dataset.source_row_idx_train[:job["selection_fit_rows"]]
    fills, _ = build_fill_values(encoded.iloc[fit_ids])
    fills = {name: float(fills.get(name, 0.)) for name in dataset.feature_names}
    filled = apply_fill_values(encoded, fills)
    for ids, expected_X in ((dataset.source_row_idx_train, dataset.X_train),
                            (dataset.source_row_idx_val, dataset.X_val),
                            (dataset.source_row_idx_test, dataset.X_test)):
        np.testing.assert_allclose(filled.iloc[ids].to_numpy(), expected_X, rtol=1e-6, atol=1e-6)
    return frame, encoded, filled, fills, path, expected


def runtime(output, summary, *, frozen_scaler=None):
    backend = summary["config"]["backend"]
    if backend == "torch":
        backend = summary["config"]["model_type"]
    family = "ict" if "_ict_" in output.name else "frvp"
    manifest = SimpleNamespace(model_id=f"{family}_{output.name}_history_audit", backend=backend,
        direction="short" if output.name.startswith("short") else "long",
        feature_manifest=SimpleNamespace(selected_feature_names=summary["selected_feature_names"],
            lag_steps=summary["lag_steps"], delta_feature_count=summary["delta_feature_count"]),
        artifact_references=SimpleNamespace(model_file=str(output / ("model.json" if backend == "xgboost" else "model.pt"))),
        context_requirements=SimpleNamespace(context_rows=summary["context_rows"], window_size=summary["window_size"]))
    # Exercise the production backend/device/precision path, including CPU TCN.
    model = _load_backend_model(manifest)
    use_amp = _resolve_runtime_amp_usage(manifest=manifest, training_summary=summary)
    scaler = frozen_scaler if frozen_scaler is not None else joblib.load(output / "scaler.joblib")
    validate_source_history_runtime(manifest, summary, scaler)
    calibrator = joblib.load(output / "calibrator.joblib") if (output / "calibrator.joblib").exists() else None
    return RuntimeModelRunner(LoadedRuntimeModel(manifest, model, scaler, calibrator, summary,
        summary["config"]["scale_clip"], summary["config"]["batch_size"], use_amp))


def audit_one(manifest, model_id, *, source_history_only=False):
    dataset, config, gaps = old.load_job(manifest, model_id)
    run_root = Path(config.output_root)
    output = run_root / dataset.target_name
    run_path = run_root / "b2_retraining_manifest.json"
    if not run_path.exists() or not output.exists():
        if not source_history_only:
            raise FileNotFoundError(
                f"No reviewed retained-row training bundle for ID {model_id}; "
                "use --source-history-only for a new replacement audit"
            )
        # Background models are intentionally not admitted from an attractive
        # legacy backtest.  Validate only the pinned source tape and the
        # frozen-prefix reconstruction; the replacement must be trained below.
        frame, encoded, filled, fills, source, source_sha = source_frame(manifest, model_id, dataset)
        contract = dict(
            mode="source_history_repair_required",
            source_dataset=str(source.relative_to(ROOT)),
            source_dataset_sha256=source_sha,
            source_rows=len(frame),
            feature_names=list(dataset.feature_names),
            selection_fit_rows=int(manifest["models"][str(model_id)]["selection_fit_rows"]),
            fill_values=fills,
            frozen_prefix_reconstruction_matches_prepared=True,
            legacy_bundle_used_for_replacement=False,
        )
        return dict(
            id=model_id,
            target=dataset.target_name,
            backend=config.backend,
            status="source_history_repair_required",
            disposition="legacy_diagnostic_only",
            collection_eligible=False,
            selected_for_collection_review=True,
            selected_feature_order_verified=True,
            selected_features=len(dataset.feature_names),
            source_dataset=contract["source_dataset"],
            source_dataset_sha256=source_sha,
            source_rows=len(frame),
            selection_fit_rows=contract["selection_fit_rows"],
            frozen_fill_reconstruction_matches_prepared=True,
            legacy_bundle_present=False,
            legacy_bundle_used_for_replacement=False,
            required_original_bar_gaps=gaps,
            limitation="Pinned source tape and frozen fills only; replacement training and runtime replay are required",
        )
    run = json.loads(run_path.read_text())
    if run["lineage"] != manifest["models"][str(model_id)] or run["config"] != asdict(config):
        raise ValueError("Archived run recipe/lineage changed")
    for name, sha in run["outputs_sha256"].items():
        old.require_hash(output / name, sha)
    summary = json.loads((output / "training_summary.json").read_text())
    names = summary["selected_feature_names"]
    if names != dataset.feature_names[:len(names)]:
        raise ValueError("Selected-feature order mismatch")
    saved = pd.read_csv(output / "test_predictions.csv")
    np.testing.assert_array_equal(saved.source_row_idx, dataset.test_source_row_idx)
    replay = replay_predictions(dataset, config, summary, output)
    np.testing.assert_allclose(replay, saved.raw_probability, rtol=1e-5, atol=1e-6)
    frame, encoded, filled, fills, source, source_sha = source_frame(manifest, model_id, dataset)
    runner = runtime(output, summary)
    # Establish the saved scaler fit scope independently of its serialized state.
    expected_scaler = t.fit_scaler(dataset.dev_X[:, :len(names)], config)
    np.testing.assert_array_equal(runner.loaded_model.scaler.center_, expected_scaler.center_)
    np.testing.assert_array_equal(runner.loaded_model.scaler.scale_, expected_scaler.scale_)
    context = summary["context_rows"]
    timeline = np.concatenate([dataset.dev_source_row_idx[-context:], dataset.test_source_row_idx])
    lags = summary["lag_steps"] if config.backend == "xgboost" else list(range(context + 1))
    changed_histories = 0
    raw_rejected = 0
    missing_cells = 0
    predictions = []
    raw_predictions = []
    for n, endpoint in enumerate(dataset.test_source_row_idx):
        retained = timeline[n + context - np.asarray(lags)]
        changed_histories += int(not np.array_equal(retained, endpoint - np.asarray(lags)))
        raw = frame.iloc[endpoint-context:endpoint+1]
        missing_cells += int((~np.isfinite(raw[names].to_numpy(dtype=float))).sum())
        try:
            raw_predictions.append(runner.predict_latest(raw).raw_score)
        except ValueError as exc:
            if "NaN" not in str(exc) and "Input X contains" not in str(exc):
                raise
            raw_predictions.append(None)
            raw_rejected += 1
        ready = raw[["source_row_idx", "datetime"]].copy()
        ready[names] = filled.iloc[endpoint-context:endpoint+1][names]
        predictions.append(runner.predict_latest(ready).raw_score)
    probability = np.asarray(predictions)
    threshold = summary["threshold"]
    raw_valid = np.array([x is not None for x in raw_predictions])
    raw_delta = np.abs(np.asarray([x or 0. for x in raw_predictions]) - probability)
    print(f"ID {model_id}: {changed_histories}/{len(saved)} histories differ; "
          f"{np.sum((probability >= threshold) != saved.predicted_label)}/{len(saved)} threshold decisions differ", flush=True)
    return dict(id=model_id, target=dataset.target_name, backend=config.backend,
        status="affected_diagnostic_only", collection_eligible=False,
        selected_feature_order_verified=True, selected_features=len(names),
        source_dataset=str(source.relative_to(ROOT)), source_dataset_sha256=source_sha,
        test_endpoints=len(saved), context_rows=context, source_rows=len(frame),
        retained_history_endpoint_mismatches=changed_histories,
        frozen_fill_reconstruction_matches_prepared=True, saved_frozen_fill_contract=False,
        raw_runtime_rejected_endpoints=raw_rejected, raw_window_nonfinite_cells=missing_cells,
        raw_vs_frozen_fill_score_differences=int(np.sum(raw_valid & (raw_delta > 1e-6))),
        saved_scaler_fit_scope="all development endpoints, exact center/scale reproduced",
        retained_row_saved_replay_max_abs_error=float(np.max(np.abs(replay-saved.raw_probability))),
        consecutive_history_frozen_fill_max_abs_error=float(np.max(np.abs(probability-saved.raw_probability))),
        consecutive_history_threshold_disagreements=int(np.sum((probability >= threshold) != saved.predicted_label)),
        frozen_archived_threshold=threshold, required_original_bar_gaps=gaps,
        original_manifest_sha256=file_sha256(run_path), outputs_sha256=run["outputs_sha256"],
        original_code_sha256=run["code_sha256"],
        original_code_paths_changed_since_fit=[p for p,h in run["code_sha256"].items() if file_sha256(ROOT/p) != h],
        limitation="Prepared-source inference only; actual live producers, availability, gaps and rolls remain separate gates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", choices=old.GROUPS["all"], default=list(IDS))
    parser.add_argument(
        "--source-history-only",
        action="store_true",
        help="Allow explicitly untrained IDs; audit only their pinned source tape and frozen fills",
    )
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a new report path")
    manifest = old.load_manifest()
    model_ids = tuple(dict.fromkeys(args.ids))
    results = [audit_one(manifest, i, source_history_only=args.source_history_only) for i in model_ids]
    dump_new(args.report, dict(status="candidate_history_audit_complete", models=results,
        model_ids=list(model_ids), source_history_only=args.source_history_only,
        validation_manifest_sha256=old.MANIFEST_SHA256, trading_model_training_run=False,
        collection_eligible=False, background_and_ote_prerequisite=False,
        deferred_long_reversal_recency=dict(half_life_days=730, floor=0.20)))


if __name__ == "__main__":
    main()
