"""Prepare, check, train or review the corrected S3 source-history replacement.

Only `train` fits a trading model. Every output version is separate from the
retained-row originals; this script never installs or activates a model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from scripts import run_b2_retraining as old
from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import MODE, FrozenHistoryScaler, file_sha256, indexed_lag_view
from preprocessing.feature_selection import encode_candidate_features, build_fill_values, apply_fill_values
from ote_live.models.loaders import LoadedRuntimeModel, validate_source_history_runtime
from ote_live.models.runners import RuntimeModelRunner

PREPARED = ROOT / "artifacts/b2_source_history_20260927_v1"
OUTPUT = ROOT / "models/b2_source_history_20260927_v1/06_long_frvp_continuation_setup3"
FEATURES = ROOT / "artifacts/b2_frvp_setups_prepared_20260921_v1/dataset.parquet"
FEATURES_SHA = "63d5ecf1789c170b46e17b231aa96cae76e3e9074e59cf25840a99e573aba9d1"
TARGET = "long_frvp_continuation_setup3"
CODE = (*old.CODE_FILES, "model_training/source_history.py", "ote_live/models/loaders.py",
        "ote_live/models/runners.py", "scripts/run_b2_source_history.py")


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(root):
    if root.exists() or not root.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("Preparation needs a new version directory under artifacts")
    old.require_hash(FEATURES, FEATURES_SHA)
    manifest = old.load_manifest()
    dataset, config, _ = old.load_job(manifest, 6)
    names = dataset.feature_names
    print("Building full source history and frozen prefix imputation", flush=True)
    frame = pd.read_parquet(FEATURES, columns=["source_row_idx", "datetime", *names])
    assert np.array_equal(frame.source_row_idx, np.arange(len(frame)))
    assert pd.to_datetime(frame.datetime, utc=True).is_monotonic_increasing and frame.datetime.is_unique
    encoded, encoding = encode_candidate_features(frame[names], names)
    if encoding["encoded_columns"]:
        raise ValueError("This S3 migration requires numeric selected predictors")
    fit_ids = dataset.source_row_idx_train[:manifest["models"]["6"]["selection_fit_rows"]]
    fills, _ = build_fill_values(encoded.iloc[fit_ids])
    fills = {name: float(fills.get(name, 0.)) for name in names}
    filled = apply_fill_values(encoded, fills)
    for ids, expected in ((dataset.source_row_idx_train, dataset.X_train),
                          (dataset.source_row_idx_val, dataset.X_val), (dataset.source_row_idx_test, dataset.X_test)):
        np.testing.assert_allclose(filled.iloc[ids].to_numpy(), expected, rtol=1e-6, atol=1e-6)
    prepared = root / "prepared" / TARGET
    prepared.mkdir(parents=True)
    for name in manifest["models"]["6"]["prepared_hashes"]:
        shutil.copyfile(dataset.prepared_dir / name, prepared / name)
    shutil.copyfile(dataset.prepared_dir.parent / "summary.json", prepared.parent / "summary.json")
    # Raw numeric predictors are saved; both loaders and runtime use the fills.
    tape = pd.concat([frame[["source_row_idx", "datetime"]], encoded], axis=1)
    tape.to_parquet(prepared / "source_history.parquet", index=False)
    contract = dict(mode=MODE, bar_minutes=5, feature_names=names, fill_values=fills,
        missing_value_rule="nonfinite_to_frozen_fill", selection_fit_rows=len(fit_ids),
        selection_fit_end_source_row_idx=int(fit_ids[-1]), source_dataset_sha256=FEATURES_SHA,
        source_history_sha256=file_sha256(prepared / "source_history.parquet"),
        clock_note="Every observed source bar in chronological order; market gaps/rolls remain separate eligibility checks")
    dump(prepared / "source_history_contract.json", contract)
    payload = dict(status="source_history_prepared", model_id=6, target=TARGET,
        source_manifest_sha256=old.MANIFEST_SHA256, source_feature_sha256=FEATURES_SHA,
        source_job=manifest["models"]["6"], config=asdict(config), rows=len(frame),
        prepared_files_sha256={str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()},
        old_models_changed=False, trading_model_training_run=False)
    dump(root / "manifest.json", payload)
    print(f"Prepared {len(frame)} source rows; preserved labels, weights, split IDs and 500-row selection", flush=True)


def load(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "source_history_prepared" or manifest["model_id"] != 6:
        raise ValueError("Expected corrected S3 preparation")
    if manifest["source_manifest_sha256"] != old.MANIFEST_SHA256 or manifest["source_feature_sha256"] != FEATURES_SHA:
        raise ValueError("Wrong preparation ancestry")
    for relative, sha in manifest["prepared_files_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Invalid prepared path")
        old.require_hash(path, sha)
    old.require_hash(FEATURES, FEATURES_SHA)
    # Preserve all recipe, gap and weighting decisions from the pinned queue.
    parents = old.load_manifest()
    old_dataset, config, gaps = old.load_job(parents, 6, output.parent)
    if manifest["source_job"] != parents["models"]["6"]:
        raise ValueError("Source recipe/selection lineage changed")
    config.prepared_root = str(root / "prepared")
    config.output_root = str(output)
    dataset = t.load_prepared_target_dataset(root / "prepared", TARGET, config)
    if dataset.source_history_X is None:
        raise ValueError("Refusing retained-row fallback for corrected S3")
    return dataset, config, gaps


def runtime_model(model, scaler, summary):
    """Evaluation-only binding through the production loader contract and runner."""
    manifest = SimpleNamespace(model_id="frvp_s3_source_history_candidate", backend="xgboost", direction="long",
        feature_manifest=SimpleNamespace(selected_feature_names=summary["selected_feature_names"],
            lag_steps=summary["lag_steps"], delta_feature_count=summary["delta_feature_count"]),
        context_requirements=SimpleNamespace(context_rows=summary["context_rows"], window_size=summary["window_size"]))
    validate_source_history_runtime(manifest, summary, scaler)
    return RuntimeModelRunner(LoadedRuntimeModel(manifest, model, scaler, None, summary,
        summary["config"]["scale_clip"], summary["config"]["batch_size"], False))


def check(dataset, config):
    checks = old.check_job(dataset, config, 6)
    # All search-window choices: indexed training tensors equal runtime tensors.
    from ote_live.models.runners import _build_sparse_lag_view
    scaler = t.fit_scaler(dataset.X_train[:500], config)
    endpoints = np.concatenate([dataset.dev_source_row_idx[:8], dataset.dev_source_row_idx[-8:], dataset.test_source_row_idx])
    feature_idx = np.arange(len(dataset.feature_names))
    tensor_checks = []
    for window in range(config.window_min, config.window_max + 1, 4):
        lags = t.make_lag_steps(window, config.lag_count_max)
        expected = indexed_lag_view(dataset.source_history_X, endpoints, feature_idx, scaler, lags,
                                    config.delta_feature_cap, config.scale_clip)
        actual = np.concatenate([_build_sparse_lag_view(t.transform_matrix(scaler,
            dataset.source_history_X[i-max(lags):i+1], clip_value=config.scale_clip),
            lag_steps=lags, delta_feature_count=config.delta_feature_cap) for i in endpoints])
        np.testing.assert_array_equal(actual, expected)
        tensor_checks.append(dict(window=window, endpoints=len(endpoints), exact_equal=True))
    return dict(status="source_history_preflight_passed", model=checks, tensor_parity=tensor_checks,
                code_sha256={p: file_sha256(ROOT / p) for p in CODE}, trading_model_training_run=False,
                promotion_eligible=False)


def review(dataset, config, output):
    run = json.loads((output / "source_history_training_manifest.json").read_text())
    if run["status"] != "trained_pending_source_history_review":
        raise ValueError("Corrected S3 training is incomplete")
    if run["config"] != asdict(config):
        raise ValueError("Corrected S3 recipe changed")
    model_dir = output / TARGET
    for name, sha in run["outputs_sha256"].items():
        old.require_hash(model_dir / name, sha)
    for name, sha in run["code_sha256"].items():
        old.require_hash(ROOT / name, sha)
    summary = json.loads((model_dir / "training_summary.json").read_text())
    if summary["model_config"].get("source_history_contract") != dataset.source_history_contract:
        raise ValueError("Trained model did not preserve the source-history contract")
    booster = xgb.Booster(); booster.load_model(model_dir / "model.json")
    scaler = joblib.load(model_dir / "scaler.joblib")
    runtime = runtime_model(booster, scaler, summary)
    # Use the RAW history here so replay actually exercises the saved fills.
    raw = pd.read_parquet(dataset.prepared_dir / "source_history.parquet").set_index("source_row_idx", drop=False)
    predictions = []
    for endpoint in dataset.test_source_row_idx:
        frame = raw.loc[endpoint-summary["context_rows"]:endpoint]
        predictions.append(runtime.predict_latest(frame, timestamp=pd.Timestamp(raw.loc[endpoint, "datetime"]).to_pydatetime()).raw_score)
    saved = pd.read_csv(model_dir / "test_predictions.csv")
    np.testing.assert_array_equal(saved.source_row_idx, dataset.test_source_row_idx)
    np.testing.assert_allclose(predictions, saved.raw_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(predictions) >= summary["threshold"], saved.predicted_label)
    oof = pd.read_csv(model_dir / "oof_predictions.csv")
    np.testing.assert_array_equal(oof.source_row_idx, dataset.dev_source_row_idx)
    valid = oof.oof_raw_probability.notna()
    threshold, _ = t.select_operating_threshold(y_true=dataset.dev_y[valid],
        y_score=oof.loc[valid, "oof_raw_probability"].to_numpy(np.float32),
        tolerance=config.event_tolerance_bars, cooldown=config.event_cooldown_bars,
        grid_size=config.threshold_grid_size, config=config)
    if not np.isclose(threshold, summary["threshold"]) or summary["calibration"]["resolved_method"] != "none":
        raise ValueError("OOF-only threshold or continuation calibration mismatch")
    return dict(status="source_history_artifact_review_passed", prediction_rows=len(predictions),
        prediction_max_abs_error=float(np.max(np.abs(predictions - saved.raw_probability))),
        threshold_decisions_equal=True, threshold=float(threshold), OOF_threshold_reproduced=True,
        model_outputs_sha256=run["outputs_sha256"], B1_evaluation_complete=False, promotion_eligible=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "check", "train", "review"))
    ap.add_argument("--prepared-root", type=Path, default=PREPARED)
    ap.add_argument("--output-root", type=Path, default=OUTPUT)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    if args.report and args.report.exists():
        ap.error("Use a new report path")
    if args.mode == "prepare":
        prepare(args.prepared_root.resolve())
        return
    data, config, gaps = load(args.prepared_root.resolve(), args.output_root.resolve())
    if args.mode == "review":
        result = review(data, config, args.output_root.resolve())
    else:
        result = check(data, config)
        print("Source-history tensors and all eight search windows pass preflight", flush=True)
        if args.mode == "train":
            out = args.output_root.resolve()
            if not out.is_relative_to(ROOT / "models"):
                ap.error("Use a new version under models")
            out.mkdir(parents=True, exist_ok=False)
            run = dict(status="training_started", config=asdict(config), source_history_contract=data.source_history_contract,
                prepared_manifest_sha256=file_sha256(args.prepared_root / "manifest.json"),
                code_sha256=result["code_sha256"], preflight=result, trusted_baseline=False, **gaps)
            path = out / "source_history_training_manifest.json"
            dump(path, run)
            trained = t.train_target_pipeline(data, config)
            run.update(status="trained_pending_source_history_review", outputs_sha256={
                p.name: file_sha256(p) for p in Path(trained["output_dir"]).iterdir() if p.is_file()})
            dump(path, run)
            result = dict(status=run["status"], output_dir=str(trained["output_dir"]), promotion_eligible=False)
    if args.report:
        dump(args.report, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
