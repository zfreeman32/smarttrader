"""Retrain R3 review: re-check one trained feature contract v2 model and replay it through the runtime (before B4).

Kept separate from ``scripts/run_b2_v2_source_history.py`` so reviewer fixes never change the training-code hashes
that each trained model pins. Checks (mirroring the v1 candidate-history reviews, for XGBoost and TCN):

* training manifest complete; recipe, preparation manifest and every pinned training-code file unchanged (live file
  and the model's code snapshot); model output files unchanged;
* selected features are the prefix of the frozen ranking; fitted source-history/fill contract; preparation report;
  purged final holdout; scaler fit scope (refit rows only);
* full search budget, best trial and its parameters; CV fold plans and frozen selection; OOF rows and folds;
  calibration contract and calibrated OOF; the threshold reselected from OOF only;
* TCN: fixed-duration final refit without validation selection;
* runtime replay: the production backend loader + ``RuntimeModelRunner`` on the raw source history (frozen fills
  applied by the saved scaler) reproduces the saved test raw and calibrated probabilities and every threshold decision.

Usage: python scripts/review_b2_v2_models.py ID [--prepared-root ...] [--output-root ...] [--report PATH]
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
from scripts import run_b2_v2_source_history as v2
from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import file_sha256, selected_contract
from ote_live.models.loaders import LoadedRuntimeModel, _load_backend_model, _resolve_runtime_amp_usage, validate_source_history_runtime
from ote_live.models.runners import RuntimeModelRunner


def runtime(model_dir: Path, summary: dict, asset: str) -> RuntimeModelRunner:
    backend = summary["config"]["backend"]
    if backend == "torch":
        backend = summary["config"]["model_type"]
    manifest = SimpleNamespace(model_id=f"b2_v2_{model_dir.parent.name}_review", backend=backend, asset=asset, timeframe="5m",
        direction="short" if model_dir.name.startswith("short") else "long",
        feature_manifest=SimpleNamespace(selected_feature_names=summary["selected_feature_names"],
            lag_steps=summary["lag_steps"], delta_feature_count=summary["delta_feature_count"]),
        artifact_references=SimpleNamespace(model_file=str(model_dir / ("model.json" if backend == "xgboost" else "model.pt"))),
        context_requirements=SimpleNamespace(context_rows=summary["context_rows"], window_size=summary["window_size"]))
    model = _load_backend_model(manifest)  # production backend/device/precision path, incl. CPU TCN
    scaler = joblib.load(model_dir / "scaler.joblib")
    validate_source_history_runtime(manifest, summary, scaler)
    calibrator = joblib.load(model_dir / "calibrator.joblib") if (model_dir / "calibrator.joblib").exists() else None
    use_amp = _resolve_runtime_amp_usage(manifest=manifest, training_summary=summary)
    return RuntimeModelRunner(LoadedRuntimeModel(manifest, model, scaler, calibrator, summary,
        summary["config"]["scale_clip"], summary["config"]["batch_size"], use_amp))


def review(model_id: int, root: Path, output: Path) -> dict:
    data, cfg, _ = v2.load(model_id, root, output)
    out = Path(cfg.output_root)
    run_path = out / "v2_training_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run["status"] != "trained_pending_v2_review" or run["model_id"] != model_id or run["config"] != asdict(cfg):
        raise ValueError("Incomplete or changed training recipe")
    if set(run["code_sha256"]) != set(v2.CODE) or run["corrected_events_sha256"] != old.EVENTS_SHA256:
        raise ValueError("Incomplete code or corrected event lineage")
    old.require_hash(root / str(model_id) / "manifest.json", run["prepared_manifest_sha256"])
    for name, sha in run["code_sha256"].items():
        old.require_hash(out / "code_snapshot" / name, sha)
        if file_sha256(ROOT / name) != sha:
            raise ValueError(f"Training code changed after this model was trained: {name}")
    model = out / data.target_name
    for name, sha in run["outputs_sha256"].items():
        old.require_hash(model / name, sha)
    summary = json.loads((model / "training_summary.json").read_text(encoding="utf-8"))
    names = summary["selected_feature_names"]
    if names != data.feature_names[:len(names)] or summary["config"] != asdict(cfg) or summary["report"] != data.report:
        raise ValueError("Feature order, recipe or preparation report changed")
    if summary["model_config"]["source_history_contract"] != selected_contract(data.source_history_contract, names):
        raise ValueError("Fitted history/fill contract changed")
    if summary["final_holdout"] != t.plan_final_holdout(data, cfg, summary["context_rows"]):
        raise ValueError("Purged final holdout changed")
    scaler = joblib.load(model / "scaler.joblib")
    fit_X = data.dev_X[:, :len(names)] if cfg.final_refit_on_dev else data.dev_X[:summary["final_holdout"]["fit_end"], :len(names)]
    expected = t.fit_scaler(fit_X, cfg)
    np.testing.assert_array_equal(scaler.scaler.center_, expected.center_)
    np.testing.assert_array_equal(scaler.scaler.scale_, expected.scale_)
    trials = pd.read_csv(model / "optuna_trials.csv")
    if len(trials) != cfg.n_trials or not set(trials.state) <= {"COMPLETE", "PRUNED"} or not (trials.state == "COMPLETE").any():
        raise ValueError("Search budget incomplete or trials failed")
    best = trials.loc[trials.number.eq(summary["study_best_trial"])].iloc[0]
    if best.state != "COMPLETE" or not np.isclose(best.value, trials.loc[trials.state.eq("COMPLETE"), "value"].max()):
        raise ValueError("Best trial selection mismatch")
    for name, value in summary["best_params"].items():
        if not np.isclose(best["params_" + name], value):
            raise ValueError("Saved best parameters differ from the selected trial")
    context = summary["context_rows"]
    purge = t.resolve_dataset_cv_purge_bars(data, cfg, context_rows=context)
    folds = t.PurgedWalkForwardSplitter(max(cfg.cv_initial_train_rows, context + 1), cfg.cv_val_rows,
        cfg.cv_step_rows, purge, cfg.cv_max_train_rows, cfg.cv_min_folds).split(len(data.dev_y))
    if summary["cv_splitter"]["fold_plans"] != [asdict(f) for f in folds]:
        raise ValueError("Saved CV geometry mismatch")
    t.validate_frozen_selection(data, folds)
    oof = pd.read_csv(model / "oof_predictions.csv")
    np.testing.assert_array_equal(oof.source_row_idx, data.dev_source_row_idx)
    fold_ids = np.full(len(data.dev_y), -1)
    for f in folds:
        fold_ids[f.val_start:f.val_end] = f.fold
    np.testing.assert_array_equal(oof.fold, fold_ids)
    valid = oof.oof_raw_probability.notna()
    np.testing.assert_array_equal(valid, fold_ids >= 0)
    raw_oof = oof.loc[valid, "oof_raw_probability"].to_numpy(np.float32)
    expected_calibration = t.resolve_calibration_method(cfg, data.target_name)
    calibration = summary["calibration"]
    if (calibration["requested_method"] != str(cfg.calibration_method).strip().lower()
            or calibration["resolved_method"] != expected_calibration
            or bool(calibration.get("disabled_for_continuation_target", False)) != ("continuation" in data.target_name.lower())):
        raise ValueError("Saved calibration contract does not match the target recipe")
    calibrator = joblib.load(model / "calibrator.joblib") if expected_calibration != "none" else None
    calibrated_oof = t.apply_calibrator(calibrator, raw_oof)
    if "oof_calibrated_probability" in oof:
        np.testing.assert_allclose(calibrated_oof, oof.loc[valid, "oof_calibrated_probability"].to_numpy(np.float32), rtol=1e-6, atol=1e-6)
    threshold, _ = t.select_operating_threshold(y_true=data.dev_y[valid], y_score=calibrated_oof,
        tolerance=cfg.event_tolerance_bars, cooldown=cfg.event_cooldown_bars, grid_size=cfg.threshold_grid_size, config=cfg)
    if not np.isclose(threshold, summary["threshold"]):
        raise ValueError("Threshold was not reselected from calibrated OOF predictions only")
    if cfg.backend == "torch":
        import torch
        checkpoint = torch.load(model / "model.pt", map_location="cpu", weights_only=True)
        if (checkpoint["selection_mode"] != "fixed_duration_refit" or checkpoint["best_val_auprc"] is not None
                or checkpoint["best_epoch"] != len(summary["refit_fixed_epoch_schedule"])):
            raise ValueError("TCN final refit used validation or changed duration")
    asset = "EURUSD" if model_id in v2.OTE_IDS else "ES"
    runner = runtime(model, summary, asset)
    raw = pd.read_parquet(data.prepared_dir / "source_history.parquet", columns=["source_row_idx", "datetime", *names])
    predictions = [runner.predict_latest(raw.iloc[int(end) - context:int(end) + 1]) for end in data.test_source_row_idx]
    raw_scores = np.asarray([p.raw_score for p in predictions])
    calibrated = np.asarray([p.calibrated_probability for p in predictions])
    saved = pd.read_csv(model / "test_predictions.csv")
    np.testing.assert_array_equal(saved.source_row_idx, data.test_source_row_idx)
    np.testing.assert_allclose(raw_scores, saved.raw_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(calibrated, saved.calibrated_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(calibrated >= threshold, saved.predicted_label)
    return dict(status="v2_artifact_review_passed", model_id=model_id, target=data.target_name, backend=cfg.backend,
        asset=asset, feature_contract="v2", selected_features=len(names), prediction_rows=len(saved), threshold=float(threshold),
        test_positive_predictions=int(saved.predicted_label.sum()),
        prediction_max_abs_error=float(np.max(np.abs(raw_scores - saved.raw_probability))),
        calibrated_prediction_max_abs_error=float(np.max(np.abs(calibrated - saved.calibrated_probability))),
        threshold_decisions_equal=True, OOF_threshold_reproduced=True, scaler_fit_scope_verified=True,
        raw_frozen_fill_runtime_replay=True, selected_feature_order_verified=True, purged_holdout_and_cv_verified=True,
        training_manifest_sha256=file_sha256(run_path), model_outputs_sha256=run["outputs_sha256"],
        prepared_manifest_sha256=run["prepared_manifest_sha256"], code_sha256=run["code_sha256"],
        reviewer_sha256=file_sha256(Path(__file__)), B1_evaluation_complete=False,
        actual_live_producer_parity_complete=False, collection_eligible=False, promotion_eligible=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=int, choices=(*v2.ES_IDS, *v2.OTE_IDS))
    parser.add_argument("--prepared-root", type=Path, default=v2.DEFAULT_PREPARED)
    parser.add_argument("--output-root", type=Path, default=v2.DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Use a new report path")
    result = review(args.model, args.prepared_root.resolve(), args.output_root.resolve())
    if args.report:
        v2.dump_new(args.report, result)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
