"""Verify completed priority or exploratory artifacts and replay predictions without fitting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_b2_retraining as runner
from scripts.prepare_b2_validation_views import PRIORITY_MANIFEST, PRIORITY_MANIFEST_SHA256
from model_training.ote_training import ote_xgboost_pipeline as t
from model_testing.artifact_lineage_audit import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contains_saved(actual, saved):
    """New preflight diagnostics may add fields without rewriting historical runs."""
    if isinstance(saved, dict):
        return isinstance(actual, dict) and all(k in actual and contains_saved(actual[k], v) for k, v in saved.items())
    if isinstance(saved, list):
        return isinstance(actual, list) and len(actual) == len(saved) and all(contains_saved(a, b) for a, b in zip(actual, saved))
    return actual == saved


def probability_metrics(y, probability, weight):
    return dict(average_precision=t.safe_average_precision(y, probability, sample_weight=weight),
                roc_auc=t.safe_roc_auc(y, probability, sample_weight=weight),
                brier=float(t.brier_score_loss(y, probability, sample_weight=weight)),
                weighted_positive_rate=float(np.average(y, weights=weight)))


def replay_predictions(dataset, config, summary, output):
    count, context = len(summary["selected_feature_names"]), summary["context_rows"]
    matrix = np.concatenate([dataset.dev_X[-context:, :count], dataset.X_test[:, :count]]) if context else dataset.X_test[:, :count]
    matrix = t.transform_matrix(joblib.load(output / "scaler.joblib"), matrix, clip_value=config.scale_clip)
    if config.backend == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(output / "model.json")
        X = t.build_sparse_lag_view(matrix, lag_steps=summary["lag_steps"], delta_feature_count=summary["delta_feature_count"])
        return t.sigmoid(model.predict(t.make_prediction_dmatrix(X)))
    import torch
    from model_training.ote_training import torch_trainer
    torch_trainer.configure_torch_runtime(device=torch_trainer.choose_device(),
        allow_tf32=config.torch_allow_tf32, cudnn_benchmark=config.torch_cudnn_benchmark)
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    require(checkpoint["selection_mode"] == "fixed_duration_refit", "TCN used validation for refit selection")
    require(checkpoint["best_val_auprc"] is None, "TCN refit reports a validation score")
    require(checkpoint["best_epoch"] == len(summary["refit_fixed_epoch_schedule"]), "TCN duration changed")
    return torch_trainer.predict_torch_model(
        model=torch_trainer.load_torch_model_from_checkpoint(checkpoint),
        X=t.SequenceWindowSource(matrix=matrix, window_size=summary["window_size"]),
        batch_size=config.batch_size, preload_to_device=config.torch_preload_to_device,
        use_amp=bool(checkpoint.get("amp_enabled", config.use_amp)))


def review(output_root, group="priority"):
    if group == "priority":
        manifest_path, manifest_hash = PRIORITY_MANIFEST, PRIORITY_MANIFEST_SHA256
        runner.require_hash(manifest_path, manifest_hash)
        manifest = json.loads(manifest_path.read_text())
        model_ids = runner.PRIORITY_IDS
    elif group == "exploratory":
        manifest_path, manifest_hash = runner.MANIFEST, runner.MANIFEST_SHA256
        manifest = runner.load_manifest()
        model_ids = runner.EXPLORATORY_IDS
    else:
        raise ValueError("Review supports priority or exploratory continuation targets only.")
    inputs = {str(manifest_path.relative_to(ROOT)): manifest_hash, **manifest["parents"],
              str(runner.EVENTS.relative_to(ROOT)): runner.EVENTS_SHA256,
              "docs/audits/b2_repaired_model_data_roster_20260921.csv": manifest["roster_sha256"]}
    if group == "exploratory":
        inputs.update({str(runner.QUEUE_POLICY.relative_to(ROOT)): runner.QUEUE_POLICY_SHA256,
                       str(runner.FIRST_PASS_REVIEW.relative_to(ROOT)): runner.FIRST_PASS_REVIEW_SHA256})
    for name, expected in inputs.items():
        runner.require_hash(ROOT / name, expected)
    results = []
    for model_id in model_ids:
        dataset, config, gaps = runner.load_job(manifest, model_id, output_root.resolve())
        checked = runner.check_job(dataset, config, model_id)
        run_path = Path(config.output_root) / "b2_retraining_manifest.json"
        run = json.loads(run_path.read_text())
        require(run["status"] == "trained_pending_B1_evaluation", f"Model {model_id} is incomplete")
        require(run["model_id"] == model_id, "Run model ID mismatch")
        require(run["validation_manifest_sha256"] == manifest_hash, "Run used different preparation")
        if group == "exploratory":
            require(run["queue_policy_sha256"] == runner.QUEUE_POLICY_SHA256, "Run queue policy mismatch")
            require(run["first_pass_review_sha256"] == runner.FIRST_PASS_REVIEW_SHA256, "Run first-pass review mismatch")
            require(run["geometry_adjustment"] is None, "Exploratory recipe geometry changed")
        require(run["lineage"] == manifest["models"][str(model_id)], "Run lineage mismatch")
        require(run["config"] == asdict(config), "Run recipe or seed differs from saved recipe")
        require(contains_saved(checked, run["validation_preflight"]), "Saved preflight changed")
        require(run["corrected_events_sha256"] == runner.EVENTS_SHA256, "Event lineage mismatch")
        require(all(run[k] == v for k, v in gaps.items()), "Run gap mismatch")
        output = Path(run["output_dir"])
        require(output == Path(config.output_root) / dataset.target_name, "Unexpected run output path")
        inputs[str(run_path.relative_to(ROOT))] = digest(run_path)
        for name, expected in run["outputs_sha256"].items():
            runner.require_hash(output / name, expected)
            inputs[str((output / name).relative_to(ROOT))] = expected
        code_changes = [name for name, expected in run["code_sha256"].items() if digest(ROOT / name) != expected]
        require(not any(not name.startswith("scripts/") for name in code_changes), "Fitting/preprocessing code changed since run")
        summary = json.loads((output / "training_summary.json").read_text())
        require(summary["config"] == run["config"], "Summary recipe mismatch")
        require(summary["report"] == dataset.report, "Embedded preparation report mismatch")
        trials = pd.read_csv(output / "optuna_trials.csv")
        states = {str(k): int(v) for k, v in trials.state.value_counts().items()}
        require(len(trials) == config.n_trials and states.get("COMPLETE", 0) > 0, "Search incomplete")
        require(set(states) <= {"COMPLETE", "PRUNED"}, "Failed or unfinished trials")
        best = trials.loc[trials.number.eq(summary["study_best_trial"])].iloc[0]
        require(best.state == "COMPLETE" and np.isclose(best.value, summary["study_best_value"]), "Best trial mismatch")
        require(np.isclose(best.value, trials.loc[trials.state.eq("COMPLETE"), "value"].max()), "Wrong best trial")
        for name, value in summary["best_params"].items():
            require(np.isclose(best["params_" + name], value), "Best parameters mismatch")
        names = summary["selected_feature_names"]
        require(names == dataset.feature_names[:len(names)] and bool(names), "Feature ordering mismatch")
        context = summary["context_rows"]
        purge = t.resolve_dataset_cv_purge_bars(dataset, config, context_rows=context)
        folds = t.PurgedWalkForwardSplitter(max(config.cv_initial_train_rows, context + 1), config.cv_val_rows,
            config.cv_step_rows, purge, config.cv_max_train_rows, config.cv_min_folds).split(len(dataset.dev_y))
        require(summary["cv_splitter"]["fold_plans"] == [asdict(f) for f in folds], "Fold geometry mismatch")
        t.validate_frozen_selection(dataset, folds)
        fold_table = pd.read_csv(output / "cv_fold_manifest.csv")
        require(len(fold_table) == len(folds), "Missing fold records")
        expected_fold = np.full(len(dataset.dev_y), -1)
        for f, row in zip(folds, fold_table.to_dict("records")):
            require(row["fold"] == f.fold and row["train_end_row"] == f.train_end - 1
                    and row["val_start_row"] == f.val_start and row["val_end_row"] == f.val_end - 1,
                    "Fold record boundaries changed")
            expected_fold[f.val_start:f.val_end] = f.fold  # Overlaps retain the last fold, as in trainer.
        holdout = t.plan_final_holdout(dataset, config, context)
        require(summary["final_holdout"] == holdout, "Final holdout mismatch")
        history = json.loads((output / "training_history.json").read_text())
        if config.backend == "torch":
            schedule = summary["refit_fixed_epoch_schedule"]
            require(schedule == summary["holdout_training_history"][:len(schedule)], "TCN schedule differs from holdout")
            require(len(history) == len(schedule) and all(h.get("checkpoint_selection") is False for h in history), "TCN refit selection")
            for actual, planned in zip(history, schedule):
                require(all(actual[k] == planned[k] for k in ("phase", "train_fraction", "learning_rate")), "TCN schedule not replayed")
        else:
            require(all(h["eval_rows"] == 0 for h in history), "XGB refit reused evaluation rows")
        if "_ict_" in dataset.target_name:
            require(summary["event_window_source"] == str(runner.EVENTS) and summary["event_window_count"] == len(dataset.event_window_lookup), "ICT event history mismatch")
            require(fold_table.sequential_bootstrap_applied.all(), "ICT bootstrap missing in a fold")
            for name in ("fit_selection_diagnostics", "final_refit_selection_diagnostics"):
                require(summary[name]["sequential_bootstrap_applied"], "ICT final bootstrap missing")
        oof = pd.read_csv(output / "oof_predictions.csv")
        test = pd.read_csv(output / "test_predictions.csv")
        for frame, source, y, w in ((oof, dataset.dev_source_row_idx, dataset.dev_y, dataset.dev_w),
                                    (test, dataset.test_source_row_idx, dataset.y_test, dataset.w_test)):
            np.testing.assert_array_equal(frame.source_row_idx, source)
            np.testing.assert_array_equal(frame.target, y)
            np.testing.assert_allclose(frame.sample_weight, w, rtol=1e-6, atol=0)
        np.testing.assert_array_equal(oof.fold, expected_fold)
        mask = expected_fold >= 0
        for raw, calibrated, frame, selected in (("oof_raw_probability", "oof_calibrated_probability", oof, mask),
                                                ("raw_probability", "calibrated_probability", test, np.ones(len(test), dtype=bool))):
            require(np.isfinite(frame.loc[selected, raw]).all() and frame.loc[selected, raw].between(0, 1).all(), "Invalid probabilities")
            require(frame.loc[~selected, raw].isna().all(), "Unexpected OOF predictions")
            np.testing.assert_array_equal(frame[raw], frame[calibrated])
        require(summary["calibration"]["resolved_method"] == "none" and not (output / "calibrator.joblib").exists(), "Continuation calibration changed")
        replay = replay_predictions(dataset, config, summary, output)
        np.testing.assert_allclose(replay, test.raw_probability, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(replay >= summary["threshold"], test.predicted_label)
        np.testing.assert_array_equal(test.predicted_label, test.calibrated_probability.ge(summary["threshold"]).astype(int))
        threshold, _ = t.select_operating_threshold(y_true=dataset.dev_y[mask],
            y_score=oof.loc[mask, "oof_calibrated_probability"].to_numpy(dtype=np.float32),
            tolerance=config.event_tolerance_bars, cooldown=config.event_cooldown_bars,
            grid_size=config.threshold_grid_size, config=config)
        require(np.isclose(threshold, summary["threshold"]), "Threshold cannot be reproduced from OOF only")
        test_metrics = probability_metrics(dataset.y_test, test.raw_probability.to_numpy(dtype=np.float32), dataset.w_test)
        for name in ("average_precision", "roc_auc", "brier"):
            require(np.isclose(test_metrics[name], summary["test_metrics"][name], atol=1e-6), "Saved test metric mismatch")
        development_prevalence = float(np.average(dataset.dev_y, weights=dataset.dev_w))
        diagnostics = dict(
            test_ap_lift_over_prevalence=test_metrics["average_precision"] / test_metrics["weighted_positive_rate"],
            development_weighted_prevalence=development_prevalence,
            test_brier_constant_development_prevalence=float(t.brier_score_loss(dataset.y_test,
                np.full(len(dataset.y_test), development_prevalence), sample_weight=dataset.w_test)),
            test_positive_rows=int(dataset.y_test.sum()), test_predicted_positive_rows=int(test.predicted_label.sum()),
            test_row_precision=float(test.loc[test.predicted_label.eq(1), "target"].mean()) if test.predicted_label.any() else None,
            cv_fold_auc_above_half=int(fold_table.roc_auc.gt(.5).sum()),
            cv_fold_auc_min=float(fold_table.roc_auc.min()), cv_fold_auc_max=float(fold_table.roc_auc.max()),
            event_tolerance_retained_rows=config.event_tolerance_bars,
            event_cooldown_retained_rows=config.event_cooldown_bars)
        results.append(dict(id=model_id, target=dataset.target_name, artifact_dir=str(output.relative_to(ROOT)),
            status="artifact_review_passed_pending_B1", trials=states, folds=len(folds), features=len(names),
            oof_rows=int(mask.sum()), test_rows=len(test), calibration=summary["calibration"],
            threshold=summary["threshold"], cv_summary=summary["cv_summary"], diagnostics=diagnostics,
            saved_test_event_metrics={k: v for k, v in summary["test_metrics"].items()
                                      if k.startswith("event_") or k in ("predicted_events", "true_events", "matched_events")},
            test_metrics=test_metrics, oof_metrics=probability_metrics(dataset.dev_y[mask],
                oof.loc[mask, "oof_raw_probability"].to_numpy(dtype=np.float32), dataset.dev_w[mask]),
            minimum_cv_source_gap=min(int(dataset.dev_source_row_idx[f.val_start] - dataset.dev_source_row_idx[f.train_end - 1] - 1) for f in folds),
            final_holdout=holdout, tcn_refit_epochs=len(history) if config.backend == "torch" else None,
            inference_replay_max_abs_error=float(np.max(np.abs(replay - test.raw_probability))),
            code_changes_since_run=code_changes, outputs_verified=len(run["outputs_sha256"])))
        print(f"{model_id:02d}: artifacts and prediction replay passed; {states}; test AUC={test_metrics['roc_auc']:.4f}", flush=True)
    return dict(status="first_pass_artifact_review_complete" if group == "priority" else "exploratory_artifact_review_complete",
                group=group, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                models=results, inputs_sha256=inputs, review_code_sha256=digest(Path(__file__)),
                B1_evaluation_complete=False, trusted_baseline=False, trading_model_training_run=False,
                interpretation="Historical label diagnostics only; no clean-period economics or promotion clearance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=runner.OUTPUT)
    parser.add_argument("--group", choices=("priority", "exploratory"), default="priority")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = review(args.output_root, args.group)
    with args.report.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
