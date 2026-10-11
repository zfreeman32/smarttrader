"""Versioned source-history repair for selected B2 collection candidates only.

Prepare/check do not fit models. Train refuses existing output directories.
Every candidate remains diagnostic-only pending its independent collection gates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from scripts import run_b2_retraining as old
from scripts.audit_b2_candidate_history import IDS, dump_new, runtime, source_frame
from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import MODE, file_sha256, indexed_lag_view, selected_contract

PREPARED = ROOT / "artifacts/b2_candidate_history_20260928_v1"
OUTPUT = ROOT / "models/b2_candidate_history_20260928_v1"
AUDIT = ROOT / "docs/audits/b2_candidate_history_20260928.json"
# The original five repairs remain supported.  The minimum-core ES branches
# are admitted first; expanded-core IDs 3 and 19 are admitted only after the
# minimum-core technical reviews have passed.
# Latency-approved extension IDs 5/14/18 are admitted only through the
# versioned catch-up roster and only into its own new roots.
CATCHUP = json.loads((ROOT / "configs/b2_candidate_history_catchup_roster_20261001.json").read_text())["es_runner"]
CATCHUP_IDS = tuple(c["id"] for c in CATCHUP["candidates"])
CANDIDATE_IDS = tuple(dict.fromkeys((*IDS, 3, 10, 12, 13, 15, 16, 19, *CATCHUP_IDS)))
CODE = tuple(dict.fromkeys((*old.CODE_FILES, "model_training/source_history.py",
    "model_training/ote_training/torch_models.py", "ote_live/models/loaders.py", "ote_live/models/runners.py",
    "ote_live/models/calibrators.py", "scripts/audit_b2_candidate_history.py", "scripts/run_b2_candidate_history.py")))
# These exact predecessor scripts were snapshotted by the first three fits.
# The only subsequent changes select the production CPU backend for review and
# record this reviewer revision. Fitting/preparation/library code must still match.
REVIEW_ONLY_PREDECESSORS = {
    "scripts/audit_b2_candidate_history.py": "fe30974354487e59c3cd2a78a39768e634e1b5c92fa0f33c8dd1cdfb8b9b521c",
    "scripts/run_b2_candidate_history.py": "3e1c957eed50537266eb464b0bd72997544b9f6ec004207a284e5cd3d358a0d7",
}
# This hash is the runner captured by the first minimum-core replacement
# before the calibration-aware reviewer fix.  It changes review/CLI behavior
# only; fitting, preparation and model-serving code are still hash-checked.
REVIEW_ONLY_PREDECESSORS["scripts/run_b2_candidate_history.py"] = "7bf2f61fc41e0afb4425f9a08e0c54dbe5ce50a689ae8f822252679dd9813116"
# Runner captured by expanded-core IDs 3/19 before the catch-up roster
# admitted IDs 5/14/18.  That change is CLI admission and a root guard only.
ADMISSION_ONLY_PREDECESSORS = {
    "scripts/run_b2_candidate_history.py": ("e4b19690ea4e32a3f8c6336b15e6773ab509b5bfd437a280dc71b274772f67ec",),
}


def require_catchup_roots(model_id, root, output, audit_path):
    if model_id not in CATCHUP_IDS:
        return
    for path, parent in ((root, "artifacts"), (output, "models")):
        if path.parent != ROOT / parent or not path.name.startswith(CATCHUP["root_prefix"]):
            raise ValueError("Catch-up IDs require their own new catch-up roots; earlier roots are frozen")
    if not audit_path.name.startswith(CATCHUP["audit_prefix"]):
        raise ValueError("Catch-up IDs require their own source-history audit report")


def prepare(model_id, root, audit_path):
    out = root / str(model_id)
    if out.exists() or not root.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Preparation requires a new candidate directory under artifacts")
    audit = json.loads(audit_path.read_text())
    reviewed = next(m for m in audit["models"] if m["id"] == model_id)
    if audit["status"] != "candidate_history_audit_complete" or reviewed["status"] not in {
        "affected_diagnostic_only", "source_history_repair_required"
    }:
        raise ValueError("Repair requires completed candidate-specific evidence")
    manifest = old.load_manifest()
    data, cfg, _ = old.load_job(manifest, model_id)
    job = manifest["models"][str(model_id)]
    frame, encoded, _, fills, source, source_sha = source_frame(manifest, model_id, data)
    prepared = out / "prepared" / data.target_name
    prepared.mkdir(parents=True)
    for name in job["prepared_hashes"]:
        shutil.copyfile(data.prepared_dir / name, prepared / name)
    shutil.copyfile(data.prepared_dir.parent / "summary.json", prepared.parent / "summary.json")
    tape = pd.concat([frame[["source_row_idx", "datetime"]], encoded], axis=1)
    tape.to_parquet(prepared / "source_history.parquet", index=False)
    fit_ids = data.source_row_idx_train[:job["selection_fit_rows"]]
    contract = dict(mode=MODE, bar_minutes=5, feature_names=data.feature_names, fill_values=fills,
        missing_value_rule="nonfinite_to_frozen_fill", selection_fit_rows=len(fit_ids),
        selection_fit_end_source_row_idx=int(fit_ids[-1]), source_dataset_sha256=source_sha,
        source_history_sha256=file_sha256(prepared / "source_history.parquet"),
        clock_note="Consecutive observed source bars; availability, scheduled gaps, closures and rolls are separate gates")
    dump_new(prepared / "source_history_contract.json", contract)
    dump_new(out / "manifest.json", dict(status="candidate_source_history_prepared", model_id=model_id,
        audit_sha256=file_sha256(audit_path), audit_path=str(audit_path.relative_to(ROOT)),
        validation_manifest_sha256=old.MANIFEST_SHA256,
        source_job=job, source_dataset=str(source.relative_to(ROOT)), source_dataset_sha256=source_sha,
        source_config=asdict(cfg), source_rows=len(frame),
        prepared_files_sha256={str(p.relative_to(out)): file_sha256(p) for p in out.rglob("*") if p.is_file()},
        collection_eligible=False, selected_for_offline_repair=True))
    print(f"ID {model_id}: prepared {len(frame)} source bars; exact labels, weights, splits and frozen feature order preserved", flush=True)


def load(model_id, root, output, audit_path):
    path = root / str(model_id)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["model_id"] != model_id or manifest["status"] != "candidate_source_history_prepared":
        raise ValueError("Wrong candidate preparation")
    if manifest["validation_manifest_sha256"] != old.MANIFEST_SHA256:
        raise ValueError("Wrong parent preparation")
    old.require_hash(audit_path, manifest["audit_sha256"])
    for name, sha in manifest["prepared_files_sha256"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path):
            raise ValueError("Invalid prepared path")
        old.require_hash(file, sha)
    old.require_hash(ROOT / manifest["source_dataset"], manifest["source_dataset_sha256"])
    parents = old.load_manifest()
    original, cfg, gaps = old.load_job(parents, model_id)
    if manifest["source_job"] != parents["models"][str(model_id)] or manifest["source_config"] != asdict(cfg):
        raise ValueError("Recipe or selection ancestry changed")
    cfg.prepared_root = str(path / "prepared")
    cfg.output_root = str(output / f"{model_id:02d}_{original.target_name}")
    data = t.load_prepared_target_dataset(path / "prepared", original.target_name, cfg)
    if data.source_history_X is None:
        raise ValueError("Refusing retained-row fallback")
    # Preserve ICT's corrected event windows and sequential-bootstrap behavior.
    data.event_window_lookup = original.event_window_lookup
    data.event_window_source = original.event_window_source
    return data, cfg, gaps


def preflight(data, cfg, model_id):
    from ote_live.models.runners import _build_sparse_lag_view, _build_sequence_windows
    result = old.check_job(data, cfg, model_id)
    scaler = t.fit_scaler(data.X_train[:data.source_history_contract["selection_fit_rows"]], cfg)
    indices = np.arange(len(data.feature_names))
    ends = np.concatenate([data.dev_source_row_idx[:8], data.dev_source_row_idx[-8:], data.test_source_row_idx])
    checks = []
    for window in range(cfg.window_min, cfg.window_max + 1, 4):
        if cfg.backend == "xgboost":
            lags = t.make_lag_steps(window, cfg.lag_count_max)
            expected = indexed_lag_view(data.source_history_X, ends, indices, scaler, lags, cfg.delta_feature_cap, cfg.scale_clip)
        else:
            expected = t.indexed_sequence_view(data.source_history_X, ends, indices, scaler, window, cfg.scale_clip)
        for n, end in enumerate(ends):
            scaled = t.transform_matrix(scaler, data.source_history_X[end-window+1:end+1], cfg.scale_clip)
            if cfg.backend == "xgboost":
                actual = _build_sparse_lag_view(scaled, lag_steps=lags, delta_feature_count=cfg.delta_feature_cap)[-1]
                np.testing.assert_array_equal(actual, expected[n])
            else:
                actual = _build_sequence_windows(scaled, window_size=window)[-1]
                start = expected.sampled_indices[n]
                np.testing.assert_array_equal(actual, expected.matrix[start:start+window])
        checks.append(dict(window=window, endpoints=len(ends), exact_training_runtime_tensor_equality=True))
    return dict(status="candidate_history_preflight_passed", model=result, tensor_checks=checks,
                collection_eligible=False)


def review(data, cfg, root, audit_path):
    out = Path(cfg.output_root)
    run_path = out / "candidate_history_training_manifest.json"
    run = json.loads(run_path.read_text())
    if run["status"] != "trained_pending_candidate_history_review" or run["config"] != asdict(cfg):
        raise ValueError("Incomplete or changed training recipe")
    if set(run["code_sha256"]) != set(CODE) or run["corrected_events_sha256"] != old.EVENTS_SHA256:
        raise ValueError("Incomplete code or corrected event lineage")
    old.require_hash(root / str(run["model_id"]) / "manifest.json", run["prepared_manifest_sha256"])
    for name, sha in run["code_sha256"].items():
        old.require_hash(out / "code_snapshot" / name, sha)
        if (file_sha256(ROOT / name) != sha and REVIEW_ONLY_PREDECESSORS.get(name) != sha
                and sha not in ADMISSION_ONLY_PREDECESSORS.get(name, ())):
            raise ValueError(f"Unreviewed training code change: {name}")
    model = out / data.target_name
    for name, sha in run["outputs_sha256"].items():
        old.require_hash(model / name, sha)
    summary = json.loads((model / "training_summary.json").read_text())
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
        if not np.isclose(best["params_"+name], value):
            raise ValueError("Saved best parameters differ from the selected trial")
    context = summary["context_rows"]
    purge = t.resolve_dataset_cv_purge_bars(data, cfg, context_rows=context)
    folds = t.PurgedWalkForwardSplitter(max(cfg.cv_initial_train_rows, context+1), cfg.cv_val_rows,
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
    requested_calibration = str(cfg.calibration_method).strip().lower()
    calibration = summary["calibration"]
    if (calibration["requested_method"] != requested_calibration
            or calibration["resolved_method"] != expected_calibration
            or bool(calibration.get("disabled_for_continuation_target", False))
            != ("continuation" in data.target_name.lower())):
        raise ValueError("Saved calibration contract does not match the target recipe")
    calibrator_path = model / "calibrator.joblib"
    calibrator = joblib.load(calibrator_path) if expected_calibration != "none" else None
    calibrated_oof = t.apply_calibrator(calibrator, raw_oof)
    if "oof_calibrated_probability" in oof:
        np.testing.assert_allclose(
            calibrated_oof,
            oof.loc[valid, "oof_calibrated_probability"].to_numpy(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
    threshold, _ = t.select_operating_threshold(y_true=data.dev_y[valid],
        y_score=calibrated_oof,
        tolerance=cfg.event_tolerance_bars, cooldown=cfg.event_cooldown_bars,
        grid_size=cfg.threshold_grid_size, config=cfg)
    if not np.isclose(threshold, summary["threshold"]):
        raise ValueError("OOF-only calibrated threshold mismatch")
    if cfg.backend == "torch":
        import torch
        checkpoint = torch.load(model / "model.pt", map_location="cpu", weights_only=True)
        if (checkpoint["selection_mode"] != "fixed_duration_refit" or checkpoint["best_val_auprc"] is not None
                or checkpoint["best_epoch"] != len(summary["refit_fixed_epoch_schedule"])):
            raise ValueError("TCN final refit used validation or changed duration")
    runner = runtime(model, summary)
    raw = pd.read_parquet(data.prepared_dir / "source_history.parquet")
    runtime_predictions = [runner.predict_latest(raw.iloc[end-context:end+1]) for end in data.test_source_row_idx]
    predictions = np.asarray([prediction.raw_score for prediction in runtime_predictions])
    calibrated_predictions = np.asarray(
        [prediction.calibrated_probability for prediction in runtime_predictions]
    )
    saved = pd.read_csv(model / "test_predictions.csv")
    np.testing.assert_array_equal(saved.source_row_idx, data.test_source_row_idx)
    np.testing.assert_allclose(predictions, saved.raw_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        calibrated_predictions,
        saved.calibrated_probability,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(calibrated_predictions >= threshold, saved.predicted_label)
    return dict(status="candidate_history_artifact_review_passed", model_id=run["model_id"],
        target=data.target_name, backend=cfg.backend, prediction_rows=len(saved), threshold=float(threshold),
        prediction_max_abs_error=float(np.max(np.abs(predictions-saved.raw_probability))),
        calibrated_prediction_max_abs_error=float(
            np.max(np.abs(calibrated_predictions-saved.calibrated_probability))
        ),
        threshold_decisions_equal=True, OOF_threshold_reproduced=True,
        scaler_fit_scope_verified=True, raw_frozen_fill_runtime_replay=True,
        selected_feature_order_verified=True, purged_holdout_and_cv_verified=True,
        training_manifest_sha256=file_sha256(run_path), model_outputs_sha256=run["outputs_sha256"],
        prepared_manifest_sha256=run["prepared_manifest_sha256"], code_sha256=run["code_sha256"],
        reviewer_code_sha256={name: file_sha256(ROOT/name) for name in CODE},
        reviewer_revision="Production backend loader and CPU TCN precision; fitting code and recipe unchanged",
        B1_evaluation_complete=False, actual_live_producer_parity_complete=False, collection_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "train", "review"))
    parser.add_argument("model", type=int, choices=CANDIDATE_IDS)
    parser.add_argument("--prepared-root", type=Path, default=PREPARED)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root, output = args.prepared_root.resolve(), args.output_root.resolve()
    audit_path = args.audit.resolve()
    if args.report and args.report.exists():
        parser.error("Use a new report path")
    require_catchup_roots(args.model, root, output, audit_path)
    if args.mode == "prepare":
        prepare(args.model, root, audit_path)
        return
    data, cfg, gaps = load(args.model, root, output, audit_path)
    if args.mode == "review":
        result = review(data, cfg, root, audit_path)
    else:
        result = preflight(data, cfg, args.model)
        print(f"ID {args.model}: all search windows, source tensors, frozen selection and purged holdout pass", flush=True)
        if args.mode == "train":
            out = Path(cfg.output_root)
            if not output.is_relative_to(ROOT / "models"):
                raise ValueError("Use a new output version under models")
            out.mkdir(parents=True, exist_ok=False)
            pins = {name: file_sha256(ROOT / name) for name in CODE}
            for name in CODE:
                dest = out / "code_snapshot" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            run = dict(status="training_started", model_id=args.model, config=asdict(cfg),
                prepared_manifest_sha256=file_sha256(root / str(args.model) / "manifest.json"),
                source_history_contract=data.source_history_contract, preflight=result,
                corrected_events_sha256=old.EVENTS_SHA256, code_sha256=pins,
                environment=dict(python=sys.version, omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                    packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost", "torch", "scikit-learn", "optuna", "joblib")}),
                collection_eligible=False, **gaps)
            path = out / "candidate_history_training_manifest.json"
            dump_new(path, run)
            trained = t.train_target_pipeline(data, cfg)
            run.update(status="trained_pending_candidate_history_review", outputs_sha256={
                p.name: file_sha256(p) for p in Path(trained["output_dir"]).iterdir() if p.is_file()})
            path.write_text(json.dumps(run, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            result = dict(status=run["status"], model_id=args.model, output_dir=str(trained["output_dir"]), collection_eligible=False)
    if args.report:
        dump_new(args.report, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
