"""Hash-checked preflight and individual training for the retained B2 repair queue."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ict.reports.leakage_control import build_ict_event_window_frame
from model_testing.artifact_lineage_audit import digest
from model_training.ote_training import ote_xgboost_pipeline as t

MANIFEST = ROOT / "artifacts/b2_validation_prepared_20260922_v2/manifest.json"
MANIFEST_SHA256 = "e99d03f09d543efdf809e3c0a315e85bc1c0be6ae463223f821327cd2e2b1936"
EVENTS = ROOT / "artifacts/b2_ict_labels_20260921_v1/ict_es_events.csv"
EVENTS_SHA256 = "e54b865f050e9172281962d2a1afc0e712a854bf201acc7b763c4daaa40ccc98"
PRIORITY_IDS = (2, 1, 11, 17)
EXPLORATORY_IDS = (6, 7)
BACKGROUND_IDS = (10, 3, 4, 8, 5, 9, 13, 12, 15, 16, 14, 18, 19)
OTE_IDS = (22, 26, 21, 25, 20, 24, 23, 27)
GROUPS = dict(priority=PRIORITY_IDS, exploratory=EXPLORATORY_IDS, background=BACKGROUND_IDS, ote=OTE_IDS,
              later=EXPLORATORY_IDS + BACKGROUND_IDS + OTE_IDS, all=tuple(range(1, 28)))
FIRST_PASS_REVIEW = ROOT / "docs/audits/b2_first_pass_review_20260922_v2.json"
FIRST_PASS_REVIEW_SHA256 = "98bab1b723c6444121dfab983882f9a69cc68bc69fe6556e162c268c44071762"
QUEUE_POLICY = ROOT / "configs/b2_retraining_queue_20260922.json"
QUEUE_POLICY_SHA256 = "e59fbba3795192c2da94067d6b7346cd7b9a450ee276d5d3d87a543345afd1f1"
OUTPUT = ROOT / "models/b2_retrained_20260922_v2"
CODE_FILES = ("preprocessing/config.py", "preprocessing/pipeline.py", "preprocessing/feature_selection.py",
              "preprocessing/feature_importance.py", "model_training/ote_training/ote_xgboost_pipeline.py",
              "model_training/ote_training/torch_trainer.py", "scripts/run_b2_retraining.py",
              "scripts/prepare_b2_validation_views.py", "scripts/materialize_frvp_recency_prepared_root.py",
              "scripts/review_b2_first_pass.py")


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Input changed; review lineage before training: {path}")


def load_manifest():
    require_hash(QUEUE_POLICY, QUEUE_POLICY_SHA256)
    require_hash(MANIFEST, MANIFEST_SHA256)
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "prepared_and_checked":
        raise ValueError("Validation preparation is incomplete.")
    for path, expected in manifest["parents"].items():
        require_hash(ROOT / path, expected)
    require_hash(ROOT / "docs/audits/b2_repaired_model_data_roster_20260921.csv", manifest["roster_sha256"])
    require_hash(EVENTS, EVENTS_SHA256)
    if set(manifest["models"]) != {str(i) for i in GROUPS["all"]}:
        raise ValueError("Expected pinned preparation for all 27 retained recipes.")
    require_hash(FIRST_PASS_REVIEW, FIRST_PASS_REVIEW_SHA256)
    review = json.loads(FIRST_PASS_REVIEW.read_text())
    if review["status"] != "first_pass_artifact_review_complete" or {m["id"] for m in review["models"]} != set(PRIORITY_IDS):
        raise ValueError("First-pass artifact review is incomplete.")
    for name, expected in review["inputs_sha256"].items():
        require_hash(ROOT / name, expected)
    return manifest


def load_job(manifest, model_id, output_root=OUTPUT):
    if model_id not in GROUPS["all"] or str(model_id) not in manifest["models"]:
        raise ValueError("Model is outside the pinned retained preparation queue.")
    job = manifest["models"][str(model_id)]
    prepared, recipe = ROOT / job["prepared_dir"], ROOT / job["recipe"]
    require_hash(recipe, job["recipe_sha256"])
    require_hash(prepared.parent / "summary.json", job["summary_sha256"])
    for name, expected in job["prepared_hashes"].items():
        require_hash(prepared / name, expected)
    config = t.OTETrainingConfig(**json.loads(recipe.read_text())["config"])
    apply_geometry_adjustment(config, model_id)
    config.prepared_root = str(prepared.parent)
    config.targets = [prepared.name]
    config.output_root = str(output_root / f"{model_id:02d}_{recipe.parent.name}")
    dataset = t.load_prepared_target_dataset(prepared.parent, prepared.name, config)
    contract = dataset.report.get("selection_contract", {})
    if contract.get("mode") != "frozen_development_prefix":
        raise ValueError("B2 requires frozen preparation from the complete pre-selection feature pool.")
    if contract["fit_rows"] != job["selection_fit_rows"]:
        raise ValueError("Frozen selection boundary differs from the checked preparation manifest.")
    if not dataset.feature_names or any(n.startswith("htf_confluence_") for n in dataset.feature_names):
        raise ValueError("Missing features or forbidden label-derived predictors.")
    source = dataset.dev_source_row_idx
    if not np.all(source >= 0) or not np.all(np.diff(source) > 0):
        raise ValueError("Development source rows must be strictly chronological.")
    raw_gap = int(dataset.report["split_geometry"]["applied_embargo_bars"])
    if "_ict_" not in prepared.name and raw_gap < 1071:
        raise ValueError("FRVP and OTE require at least 1,071 original bars of label separation.")
    if int(contract["required_gap_bars"]) != raw_gap:
        raise ValueError("Selection gap differs from the repaired label gap.")
    for left, right in ((dataset.source_row_idx_train, dataset.source_row_idx_val),
                        (dataset.source_row_idx_val, dataset.source_row_idx_test)):
        if not len(left) or not len(right) or int(right[0] - left[-1] - 1) < raw_gap:
            raise ValueError("Outer split violates the repaired original-bar gap.")
    if not np.all(np.diff(dataset.source_row_idx_test) > 0):
        raise ValueError("Test source rows must be strictly chronological.")
    if model_id == 10:
        validate_recency_weights(job, prepared, dataset.prepared_summary)
    retained_gap = int(np.max(np.searchsorted(source, source + raw_gap, side="right") - np.arange(len(source)) - 1))
    minimum_base = max(config.label_max_holding_bars,
                       config.label_exclusion_pre_bars + config.label_zone_pre_bars,
                       config.event_tolerance_bars + config.event_cooldown_bars)
    config.purge_buffer_bars = max(config.purge_buffer_bars, retained_gap - minimum_base)
    if "_ict_" in prepared.name:
        events = build_ict_event_window_frame(pd.read_csv(EVENTS))
        selected = events.loc[events.target_name.eq(prepared.name)]
        dataset.event_window_lookup = {int(k): int(v) for k, v in selected.groupby("signal_index").barrier_end_index.max().items()}
        dataset.event_window_source = str(EVENTS)
        if not dataset.event_window_lookup or not t.resolve_sequential_bootstrap_enabled(dataset=dataset, config=config):
            raise ValueError("Corrected ICT event history must enable sequential bootstrap.")
        missing = set(source[dataset.dev_y > 0]) - set(dataset.event_window_lookup)
        if missing:
            raise ValueError(f"Missing corrected ICT event windows for {len(missing)} positive rows.")
    return dataset, config, dict(raw_bar_gap=raw_gap, retained_row_gap=retained_gap)


def apply_geometry_adjustment(config, model_id):
    require_hash(QUEUE_POLICY, QUEUE_POLICY_SHA256)
    policy = json.loads(QUEUE_POLICY.read_text())
    adjustment = policy["geometry_adjustments"].get(str(model_id))
    if adjustment:
        for key, value in adjustment["effective"].items():
            if (key not in ("cv_initial_train_rows", "cv_val_rows")
                    or getattr(config, key) != adjustment["saved"][key]
                    or value < getattr(config, key)):
                raise ValueError("Invalid or stale CV geometry adjustment.")
            setattr(config, key, value)


def validate_recency_weights(job, prepared, summary):
    contract = job.get("recency_weighting") or {}
    saved = summary.get("recency_weighting") or {}
    detail = json.loads((prepared / "recency_weighting_summary.json").read_text())
    for record in (contract, saved, detail):
        if record.get("half_life_days") != 730 or record.get("floor") != .20:
            raise ValueError("FRVP long reversal requires 730-day half-life and 0.20 floor.")
    if len({record["latest_ts_utc"] for record in (contract, saved, detail)}) != 1:
        raise ValueError("FRVP recency reference timestamps differ.")
    columns = ["source_row_idx", "target", "sample_weight"]
    for split in ("train", "val", "test"):
        before = pd.read_csv(ROOT / job["old_prepared_dir"] / f"{split}.csv", usecols=columns)
        after = pd.read_csv(prepared / f"{split}.csv", usecols=columns)
        pd.testing.assert_frame_equal(before[columns], after[columns], check_exact=True)


def check_job(dataset, config, model_id):
    t.validate_backend_config(config)
    t.validate_cv_geometry_for_dataset(dataset, config)
    windows = range(config.window_min, config.window_max + 1, 4)
    source, y = dataset.dev_source_row_idx, dataset.dev_y
    raw_gap = int(dataset.report["split_geometry"]["applied_embargo_bars"])
    checks = []
    for window in windows:
        context = window - 1
        purge = t.resolve_dataset_cv_purge_bars(dataset, config, context_rows=context)
        folds = t.PurgedWalkForwardSplitter(max(config.cv_initial_train_rows, context + 1),
            config.cv_val_rows, config.cv_step_rows, purge, config.cv_max_train_rows, config.cv_min_folds).split(len(y))
        t.validate_frozen_selection(dataset, folds)
        gaps = [int(source[f.val_start] - source[f.train_end - 1] - 1) for f in folds]
        if min(gaps) < raw_gap:
            raise ValueError("A walk-forward fold violates the repaired original-bar gap.")
        counts = []
        for fold in folds:
            # Match the trainer: rolling cap applies to model rows, context precedes it.
            label_context = 0 if dataset.source_history_X is not None else context
            train_start = max(0, fold.train_start - label_context)
            train_y = y[train_start + label_context:fold.train_end]
            val_y = y[fold.val_start:fold.val_end]
            count = dict(fold=fold.fold, train_positive_rows=int(train_y.sum()),
                         val_positive_rows=int(val_y.sum()), val_true_events=len(t.group_positive_zones(val_y)))
            if count["train_positive_rows"] < config.min_train_positive_rows or count["val_positive_rows"] < config.min_val_positive_rows or count["val_true_events"] < config.min_val_true_events:
                raise ValueError(f"Saved positive/event safeguards fail: model {model_id}, window {window}, {count}")
            counts.append(count)
        final_holdout = t.plan_final_holdout(dataset, config, context)
        final_counts = dict(train_positive_rows=int(y[label_context:final_holdout["fit_end"]].sum()),
            val_positive_rows=int(y[final_holdout["val_start"]:].sum()),
            val_true_events=len(t.group_positive_zones(y[final_holdout["val_start"]:])))
        if (final_counts["train_positive_rows"] < config.min_train_positive_rows
                or final_counts["val_positive_rows"] < config.min_val_positive_rows
                or final_counts["val_true_events"] < config.min_val_true_events):
            raise ValueError(f"Final holdout positive/event safeguards fail: model {model_id}, {final_counts}")
        checks.append(dict(window=window, folds=len(folds), minimum_source_gap=min(gaps),
                           selection_validation_overlap_rows=0, fold_counts=counts, final_holdout=final_holdout,
                           final_holdout_counts=final_counts))
    return dict(id=model_id, target=dataset.target_name, backend=config.backend, trials=config.n_trials,
                prepared_dir=str(dataset.prepared_dir.relative_to(ROOT)),
                selection_fit_rows=dataset.report["selection_contract"]["fit_rows"],
                loaded_features=len(dataset.feature_names), corrected_event_windows=len(dataset.event_window_lookup),
                final_refit_uses_validation=False, windows=checks,
                calibration_method=t.resolve_calibration_method(config, dataset.target_name),
                recency_weighting=dataset.prepared_summary.get("recency_weighting"),
                effective_cv_geometry={k: getattr(config, k) for k in (
                    "cv_initial_train_rows", "cv_val_rows", "cv_step_rows", "cv_max_train_rows", "cv_min_folds")},
                trusted_baseline=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train"))
    parser.add_argument("model", help="Check priority/exploratory/background/ote/later/all, or an individual retained model ID")
    parser.add_argument("--report", type=Path, help="Save read-only preflight evidence as JSON")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.report and args.mode != "check":
        parser.error("--report is reserved for preflight checks.")
    if args.mode == "train" and args.model in GROUPS:
        parser.error("Train one reviewed model at a time.")
    if args.mode == "train" and args.model == "6":
        parser.error("S3 retained-row training is retired. Use scripts/run_b2_source_history.py train for the corrected source-bar recipe.")
    if args.mode == "train" and args.model in {"2", "1", "11", "17", "7"}:
        parser.error("Candidate retained-row training is retired. Use scripts/run_b2_candidate_history.py train ID for the audited source-bar recipe.")
    if args.model not in GROUPS and (not args.model.isdigit() or int(args.model) not in GROUPS["all"]):
        parser.error("Choose a retained model ID (1-27) or a named check group.")
    ids = GROUPS[args.model] if args.model in GROUPS else [int(args.model)]
    manifest = load_manifest()
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  validation_manifest_sha256=MANIFEST_SHA256, corrected_events_sha256=EVENTS_SHA256,
                  first_pass_review_sha256=FIRST_PASS_REVIEW_SHA256,
                  queue_policy_sha256=QUEUE_POLICY_SHA256,
                  geometry_adjustments=json.loads(QUEUE_POLICY.read_text())["geometry_adjustments"],
                  code_sha256={name: digest(ROOT / name) for name in CODE_FILES}, models=[], errors=[],
                  trading_model_training_run=False, trusted_baseline=False)
    for model_id in ids:
        try:
            dataset, config, gaps = load_job(manifest, model_id, args.output_root.resolve())
            checked = check_job(dataset, config, model_id)
        except (ValueError, FileNotFoundError, AssertionError) as exc:
            if args.mode == "train":
                raise
            report["errors"].append(dict(id=model_id, error=str(exc)))
            print(f"{model_id:02d} BLOCKED: {exc}", flush=True)
            continue
        report["models"].append(checked)
        print(f"{model_id:02d} {dataset.target_name}: {len(checked['windows'])} windows checked; selection frozen at {checked['selection_fit_rows']} rows; purged final holdout checked; trials={config.n_trials}", flush=True)
        if args.mode == "check":
            continue
        out = Path(config.output_root)
        out.mkdir(parents=True, exist_ok=False)
        training = dict(status="training_started", model_id=model_id, config=asdict(config),
                        validation_preflight=checked, validation_manifest_sha256=MANIFEST_SHA256,
                        first_pass_review_sha256=FIRST_PASS_REVIEW_SHA256,
                        queue_policy_sha256=QUEUE_POLICY_SHA256,
                        geometry_adjustment=report["geometry_adjustments"].get(str(model_id)),
                        code_sha256=report["code_sha256"], corrected_events_sha256=EVENTS_SHA256,
                        lineage=manifest["models"][str(model_id)], trusted_baseline=False, **gaps)
        path = out / "b2_retraining_manifest.json"
        path.write_text(json.dumps(training, indent=2) + "\n")
        result = t.train_target_pipeline(dataset=dataset, config=config)
        training.update(status="trained_pending_B1_evaluation", output_dir=str(result["output_dir"]),
                        outputs_sha256={p.name: digest(p) for p in result["output_dir"].iterdir() if p.is_file()})
        path.write_text(json.dumps(training, indent=2) + "\n")
        print(f"Saved {result['output_dir']}; pending B1 evaluation.", flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
