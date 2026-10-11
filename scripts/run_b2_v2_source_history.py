"""Retrain R2c part 2 (prepare, check) and R3 (train): per-model source history on the feature contract v2 views.

Covers the 16 roster models in ``docs/audits/b2_v2_model_data_roster_20261006.csv``, XGBoost and TCN alike. The v1
runners (``run_b2_candidate_history.py`` for ES, ``run_b2_ote_candidate_history.py`` for EURUSD) stay unchanged: earlier
models' reviews pin their hashes. Their generic pieces are reused here: ``run_b2_retraining.load_job`` / ``check_job``
(recipe, CV geometry adjustment, gaps, ICT event windows, recency check), ``audit_b2_candidate_history.source_frame``
(ES tape + frozen fills + endpoint equality) and ``run_b2_candidate_history.preflight`` (training tensors == runtime
tensors for every search window and backend).

* ES: each model's tape is the v2 R2b ``dataset.parquet`` (every source bar, ``source_row_idx`` 0..N-1) for its loaded
  features, with fills fit on its frozen selection prefix.
* EURUSD: one union tape for the five OTE models. The 250 warm-up bars (in the label file, dropped from the feature
  tape) are prepended as unavailable rows so the earliest endpoints have causal context, and split ids shift by that
  offset (as in the v1 OTE runner). Fills are per model, fit on its own frozen prefix.

Only ``train`` fits a model. Nothing is installed or activated. Outputs are new versioned roots.

Usage: python scripts/run_b2_v2_source_history.py prepare|check|train ID [--prepared-root ...] [--output-root ...]
       python scripts/run_b2_v2_source_history.py prepare ote   (all five EURUSD models, shared tape)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import run_b2_retraining as old
from scripts.audit_b2_candidate_history import source_frame
from scripts.run_b2_candidate_history import preflight as source_history_preflight
from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import MODE, file_sha256
from preprocessing.feature_selection import build_fill_values

LOCK = ROOT / "configs/b2_v2_validation_views_lock_20261006.json"
DEFAULT_PREPARED = ROOT / "artifacts/b2_v2_source_history_20261006"
DEFAULT_OUTPUT = ROOT / "models/b2_v2_retrained_20261006"
OTE_LABELS = ROOT / "artifacts/b2_ote_labels_20260920_v2/labels.csv.gz"
ES_IDS = (2, 3, 5, 10, 12, 13, 14, 15, 16, 18, 19)
OTE_IDS = (21, 22, 23, 25, 26)
CODE = tuple(dict.fromkeys((*old.CODE_FILES, "model_training/source_history.py",
    "model_training/ote_training/torch_models.py", "ote_live/models/loaders.py", "ote_live/models/runners.py",
    "ote_live/models/calibrators.py", "scripts/audit_b2_candidate_history.py", "scripts/run_b2_candidate_history.py",
    "scripts/prepare_b2_v2_validation_views.py", "scripts/build_b2_v2_roster.py", "scripts/run_b2_v2_source_history.py")))


def dump_new(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_manifest() -> tuple[dict, str]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    path = ROOT / lock["manifest"]
    old.require_hash(path, lock["manifest_sha256"])
    old.require_hash(ROOT / lock["roster"], lock["roster_sha256"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "prepared_and_checked" or manifest["roster_sha256"] != lock["roster_sha256"]:
        raise ValueError("v2 validation views are incomplete.")
    for relative, sha in manifest["parents"].items():
        old.require_hash(ROOT / relative, sha)
    old.require_hash(old.QUEUE_POLICY, old.QUEUE_POLICY_SHA256)
    old.require_hash(old.EVENTS, old.EVENTS_SHA256)
    if sorted(manifest["models"], key=int) != [str(i) for i in (*ES_IDS, *OTE_IDS)] or lock["models"] != sorted(manifest["models"], key=int):
        raise ValueError("Expected the 16 roster models.")
    return manifest, lock["manifest_sha256"]


def copy_prepared(data, job: dict, prepared: Path) -> None:
    prepared.mkdir(parents=True)
    for name in job["prepared_hashes"]:
        shutil.copyfile(data.prepared_dir / name, prepared / name)
    shutil.copyfile(data.prepared_dir.parent / "summary.json", prepared.parent / "summary.json")


def write_model_manifest(model_id, out, manifest_sha, job, cfg, source, source_sha, rows, **extra) -> None:
    dump_new(out / "manifest.json", dict(status="v2_source_history_prepared", model_id=model_id,
        validation_manifest_sha256=manifest_sha, source_job=job, source_dataset=str(source.relative_to(ROOT)),
        source_dataset_sha256=source_sha, source_config=asdict(cfg), source_rows=rows, feature_contract="v2",
        prepared_files_sha256={str(p.relative_to(out)): file_sha256(p) for p in out.rglob("*") if p.is_file()},
        collection_eligible=False, trading_model_training_run=False, **extra))


def prepare_es(model_id: int, root: Path) -> None:
    out = root / str(model_id)
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}")
    manifest, manifest_sha = load_manifest()
    data, cfg, _ = old.load_job(manifest, model_id)
    job = manifest["models"][str(model_id)]
    frame, encoded, _, fills, source, source_sha = source_frame(manifest, model_id, data)
    prepared = out / "prepared" / data.target_name
    copy_prepared(data, job, prepared)
    pd.concat([frame[["source_row_idx", "datetime"]], encoded], axis=1).to_parquet(prepared / "source_history.parquet", index=False)
    fit_ids = data.source_row_idx_train[:job["selection_fit_rows"]]
    dump_new(prepared / "source_history_contract.json", dict(mode=MODE, bar_minutes=5, feature_names=data.feature_names,
        fill_values=fills, missing_value_rule="nonfinite_to_frozen_fill", selection_fit_rows=len(fit_ids),
        selection_fit_end_source_row_idx=int(fit_ids[-1]), source_dataset_sha256=source_sha,
        source_history_sha256=file_sha256(prepared / "source_history.parquet"),
        clock_note="Consecutive observed source bars; availability, scheduled gaps, closures and rolls are separate gates"))
    write_model_manifest(model_id, out, manifest_sha, job, cfg, source, source_sha, len(frame))
    print(f"ID {model_id}: {len(frame)} source bars x {len(data.feature_names)} loaded features; endpoints == prepared splits", flush=True)


def shift_ids(prepared: Path, offset: int) -> None:
    for split in ("train", "val", "test"):
        path = prepared / f"{split}.csv"
        frame = pd.read_csv(path)
        frame["source_row_idx"] = frame["source_row_idx"].astype(np.int64) + offset
        frame.to_csv(path, index=False)
    report_path = prepared / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    geometry = report.get("split_geometry", {})
    for bounds in geometry.get("split_source_row_ranges", {}).values():
        bounds["start_source_row_idx"] += offset
        bounds["end_source_row_idx"] += offset
    for boundary in geometry.get("boundaries", []):
        for key in ("current_split_last_source_row_idx", "next_split_first_source_row_idx"):
            if key in boundary:
                boundary[key] += offset
    selection = report.get("selection_contract", {})
    if "fit_end_source_row_idx" in selection:
        selection["fit_end_source_row_idx"] += offset
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def prepare_ote(root: Path) -> None:
    if any((root / str(i)).exists() for i in OTE_IDS) or (root / "source_history_union.parquet").exists():
        raise ValueError("OTE preparation exists in this root; use a new version.")
    manifest, manifest_sha = load_manifest()
    loaded = {i: old.load_job(manifest, i) for i in OTE_IDS}
    names = list(dict.fromkeys(name for data, _, _ in loaded.values() for name in data.feature_names))
    summary = json.loads((ROOT / Path(manifest["models"]["21"]["old_prepared_dir"]).parent / "summary.json").read_text())
    source = Path(summary["input_file"]).resolve()
    source_sha = manifest["parents"][str(source.relative_to(ROOT))]
    old.require_hash(source, source_sha)
    print(f"OTE: reading {len(names)} union features from {source.relative_to(ROOT)}", flush=True)
    frame = pd.read_parquet(source, columns=["source_row_idx", "datetime", *names])
    np.testing.assert_array_equal(frame.source_row_idx, np.arange(len(frame)))
    stamps = pd.to_datetime(frame.datetime, utc=True)
    if not stamps.is_monotonic_increasing or not stamps.is_unique:
        raise ValueError("OTE source timestamps must be unique and chronological")
    labels = pd.to_datetime(pd.read_csv(OTE_LABELS, usecols=["timestamp"])["timestamp"], utc=True)
    prefix = labels[labels < stamps.iloc[0]]
    # The label file holds every source bar; the feature tape is it minus the dropped warm-up rows.
    if not labels.iloc[len(prefix):].reset_index(drop=True).equals(stamps.reset_index(drop=True)):
        raise ValueError("OTE label bars after the warm-up prefix differ from the feature tape")
    offset = len(prefix)
    head = pd.DataFrame({name: np.full(offset, np.nan, dtype=np.float32) for name in names})
    head.insert(0, "datetime", prefix.to_numpy())
    body = frame.drop(columns="source_row_idx")
    body["datetime"] = stamps.to_numpy()
    tape = pd.concat([head, body.astype({name: np.float32 for name in names})], ignore_index=True)
    tape.insert(0, "source_row_idx", np.arange(len(tape), dtype=np.int64))
    union = root / "source_history_union.parquet"
    root.mkdir(parents=True, exist_ok=True)
    tape.to_parquet(union, index=False, compression="zstd")
    union_sha = file_sha256(union)
    dump_new(root / "source_history_union.json", dict(status="v2_ote_source_history_tape", feature_names=names,
        source_dataset=str(source.relative_to(ROOT)), source_dataset_sha256=source_sha, labels=str(OTE_LABELS.relative_to(ROOT)),
        labels_sha256=file_sha256(OTE_LABELS), source_rows=len(tape), source_row_offset=offset, warmup_prefix_rows=offset,
        source_history_sha256=union_sha, prefix_feature_availability="warm-up bars without features; per-model frozen fills"))
    for model_id, (data, cfg, _) in loaded.items():
        out = root / str(model_id)
        job = manifest["models"][str(model_id)]
        prepared = out / "prepared" / data.target_name
        copy_prepared(data, job, prepared)
        shift_ids(prepared, offset)
        fit_ids = data.source_row_idx_train[:job["selection_fit_rows"]] + offset
        computed, _ = build_fill_values(tape.loc[fit_ids, data.feature_names])
        fills = {name: float(computed.get(name, 0.0)) for name in data.feature_names}
        os.link(union, prepared / "source_history.parquet")
        dump_new(prepared / "source_history_contract.json", dict(mode=MODE, bar_minutes=5,
            feature_names=list(data.feature_names), fill_values=fills, missing_value_rule="nonfinite_to_frozen_fill",
            selection_fit_rows=len(fit_ids), selection_fit_end_source_row_idx=int(fit_ids[-1]),
            source_dataset_sha256=source_sha, source_history_sha256=union_sha, source_row_offset=offset,
            clock_note="Every observed EURUSD source bar in order; warm-up prefix bars carry frozen fills"))
        write_model_manifest(model_id, out, manifest_sha, job, cfg, source, source_sha, len(tape),
                             source_row_offset=offset, source_history_union_sha256=union_sha)
        print(f"ID {model_id}: {len(tape)} source bars (offset {offset}) x {len(data.feature_names)} loaded features", flush=True)


def load(model_id: int, root: Path, output: Path):
    path = root / str(model_id)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != model_id or manifest["status"] != "v2_source_history_prepared":
        raise ValueError("Wrong v2 source-history preparation")
    for relative, sha in manifest["prepared_files_sha256"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path.resolve()):
            raise ValueError("Invalid prepared path")
        old.require_hash(file, sha)
    old.require_hash(ROOT / manifest["source_dataset"], manifest["source_dataset_sha256"])
    parents, manifest_sha = load_manifest()
    if manifest["validation_manifest_sha256"] != manifest_sha:
        raise ValueError("Wrong parent validation views")
    original, cfg, gaps = old.load_job(parents, model_id)
    if manifest["source_job"] != parents["models"][str(model_id)] or manifest["source_config"] != asdict(cfg):
        raise ValueError("Recipe or selection ancestry changed")
    cfg.prepared_root = str(path / "prepared")
    cfg.output_root = str(output / f"{model_id:02d}_{original.target_name}")
    # Loading re-checks the tape hash and that frozen-filled history at every endpoint equals the prepared splits.
    data = t.load_prepared_target_dataset(path / "prepared", original.target_name, cfg)
    if data.source_history_X is None:
        raise ValueError("Refusing retained-row fallback")
    data.event_window_lookup = original.event_window_lookup  # ICT corrected event windows / sequential bootstrap
    data.event_window_source = original.event_window_source
    return data, cfg, gaps


def check(model_id: int, root: Path, output: Path) -> dict:
    data, cfg, gaps = load(model_id, root, output)
    result = source_history_preflight(data, cfg, model_id)
    result.update(status="v2_source_history_preflight_passed", model_id=model_id, target=data.target_name,
                  backend=cfg.backend, loaded_features=len(data.feature_names), feature_contract="v2", **gaps)
    return result


def train(model_id: int, root: Path, output: Path) -> dict:
    if not output.is_relative_to(ROOT / "models"):
        raise ValueError("Use a new output version under models")
    data, cfg, gaps = load(model_id, root, output)
    checked = source_history_preflight(data, cfg, model_id)
    out = Path(cfg.output_root)
    out.mkdir(parents=True, exist_ok=False)
    pins = {name: file_sha256(ROOT / name) for name in CODE}
    for name in CODE:
        destination = out / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    run = dict(status="training_started", model_id=model_id, config=asdict(cfg), feature_contract="v2",
        prepared_manifest_sha256=file_sha256(root / str(model_id) / "manifest.json"),
        source_history_contract=data.source_history_contract, preflight=checked,
        corrected_events_sha256=old.EVENTS_SHA256, code_sha256=pins,
        environment=dict(python=sys.version, omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost", "torch", "scikit-learn", "optuna", "joblib")}),
        collection_eligible=False, **gaps)
    path = out / "v2_training_manifest.json"
    dump_new(path, run)
    trained = t.train_target_pipeline(data, cfg)
    run.update(status="trained_pending_v2_review", outputs_sha256={
        p.name: file_sha256(p) for p in Path(trained["output_dir"]).iterdir() if p.is_file()})
    path.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(status=run["status"], model_id=model_id, output_dir=str(trained["output_dir"]), collection_eligible=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "train"))
    parser.add_argument("model", help="Roster model ID, or 'ote' for prepare (shared EURUSD tape)")
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root, output = args.prepared_root.resolve(), args.output_root.resolve()
    if not root.is_relative_to(ROOT / "artifacts"):
        parser.error("Prepared roots live under artifacts")
    if args.report and args.report.exists():
        parser.error("Use a new report path")
    if args.mode == "prepare" and args.model == "ote":
        prepare_ote(root)
        return
    if not args.model.isdigit() or int(args.model) not in (*ES_IDS, *OTE_IDS):
        parser.error("Choose a roster model ID")
    model_id = int(args.model)
    if args.mode == "prepare":
        if model_id in OTE_IDS:
            parser.error("EURUSD models share one tape: use 'prepare ote'")
        prepare_es(model_id, root)
        return
    result = check(model_id, root, output) if args.mode == "check" else train(model_id, root, output)
    if args.report:
        dump_new(args.report, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("model", "tensor_checks")}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
