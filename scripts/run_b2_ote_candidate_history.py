"""Prepare, train, and review source-history OTE candidates.

This runner is deliberately separate from the retained-row B2 route.  It
uses the corrected completed-HTF publication artifact as the source tape,
keeps every observed source bar in order, and carries frozen-prefix fills
through the saved scaler.  Only ``train`` fits a model; no mode installs or
activates anything.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import run_b2_retraining as old
from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import MODE, file_sha256, indexed_lag_view, selected_contract
from preprocessing.feature_selection import build_fill_values, encode_candidate_features
from ote_live.models.loaders import LoadedRuntimeModel, _load_backend_model, _resolve_runtime_amp_usage, validate_source_history_runtime
from ote_live.models.runners import RuntimeModelRunner


MINIMUM_CORE_IDS = (22, 26)
EXPANDED_CORE_IDS = (22, 26, 21, 25)
RESERVE_IDS = (22, 26, 21, 25, 23, 27)
ALLOWED_IDS = tuple(sorted(set(RESERVE_IDS)))

DEFAULT_PREPARED = ROOT / "artifacts/b2_candidate_history_minimum_core_ote_20260930_v3"
DEFAULT_OUTPUT = ROOT / "models/b2_candidate_history_minimum_core_ote_20260930_v3"

OTE_FEATURES = ROOT / "artifacts/b2_ote_features_20260920_v1/features.csv.gz"
OTE_FEATURES_METADATA = ROOT / "artifacts/b2_ote_features_20260920_v1/features.metadata.json"
OTE_LABELS = ROOT / "artifacts/b2_ote_labels_20260920_v2/labels.csv.gz"
OTE_LABELS_METADATA = ROOT / "artifacts/b2_ote_labels_20260920_v2/labels.metadata.json"

CODE = tuple(dict.fromkeys((*old.CODE_FILES,
    "model_training/source_history.py", "ote_live/models/loaders.py",
    "ote_live/models/runners.py", "scripts/run_b2_ote_candidate_history.py")))


def dump_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def dump_replace(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_group(root: Path, model_id: int) -> tuple[int, ...]:
    name = root.name.lower()
    if "minimum_core" in name:
        group = MINIMUM_CORE_IDS
    elif "expanded_core" in name:
        group = EXPANDED_CORE_IDS
    else:
        group = (model_id,)
    if model_id not in group:
        group = tuple(dict.fromkeys((*group, model_id)))
    return group


def require_phase(model_id: int, root: Path, *, expand_to_12: bool, promote_reserve: bool) -> None:
    if model_id in (21, 25) and not expand_to_12:
        raise ValueError("IDs 21/25 require --expand-to-12; minimum core is IDs 22/26.")
    if model_id in (23, 27) and not promote_reserve:
        raise ValueError("IDs 23/27 are reserve candidates; pass --promote-reserve only after promotion.")
    if model_id in (21, 25) and "expanded_core" not in root.name.lower() and expand_to_12:
        raise ValueError("Expanded IDs require a new expanded_core OTE prepared root.")
    if model_id in (23, 27) and "reserve" not in root.name.lower() and promote_reserve:
        raise ValueError("Reserve IDs require a separately named promoted reserve root.")


def corrected_publication_contract() -> dict:
    feature_meta = json.loads(OTE_FEATURES_METADATA.read_text(encoding="utf-8"))
    labels_meta = json.loads(OTE_LABELS_METADATA.read_text(encoding="utf-8"))
    if feature_meta.get("config", {}).get("htf_bar_timestamp_semantics") != "bar_open":
        raise ValueError("OTE feature artifact is not pinned to open-stamped bars.")
    if feature_meta.get("upstream_bar_timestamp_semantics") != "bar_open":
        raise ValueError("OTE feature upstream timestamp semantics are not bar_open.")
    if feature_meta.get("b2_repair", {}).get("version") != "b2_ote_features_20260920_v1":
        raise ValueError("OTE feature artifact is not the corrected publication repair.")
    if labels_meta.get("bar_timestamp_semantics") != "bar_open":
        raise ValueError("OTE labels are not pinned to open-stamped bars.")
    if labels_meta.get("b2_repair", {}).get("families") is None:
        raise ValueError("OTE labels do not carry the corrected publication repair evidence.")
    if not OTE_FEATURES.is_file() or not OTE_LABELS.is_file():
        raise FileNotFoundError("Corrected OTE feature/label artifacts are missing.")
    return {
        "feature_artifact": str(OTE_FEATURES.relative_to(ROOT)),
        "feature_artifact_sha256": file_sha256(OTE_FEATURES),
        "feature_metadata": str(OTE_FEATURES_METADATA.relative_to(ROOT)),
        "feature_metadata_sha256": file_sha256(OTE_FEATURES_METADATA),
        "label_artifact": str(OTE_LABELS.relative_to(ROOT)),
        "label_artifact_sha256": file_sha256(OTE_LABELS),
        "label_metadata": str(OTE_LABELS_METADATA.relative_to(ROOT)),
        "label_metadata_sha256": file_sha256(OTE_LABELS_METADATA),
        "source_bar_timestamp_semantics": "bar_open",
        "publication_rule": "completed_30m_and_1h_structural_buckets_only",
        "feature_repair_version": "b2_ote_features_20260920_v1",
        "label_repair_version": "b2_ote_labels_20260920_v2",
    }


def prepared_feature_names(model_id: int, parent_manifest: dict) -> list[str]:
    job = parent_manifest["models"][str(model_id)]
    path = ROOT / job["prepared_dir"] / "features.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("features")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid frozen selected-feature list for ID {model_id}.")
    return [str(name) for name in names]


def verify_source_lineage(dataset, source_contract: dict) -> None:
    lineage = dataset.report.get("source_lineage", {})
    declared = lineage.get("feature_builder_source_path")
    if not declared:
        raise ValueError("Prepared OTE target lacks feature-builder source lineage.")
    declared_path = Path(declared)
    declared_path = declared_path if declared_path.is_absolute() else ROOT / declared_path
    if declared_path.resolve() != OTE_FEATURES.resolve():
        raise ValueError("Prepared OTE target does not point at the corrected feature source.")
    if lineage.get("upstream_bar_timestamp_semantics") != "bar_open":
        raise ValueError("Prepared OTE target has non-open upstream timestamp semantics.")
    if source_contract["feature_repair_version"] != "b2_ote_features_20260920_v1":
        raise ValueError("Unexpected OTE feature repair version.")


def endpoint_expectations(dataset):
    ids = np.concatenate([dataset.source_row_idx_train, dataset.source_row_idx_val, dataset.source_row_idx_test]).astype(np.int64)
    expected = np.concatenate([dataset.X_train, dataset.X_val, dataset.X_test]).astype(np.float32, copy=False)
    if len(ids) != len(np.unique(ids)) or not np.all(np.diff(ids) > 0):
        raise ValueError("OTE split source endpoints must be unique and chronological.")
    return ids, expected


def source_prefix_timestamps() -> pd.DatetimeIndex:
    first_suffix = pd.to_datetime(
        pd.read_csv(OTE_FEATURES, usecols=["datetime"], nrows=1)["datetime"], utc=True
    ).iloc[0]
    pieces = []
    for chunk in pd.read_csv(OTE_LABELS, usecols=["timestamp"], chunksize=50000):
        timestamps = pd.to_datetime(chunk["timestamp"], utc=True)
        before = timestamps[timestamps < first_suffix]
        pieces.append(before)
        if len(before) != len(timestamps):
            break
    prefix = pd.DatetimeIndex(pd.concat(pieces, ignore_index=True))
    if len(prefix) == 0 or not prefix.is_monotonic_increasing or not prefix.is_unique:
        raise ValueError("Could not identify a unique chronological OTE warmup source prefix.")
    return prefix


def endpoint_expectations_with_offset(dataset, source_offset: int):
    ids, expected = endpoint_expectations(dataset)
    return ids + int(source_offset), expected


def write_shared_source_tape(root: Path, names: list[str], datasets: dict[int, object], source_contract: dict) -> tuple[Path, dict]:
    tape = root / "source_history_union.parquet"
    if tape.exists():
        shared = json.loads((root / "source_history_union.json").read_text(encoding="utf-8"))
        if shared.get("feature_names") != names:
            raise ValueError("Existing source tape feature union differs; use a new prepared root.")
        old.require_hash(tape, shared["source_history_sha256"])
        return tape, shared

    root.mkdir(parents=True, exist_ok=True)
    source_contract = dict(source_contract)
    dtype = {name: "float32" for name in names}
    usecols = ["datetime", *names]
    temp = root / "source_history_union.parquet.partial"
    if temp.exists():
        raise ValueError(f"Partial source tape exists; preserve it and use a new root: {temp}")

    prefix = source_prefix_timestamps()
    endpoint_sets = {}
    for model_id, dataset in datasets.items():
        endpoint_sets[model_id] = endpoint_expectations_with_offset(dataset, len(prefix))
    writer = None
    row_start = len(prefix)
    previous_ts = None
    bad_columns = {name: 0 for name in names}
    fit_frames: list[pd.DataFrame] = []
    try:
        prefix_frame = pd.DataFrame({name: np.full(len(prefix), np.nan, dtype=np.float32) for name in names})
        prefix_frame.insert(0, "source_row_idx", np.arange(len(prefix), dtype=np.int64))
        prefix_frame.insert(1, "datetime", pd.Series(prefix, dtype="datetime64[ns, UTC]"))
        prefix_table = pa.Table.from_pandas(prefix_frame, preserve_index=False)
        writer_schema = prefix_table.schema
        writer = pq.ParquetWriter(temp, prefix_table.schema, compression="zstd")
        writer.write_table(prefix_table)
        for chunk in pd.read_csv(OTE_FEATURES, usecols=usecols, dtype=dtype, chunksize=20000):
            timestamps = pd.to_datetime(chunk.pop("datetime"), utc=True)
            if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
                raise ValueError("Corrected OTE source timestamps are not valid and chronological.")
            if previous_ts is not None and timestamps.iloc[0] <= previous_ts:
                raise ValueError("Corrected OTE source timestamps are not strictly increasing.")
            previous_ts = timestamps.iloc[-1]
            values = chunk[names].to_numpy(dtype=np.float32, copy=True)
            bad = ~np.isfinite(values)
            if bad.any():
                for index, name in enumerate(names):
                    count = int(bad[:, index].sum())
                    bad_columns[name] += count
                fit_end = max(
                    len(prefix) + int(dataset.source_row_idx_train[dataset.report["selection_contract"]["fit_rows"] - 1]) + 1
                    for dataset in datasets.values()
                )
                local_ids = np.arange(row_start, row_start + len(chunk), dtype=np.int64)
                fit_mask = local_ids < fit_end
                if fit_mask.any():
                    fit_frames.append(chunk.loc[fit_mask, names].copy())

            table_frame = chunk.copy()
            table_frame.insert(0, "source_row_idx", np.arange(row_start, row_start + len(chunk), dtype=np.int64))
            table_frame.insert(1, "datetime", timestamps.to_numpy())
            table = pa.Table.from_pandas(table_frame, schema=writer_schema, preserve_index=False, safe=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)

            for model_id, dataset in datasets.items():
                ids, expected = endpoint_sets[model_id]
                left = int(np.searchsorted(ids, row_start, side="left"))
                right = int(np.searchsorted(ids, row_start + len(chunk), side="left"))
                if left == right:
                    continue
                selected = chunk.iloc[ids[left:right] - row_start][dataset.feature_names].to_numpy(dtype=np.float32, copy=True)
                if bad.any():
                    # The normal corrected artifact has no nonfinite selected
                    # cells.  If that changes, the second pass below applies
                    # the frozen-prefix fills before comparison.
                    continue
                np.testing.assert_allclose(selected, expected[left:right], rtol=1e-6, atol=1e-6)
            row_start += len(chunk)
            if row_start % 100000 == 0:
                print(f"source rows written: {row_start}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if row_start == 0:
        raise ValueError("Corrected OTE source tape is empty.")

    fills = {name: 0.0 for name in names}
    if any(bad_columns.values()):
        fit_frame = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame(columns=names)
        calculated, _ = build_fill_values(fit_frame)
        fills.update({name: float(calculated.get(name, 0.0)) for name in names})
        # Re-read only the endpoint rows through the Parquet tape so the
        # comparison uses exactly the persisted raw history and frozen fills.
        for model_id, dataset in datasets.items():
            ids, expected = endpoint_sets[model_id]
            history = pd.read_parquet(temp, columns=["source_row_idx", *dataset.feature_names])
            raw = history[dataset.feature_names].to_numpy(dtype=np.float32, copy=True)
            mask = ~np.isfinite(raw)
            if mask.any():
                fill_array = np.asarray([fills[name] for name in dataset.feature_names], dtype=np.float32)
                raw[mask] = np.broadcast_to(fill_array, raw.shape)[mask]
            np.testing.assert_allclose(raw[ids], expected, rtol=1e-6, atol=1e-6)

    os.replace(temp, tape)
    shared = dict(status="corrected_ote_source_history_tape",
                  source_contract=source_contract, feature_names=names,
                  source_rows=row_start, source_row_offset=len(prefix), warmup_prefix_rows=len(prefix),
                  source_history_sha256=file_sha256(tape),
                  fill_values=fills, raw_nonfinite_cells=int(sum(bad_columns.values())),
                  consecutive_source_rows=True, prepared_from_corrected_timestamp_publication=True,
                  prefix_feature_availability="unavailable_before_corrected_feature_artifact_start; frozen_imputation_applied")
    dump_new(root / "source_history_union.json", shared)
    return tape, shared


def prepare(model_id: int, root: Path, *, expand_to_12: bool, promote_reserve: bool) -> None:
    require_phase(model_id, root, expand_to_12=expand_to_12, promote_reserve=promote_reserve)
    if not root.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("Preparation requires a new root under artifacts.")
    target_root = root / str(model_id)
    if target_root.exists():
        raise ValueError(f"Refusing to overwrite existing OTE candidate preparation: {target_root}")

    parent_manifest = old.load_manifest()
    source_contract = corrected_publication_contract()
    group = resolve_group(root, model_id)
    names_union = []
    for group_id in group:
        names_union.extend(prepared_feature_names(group_id, parent_manifest))
    names_union = list(dict.fromkeys(names_union))

    dataset, config, _ = old.load_job(parent_manifest, model_id)
    verify_source_lineage(dataset, source_contract)
    prepared = target_root / "prepared" / dataset.target_name
    prepared.mkdir(parents=True)
    job = parent_manifest["models"][str(model_id)]
    for name in job["prepared_hashes"]:
        shutil.copyfile(dataset.prepared_dir / name, prepared / name)
    shutil.copyfile(dataset.prepared_dir.parent / "summary.json", prepared.parent / "summary.json")

    datasets = {model_id: dataset}
    # A minimum-core tape is built once for both 22 and 26.  It contains only
    # their selected features; reserve and expanded feature families are not
    # materialized in this root.
    if not (root / "source_history_union.parquet").exists():
        for other_id in group:
            if other_id == model_id:
                continue
            other_dataset, _, _ = old.load_job(parent_manifest, other_id)
            verify_source_lineage(other_dataset, source_contract)
            datasets[other_id] = other_dataset

    print(f"Building corrected consecutive OTE source history for ID {model_id}", flush=True)
    tape, shared = write_shared_source_tape(root, names_union, datasets, source_contract)
    source_offset = int(shared["source_row_offset"])
    for split in ("train", "val", "test"):
        split_path = prepared / f"{split}.csv"
        split_frame = pd.read_csv(split_path)
        split_frame["source_row_idx"] = split_frame["source_row_idx"].astype(np.int64) + source_offset
        split_frame.to_csv(split_path, index=False)
    report_path = prepared / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    geometry = report.get("split_geometry", {})
    for bounds in geometry.get("split_source_row_ranges", {}).values():
        bounds["start_source_row_idx"] += source_offset
        bounds["end_source_row_idx"] += source_offset
    for boundary in geometry.get("boundaries", []):
        for key in ("current_split_last_source_row_idx", "next_split_first_source_row_idx"):
            if key in boundary:
                boundary[key] += source_offset
    selection = report.get("selection_contract", {})
    if "fit_end_source_row_idx" in selection:
        selection["fit_end_source_row_idx"] += source_offset
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dataset.source_row_idx_train = dataset.source_row_idx_train + source_offset
    dataset.source_row_idx_val = dataset.source_row_idx_val + source_offset
    dataset.source_row_idx_test = dataset.source_row_idx_test + source_offset
    dataset._dev_source_row_idx_cache = None
    link = prepared / "source_history.parquet"
    os.link(tape, link)
    fill_values = {name: float(shared["fill_values"].get(name, 0.0)) for name in dataset.feature_names}
    fit_ids = dataset.source_row_idx_train[:job["selection_fit_rows"]]
    contract = dict(mode=MODE, bar_minutes=5, feature_names=list(dataset.feature_names),
                    fill_values=fill_values, missing_value_rule="nonfinite_to_frozen_fill",
                    selection_fit_rows=len(fit_ids), selection_fit_end_source_row_idx=int(fit_ids[-1]),
                    source_dataset_sha256=source_contract["feature_artifact_sha256"],
                    source_history_sha256=file_sha256(link),
                    source_row_offset=source_offset,
                    clock_note="Every observed EURUSD source bar in chronological order; corrected completed 30m/1h publication is pinned separately")
    dump_new(prepared / "source_history_contract.json", contract)
    payload = dict(status="candidate_source_history_prepared", model_id=model_id,
                   target=dataset.target_name, phase=("minimum_core" if model_id in MINIMUM_CORE_IDS else "expanded_or_promoted"),
                   source_manifest_sha256=old.MANIFEST_SHA256, source_job=job, source_config=asdict(config),
                   corrected_publication=source_contract, source_rows=shared["source_rows"],
                   source_row_offset=source_offset,
                   source_history_union_sha256=shared["source_history_sha256"],
                   prepared_files_sha256={str(p.relative_to(target_root)): file_sha256(p)
                                          for p in target_root.rglob("*") if p.is_file()},
                   collection_eligible=False, legacy_bundle_used_for_replacement=False)
    dump_new(target_root / "manifest.json", payload)
    print(f"ID {model_id}: prepared {shared['source_rows']} consecutive source bars; frozen-prefix fills and original split identities preserved", flush=True)


def load(model_id: int, root: Path, output: Path):
    path = root / str(model_id)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No OTE source-history preparation for ID {model_id}: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_source_history_prepared" or manifest.get("model_id") != model_id:
        raise ValueError("Wrong OTE candidate preparation status.")
    if manifest["source_manifest_sha256"] != old.MANIFEST_SHA256:
        raise ValueError("Wrong pinned OTE parent preparation manifest.")
    for relative, sha in manifest["prepared_files_sha256"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path.resolve()):
            raise ValueError("Invalid OTE prepared path.")
        old.require_hash(file, sha)
    shared = json.loads((root / "source_history_union.json").read_text(encoding="utf-8"))
    old.require_hash(root / "source_history_union.parquet", shared["source_history_sha256"])
    source_contract = corrected_publication_contract()
    if manifest["corrected_publication"] != source_contract:
        raise ValueError("Corrected OTE publication artifact changed after preparation.")

    parents = old.load_manifest()
    original, config, gaps = old.load_job(parents, model_id)
    if manifest["source_job"] != parents["models"][str(model_id)] or manifest["source_config"] != asdict(config):
        raise ValueError("Pinned OTE recipe or preparation ancestry changed.")
    config.prepared_root = str(path / "prepared")
    config.output_root = str(output / f"{model_id:02d}_{original.target_name}")
    dataset = t.load_prepared_target_dataset(path / "prepared", original.target_name, config)
    if dataset.source_history_X is None:
        raise ValueError("Refusing a retained-row fallback for OTE source history.")
    if dataset.source_history_contract.get("source_dataset_sha256") != source_contract["feature_artifact_sha256"]:
        raise ValueError("OTE source-history contract does not pin the corrected feature artifact.")
    return dataset, config, gaps


def preflight(data, config, model_id: int) -> dict:
    from ote_live.models.runners import _build_sequence_windows, _build_sparse_lag_view
    result = old.check_job(data, config, model_id)
    scaler = t.fit_scaler(data.X_train[:data.source_history_contract["selection_fit_rows"]], config)
    indices = np.arange(len(data.feature_names))
    ends = np.concatenate([data.dev_source_row_idx[:8], data.dev_source_row_idx[-8:], data.test_source_row_idx])
    checks = []
    for window in range(config.window_min, config.window_max + 1, 4):
        if config.backend == "xgboost":
            lags = t.make_lag_steps(window, config.lag_count_max)
            expected = indexed_lag_view(data.source_history_X, ends, indices, scaler, lags, config.delta_feature_cap, config.scale_clip)
        else:
            expected = t.indexed_sequence_view(data.source_history_X, ends, indices, scaler, window, config.scale_clip)
        for n, end in enumerate(ends):
            scaled = t.transform_matrix(scaler, data.source_history_X[end - window + 1:end + 1], config.scale_clip)
            if config.backend == "xgboost":
                actual = _build_sparse_lag_view(scaled, lag_steps=lags, delta_feature_count=config.delta_feature_cap)[-1]
                np.testing.assert_array_equal(actual, expected[n])
            else:
                actual = _build_sequence_windows(scaled, window_size=window)[-1]
                start = expected.sampled_indices[n]
                np.testing.assert_array_equal(actual, expected.matrix[start:start + window])
        checks.append(dict(window=window, endpoints=len(ends), exact_training_runtime_tensor_equality=True))
    return dict(status="candidate_ote_history_preflight_passed", model=result, tensor_checks=checks,
                corrected_timestamp_publication=True, collection_eligible=False)


def runtime_model(output: Path, summary: dict):
    backend = summary["config"]["backend"]
    if backend == "torch":
        backend = summary["config"]["model_type"]
    manifest = SimpleNamespace(model_id=f"ote_{output.name}_candidate", backend=backend,
        asset="EURUSD", timeframe="5m", direction="short" if output.name.startswith("short") else "long",
        feature_manifest=SimpleNamespace(selected_feature_names=summary["selected_feature_names"],
            lag_steps=summary["lag_steps"], delta_feature_count=summary["delta_feature_count"]),
        artifact_references=SimpleNamespace(model_file=str(output / ("model.json" if backend == "xgboost" else "model.pt"))),
        context_requirements=SimpleNamespace(context_rows=summary["context_rows"], window_size=summary["window_size"]))
    model = _load_backend_model(manifest)
    scaler = joblib.load(output / "scaler.joblib")
    validate_source_history_runtime(manifest, summary, scaler)
    calibrator = joblib.load(output / "calibrator.joblib") if (output / "calibrator.joblib").exists() else None
    use_amp = _resolve_runtime_amp_usage(manifest=manifest, training_summary=summary)
    return RuntimeModelRunner(LoadedRuntimeModel(manifest, model, scaler, calibrator, summary,
        summary["config"]["scale_clip"], summary["config"]["batch_size"], use_amp))


def review(data, config, root: Path, output: Path) -> dict:
    run_path = output / "candidate_ote_training_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "trained_pending_candidate_ote_history_review" or run["config"] != asdict(config):
        raise ValueError("Incomplete or changed OTE candidate training recipe.")
    if run["corrected_publication"] != corrected_publication_contract():
        raise ValueError("Corrected OTE publication lineage changed after training.")
    for name, sha in run["code_sha256"].items():
        old.require_hash(output / "code_snapshot" / name, sha)
        old.require_hash(ROOT / name, sha)
    model = output / data.target_name
    for name, sha in run["outputs_sha256"].items():
        old.require_hash(model / name, sha)
    summary = json.loads((model / "training_summary.json").read_text(encoding="utf-8"))
    names = summary["selected_feature_names"]
    if names != data.feature_names[:len(names)] or summary["config"] != asdict(config) or summary["report"] != data.report:
        raise ValueError("OTE selected features, recipe, or preparation report changed.")
    if summary["model_config"]["source_history_contract"] != selected_contract(data.source_history_contract, names):
        raise ValueError("Fitted OTE history/fill contract changed.")
    scaler = joblib.load(model / "scaler.joblib")
    fit_X = data.dev_X[:, :len(names)] if config.final_refit_on_dev else data.dev_X[:summary["final_holdout"]["fit_end"], :len(names)]
    expected_scaler = t.fit_scaler(fit_X, config)
    np.testing.assert_array_equal(scaler.scaler.center_, expected_scaler.center_)
    np.testing.assert_array_equal(scaler.scaler.scale_, expected_scaler.scale_)
    trials = pd.read_csv(model / "optuna_trials.csv")
    if len(trials) != config.n_trials or not set(trials.state) <= {"COMPLETE", "PRUNED"} or not (trials.state == "COMPLETE").any():
        raise ValueError("OTE search budget incomplete or trials failed.")
    best = trials.loc[trials.number.eq(summary["study_best_trial"])].iloc[0]
    if best.state != "COMPLETE" or not np.isclose(best.value, trials.loc[trials.state.eq("COMPLETE"), "value"].max()):
        raise ValueError("OTE best-trial selection mismatch.")
    for name, value in summary["best_params"].items():
        if not np.isclose(best["params_" + name], value):
            raise ValueError("Saved OTE best parameters differ from selected trial.")

    context = summary["context_rows"]
    purge = t.resolve_dataset_cv_purge_bars(data, config, context_rows=context)
    folds = t.PurgedWalkForwardSplitter(max(config.cv_initial_train_rows, context + 1), config.cv_val_rows,
        config.cv_step_rows, purge, config.cv_max_train_rows, config.cv_min_folds).split(len(data.dev_y))
    if summary["cv_splitter"]["fold_plans"] != [asdict(fold) for fold in folds]:
        raise ValueError("Saved OTE CV geometry mismatch.")
    t.validate_frozen_selection(data, folds)
    oof = pd.read_csv(model / "oof_predictions.csv")
    np.testing.assert_array_equal(oof.source_row_idx, data.dev_source_row_idx)
    fold_ids = np.full(len(data.dev_y), -1)
    for fold in folds:
        fold_ids[fold.val_start:fold.val_end] = fold.fold
    np.testing.assert_array_equal(oof.fold, fold_ids)
    valid = oof.oof_raw_probability.notna()
    np.testing.assert_array_equal(valid, fold_ids >= 0)
    raw_oof = oof.loc[valid, "oof_raw_probability"].to_numpy(np.float32)
    expected_calibration = t.resolve_calibration_method(config, data.target_name)
    requested_calibration = str(config.calibration_method).strip().lower()
    calibration = summary["calibration"]
    if (calibration["requested_method"] != requested_calibration
            or calibration["resolved_method"] != expected_calibration
            or bool(calibration.get("disabled_for_continuation_target", False))):
        raise ValueError("Saved OTE calibration contract does not match the recipe.")
    calibrator = joblib.load(model / "calibrator.joblib") if expected_calibration != "none" else None
    calibrated_oof = t.apply_calibrator(calibrator, raw_oof)
    np.testing.assert_allclose(calibrated_oof, oof.loc[valid, "oof_calibrated_probability"].to_numpy(np.float32), rtol=1e-6, atol=1e-6)
    threshold, _ = t.select_operating_threshold(y_true=data.dev_y[valid], y_score=calibrated_oof,
        tolerance=config.event_tolerance_bars, cooldown=config.event_cooldown_bars,
        grid_size=config.threshold_grid_size, config=config)
    if not np.isclose(threshold, summary["threshold"]):
        raise ValueError("OTE threshold was not selected from calibrated OOF predictions only.")

    if config.backend == "torch":
        import torch
        checkpoint = torch.load(model / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint["selection_mode"] != "fixed_duration_refit" or checkpoint["best_val_auprc"] is not None or checkpoint["best_epoch"] != len(summary["refit_fixed_epoch_schedule"]):
            raise ValueError("OTE TCN final refit used validation or changed duration.")

    runner = runtime_model(model, summary)
    raw = pd.read_parquet(data.prepared_dir / "source_history.parquet",
                           columns=["source_row_idx", "datetime", *names])
    runtime_predictions = []
    for endpoint in data.test_source_row_idx:
        endpoint = int(endpoint)
        runtime_predictions.append(runner.predict_latest(raw.iloc[endpoint - context:endpoint + 1]))
    raw_predictions = np.asarray([prediction.raw_score for prediction in runtime_predictions])
    calibrated_predictions = np.asarray([prediction.calibrated_probability for prediction in runtime_predictions])
    saved = pd.read_csv(model / "test_predictions.csv")
    np.testing.assert_array_equal(saved.source_row_idx, data.test_source_row_idx)
    np.testing.assert_allclose(raw_predictions, saved.raw_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(calibrated_predictions, saved.calibrated_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(calibrated_predictions >= threshold, saved.predicted_label)
    return dict(status="candidate_ote_history_artifact_review_passed", model_id=run["model_id"],
        target=data.target_name, backend=config.backend, prediction_rows=len(saved), threshold=float(threshold),
        corrected_timestamp_publication_verified=True, consecutive_source_history_verified=True,
        prediction_max_abs_error=float(np.max(np.abs(raw_predictions - saved.raw_probability))),
        calibrated_prediction_max_abs_error=float(np.max(np.abs(calibrated_predictions - saved.calibrated_probability))),
        threshold_decisions_equal=True, OOF_threshold_reproduced=True, calibration_selection_source="validation_only_OOF",
        scaler_fit_scope_verified=True, frozen_prefix_imputation_verified=True,
        selected_feature_order_verified=True, purged_holdout_and_original_cv_verified=True,
        training_manifest_sha256=file_sha256(run_path), model_outputs_sha256=run["outputs_sha256"],
        prepared_manifest_sha256=file_sha256(root / str(run["model_id"]) / "manifest.json"),
        B1_evaluation_complete=False, collection_eligible=False, promotion_eligible=False)


def train_or_check(mode: str, model_id: int, root: Path, output: Path, *, expand_to_12: bool, promote_reserve: bool, report: Path | None) -> None:
    require_phase(model_id, root, expand_to_12=expand_to_12, promote_reserve=promote_reserve)
    if mode == "prepare":
        prepare(model_id, root, expand_to_12=expand_to_12, promote_reserve=promote_reserve)
        return
    data, config, gaps = load(model_id, root, output)
    if mode == "review":
        result = review(data, config, root, Path(config.output_root))
    else:
        result = preflight(data, config, model_id)
        print(f"ID {model_id}: corrected publication, source tensors, frozen selection and purged holdout pass", flush=True)
        if mode == "train":
            out = output / f"{model_id:02d}_{data.target_name}"
            if not output.resolve().is_relative_to(ROOT / "models"):
                raise ValueError("Use a new versioned root under models.")
            out.mkdir(parents=True, exist_ok=False)
            pins = {name: file_sha256(ROOT / name) for name in CODE}
            for name in CODE:
                destination = out / "code_snapshot" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            run = dict(status="training_started", model_id=model_id, config=asdict(config),
                       source_history_contract=data.source_history_contract, preflight=result,
                       prepared_manifest_sha256=file_sha256(root / str(model_id) / "manifest.json"),
                       corrected_publication=corrected_publication_contract(), code_sha256=pins,
                       environment=dict(python=sys.version, omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                           packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost", "torch", "scikit-learn", "optuna", "joblib", "pyarrow")}),
                       collection_eligible=False, **gaps)
            path = out / "candidate_ote_training_manifest.json"
            dump_new(path, run)
            trained = t.train_target_pipeline(data, config)
            run.update(status="trained_pending_candidate_ote_history_review", outputs_sha256={
                p.name: file_sha256(p) for p in Path(trained["output_dir"]).iterdir() if p.is_file()})
            dump_replace(path, run)
            result = dict(status=run["status"], model_id=model_id, output_dir=str(trained["output_dir"]), collection_eligible=False)
    if report:
        dump_new(report, result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "train", "review"))
    parser.add_argument("model", type=int, choices=ALLOWED_IDS)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expand-to-12", action="store_true")
    parser.add_argument("--promote-reserve", action="store_true")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Use a new report path.")
    train_or_check(args.mode, args.model, args.prepared_root.resolve(), args.output_root.resolve(),
                   expand_to_12=args.expand_to_12, promote_reserve=args.promote_reserve, report=args.report)


if __name__ == "__main__":
    main()
