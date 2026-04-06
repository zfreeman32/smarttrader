from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.models.loaders import LoadedRuntimeModel, load_runtime_model
from ote_live.models.runners import RuntimeModelRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XGBOOST_RAW_PROBABILITY_ATOL = 1e-6
DEFAULT_XGBOOST_CALIBRATED_PROBABILITY_ATOL = 1e-6
DEFAULT_TORCH_RAW_PROBABILITY_ATOL = 2e-4
DEFAULT_TORCH_CALIBRATED_PROBABILITY_ATOL = 1e-6


@dataclass(frozen=True)
class PredictionReplayMismatch:
    row_index: int
    source_row_idx: int
    expected_raw_probability: float
    actual_raw_probability: float
    raw_probability_abs_diff: float
    expected_calibrated_probability: float
    actual_calibrated_probability: float
    calibrated_probability_abs_diff: float
    expected_predicted_label: int | None = None
    actual_predicted_label: int | None = None


@dataclass(frozen=True)
class PredictionReplayReport:
    model_id: str
    backend: str
    direction: str
    artifact_threshold: float | None
    requested_evaluation_rows: int
    row_offset: int
    rows_evaluated: int
    raw_probability_atol: float
    calibrated_probability_atol: float
    raw_probability_matches: int
    raw_probability_mismatches: int
    calibrated_probability_matches: int
    calibrated_probability_mismatches: int
    predicted_label_compared_rows: int
    predicted_label_matches: int
    predicted_label_mismatches: int
    max_raw_probability_abs_diff: float | None
    max_calibrated_probability_abs_diff: float | None
    sample_mismatches: tuple[PredictionReplayMismatch, ...]


def replay_prediction_parity(
    loaded_model_or_manifest_or_path: LoadedRuntimeModel | LiveRuntimeManifest | str | Path,
    *,
    evaluation_rows: int = 25,
    row_offset: int = 0,
    raw_probability_atol: float | None = None,
    calibrated_probability_atol: float | None = None,
    max_mismatch_examples: int = 20,
) -> PredictionReplayReport:
    if evaluation_rows <= 0:
        raise ValueError("evaluation_rows must be positive.")
    if row_offset < 0:
        raise ValueError("row_offset must be non-negative.")

    loaded_model = _coerce_loaded_model(loaded_model_or_manifest_or_path)
    raw_probability_atol = float(
        raw_probability_atol
        if raw_probability_atol is not None
        else (
            DEFAULT_XGBOOST_RAW_PROBABILITY_ATOL
            if loaded_model.backend == "xgboost"
            else DEFAULT_TORCH_RAW_PROBABILITY_ATOL
        )
    )
    calibrated_probability_atol = float(
        calibrated_probability_atol
        if calibrated_probability_atol is not None
        else (
            DEFAULT_XGBOOST_CALIBRATED_PROBABILITY_ATOL
            if loaded_model.backend == "xgboost"
            else DEFAULT_TORCH_CALIBRATED_PROBABILITY_ATOL
        )
    )

    reference_predictions = _load_reference_prediction_frame(
        loaded_model,
        row_offset=row_offset,
        max_rows=evaluation_rows,
    )
    if reference_predictions.empty:
        return PredictionReplayReport(
            model_id=loaded_model.model_id,
            backend=loaded_model.backend,
            direction=loaded_model.manifest.direction,
            artifact_threshold=_resolve_artifact_threshold(loaded_model),
            requested_evaluation_rows=int(evaluation_rows),
            row_offset=int(row_offset),
            rows_evaluated=0,
            raw_probability_atol=raw_probability_atol,
            calibrated_probability_atol=calibrated_probability_atol,
            raw_probability_matches=0,
            raw_probability_mismatches=0,
            calibrated_probability_matches=0,
            calibrated_probability_mismatches=0,
            predicted_label_compared_rows=0,
            predicted_label_matches=0,
            predicted_label_mismatches=0,
            max_raw_probability_abs_diff=None,
            max_calibrated_probability_abs_diff=None,
            sample_mismatches=(),
        )

    feature_columns = _prepared_feature_columns(loaded_model)
    prefill_frame = _load_prefill_feature_frame(loaded_model, feature_columns=feature_columns)
    stop_row = row_offset + len(reference_predictions)
    test_feature_frame = _read_csv_rows(
        _resolve_prepared_dir(loaded_model) / "test.csv",
        columns=feature_columns,
        start_row=0,
        max_rows=stop_row,
    )
    if len(test_feature_frame) < stop_row:
        raise ValueError(
            f"Prepared test split for {loaded_model.model_id} only provided {len(test_feature_frame)} rows, "
            f"but replay requested rows through index {stop_row - 1}."
        )

    combined_frame = pd.concat([prefill_frame, test_feature_frame], ignore_index=True)
    runner = RuntimeModelRunner(loaded_model)
    artifact_threshold = _resolve_artifact_threshold(loaded_model)

    raw_probability_matches = 0
    calibrated_probability_matches = 0
    predicted_label_compared_rows = 0
    predicted_label_matches = 0
    max_raw_probability_abs_diff: float | None = None
    max_calibrated_probability_abs_diff: float | None = None
    sample_mismatches: list[PredictionReplayMismatch] = []

    context_window_rows = loaded_model.context_rows + 1
    reference_predictions = reference_predictions.reset_index(drop=True)
    for local_row_index, reference_row in reference_predictions.iterrows():
        absolute_row_index = row_offset + local_row_index
        end_position = len(prefill_frame) + absolute_row_index + 1
        start_position = end_position - context_window_rows
        feature_window = combined_frame.iloc[start_position:end_position].reset_index(drop=True)
        if len(feature_window) != context_window_rows:
            raise ValueError(
                f"Replay window for {loaded_model.model_id} expected {context_window_rows} rows, "
                f"but assembled {len(feature_window)}."
            )

        expected_source_row_idx = _coerce_required_int(reference_row["source_row_idx"], "source_row_idx")
        actual_source_row_idx = _coerce_required_int(feature_window["source_row_idx"].iloc[-1], "source_row_idx")
        if expected_source_row_idx != actual_source_row_idx:
            raise ValueError(
                f"Replay row misalignment for {loaded_model.model_id}: "
                f"reference source_row_idx={expected_source_row_idx} but replay assembled {actual_source_row_idx}."
            )

        prediction = runner.predict_latest(
            feature_window,
            timestamp=_synthetic_replay_timestamp(expected_source_row_idx),
            source_row_idx=expected_source_row_idx,
        )
        expected_raw_probability = _coerce_required_float(reference_row["raw_probability"], "raw_probability")
        expected_calibrated_probability = _coerce_required_float(
            reference_row["calibrated_probability"],
            "calibrated_probability",
        )
        actual_raw_probability = float(prediction.raw_score)
        actual_calibrated_probability = float(prediction.calibrated_probability)
        raw_probability_abs_diff = abs(actual_raw_probability - expected_raw_probability)
        calibrated_probability_abs_diff = abs(
            actual_calibrated_probability - expected_calibrated_probability
        )
        raw_matches = raw_probability_abs_diff <= raw_probability_atol
        calibrated_matches = calibrated_probability_abs_diff <= calibrated_probability_atol

        if raw_matches:
            raw_probability_matches += 1
        if calibrated_matches:
            calibrated_probability_matches += 1

        max_raw_probability_abs_diff = _update_max_diff(max_raw_probability_abs_diff, raw_probability_abs_diff)
        max_calibrated_probability_abs_diff = _update_max_diff(
            max_calibrated_probability_abs_diff,
            calibrated_probability_abs_diff,
        )

        expected_predicted_label = _coerce_optional_int(reference_row.get("predicted_label"))
        actual_predicted_label: int | None = None
        predicted_label_matches_row = True
        if expected_predicted_label is not None and artifact_threshold is not None:
            predicted_label_compared_rows += 1
            actual_predicted_label = int(actual_calibrated_probability >= artifact_threshold)
            predicted_label_matches_row = actual_predicted_label == expected_predicted_label
            if predicted_label_matches_row:
                predicted_label_matches += 1

        if (
            (not raw_matches or not calibrated_matches or not predicted_label_matches_row)
            and len(sample_mismatches) < max_mismatch_examples
        ):
            sample_mismatches.append(
                PredictionReplayMismatch(
                    row_index=int(absolute_row_index),
                    source_row_idx=expected_source_row_idx,
                    expected_raw_probability=expected_raw_probability,
                    actual_raw_probability=actual_raw_probability,
                    raw_probability_abs_diff=raw_probability_abs_diff,
                    expected_calibrated_probability=expected_calibrated_probability,
                    actual_calibrated_probability=actual_calibrated_probability,
                    calibrated_probability_abs_diff=calibrated_probability_abs_diff,
                    expected_predicted_label=expected_predicted_label,
                    actual_predicted_label=actual_predicted_label,
                )
            )

    rows_evaluated = int(len(reference_predictions))
    return PredictionReplayReport(
        model_id=loaded_model.model_id,
        backend=loaded_model.backend,
        direction=loaded_model.manifest.direction,
        artifact_threshold=artifact_threshold,
        requested_evaluation_rows=int(evaluation_rows),
        row_offset=int(row_offset),
        rows_evaluated=rows_evaluated,
        raw_probability_atol=raw_probability_atol,
        calibrated_probability_atol=calibrated_probability_atol,
        raw_probability_matches=raw_probability_matches,
        raw_probability_mismatches=rows_evaluated - raw_probability_matches,
        calibrated_probability_matches=calibrated_probability_matches,
        calibrated_probability_mismatches=rows_evaluated - calibrated_probability_matches,
        predicted_label_compared_rows=predicted_label_compared_rows,
        predicted_label_matches=predicted_label_matches,
        predicted_label_mismatches=predicted_label_compared_rows - predicted_label_matches,
        max_raw_probability_abs_diff=max_raw_probability_abs_diff,
        max_calibrated_probability_abs_diff=max_calibrated_probability_abs_diff,
        sample_mismatches=tuple(sample_mismatches),
    )


def _coerce_loaded_model(
    loaded_model_or_manifest_or_path: LoadedRuntimeModel | LiveRuntimeManifest | str | Path,
) -> LoadedRuntimeModel:
    if isinstance(loaded_model_or_manifest_or_path, LoadedRuntimeModel):
        return loaded_model_or_manifest_or_path
    return load_runtime_model(loaded_model_or_manifest_or_path)


def _resolve_prepared_dir(loaded_model: LoadedRuntimeModel) -> Path:
    prepared_dir = loaded_model.training_summary.get("prepared_dir")
    if not prepared_dir:
        raise ValueError(f"Training summary for {loaded_model.model_id} does not contain a prepared_dir.")
    return _resolve_repo_relative_path(prepared_dir)


def _resolve_reference_predictions_path(loaded_model: LoadedRuntimeModel) -> Path:
    prediction_file = loaded_model.manifest.artifact_references.test_predictions_file
    if not prediction_file:
        raise ValueError(f"Runtime manifest for {loaded_model.model_id} does not define test_predictions_file.")
    return _resolve_repo_relative_path(prediction_file)


def _load_reference_prediction_frame(
    loaded_model: LoadedRuntimeModel,
    *,
    row_offset: int,
    max_rows: int,
) -> pd.DataFrame:
    return _read_csv_rows(
        _resolve_reference_predictions_path(loaded_model),
        columns=(
            "source_row_idx",
            "row_index",
            "target",
            "sample_weight",
            "raw_probability",
            "calibrated_probability",
            "predicted_label",
        ),
        start_row=row_offset,
        max_rows=max_rows,
    )


def _load_prefill_feature_frame(
    loaded_model: LoadedRuntimeModel,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    if loaded_model.context_rows <= 0:
        return pd.DataFrame(columns=list(feature_columns))

    prefill_frame = _read_csv_tail_rows(
        _resolve_prepared_dir(loaded_model) / "val.csv",
        columns=feature_columns,
        max_rows=loaded_model.context_rows,
    )
    if len(prefill_frame) < loaded_model.context_rows:
        raise ValueError(
            f"Prepared val split for {loaded_model.model_id} only provided {len(prefill_frame)} context rows, "
            f"but runtime replay requires {loaded_model.context_rows}."
        )
    return prefill_frame.reset_index(drop=True)


def _prepared_feature_columns(loaded_model: LoadedRuntimeModel) -> tuple[str, ...]:
    columns = ["source_row_idx", *loaded_model.selected_feature_names]
    return tuple(dict.fromkeys(columns))


def _read_csv_rows(
    path: Path,
    *,
    columns: tuple[str, ...] | list[str],
    start_row: int,
    max_rows: int,
) -> pd.DataFrame:
    requested_columns = tuple(columns)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        available_columns = tuple(column for column in requested_columns if column in (reader.fieldnames or []))
        for row_index, row in enumerate(reader):
            if row_index < start_row:
                continue
            rows.append({column: row[column] for column in available_columns})
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows, columns=list(available_columns))


def _read_csv_tail_rows(
    path: Path,
    *,
    columns: tuple[str, ...] | list[str],
    max_rows: int,
) -> pd.DataFrame:
    requested_columns = tuple(columns)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        available_columns = tuple(column for column in requested_columns if column in (reader.fieldnames or []))
        tail_rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
        for row in reader:
            tail_rows.append({column: row[column] for column in available_columns})
    return pd.DataFrame(list(tail_rows), columns=list(available_columns))


def _resolve_artifact_threshold(loaded_model: LoadedRuntimeModel) -> float | None:
    threshold = loaded_model.manifest.artifact_test_metrics.threshold
    if threshold is not None:
        return float(threshold)

    training_summary_threshold = (
        loaded_model.training_summary.get("test_metrics", {}) if isinstance(loaded_model.training_summary, dict) else {}
    )
    resolved = training_summary_threshold.get("threshold")
    return float(resolved) if resolved is not None else None


def _resolve_repo_relative_path(path_value: str | Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _coerce_required_float(value: Any, column_name: str) -> float:
    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(coerced):
        raise ValueError(f"Replay reference column {column_name!r} contains a non-numeric value: {value!r}")
    return float(coerced)


def _coerce_required_int(value: Any, column_name: str) -> int:
    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(coerced):
        raise ValueError(f"Replay reference column {column_name!r} contains a non-numeric value: {value!r}")
    return int(coerced)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(coerced):
        return None
    return int(coerced)


def _synthetic_replay_timestamp(source_row_idx: int) -> datetime:
    return datetime(2000, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * int(source_row_idx))


def _update_max_diff(current: float | None, candidate: float) -> float:
    if current is None:
        return float(candidate)
    return max(float(current), float(candidate))
