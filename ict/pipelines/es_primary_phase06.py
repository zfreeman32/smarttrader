from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from features.io import save_dataset, standardize_market_frame
from ict.reports.leakage_control import (
    build_ict_event_window_frame,
    resolve_ict_recommended_embargo_bars,
    resolve_ict_swing_confirm_bars,
)
from preprocessing.config import ICT_TARGET_COLUMNS, PreprocessingConfig
from preprocessing.pipeline import FeaturePreprocessingPipeline

from .layout import build_ict_artifact_layout


DEFAULT_BASE_DIR = Path("artifacts")
DEFAULT_RUN_ID = "ict_es_primary"
DEFAULT_TARGET_COLUMNS = tuple(ICT_TARGET_COLUMNS)
ICT_HELPER_PREFIXES = (
    "label_",
    "label_quality_",
    "sample_weight_",
    "exclude_",
    "neg_ok_",
    "concurrency_",
    "htf_confluence_",
)
PHASE06_FORBIDDEN_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ts_event",
    "symbol",
    "instrument_id",
    "contract_id",
    "contract_symbol",
    "contract_expiration",
    "is_roll_boundary",
    "bars_since_roll",
    "market_day_close",
    "market_day_index",
    "in_roll_bracket",
}


@dataclass(frozen=True)
class ICTESPrimaryPhase06Config:
    feature_csv_path: Path
    labels_csv_path: Path
    run_id: str = DEFAULT_RUN_ID
    base_dir: Path = DEFAULT_BASE_DIR
    feature_metadata_path: Path | None = None
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS
    min_usable_rows: int = 250
    min_train_rows: int = 100
    min_positive_samples: int = 25
    top_n_features: int = 25
    scaler_type: str = "none"


def run(config: ICTESPrimaryPhase06Config) -> dict[str, Any]:
    feature_csv_path = Path(config.feature_csv_path)
    labels_csv_path = Path(config.labels_csv_path)
    feature_metadata_path = _resolve_feature_metadata_path(
        feature_csv_path,
        config.feature_metadata_path,
    )

    layout = build_ict_artifact_layout(
        config.run_id,
        base_dir=config.base_dir,
        ensure_directories=True,
    )
    phase04_dir = layout.phase04_prepared
    prepared_root = phase04_dir / "prepared"

    feature_dataset, feature_metadata = _load_existing_feature_dataset(
        feature_csv_path,
        feature_metadata_path,
    )
    labels_dataset = _load_ict_label_dataset(labels_csv_path)
    _validate_unique_datetimes(feature_dataset, path=feature_csv_path, frame_name="feature dataset")
    _validate_unique_datetimes(labels_dataset, path=labels_csv_path, frame_name="ICT labels")

    merged = feature_dataset.merge(
        labels_dataset,
        on="datetime",
        how="inner",
        validate="one_to_one",
    )
    cleaned_dataset, cleaned_metadata = _build_clean_phase06_dataset(
        merged,
        feature_metadata=feature_metadata,
        feature_source_csv=feature_csv_path,
        feature_source_metadata=feature_metadata_path,
        label_source_csv=labels_csv_path,
    )
    cleaned_csv_path, cleaned_metadata_path = save_dataset(
        cleaned_dataset,
        cleaned_metadata,
        phase04_dir / "ict_es_phase06_merged_dataset.csv",
    )
    leakage_control_contract = _resolve_ict_leakage_control_contract(
        labels_csv_path=labels_csv_path,
        phase02_metadata=feature_metadata,
        target_columns=config.target_columns,
    )

    preprocessing_summary = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=list(config.target_columns),
            load_time_column=False,
            scaler_type=str(config.scaler_type),
            min_usable_rows=int(config.min_usable_rows),
            min_train_rows=int(config.min_train_rows),
            min_positive_samples=int(config.min_positive_samples),
            top_n_features=int(config.top_n_features),
            target_split_embargo_bars=dict(leakage_control_contract.get("target_split_embargo_bars", {})),
        )
    ).run(
        cleaned_csv_path,
        prepared_root,
        metadata_path=cleaned_metadata_path,
    )
    preprocessing_summary["output_dir"] = str(prepared_root)

    summary = {
        "config": _make_json_safe(asdict(config)),
        "artifact_root": str(layout.root),
        "phase04_prepared_dir": str(phase04_dir),
        "prepared_root": str(prepared_root),
        "rows": {
            "feature_rows": int(len(feature_dataset)),
            "label_rows": int(len(labels_dataset)),
            "merged_rows": int(len(merged)),
            "cleaned_rows": int(len(cleaned_dataset)),
        },
        "artifacts": {
            "feature_csv": str(feature_csv_path),
            "feature_metadata": str(feature_metadata_path),
            "labels_csv": str(labels_csv_path),
            "cleaned_dataset_csv": str(cleaned_csv_path),
            "cleaned_dataset_metadata": str(cleaned_metadata_path),
            "prepared_summary": str(prepared_root / "summary.json"),
        },
        "leakage_control": leakage_control_contract,
        "preprocessing_summary": preprocessing_summary,
    }
    summary_path = phase04_dir / "phase06_summary.json"
    summary_path.write_text(json.dumps(_make_json_safe(summary), indent=2), encoding="utf-8")
    summary["artifacts"]["phase06_summary"] = str(summary_path)
    return _make_json_safe(summary)


def _resolve_feature_metadata_path(
    feature_csv_path: Path,
    configured_path: Path | None,
) -> Path:
    if configured_path is not None:
        metadata_path = Path(configured_path)
    else:
        metadata_path = feature_csv_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Feature metadata sidecar was not found: {metadata_path}")
    return metadata_path


def _load_existing_feature_dataset(
    feature_csv_path: Path,
    feature_metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
    usecols = {"datetime", *feature_columns}
    sample = pd.read_csv(
        feature_csv_path,
        nrows=64,
        usecols=lambda column: str(column) in usecols,
    )
    dtype_map: dict[str, Any] = {}
    for column in feature_columns:
        if column not in sample.columns:
            continue
        if pd.api.types.is_numeric_dtype(sample[column]):
            dtype_map[column] = np.float32
    dataset = pd.read_csv(
        feature_csv_path,
        usecols=lambda column: str(column) in usecols,
        dtype=dtype_map,
    )
    if "datetime" not in dataset.columns:
        raise ValueError(f"Feature dataset is missing a datetime column: {feature_csv_path}")
    dataset = standardize_market_frame(dataset)
    return dataset, metadata


def _load_ict_label_dataset(labels_csv_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(labels_csv_path)
    if "datetime" not in dataset.columns:
        unnamed_candidates = [column for column in dataset.columns if str(column).startswith("Unnamed:")]
        if unnamed_candidates:
            dataset = dataset.rename(columns={unnamed_candidates[0]: "datetime"})
    if "datetime" not in dataset.columns and len(dataset.columns) > 0:
        candidate = str(dataset.columns[0])
        parsed = pd.to_datetime(dataset.iloc[:, 0], errors="coerce", utc=True)
        if int(parsed.notna().sum()) >= max(1, int(len(parsed) * 0.8)):
            dataset = dataset.rename(columns={candidate: "datetime"})

    dataset = standardize_market_frame(dataset)
    if "datetime" not in dataset.columns:
        raise ValueError(f"ICT label dataset is missing a datetime column: {labels_csv_path}")
    return dataset


def _validate_unique_datetimes(df: pd.DataFrame, *, path: Path, frame_name: str) -> None:
    duplicated = df["datetime"].duplicated(keep=False)
    if bool(duplicated.any()):
        duplicate_count = int(duplicated.sum())
        raise ValueError(
            f"{frame_name} contains duplicate datetime rows ({duplicate_count}) and cannot be merged one-to-one: {path}"
        )


def _build_clean_phase06_dataset(
    merged: pd.DataFrame,
    *,
    feature_metadata: dict[str, Any],
    feature_source_csv: Path,
    feature_source_metadata: Path,
    label_source_csv: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    generated_features = [
        column
        for column in feature_metadata.get("feature_columns", [])
        if column in merged.columns and column not in PHASE06_FORBIDDEN_COLUMNS
    ]
    helper_columns = [
        column
        for column in merged.columns
        if any(column.startswith(prefix) for prefix in ICT_HELPER_PREFIXES)
    ]
    carry_through_columns = [
        column
        for column in helper_columns
        if column.startswith("htf_confluence_")
    ]
    label_helper_columns = [column for column in helper_columns if column not in carry_through_columns]

    keep = [
        "datetime",
        *generated_features,
        *carry_through_columns,
        *label_helper_columns,
        "warmup_mask",
    ]
    keep = [column for column in list(dict.fromkeys(keep)) if column in merged.columns]
    cleaned = merged.loc[:, keep].copy()

    upstream_source_path = feature_metadata.get("upstream_source_path", feature_metadata.get("source_path"))
    upstream_metadata_file = feature_metadata.get(
        "upstream_metadata_file",
        feature_metadata.get("source_metadata_file"),
    )
    upstream_timezone_contract = feature_metadata.get(
        "upstream_timezone_contract",
        feature_metadata.get("timezone_contract", {}),
    )

    metadata = {
        "feature_columns": generated_features,
        "timezone_contract": feature_metadata.get("timezone_contract", {}),
        "source_path": str(feature_source_csv),
        "source_metadata_file": str(feature_source_metadata),
        "upstream_source_path": upstream_source_path,
        "upstream_metadata_file": upstream_metadata_file,
        "upstream_bar_timestamp_semantics": feature_metadata.get("upstream_bar_timestamp_semantics"),
        "upstream_timezone_contract": upstream_timezone_contract,
        "config": {
            "drop_warmup_rows": False,
            "warmup_rows": 0,
            "fillna_numeric": False,
        },
        "prepared_contract": {
            "datetime_column": "datetime",
            "duplicate_time_columns_removed": ["ts_event", "timestamp"],
            "raw_price_columns_removed": ["open", "high", "low", "close", "volume"],
            "roll_lineage_columns_removed": sorted(PHASE06_FORBIDDEN_COLUMNS.intersection(merged.columns)),
            "target_context_columns": carry_through_columns,
            "label_source_path": str(label_source_csv),
            "label_columns_present": sorted(column for column in merged.columns if str(column).startswith("label_")),
        },
    }
    return cleaned, metadata


def _resolve_ict_leakage_control_contract(
    *,
    labels_csv_path: Path,
    phase02_metadata: dict[str, Any],
    target_columns: tuple[str, ...],
) -> dict[str, Any]:
    phase03_dir = labels_csv_path.parent
    events_csv_path = phase03_dir / "ict_es_events.csv"
    contract = {
        "available": False,
        "phase03_dir": str(phase03_dir),
        "events_csv": str(events_csv_path),
        "swing_confirm_bars": int(resolve_ict_swing_confirm_bars(phase02_metadata)),
        "target_split_embargo_bars": {},
        "overall_recommended_embargo_bars": 0,
        "formula": "recommended_embargo_bars = max(realized_window_bars) + swing_confirm_bars",
    }
    if not events_csv_path.exists():
        contract["reason"] = "missing_events_csv"
        return contract

    events = pd.read_csv(events_csv_path)
    event_windows = build_ict_event_window_frame(events)
    swing_confirm_bars = int(contract["swing_confirm_bars"])
    target_names = [
        _target_column_to_target_name(column)
        for column in target_columns
        if _target_column_to_target_name(column) is not None
    ]

    target_embargo_bars: dict[str, int] = {}
    for target_name in list(dict.fromkeys(target_names)):
        target_events = event_windows.loc[event_windows["target_name"].eq(target_name)].copy()
        target_embargo_bars[target_name] = int(
            resolve_ict_recommended_embargo_bars(
                target_events,
                swing_confirm_bars=swing_confirm_bars,
            )
        )

    contract["available"] = True
    contract["target_split_embargo_bars"] = target_embargo_bars
    contract["overall_recommended_embargo_bars"] = (
        max(target_embargo_bars.values()) if target_embargo_bars else int(max(1, swing_confirm_bars))
    )
    return contract


def _target_column_to_target_name(target_column: str) -> str | None:
    if not str(target_column).startswith("label_"):
        return None
    return str(target_column).removeprefix("label_")


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ICT Phase 6 prepared-dataset path from an existing feature dataset plus ICT labels."
    )
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--feature-metadata", default=None)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--target", action="append", default=None, help="Specific ICT target column(s) to prepare.")
    parser.add_argument("--min-usable-rows", type=int, default=250)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-positive-samples", type=int, default=25)
    parser.add_argument("--top-n-features", type=int, default=25)
    parser.add_argument("--scaler", default="none", choices=["none", "robust", "standard"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run(
        ICTESPrimaryPhase06Config(
            feature_csv_path=Path(args.feature_csv),
            labels_csv_path=Path(args.labels_csv),
            feature_metadata_path=Path(args.feature_metadata) if args.feature_metadata else None,
            run_id=str(args.run_id).strip(),
            base_dir=Path(args.base_dir),
            target_columns=tuple(args.target or DEFAULT_TARGET_COLUMNS),
            min_usable_rows=int(args.min_usable_rows),
            min_train_rows=int(args.min_train_rows),
            min_positive_samples=int(args.min_positive_samples),
            top_n_features=int(args.top_n_features),
            scaler_type=str(args.scaler),
        )
    )
    print(json.dumps(_make_json_safe(summary), indent=2))
    return 0


__all__ = ["ICTESPrimaryPhase06Config", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
