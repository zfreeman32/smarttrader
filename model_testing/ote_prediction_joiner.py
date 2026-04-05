from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from features.io import standardize_market_frame
from features.preprocessing import FeaturePreprocessingPipeline, PreprocessingConfig
from preprocessing.feature_selection import (
    build_split_indices,
    build_target_context,
    discover_targets,
    ordered_usable_positions,
    resolve_source_row_idx,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TARGETS = {"long_entry", "short_entry", "long_ote", "short_ote"}


def join_prediction_file(
    prediction_path: str | Path,
    *,
    output_path: str | Path | None = None,
    source_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    target_name: str | None = None,
    prepared_root: str | Path | None = None,
    source_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    prediction_path = Path(prediction_path)
    predictions = pd.read_csv(prediction_path)
    target_name = target_name or _infer_target_name(prediction_path)
    source_file = Path(source_file) if source_file is not None else None
    metadata_file = Path(metadata_file) if metadata_file is not None else None
    prepared_root = Path(prepared_root) if prepared_root is not None else None

    resolved_source_row_idx = resolve_prediction_source_row_idx(
        predictions,
        prediction_path=prediction_path,
        target_name=target_name,
        source_file=source_file,
        metadata_file=metadata_file,
        prepared_root=prepared_root,
    )

    source_frame = load_source_frame(
        prediction_path=prediction_path,
        source_file=source_file,
        source_columns=source_columns,
    )
    source_subset = select_source_columns(source_frame, source_columns)

    joined = predictions.copy()
    joined["source_row_idx"] = resolved_source_row_idx
    joined = joined.merge(source_subset, on="source_row_idx", how="left", validate="many_to_one")
    joined = reorder_joined_columns(joined, predictions.columns)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(output_path, index=False)

    return joined


def resolve_prediction_source_row_idx(
    predictions: pd.DataFrame,
    *,
    prediction_path: Path,
    target_name: str,
    source_file: Path | None = None,
    metadata_file: Path | None = None,
    prepared_root: Path | None = None,
) -> np.ndarray:
    direct = _extract_direct_source_row_idx(predictions)
    if direct is not None:
        return direct

    mapping = reconstruct_source_row_idx_mapping(
        prediction_path=prediction_path,
        target_name=target_name,
        source_file=source_file,
        metadata_file=metadata_file,
        prepared_root=prepared_root,
    )

    if "row_index_in_dev" in predictions.columns:
        row_index = pd.to_numeric(predictions["row_index_in_dev"], errors="raise").astype(np.int64)
        dev_source_row_idx = np.concatenate(
            [
                mapping["ordered_source_row_idx"][mapping["split_indices"]["train"]],
                mapping["ordered_source_row_idx"][mapping["split_indices"]["val"]],
            ]
        )
        return _resolve_row_index_lookup(row_index.to_numpy(copy=True), dev_source_row_idx, "row_index_in_dev")

    if "row_index" in predictions.columns:
        row_index = pd.to_numeric(predictions["row_index"], errors="raise").astype(np.int64)
        test_source_row_idx = mapping["ordered_source_row_idx"][mapping["split_indices"]["test"]]
        return _resolve_row_index_lookup(row_index.to_numpy(copy=True), test_source_row_idx, "row_index")

    raise ValueError(
        "Prediction file must contain either 'source_row_idx', 'row_index_in_dev', or 'row_index'."
    )


def reconstruct_source_row_idx_mapping(
    *,
    prediction_path: Path,
    target_name: str,
    source_file: Path | None = None,
    metadata_file: Path | None = None,
    prepared_root: Path | None = None,
) -> Dict[str, Any]:
    if target_name not in SUPPORTED_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_TARGETS))
        raise ValueError(f"Unsupported target_name={target_name!r}. Supported values: {supported}")

    prepared_root = prepared_root or infer_prepared_root(prediction_path)
    summary = load_prepared_summary(prepared_root)

    source_file = _resolve_repo_path(source_file or summary["input_file"])
    metadata_file = _resolve_repo_path(metadata_file or summary["metadata_file"])

    config = PreprocessingConfig(**summary.get("config", {}))
    pipeline = FeaturePreprocessingPipeline(config)

    source_frame = _load_mapping_source_frame(
        source_file=source_file,
        target_name=target_name,
        time_column=config.time_column,
    )

    metadata = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    upstream_info = pipeline._detect_upstream_preprocessing(metadata)
    source_row_idx = resolve_source_row_idx(source_frame)
    source_frame, source_row_idx, _ = pipeline._apply_row_windowing(
        source_frame,
        source_row_idx,
        upstream_info,
    )

    target_specs = {spec.name: spec for spec in discover_targets(source_frame, config)}
    if target_name not in target_specs:
        available = ", ".join(sorted(target_specs))
        raise ValueError(f"Target {target_name!r} was not found in source dataset. Available targets: {available}")

    spec = target_specs[target_name]
    target_context = build_target_context(source_frame, [spec], config.time_column)
    target_context["source_row_idx"] = source_row_idx.to_numpy(copy=False)
    target_raw = pd.to_numeric(target_context[spec.target_column], errors="coerce")
    positive_mask = target_raw.fillna(0).astype(float) > 0
    warmup_mask = (
        target_context["warmup_mask"].fillna(False).astype(bool)
        if "warmup_mask" in target_context.columns
        else pd.Series(False, index=target_context.index)
    )

    negative_mask = ~positive_mask
    if spec.safe_negative_column:
        negative_mask &= target_context[spec.safe_negative_column].fillna(False).astype(bool)
    if spec.exclude_column:
        negative_mask &= ~target_context[spec.exclude_column].fillna(False).astype(bool)

    usable_mask = ~warmup_mask & (positive_mask | negative_mask)
    ordered_positions, _ = ordered_usable_positions(target_context, usable_mask, config.time_column)
    ordered_source_row_idx = (
        target_context["source_row_idx"]
        .iloc[ordered_positions]
        .astype(np.int64)
        .to_numpy(copy=True)
    )
    split_indices = build_split_indices(len(ordered_positions), config)

    return {
        "prepared_root": prepared_root,
        "source_file": source_file,
        "metadata_file": metadata_file,
        "ordered_source_row_idx": ordered_source_row_idx,
        "split_indices": split_indices,
    }


def load_source_frame(
    *,
    prediction_path: Path,
    source_file: Path | None = None,
    source_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if source_file is None:
        prepared_root = infer_prepared_root(prediction_path)
        summary = load_prepared_summary(prepared_root)
        source_file = _resolve_repo_path(summary["input_file"])

    read_csv_kwargs: Dict[str, Any] = {}
    usecols = _build_source_usecols(source_columns)
    if usecols is not None:
        read_csv_kwargs["usecols"] = usecols

    source_frame = pd.read_csv(source_file, **read_csv_kwargs)
    source_frame = standardize_market_frame(source_frame)
    if "source_row_idx" not in source_frame.columns:
        source_frame = source_frame.copy()
        source_frame.insert(0, "source_row_idx", np.arange(len(source_frame), dtype=np.int64))
    else:
        source_frame["source_row_idx"] = (
            pd.to_numeric(source_frame["source_row_idx"], errors="coerce").fillna(-1).astype(np.int64)
        )
    return source_frame


def load_source_columns(
    *,
    prediction_path: Path,
    source_file: Path | None = None,
) -> list[str]:
    if source_file is None:
        prepared_root = infer_prepared_root(prediction_path)
        summary = load_prepared_summary(prepared_root)
        source_file = _resolve_repo_path(summary["input_file"])

    source_header = pd.read_csv(source_file, nrows=0)
    source_header = standardize_market_frame(source_header)

    columns = list(source_header.columns)
    if "source_row_idx" not in columns:
        columns.insert(0, "source_row_idx")
    return columns


def select_source_columns(
    source_frame: pd.DataFrame,
    source_columns: Optional[Sequence[str]],
) -> pd.DataFrame:
    if source_columns is None:
        return source_frame

    requested = ["source_row_idx"] + [column for column in source_columns if column != "source_row_idx"]
    missing = [column for column in requested if column not in source_frame.columns]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(f"Requested source columns not found in source frame: {missing_csv}")
    return source_frame.loc[:, requested]


def reorder_joined_columns(joined: pd.DataFrame, prediction_columns: Iterable[str]) -> pd.DataFrame:
    preferred = [column for column in ("source_row_idx", "datetime", "open", "high", "low", "close", "volume") if column in joined.columns]
    prediction_cols = [column for column in prediction_columns if column in joined.columns and column not in preferred]
    remaining = [column for column in joined.columns if column not in preferred and column not in prediction_cols]
    return joined.loc[:, preferred + prediction_cols + remaining]


def infer_prepared_root(prediction_path: Path) -> Path:
    training_summary_path = prediction_path.parent / "training_summary.json"
    if not training_summary_path.exists():
        raise FileNotFoundError(
            "Could not infer prepared_root because training_summary.json was not found next to the prediction file."
        )

    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    prepared_dir = training_summary.get("prepared_dir")
    if not prepared_dir:
        raise ValueError(f"training_summary.json at {training_summary_path} is missing 'prepared_dir'.")
    return _resolve_repo_path(prepared_dir).parent


def load_prepared_summary(prepared_root: Path) -> Dict[str, Any]:
    summary_path = prepared_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Prepared dataset summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _extract_direct_source_row_idx(predictions: pd.DataFrame) -> Optional[np.ndarray]:
    if "source_row_idx" not in predictions.columns:
        return None

    source_row_idx = pd.to_numeric(predictions["source_row_idx"], errors="coerce")
    if source_row_idx.isna().any():
        return None

    source_row_idx = source_row_idx.astype(np.int64)
    if (source_row_idx < 0).any():
        return None
    return source_row_idx.to_numpy(copy=True)


def _resolve_row_index_lookup(row_index: np.ndarray, lookup: np.ndarray, column_name: str) -> np.ndarray:
    if row_index.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(row_index < 0):
        raise ValueError(f"Prediction column {column_name} contains negative indices.")
    if np.any(row_index >= len(lookup)):
        raise ValueError(
            f"Prediction column {column_name} contains indices outside the available range 0..{len(lookup) - 1}."
        )
    return np.ascontiguousarray(lookup[row_index], dtype=np.int64)


def _infer_target_name(prediction_path: Path) -> str:
    target_name = prediction_path.parent.name
    if target_name in SUPPORTED_TARGETS:
        return target_name
    raise ValueError(
        "Could not infer target_name from prediction path. Pass --target-name explicitly."
    )


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _build_source_usecols(
    source_columns: Optional[Sequence[str]],
):
    if source_columns is None:
        return None

    allowed = {column.strip().lower() for column in source_columns}
    allowed.update({"source_row_idx"})
    if "datetime" in allowed:
        allowed.update({"timestamp", "date", "time"})

    def include(raw_column: str) -> bool:
        return raw_column.strip().lower() in allowed

    return include


def _load_mapping_source_frame(
    *,
    source_file: Path,
    target_name: str,
    time_column: str,
) -> pd.DataFrame:
    source_frame = pd.read_csv(
        source_file,
        usecols=_build_mapping_usecols(target_name=target_name, time_column=time_column),
    )
    source_frame = standardize_market_frame(source_frame)
    if "source_row_idx" not in source_frame.columns:
        source_frame = source_frame.copy()
        source_frame.insert(0, "source_row_idx", np.arange(len(source_frame), dtype=np.int64))
    else:
        source_frame["source_row_idx"] = (
            pd.to_numeric(source_frame["source_row_idx"], errors="coerce").fillna(-1).astype(np.int64)
        )
    return source_frame


def _build_mapping_usecols(
    *,
    target_name: str,
    time_column: str,
):
    direction = "long" if target_name.startswith("long_") else "short"
    label_kind = "entry" if target_name.endswith("_entry") else "ote"
    target_column = f"label_{direction}_{label_kind}"
    sample_weight_column = f"sample_weight_{direction}"
    quality_column = f"label_quality_{direction}"
    if label_kind == "entry":
        sample_weight_column = f"sample_weight_entry_{direction}"
        quality_column = f"entry_quality_{direction}"

    allowed = {
        "source_row_idx",
        "warmup_mask",
        target_column,
        sample_weight_column,
        quality_column,
        f"exclude_{direction}",
        f"neg_ok_{direction}",
        time_column.strip().lower(),
    }
    if time_column.strip().lower() == "datetime":
        allowed.update({"datetime", "timestamp", "date", "time"})

    def include(raw_column: str) -> bool:
        return raw_column.strip().lower() in allowed

    return include


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join OTE prediction exports back to the timestamped source feature dataset."
    )
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--target-name", type=str, default=None)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument(
        "--source-columns",
        nargs="*",
        default=None,
        help="Optional subset of source columns to keep. Defaults to all source columns.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.prediction_path.with_name(f"{args.prediction_path.stem}.joined.csv")

    joined = join_prediction_file(
        args.prediction_path,
        output_path=output_path,
        source_file=args.source_file,
        metadata_file=args.metadata_file,
        target_name=args.target_name,
        prepared_root=args.prepared_root,
        source_columns=args.source_columns,
    )
    print(
        json.dumps(
            {
                "prediction_path": str(args.prediction_path),
                "output_path": str(output_path),
                "rows_joined": int(len(joined)),
                "columns_written": int(joined.shape[1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
