from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ict.pipelines.layout import build_ict_artifact_layout
from ict.reports.leakage_control import (
    build_ict_event_window_frame,
    resolve_ict_recommended_embargo_bars,
    resolve_ict_swing_confirm_bars,
)
from preprocessing.config import PreprocessingConfig
from preprocessing.pipeline import FeaturePreprocessingPipeline


DEFAULT_BASE_ROOT = Path("artifacts/ict_es_primary_refresh_20260724_spacing_refit_final_confirm")
DEFAULT_RUN_ID = "ict_short_setup_targets_20260811_audit"
DEFAULT_DIRECTION = "short"
SETUP_TARGET_LABEL_FAMILIES = frozenset({"ict_reversal", "ict_continuation", "ict_classic_breaker"})


def _normalize_setup_type(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "" if text in {"", "none", "nan"} else text


def _as_bool_series(values: object, *, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _as_bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _as_float_value(value: object, *, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return float(default)
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(default)
    return resolved if np.isfinite(resolved) else float(default)


def _as_int_value(value: object, *, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _positive_event_mask(events: pd.DataFrame) -> pd.Series:
    return events.get("tb_outcome", "").fillna("").astype(str).str.lower().isin(
        {"tp", "timeout_profit"}
    )


def _selected_events(events: pd.DataFrame, direction: str) -> pd.DataFrame:
    working = events.copy()
    working["excluded"] = _as_bool_series(working.get("excluded", False), index=working.index)
    working["event_direction"] = working.get("event_direction", "").fillna("").astype(str).str.strip().str.lower()
    working["label_family"] = working.get("label_family", "").fillna("").astype(str).str.strip().str.lower()
    working["setup_type"] = working.get("setup_type", "").map(_normalize_setup_type)
    working = working.loc[
        (~working["excluded"])
        & working["event_direction"].eq(direction)
        & working["label_family"].isin(SETUP_TARGET_LABEL_FAMILIES)
        & working["setup_type"].ne("")
    ].copy()
    working["signal_index"] = pd.to_numeric(working["signal_index"], errors="coerce")
    working = working.loc[working["signal_index"].notna()].copy()
    working["signal_index"] = working["signal_index"].astype(np.int64)
    working["positive_event"] = _positive_event_mask(working)
    return working


def _target_name(direction: str, label_family: str, setup_type: str) -> str:
    label_family = str(label_family).strip().lower()
    if label_family == "ict_classic_breaker":
        return f"{direction}_{label_family}"
    return f"{direction}_{label_family}_{setup_type}"


def _target_columns(events: pd.DataFrame, direction: str) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for label_family, setup_type in (
        events.loc[:, ["label_family", "setup_type"]]
        .drop_duplicates()
        .sort_values(["label_family", "setup_type"])
        .itertuples(index=False, name=None)
    ):
        target_column = f"label_{_target_name(direction, str(label_family), str(setup_type))}"
        if target_column in seen:
            continue
        seen.add(target_column)
        targets.append(target_column)
    return targets


def _copy_phase03_events(*, source_events_path: Path, output_root: Path) -> Path:
    destination = output_root / "phase03_labeling" / "ict_es_events.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_events_path, destination)
    return destination


def _resolve_source_events_path(source_phase03_dir: Path) -> Path:
    if source_phase03_dir.is_file():
        return source_phase03_dir
    return source_phase03_dir / "ict_es_events.csv"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _materialize_labels(
    *,
    base_dataset: pd.DataFrame,
    events: pd.DataFrame,
    direction: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = base_dataset.copy()
    if "source_row_idx" not in out.columns:
        out.insert(0, "source_row_idx", np.arange(len(out), dtype=np.int64))

    source_to_position = {
        int(source_row_idx): position
        for position, source_row_idx in enumerate(pd.to_numeric(out["source_row_idx"], errors="coerce"))
        if pd.notna(source_row_idx)
    }

    target_columns = _target_columns(events, direction)
    target_names = [column.removeprefix("label_") for column in target_columns]
    for target_name in target_names:
        out[f"label_{target_name}"] = np.zeros(len(out), dtype=np.int8)
        out[f"label_quality_{target_name}"] = np.zeros(len(out), dtype=np.float32)
        out[f"sample_weight_{target_name}"] = np.ones(len(out), dtype=np.float32)
        out[f"exclude_{target_name}"] = np.ones(len(out), dtype=bool)
        out[f"neg_ok_{target_name}"] = np.zeros(len(out), dtype=bool)
        out[f"concurrency_{target_name}"] = np.zeros(len(out), dtype=np.int16)
        out[f"htf_confluence_{target_name}"] = np.zeros(len(out), dtype=np.int8)

    missing_signal_rows = 0
    for row in events.itertuples(index=False):
        signal_index = int(getattr(row, "signal_index"))
        position = source_to_position.get(signal_index)
        if position is None:
            missing_signal_rows += 1
            continue

        target_name = _target_name(
            direction,
            str(getattr(row, "label_family")),
            str(getattr(row, "setup_type")),
        )
        out.iat[position, out.columns.get_loc(f"label_{target_name}")] = int(bool(getattr(row, "positive_event")))
        label_quality = _as_float_value(getattr(row, "label_quality", 0.0), default=0.0)
        out.iat[position, out.columns.get_loc(f"label_quality_{target_name}")] = label_quality
        sample_weight = getattr(row, "sample_weight", np.nan)
        if sample_weight is None or pd.isna(sample_weight):
            sample_weight = max(label_quality or 1.0, 0.1)
        out.iat[position, out.columns.get_loc(f"sample_weight_{target_name}")] = _as_float_value(
            sample_weight,
            default=1.0,
        )
        out.iat[position, out.columns.get_loc(f"exclude_{target_name}")] = False
        out.iat[position, out.columns.get_loc(f"neg_ok_{target_name}")] = True
        out.iat[position, out.columns.get_loc(f"concurrency_{target_name}")] = _as_int_value(
            getattr(row, "ict_concurrency", 0),
            default=0,
        )
        out.iat[position, out.columns.get_loc(f"htf_confluence_{target_name}")] = int(
            _as_bool_value(getattr(row, "htf_confluence_flag", False))
        )

    counts: dict[str, Any] = {"missing_signal_rows": int(missing_signal_rows), "targets": {}}
    for target_name in target_names:
        usable = ~out[f"exclude_{target_name}"].astype(bool)
        positives = pd.to_numeric(out.loc[usable, f"label_{target_name}"], errors="coerce").fillna(0).astype(int)
        counts["targets"][target_name] = {
            "usable_rows": int(usable.sum()),
            "positive_rows": int(positives.sum()),
            "positive_rate": float(positives.mean()) if len(positives) else 0.0,
        }
    return out, target_columns, counts


def _embargo_map(
    *,
    events: pd.DataFrame,
    target_columns: list[str],
    feature_metadata: dict[str, Any],
) -> dict[str, int]:
    event_windows = build_ict_event_window_frame(events)
    swing_confirm_bars = resolve_ict_swing_confirm_bars(feature_metadata)
    result: dict[str, int] = {}
    for target_column in target_columns:
        target_name = target_column.removeprefix("label_")
        target_events = event_windows.loc[event_windows["target_name"].eq(target_name)]
        result[target_name] = int(
            resolve_ict_recommended_embargo_bars(
                target_events,
                swing_confirm_bars=swing_confirm_bars,
            )
        )
    return result


def materialize_ict_setup_prepared_root(
    *,
    base_root: Path,
    source_phase03_dir: Path | None,
    artifact_base_dir: Path | None,
    run_id: str,
    direction: str,
    min_usable_rows: int,
    min_train_rows: int,
    min_positive_samples: int,
    top_n_features: int,
) -> dict[str, Any]:
    direction = str(direction).strip().lower()
    if direction not in {"long", "short"}:
        raise ValueError("direction must be either 'long' or 'short'.")

    base_root = Path(base_root)
    base_phase04 = base_root / "phase04_prepared"
    base_dataset_path = base_phase04 / "ict_es_phase06_merged_dataset.csv"
    base_metadata_path = base_phase04 / "ict_es_phase06_merged_dataset.metadata.json"
    source_phase03_dir = Path(source_phase03_dir) if source_phase03_dir is not None else base_root / "phase03_labeling"
    source_events_path = _resolve_source_events_path(source_phase03_dir)
    if not base_dataset_path.exists():
        raise FileNotFoundError(base_dataset_path)
    if not base_metadata_path.exists():
        raise FileNotFoundError(base_metadata_path)
    if not source_events_path.exists():
        raise FileNotFoundError(source_events_path)

    layout = build_ict_artifact_layout(
        run_id,
        base_dir=artifact_base_dir or Path("artifacts"),
        ensure_directories=True,
    )
    copied_events_path = _copy_phase03_events(source_events_path=source_events_path, output_root=layout.root)

    base_dataset = pd.read_csv(base_dataset_path)
    feature_metadata = _load_json(base_metadata_path)
    events = _selected_events(pd.read_csv(source_events_path), direction)
    if events.empty:
        raise ValueError(f"No usable {direction} ICT setup events found in {source_events_path}.")

    setup_dataset, target_columns, count_summary = _materialize_labels(
        base_dataset=base_dataset,
        events=events,
        direction=direction,
    )
    output_dataset_path = layout.phase04_prepared / "ict_es_setup_phase06_merged_dataset.csv"
    output_metadata_path = output_dataset_path.with_suffix(".metadata.json")
    setup_dataset.to_csv(output_dataset_path, index=False)

    metadata = dict(feature_metadata)
    prepared_contract = dict(metadata.get("prepared_contract") or {})
    target_context_columns = [
        str(column)
        for column in prepared_contract.get("target_context_columns", [])
        if str(column).strip()
    ]
    if "source_row_idx" not in target_context_columns:
        target_context_columns.append("source_row_idx")
    prepared_contract.update(
        {
            "setup_target_source_root": str(base_root),
            "setup_target_source_phase03_dir": str(source_phase03_dir),
            "setup_target_source_events": str(source_events_path),
            "setup_target_copied_events": str(copied_events_path),
            "setup_target_direction": direction,
            "setup_target_columns": target_columns,
            "target_context_columns": target_context_columns,
        }
    )
    metadata["prepared_contract"] = prepared_contract
    _write_json(output_metadata_path, metadata)

    target_split_embargo_bars = _embargo_map(
        events=events,
        target_columns=target_columns,
        feature_metadata=feature_metadata,
    )
    prepared_root = layout.phase04_prepared / "prepared"
    preprocessing_summary = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=target_columns,
            load_time_column=False,
            scaler_type="none",
            min_usable_rows=int(min_usable_rows),
            min_train_rows=int(min_train_rows),
            min_positive_samples=int(min_positive_samples),
            top_n_features=int(top_n_features),
            target_split_embargo_bars=target_split_embargo_bars,
        )
    ).run(
        output_dataset_path,
        prepared_root,
        metadata_path=output_metadata_path,
    )

    summary = {
        "run_id": run_id,
        "artifact_base_dir": str(artifact_base_dir or Path("artifacts")),
        "base_root": str(base_root),
        "source_phase03_dir": str(source_phase03_dir),
        "direction": direction,
        "event_rows": int(len(events)),
        "target_columns": target_columns,
        "target_split_embargo_bars": target_split_embargo_bars,
        "counts": count_summary,
        "artifacts": {
            "events_csv": str(copied_events_path),
            "merged_dataset_csv": str(output_dataset_path),
            "merged_dataset_metadata": str(output_metadata_path),
            "prepared_root": str(prepared_root),
            "prepared_summary": str(prepared_root / "summary.json"),
        },
        "preprocessing_summary": preprocessing_summary,
    }
    summary_path = layout.phase04_prepared / "ict_setup_phase06_summary.json"
    _write_json(summary_path, summary)
    summary["artifacts"]["setup_phase06_summary"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize ICT setup-specific prepared targets from an existing ICT Phase 6 root."
    )
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--artifact-base-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--source-phase03-dir",
        type=Path,
        default=None,
        help="Phase 3 directory or events CSV to use for setup labels. Defaults to <base-root>/phase03_labeling.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--direction", choices=("long", "short"), default=DEFAULT_DIRECTION)
    parser.add_argument("--min-usable-rows", type=int, default=250)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-positive-samples", type=int, default=25)
    parser.add_argument("--top-n-features", type=int, default=25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = materialize_ict_setup_prepared_root(
        base_root=args.base_root,
        source_phase03_dir=args.source_phase03_dir,
        artifact_base_dir=args.artifact_base_dir,
        run_id=args.run_id,
        direction=args.direction,
        min_usable_rows=args.min_usable_rows,
        min_train_rows=args.min_train_rows,
        min_positive_samples=args.min_positive_samples,
        top_n_features=args.top_n_features,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
