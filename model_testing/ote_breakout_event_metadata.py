from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from features.fx_calendar import normalize_datetime_series


REPO_ROOT = Path(__file__).resolve().parents[1]
BREAKOUT_LABEL_KINDS = frozenset({"breakout", "breakout_entry"})
BREAKOUT_EVENT_METADATA_COLUMNS = (
    "breakout_event_confirm_datetime",
    "breakout_event_confirm_index",
    "breakout_event_entry_datetime",
    "breakout_event_entry_index",
    "breakout_event_entry_price",
    "breakout_event_entry_score",
    "breakout_event_entry_delay_bars",
    "breakout_event_exit_datetime",
    "breakout_event_exit_index",
    "breakout_event_exit_price",
    "breakout_event_remaining_bars_after_entry",
    "breakout_event_tb_bars_held",
    "breakout_event_tb_outcome",
    "breakout_event_tb_return",
    "breakout_event_label_quality",
    "breakout_event_label_tier",
    "breakout_event_trade_available",
)


def enrich_prediction_frame_with_breakout_event_metadata(
    joined: pd.DataFrame,
    *,
    training_summary: Mapping[str, object],
    direction: str,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    if joined.empty or "datetime" not in joined.columns:
        return joined

    context = _resolve_breakout_context(
        training_summary,
        repo_root=repo_root or REPO_ROOT,
    )
    if context is None:
        return joined

    label_frame = _load_breakout_label_frame(
        context["label_path"],
        source_timezone=context["csv_timezone"],
        canonical_timezone=context["canonical_timezone"],
    )
    events_frame = _load_breakout_event_frame(
        context["events_path"],
        source_timezone=context["csv_timezone"],
        canonical_timezone=context["canonical_timezone"],
    )
    metadata = build_breakout_event_metadata_frame(
        label_frame,
        events_frame,
        direction=direction,
        label_kind=context["label_kind"],
        zone_pre_bars=context["zone_pre_bars"],
        zone_post_bars=context["zone_post_bars"],
        source_timezone=context["csv_timezone"],
        canonical_timezone=context["canonical_timezone"],
    )
    if metadata.empty:
        return joined

    working = joined.copy()
    working["__breakout_join_datetime"] = normalize_datetime_series(
        working["datetime"],
        source_timezone=context["csv_timezone"],
        canonical_timezone=context["canonical_timezone"],
    )
    merged = working.merge(
        metadata.rename(columns={"datetime": "__breakout_join_datetime"}),
        on="__breakout_join_datetime",
        how="left",
        validate="many_to_one",
    )
    return merged.drop(columns="__breakout_join_datetime")


def build_breakout_event_metadata_frame(
    label_frame: pd.DataFrame,
    events_frame: pd.DataFrame,
    *,
    direction: str,
    label_kind: str,
    zone_pre_bars: int,
    zone_post_bars: int,
    source_timezone: str = "UTC",
    canonical_timezone: str = "UTC",
) -> pd.DataFrame:
    if direction not in {"long", "short"}:
        raise ValueError(f"Unsupported breakout direction: {direction!r}")

    normalized_label_kind = str(label_kind).strip().lower()
    if normalized_label_kind not in BREAKOUT_LABEL_KINDS:
        return _empty_breakout_metadata_frame()

    labels = label_frame.copy()
    if "datetime" not in labels.columns and "timestamp" in labels.columns:
        labels = labels.rename(columns={"timestamp": "datetime"})
    if "datetime" not in labels.columns or "close" not in labels.columns:
        raise ValueError("Breakout label frame must include 'datetime' and 'close' columns.")

    labels["datetime"] = normalize_datetime_series(
        labels["datetime"],
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    labels["close"] = pd.to_numeric(labels["close"], errors="coerce")
    labels = labels.reset_index(drop=True)
    label_count = len(labels)
    if label_count == 0:
        return _empty_breakout_metadata_frame()

    events = events_frame.copy()
    if "label_family" in events.columns:
        events = events.loc[events["label_family"].astype(str).str.lower() == "breakout"].copy()
    if events.empty:
        return _empty_breakout_metadata_frame()

    direction_mask = _resolve_direction_mask(events, direction)
    events = events.loc[direction_mask].copy()
    if events.empty:
        return _empty_breakout_metadata_frame()

    for column in (
        "confirm_index",
        "entry_index",
        "entry_price",
        "entry_score",
        "tb_bars_held",
        "tb_return",
        "label_quality",
    ):
        if column in events.columns:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    for column in ("confirm_time", "entry_time"):
        if column in events.columns:
            events[column] = normalize_datetime_series(
                events[column],
                source_timezone=source_timezone,
                canonical_timezone=canonical_timezone,
            )

    assignments: list[dict[str, Any]] = []
    label_datetimes = labels["datetime"]
    label_closes = labels["close"]
    for event in events.to_dict(orient="records"):
        confirm_index = _optional_int(event.get("confirm_index"))
        if confirm_index is None or confirm_index < 0 or confirm_index >= label_count:
            continue

        entry_index = _optional_int(event.get("entry_index"))
        if entry_index is not None and (entry_index < 0 or entry_index >= label_count):
            entry_index = None

        mapped_row_indices = _resolve_mapped_label_rows(
            confirm_index=confirm_index,
            entry_index=entry_index,
            label_kind=normalized_label_kind,
            zone_pre_bars=zone_pre_bars,
            zone_post_bars=zone_post_bars,
            label_count=label_count,
        )
        if not mapped_row_indices:
            continue

        confirm_datetime = _coalesce_timestamp(event.get("confirm_time"), label_datetimes.iat[confirm_index])
        entry_datetime = _coalesce_timestamp(
            event.get("entry_time"),
            _value_at_position(label_datetimes, entry_index),
        )
        entry_price = _coalesce_numeric(
            event.get("entry_price"),
            _value_at_position(label_closes, entry_index),
        )
        tb_bars_held = _optional_int(event.get("tb_bars_held"))
        entry_delay_bars = None if entry_index is None else int(entry_index - confirm_index)
        remaining_bars_after_entry = None
        exit_index = None
        if tb_bars_held is not None:
            if entry_index is not None:
                effective_entry_delay = max(entry_delay_bars or 0, 0)
                remaining_bars_after_entry = max(tb_bars_held - effective_entry_delay, 0)
                exit_index = min(entry_index + remaining_bars_after_entry, label_count - 1)
            else:
                remaining_bars_after_entry = max(tb_bars_held, 0)
                exit_index = min(confirm_index + remaining_bars_after_entry, label_count - 1)
        exit_datetime = _value_at_position(label_datetimes, exit_index)
        exit_price = _value_at_position(label_closes, exit_index)

        base_trade_available = (
            entry_index is not None
            and exit_index is not None
            and pd.notna(entry_datetime)
            and pd.notna(exit_datetime)
            and entry_price is not None
            and exit_price is not None
        )

        for mapped_row_index in mapped_row_indices:
            assignments.append(
                {
                    "datetime": label_datetimes.iat[mapped_row_index],
                    "breakout_event_confirm_datetime": confirm_datetime,
                    "breakout_event_confirm_index": confirm_index,
                    "breakout_event_entry_datetime": entry_datetime,
                    "breakout_event_entry_index": entry_index,
                    "breakout_event_entry_price": entry_price,
                    "breakout_event_entry_score": _optional_float(event.get("entry_score")),
                    "breakout_event_entry_delay_bars": entry_delay_bars,
                    "breakout_event_exit_datetime": exit_datetime,
                    "breakout_event_exit_index": exit_index,
                    "breakout_event_exit_price": _optional_float(exit_price),
                    "breakout_event_remaining_bars_after_entry": remaining_bars_after_entry,
                    "breakout_event_tb_bars_held": tb_bars_held,
                    "breakout_event_tb_outcome": _optional_string(event.get("tb_outcome")),
                    "breakout_event_tb_return": _optional_float(event.get("tb_return")),
                    "breakout_event_label_quality": _optional_float(event.get("label_quality")),
                    "breakout_event_label_tier": _optional_string(event.get("label_tier")),
                    "breakout_event_trade_available": bool(
                        base_trade_available and entry_index is not None and mapped_row_index <= entry_index
                    ),
                    "__mapped_label_row_idx": mapped_row_index,
                }
            )

    if not assignments:
        return _empty_breakout_metadata_frame()

    metadata = pd.DataFrame(assignments)
    metadata["__trade_available_rank"] = metadata["breakout_event_trade_available"].fillna(False).astype(int)
    metadata["__label_quality_rank"] = pd.to_numeric(
        metadata["breakout_event_label_quality"],
        errors="coerce",
    ).fillna(-np.inf)
    metadata["__entry_score_rank"] = pd.to_numeric(
        metadata["breakout_event_entry_score"],
        errors="coerce",
    ).fillna(-np.inf)
    metadata = metadata.sort_values(
        [
            "__mapped_label_row_idx",
            "__trade_available_rank",
            "__label_quality_rank",
            "__entry_score_rank",
            "breakout_event_confirm_index",
        ],
        ascending=[True, False, False, False, True],
    )
    metadata = metadata.drop_duplicates(subset="__mapped_label_row_idx", keep="first")
    metadata = metadata.sort_values("datetime").reset_index(drop=True)
    return metadata.loc[:, ["datetime", *BREAKOUT_EVENT_METADATA_COLUMNS]]


def _resolve_breakout_context(
    training_summary: Mapping[str, object],
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    report = _mapping(training_summary.get("report"))
    prepared_summary = _mapping(training_summary.get("prepared_summary"))
    source_lineage = _mapping(report.get("source_lineage")) or _mapping(prepared_summary.get("source_lineage"))

    label_kind = str(report.get("label_kind") or "").strip().lower()
    if not label_kind:
        target_name = str(training_summary.get("target") or "").strip().lower()
        if "breakout_entry" in target_name:
            label_kind = "breakout_entry"
        elif "breakout" in target_name:
            label_kind = "breakout"
    if label_kind not in BREAKOUT_LABEL_KINDS:
        return None

    label_metadata_path = _resolve_repo_path(
        source_lineage.get("feature_builder_source_metadata_file")
        or source_lineage.get("upstream_metadata_file"),
        repo_root=repo_root,
    )
    label_metadata = _load_json(label_metadata_path) if label_metadata_path is not None and label_metadata_path.exists() else {}

    label_path = _resolve_repo_path(source_lineage.get("feature_builder_source_path"), repo_root=repo_root)
    if label_path is None:
        output_path = _mapping(label_metadata).get("output_path")
        label_path = _resolve_repo_path(output_path, repo_root=repo_root)
    if label_path is None or not label_path.exists():
        raise FileNotFoundError("Could not resolve the breakout label CSV needed for model-testing enrichment.")

    events_path = _derive_events_path(label_path)
    if not events_path.exists():
        raise FileNotFoundError(
            f"Expected breakout event CSV at {events_path}, but it does not exist."
        )

    timezone_contract = _mapping(label_metadata.get("timezone_contract")) or _mapping(report.get("timezone_contract"))
    breakout_params = _mapping(_mapping(label_metadata.get("labeling_params")).get("breakout"))
    return {
        "label_kind": label_kind,
        "label_path": label_path,
        "events_path": events_path,
        "csv_timezone": str(timezone_contract.get("csv_timezone") or timezone_contract.get("canonical_timezone") or "UTC"),
        "canonical_timezone": str(timezone_contract.get("canonical_timezone") or "UTC"),
        "zone_pre_bars": int(breakout_params.get("zone_pre_bars", 0)),
        "zone_post_bars": int(breakout_params.get("zone_post_bars", 0)),
    }


def _load_breakout_label_frame(
    path: Path,
    *,
    source_timezone: str,
    canonical_timezone: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "close"])
    frame = frame.rename(columns={"timestamp": "datetime"})
    frame["datetime"] = normalize_datetime_series(
        frame["datetime"],
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.reset_index(drop=True)


def _load_breakout_event_frame(
    path: Path,
    *,
    source_timezone: str,
    canonical_timezone: str,
) -> pd.DataFrame:
    wanted = {
        "label_family",
        "swing_type",
        "breakout_direction",
        "confirm_time",
        "confirm_index",
        "entry_time",
        "entry_index",
        "entry_price",
        "entry_score",
        "tb_bars_held",
        "tb_outcome",
        "tb_return",
        "label_quality",
        "label_tier",
    }
    frame = pd.read_csv(path, usecols=lambda column: column in wanted)
    for column in ("confirm_time", "entry_time"):
        if column in frame.columns:
            frame[column] = normalize_datetime_series(
                frame[column],
                source_timezone=source_timezone,
                canonical_timezone=canonical_timezone,
            )
    return frame


def _derive_events_path(label_path: Path) -> Path:
    if "_labels_" in label_path.name:
        return label_path.with_name(label_path.name.replace("_labels_", "_swings_", 1))
    if "labels" in label_path.name:
        return label_path.with_name(label_path.name.replace("labels", "swings", 1))
    return label_path.with_name(f"{label_path.stem}_swings{label_path.suffix}")


def _resolve_direction_mask(events: pd.DataFrame, direction: str) -> pd.Series:
    breakout_direction = events.get("breakout_direction")
    swing_type = events.get("swing_type")
    if direction == "long":
        if breakout_direction is not None and breakout_direction.notna().any():
            return breakout_direction.astype(str).str.lower() == "up"
        if swing_type is not None:
            return swing_type.astype(str).str.lower() == "low"
    else:
        if breakout_direction is not None and breakout_direction.notna().any():
            return breakout_direction.astype(str).str.lower() == "down"
        if swing_type is not None:
            return swing_type.astype(str).str.lower() == "high"
    return pd.Series(False, index=events.index, dtype=bool)


def _resolve_mapped_label_rows(
    *,
    confirm_index: int,
    entry_index: int | None,
    label_kind: str,
    zone_pre_bars: int,
    zone_post_bars: int,
    label_count: int,
) -> list[int]:
    if label_kind == "breakout_entry":
        if entry_index is None:
            return []
        return [entry_index]

    start = max(confirm_index - int(zone_pre_bars), 0)
    end = min(confirm_index + int(zone_post_bars), label_count - 1)
    return list(range(start, end + 1))


def _empty_breakout_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", *BREAKOUT_EVENT_METADATA_COLUMNS])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(value: object, *, repo_root: Path) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _coalesce_numeric(*values: object) -> float | None:
    for value in values:
        resolved = _optional_float(value)
        if resolved is not None:
            return resolved
    return None


def _coalesce_timestamp(*values: object) -> pd.Timestamp | pd.NaT:
    for value in values:
        if value is None:
            continue
        if isinstance(value, pd.Timestamp):
            if pd.notna(value):
                return value
            continue
        if pd.notna(value):
            parsed = pd.Timestamp(value)
            if pd.notna(parsed):
                return parsed
    return pd.NaT


def _value_at_position(series: pd.Series, position: int | None) -> object:
    if position is None or position < 0 or position >= len(series):
        return None
    return series.iat[position]
