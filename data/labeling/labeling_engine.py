"""
Unified EURUSD labeling workflow.

This script generates reversal, continuation-pullback, breakout, and optional
FRVP labels in one pass and saves a single bar-level CSV with separate named
columns for each label family.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.fx_calendar import normalize_datetime_series

try:
    from .frvp_labeling_engine import FRVPLabelingParams, build_frvp_labels, frvp_events_to_frame
    from .ote_breakout_labeling_engine import BreakoutParams, build_breakout_labels, plot_breakouts
    from .ote_continuation_pullback_labeling_engine import (
        ContinuationPullbackParams,
        build_continuation_pullback_labels,
        plot_continuation_pullbacks,
    )
    from .reversal_labeling_engine import (
        ReversalParams,
        build_reversal_labels,
        format_bar_timeframe,
        infer_bar_minutes,
        plot_reversal_swings,
        retune_bar_count_params,
    )
except ImportError:
    from frvp_labeling_engine import FRVPLabelingParams, build_frvp_labels, frvp_events_to_frame  # type: ignore
    from ote_breakout_labeling_engine import BreakoutParams, build_breakout_labels, plot_breakouts  # type: ignore
    from ote_continuation_pullback_labeling_engine import (  # type: ignore
        ContinuationPullbackParams,
        build_continuation_pullback_labels,
        plot_continuation_pullbacks,
    )
    from reversal_labeling_engine import (  # type: ignore
        ReversalParams,
        build_reversal_labels,
        format_bar_timeframe,
        infer_bar_minutes,
        plot_reversal_swings,
        retune_bar_count_params,
    )

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "currency_data" / "EURUSD_5min.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_all_labels.csv"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_all_label_events.csv"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "data" / "labeling" / "plots" / "eurusd_5min_label_overview.png"

TIMING_COLUMNS = {
    "bars_since_30m_swing_low",
    "bars_since_30m_swing_high",
    "bars_since_1h_swing_low",
    "bars_since_1h_swing_high",
}


@dataclass
class CombinedLabelingParams:
    reversal: ReversalParams = field(default_factory=ReversalParams)
    continuation: ContinuationPullbackParams = field(default_factory=ContinuationPullbackParams)
    breakout: BreakoutParams = field(default_factory=BreakoutParams)
    frvp: FRVPLabelingParams = field(default_factory=FRVPLabelingParams)


def detect_anomalies(df: pd.DataFrame, atr_period: int = 20, range_atr_mult: float = 10.0) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    invalid_ohlc = (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
    huge_range = (atr > 0) & ((df["high"] - df["low"]) > range_atr_mult * atr)
    zero_volume = df["volume"] <= 0
    return invalid_ohlc | huge_range | zero_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label EURUSD 5-minute reversal, continuation, and breakout structures."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Source OHLCV CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Combined bar-level labeled CSV.")
    parser.add_argument(
        "--events-output",
        dest="events_output",
        type=Path,
        default=DEFAULT_EVENTS_PATH,
        help="Combined label-event metadata CSV.",
    )
    parser.add_argument("--swings-output", dest="events_output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plot", action="store_true", help="Save diagnostic charts for each label family.")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help="Base plot path. Family-specific suffixes will be appended when --plot is enabled.",
    )
    parser.add_argument(
        "--source-timezone",
        type=str,
        default="UTC",
        help="Timezone of naive timestamps in the raw CSV. Use an IANA timezone or fixed offset like GMT-6.",
    )
    parser.add_argument(
        "--canonical-timezone",
        type=str,
        default="UTC",
        help="Canonical timezone to store during labeling and downstream training. Defaults to UTC.",
    )
    parser.add_argument(
        "--bar-timestamp-semantics",
        type=str,
        default="as_provided",
        help="Metadata-only description of whether timestamps represent bar opens, closes, or another convention.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive lower bound for labeled bars in canonical time, for example 2019-01-01.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive upper bound for labeled bars in canonical time. Date-only values include the full day.",
    )
    return parser.parse_args()


def load_fx_csv(
    path: Path,
    *,
    source_timezone: str = "UTC",
    canonical_timezone: str = "UTC",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    raw_row_count = int(len(df))
    columns_lower = {col.strip().lower(): col for col in df.columns}
    timestamp_source_column: Optional[str] = None

    if "date" in columns_lower and "time" in columns_lower:
        df["timestamp"] = pd.to_datetime(
            df[columns_lower["date"]].astype(str) + " " + df[columns_lower["time"]].astype(str),
            format="%Y%m%d %H:%M:%S",
            errors="coerce",
        )
        df = df.drop(columns=[columns_lower["date"], columns_lower["time"]])
    elif "timestamp" in columns_lower:
        timestamp_source_column = columns_lower["timestamp"]
        df["timestamp"] = pd.to_datetime(df[timestamp_source_column], errors="coerce")
    elif "datetime" in columns_lower:
        timestamp_source_column = columns_lower["datetime"]
        df["timestamp"] = pd.to_datetime(df[timestamp_source_column], errors="coerce")
    elif "ts_event" in columns_lower:
        timestamp_source_column = columns_lower["ts_event"]
        df["timestamp"] = pd.to_datetime(df[timestamp_source_column], errors="coerce")
    else:
        raise ValueError(f"Could not find a timestamp column in {path}. Columns: {df.columns.tolist()}")

    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower()
        if normalized in {"open", "o"}:
            rename_map[col] = "open"
        elif normalized in {"high", "h"}:
            rename_map[col] = "high"
        elif normalized in {"low", "l"}:
            rename_map[col] = "low"
        elif normalized in {"close", "c", "adj close"}:
            rename_map[col] = "close"
        elif normalized in {"volume", "vol", "v", "tick_volume"}:
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    df["timestamp"] = normalize_datetime_series(
        df["timestamp"],
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    if timestamp_source_column and timestamp_source_column in df.columns and timestamp_source_column != "timestamp":
        df[timestamp_source_column] = df["timestamp"]
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if "volume" not in df.columns:
        df["volume"] = 0

    ordered_columns = ["open", "high", "low", "close", "volume"]
    ordered_columns.extend(
        [
            column
            for column in df.columns
            if column not in {"open", "high", "low", "close", "volume"}
        ]
    )
    df = df.loc[:, ordered_columns].copy()
    anomaly_mask = detect_anomalies(df)
    anomaly_count = int(anomaly_mask.sum())
    df["is_anomaly"] = anomaly_mask.astype(bool)
    if anomaly_mask.any():
        df = df.loc[~anomaly_mask].copy()
        df["is_anomaly"] = False
    df.attrs["anomaly_count"] = anomaly_count
    df.attrs["source_row_count"] = raw_row_count
    df.attrs["source_timezone"] = source_timezone
    df.attrs["canonical_timezone"] = canonical_timezone

    return df


def prepare_frame_for_csv(
    frame: pd.DataFrame,
    *,
    canonical_timezone: str,
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        series = prepared[column]
        if not pd.api.types.is_datetime64_any_dtype(series):
            continue
        normalized = normalize_datetime_series(
            series,
            source_timezone=canonical_timezone,
            canonical_timezone=canonical_timezone,
        )
        prepared[column] = normalized.dt.tz_localize(None)
    return prepared


def _normalize_boundary(
    raw_value: str,
    *,
    canonical_timezone: str,
) -> pd.Timestamp:
    normalized = normalize_datetime_series(
        pd.Series([raw_value]),
        source_timezone=canonical_timezone,
        canonical_timezone=canonical_timezone,
    )
    boundary = normalized.iloc[0]
    if pd.isna(boundary):
        raise ValueError(f"Could not parse date boundary: {raw_value!r}")
    return pd.Timestamp(boundary)


def _is_date_only(raw_value: str) -> bool:
    value = str(raw_value).strip()
    if not value:
        return False
    return (" " not in value) and ("T" not in value) and (len(value) <= 10)


def filter_date_range(
    df: pd.DataFrame,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    canonical_timezone: str,
) -> pd.DataFrame:
    if not start_date and not end_date:
        return df

    if df.empty:
        return df.copy()

    original_start = df.index.min()
    original_end = df.index.max()

    start_ts = _normalize_boundary(start_date, canonical_timezone=canonical_timezone) if start_date else None
    end_ts = _normalize_boundary(end_date, canonical_timezone=canonical_timezone) if end_date else None

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            f"start-date must be <= end-date after timezone normalization. Got {start_ts.isoformat()} > {end_ts.isoformat()}."
        )

    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= df.index >= start_ts
    if end_ts is not None:
        if end_date and _is_date_only(end_date):
            mask &= df.index < (end_ts + pd.Timedelta(days=1))
        else:
            mask &= df.index <= end_ts

    filtered = df.loc[mask].copy()
    filtered.attrs.update(df.attrs)
    filtered.attrs["rows_before_date_filter"] = int(len(df))
    filtered.attrs["rows_removed_by_date_filter"] = int(len(df) - len(filtered))
    filtered.attrs["requested_start_date"] = start_date
    filtered.attrs["requested_end_date"] = end_date
    filtered.attrs["date_filter_start_canonical"] = start_ts.isoformat() if start_ts is not None else None
    filtered.attrs["date_filter_end_canonical"] = end_ts.isoformat() if end_ts is not None else None
    filtered.attrs["source_time_range_before_date_filter_start"] = (
        original_start.isoformat() if original_start is not None else None
    )
    filtered.attrs["source_time_range_before_date_filter_end"] = (
        original_end.isoformat() if original_end is not None else None
    )
    return filtered


def write_metadata(path: Path, payload: dict) -> Path:
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(_make_json_safe(payload), indent=2), encoding="utf-8")
    return metadata_path


def _serialize_labeling_params(params: Any) -> Any:
    if is_dataclass(params):
        return asdict(params)
    if isinstance(params, dict):
        return params
    return str(params)


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
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_make_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_make_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [_make_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if is_dataclass(value):
        return _make_json_safe(asdict(value))
    if pd.isna(value):
        return None
    return str(value)


def build_output_metadata(
    *,
    output_kind: str,
    source_path: Path,
    output_path: Path,
    df_5m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1hr: pd.DataFrame,
    params: Any,
    source_timezone: str,
    canonical_timezone: str,
    bar_timestamp_semantics: str,
    output_rows: int,
) -> dict:
    base_bar_minutes = infer_bar_minutes(df_5m.index)
    base_timeframe = format_bar_timeframe(base_bar_minutes)
    return {
        "output_kind": output_kind,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_bar_minutes": base_bar_minutes,
        "base_timeframe": base_timeframe,
        "row_counts": {
            "source_rows": int(df_5m.attrs.get("source_row_count", len(df_5m))),
            "rows_before_date_filter": int(df_5m.attrs.get("rows_before_date_filter", len(df_5m))),
            "rows_removed_by_date_filter": int(df_5m.attrs.get("rows_removed_by_date_filter", 0)),
            "rows_base_after_cleanup": int(len(df_5m)),
            "rows_5m_after_cleanup": int(len(df_5m)),
            "rows_30m": int(len(df_30m)),
            "rows_1h": int(len(df_1hr)),
            "output_rows": int(output_rows),
            "anomaly_rows_removed": int(df_5m.attrs.get("anomaly_count", 0)),
        },
        "time_range": {
            "source_start_before_date_filter": df_5m.attrs.get("source_time_range_before_date_filter_start"),
            "source_end_before_date_filter": df_5m.attrs.get("source_time_range_before_date_filter_end"),
            "requested_start_date": df_5m.attrs.get("requested_start_date"),
            "requested_end_date": df_5m.attrs.get("requested_end_date"),
            "date_filter_start_canonical": df_5m.attrs.get("date_filter_start_canonical"),
            "date_filter_end_canonical": df_5m.attrs.get("date_filter_end_canonical"),
            "start": df_5m.index.min().isoformat() if not df_5m.empty else None,
            "end": df_5m.index.max().isoformat() if not df_5m.empty else None,
        },
        "timezone_contract": {
            "source_timezone": source_timezone,
            "canonical_timezone": canonical_timezone,
            "csv_timezone": canonical_timezone,
            "csv_timestamps_are_timezone_aware": False,
            "timestamp_column": "timestamp" if output_kind == "bar_labels" else None,
        },
        "bar_timestamp_semantics": bar_timestamp_semantics,
        "labeling_params": _serialize_labeling_params(params),
    }


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def build_default_params() -> CombinedLabelingParams:
    return CombinedLabelingParams(
        reversal=ReversalParams(
            atr_period=14,
            atr_smoothing="ema",
            structural_atr_tf="1hr",
            cusum_atr_mult=0.5,
            cusum_use_dynamic=True,
            confirm_atr_mult=0.8,
            confirm_use_close=True,
            min_swing_atr=1.35,
            min_swing_distance_atr=0.75,
            min_bars_between_swings=18,
            tb_enable=True,
            tb_profit_atr=1.0,
            tb_stop_atr=2.0,
            tb_max_bars=120,
            zone_pre_bars=2,
            zone_post_bars=0,
            entry_lookback_bars=3,
            entry_max_delay_after_swing=4,
            exclusion_pre_bars=10,
            htf_confirm_atr_mult=0.6,
            htf_min_swing_atr=0.8,
            htf_min_bars_between=4,
            htf_confluence_window_minutes=120,
            compute_uniqueness=True,
            warmup_bars=50,
        ),
        continuation=ContinuationPullbackParams(),
        breakout=BreakoutParams(),
        frvp=FRVPLabelingParams(),
    )


def retune_combined_params(
    params: CombinedLabelingParams,
    base_bar_minutes: float,
) -> CombinedLabelingParams:
    return CombinedLabelingParams(
        reversal=retune_bar_count_params(params.reversal, base_bar_minutes),
        continuation=retune_bar_count_params(
            params.continuation,
            base_bar_minutes,
            extra_count_fields=("max_pullback_bars",),
        ),
        breakout=retune_bar_count_params(
            params.breakout,
            base_bar_minutes,
            extra_count_fields=("breakout_lookback_bars", "compression_lookback_bars"),
        ),
        frvp=retune_bar_count_params(
            params.frvp,
            base_bar_minutes,
            extra_count_fields=(
                "mean_reversion_max_bars",
                "continuation_max_bars",
                "failed_auction_max_bars",
                "first_rth_exclusion_bars",
            ),
        ),
    )


def summarize_labels(df_labeled: pd.DataFrame) -> str:
    usable = ~df_labeled["warmup_mask"]
    families = [
        (
            "Reversal",
            "label_long_reversal",
            "label_short_reversal",
            "label_long_reversal_entry",
            "label_short_reversal_entry",
            "label_quality_long_reversal",
            "label_quality_short_reversal",
        ),
        (
            "Continuation",
            "label_long_continuation_pullback",
            "label_short_continuation_pullback",
            "label_long_continuation_entry",
            "label_short_continuation_entry",
            "label_quality_long_continuation",
            "label_quality_short_continuation",
        ),
        (
            "Breakout",
            "label_long_breakout",
            "label_short_breakout",
            "label_long_breakout_entry",
            "label_short_breakout_entry",
            "label_quality_long_breakout",
            "label_quality_short_breakout",
        ),
        (
            "FRVP Reversal",
            "label_long_frvp_reversal",
            "label_short_frvp_reversal",
            None,
            None,
            "label_quality_long_frvp_reversal",
            "label_quality_short_frvp_reversal",
        ),
        (
            "FRVP Continuation",
            "label_long_frvp_continuation",
            "label_short_frvp_continuation",
            None,
            None,
            "label_quality_long_frvp_continuation",
            "label_quality_short_frvp_continuation",
        ),
    ]

    lines = ["", f"Bars after warmup: {int(usable.sum()):,}"]
    for family_name, long_col, short_col, long_entry, short_entry, long_quality, short_quality in families:
        if long_col not in df_labeled.columns or short_col not in df_labeled.columns:
            continue
        long_quality_values = df_labeled.loc[usable & (df_labeled[long_quality] > 0), long_quality]
        short_quality_values = df_labeled.loc[usable & (df_labeled[short_quality] > 0), short_quality]
        lines.append(f"{family_name} long positives:   {int(df_labeled.loc[usable, long_col].sum()):,}")
        lines.append(f"{family_name} short positives:  {int(df_labeled.loc[usable, short_col].sum()):,}")
        if long_entry and short_entry and long_entry in df_labeled.columns and short_entry in df_labeled.columns:
            lines.append(f"{family_name} long entries:     {int(df_labeled.loc[usable, long_entry].sum()):,}")
            lines.append(f"{family_name} short entries:    {int(df_labeled.loc[usable, short_entry].sum()):,}")
        lines.append(
            f"{family_name} quality L/S:    "
            f"{float(long_quality_values.mean()) if not long_quality_values.empty else 0.0:.3f} / "
            f"{float(short_quality_values.mean()) if not short_quality_values.empty else 0.0:.3f}"
        )
    return "\n".join(lines)


def reversal_events_to_frame(swings: list) -> pd.DataFrame:
    rows = []
    for swing in swings:
        rows.append(
            {
                "label_family": "reversal",
                "swing_type": swing.swing_type,
                "source_tf": swing.source_tf,
                "swing_time": swing.swing_time,
                "swing_index": swing.swing_index,
                "swing_price": swing.swing_price,
                "confirm_time": swing.confirm_time,
                "confirm_index": swing.confirm_index,
                "confirm_lag_bars": swing.confirm_lag,
                "atr_at_swing": swing.atr_at_swing,
                "swing_size_atr": swing.swing_size_atr,
                "tb_outcome": swing.tb_outcome,
                "tb_return": swing.tb_return,
                "tb_bars_held": swing.tb_bars_held,
                "entry_time": swing.entry_time,
                "entry_index": swing.entry_index,
                "entry_price": swing.entry_price,
                "entry_score": swing.entry_score,
                "entry_rr": swing.entry_rr,
                "entry_followthrough_atr": swing.entry_followthrough_atr,
                "trend_scan_t": swing.trend_scan_t,
                "trend_scan_window": swing.trend_scan_window,
                "trend_scan_pass": swing.trend_scan_pass,
                "htf_match_30m": swing.htf_match_30m,
                "htf_match_1h": swing.htf_match_1h,
                "bars_since_prev_30m_same": swing.bars_since_prev_30m_same,
                "bars_since_prev_1h_same": swing.bars_since_prev_1h_same,
                "label_quality": swing.label_quality,
                "label_tier": swing.label_tier,
            }
        )
    return pd.DataFrame(rows)


def continuation_events_to_frame(events: list) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "label_family": "continuation_pullback",
                "swing_type": event.swing_type,
                "source_tf": event.source_tf,
                "trend_direction": getattr(event, "trend_direction", None),
                "swing_time": event.swing_time,
                "swing_index": event.swing_index,
                "swing_price": event.swing_price,
                "confirm_time": event.confirm_time,
                "confirm_index": event.confirm_index,
                "confirm_lag_bars": event.confirm_lag,
                "atr_at_swing": event.atr_at_swing,
                "swing_size_atr": event.swing_size_atr,
                "pullback_depth_atr": getattr(event, "pullback_depth_atr", None),
                "pullback_bars": getattr(event, "pullback_bars", None),
                "trend_spread_30m_atr": getattr(event, "trend_spread_30m_atr", None),
                "trend_spread_1h_atr": getattr(event, "trend_spread_1h_atr", None),
                "price_vs_trend_floor_atr": getattr(event, "price_vs_trend_floor_atr", None),
                "same_direction_break_atr": getattr(event, "same_direction_break_atr", None),
                "resume_move_at_confirm_atr": getattr(event, "resume_move_at_confirm_atr", None),
                "tb_outcome": event.tb_outcome,
                "tb_return": event.tb_return,
                "tb_bars_held": event.tb_bars_held,
                "entry_time": event.entry_time,
                "entry_index": event.entry_index,
                "entry_price": event.entry_price,
                "entry_score": event.entry_score,
                "entry_rr": event.entry_rr,
                "entry_followthrough_atr": event.entry_followthrough_atr,
                "trend_scan_t": event.trend_scan_t,
                "trend_scan_window": event.trend_scan_window,
                "trend_scan_pass": event.trend_scan_pass,
                "htf_match_30m": event.htf_match_30m,
                "htf_match_1h": event.htf_match_1h,
                "bars_since_prev_30m_same": event.bars_since_prev_30m_same,
                "bars_since_prev_1h_same": event.bars_since_prev_1h_same,
                "label_quality": event.label_quality,
                "label_tier": event.label_tier,
            }
        )
    return pd.DataFrame(rows)


def breakout_events_to_frame(events: list) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "label_family": "breakout",
                "swing_type": event.swing_type,
                "source_tf": event.source_tf,
                "breakout_direction": getattr(event, "breakout_direction", None),
                "swing_time": event.swing_time,
                "swing_index": event.swing_index,
                "swing_price": event.swing_price,
                "confirm_time": event.confirm_time,
                "confirm_index": event.confirm_index,
                "confirm_lag_bars": event.confirm_lag,
                "atr_at_swing": event.atr_at_swing,
                "swing_size_atr": event.swing_size_atr,
                "breakout_level": getattr(event, "breakout_level", None),
                "prior_range_high": getattr(event, "prior_range_high", None),
                "prior_range_low": getattr(event, "prior_range_low", None),
                "prior_range_width_atr": getattr(event, "prior_range_width_atr", None),
                "compression_range_atr": getattr(event, "compression_range_atr", None),
                "breakout_extension_atr": getattr(event, "breakout_extension_atr", None),
                "breakout_close_strength": getattr(event, "breakout_close_strength", None),
                "breakout_body_fraction": getattr(event, "breakout_body_fraction", None),
                "breakout_trend_spread_30m_atr": getattr(event, "breakout_trend_spread_30m_atr", None),
                "breakout_trend_spread_1h_atr": getattr(event, "breakout_trend_spread_1h_atr", None),
                "breakout_is_compressed": getattr(event, "breakout_is_compressed", None),
                "tb_outcome": event.tb_outcome,
                "tb_return": event.tb_return,
                "tb_bars_held": event.tb_bars_held,
                "entry_time": event.entry_time,
                "entry_index": event.entry_index,
                "entry_price": event.entry_price,
                "entry_score": event.entry_score,
                "entry_rr": event.entry_rr,
                "entry_followthrough_atr": event.entry_followthrough_atr,
                "trend_scan_t": event.trend_scan_t,
                "trend_scan_window": event.trend_scan_window,
                "trend_scan_pass": event.trend_scan_pass,
                "htf_match_30m": event.htf_match_30m,
                "htf_match_1h": event.htf_match_1h,
                "bars_since_prev_30m_same": event.bars_since_prev_30m_same,
                "bars_since_prev_1h_same": event.bars_since_prev_1h_same,
                "label_quality": event.label_quality,
                "label_tier": event.label_tier,
            }
        )
    return pd.DataFrame(rows)


def combine_event_frames(
    reversal_events: list,
    continuation_events: list,
    breakout_events: list,
    frvp_events: Optional[list] = None,
) -> pd.DataFrame:
    frames = [
        reversal_events_to_frame(reversal_events),
        continuation_events_to_frame(continuation_events),
        breakout_events_to_frame(breakout_events),
    ]
    if frvp_events:
        frames.append(frvp_events_to_frame(frvp_events))
    populated = [frame for frame in frames if not frame.empty]
    if not populated:
        return pd.DataFrame()
    combined = pd.concat(populated, ignore_index=True, sort=False)
    if "confirm_time" in combined.columns:
        combined = combined.sort_values(["confirm_time", "label_family", "confirm_index"], kind="stable")
    return combined.reset_index(drop=True)


def merge_reversal_output(base_df: pd.DataFrame, reversal_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.copy()
    for column in reversal_df.columns:
        if column not in merged.columns:
            merged[column] = reversal_df[column]
    if "atr" in reversal_df.columns:
        merged["reversal_atr"] = reversal_df["atr"]
    if "structural_atr" in reversal_df.columns:
        merged["reversal_structural_atr"] = reversal_df["structural_atr"]
    return merged


def merge_prefixed_output(base_df: pd.DataFrame, family_df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    merged = base_df.copy()
    if "atr" in family_df.columns:
        merged[f"{family_name}_atr"] = family_df["atr"]
    if "structural_atr" in family_df.columns:
        merged[f"{family_name}_structural_atr"] = family_df["structural_atr"]
    for column in family_df.columns:
        if column in merged.columns or column in {"atr", "structural_atr", "warmup_mask"} or column in TIMING_COLUMNS:
            continue
        merged[column] = family_df[column]
    return merged


def build_combined_label_frame(
    df_5m: pd.DataFrame,
    reversal_df: pd.DataFrame,
    continuation_df: pd.DataFrame,
    breakout_df: pd.DataFrame,
    frvp_df: pd.DataFrame,
    params: CombinedLabelingParams,
) -> pd.DataFrame:
    merged = merge_reversal_output(df_5m, reversal_df)
    merged = merge_prefixed_output(merged, continuation_df, "continuation")
    merged = merge_prefixed_output(merged, breakout_df, "breakout")
    merged = merge_prefixed_output(merged, frvp_df, "frvp")

    warmup_bars = max(
        params.reversal.warmup_bars,
        params.continuation.warmup_bars,
        params.breakout.warmup_bars,
        params.frvp.warmup_bars,
    )
    warmup_mask = pd.Series(False, index=merged.index)
    warmup_mask.iloc[:warmup_bars] = True
    merged["warmup_mask"] = warmup_mask.values
    return merged


def _plot_path_with_suffix(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def maybe_save_plots(
    df_labeled: pd.DataFrame,
    reversal_events: list,
    continuation_events: list,
    breakout_events: list,
    plot_output: Path,
) -> None:
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    if reversal_events:
        plot_reversal_swings(
            df_labeled,
            reversal_events,
            df_labeled["label_long_reversal"].values,
            df_labeled["label_short_reversal"].values,
            save_path=str(_plot_path_with_suffix(plot_output, "reversal")),
        )
    if continuation_events:
        plot_continuation_pullbacks(
            df_labeled,
            continuation_events,
            df_labeled["label_long_continuation_pullback"].values,
            df_labeled["label_short_continuation_pullback"].values,
            save_path=str(_plot_path_with_suffix(plot_output, "continuation")),
        )
    if breakout_events:
        plot_breakouts(
            df_labeled,
            breakout_events,
            df_labeled["label_long_breakout"].values,
            df_labeled["label_short_breakout"].values,
            save_path=str(_plot_path_with_suffix(plot_output, "breakout")),
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    print(f"Loading source data: {args.input}")
    df_5m = load_fx_csv(
        args.input,
        source_timezone=args.source_timezone,
        canonical_timezone=args.canonical_timezone,
    )
    df_5m = filter_date_range(
        df_5m,
        start_date=args.start_date,
        end_date=args.end_date,
        canonical_timezone=args.canonical_timezone,
    )
    if df_5m.empty:
        raise ValueError(
            "No rows remain after applying the requested date filter. "
            f"start-date={args.start_date!r}, end-date={args.end_date!r}"
        )
    base_bar_minutes = infer_bar_minutes(df_5m.index)
    base_timeframe = format_bar_timeframe(base_bar_minutes)
    df_30m = resample_ohlcv(df_5m, "30min")
    df_1hr = resample_ohlcv(df_5m, "1h")

    print(f"Base rows ({base_timeframe}): {len(df_5m):,}")
    print(f"30m rows: {len(df_30m):,}")
    print(f"1h rows:  {len(df_1hr):,}")
    print(f"Anomaly bars removed: {int(df_5m.attrs.get('anomaly_count', 0)):,}")
    if args.start_date or args.end_date:
        print(
            "Date filter: "
            f"{args.start_date or 'beginning'} -> {args.end_date or 'latest'} "
            f"(removed {int(df_5m.attrs.get('rows_removed_by_date_filter', 0)):,} rows)"
        )
    print(f"Timezone normalization: {args.source_timezone} -> {args.canonical_timezone}")
    print(f"Date range: {df_5m.index[0]} -> {df_5m.index[-1]}")
    if abs(base_bar_minutes - 5.0) >= 1e-9:
        print(f"Runtime bar-count tuning enabled: 5m defaults -> {base_timeframe}")

    params = retune_combined_params(build_default_params(), base_bar_minutes)

    reversal_df, reversal_diag, reversal_events = build_reversal_labels(
        df_5m=df_5m,
        df_30m=df_30m,
        df_1hr=df_1hr,
        params=params.reversal,
        verbose=True,
    )
    continuation_df, continuation_diag, continuation_events = build_continuation_pullback_labels(
        df_5m=df_5m,
        df_30m=df_30m,
        df_1hr=df_1hr,
        params=params.continuation,
        verbose=True,
    )
    breakout_df, breakout_diag, breakout_events = build_breakout_labels(
        df_5m=df_5m,
        df_30m=df_30m,
        df_1hr=df_1hr,
        params=params.breakout,
        verbose=True,
    )
    frvp_df, frvp_diag, frvp_events = build_frvp_labels(
        df_5m=df_5m,
        df_30m=df_30m,
        df_1hr=df_1hr,
        params=params.frvp,
        verbose=True,
    )

    df_labeled = build_combined_label_frame(
        df_5m,
        reversal_df,
        continuation_df,
        breakout_df,
        frvp_df,
        params,
    )
    combined_events = combine_event_frames(reversal_events, continuation_events, breakout_events, frvp_events)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled_to_save = df_labeled.reset_index().rename(columns={"index": "timestamp"})
    labeled_to_save = prepare_frame_for_csv(
        labeled_to_save,
        canonical_timezone=args.canonical_timezone,
    )
    labeled_to_save.to_csv(args.output, index=False)

    args.events_output.parent.mkdir(parents=True, exist_ok=True)
    events_to_save = prepare_frame_for_csv(
        combined_events,
        canonical_timezone=args.canonical_timezone,
    )
    events_to_save.to_csv(args.events_output, index=False)

    labels_metadata = build_output_metadata(
        output_kind="bar_labels",
        source_path=args.input,
        output_path=args.output,
        df_5m=df_5m,
        df_30m=df_30m,
        df_1hr=df_1hr,
        params=params,
        source_timezone=args.source_timezone,
        canonical_timezone=args.canonical_timezone,
        bar_timestamp_semantics=args.bar_timestamp_semantics,
        output_rows=len(labeled_to_save),
    )
    labels_metadata["family_diagnostics"] = {
        "reversal": reversal_diag,
        "continuation": continuation_diag,
        "breakout": breakout_diag,
        "frvp": frvp_diag,
    }
    events_metadata = build_output_metadata(
        output_kind="combined_label_events",
        source_path=args.input,
        output_path=args.events_output,
        df_5m=df_5m,
        df_30m=df_30m,
        df_1hr=df_1hr,
        params=params,
        source_timezone=args.source_timezone,
        canonical_timezone=args.canonical_timezone,
        bar_timestamp_semantics=args.bar_timestamp_semantics,
        output_rows=len(events_to_save),
    )
    events_metadata["event_counts"] = {
        "reversal": len(reversal_events),
        "continuation": len(continuation_events),
        "breakout": len(breakout_events),
        "frvp": len(frvp_events),
    }
    labels_metadata_path = write_metadata(args.output, labels_metadata)
    events_metadata_path = write_metadata(args.events_output, events_metadata)

    print(f"\nSaved combined bar labels to: {args.output}")
    print(f"Saved combined bar-label metadata to: {labels_metadata_path}")
    print(f"Saved combined label events to: {args.events_output}")
    print(f"Saved combined event metadata sidecar to: {events_metadata_path}")
    print(summarize_labels(df_labeled))

    if args.plot:
        maybe_save_plots(
            df_labeled,
            reversal_events,
            continuation_events,
            breakout_events,
            args.plot_output,
        )
        print(f"Saved diagnostic plots using base path: {args.plot_output}")


if __name__ == "__main__":
    main()
