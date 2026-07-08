from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from features.fx_calendar import (
    build_intraday_calendar_frame,
    classify_session_regime,
    normalize_datetime_series,
)


FRVP_OPEN_TYPE_LABELS = {
    -1: "below_value",
    0: "inside_value",
    1: "above_value",
}
FRVP_OPEN_TYPE_STATUS_CLASSIFIED = "classified"
FRVP_OPEN_TYPE_STATUS_NOT_YET_KNOWN = "not_yet_known"
FRVP_OPEN_TYPE_STATUS_MISSING_AFTER_OPEN = "missing_after_open"
FRVP_OPEN_TYPE_STATUS_UNKNOWN = "unknown"

FRVP_DAY_TYPE_LABELS = {
    1: "normal",
    2: "normal_variation",
    3: "trend",
    4: "double_distribution_trend",
    5: "neutral",
}


@dataclass(frozen=True)
class RegimeLabelConfig:
    time_column: str = "datetime"
    adx_column: str = "adx_14"
    ema_alignment_column: str = "ema_alignment"
    atr_column: str = "atr_14"
    range_shock_column: str = "range_shock_20"
    ema50_column: str = "ema_50"
    close_column: str = "close"
    close_vs_ema50_column: str = "close_vs_ema_50_atr"
    hour_column: str = "hour"
    hour_sin_column: str = "hour_sin"
    hour_cos_column: str = "hour_cos"
    asian_session_column: str = "in_asian_session"
    london_session_column: str = "in_london_session"
    new_york_session_column: str = "in_newyork_session"
    source_timezone: str = "UTC"
    canonical_timezone: str = "UTC"
    feature_clock_timezone: str = "America/New_York"
    london_timezone: str = "Europe/London"
    new_york_timezone: str = "America/New_York"
    asia_session_reference_timezone: str = "America/New_York"
    asia_session_start_hour: int = 19
    asia_session_end_hour: int = 2
    london_session_start_hour: int = 7
    london_session_end_hour: int = 16
    new_york_session_start_hour: int = 8
    new_york_session_end_hour: int = 17
    atr_percentile_window: int = 504
    adx_ranging_threshold: float = 20.0
    adx_strong_threshold: float = 30.0
    ema_alignment_ranging_threshold: float = 0.2
    ema_alignment_strong_threshold: float = 0.7
    stress_elevated_threshold: float = 2.0
    stress_high_threshold: float = 3.0


def label_regimes(
    df: pd.DataFrame,
    *,
    config: RegimeLabelConfig | None = None,
) -> pd.DataFrame:
    """
    Compute deterministic OTE regime labels from existing feature columns.

    Returns a copy of ``df`` with:
    ``trend_regime``, ``vol_regime``, ``session_regime``,
    ``stress_regime``, ``composite_regime``, ``year``,
    ``session_hour_utc``, and ``atr_percentile_rank``.
    """

    config = config or RegimeLabelConfig()

    ema_alignment = _require_numeric_column(df, config.ema_alignment_column)
    atr = _require_numeric_column(df, config.atr_column)
    range_shock = _require_numeric_column(df, config.range_shock_column)
    close_bias = _resolve_close_bias(df, config)
    datetime_utc = _resolve_datetime_utc(df, config)
    session_regime, session_hour = _resolve_session_information(df, config, datetime_utc)

    atr_percentile_rank = (
        atr.rolling(window=config.atr_percentile_window, min_periods=1).rank(pct=True) * 100.0
    )

    trend_regime = _resolve_trend_regime(
        df,
        config=config,
        ema_alignment=ema_alignment,
        close_bias=close_bias,
    )

    vol_regime = pd.Series("medium", index=df.index, dtype="object")
    vol_regime.loc[atr_percentile_rank < 25.0] = "low"
    vol_regime.loc[atr_percentile_rank > 75.0] = "high"
    vol_regime = vol_regime.fillna("medium")

    stress_regime = pd.Series("normal", index=df.index, dtype="object")
    stress_regime.loc[
        (range_shock >= config.stress_elevated_threshold)
        & (range_shock <= config.stress_high_threshold)
    ] = "elevated"
    stress_regime.loc[range_shock > config.stress_high_threshold] = "high"
    stress_regime = stress_regime.fillna("normal")

    year = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if datetime_utc is not None:
        year = datetime_utc.dt.year.astype("Int64")
    elif "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    labeled = df.copy()
    labeled["session_hour_utc"] = (
        session_hour.astype("Int64")
        if session_hour is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    labeled["atr_percentile_rank"] = atr_percentile_rank.astype(np.float32)
    labeled["year"] = year
    labeled["trend_regime"] = trend_regime
    labeled["vol_regime"] = vol_regime
    labeled["session_regime"] = session_regime
    labeled["stress_regime"] = stress_regime
    labeled["composite_regime"] = trend_regime + "_" + vol_regime
    _attach_frvp_context_labels(
        labeled,
        datetime_utc=datetime_utc,
        config=config,
    )
    return labeled


def _require_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Required regime feature column is missing: {column}")
    return pd.to_numeric(df[column], errors="coerce")


def _resolve_trend_regime(
    df: pd.DataFrame,
    *,
    config: RegimeLabelConfig,
    ema_alignment: pd.Series,
    close_bias: pd.Series,
) -> pd.Series:
    trend_regime = pd.Series("ranging", index=df.index, dtype="object")

    if config.adx_column in df.columns:
        adx = _require_numeric_column(df, config.adx_column)
        weak_trend_mask = adx.between(
            config.adx_ranging_threshold,
            config.adx_strong_threshold,
            inclusive="both",
        )
        strong_up_mask = (
            (adx > config.adx_strong_threshold)
            & (ema_alignment > config.ema_alignment_strong_threshold)
        )
        strong_down_mask = (
            (adx > config.adx_strong_threshold)
            & (ema_alignment < -config.ema_alignment_strong_threshold)
        )
        weak_up_mask = weak_trend_mask & (close_bias > 0)
        weak_down_mask = weak_trend_mask & (close_bias < 0)

        trend_regime.loc[weak_up_mask] = "weak_up"
        trend_regime.loc[weak_down_mask] = "weak_down"
        trend_regime.loc[strong_up_mask] = "strong_up"
        trend_regime.loc[strong_down_mask] = "strong_down"
        trend_regime.loc[adx < config.adx_ranging_threshold] = "ranging"
        return trend_regime.fillna("ranging")

    alignment_strength = ema_alignment.abs()
    strong_up_mask = ema_alignment > config.ema_alignment_strong_threshold
    strong_down_mask = ema_alignment < -config.ema_alignment_strong_threshold
    weak_alignment_mask = alignment_strength >= config.ema_alignment_ranging_threshold
    weak_up_mask = weak_alignment_mask & ~strong_up_mask & (close_bias > 0)
    weak_down_mask = weak_alignment_mask & ~strong_down_mask & (close_bias < 0)

    trend_regime.loc[weak_up_mask] = "weak_up"
    trend_regime.loc[weak_down_mask] = "weak_down"
    trend_regime.loc[strong_up_mask] = "strong_up"
    trend_regime.loc[strong_down_mask] = "strong_down"
    trend_regime.loc[alignment_strength < config.ema_alignment_ranging_threshold] = "ranging"
    return trend_regime.fillna("ranging")


def _resolve_close_bias(df: pd.DataFrame, config: RegimeLabelConfig) -> pd.Series:
    if {config.close_column, config.ema50_column}.issubset(df.columns):
        close = pd.to_numeric(df[config.close_column], errors="coerce")
        ema_50 = pd.to_numeric(df[config.ema50_column], errors="coerce")
        return close - ema_50

    if config.close_vs_ema50_column in df.columns:
        return pd.to_numeric(df[config.close_vs_ema50_column], errors="coerce")

    if config.ema_alignment_column in df.columns:
        return pd.to_numeric(df[config.ema_alignment_column], errors="coerce")

    raise ValueError(
        "Could not derive close-vs-EMA50 bias. Provide either "
        f"{config.close_column!r} + {config.ema50_column!r}, "
        f"{config.close_vs_ema50_column!r}, or {config.ema_alignment_column!r}."
    )


def _resolve_datetime_utc(
    df: pd.DataFrame,
    config: RegimeLabelConfig,
) -> Optional[pd.Series]:
    if config.time_column in df.columns:
        return normalize_datetime_series(
            df[config.time_column],
            source_timezone=config.source_timezone,
            canonical_timezone=config.canonical_timezone,
        )

    if isinstance(df.index, pd.DatetimeIndex):
        return normalize_datetime_series(
            pd.Series(df.index, index=df.index),
            source_timezone=config.source_timezone,
            canonical_timezone=config.canonical_timezone,
        )

    return None


def _resolve_session_information(
    df: pd.DataFrame,
    config: RegimeLabelConfig,
    datetime_utc: Optional[pd.Series],
) -> tuple[pd.Series, Optional[pd.Series]]:
    if datetime_utc is not None:
        calendar = build_intraday_calendar_frame(
            datetime_utc,
            source_timezone=config.canonical_timezone,
            canonical_timezone=config.canonical_timezone,
            feature_clock_timezone=config.feature_clock_timezone,
            london_timezone=config.london_timezone,
            new_york_timezone=config.new_york_timezone,
            asia_session_reference_timezone=config.asia_session_reference_timezone,
            asia_session_start_hour=config.asia_session_start_hour,
            asia_session_end_hour=config.asia_session_end_hour,
            london_session_start_hour=config.london_session_start_hour,
            london_session_end_hour=config.london_session_end_hour,
            new_york_session_start_hour=config.new_york_session_start_hour,
            new_york_session_end_hour=config.new_york_session_end_hour,
        )
        return calendar["session_regime"], calendar["datetime_utc"].dt.hour.astype("Int64")

    if {
        config.asian_session_column,
        config.london_session_column,
        config.new_york_session_column,
    }.issubset(df.columns):
        session_regime = classify_session_regime(
            in_asian_session=pd.to_numeric(df[config.asian_session_column], errors="coerce").fillna(0),
            in_london_session=pd.to_numeric(df[config.london_session_column], errors="coerce").fillna(0),
            in_newyork_session=pd.to_numeric(df[config.new_york_session_column], errors="coerce").fillna(0),
        )
        return session_regime, _resolve_session_hour(df, config, datetime_utc)

    session_hour = _resolve_session_hour(df, config, datetime_utc)
    session_regime = pd.Series("off_hours", index=df.index, dtype="object")
    if session_hour is not None:
        session_regime.loc[(session_hour >= 19) | (session_hour < 2)] = "asia"
        session_regime.loc[(session_hour >= 2) & (session_hour < 8)] = "london"
        session_regime.loc[(session_hour >= 8) & (session_hour < 11)] = "overlap"
        session_regime.loc[(session_hour >= 11) & (session_hour < 17)] = "new_york"
    return session_regime, session_hour


def _resolve_session_hour(
    df: pd.DataFrame,
    config: RegimeLabelConfig,
    datetime_utc: Optional[pd.Series],
) -> Optional[pd.Series]:
    if datetime_utc is not None:
        return datetime_utc.dt.hour.astype("Int64")

    if config.hour_column in df.columns:
        hour = pd.to_numeric(df[config.hour_column], errors="coerce").round()
        return (hour % 24).astype("Int64")

    if {config.hour_sin_column, config.hour_cos_column}.issubset(df.columns):
        hour_sin = pd.to_numeric(df[config.hour_sin_column], errors="coerce")
        hour_cos = pd.to_numeric(df[config.hour_cos_column], errors="coerce")
        angle = np.mod(np.arctan2(hour_sin, hour_cos), 2.0 * np.pi)
        hour = np.rint((angle * 24.0) / (2.0 * np.pi)) % 24.0
        return pd.Series(hour, index=df.index).astype("Int64")

    return None


def _attach_frvp_context_labels(
    labeled: pd.DataFrame,
    *,
    datetime_utc: Optional[pd.Series],
    config: RegimeLabelConfig,
) -> None:
    if "frvp_open_type" in labeled.columns:
        open_type = pd.to_numeric(labeled["frvp_open_type"], errors="coerce")
        open_type_label = (
            open_type.round()
            .astype("Int64")
            .map(FRVP_OPEN_TYPE_LABELS)
            .astype("object")
        )
        open_type_status = _resolve_frvp_open_type_status(
            open_type_label,
            datetime_utc=datetime_utc,
            config=config,
        )
        labeled["frvp_open_type_label"] = open_type_label
        labeled["frvp_open_type_status"] = open_type_status.astype("object")

    if "frvp_day_type" in labeled.columns:
        day_type = pd.to_numeric(labeled["frvp_day_type"], errors="coerce")
        labeled["frvp_day_type_label"] = (
            day_type.round()
            .astype("Int64")
            .map(FRVP_DAY_TYPE_LABELS)
            .fillna("unknown")
            .astype("object")
        )


def _resolve_frvp_open_type_status(
    open_type_label: pd.Series,
    *,
    datetime_utc: Optional[pd.Series],
    config: RegimeLabelConfig,
) -> pd.Series:
    status = pd.Series(
        FRVP_OPEN_TYPE_STATUS_CLASSIFIED,
        index=open_type_label.index,
        dtype="object",
    )
    missing_mask = open_type_label.isna()
    if not bool(missing_mask.any()):
        return status

    if datetime_utc is None:
        status.loc[missing_mask] = FRVP_OPEN_TYPE_STATUS_UNKNOWN
        return status

    local_datetimes = datetime_utc.dt.tz_convert(config.feature_clock_timezone)
    local_minutes = (local_datetimes.dt.hour * 60) + local_datetimes.dt.minute
    open_known_mask = local_minutes.ge((9 * 60) + 30) & local_minutes.lt(16 * 60)
    status.loc[missing_mask & ~open_known_mask] = FRVP_OPEN_TYPE_STATUS_NOT_YET_KNOWN
    status.loc[missing_mask & open_known_mask] = FRVP_OPEN_TYPE_STATUS_MISSING_AFTER_OPEN
    return status
