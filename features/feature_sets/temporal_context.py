from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..fx_calendar import build_intraday_calendar_frame
from ..registry import register_feature_set
from ..transforms import bars_since_event, calculate_atr, safe_divide


def _event_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna(0).astype(bool)
    return pd.Series(False, index=df.index)


def _last_seen_index(event: pd.Series) -> np.ndarray:
    result = np.full(len(event), -1, dtype=int)
    last_seen = -1
    for idx, flag in enumerate(event.to_numpy(dtype=bool)):
        if flag:
            last_seen = idx
        result[idx] = last_seen
    return result


@register_feature_set(
    name="temporal_context",
    category="context",
    description="Event-age, sequence-order, clustering, and session-phase timing features",
    required_columns=("datetime", "open", "high", "low", "close"),
)
def build_temporal_context(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    candle_range = df["candle_range"] if "candle_range" in df.columns else (df["high"] - df["low"])
    candle_body = df["candle_body"] if "candle_body" in df.columns else (df["close"] - df["open"])
    body_abs = candle_body.abs()
    upper_wick = (
        df["upper_wick"]
        if "upper_wick" in df.columns
        else df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    )
    lower_wick = (
        df["lower_wick"]
        if "lower_wick" in df.columns
        else pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
    )
    close_location = (
        df["close_location"]
        if "close_location" in df.columns
        else safe_divide(df["close"] - df["low"], candle_range, fill_value=0.5)
    )

    bullish_rejection = ((lower_wick > body_abs * 1.5) & (close_location > 0.6)).fillna(False)
    bearish_rejection = ((upper_wick > body_abs * 1.5) & (close_location < 0.4)).fillna(False)
    out["bullish_rejection"] = bullish_rejection.astype(int)
    out["bearish_rejection"] = bearish_rejection.astype(int)

    sweep_high = _event_series(df, "sweep_high_20")
    sweep_low = _event_series(df, "sweep_low_20")
    displacement_bull = _event_series(df, "displacement_bullish")
    displacement_bear = _event_series(df, "displacement_bearish")
    bos_bull = _event_series(df, "ict_bos_bull")
    bos_bear = _event_series(df, "ict_bos_bear")
    choch_bull = _event_series(df, "ict_choch_bull")
    choch_bear = _event_series(df, "ict_choch_bear")
    swing_high = _event_series(df, "ict_confirmed_swing_high")
    swing_low = _event_series(df, "ict_confirmed_swing_low")

    sweep_any = sweep_high | sweep_low
    bos_any = bos_bull | bos_bear
    choch_any = choch_bull | choch_bear
    displacement_any = displacement_bull | displacement_bear
    rejection_any = bullish_rejection | bearish_rejection
    swing_any = swing_high | swing_low
    major_events = sweep_any | bos_any | choch_any | displacement_any | rejection_any

    out["bars_since_last_sweep"] = bars_since_event(sweep_any)
    out["bars_since_last_bos"] = bars_since_event(bos_any)
    out["bars_since_last_choch"] = bars_since_event(choch_any)
    out["bars_since_last_displacement"] = bars_since_event(displacement_any)
    out["bars_since_last_rejection"] = bars_since_event(rejection_any)
    out["cooldown_since_major_event"] = bars_since_event(major_events)

    out["major_event_density_10"] = major_events.astype(int).rolling(10, min_periods=1).mean()
    out["major_event_density_25"] = major_events.astype(int).rolling(25, min_periods=1).mean()
    out["major_event_cluster_10"] = major_events.astype(int).rolling(10, min_periods=1).sum()
    out["major_event_cluster_25"] = major_events.astype(int).rolling(25, min_periods=1).sum()
    out["swing_density_20"] = swing_any.astype(int).rolling(20, min_periods=1).mean()
    out["swing_density_50"] = swing_any.astype(int).rolling(50, min_periods=1).mean()

    last_sweep_low = _last_seen_index(sweep_low)
    last_sweep_high = _last_seen_index(sweep_high)
    last_displacement_bull = _last_seen_index(displacement_bull)
    last_displacement_bear = _last_seen_index(displacement_bear)
    last_bull_structure = np.maximum(_last_seen_index(bos_bull), _last_seen_index(choch_bull))
    last_bear_structure = np.maximum(_last_seen_index(bos_bear), _last_seen_index(choch_bear))

    atr = df["atr_14"] if "atr_14" in df.columns else calculate_atr(df)
    ema_distance = (
        df["close_vs_ema_8_atr"]
        if "close_vs_ema_8_atr" in df.columns
        else safe_divide(df["close"] - df["close"].ewm(span=8, adjust=False).mean(), atr)
    )
    near_trend = ema_distance.abs() <= 1.0
    current_index = np.arange(len(df))

    bull_sequence = (
        (last_sweep_low >= 0)
        & (last_displacement_bull > last_sweep_low)
        & ((current_index - last_displacement_bull) <= 10)
        & ((last_displacement_bull - last_sweep_low) <= 8)
    )
    bear_sequence = (
        (last_sweep_high >= 0)
        & (last_displacement_bear > last_sweep_high)
        & ((current_index - last_displacement_bear) <= 10)
        & ((last_displacement_bear - last_sweep_high) <= 8)
    )

    out["bull_sequence_sweep_displacement"] = bull_sequence.astype(int)
    out["bear_sequence_sweep_displacement"] = bear_sequence.astype(int)
    out["bull_sequence_sweep_displacement_retrace"] = (
        bull_sequence & near_trend.to_numpy(dtype=bool) & (last_bull_structure >= last_displacement_bull)
    ).astype(int)
    out["bear_sequence_sweep_displacement_retrace"] = (
        bear_sequence & near_trend.to_numpy(dtype=bool) & (last_bear_structure >= last_displacement_bear)
    ).astype(int)

    calendar = build_intraday_calendar_frame(
        df["datetime"],
        source_timezone=config.source_timezone,
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
        london_killzone_reference_timezone=config.london_killzone_reference_timezone,
        london_killzone_start_hour=config.london_killzone_start_hour,
        london_killzone_end_hour=config.london_killzone_end_hour,
        new_york_killzone_reference_timezone=config.new_york_killzone_reference_timezone,
        new_york_killzone_start_hour=config.new_york_killzone_start_hour,
        new_york_killzone_end_hour=config.new_york_killzone_end_hour,
    )
    hour = calendar["feature_hour"].fillna(0).astype(int)
    out["in_london_early"] = calendar["in_london_killzone"].astype(int)
    out["in_london_mid"] = (
        calendar["in_london_session"].astype(bool)
        & ~calendar["in_london_killzone"].astype(bool)
        & (hour < 8)
    ).astype(int)
    out["in_london_late"] = (
        calendar["in_london_session"].astype(bool) & ~out["in_london_early"].astype(bool) & (hour >= 8)
    ).astype(int)
    out["in_newyork_early"] = calendar["in_newyork_killzone"].astype(int)
    out["in_newyork_mid"] = (
        (hour >= config.new_york_killzone_end_hour)
        & calendar["in_newyork_session"].astype(bool)
        & (hour < 14)
    ).astype(int)
    out["in_newyork_late"] = ((hour >= 14) & calendar["in_newyork_session"].astype(bool)).astype(int)

    return out
