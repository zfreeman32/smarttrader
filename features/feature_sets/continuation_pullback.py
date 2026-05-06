from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import bars_since_event, calculate_atr, detect_confirmed_swings, proximity_score, safe_divide


def _series_or_fallback(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    fallback: pd.Series,
) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.to_numeric(fallback, errors="coerce")


def _resolve_trend_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "continuation_trend_up_30m" in df.columns and "continuation_trend_down_30m" in df.columns:
        bull_30m = df["continuation_trend_up_30m"].fillna(0).astype(bool)
        bear_30m = df["continuation_trend_down_30m"].fillna(0).astype(bool)
    elif "htf_30m_ema_alignment" in df.columns:
        bull_30m = df["htf_30m_ema_alignment"].fillna(0) > 0
        bear_30m = df["htf_30m_ema_alignment"].fillna(0) < 0
    else:
        bull_30m = df.get("ema_alignment", pd.Series(0, index=df.index)).fillna(0) > 0
        bear_30m = df.get("ema_alignment", pd.Series(0, index=df.index)).fillna(0) < 0

    if "continuation_trend_up_1h" in df.columns and "continuation_trend_down_1h" in df.columns:
        bull_1h = df["continuation_trend_up_1h"].fillna(0).astype(bool)
        bear_1h = df["continuation_trend_down_1h"].fillna(0).astype(bool)
        return bull_30m & bull_1h, bear_30m & bear_1h

    if "htf_1h_ema_alignment" in df.columns:
        bull_1h = df["htf_1h_ema_alignment"].fillna(0) > 0
        bear_1h = df["htf_1h_ema_alignment"].fillna(0) < 0
        return bull_30m & bull_1h, bear_30m & bear_1h

    return bull_30m, bear_30m


def _depth_ratio_quality(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    if value < 0.15:
        return float(value / 0.15)
    if value <= 0.65:
        return 1.0
    if value <= 1.0:
        return float(max(0.0, 1.0 - ((value - 0.65) / 0.35)))
    return 0.0


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if np.isfinite(value)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


@register_feature_set(
    name="continuation_pullback",
    category="context",
    description="Pullback geometry, retest quality, and trend-resumption structure features",
    required_columns=("open", "high", "low", "close"),
)
def build_continuation_pullback(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    atr = df["atr_14"] if "atr_14" in df.columns else calculate_atr(df)
    atr = pd.to_numeric(atr, errors="coerce").replace(0.0, np.nan)

    bull_trend, bear_trend = _resolve_trend_flags(df)

    prior_high = (
        df["prior_high_20"]
        if "prior_high_20" in df.columns
        else df["high"].shift(1).rolling(20, min_periods=20).max()
    )
    prior_low = (
        df["prior_low_20"]
        if "prior_low_20" in df.columns
        else df["low"].shift(1).rolling(20, min_periods=20).min()
    )

    breakout_high = (
        df["breakout_high_20"].fillna(0).astype(bool)
        if "breakout_high_20" in df.columns
        else (df["high"] > prior_high).fillna(False)
    )
    breakout_low = (
        df["breakout_low_20"].fillna(0).astype(bool)
        if "breakout_low_20" in df.columns
        else (df["low"] < prior_low).fillna(False)
    )
    displacement_bull = df.get("displacement_bullish", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    displacement_bear = df.get("displacement_bearish", pd.Series(0, index=df.index)).fillna(0).astype(bool)

    candle_range = (
        df["candle_range"]
        if "candle_range" in df.columns
        else (df["high"] - df["low"])
    )
    candle_body = (
        df["candle_body"]
        if "candle_body" in df.columns
        else (df["close"] - df["open"])
    )
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
    ).fillna(0.5)

    floor_distance = _series_or_fallback(
        df,
        (
            "continuation_close_vs_slow_30m_atr",
            "htf_30m_close_vs_ema_21_atr",
            "close_vs_ema_21_atr",
        ),
        safe_divide(df["close"] - df["close"].ewm(span=21, adjust=False).mean(), atr),
    )

    _, confirmed_high_levels, _, confirmed_low_levels = detect_confirmed_swings(
        df["high"],
        df["low"],
        window=config.swing_window,
    )
    latest_confirmed_high = confirmed_high_levels.ffill()
    latest_confirmed_low = confirmed_low_levels.ffill()

    bull_impulse_raw = (
        bull_trend
        & (displacement_bull | breakout_high)
        & (df["close"] > df["open"])
        & (close_location > 0.55)
    )
    bear_impulse_raw = (
        bear_trend
        & (displacement_bear | breakout_low)
        & (df["close"] < df["open"])
        & (close_location < 0.45)
    )
    bull_impulse_event = bull_impulse_raw & ~bull_impulse_raw.shift(1, fill_value=False)
    bear_impulse_event = bear_impulse_raw & ~bear_impulse_raw.shift(1, fill_value=False)

    n_rows = len(df)
    bull_bars_since_impulse = np.full(n_rows, np.nan, dtype=float)
    bear_bars_since_impulse = np.full(n_rows, np.nan, dtype=float)
    bull_bars_since_peak = np.full(n_rows, np.nan, dtype=float)
    bear_bars_since_trough = np.full(n_rows, np.nan, dtype=float)
    bull_impulse_magnitude = np.full(n_rows, np.nan, dtype=float)
    bear_impulse_magnitude = np.full(n_rows, np.nan, dtype=float)
    bull_pullback_depth = np.full(n_rows, np.nan, dtype=float)
    bear_pullback_depth = np.full(n_rows, np.nan, dtype=float)
    bull_pullback_ratio = np.full(n_rows, np.nan, dtype=float)
    bear_pullback_ratio = np.full(n_rows, np.nan, dtype=float)
    bull_retest_quality = np.zeros(n_rows, dtype=float)
    bear_retest_quality = np.zeros(n_rows, dtype=float)
    bull_pullback_active = np.zeros(n_rows, dtype=np.int8)
    bear_pullback_active = np.zeros(n_rows, dtype=np.int8)

    low_values = df["low"].to_numpy(dtype=float, copy=False)
    high_values = df["high"].to_numpy(dtype=float, copy=False)
    close_values = df["close"].to_numpy(dtype=float, copy=False)
    atr_values = atr.to_numpy(dtype=float, copy=False)
    prior_high_values = pd.to_numeric(prior_high, errors="coerce").to_numpy(dtype=float, copy=False)
    prior_low_values = pd.to_numeric(prior_low, errors="coerce").to_numpy(dtype=float, copy=False)
    latest_confirmed_high_values = latest_confirmed_high.to_numpy(dtype=float, copy=False)
    latest_confirmed_low_values = latest_confirmed_low.to_numpy(dtype=float, copy=False)
    floor_distance_values = pd.to_numeric(floor_distance, errors="coerce").to_numpy(dtype=float, copy=False)
    bull_trend_values = bull_trend.to_numpy(dtype=bool, copy=False)
    bear_trend_values = bear_trend.to_numpy(dtype=bool, copy=False)
    bull_event_values = bull_impulse_event.to_numpy(dtype=bool, copy=False)
    bear_event_values = bear_impulse_event.to_numpy(dtype=bool, copy=False)

    bull_last_impulse_idx = -1
    bull_last_peak_idx = -1
    bull_origin_price = np.nan
    bull_peak_price = np.nan
    bull_breakout_level = np.nan

    bear_last_impulse_idx = -1
    bear_last_trough_idx = -1
    bear_origin_price = np.nan
    bear_trough_price = np.nan
    bear_breakout_level = np.nan

    fallback_window = max(4, config.swing_window * 2)
    for index in range(n_rows):
        atr_at_bar = atr_values[index]
        atr_valid = np.isfinite(atr_at_bar) and atr_at_bar > 0.0

        if bull_event_values[index]:
            origin_price = latest_confirmed_low_values[index]
            if not np.isfinite(origin_price):
                start = max(0, index - fallback_window)
                window_low = np.nanmin(low_values[start : index + 1])
                origin_price = window_low if np.isfinite(window_low) else low_values[index]
            bull_last_impulse_idx = index
            bull_last_peak_idx = index
            bull_origin_price = origin_price
            bull_peak_price = high_values[index]
            bull_breakout_level = prior_high_values[index] if np.isfinite(prior_high_values[index]) else close_values[index]
        elif bull_last_impulse_idx >= 0 and np.isfinite(high_values[index]) and (
            not np.isfinite(bull_peak_price) or high_values[index] >= bull_peak_price
        ):
            bull_peak_price = high_values[index]
            bull_last_peak_idx = index

        if bear_event_values[index]:
            origin_price = latest_confirmed_high_values[index]
            if not np.isfinite(origin_price):
                start = max(0, index - fallback_window)
                window_high = np.nanmax(high_values[start : index + 1])
                origin_price = window_high if np.isfinite(window_high) else high_values[index]
            bear_last_impulse_idx = index
            bear_last_trough_idx = index
            bear_origin_price = origin_price
            bear_trough_price = low_values[index]
            bear_breakout_level = prior_low_values[index] if np.isfinite(prior_low_values[index]) else close_values[index]
        elif bear_last_impulse_idx >= 0 and np.isfinite(low_values[index]) and (
            not np.isfinite(bear_trough_price) or low_values[index] <= bear_trough_price
        ):
            bear_trough_price = low_values[index]
            bear_last_trough_idx = index

        if bull_last_impulse_idx >= 0 and atr_valid and np.isfinite(bull_origin_price) and np.isfinite(bull_peak_price):
            bull_bars_since_impulse[index] = float(index - bull_last_impulse_idx)
            bull_bars_since_peak[index] = float(index - bull_last_peak_idx)
            bull_impulse_magnitude[index] = max(bull_peak_price - bull_origin_price, 0.0) / atr_at_bar
            bull_pullback_depth[index] = max(bull_peak_price - low_values[index], 0.0) / atr_at_bar
            if bull_impulse_magnitude[index] > 1e-9:
                bull_pullback_ratio[index] = bull_pullback_depth[index] / bull_impulse_magnitude[index]

            pullback_is_active = (
                bull_trend_values[index]
                and bull_bars_since_peak[index] >= 1.0
                and bull_bars_since_peak[index] <= 48.0
                and bull_impulse_magnitude[index] >= 0.25
                and bull_pullback_depth[index] >= 0.10
                and bull_pullback_ratio[index] <= 1.25
            )
            bull_pullback_active[index] = int(pullback_is_active)

            ema_proximity = proximity_score(
                np.array([abs(floor_distance_values[index]) if np.isfinite(floor_distance_values[index]) else np.nan]),
                threshold=1.25,
            ).iat[0]
            breakout_distance_atr = (
                abs(close_values[index] - bull_breakout_level) / atr_at_bar
                if np.isfinite(bull_breakout_level)
                else np.nan
            )
            breakout_proximity = proximity_score(np.array([breakout_distance_atr]), threshold=1.5).iat[0]
            floor_quality = (
                float(np.clip((floor_distance_values[index] + 0.75) / 1.5, 0.0, 1.0))
                if np.isfinite(floor_distance_values[index])
                else 0.0
            )
            bull_retest_quality[index] = (
                _nanmean(
                    [
                        float(ema_proximity),
                        float(breakout_proximity),
                        floor_quality,
                        _depth_ratio_quality(float(bull_pullback_ratio[index])),
                    ]
                )
                if pullback_is_active
                else 0.0
            )

        if bear_last_impulse_idx >= 0 and atr_valid and np.isfinite(bear_origin_price) and np.isfinite(bear_trough_price):
            bear_bars_since_impulse[index] = float(index - bear_last_impulse_idx)
            bear_bars_since_trough[index] = float(index - bear_last_trough_idx)
            bear_impulse_magnitude[index] = max(bear_origin_price - bear_trough_price, 0.0) / atr_at_bar
            bear_pullback_depth[index] = max(high_values[index] - bear_trough_price, 0.0) / atr_at_bar
            if bear_impulse_magnitude[index] > 1e-9:
                bear_pullback_ratio[index] = bear_pullback_depth[index] / bear_impulse_magnitude[index]

            pullback_is_active = (
                bear_trend_values[index]
                and bear_bars_since_trough[index] >= 1.0
                and bear_bars_since_trough[index] <= 48.0
                and bear_impulse_magnitude[index] >= 0.25
                and bear_pullback_depth[index] >= 0.10
                and bear_pullback_ratio[index] <= 1.25
            )
            bear_pullback_active[index] = int(pullback_is_active)

            ema_proximity = proximity_score(
                np.array([abs(floor_distance_values[index]) if np.isfinite(floor_distance_values[index]) else np.nan]),
                threshold=1.25,
            ).iat[0]
            breakout_distance_atr = (
                abs(close_values[index] - bear_breakout_level) / atr_at_bar
                if np.isfinite(bear_breakout_level)
                else np.nan
            )
            breakout_proximity = proximity_score(np.array([breakout_distance_atr]), threshold=1.5).iat[0]
            floor_quality = (
                float(np.clip(((-floor_distance_values[index]) + 0.75) / 1.5, 0.0, 1.0))
                if np.isfinite(floor_distance_values[index])
                else 0.0
            )
            bear_retest_quality[index] = (
                _nanmean(
                    [
                        float(ema_proximity),
                        float(breakout_proximity),
                        floor_quality,
                        _depth_ratio_quality(float(bear_pullback_ratio[index])),
                    ]
                )
                if pullback_is_active
                else 0.0
            )

    smoothed_close = df["close"].rolling(3, min_periods=1).mean()
    raw_channel_slope_8 = safe_divide(
        smoothed_close - smoothed_close.shift(8),
        atr * 8.0,
        fill_value=np.nan,
    )
    bull_channel_slope = (-raw_channel_slope_8).clip(lower=0.0)
    bear_channel_slope = raw_channel_slope_8.clip(lower=0.0)

    lower_wick_ratio = safe_divide(lower_wick, candle_range, fill_value=0.0).clip(lower=0.0)
    upper_wick_ratio = safe_divide(upper_wick, candle_range, fill_value=0.0).clip(lower=0.0)
    body_ratio = safe_divide(body_abs, candle_range, fill_value=0.0).clip(lower=0.0)

    out["continuation_bull_impulse_event"] = bull_impulse_event.astype(int)
    out["continuation_bear_impulse_event"] = bear_impulse_event.astype(int)
    out["continuation_bars_since_bull_impulse"] = bull_bars_since_impulse
    out["continuation_bars_since_bear_impulse"] = bear_bars_since_impulse
    out["continuation_bull_impulse_magnitude_atr"] = bull_impulse_magnitude
    out["continuation_bear_impulse_magnitude_atr"] = bear_impulse_magnitude
    out["continuation_bull_pullback_depth_atr"] = bull_pullback_depth
    out["continuation_bear_pullback_depth_atr"] = bear_pullback_depth
    out["continuation_bull_pullback_depth_ratio"] = bull_pullback_ratio
    out["continuation_bear_pullback_depth_ratio"] = bear_pullback_ratio
    out["continuation_bull_pullback_active"] = bull_pullback_active
    out["continuation_bear_pullback_active"] = bear_pullback_active
    out["continuation_bull_pullback_channel_slope_8_atr"] = bull_channel_slope.where(
        pd.Series(bull_pullback_active.astype(bool), index=df.index),
        0.0,
    )
    out["continuation_bear_pullback_channel_slope_8_atr"] = bear_channel_slope.where(
        pd.Series(bear_pullback_active.astype(bool), index=df.index),
        0.0,
    )
    out["continuation_bull_retest_quality"] = bull_retest_quality
    out["continuation_bear_retest_quality"] = bear_retest_quality
    out["continuation_bull_wick_rejection_score"] = (
        lower_wick_ratio
        * close_location.clip(lower=0.0, upper=1.0)
        * (1.0 - body_ratio.clip(lower=0.0, upper=1.0))
        * pd.Series(bull_pullback_active, index=df.index)
    )
    out["continuation_bear_wick_rejection_score"] = (
        upper_wick_ratio
        * (1.0 - close_location.clip(lower=0.0, upper=1.0))
        * (1.0 - body_ratio.clip(lower=0.0, upper=1.0))
        * pd.Series(bear_pullback_active, index=df.index)
    )

    recent_high_3 = df["high"].shift(1).rolling(3, min_periods=3).max()
    recent_low_3 = df["low"].shift(1).rolling(3, min_periods=3).min()
    bull_pullback_recent = (
        pd.Series(bull_pullback_active, index=df.index)
        .shift(1, fill_value=0)
        .rolling(8, min_periods=1)
        .max()
        .astype(bool)
    )
    bear_pullback_recent = (
        pd.Series(bear_pullback_active, index=df.index)
        .shift(1, fill_value=0)
        .rolling(8, min_periods=1)
        .max()
        .astype(bool)
    )

    bull_resume_event = (
        bull_trend
        & bull_pullback_recent
        & (df["close"] > df["open"])
        & (close_location > 0.55)
        & ((df["close"] > recent_high_3) | displacement_bull | breakout_high)
    )
    bear_resume_event = (
        bear_trend
        & bear_pullback_recent
        & (df["close"] < df["open"])
        & (close_location < 0.45)
        & ((df["close"] < recent_low_3) | displacement_bear | breakout_low)
    )

    out["continuation_bull_resume_event"] = bull_resume_event.astype(int)
    out["continuation_bear_resume_event"] = bear_resume_event.astype(int)
    out["continuation_bars_since_bull_resume"] = bars_since_event(bull_resume_event)
    out["continuation_bars_since_bear_resume"] = bars_since_event(bear_resume_event)

    return out
