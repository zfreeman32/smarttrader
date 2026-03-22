from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import (
    bars_since_event,
    calculate_atr,
    confirm_event_series,
    detect_swing_highs,
    detect_swing_lows,
    safe_divide,
)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@register_feature_set(
    name="exhaustion",
    category="core",
    description="Momentum deceleration, divergence, compression, and failed-continuation features",
    required_columns=("open", "high", "low", "close"),
)
def build_exhaustion(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    atr = df["atr_14"] if "atr_14" in df.columns else calculate_atr(df)
    rsi = df["rsi_14"] if "rsi_14" in df.columns else _compute_rsi(df["close"])
    returns = df["close"].pct_change()
    bar_range = df["high"] - df["low"]

    out["return_slope_3"] = returns.rolling(3, min_periods=3).mean()
    out["return_slope_8"] = returns.rolling(8, min_periods=8).mean()
    out["return_slope_decay_3_8"] = out["return_slope_3"] - out["return_slope_8"]
    out["return_slope_decay_8_21"] = out["return_slope_8"] - returns.rolling(21, min_periods=21).mean()

    out["rsi_slope_3"] = rsi.diff(3) / 3.0
    out["rsi_slope_8"] = rsi.diff(8) / 8.0
    out["rsi_slope_decay_3_8"] = out["rsi_slope_3"] - out["rsi_slope_8"]
    out["rsi_slope_decay_8_21"] = out["rsi_slope_8"] - (rsi.diff(21) / 21.0)

    raw_swing_highs = detect_swing_highs(df["high"], window=config.swing_window)
    raw_swing_lows = detect_swing_lows(df["low"], window=config.swing_window)
    confirmed_highs, confirmed_high_levels = confirm_event_series(raw_swing_highs, df["high"], delay=config.swing_window)
    confirmed_lows, confirmed_low_levels = confirm_event_series(raw_swing_lows, df["low"], delay=config.swing_window)
    confirmed_high_rsi = rsi.where(raw_swing_highs).shift(config.swing_window)
    confirmed_low_rsi = rsi.where(raw_swing_lows).shift(config.swing_window)

    latest_swing_high = confirmed_high_levels.ffill()
    latest_swing_low = confirmed_low_levels.ffill()
    out["distance_from_bull_origin_atr"] = safe_divide(df["close"] - latest_swing_low, atr)
    out["distance_from_bear_origin_atr"] = safe_divide(latest_swing_high - df["close"], atr)

    bull_divergence = np.zeros(len(df), dtype=int)
    bear_divergence = np.zeros(len(df), dtype=int)
    bull_divergence_strength = np.zeros(len(df), dtype=float)
    bear_divergence_strength = np.zeros(len(df), dtype=float)

    previous_high_price = np.nan
    previous_high_rsi = np.nan
    previous_low_price = np.nan
    previous_low_rsi = np.nan

    atr_values = atr.ffill().fillna(0.0).to_numpy(dtype=float)
    for i in range(len(df)):
        current_atr = atr_values[i] if atr_values[i] > 0 else 1.0

        if bool(confirmed_highs.iloc[i]) and np.isfinite(confirmed_high_levels.iloc[i]) and np.isfinite(confirmed_high_rsi.iloc[i]):
            current_price = float(confirmed_high_levels.iloc[i])
            current_rsi = float(confirmed_high_rsi.iloc[i])
            if np.isfinite(previous_high_price) and np.isfinite(previous_high_rsi):
                if current_price > previous_high_price and current_rsi < previous_high_rsi:
                    bear_divergence[i] = 1
                    bear_divergence_strength[i] = (
                        max(current_price - previous_high_price, 0.0) / current_atr
                    ) * max(previous_high_rsi - current_rsi, 0.0) / 50.0
            previous_high_price = current_price
            previous_high_rsi = current_rsi

        if bool(confirmed_lows.iloc[i]) and np.isfinite(confirmed_low_levels.iloc[i]) and np.isfinite(confirmed_low_rsi.iloc[i]):
            current_price = float(confirmed_low_levels.iloc[i])
            current_rsi = float(confirmed_low_rsi.iloc[i])
            if np.isfinite(previous_low_price) and np.isfinite(previous_low_rsi):
                if current_price < previous_low_price and current_rsi > previous_low_rsi:
                    bull_divergence[i] = 1
                    bull_divergence_strength[i] = (
                        max(previous_low_price - current_price, 0.0) / current_atr
                    ) * max(current_rsi - previous_low_rsi, 0.0) / 50.0
            previous_low_price = current_price
            previous_low_rsi = current_rsi

    out["bullish_divergence_event"] = bull_divergence
    out["bearish_divergence_event"] = bear_divergence
    out["bullish_divergence_strength"] = bull_divergence_strength
    out["bearish_divergence_strength"] = bear_divergence_strength
    out["bars_since_bullish_divergence"] = bars_since_event(pd.Series(bull_divergence.astype(bool), index=df.index))
    out["bars_since_bearish_divergence"] = bars_since_event(pd.Series(bear_divergence.astype(bool), index=df.index))

    rolling_range_10 = bar_range.rolling(10, min_periods=10).mean()
    rolling_range_50 = bar_range.rolling(50, min_periods=50).mean()
    rolling_vol_10 = returns.rolling(10, min_periods=10).std()
    rolling_vol_50 = returns.rolling(50, min_periods=50).std()
    out["range_compression_10_50"] = safe_divide(rolling_range_10, rolling_range_50)
    out["volatility_compression_10_50"] = safe_divide(rolling_vol_10, rolling_vol_50)
    out["compression_regime"] = (
        (out["range_compression_10_50"] < 0.8) & (out["volatility_compression_10_50"] < 0.8)
    ).astype(int)
    out["range_expansion_shock_10"] = safe_divide(bar_range, rolling_range_10)

    if "volume_relative_20" in df.columns:
        volume_relative = df["volume_relative_20"]
    elif "volume" in df.columns:
        volume_relative = safe_divide(df["volume"], df["volume"].rolling(20).mean(), fill_value=1.0)
    else:
        volume_relative = pd.Series(1.0, index=df.index)
    volume_fade = (volume_relative.rolling(3, min_periods=3).mean() - volume_relative.rolling(10, min_periods=10).mean()).mul(-1)

    bull_impulse = out["distance_from_bull_origin_atr"].clip(lower=0)
    bear_impulse = out["distance_from_bear_origin_atr"].clip(lower=0)
    out["bull_volume_exhaustion"] = bull_impulse * volume_fade.clip(lower=0) * ((rsi - 50).clip(lower=0) / 50.0)
    out["bear_volume_exhaustion"] = bear_impulse * volume_fade.clip(lower=0) * ((50 - rsi).clip(lower=0) / 50.0)

    prior_high = df["prior_high_20"] if "prior_high_20" in df.columns else df["high"].shift(1).rolling(20).max()
    prior_low = df["prior_low_20"] if "prior_low_20" in df.columns else df["low"].shift(1).rolling(20).min()
    breakout_high = (df["high"] > prior_high).fillna(False)
    breakout_low = (df["low"] < prior_low).fillna(False)

    out["failed_breakout_bull"] = (breakout_high & (df["close"] < prior_high)).astype(int)
    out["failed_breakout_bear"] = (breakout_low & (df["close"] > prior_low)).astype(int)
    out["failed_continuation_bull"] = (breakout_high.shift(1, fill_value=False) & (df["close"] < prior_high)).astype(int)
    out["failed_continuation_bear"] = (breakout_low.shift(1, fill_value=False) & (df["close"] > prior_low)).astype(int)

    return out
