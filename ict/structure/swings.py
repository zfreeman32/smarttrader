from __future__ import annotations

import numpy as np
import pandas as pd

from features.transforms import detect_swing_highs, detect_swing_lows, safe_divide

from ..common import cfg_value, get_atr_like

Zone = tuple[float, float, float, int]


def _distance_to_zone(price: float, lower: float, upper: float) -> float:
    if np.isnan(lower) or np.isnan(upper):
        return np.nan
    if lower <= price <= upper:
        return 0.0
    if price < lower:
        return lower - price
    return price - upper


def _nearest_zone_stats(
    zones: list[Zone],
    *,
    price: float,
    atr: float,
    current_index: int,
) -> tuple[float, float, float]:
    if not zones or atr <= 0:
        return np.nan, np.nan, np.nan

    nearest_distance = np.inf
    nearest_age = np.nan
    nearest_mid = np.nan
    for lower, upper, mid, formed_idx in zones:
        distance = _distance_to_zone(price, lower, upper)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_age = float(current_index - formed_idx)
            nearest_mid = mid

    if not np.isfinite(nearest_distance):
        return np.nan, np.nan, np.nan
    return nearest_distance / atr, nearest_age, nearest_mid


def detect_ict_swings(
    df: pd.DataFrame,
    config: object,
) -> pd.DataFrame:
    """Detect causal confirmed swings and equal-high/low pools."""

    out = pd.DataFrame(index=df.index)
    if not {"high", "low", "close"}.issubset(df.columns):
        return out

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    atr = get_atr_like(df)
    window = int(cfg_value(config, "swing_window", default=3))
    max_zone_age = int(cfg_value(config, "ict_zone_max_age", default=120))
    tolerance_atr = float(cfg_value(config, "ict_liquidity_tolerance_atr", default=0.2))

    raw_high = detect_swing_highs(high, window=window)
    raw_low = detect_swing_lows(low, window=window)

    raw_high_index = pd.Series(np.where(raw_high, np.arange(len(df), dtype=float), np.nan), index=df.index)
    raw_low_index = pd.Series(np.where(raw_low, np.arange(len(df), dtype=float), np.nan), index=df.index)

    confirmed_high = raw_high.shift(window, fill_value=False)
    confirmed_low = raw_low.shift(window, fill_value=False)
    confirmed_high_level = high.where(raw_high).shift(window)
    confirmed_low_level = low.where(raw_low).shift(window)
    confirmed_high_pivot_idx = raw_high_index.shift(window)
    confirmed_low_pivot_idx = raw_low_index.shift(window)

    high_tax_price = (confirmed_high_level - close).where(confirmed_high)
    low_tax_price = (close - confirmed_low_level).where(confirmed_low)

    out["ict_raw_swing_high"] = raw_high.astype(np.int8)
    out["ict_raw_swing_low"] = raw_low.astype(np.int8)
    out["ict_confirmed_swing_high"] = confirmed_high.astype(np.int8)
    out["ict_confirmed_swing_low"] = confirmed_low.astype(np.int8)
    out["ict_confirmed_swing_high_level"] = confirmed_high_level
    out["ict_confirmed_swing_low_level"] = confirmed_low_level
    out["ict_confirmed_swing_high_pivot_index"] = confirmed_high_pivot_idx
    out["ict_confirmed_swing_low_pivot_index"] = confirmed_low_pivot_idx
    out["ict_swing_high_confirmation_tax_price"] = high_tax_price
    out["ict_swing_low_confirmation_tax_price"] = low_tax_price
    out["ict_swing_high_confirmation_tax_atr"] = safe_divide(high_tax_price, atr)
    out["ict_swing_low_confirmation_tax_atr"] = safe_divide(low_tax_price, atr)
    out["ict_latest_swing_high"] = confirmed_high_level.ffill()
    out["ict_latest_swing_low"] = confirmed_low_level.ffill()
    out["ict_latest_swing_high_index"] = confirmed_high_pivot_idx.ffill()
    out["ict_latest_swing_low_index"] = confirmed_low_pivot_idx.ffill()
    out["ict_dist_to_swing_high_atr"] = safe_divide(out["ict_latest_swing_high"] - close, atr)
    out["ict_dist_to_swing_low_atr"] = safe_divide(close - out["ict_latest_swing_low"], atr)

    active_equal_highs: list[Zone] = []
    active_equal_lows: list[Zone] = []
    previous_swing_high_level = np.nan
    previous_swing_high_idx = -1
    previous_swing_high_tolerance = np.nan
    previous_swing_low_level = np.nan
    previous_swing_low_idx = -1
    previous_swing_low_tolerance = np.nan

    equal_high_distance = np.full(len(df), np.nan)
    equal_low_distance = np.full(len(df), np.nan)
    equal_high_age = np.full(len(df), np.nan)
    equal_low_age = np.full(len(df), np.nan)
    equal_high_mid = np.full(len(df), np.nan)
    equal_low_mid = np.full(len(df), np.nan)
    equal_high_count = np.zeros(len(df), dtype=float)
    equal_low_count = np.zeros(len(df), dtype=float)

    highs = high.to_numpy(dtype=float, copy=False)
    lows = low.to_numpy(dtype=float, copy=False)
    closes = close.to_numpy(dtype=float, copy=False)
    atr_values = atr.ffill().fillna(0.0).to_numpy(dtype=float, copy=False)
    confirmed_high_values = confirmed_high.to_numpy(dtype=bool, copy=False)
    confirmed_low_values = confirmed_low.to_numpy(dtype=bool, copy=False)
    confirmed_high_level_values = confirmed_high_level.to_numpy(dtype=float, copy=False)
    confirmed_low_level_values = confirmed_low_level.to_numpy(dtype=float, copy=False)

    for i in range(len(df)):
        current_atr = atr_values[i] if atr_values[i] > 0 else 1.0
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]

        active_equal_highs = [
            zone for zone in active_equal_highs if (i - zone[3]) <= max_zone_age and current_high <= zone[1]
        ]
        active_equal_lows = [
            zone for zone in active_equal_lows if (i - zone[3]) <= max_zone_age and current_low >= zone[0]
        ]

        if confirmed_high_values[i] and np.isfinite(confirmed_high_level_values[i]):
            current_level = float(confirmed_high_level_values[i])
            tolerance = current_atr * tolerance_atr
            if (
                np.isfinite(previous_swing_high_level)
                and (i - previous_swing_high_idx) >= window
                and abs(current_level - previous_swing_high_level) <= max(tolerance, previous_swing_high_tolerance)
            ):
                mid = (current_level + previous_swing_high_level) / 2.0
                zone_tolerance = max(tolerance, previous_swing_high_tolerance)
                active_equal_highs.append((mid - zone_tolerance, mid + zone_tolerance, mid, i))
            previous_swing_high_level = current_level
            previous_swing_high_idx = i
            previous_swing_high_tolerance = tolerance

        if confirmed_low_values[i] and np.isfinite(confirmed_low_level_values[i]):
            current_level = float(confirmed_low_level_values[i])
            tolerance = current_atr * tolerance_atr
            if (
                np.isfinite(previous_swing_low_level)
                and (i - previous_swing_low_idx) >= window
                and abs(current_level - previous_swing_low_level) <= max(tolerance, previous_swing_low_tolerance)
            ):
                mid = (current_level + previous_swing_low_level) / 2.0
                zone_tolerance = max(tolerance, previous_swing_low_tolerance)
                active_equal_lows.append((mid - zone_tolerance, mid + zone_tolerance, mid, i))
            previous_swing_low_level = current_level
            previous_swing_low_idx = i
            previous_swing_low_tolerance = tolerance

        equal_high_distance[i], equal_high_age[i], equal_high_mid[i] = _nearest_zone_stats(
            active_equal_highs,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        equal_low_distance[i], equal_low_age[i], equal_low_mid[i] = _nearest_zone_stats(
            active_equal_lows,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        equal_high_count[i] = len(active_equal_highs)
        equal_low_count[i] = len(active_equal_lows)

    out["dist_to_equal_high_pool_atr"] = equal_high_distance
    out["dist_to_equal_low_pool_atr"] = equal_low_distance
    out["equal_high_pool_age_bars"] = equal_high_age
    out["equal_low_pool_age_bars"] = equal_low_age
    out["ict_equal_high_pool_level"] = equal_high_mid
    out["ict_equal_low_pool_level"] = equal_low_mid
    out["active_equal_high_pool_count"] = equal_high_count
    out["active_equal_low_pool_count"] = equal_low_count
    return out
