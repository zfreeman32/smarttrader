from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import bars_since_event, calculate_atr, detect_confirmed_swings, safe_divide

Zone = tuple[float, float, int]


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
) -> tuple[float, float]:
    if not zones or atr <= 0:
        return np.nan, np.nan

    nearest_distance = np.inf
    nearest_age = np.nan
    for lower, upper, formed_idx in zones:
        distance = _distance_to_zone(price, lower, upper)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_age = float(current_index - formed_idx)

    if not np.isfinite(nearest_distance):
        return np.nan, np.nan
    return nearest_distance / atr, nearest_age


@register_feature_set(
    name="ict_context",
    category="context",
    description="ICT proximity features for FVGs, order blocks, liquidity pools, and structure breaks",
    required_columns=("open", "high", "low", "close"),
)
def build_ict_context(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    atr = df["atr_14"] if "atr_14" in df.columns else calculate_atr(df)
    candle_range = df["high"] - df["low"]
    body_size = (df["close"] - df["open"]).abs()
    body_to_range = safe_divide(body_size, candle_range, fill_value=0.0)
    range_atr = safe_divide(candle_range, atr, fill_value=0.0)

    sweep_high = (
        df["sweep_high_20"]
        if "sweep_high_20" in df.columns
        else ((df["high"] > df["high"].shift(1).rolling(20).max()) & (df["close"] < df["high"].shift(1).rolling(20).max())).astype(int)
    )
    sweep_low = (
        df["sweep_low_20"]
        if "sweep_low_20" in df.columns
        else ((df["low"] < df["low"].shift(1).rolling(20).min()) & (df["close"] > df["low"].shift(1).rolling(20).min())).astype(int)
    )

    swing_high_event, swing_high_levels, swing_low_event, swing_low_levels = detect_confirmed_swings(
        df["high"],
        df["low"],
        window=config.swing_window,
    )
    latest_swing_high = swing_high_levels.ffill()
    latest_swing_low = swing_low_levels.ffill()

    out["ict_confirmed_swing_high"] = swing_high_event.astype(int)
    out["ict_confirmed_swing_low"] = swing_low_event.astype(int)
    out["ict_latest_swing_high"] = latest_swing_high
    out["ict_latest_swing_low"] = latest_swing_low
    out["ict_dist_to_swing_high_atr"] = safe_divide(latest_swing_high - df["close"], atr)
    out["ict_dist_to_swing_low_atr"] = safe_divide(df["close"] - latest_swing_low, atr)

    min_gap = atr * config.ict_fvg_min_gap_atr
    bull_fvg_form = ((df["low"] - df["high"].shift(2)) > min_gap).fillna(False)
    bear_fvg_form = ((df["low"].shift(2) - df["high"]) > min_gap).fillna(False)

    bull_ob_form = (
        (df["close"] > df["open"])
        & (df["close"].shift(1) < df["open"].shift(1))
        & (body_to_range > 0.6)
        & (range_atr >= config.ict_order_block_range_atr)
        & (df["close"] > df["high"].shift(1))
    ).fillna(False)
    bear_ob_form = (
        (df["close"] < df["open"])
        & (df["close"].shift(1) > df["open"].shift(1))
        & (body_to_range > 0.6)
        & (range_atr >= config.ict_order_block_range_atr)
        & (df["close"] < df["low"].shift(1))
    ).fillna(False)

    n_rows = len(df)
    closes = df["close"].to_numpy(dtype=float, copy=False)
    highs = df["high"].to_numpy(dtype=float, copy=False)
    lows = df["low"].to_numpy(dtype=float, copy=False)
    atr_values = atr.ffill().fillna(0.0).to_numpy(dtype=float, copy=False)
    bull_fvg_form_values = bull_fvg_form.to_numpy(dtype=bool, copy=False)
    bear_fvg_form_values = bear_fvg_form.to_numpy(dtype=bool, copy=False)
    bull_ob_form_values = bull_ob_form.to_numpy(dtype=bool, copy=False)
    bear_ob_form_values = bear_ob_form.to_numpy(dtype=bool, copy=False)
    swing_high_event_values = swing_high_event.to_numpy(dtype=bool, copy=False)
    swing_low_event_values = swing_low_event.to_numpy(dtype=bool, copy=False)
    swing_high_levels_values = swing_high_levels.to_numpy(dtype=float, copy=False)
    swing_low_levels_values = swing_low_levels.to_numpy(dtype=float, copy=False)
    latest_swing_high_values = latest_swing_high.to_numpy(dtype=float, copy=False)
    latest_swing_low_values = latest_swing_low.to_numpy(dtype=float, copy=False)
    prior_high_values = df["high"].shift(1).to_numpy(dtype=float)
    prior_low_values = df["low"].shift(1).to_numpy(dtype=float)
    earlier_high_values = df["high"].shift(2).to_numpy(dtype=float)
    earlier_low_values = df["low"].shift(2).to_numpy(dtype=float)
    sweep_high_values = pd.to_numeric(sweep_high, errors="coerce").fillna(0).to_numpy(dtype=int, copy=False)
    sweep_low_values = pd.to_numeric(sweep_low, errors="coerce").fillna(0).to_numpy(dtype=int, copy=False)
    max_zone_age = config.ict_zone_max_age

    bull_fvg_count = np.zeros(n_rows, dtype=float)
    bear_fvg_count = np.zeros(n_rows, dtype=float)
    bull_fvg_distance = np.full(n_rows, np.nan)
    bear_fvg_distance = np.full(n_rows, np.nan)

    bull_ob_distance = np.full(n_rows, np.nan)
    bear_ob_distance = np.full(n_rows, np.nan)
    bull_ob_age = np.full(n_rows, np.nan)
    bear_ob_age = np.full(n_rows, np.nan)

    equal_high_distance = np.full(n_rows, np.nan)
    equal_low_distance = np.full(n_rows, np.nan)
    equal_high_count = np.zeros(n_rows, dtype=float)
    equal_low_count = np.zeros(n_rows, dtype=float)

    bos_bull = np.zeros(n_rows, dtype=int)
    bos_bear = np.zeros(n_rows, dtype=int)
    choch_bull = np.zeros(n_rows, dtype=int)
    choch_bear = np.zeros(n_rows, dtype=int)

    active_bull_fvgs: list[Zone] = []
    active_bear_fvgs: list[Zone] = []
    active_bull_obs: list[Zone] = []
    active_bear_obs: list[Zone] = []
    active_equal_highs: list[Zone] = []
    active_equal_lows: list[Zone] = []

    previous_swing_high_level = np.nan
    previous_swing_high_idx = -1
    previous_swing_high_tolerance = np.nan
    previous_swing_low_level = np.nan
    previous_swing_low_idx = -1
    previous_swing_low_tolerance = np.nan

    trend_state = 0
    last_broken_high = np.nan
    last_broken_low = np.nan

    for i in range(n_rows):
        current_atr = atr_values[i] if atr_values[i] > 0 else 1.0
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]

        active_bull_fvgs = [
            zone for zone in active_bull_fvgs if (i - zone[2]) <= max_zone_age and current_low > zone[0]
        ]
        active_bear_fvgs = [
            zone for zone in active_bear_fvgs if (i - zone[2]) <= max_zone_age and current_high < zone[1]
        ]
        active_bull_obs = [
            zone for zone in active_bull_obs if (i - zone[2]) <= max_zone_age and current_close >= zone[0]
        ]
        active_bear_obs = [
            zone for zone in active_bear_obs if (i - zone[2]) <= max_zone_age and current_close <= zone[1]
        ]
        active_equal_highs = [
            zone for zone in active_equal_highs if (i - zone[2]) <= max_zone_age and current_high <= zone[1]
        ]
        active_equal_lows = [
            zone for zone in active_equal_lows if (i - zone[2]) <= max_zone_age and current_low >= zone[0]
        ]

        if bull_fvg_form_values[i] and np.isfinite(earlier_high_values[i]):
            active_bull_fvgs.append((float(earlier_high_values[i]), float(lows[i]), i))
        if bear_fvg_form_values[i] and np.isfinite(earlier_low_values[i]):
            active_bear_fvgs.append((float(highs[i]), float(earlier_low_values[i]), i))
        if bull_ob_form_values[i] and i >= 1 and np.isfinite(prior_low_values[i]) and np.isfinite(prior_high_values[i]):
            active_bull_obs.append((float(prior_low_values[i]), float(prior_high_values[i]), i))
        if bear_ob_form_values[i] and i >= 1 and np.isfinite(prior_low_values[i]) and np.isfinite(prior_high_values[i]):
            active_bear_obs.append((float(prior_low_values[i]), float(prior_high_values[i]), i))

        if swing_high_event_values[i] and np.isfinite(swing_high_levels_values[i]):
            current_level = float(swing_high_levels_values[i])
            tolerance = current_atr * config.ict_liquidity_tolerance_atr
            if (
                np.isfinite(previous_swing_high_level)
                and (i - previous_swing_high_idx) >= config.swing_window
                and abs(current_level - previous_swing_high_level)
                <= max(tolerance, previous_swing_high_tolerance)
            ):
                pool_level = (current_level + previous_swing_high_level) / 2.0
                zone_tolerance = max(tolerance, previous_swing_high_tolerance)
                active_equal_highs.append((pool_level - zone_tolerance, pool_level + zone_tolerance, i))
            previous_swing_high_level = current_level
            previous_swing_high_idx = i
            previous_swing_high_tolerance = tolerance

        if swing_low_event_values[i] and np.isfinite(swing_low_levels_values[i]):
            current_level = float(swing_low_levels_values[i])
            tolerance = current_atr * config.ict_liquidity_tolerance_atr
            if (
                np.isfinite(previous_swing_low_level)
                and (i - previous_swing_low_idx) >= config.swing_window
                and abs(current_level - previous_swing_low_level)
                <= max(tolerance, previous_swing_low_tolerance)
            ):
                pool_level = (current_level + previous_swing_low_level) / 2.0
                zone_tolerance = max(tolerance, previous_swing_low_tolerance)
                active_equal_lows.append((pool_level - zone_tolerance, pool_level + zone_tolerance, i))
            previous_swing_low_level = current_level
            previous_swing_low_idx = i
            previous_swing_low_tolerance = tolerance

        latest_high = latest_swing_high_values[i]
        latest_low = latest_swing_low_values[i]
        break_buffer = current_atr * config.ict_break_buffer_atr

        bull_break = (
            np.isfinite(latest_high)
            and current_high > float(latest_high) + break_buffer
            and (not np.isfinite(last_broken_high) or abs(float(latest_high) - last_broken_high) > break_buffer / 2.0)
        )
        bear_break = (
            np.isfinite(latest_low)
            and current_low < float(latest_low) - break_buffer
            and (not np.isfinite(last_broken_low) or abs(float(latest_low) - last_broken_low) > break_buffer / 2.0)
        )

        if bull_break:
            if trend_state == -1:
                choch_bull[i] = 1
            else:
                bos_bull[i] = 1
            trend_state = 1
            last_broken_high = float(latest_high)
        elif bear_break:
            if trend_state == 1:
                choch_bear[i] = 1
            else:
                bos_bear[i] = 1
            trend_state = -1
            last_broken_low = float(latest_low)

        bull_fvg_count[i] = len(active_bull_fvgs)
        bear_fvg_count[i] = len(active_bear_fvgs)
        bull_fvg_distance[i], _ = _nearest_zone_stats(
            active_bull_fvgs,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        bear_fvg_distance[i], _ = _nearest_zone_stats(
            active_bear_fvgs,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        bull_ob_distance[i], bull_ob_age[i] = _nearest_zone_stats(
            active_bull_obs,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        bear_ob_distance[i], bear_ob_age[i] = _nearest_zone_stats(
            active_bear_obs,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        equal_high_distance[i], _ = _nearest_zone_stats(
            active_equal_highs,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        equal_low_distance[i], _ = _nearest_zone_stats(
            active_equal_lows,
            price=current_close,
            atr=current_atr,
            current_index=i,
        )
        equal_high_count[i] = len(active_equal_highs)
        equal_low_count[i] = len(active_equal_lows)

    out["dist_to_bull_fvg_atr"] = bull_fvg_distance
    out["dist_to_bear_fvg_atr"] = bear_fvg_distance
    out["active_bull_fvg_count"] = bull_fvg_count
    out["active_bear_fvg_count"] = bear_fvg_count

    out["dist_to_bull_order_block_atr"] = bull_ob_distance
    out["dist_to_bear_order_block_atr"] = bear_ob_distance
    out["bull_order_block_age_bars"] = bull_ob_age
    out["bear_order_block_age_bars"] = bear_ob_age

    out["dist_to_equal_high_pool_atr"] = equal_high_distance
    out["dist_to_equal_low_pool_atr"] = equal_low_distance
    out["active_equal_high_pool_count"] = equal_high_count
    out["active_equal_low_pool_count"] = equal_low_count

    out["ict_bos_bull"] = bos_bull
    out["ict_bos_bear"] = bos_bear
    out["ict_choch_bull"] = choch_bull
    out["ict_choch_bear"] = choch_bear

    bos_any = pd.Series((bos_bull + bos_bear) > 0, index=df.index)
    choch_any = pd.Series((choch_bull + choch_bear) > 0, index=df.index)
    sweep_any = pd.Series((sweep_high_values + sweep_low_values) > 0, index=df.index)
    out["ict_bars_since_sweep_any"] = bars_since_event(sweep_any)
    out["ict_bars_since_bos_any"] = bars_since_event(bos_any)
    out["ict_bars_since_choch_any"] = bars_since_event(choch_any)

    out["ict_bull_confluence_1atr"] = (
        (out["dist_to_bull_fvg_atr"] <= 1.0).astype(int)
        + (out["dist_to_bull_order_block_atr"] <= 1.0).astype(int)
        + (out["dist_to_equal_low_pool_atr"] <= 1.0).astype(int)
    )
    out["ict_bear_confluence_1atr"] = (
        (out["dist_to_bear_fvg_atr"] <= 1.0).astype(int)
        + (out["dist_to_bear_order_block_atr"] <= 1.0).astype(int)
        + (out["dist_to_equal_high_pool_atr"] <= 1.0).astype(int)
    )
    out["ict_total_confluence_1atr"] = out["ict_bull_confluence_1atr"] + out["ict_bear_confluence_1atr"]
    out["ict_zone_balance_1atr"] = out["ict_bull_confluence_1atr"] - out["ict_bear_confluence_1atr"]

    return out
