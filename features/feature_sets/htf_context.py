from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..fx_calendar import (
    build_market_day_close_labels,
    build_market_week_close_labels,
    normalize_datetime_series,
    resample_ohlcv_by_close_labels,
)
from ..registry import register_feature_set
from ..transforms import bars_since_event, calculate_atr, detect_confirmed_swings, safe_divide


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        aggregation["volume"] = "sum"

    out = df.resample(rule, label="right", closed="right").agg(aggregation)
    return out.dropna(subset=["open", "high", "low", "close"])


def _align_continuous(
    frame: pd.DataFrame,
    target_dt_index: pd.DatetimeIndex,
    output_index: pd.Index,
) -> pd.DataFrame:
    aligned = frame.reindex(target_dt_index, method="ffill")
    aligned.index = output_index
    return aligned


def _align_events(
    event: pd.Series,
    target_dt_index: pd.DatetimeIndex,
    output_index: pd.Index,
) -> pd.Series:
    aligned = event.astype(bool).reindex(target_dt_index, fill_value=False)
    return pd.Series(aligned.to_numpy(dtype=bool), index=output_index)


def _build_htf_block(
    frame: pd.DataFrame,
    *,
    prefix: str,
    swing_window: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    out = pd.DataFrame(index=frame.index)
    atr = calculate_atr(frame)
    ema_8 = frame["close"].ewm(span=8, adjust=False).mean()
    ema_21 = frame["close"].ewm(span=21, adjust=False).mean()
    ema_50 = frame["close"].ewm(span=50, adjust=False).mean()

    swing_high_event, swing_high_levels, swing_low_event, swing_low_levels = detect_confirmed_swings(
        frame["high"],
        frame["low"],
        window=swing_window,
    )

    latest_swing_high = swing_high_levels.ffill()
    latest_swing_low = swing_low_levels.ffill()

    out[f"{prefix}_latest_swing_high"] = latest_swing_high
    out[f"{prefix}_latest_swing_low"] = latest_swing_low
    out[f"{prefix}_dist_to_swing_high_atr"] = safe_divide(latest_swing_high - frame["close"], atr)
    out[f"{prefix}_dist_to_swing_low_atr"] = safe_divide(frame["close"] - latest_swing_low, atr)
    out[f"{prefix}_ema_alignment"] = np.where(
        (ema_8 > ema_21) & (ema_21 > ema_50),
        1,
        np.where((ema_8 < ema_21) & (ema_21 < ema_50), -1, 0),
    )
    out[f"{prefix}_close_vs_ema_21_atr"] = safe_divide(frame["close"] - ema_21, atr)
    out[f"{prefix}_ema_spread_8_21_atr"] = safe_divide(ema_8 - ema_21, atr)
    out[f"{prefix}_ema_spread_21_50_atr"] = safe_divide(ema_21 - ema_50, atr)

    return out, swing_high_event, swing_low_event


@register_feature_set(
    name="htf_context",
    category="context",
    description="Higher-timeframe swing, daily/weekly range, and HTF trend-alignment context",
    required_columns=("datetime", "open", "high", "low", "close"),
)
def build_htf_context(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    datetime_series = normalize_datetime_series(
        df["datetime"],
        source_timezone=config.source_timezone,
        canonical_timezone=config.canonical_timezone,
    )
    valid_mask = datetime_series.notna()
    if not valid_mask.any():
        return out

    base_columns = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        base_columns.append("volume")

    market = df.loc[valid_mask, base_columns].copy()
    market.index = pd.DatetimeIndex(datetime_series.loc[valid_mask])
    market = market[~market.index.duplicated(keep="last")].sort_index()

    target_dt_index = market.index
    output_index = df.index[valid_mask]
    atr_like = df["atr_14"] if "atr_14" in df.columns else calculate_atr(df)

    for rule, prefix in (("30min", "htf_30m"), ("1h", "htf_1h")):
        resampled = _resample_ohlcv(market, rule)
        if resampled.empty:
            continue

        block, swing_high_event, swing_low_event = _build_htf_block(
            resampled,
            prefix=prefix,
            swing_window=config.htf_swing_window,
        )
        out = out.join(_align_continuous(block, target_dt_index, output_index))

        aligned_high_event = _align_events(swing_high_event, target_dt_index, output_index)
        aligned_low_event = _align_events(swing_low_event, target_dt_index, output_index)
        out[f"{prefix}_bars_since_swing_high_event"] = bars_since_event(aligned_high_event)
        out[f"{prefix}_bars_since_swing_low_event"] = bars_since_event(aligned_low_event)

    daily_labels = build_market_day_close_labels(
        pd.Series(market.index, index=market.index),
        source_timezone=config.canonical_timezone,
        canonical_timezone=config.canonical_timezone,
        market_close_timezone=config.market_close_timezone,
        market_close_hour=config.market_close_hour,
        market_close_minute=config.market_close_minute,
    )
    daily = resample_ohlcv_by_close_labels(market, daily_labels)
    if not daily.empty:
        daily_levels = pd.DataFrame(index=daily.index)
        daily_levels["htf_prev_day_high"] = daily["high"].shift(1)
        daily_levels["htf_prev_day_low"] = daily["low"].shift(1)
        daily_levels["htf_rolling_daily_high"] = daily["high"].shift(1).rolling(5, min_periods=1).max()
        daily_levels["htf_rolling_daily_low"] = daily["low"].shift(1).rolling(5, min_periods=1).min()
        out = out.join(_align_continuous(daily_levels, target_dt_index, output_index))

    weekly_labels = build_market_week_close_labels(
        pd.Series(market.index, index=market.index),
        source_timezone=config.canonical_timezone,
        canonical_timezone=config.canonical_timezone,
        market_close_timezone=config.market_close_timezone,
        market_close_hour=config.market_close_hour,
        market_close_minute=config.market_close_minute,
    )
    weekly = resample_ohlcv_by_close_labels(market, weekly_labels)
    if not weekly.empty:
        weekly_levels = pd.DataFrame(index=weekly.index)
        weekly_levels["htf_rolling_weekly_high"] = weekly["high"].shift(1).rolling(4, min_periods=1).max()
        weekly_levels["htf_rolling_weekly_low"] = weekly["low"].shift(1).rolling(4, min_periods=1).min()
        out = out.join(_align_continuous(weekly_levels, target_dt_index, output_index))

    prev_day_high = out["htf_prev_day_high"] if "htf_prev_day_high" in out.columns else pd.Series(np.nan, index=df.index)
    prev_day_low = out["htf_prev_day_low"] if "htf_prev_day_low" in out.columns else pd.Series(np.nan, index=df.index)
    rolling_daily_high = (
        out["htf_rolling_daily_high"] if "htf_rolling_daily_high" in out.columns else pd.Series(np.nan, index=df.index)
    )
    rolling_daily_low = (
        out["htf_rolling_daily_low"] if "htf_rolling_daily_low" in out.columns else pd.Series(np.nan, index=df.index)
    )
    rolling_weekly_high = (
        out["htf_rolling_weekly_high"] if "htf_rolling_weekly_high" in out.columns else pd.Series(np.nan, index=df.index)
    )
    rolling_weekly_low = (
        out["htf_rolling_weekly_low"] if "htf_rolling_weekly_low" in out.columns else pd.Series(np.nan, index=df.index)
    )

    out["htf_dist_to_prev_day_high_atr"] = safe_divide(prev_day_high - df["close"], atr_like)
    out["htf_dist_to_prev_day_low_atr"] = safe_divide(df["close"] - prev_day_low, atr_like)
    out["htf_dist_to_rolling_daily_high_atr"] = safe_divide(
        rolling_daily_high - df["close"],
        atr_like,
    )
    out["htf_dist_to_rolling_daily_low_atr"] = safe_divide(
        df["close"] - rolling_daily_low,
        atr_like,
    )
    out["htf_dist_to_rolling_weekly_high_atr"] = safe_divide(
        rolling_weekly_high - df["close"],
        atr_like,
    )
    out["htf_dist_to_rolling_weekly_low_atr"] = safe_divide(
        df["close"] - rolling_weekly_low,
        atr_like,
    )

    if {"htf_30m_ema_alignment", "htf_1h_ema_alignment"}.issubset(out.columns):
        out["htf_alignment_score"] = out["htf_30m_ema_alignment"] + out["htf_1h_ema_alignment"]

    return out
