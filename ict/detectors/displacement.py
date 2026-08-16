from __future__ import annotations

import numpy as np
import pandas as pd

from features.transforms import bars_since_event, safe_divide

from ..common import cfg_value, get_atr_like, rolling_group_zscore


def detect_ict_displacement(
    df: pd.DataFrame,
    config: object,
    *,
    liquidity_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect objective displacement with ES volume confirmation."""

    out = pd.DataFrame(index=df.index)
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return out

    atr = get_atr_like(df)
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df.get("volume", pd.Series(np.nan, index=df.index)), errors="coerce")

    candle_range = high - low
    body = (close - open_).abs()
    body_to_range = safe_divide(body, candle_range, fill_value=0.0)
    range_atr = safe_divide(candle_range, atr, fill_value=0.0)
    body_atr = safe_divide(body, atr, fill_value=0.0)
    close_location = safe_divide(close - low, candle_range, fill_value=0.5)

    phase_groups = (
        liquidity_features["ict_session_phase_code"]
        if liquidity_features is not None and "ict_session_phase_code" in liquidity_features.columns
        else pd.Series(0, index=df.index)
    )
    volume_zscore = rolling_group_zscore(volume, phase_groups, window=50)
    volume_relative = safe_divide(
        volume,
        volume.groupby(phase_groups).transform(lambda series: series.shift(1).rolling(20, min_periods=5).mean()),
        fill_value=np.nan,
    )

    min_range_atr = float(cfg_value(config, "displacement_range_atr", "ict_displacement_range_atr", default=1.5))
    min_body_to_range = float(
        cfg_value(config, "displacement_body_to_range", "ict_displacement_body_to_range", default=0.65)
    )
    close_location_threshold = float(
        cfg_value(config, "displacement_close_location", "ict_displacement_close_location", default=0.75)
    )
    min_volume_zscore = float(
        cfg_value(config, "displacement_volume_zscore", "ict_displacement_volume_zscore", default=0.5)
    )

    volume_ok = volume_zscore.ge(min_volume_zscore).fillna(False)
    bullish = (
        (close > open_)
        & range_atr.ge(min_range_atr)
        & body_to_range.ge(min_body_to_range)
        & close_location.ge(close_location_threshold)
        & (volume_ok | volume.isna())
    )
    bearish = (
        (close < open_)
        & range_atr.ge(min_range_atr)
        & body_to_range.ge(min_body_to_range)
        & close_location.le(1.0 - close_location_threshold)
        & (volume_ok | volume.isna())
    )

    score = (
        (range_atr.clip(lower=0.0) * 0.45)
        + (body_to_range.clip(lower=0.0) * 0.25)
        + ((volume_zscore.fillna(0.0).clip(lower=0.0) / 3.0) * 0.20)
        + ((close_location - 0.5).abs().clip(lower=0.0) * 2.0 * 0.10)
    )

    bull_event = bullish.astype(bool)
    bear_event = bearish.astype(bool)
    any_event = bull_event | bear_event
    event_id = pd.Series(np.where(any_event, np.arange(len(df), dtype=float) + 1.0, np.nan), index=df.index)
    bull_event_id = event_id.where(bull_event)
    bear_event_id = event_id.where(bear_event)
    bull_origin = low.where(bull_event).ffill()
    bear_origin = high.where(bear_event).ffill()
    bull_index = pd.Series(np.where(bull_event, np.arange(len(df), dtype=float), np.nan), index=df.index).ffill()
    bear_index = pd.Series(np.where(bear_event, np.arange(len(df), dtype=float), np.nan), index=df.index).ffill()

    out["ict_displacement_body_atr"] = body_atr
    out["ict_displacement_range_atr"] = range_atr
    out["ict_displacement_body_to_range"] = body_to_range
    out["ict_displacement_close_location"] = close_location
    out["ict_displacement_volume_zscore"] = volume_zscore
    out["ict_displacement_volume_relative"] = volume_relative
    out["ict_displacement_score"] = score
    out["displacement_bullish"] = bullish.astype(np.int8)
    out["displacement_bearish"] = bearish.astype(np.int8)
    out["ict_displacement_event_id"] = pd.Series(event_id, index=df.index, dtype="Int64")
    out["ict_bull_displacement_event_id"] = pd.Series(bull_event_id, index=df.index, dtype="Int64")
    out["ict_bear_displacement_event_id"] = pd.Series(bear_event_id, index=df.index, dtype="Int64")
    out["ict_latest_bull_displacement_id"] = pd.Series(bull_event_id.ffill(), index=df.index, dtype="Int64")
    out["ict_latest_bear_displacement_id"] = pd.Series(bear_event_id.ffill(), index=df.index, dtype="Int64")
    out["ict_latest_bull_displacement_origin"] = bull_origin
    out["ict_latest_bear_displacement_origin"] = bear_origin
    out["ict_latest_bull_displacement_index"] = bull_index
    out["ict_latest_bear_displacement_index"] = bear_index
    out["ict_bars_since_displacement_bull"] = bars_since_event(pd.Series(bull_event, index=df.index))
    out["ict_bars_since_displacement_bear"] = bars_since_event(pd.Series(bear_event, index=df.index))
    out["ict_bars_since_displacement_any"] = bars_since_event(pd.Series(any_event, index=df.index))
    return out
