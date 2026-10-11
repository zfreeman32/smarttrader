"""Completed-bar adapter for the versioned OTE training-data repair.

Base input rows are identified by opening timestamps. Label engines calculate
on their decision/close timestamps; HTF rows are published at completion. The
adapter leaves historical target definitions and price arrays unchanged.
"""
from __future__ import annotations

import pandas as pd

from features.feature_sets.htf_context import _resample_ohlcv


def completed_ote_frames(base: pd.DataFrame, *, bar_minutes: float = 5):
    if not isinstance(base.index, pd.DatetimeIndex) or not base.index.is_unique or not base.index.is_monotonic_increasing:
        raise ValueError("Expected unique chronological bar-open timestamps")
    if bar_minutes <= 0:
        raise ValueError("Expected positive base bar duration")
    decisions = base.copy()
    decisions.index = base.index + pd.Timedelta(minutes=bar_minutes)
    halfhour = _resample_ohlcv(base, "30min", bar_minutes=bar_minutes)
    hour = _resample_ohlcv(base, "1h", bar_minutes=bar_minutes)
    return decisions, halfhour, hour


def publication_frames(base: pd.DataFrame, *, timestamp_semantics: str, bar_minutes: float):
    """Normalize either explicit convention to a completed-bar decision clock."""
    if timestamp_semantics not in {"bar_open", "bar_close"}:
        raise ValueError("Declare --bar-timestamp-semantics bar_open or bar_close before building training labels.")
    opening = base.copy()
    if timestamp_semantics == "bar_close":
        opening.index -= pd.Timedelta(minutes=bar_minutes)
    return completed_ote_frames(opening, bar_minutes=bar_minutes)


def completed_continuation_context(frame: pd.DataFrame, *, bar_minutes: float = 5):
    """Shared historical/live trend inputs, independent of labels or outcomes."""
    from data.labeling.labeling_engine import build_default_params
    from data.labeling.ote_continuation_pullback_labeling_engine import build_trend_context
    from data.labeling.reversal_labeling_engine import compute_atr
    raw = frame[['open', 'high', 'low', 'close', 'volume']].copy()
    raw.index = pd.DatetimeIndex(pd.to_datetime(frame['datetime'], utc=True))
    decisions, half, hour = completed_ote_frames(raw, bar_minutes=bar_minutes)
    params = build_default_params().continuation
    a30 = compute_atr(half, params.atr_period, params.atr_smoothing)
    a1 = compute_atr(hour, params.atr_period, params.atr_smoothing)
    context = build_trend_context(decisions, half, a30, params, hour, a1)
    context.index = frame.index
    return context
