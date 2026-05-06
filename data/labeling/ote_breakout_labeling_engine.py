"""
Breakout labeling engine for 5-minute FX data.

This module mirrors the broader structure of the reversal and continuation
labelers while targeting compression-to-expansion breakout setups.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from .reversal_labeling_engine import (
        ReversalParams,
        SwingPoint,
        _prep,
        compute_atr,
        compute_sample_weights,
        compute_swing_quality,
        create_exclusion_masks,
        detect_swings_zigzag,
        is_positive_swing,
        map_htf_atr_to_ltf,
        print_diagnostics,
        run_diagnostics,
        trend_scan_swings,
        validate_swings_triple_barrier,
    )
except ImportError:
    from reversal_labeling_engine import (  # type: ignore
        ReversalParams,
        SwingPoint,
        _prep,
        compute_atr,
        compute_sample_weights,
        compute_swing_quality,
        create_exclusion_masks,
        detect_swings_zigzag,
        is_positive_swing,
        map_htf_atr_to_ltf,
        print_diagnostics,
        run_diagnostics,
        trend_scan_swings,
        validate_swings_triple_barrier,
    )


@dataclass
class BreakoutParams(ReversalParams):
    """Parameter set tuned for smaller 5-minute breakout structures."""

    structural_atr_tf: str = "30m"
    confirm_atr_mult: float = 0.25
    min_swing_atr: float = 0.50
    min_swing_distance_atr: float = 0.25
    min_bars_between_swings: int = 12

    tb_profit_atr: float = 0.90
    tb_stop_atr: float = 0.70
    tb_max_bars: int = 24

    zone_pre_bars: int = 0
    zone_post_bars: int = 2
    entry_lookback_bars: int = 0
    entry_max_delay_after_swing: int = 4
    entry_eval_bars_after_confirm: int = 10
    entry_price_tolerance_atr: float = 0.55
    entry_min_followthrough_atr: float = 0.30
    entry_min_rr: float = 1.00

    trend_scan_windows: Tuple[int, ...] = (6, 12, 24)
    trend_scan_min_abs_t: float = 1.10
    min_label_quality: float = 0.25
    event_match_bars: int = 2
    exclusion_pre_bars: int = 5

    htf_confirm_atr_mult: float = 0.45
    htf_min_swing_atr: float = 0.50
    htf_min_bars_between: int = 3
    htf_confluence_window_minutes: int = 90

    breakout_lookback_bars: int = 20
    compression_lookback_bars: int = 12
    compression_max_range_atr: float = 1.80
    min_breakout_range_atr: float = 0.50
    breakout_close_buffer_atr: float = 0.05
    breakout_min_body_fraction: float = 0.25
    breakout_min_close_location: float = 0.65
    breakout_requires_compression: bool = True
    max_breakout_extension_atr: float = 0.80
    breakout_failure_tolerance_atr: float = 0.25
    entry_retest_tolerance_atr: float = 0.40

    trend_ema_fast: int = 21
    trend_ema_slow: int = 50
    require_30m_trend_alignment: bool = False
    require_1h_trend_alignment: bool = False
    min_trend_spread_atr: float = 0.05
    min_htf_trend_spread_atr: float = 0.03

    def validate(self):
        super().validate()
        assert self.structural_atr_tf in {"30m", "1hr"}
        assert self.breakout_lookback_bars >= 5
        assert self.compression_lookback_bars >= 3
        assert self.compression_max_range_atr > 0
        assert self.min_breakout_range_atr > 0
        assert self.breakout_close_buffer_atr >= 0
        assert 0 <= self.breakout_min_body_fraction <= 1
        assert 0 <= self.breakout_min_close_location <= 1
        assert self.max_breakout_extension_atr > 0
        assert self.breakout_failure_tolerance_atr >= 0
        assert self.entry_retest_tolerance_atr > 0
        assert self.trend_ema_fast >= 2
        assert self.trend_ema_slow > self.trend_ema_fast


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _map_series_to_ltf(series: pd.Series, ltf_index: pd.DatetimeIndex) -> pd.Series:
    combined = series.reindex(series.index.union(ltf_index)).ffill()
    return combined.reindex(ltf_index)


def _safe_atr(value: float, fallback: float) -> float:
    if np.isnan(value) or value <= 0:
        return fallback
    return float(value)


def _safe_bar_range(open_price: float, high: float, low: float, close_price: float) -> float:
    bar_range = high - low
    if bar_range > 0:
        return bar_range
    return max(abs(close_price - open_price), 1e-8)


def build_breakout_context(
    df_5m: pd.DataFrame,
    df_30m: pd.DataFrame,
    structural_atr: pd.Series,
    atr_30m: pd.Series,
    params: BreakoutParams,
    df_1hr: pd.DataFrame | None = None,
    atr_1hr: pd.Series | None = None,
) -> pd.DataFrame:
    """Build causal range/compression/trend context and map HTF fields to 5m."""

    ctx = pd.DataFrame(index=df_5m.index)
    atr_safe = structural_atr.replace(0.0, np.nan)

    prior_high = df_5m["high"].shift(1).rolling(
        window=params.breakout_lookback_bars,
        min_periods=params.breakout_lookback_bars,
    ).max()
    prior_low = df_5m["low"].shift(1).rolling(
        window=params.breakout_lookback_bars,
        min_periods=params.breakout_lookback_bars,
    ).min()
    compression_high = df_5m["high"].shift(1).rolling(
        window=params.compression_lookback_bars,
        min_periods=params.compression_lookback_bars,
    ).max()
    compression_low = df_5m["low"].shift(1).rolling(
        window=params.compression_lookback_bars,
        min_periods=params.compression_lookback_bars,
    ).min()

    bar_range = (df_5m["high"] - df_5m["low"]).replace(0.0, np.nan)
    body_fraction = (df_5m["close"] - df_5m["open"]).abs() / bar_range
    close_location_long = (df_5m["close"] - df_5m["low"]) / bar_range
    close_location_short = (df_5m["high"] - df_5m["close"]) / bar_range

    ctx["breakout_prior_range_high"] = prior_high
    ctx["breakout_prior_range_low"] = prior_low
    ctx["breakout_prior_range_width_atr"] = (prior_high - prior_low) / atr_safe
    ctx["breakout_compression_range_atr"] = (compression_high - compression_low) / atr_safe
    ctx["breakout_is_compressed"] = (
        ctx["breakout_compression_range_atr"] <= params.compression_max_range_atr
    ).fillna(False).astype(np.int8)
    ctx["breakout_body_fraction"] = body_fraction.fillna(0.0)
    ctx["breakout_close_location_long"] = close_location_long.fillna(0.0)
    ctx["breakout_close_location_short"] = close_location_short.fillna(0.0)

    ema_fast_30m = _ema(df_30m["close"], params.trend_ema_fast)
    ema_slow_30m = _ema(df_30m["close"], params.trend_ema_slow)
    spread_30m_atr = (ema_fast_30m - ema_slow_30m) / atr_30m.replace(0.0, np.nan)
    trend_up_30m = (ema_fast_30m > ema_slow_30m) & (spread_30m_atr >= params.min_trend_spread_atr)
    trend_down_30m = (ema_fast_30m < ema_slow_30m) & (-spread_30m_atr >= params.min_trend_spread_atr)
    ctx["breakout_trend_spread_30m_atr"] = _map_series_to_ltf(spread_30m_atr, df_5m.index)
    ctx["breakout_trend_up_30m"] = _map_series_to_ltf(trend_up_30m.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)
    ctx["breakout_trend_down_30m"] = _map_series_to_ltf(trend_down_30m.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)

    if df_1hr is not None and atr_1hr is not None:
        ema_fast_1h = _ema(df_1hr["close"], params.trend_ema_fast)
        ema_slow_1h = _ema(df_1hr["close"], params.trend_ema_slow)
        spread_1h_atr = (ema_fast_1h - ema_slow_1h) / atr_1hr.replace(0.0, np.nan)
        trend_up_1h = (ema_fast_1h > ema_slow_1h) & (spread_1h_atr >= params.min_htf_trend_spread_atr)
        trend_down_1h = (ema_fast_1h < ema_slow_1h) & (-spread_1h_atr >= params.min_htf_trend_spread_atr)
        ctx["breakout_trend_spread_1h_atr"] = _map_series_to_ltf(spread_1h_atr, df_5m.index)
        ctx["breakout_trend_up_1h"] = _map_series_to_ltf(trend_up_1h.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)
        ctx["breakout_trend_down_1h"] = _map_series_to_ltf(trend_down_1h.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)
    else:
        ctx["breakout_trend_spread_1h_atr"] = np.nan
        ctx["breakout_trend_up_1h"] = 0
        ctx["breakout_trend_down_1h"] = 0

    return ctx


def detect_breakout_events(
    df_5m: pd.DataFrame,
    structural_atr: pd.Series,
    breakout_context: pd.DataFrame,
    params: BreakoutParams,
) -> List[SwingPoint]:
    """Detect causal breakout candidates from prior range breaks after compression."""

    n = len(df_5m)
    if n < params.warmup_bars + params.breakout_lookback_bars + 5:
        return []

    opens = df_5m["open"].values
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values
    timestamps = df_5m.index
    atr_vals = structural_atr.values

    events: List[SwingPoint] = []
    last_long_idx = -params.min_bars_between_swings - 1
    last_short_idx = -params.min_bars_between_swings - 1
    start_idx = max(params.warmup_bars, params.breakout_lookback_bars, params.compression_lookback_bars)

    for i in range(start_idx, n):
        atr_at = _safe_atr(atr_vals[i], np.nan)
        range_high = breakout_context["breakout_prior_range_high"].iat[i]
        range_low = breakout_context["breakout_prior_range_low"].iat[i]
        range_width_atr = breakout_context["breakout_prior_range_width_atr"].iat[i]
        compression_range_atr = breakout_context["breakout_compression_range_atr"].iat[i]
        if np.isnan(atr_at) or atr_at <= 0:
            continue
        if np.isnan(range_high) or np.isnan(range_low) or np.isnan(range_width_atr):
            continue
        if range_width_atr < params.min_breakout_range_atr:
            continue
        if params.breakout_requires_compression and (
            np.isnan(compression_range_atr) or compression_range_atr > params.compression_max_range_atr
        ):
            continue

        body_frac = float(breakout_context["breakout_body_fraction"].iat[i])
        close_loc_long = float(breakout_context["breakout_close_location_long"].iat[i])
        close_loc_short = float(breakout_context["breakout_close_location_short"].iat[i])
        prev_close = closes[i - 1] if i > 0 else closes[i]

        long_trend_ok = True
        short_trend_ok = True
        if params.require_30m_trend_alignment:
            long_trend_ok = bool(breakout_context["breakout_trend_up_30m"].iat[i])
            short_trend_ok = bool(breakout_context["breakout_trend_down_30m"].iat[i])
        if params.require_1h_trend_alignment:
            long_trend_ok = long_trend_ok and bool(breakout_context["breakout_trend_up_1h"].iat[i])
            short_trend_ok = short_trend_ok and bool(breakout_context["breakout_trend_down_1h"].iat[i])

        long_extension_atr = (closes[i] - range_high) / atr_at
        short_extension_atr = (range_low - closes[i]) / atr_at

        long_break = (
            i - last_long_idx >= params.min_bars_between_swings
            and long_trend_ok
            and prev_close <= range_high + params.breakout_close_buffer_atr * atr_at
            and highs[i] >= range_high
            and closes[i] >= range_high + params.breakout_close_buffer_atr * atr_at
            and long_extension_atr <= params.max_breakout_extension_atr
            and body_frac >= params.breakout_min_body_fraction
            and close_loc_long >= params.breakout_min_close_location
        )

        short_break = (
            i - last_short_idx >= params.min_bars_between_swings
            and short_trend_ok
            and prev_close >= range_low - params.breakout_close_buffer_atr * atr_at
            and lows[i] <= range_low
            and closes[i] <= range_low - params.breakout_close_buffer_atr * atr_at
            and short_extension_atr <= params.max_breakout_extension_atr
            and body_frac >= params.breakout_min_body_fraction
            and close_loc_short >= params.breakout_min_close_location
        )

        if long_break:
            event = SwingPoint(
                "low",
                timestamps[i],
                i,
                float(range_high),
                timestamps[i],
                i,
                1,
                float(atr_at),
                float(range_width_atr),
                "5m_breakout",
            )
            event.breakout_direction = "up"
            event.breakout_level = float(range_high)
            event.prior_range_high = float(range_high)
            event.prior_range_low = float(range_low)
            event.prior_range_width_atr = float(range_width_atr)
            event.compression_range_atr = float(compression_range_atr)
            event.breakout_extension_atr = float(long_extension_atr)
            event.breakout_close_strength = close_loc_long
            event.breakout_body_fraction = body_frac
            event.breakout_trend_spread_30m_atr = float(breakout_context["breakout_trend_spread_30m_atr"].iat[i])
            event.breakout_trend_spread_1h_atr = float(breakout_context["breakout_trend_spread_1h_atr"].iat[i])
            event.breakout_is_compressed = bool(breakout_context["breakout_is_compressed"].iat[i])
            events.append(event)
            last_long_idx = i

        if short_break:
            event = SwingPoint(
                "high",
                timestamps[i],
                i,
                float(range_low),
                timestamps[i],
                i,
                1,
                float(atr_at),
                float(range_width_atr),
                "5m_breakout",
            )
            event.breakout_direction = "down"
            event.breakout_level = float(range_low)
            event.prior_range_high = float(range_high)
            event.prior_range_low = float(range_low)
            event.prior_range_width_atr = float(range_width_atr)
            event.compression_range_atr = float(compression_range_atr)
            event.breakout_extension_atr = float(short_extension_atr)
            event.breakout_close_strength = close_loc_short
            event.breakout_body_fraction = body_frac
            event.breakout_trend_spread_30m_atr = float(breakout_context["breakout_trend_spread_30m_atr"].iat[i])
            event.breakout_trend_spread_1h_atr = float(breakout_context["breakout_trend_spread_1h_atr"].iat[i])
            event.breakout_is_compressed = bool(breakout_context["breakout_is_compressed"].iat[i])
            events.append(event)
            last_short_idx = i

    return events


def annotate_breakout_context(
    events: List[SwingPoint],
    htf_swings_30m: List[SwingPoint] | None = None,
    htf_swings_1h: List[SwingPoint] | None = None,
    *,
    window_minutes_30m: int,
    window_minutes_1h: int,
) -> List[SwingPoint]:
    """Attach HTF breakout agreement using highs for longs and lows for shorts."""

    highs_30m = sorted(s.swing_time.value for s in (htf_swings_30m or []) if s.swing_type == "high")
    lows_30m = sorted(s.swing_time.value for s in (htf_swings_30m or []) if s.swing_type == "low")
    highs_1h = sorted(s.swing_time.value for s in (htf_swings_1h or []) if s.swing_type == "high")
    lows_1h = sorted(s.swing_time.value for s in (htf_swings_1h or []) if s.swing_type == "low")
    bar_ns = int(pd.Timedelta(minutes=5).value)
    w30 = int(pd.Timedelta(minutes=window_minutes_30m).value)
    w1h = int(pd.Timedelta(minutes=window_minutes_1h).value)

    for event in sorted(events, key=lambda item: item.swing_index):
        target30 = highs_30m if getattr(event, "breakout_direction", "up") == "up" else lows_30m
        target1h = highs_1h if getattr(event, "breakout_direction", "up") == "up" else lows_1h

        pos30 = bisect_left(target30, event.swing_time.value)
        event.htf_match_30m = any(
            0 <= j < len(target30) and abs(target30[j] - event.swing_time.value) <= w30
            for j in (pos30 - 1, pos30)
        )
        prev30 = pos30 - 1
        if prev30 >= 0:
            event.bars_since_prev_30m_same = int((event.swing_time.value - target30[prev30]) / bar_ns)

        pos1h = bisect_left(target1h, event.swing_time.value)
        event.htf_match_1h = any(
            0 <= j < len(target1h) and abs(target1h[j] - event.swing_time.value) <= w1h
            for j in (pos1h - 1, pos1h)
        )
        prev1h = pos1h - 1
        if prev1h >= 0:
            event.bars_since_prev_1h_same = int((event.swing_time.value - target1h[prev1h]) / bar_ns)

    return events


def compute_breakout_htf_confluence(
    htf_swings: List[SwingPoint],
    n_ltf: int,
    ltf_timestamps: pd.DatetimeIndex,
    window_minutes: int,
):
    """Mark LTF bars near HTF highs for long breakouts and HTF lows for shorts."""

    long_conf = np.zeros(n_ltf, dtype=np.int8)
    short_conf = np.zeros(n_ltf, dtype=np.int8)
    if not htf_swings:
        return long_conf, short_conf

    ltf_ns = ltf_timestamps.values.astype(np.int64)
    window_ns = int(pd.Timedelta(minutes=window_minutes).value)
    for sp in htf_swings:
        t_ns = sp.swing_time.value
        start = int(np.searchsorted(ltf_ns, t_ns - window_ns, side="left"))
        end = int(np.searchsorted(ltf_ns, t_ns + window_ns, side="right"))
        start, end = max(0, start), min(n_ltf, end)
        if sp.swing_type == "high":
            long_conf[start:end] = 1
        else:
            short_conf[start:end] = 1
    return long_conf, short_conf


def assign_breakout_entry_labels(df, events, structural_atr, params):
    """Pick the best executable breakout or retest bar after confirmation."""

    n = len(df)
    long_labels = np.zeros(n, dtype=np.int8)
    short_labels = np.zeros(n, dtype=np.int8)
    if not events:
        return long_labels, short_labels

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index
    atr_vals = structural_atr.values

    for event in events:
        if not is_positive_swing(event, params):
            continue

        window_start = event.confirm_index
        window_end = min(n - 2, event.confirm_index + params.entry_max_delay_after_swing)
        if window_end < window_start:
            continue
        eval_end = min(n - 1, event.confirm_index + params.entry_eval_bars_after_confirm)

        best_idx = None
        best_score = -np.inf
        best_rr = None
        best_followthrough_atr = None
        breakout_level = float(getattr(event, "breakout_level", event.swing_price))

        for idx in range(window_start, window_end + 1):
            atr_at = _safe_atr(atr_vals[idx], event.atr_at_swing)
            if atr_at <= 0 or np.isnan(atr_at):
                continue
            if idx + 1 > eval_end:
                continue

            entry = closes[idx]
            future_high = np.max(highs[idx + 1 : eval_end + 1])
            future_low = np.min(lows[idx + 1 : eval_end + 1])
            bar_range = _safe_bar_range(opens[idx], highs[idx], lows[idx], closes[idx])
            body_frac = abs(closes[idx] - opens[idx]) / bar_range
            timing_bonus = 1.0 - ((idx - event.confirm_index) / max(params.entry_max_delay_after_swing, 1))

            if event.swing_type == "low":
                favorable = future_high - entry
                adverse = max(0.0, entry - future_low)
                close_location = (closes[idx] - lows[idx]) / bar_range
                extension_atr = max(0.0, (entry - breakout_level) / atr_at)
                retest_distance_atr = abs(lows[idx] - breakout_level) / atr_at
                hold_score = 1.0 if lows[idx] >= breakout_level - params.breakout_failure_tolerance_atr * atr_at else 0.0
            else:
                favorable = entry - future_low
                adverse = max(0.0, future_high - entry)
                close_location = (highs[idx] - closes[idx]) / bar_range
                extension_atr = max(0.0, (breakout_level - entry) / atr_at)
                retest_distance_atr = abs(highs[idx] - breakout_level) / atr_at
                hold_score = 1.0 if highs[idx] <= breakout_level + params.breakout_failure_tolerance_atr * atr_at else 0.0

            rr = favorable / max(adverse, 0.10 * atr_at)
            followthrough_atr = favorable / atr_at
            retest_score = 1.0 - min(retest_distance_atr / params.entry_retest_tolerance_atr, 1.0)

            score = (
                0.30 * rr
                + 0.25 * followthrough_atr
                + 0.15 * retest_score
                + 0.10 * close_location
                + 0.10 * body_frac
                + 0.10 * timing_bonus
            )

            if (
                followthrough_atr >= params.entry_min_followthrough_atr
                and rr >= params.entry_min_rr
                and extension_atr <= params.max_breakout_extension_atr
                and hold_score > 0
                and score > best_score
            ):
                best_idx = idx
                best_score = score
                best_rr = rr
                best_followthrough_atr = followthrough_atr

        if best_idx is None:
            continue

        event.entry_index = best_idx
        event.entry_time = timestamps[best_idx]
        event.entry_price = closes[best_idx]
        event.entry_score = best_score
        event.entry_rr = best_rr
        event.entry_followthrough_atr = best_followthrough_atr

        label_end = min(n, best_idx + params.entry_label_bars)
        if event.swing_type == "low":
            long_labels[best_idx:label_end] = 1
        else:
            short_labels[best_idx:label_end] = 1

    return long_labels, short_labels


def create_breakout_zone_labels(n, events, params):
    """Binary labels around breakout confirmation bars."""

    long_labels = np.zeros(n, dtype=np.int8)
    short_labels = np.zeros(n, dtype=np.int8)
    for event in events:
        if not is_positive_swing(event, params):
            continue
        start = max(0, event.confirm_index - params.zone_pre_bars)
        end = min(n, event.confirm_index + params.zone_post_bars + 1)
        if event.swing_type == "low":
            long_labels[start:end] = 1
        else:
            short_labels[start:end] = 1
    return long_labels, short_labels


def create_breakout_quality_arrays(n, events, params):
    long_quality = np.zeros(n, dtype=np.float32)
    short_quality = np.zeros(n, dtype=np.float32)
    entry_long_quality = np.zeros(n, dtype=np.float32)
    entry_short_quality = np.zeros(n, dtype=np.float32)
    for event in events:
        if not is_positive_swing(event, params):
            continue
        start = max(0, event.confirm_index - params.zone_pre_bars)
        end = min(n, event.confirm_index + params.zone_post_bars + 1)
        if event.swing_type == "low":
            long_quality[start:end] = np.maximum(long_quality[start:end], event.label_quality)
            if event.entry_index is not None:
                entry_long_quality[event.entry_index : min(n, event.entry_index + params.entry_label_bars)] = np.maximum(
                    entry_long_quality[event.entry_index : min(n, event.entry_index + params.entry_label_bars)],
                    event.label_quality,
                )
        else:
            short_quality[start:end] = np.maximum(short_quality[start:end], event.label_quality)
            if event.entry_index is not None:
                entry_short_quality[event.entry_index : min(n, event.entry_index + params.entry_label_bars)] = np.maximum(
                    entry_short_quality[event.entry_index : min(n, event.entry_index + params.entry_label_bars)],
                    event.label_quality,
                )
    return long_quality, short_quality, entry_long_quality, entry_short_quality


def compute_breakout_label_concurrency(n, events, params):
    """Count overlapping breakout label windows at each bar."""

    long_conc = np.zeros(n, dtype=np.int32)
    short_conc = np.zeros(n, dtype=np.int32)
    for event in events:
        start = max(0, event.confirm_index - params.zone_pre_bars)
        end = min(n, event.confirm_index + params.zone_post_bars + 1)
        if event.swing_type == "low":
            long_conc[start:end] += 1
        else:
            short_conc[start:end] += 1
    return long_conc, short_conc


def plot_breakouts(df, events, ll, ls, start_date=None, end_date=None, save_path=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available.")
        return

    ts, cl = df.index, df["close"].values
    mask = np.ones(len(ts), dtype=bool)
    if start_date:
        mask &= ts >= pd.Timestamp(start_date)
    if end_date:
        mask &= ts <= pd.Timestamp(end_date)
    idx = np.where(mask)[0]
    if not len(idx):
        print("No data in range.")
        return

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(ts[mask], cl[mask], color="#333", linewidth=0.7)

    for i in idx:
        if ll[i]:
            ax.axvspan(ts[i], ts[min(i + 1, len(ts) - 1)], alpha=0.20, color="green", lw=0)
        if ls[i]:
            ax.axvspan(ts[i], ts[min(i + 1, len(ts) - 1)], alpha=0.20, color="red", lw=0)

    idx_set = set(idx)
    for event in events:
        if event.confirm_index not in idx_set:
            continue
        if event.tb_outcome == "sl":
            continue
        marker, color = ("^", "green") if event.swing_type == "low" else ("v", "red")
        ax.plot(
            event.confirm_time,
            cl[event.confirm_index],
            marker,
            color=color,
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
        )

    ax.set_title("Breakout Labels: Compression-to-Expansion Structures", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#333", lw=1, label="Close"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="green", ms=10, label="Long Breakout"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="red", ms=10, label="Short Breakout"),
        ],
        loc="upper left",
        fontsize=9,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def build_breakout_labels(
    df_5m,
    df_30m,
    params: BreakoutParams | None = None,
    df_1hr=None,
    verbose: bool = True,
):
    """Master pipeline for breakout labels."""

    if params is None:
        params = BreakoutParams()
    params.validate()

    df_5m = _prep(df_5m)
    df_30m = _prep(df_30m)
    anomaly_count = int(df_5m.attrs.get("anomaly_count", 0))
    n = len(df_5m)
    if verbose:
        print(f"[Breakout] {n:,} bars on 5m")

    atr_5m = compute_atr(df_5m, params.atr_period, params.atr_smoothing)
    atr_30m = compute_atr(df_30m, params.atr_period, params.atr_smoothing)
    atr_1hr = None
    if df_1hr is not None:
        df_1hr = _prep(df_1hr)
        atr_1hr = compute_atr(df_1hr, params.atr_period, params.atr_smoothing)

    if params.structural_atr_tf == "1hr" and df_1hr is not None and atr_1hr is not None:
        satr_5m = map_htf_atr_to_ltf(atr_1hr, df_5m.index)
        satr_30m = map_htf_atr_to_ltf(atr_1hr, df_30m.index)
        if verbose:
            valid = satr_5m.dropna()
            if len(valid):
                print(f"[Breakout] Structural ATR (1hr->5m): {valid.mean():.6f} ({valid.mean() * 10000:.1f} pips)")
    else:
        satr_5m = map_htf_atr_to_ltf(atr_30m, df_5m.index)
        satr_30m = atr_30m
        if verbose:
            valid = satr_5m.dropna()
            if len(valid):
                print(f"[Breakout] Structural ATR (30m->5m): {valid.mean():.6f} ({valid.mean() * 10000:.1f} pips)")

    df_5m["atr"] = atr_5m
    df_5m["structural_atr"] = satr_5m

    breakout_context = build_breakout_context(
        df_5m,
        df_30m,
        satr_5m,
        atr_30m,
        params,
        df_1hr=df_1hr,
        atr_1hr=atr_1hr,
    )
    for column in breakout_context.columns:
        df_5m[column] = breakout_context[column]

    sw30 = detect_swings_zigzag(
        df_30m,
        satr_30m,
        params.htf_confirm_atr_mult,
        params.htf_min_swing_atr,
        params.min_swing_distance_atr * 0.8,
        params.htf_min_bars_between,
        params.warmup_bars,
        params.confirm_use_close,
        "30m",
    )
    if verbose:
        print(f"[Breakout] 30m swings: {len(sw30)}")

    sw1h: List[SwingPoint] = []
    if df_1hr is not None and atr_1hr is not None:
        sw1h = detect_swings_zigzag(
            df_1hr,
            atr_1hr,
            params.htf_confirm_atr_mult,
            params.htf_min_swing_atr * 0.8,
            params.min_swing_distance_atr * 0.7,
            max(2, params.htf_min_bars_between // 2),
            params.warmup_bars,
            params.confirm_use_close,
            "1hr",
        )
        if verbose:
            print(f"[Breakout] 1hr swings: {len(sw1h)}")

    events = detect_breakout_events(df_5m, satr_5m, breakout_context, params)
    if verbose:
        print(f"[Breakout] Raw breakout events: {len(events)}")

    events = annotate_breakout_context(
        events,
        sw30,
        sw1h,
        window_minutes_30m=params.htf_confluence_window_minutes,
        window_minutes_1h=params.htf_confluence_window_minutes * 2,
    )

    events = validate_swings_triple_barrier(df_5m, events, satr_5m, params)
    events = trend_scan_swings(df_5m, events, params)
    _, _ = assign_breakout_entry_labels(df_5m, events, satr_5m, params)
    events = compute_swing_quality(events, params)
    entry_long, entry_short = assign_breakout_entry_labels(df_5m, events, satr_5m, params)

    ll, ls = create_breakout_zone_labels(n, events, params)
    ql, qs, eql, eqs = create_breakout_quality_arrays(n, events, params)
    el, es, nl, ns = create_exclusion_masks(n, events, ll, ls, params)

    df_5m["label_long_breakout"] = ll
    df_5m["label_short_breakout"] = ls
    df_5m["label_long_breakout_entry"] = entry_long
    df_5m["label_short_breakout_entry"] = entry_short
    df_5m["label_quality_long_breakout"] = ql
    df_5m["label_quality_short_breakout"] = qs
    df_5m["entry_quality_long_breakout"] = eql
    df_5m["entry_quality_short_breakout"] = eqs
    df_5m["exclude_long_breakout"] = el
    df_5m["exclude_short_breakout"] = es
    df_5m["neg_ok_long_breakout"] = nl
    df_5m["neg_ok_short_breakout"] = ns

    hcl, hcs = compute_breakout_htf_confluence(sw30, n, df_5m.index, params.htf_confluence_window_minutes)
    df_5m["htf_confluence_long_breakout"] = hcl
    df_5m["htf_confluence_short_breakout"] = hcs

    if sw1h:
        h1l, h1s = compute_breakout_htf_confluence(sw1h, n, df_5m.index, params.htf_confluence_window_minutes * 2)
        df_5m["htf_confluence_long_breakout_1hr"] = h1l
        df_5m["htf_confluence_short_breakout_1hr"] = h1s

    if params.compute_uniqueness:
        cl, cs = compute_breakout_label_concurrency(n, events, params)
        df_5m["concurrency_long_breakout"] = cl
        df_5m["concurrency_short_breakout"] = cs
        df_5m["sample_weight_long_breakout"] = compute_sample_weights(cl, ll) * np.where(
            ll > 0,
            np.maximum(ql, params.min_label_quality),
            1.0,
        )
        df_5m["sample_weight_short_breakout"] = compute_sample_weights(cs, ls) * np.where(
            ls > 0,
            np.maximum(qs, params.min_label_quality),
            1.0,
        )
        df_5m["sample_weight_entry_long_breakout"] = compute_sample_weights(np.maximum(cl, 1), entry_long) * np.where(
            entry_long > 0,
            np.maximum(eql, params.min_label_quality),
            1.0,
        )
        df_5m["sample_weight_entry_short_breakout"] = compute_sample_weights(np.maximum(cs, 1), entry_short) * np.where(
            entry_short > 0,
            np.maximum(eqs, params.min_label_quality),
            1.0,
        )

    warmup_mask = np.zeros(n, dtype=bool)
    warmup_mask[: params.warmup_bars] = True
    df_5m["warmup_mask"] = warmup_mask

    diag = run_diagnostics(
        df_5m,
        events,
        ll,
        ls,
        el,
        es,
        nl,
        ns,
        satr_5m,
        params,
        hcl,
        hcs,
        {event.swing_index: event.htf_match_30m for event in events},
        entry_long,
        entry_short,
        anomaly_count,
    )
    diag["breakout_events_raw"] = len(events)
    if verbose:
        print_diagnostics(diag)

    return df_5m, diag, events
