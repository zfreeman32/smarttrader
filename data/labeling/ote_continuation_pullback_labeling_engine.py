"""
Continuation pullback labeling engine for FX data.

This module intentionally mirrors the structure of the reversal labeler
while targeting a different setup family: shallow pullbacks that resume an
existing higher-timeframe trend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from .reversal_labeling_engine import (
        ReversalParams,
        SwingPoint,
        _prep,
        annotate_swing_context,
        assign_entry_labels,
        compute_atr,
        compute_htf_confluence,
        compute_htf_swing_match,
        compute_htf_timing_features,
        compute_label_concurrency,
        compute_sample_weights,
        compute_swing_quality,
        create_exclusion_masks,
        create_quality_arrays,
        create_zone_labels,
        detect_swings_zigzag,
        format_bar_timeframe,
        infer_bar_minutes,
        map_htf_atr_to_ltf,
        print_diagnostics,
        retune_bar_count_params,
        run_diagnostics,
        trend_scan_swings,
        validate_swings_triple_barrier,
    )
except ImportError:
    from reversal_labeling_engine import (  # type: ignore
        ReversalParams,
        SwingPoint,
        _prep,
        annotate_swing_context,
        assign_entry_labels,
        compute_atr,
        compute_htf_confluence,
        compute_htf_swing_match,
        compute_htf_timing_features,
        compute_label_concurrency,
        compute_sample_weights,
        compute_swing_quality,
        create_exclusion_masks,
        create_quality_arrays,
        create_zone_labels,
        detect_swings_zigzag,
        format_bar_timeframe,
        infer_bar_minutes,
        map_htf_atr_to_ltf,
        print_diagnostics,
        retune_bar_count_params,
        run_diagnostics,
        trend_scan_swings,
        validate_swings_triple_barrier,
    )


@dataclass
class ContinuationPullbackParams(ReversalParams):
    """Parameter set tuned for smaller continuation-pullback structures."""

    structural_atr_tf: str = "30m"
    confirm_atr_mult: float = 0.45
    min_swing_atr: float = 0.70
    min_swing_distance_atr: float = 0.35
    min_bars_between_swings: int = 8

    tb_profit_atr: float = 0.80
    tb_stop_atr: float = 0.70
    tb_max_bars: int = 36

    zone_pre_bars: int = 1
    zone_post_bars: int = 1
    entry_lookback_bars: int = 2
    entry_max_delay_after_swing: int = 6
    entry_eval_bars_after_confirm: int = 10
    entry_price_tolerance_atr: float = 0.50
    entry_min_followthrough_atr: float = 0.25
    entry_min_rr: float = 1.05

    trend_scan_windows: Tuple[int, ...] = (6, 12, 24)
    trend_scan_min_abs_t: float = 1.25
    min_label_quality: float = 0.30
    event_match_bars: int = 3
    exclusion_pre_bars: int = 6

    htf_confirm_atr_mult: float = 0.45
    htf_min_swing_atr: float = 0.50
    htf_min_bars_between: int = 3
    htf_confluence_window_minutes: int = 90

    trend_ema_fast: int = 21
    trend_ema_slow: int = 50
    min_trend_spread_atr: float = 0.10
    require_1h_trend_alignment: bool = True
    min_htf_trend_spread_atr: float = 0.02
    pullback_min_atr: float = 0.18
    pullback_max_atr: float = 2.00
    max_pullback_bars: int = 48
    trend_floor_tolerance_atr: float = 0.50
    max_same_direction_break_atr: float = 0.40
    min_bounce_at_confirm_atr: float = 0.10

    def validate(self):
        super().validate()
        assert self.structural_atr_tf in {"30m", "1hr"}
        assert self.trend_ema_fast >= 2
        assert self.trend_ema_slow > self.trend_ema_fast
        assert self.min_trend_spread_atr >= 0
        assert self.min_htf_trend_spread_atr >= 0
        assert self.pullback_min_atr >= 0
        assert self.pullback_max_atr > self.pullback_min_atr
        assert self.max_pullback_bars >= 2
        assert self.trend_floor_tolerance_atr >= 0
        assert self.max_same_direction_break_atr >= 0
        assert self.min_bounce_at_confirm_atr >= 0


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _map_series_to_ltf(series: pd.Series, ltf_index: pd.DatetimeIndex) -> pd.Series:
    combined = series.reindex(series.index.union(ltf_index)).ffill()
    return combined.reindex(ltf_index)


def _safe_atr(value: float, fallback: float) -> float:
    if np.isnan(value) or value <= 0:
        return fallback
    return float(value)


def build_trend_context(
    df_5m: pd.DataFrame,
    df_30m: pd.DataFrame,
    atr_30m: pd.Series,
    params: ContinuationPullbackParams,
    df_1hr: pd.DataFrame | None = None,
    atr_1hr: pd.Series | None = None,
) -> pd.DataFrame:
    """Build causal trend context on 30m/1h and map it to the base bars."""

    ctx = pd.DataFrame(index=df_5m.index)

    ema_fast_30m = _ema(df_30m["close"], params.trend_ema_fast)
    ema_slow_30m = _ema(df_30m["close"], params.trend_ema_slow)
    spread_30m_atr = (ema_fast_30m - ema_slow_30m) / atr_30m.replace(0.0, np.nan)
    close_vs_slow_30m_atr = (df_30m["close"] - ema_slow_30m) / atr_30m.replace(0.0, np.nan)
    trend_up_30m = (ema_fast_30m > ema_slow_30m) & (spread_30m_atr >= params.min_trend_spread_atr)
    trend_down_30m = (ema_fast_30m < ema_slow_30m) & (-spread_30m_atr >= params.min_trend_spread_atr)

    ctx["continuation_ema_fast_30m"] = _map_series_to_ltf(ema_fast_30m, df_5m.index)
    ctx["continuation_ema_slow_30m"] = _map_series_to_ltf(ema_slow_30m, df_5m.index)
    ctx["continuation_trend_spread_30m_atr"] = _map_series_to_ltf(spread_30m_atr, df_5m.index)
    ctx["continuation_close_vs_slow_30m_atr"] = _map_series_to_ltf(close_vs_slow_30m_atr, df_5m.index)
    ctx["continuation_trend_up_30m"] = _map_series_to_ltf(trend_up_30m.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)
    ctx["continuation_trend_down_30m"] = _map_series_to_ltf(trend_down_30m.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)

    if df_1hr is not None and atr_1hr is not None:
        ema_fast_1h = _ema(df_1hr["close"], params.trend_ema_fast)
        ema_slow_1h = _ema(df_1hr["close"], params.trend_ema_slow)
        spread_1h_atr = (ema_fast_1h - ema_slow_1h) / atr_1hr.replace(0.0, np.nan)
        close_vs_slow_1h_atr = (df_1hr["close"] - ema_slow_1h) / atr_1hr.replace(0.0, np.nan)
        trend_up_1h = (ema_fast_1h > ema_slow_1h) & (spread_1h_atr >= params.min_htf_trend_spread_atr)
        trend_down_1h = (ema_fast_1h < ema_slow_1h) & (-spread_1h_atr >= params.min_htf_trend_spread_atr)

        ctx["continuation_ema_fast_1h"] = _map_series_to_ltf(ema_fast_1h, df_5m.index)
        ctx["continuation_ema_slow_1h"] = _map_series_to_ltf(ema_slow_1h, df_5m.index)
        ctx["continuation_trend_spread_1h_atr"] = _map_series_to_ltf(spread_1h_atr, df_5m.index)
        ctx["continuation_close_vs_slow_1h_atr"] = _map_series_to_ltf(close_vs_slow_1h_atr, df_5m.index)
        ctx["continuation_trend_up_1h"] = _map_series_to_ltf(trend_up_1h.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)
        ctx["continuation_trend_down_1h"] = _map_series_to_ltf(trend_down_1h.astype(np.int8), df_5m.index).fillna(0).astype(np.int8)
    else:
        ctx["continuation_ema_fast_1h"] = np.nan
        ctx["continuation_ema_slow_1h"] = np.nan
        ctx["continuation_trend_spread_1h_atr"] = np.nan
        ctx["continuation_close_vs_slow_1h_atr"] = np.nan
        ctx["continuation_trend_up_1h"] = 0
        ctx["continuation_trend_down_1h"] = 0

    return ctx


def filter_continuation_candidates(
    df_5m: pd.DataFrame,
    swings: List[SwingPoint],
    structural_atr: pd.Series,
    trend_context: pd.DataFrame,
    params: ContinuationPullbackParams,
) -> List[SwingPoint]:
    """Keep only swings that look like trend pullbacks rather than reversals."""

    filtered: List[SwingPoint] = []
    prev_low: SwingPoint | None = None
    prev_high: SwingPoint | None = None
    closes = df_5m["close"].values
    atr_vals = structural_atr.values

    for sp in sorted(swings, key=lambda item: item.swing_index):
        atr_at = _safe_atr(atr_vals[sp.swing_index], sp.atr_at_swing)
        if atr_at <= 0:
            if sp.swing_type == "low":
                prev_low = sp
            else:
                prev_high = sp
            continue

        if sp.swing_type == "low":
            trend_ok = bool(trend_context["continuation_trend_up_30m"].iat[sp.swing_index])
            if params.require_1h_trend_alignment:
                trend_ok = trend_ok and bool(trend_context["continuation_trend_up_1h"].iat[sp.swing_index])
            anchor = prev_high
            same_side = prev_low
            if trend_ok and anchor is not None:
                pullback_depth_atr = (anchor.swing_price - sp.swing_price) / atr_at
                pullback_bars = sp.swing_index - anchor.swing_index
                trend_floor_atr = (
                    sp.swing_price - trend_context["continuation_ema_slow_30m"].iat[sp.swing_index]
                ) / atr_at
                same_direction_break_atr = 0.0
                if same_side is not None:
                    same_direction_break_atr = max(0.0, (same_side.swing_price - sp.swing_price) / atr_at)
                bounce_at_confirm_atr = (closes[sp.confirm_index] - sp.swing_price) / atr_at

                if (
                    params.pullback_min_atr <= pullback_depth_atr <= params.pullback_max_atr
                    and pullback_bars <= params.max_pullback_bars
                    and trend_floor_atr >= -params.trend_floor_tolerance_atr
                    and same_direction_break_atr <= params.max_same_direction_break_atr
                    and bounce_at_confirm_atr >= params.min_bounce_at_confirm_atr
                ):
                    sp.trend_direction = "up"
                    sp.pullback_depth_atr = float(pullback_depth_atr)
                    sp.pullback_bars = int(pullback_bars)
                    sp.trend_spread_30m_atr = float(trend_context["continuation_trend_spread_30m_atr"].iat[sp.swing_index])
                    sp.trend_spread_1h_atr = float(trend_context["continuation_trend_spread_1h_atr"].iat[sp.swing_index])
                    sp.price_vs_trend_floor_atr = float(trend_floor_atr)
                    sp.same_direction_break_atr = float(same_direction_break_atr)
                    sp.resume_move_at_confirm_atr = float(bounce_at_confirm_atr)
                    filtered.append(sp)
        else:
            trend_ok = bool(trend_context["continuation_trend_down_30m"].iat[sp.swing_index])
            if params.require_1h_trend_alignment:
                trend_ok = trend_ok and bool(trend_context["continuation_trend_down_1h"].iat[sp.swing_index])
            anchor = prev_low
            same_side = prev_high
            if trend_ok and anchor is not None:
                pullback_depth_atr = (sp.swing_price - anchor.swing_price) / atr_at
                pullback_bars = sp.swing_index - anchor.swing_index
                trend_floor_atr = (
                    trend_context["continuation_ema_slow_30m"].iat[sp.swing_index] - sp.swing_price
                ) / atr_at
                same_direction_break_atr = 0.0
                if same_side is not None:
                    same_direction_break_atr = max(0.0, (sp.swing_price - same_side.swing_price) / atr_at)
                bounce_at_confirm_atr = (sp.swing_price - closes[sp.confirm_index]) / atr_at

                if (
                    params.pullback_min_atr <= pullback_depth_atr <= params.pullback_max_atr
                    and pullback_bars <= params.max_pullback_bars
                    and trend_floor_atr >= -params.trend_floor_tolerance_atr
                    and same_direction_break_atr <= params.max_same_direction_break_atr
                    and bounce_at_confirm_atr >= params.min_bounce_at_confirm_atr
                ):
                    sp.trend_direction = "down"
                    sp.pullback_depth_atr = float(pullback_depth_atr)
                    sp.pullback_bars = int(pullback_bars)
                    sp.trend_spread_30m_atr = float(trend_context["continuation_trend_spread_30m_atr"].iat[sp.swing_index])
                    sp.trend_spread_1h_atr = float(trend_context["continuation_trend_spread_1h_atr"].iat[sp.swing_index])
                    sp.price_vs_trend_floor_atr = float(trend_floor_atr)
                    sp.same_direction_break_atr = float(same_direction_break_atr)
                    sp.resume_move_at_confirm_atr = float(bounce_at_confirm_atr)
                    filtered.append(sp)

        if sp.swing_type == "low":
            prev_low = sp
        else:
            prev_high = sp

    return filtered


def plot_continuation_pullbacks(df, swings, ll, ls, start_date=None, end_date=None, save_path=None):
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
    for sp in swings:
        if sp.swing_index not in idx_set:
            continue
        if sp.tb_outcome == "sl":
            continue
        marker, color = ("^", "green") if sp.swing_type == "low" else ("v", "red")
        ax.plot(
            sp.swing_time,
            sp.swing_price,
            marker,
            color=color,
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
        )
        if sp.confirm_index in idx_set:
            confirm_color = "blue" if sp.swing_type == "low" else "orange"
            ax.plot(
                sp.confirm_time,
                cl[sp.confirm_index],
                "x",
                color=confirm_color,
                markersize=8,
                markeredgewidth=2,
                zorder=4,
            )

    ax.set_title("Continuation Pullback Labels: Trend Resumption Structures", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#333", lw=1, label="Close"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="green", ms=10, label="Long Pullback"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="red", ms=10, label="Short Pullback"),
            Line2D([0], [0], marker="x", color="blue", ms=8, mew=2, ls="None", label="Confirm (Long)"),
            Line2D([0], [0], marker="x", color="orange", ms=8, mew=2, ls="None", label="Confirm (Short)"),
        ],
        loc="upper left",
        fontsize=9,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def build_continuation_pullback_labels(
    df_5m,
    df_30m,
    params: ContinuationPullbackParams | None = None,
    df_1hr=None,
    verbose: bool = True,
):
    """Master pipeline for continuation-pullback labels."""

    raw_params = params or ContinuationPullbackParams()

    df_5m = _prep(df_5m)
    df_30m = _prep(df_30m)
    base_bar_minutes = infer_bar_minutes(df_5m.index)
    base_tf = format_bar_timeframe(base_bar_minutes)
    params = retune_bar_count_params(
        raw_params,
        base_bar_minutes,
        extra_count_fields=("max_pullback_bars",),
    )
    params.validate()
    anomaly_count = int(df_5m.attrs.get("anomaly_count", 0))
    n = len(df_5m)
    htf_warmup_bars = max(raw_params.warmup_bars, raw_params.atr_period * 2)
    if verbose:
        print(f"[Continuation] {n:,} bars on {base_tf}")
        if abs(base_bar_minutes - params.bar_count_reference_minutes) >= 1e-9:
            print(
                f"[Continuation] Runtime bar-count tuning: "
                f"{format_bar_timeframe(params.bar_count_reference_minutes)} -> {base_tf}"
            )

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
                print(f"[Continuation] Structural ATR (1hr->{base_tf}): {valid.mean():.6f} ({valid.mean() * 10000:.1f} pips)")
    else:
        satr_5m = map_htf_atr_to_ltf(atr_30m, df_5m.index)
        satr_30m = atr_30m
        if verbose:
            valid = satr_5m.dropna()
            if len(valid):
                print(f"[Continuation] Structural ATR (30m->{base_tf}): {valid.mean():.6f} ({valid.mean() * 10000:.1f} pips)")

    df_5m["atr"] = atr_5m
    df_5m["structural_atr"] = satr_5m

    trend_context = build_trend_context(df_5m, df_30m, atr_30m, params, df_1hr=df_1hr, atr_1hr=atr_1hr)
    for column in trend_context.columns:
        df_5m[column] = trend_context[column]

    raw_swings = detect_swings_zigzag(
        df_5m,
        satr_5m,
        params.confirm_atr_mult,
        params.min_swing_atr,
        params.min_swing_distance_atr,
        params.min_bars_between_swings,
        params.warmup_bars,
        params.confirm_use_close,
        base_tf,
    )
    if verbose:
        print(f"[Continuation] Raw {base_tf} swings: {len(raw_swings)}")

    sw30 = detect_swings_zigzag(
        df_30m,
        satr_30m,
        params.htf_confirm_atr_mult,
        params.htf_min_swing_atr,
        params.min_swing_distance_atr * 0.8,
        params.htf_min_bars_between,
        htf_warmup_bars,
        params.confirm_use_close,
        "30m",
    )
    if verbose:
        print(f"[Continuation] 30m swings: {len(sw30)}")

    sw1h: List[SwingPoint] = []
    if df_1hr is not None and atr_1hr is not None:
        sw1h = detect_swings_zigzag(
            df_1hr,
            atr_1hr,
            params.htf_confirm_atr_mult,
            params.htf_min_swing_atr * 0.8,
            params.min_swing_distance_atr * 0.7,
            max(2, params.htf_min_bars_between // 2),
            htf_warmup_bars,
            params.confirm_use_close,
            "1hr",
        )
        if verbose:
            print(f"[Continuation] 1hr swings: {len(sw1h)}")

    htf_match_30m = compute_htf_swing_match(raw_swings, sw30, params.htf_confluence_window_minutes)
    htf_match_1h = (
        compute_htf_swing_match(raw_swings, sw1h, params.htf_confluence_window_minutes * 2)
        if sw1h
        else {}
    )
    raw_swings = annotate_swing_context(
        raw_swings,
        sw30,
        sw1h,
        htf_match_30m,
        htf_match_1h,
        base_bar_minutes=base_bar_minutes,
    )

    cont_swings = filter_continuation_candidates(df_5m, raw_swings, satr_5m, trend_context, params)
    if verbose:
        print(f"[Continuation] Filtered continuation swings: {len(cont_swings)}")

    cont_swings = validate_swings_triple_barrier(df_5m, cont_swings, satr_5m, params)
    cont_swings = trend_scan_swings(df_5m, cont_swings, params)
    _, _ = assign_entry_labels(df_5m, cont_swings, satr_5m, params)
    cont_swings = compute_swing_quality(cont_swings, params)
    entry_long, entry_short = assign_entry_labels(df_5m, cont_swings, satr_5m, params)

    ll, ls = create_zone_labels(n, cont_swings, params)
    ql, qs, eql, eqs = create_quality_arrays(n, cont_swings, params)
    el, es, nl, ns = create_exclusion_masks(n, cont_swings, ll, ls, params)

    df_5m["label_long_continuation_pullback"] = ll
    df_5m["label_short_continuation_pullback"] = ls
    df_5m["label_long_continuation_entry"] = entry_long
    df_5m["label_short_continuation_entry"] = entry_short
    df_5m["label_quality_long_continuation"] = ql
    df_5m["label_quality_short_continuation"] = qs
    df_5m["entry_quality_long_continuation"] = eql
    df_5m["entry_quality_short_continuation"] = eqs
    df_5m["exclude_long_continuation"] = el
    df_5m["exclude_short_continuation"] = es
    df_5m["neg_ok_long_continuation"] = nl
    df_5m["neg_ok_short_continuation"] = ns

    hcl, hcs = compute_htf_confluence(cont_swings, sw30, n, df_5m.index, params.htf_confluence_window_minutes)
    df_5m["htf_confluence_long_continuation"] = hcl
    df_5m["htf_confluence_short_continuation"] = hcs

    if sw1h:
        h1l, h1s = compute_htf_confluence(
            cont_swings,
            sw1h,
            n,
            df_5m.index,
            params.htf_confluence_window_minutes * 2,
        )
        df_5m["htf_confluence_long_continuation_1hr"] = h1l
        df_5m["htf_confluence_short_continuation_1hr"] = h1s
        for column, values in compute_htf_timing_features(n, df_5m.index, sw1h, "1h").items():
            df_5m[column] = values

    for column, values in compute_htf_timing_features(n, df_5m.index, sw30, "30m").items():
        df_5m[column] = values

    if params.compute_uniqueness:
        cl, cs = compute_label_concurrency(n, cont_swings, params)
        df_5m["concurrency_long_continuation"] = cl
        df_5m["concurrency_short_continuation"] = cs
        df_5m["sample_weight_long_continuation"] = compute_sample_weights(cl, ll) * np.where(
            ll > 0,
            np.maximum(ql, params.min_label_quality),
            1.0,
        )
        df_5m["sample_weight_short_continuation"] = compute_sample_weights(cs, ls) * np.where(
            ls > 0,
            np.maximum(qs, params.min_label_quality),
            1.0,
        )
        df_5m["sample_weight_entry_long_continuation"] = compute_sample_weights(np.maximum(cl, 1), entry_long) * np.where(
            entry_long > 0,
            np.maximum(eql, params.min_label_quality),
            1.0,
        )
        df_5m["sample_weight_entry_short_continuation"] = compute_sample_weights(np.maximum(cs, 1), entry_short) * np.where(
            entry_short > 0,
            np.maximum(eqs, params.min_label_quality),
            1.0,
        )

    warmup_mask = np.zeros(n, dtype=bool)
    warmup_mask[: params.warmup_bars] = True
    df_5m["warmup_mask"] = warmup_mask

    diag = run_diagnostics(
        df_5m,
        cont_swings,
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
        {sp.swing_index: sp.htf_match_30m for sp in cont_swings},
        entry_long,
        entry_short,
        anomaly_count,
    )
    diag["raw_swings_5m"] = len(raw_swings)
    diag["continuation_candidate_pass_rate"] = len(cont_swings) / max(len(raw_swings), 1)
    if verbose:
        print_diagnostics(diag)

    return df_5m, diag, cont_swings
