"""Feature version 2: causal, bounded-memory feature contract (retrain R1, approved 2026-10-05).

Version 1 (``FeatureBuilderConfig.feature_version == 1``, the default) is byte-identical to the frozen B2 builds.
Version 2 changes, all gated on ``feature_version >= 2``:

* Bounded memory (anchored warm-start, approved Q-R1): each producer in ``V2_BOUNDED_MEMORY_FEATURE_SETS`` is computed
  per weekly block. A block holds the bars from one anchor (Saturday 12:00 America/New_York, inside the weekend closure
  of both ES and EURUSD, so the first bar after it is the weekly open) to the next. Its state restarts
  ``bounded_memory_warmup_bars`` (default 2,000) bars before the anchor. A value at t therefore depends only on bars
  from (latest anchor <= t) - W to t: at most about one week + W, well inside the app's 8,000-bar runtime window.
* Positional ICT outputs (row positions in the frame) gain translation-invariant ``*_bars_since`` forms. The positional and
  identity columns stay in the frame for the setup detectors and dashboards, but are excluded as predictors.
* Non-causal producers: ``ict_raw_swing_*`` and other look-ahead columns are excluded as predictors (the causal
  ``ict_confirmed_swing_*`` forms remain). The FRVP swing-to-swing window uses consistent datetime units. The selected
  look-ahead strategies are replaced by causal v2 modules; random-data and other look-ahead strategies are not built.
* Rolling mean/std (rolling statistics, z-scores, sigma normalization) are computed exactly from each window's own
  values. pandas' online rolling variance carries error proportional to the scale of all earlier history (after values of
  1e6, a constant window can report std ~0.03 instead of 0), so the v1 values depend on how much history preceded them.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

FEATURE_VERSION_LATEST = 2
V2_BOUNDED_MEMORY_FEATURE_SETS = ("ict_context", "frvp_context", "temporal_context", "strategy_signals")
V2_ANCHOR_TIMEZONE = "America/New_York"
V2_ANCHOR_WEEKDAY = 5  # Saturday
V2_ANCHOR_HOUR = 12
V2_DEFAULT_WARMUP_BARS = 2000

# Selected look-ahead strategies replaced by causal modules (new ids, so v2 columns carry new names).
V2_STRATEGY_REPLACEMENTS = {
    "high_volume_points": "high_volume_points_causal_v2",
    "ichimoku_signals__ichimoku_strategy_strategy": "ichimoku_strategy_causal_v2",
    "tesla_stock_boom_signals": "tesla_stock_boom_causal_v2",
}
# Not built in v2: random-number data (non-deterministic), static look-ahead (negative shifts, centered windows,
# reversed scans) and dynamic look-ahead found by the R0 causal-prefix audit. None is selected by a roster model.
V2_DROPPED_STRATEGIES = frozenset({
    # random data
    "inflation_trading_signals", "market_sentiment_signals__the_friday_seasonality_in_uso_oil_strategy",
    "membership_plan_signals__india_trading_strategy_strategy",
    "membership_plans_signals__a_tradeable_pattern_in_gold_miners_gdx_strategy",
    "membership_plans_signals__macro_index_spread_trading_strategy_strategy", "random_walk_signals",
    # static look-ahead
    "fractal_chaos_bands_signals", "gap_up_le_signals", "golden_triangle_signals", "momentumle_signals",
    "oversold_market_signals", "price_action_signals", "profit_target_lx_signals", "three_bar_inside_bar_le",
    "three_day_weekend_signals", "three_days_down_strategy",
    # dynamic look-ahead (R0 audit)
    "atr_trading_signals", "dma_signals", "end_of_month_effect_signals",
    "membership_plans_signals__unlocking_the_power_of_trading_days_analyzing_sp_500_gold_and_treasuries_strategy",
    # dynamic look-ahead (R1 v2 acceptance audit, EURUSD 2026-03-10): whole-frame dependence (e.g. `.iloc[-1]`)
    "adaptive_cyber_cycle_signals", "eurcad_trading_signals", "warren_buffett_etf_signals",
    # whole-frame mean/std inside the block (R1 EURUSD acceptance re-run, 2026-10-06)
    "anomaly_detection_signals", "membership_plans_signals__big_moves_trading_strategy_strategy",
    # R2a EURUSD acceptance (2026-10-06): whole-block reductions see the rest of the anchor week (rocwb RMS over the
    # block flipped 3 bars at 2026-03-10 with 983 later bars in the tape's block); the rest found by a source scan.
    "rocwb_signals", "volume_flow_signals", "london_breakout_signals",
    "end_of_month_effect_signals__end_of_month_effect_in_nifty_50_does_it_exist_turn_of_the_month_strategy",
    # monthly-sparse values carried across blocks by the numeric forward fill (window-dependent)
    "dollar_cost_averaging_signals",
})

# Strategy one-hot columns exist only for categories seen in the frame, so a short window can lack a column the tape has
# (exactly 0 there). v2 configs freeze the tape's one-hot vocabulary (``v2_strategy_one_hot_columns``); a listed column
# absent from a build is 0 when a sibling of the same strategy output exists (that output was built, the category not
# seen). Without a sibling the strategy did not build, and the column stays missing.


def strategy_one_hot_prefix(column: str) -> str | None:
    """``strategy__<id>__<output>__`` for a one-hot strategy column (four ``__`` parts), else None."""
    parts = column.split("__")
    return "__".join(parts[:3]) + "__" if len(parts) == 4 and parts[0] == "strategy" else None


def add_absent_strategy_one_hots(frame: pd.DataFrame, vocabulary) -> pd.DataFrame:
    if not vocabulary:
        return frame
    present = {strategy_one_hot_prefix(c) for c in frame.columns} - {None}
    absent = [c for c in vocabulary if c not in frame.columns and strategy_one_hot_prefix(c) in present]
    if not absent:
        return frame
    return pd.concat([frame, pd.DataFrame(0, index=frame.index, columns=absent)], axis=1)

# strategy_signals is blocked as a whole. A partial "memory strategy" list cannot be complete: pandas rolling means carry
# float residue from earlier history, so near-tie comparisons (e.g. short MA vs long MA) flip with the window start
# (25 strategies showed it at just 2 EURUSD end bars, most of them plain rolling-mean crossovers). Blocking makes every
# strategy window-invariant. Performance: run strategies in-process (strategy_timeout_seconds=None). The 600 s timeout of
# the frozen EURUSD config runs each strategy in its own subprocess (8,000 bars: 417 s unblocked vs 14.6 s in-process;
# 40.4 s blocked in-process).

_POSITIONAL = re.compile(r"^ict_.*_index$")
_IDENTITY = re.compile(r"^ict_.*_id$")
_EXCLUDED_BASES = (
    "ict_raw_swing_high", "ict_raw_swing_low",  # centered pivot (look-ahead)
    "ict_failed_buy_side_sweep", "ict_failed_sell_side_sweep",  # R0 causal-prefix audit
    "frvp_open_drive_flag",  # R0 causal-prefix audit
)


def is_v2(config) -> bool:
    return int(getattr(config, "feature_version", 1) or 1) >= 2


def v2_strategy_ids(strategy_ids) -> list[str]:
    """Requested strategy ids under v2: replacements applied, dropped ids removed, order kept, duplicates removed."""
    out: list[str] = []
    for strategy_id in strategy_ids:
        if strategy_id in V2_DROPPED_STRATEGIES:
            continue
        mapped = V2_STRATEGY_REPLACEMENTS.get(strategy_id, strategy_id)
        if mapped not in out:
            out.append(mapped)
    return out


def bars_since_name(column: str) -> str:
    return column[: -len("_index")] + "_bars_since"


def add_bars_since_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Translation-invariant forms of the positional ICT outputs: position in the frame minus the event position."""
    position = np.arange(len(frame), dtype=float)
    additions = {}
    for column in frame.columns:
        if _POSITIONAL.match(column):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            additions[bars_since_name(column)] = position - values
    if not additions:
        return frame
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)


def v2_predictor_excluded(column: str) -> bool:
    """True for v2 columns that must never be selected as model inputs (positional, identity, look-ahead), including
    any transform derived from them (lags, rolling stats, z-scores, sigma norms, winsorized and interaction forms)."""
    if _POSITIONAL.match(column) or _IDENTITY.match(column):
        return True
    for base in _EXCLUDED_BASES:
        if column == base or column.startswith(base + "_"):
            return True
    base = re.sub(r"_(lag_\d+|roll_(mean|std)_\d+|zscore_\d+|sigma_norm_\d+|winsor_\d+|pct_rank_\d+)$", "", column)
    return base != column and v2_predictor_excluded(base)


def weekly_block_ids(datetimes: pd.Series) -> np.ndarray:
    """Block key per row: the date of the latest Saturday 12:00 America/New_York anchor at or before the row."""
    # Local wall clock, then naive: calendar-day arithmetic must not cross DST transitions as fixed 24-hour steps.
    local = pd.to_datetime(datetimes, utc=True).dt.tz_convert(V2_ANCHOR_TIMEZONE).dt.tz_localize(None)
    shifted = local - pd.Timedelta(hours=V2_ANCHOR_HOUR)
    days = shifted.dt.normalize()
    back = (shifted.dt.weekday - V2_ANCHOR_WEEKDAY) % 7
    anchor_days = days - pd.to_timedelta(back, unit="D")
    return anchor_days.dt.strftime("%Y-%m-%d").to_numpy()


def exact_rolling_mean_std(series: pd.Series, window: int, *, chunk_rows: int = 20000) -> tuple[pd.Series, pd.Series]:
    """Rolling mean and sample std (ddof=1) from each window's own values; NaN where the window is incomplete or holds a
    NaN (pandas ``rolling(window)`` with ``min_periods=window`` semantics). Depends only on the window, not on history."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = len(values)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    window = int(window)
    if window >= 1 and n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        for low in range(0, len(windows), chunk_rows):
            block = windows[low:low + chunk_rows]
            rows = slice(low + window - 1, low + window - 1 + len(block))
            # A constant window is exactly its value with std 0 (summation would leave ~1e-16 residue instead).
            constant = block.max(axis=1) == block.min(axis=1)
            mean[rows] = np.where(constant, block[:, 0], block.mean(axis=1))
            if window > 1:
                std[rows] = np.where(constant, 0.0, block.std(axis=1, ddof=1))
    return pd.Series(mean, index=series.index), pd.Series(std, index=series.index)
