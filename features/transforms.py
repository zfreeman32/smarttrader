from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    fill_value: float | None = np.nan,
) -> pd.Series:
    """Safe division helper that suppresses divide-by-zero blowups."""
    numerator_series = pd.Series(numerator)
    denominator_series = pd.Series(denominator).replace(0, np.nan)
    result = numerator_series / denominator_series
    if fill_value is not None:
        result = result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return result


def bars_since_event(event: pd.Series) -> pd.Series:
    """Return the number of bars since the last True event."""
    values = event.fillna(0).astype(bool).to_numpy()
    result = np.full(len(values), np.nan)
    last_seen = None

    for index, flag in enumerate(values):
        if flag:
            last_seen = index
            result[index] = 0
        elif last_seen is not None:
            result[index] = index - last_seen

    return pd.Series(result, index=event.index)


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """Standard true-range calculation used across feature families."""
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple rolling ATR."""
    return calculate_true_range(df).rolling(period).mean()


def detect_swing_highs(series: pd.Series, window: int = 3) -> pd.Series:
    """Detect raw swing highs using a symmetric window."""
    if window < 1:
        raise ValueError("window must be at least 1")

    left_max = series.shift(1).rolling(window, min_periods=window).max()
    right_max = series.iloc[::-1].shift(1).rolling(window, min_periods=window).max().iloc[::-1]
    return ((series > left_max) & (series > right_max)).fillna(False)


def detect_swing_lows(series: pd.Series, window: int = 3) -> pd.Series:
    """Detect raw swing lows using a symmetric window."""
    if window < 1:
        raise ValueError("window must be at least 1")

    left_min = series.shift(1).rolling(window, min_periods=window).min()
    right_min = series.iloc[::-1].shift(1).rolling(window, min_periods=window).min().iloc[::-1]
    return ((series < left_min) & (series < right_min)).fillna(False)


def confirm_event_series(
    event: pd.Series,
    values: pd.Series | None = None,
    *,
    delay: int = 1,
) -> tuple[pd.Series, pd.Series | None]:
    """Shift an event forward to the bar where it becomes knowable."""
    confirmed = event.fillna(False).astype(bool).shift(delay, fill_value=False)
    if values is None:
        return confirmed, None
    confirmed_values = values.where(event.fillna(False).astype(bool)).shift(delay)
    return confirmed, confirmed_values


def detect_confirmed_swings(
    high: pd.Series,
    low: pd.Series,
    window: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Detect swing highs/lows and return confirmed events plus confirmed levels."""
    raw_highs = detect_swing_highs(high, window=window)
    raw_lows = detect_swing_lows(low, window=window)
    confirmed_highs, confirmed_high_levels = confirm_event_series(raw_highs, high, delay=window)
    confirmed_lows, confirmed_low_levels = confirm_event_series(raw_lows, low, delay=window)
    return confirmed_highs, confirmed_high_levels, confirmed_lows, confirmed_low_levels


def fractional_diff_weights(
    d: float,
    *,
    threshold: float = 1e-3,
    max_size: int = 200,
) -> np.ndarray:
    """Build fixed-width fractional-differencing weights."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if max_size < 1:
        raise ValueError("max_size must be at least 1")

    weights = [1.0]
    for k in range(1, max_size):
        next_weight = -weights[-1] * (d - k + 1) / k
        if abs(next_weight) < threshold:
            break
        weights.append(float(next_weight))

    return np.array(list(reversed(weights)), dtype=float)


def apply_fractional_diff(
    series: pd.Series,
    *,
    d: float = 0.4,
    threshold: float = 1e-3,
    max_size: int = 200,
) -> pd.Series:
    """Apply fixed-width fractional differentiation to a price series."""
    weights = fractional_diff_weights(d, threshold=threshold, max_size=max_size)
    width = len(weights)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().shape[0] < width:
        return pd.Series(np.nan, index=series.index, dtype=float)

    filled = numeric.ffill()
    values = filled.fillna(0.0).to_numpy(dtype=float)
    transformed = np.convolve(values, weights, mode="valid")

    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.iloc[width - 1 :] = transformed
    out.loc[numeric.isna()] = np.nan
    return out


def rolling_winsorize_series(
    series: pd.Series,
    window: int,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """Winsorize a series using rolling historical quantiles."""
    numeric = pd.to_numeric(series, errors="coerce")
    history = numeric.shift(1)
    lower = history.rolling(window, min_periods=window).quantile(lower_quantile)
    upper = history.rolling(window, min_periods=window).quantile(upper_quantile)

    clipped = numeric.copy()
    lower_mask = lower.notna() & (numeric < lower)
    upper_mask = upper.notna() & (numeric > upper)
    clipped = clipped.where(~lower_mask, lower)
    clipped = clipped.where(~upper_mask, upper)
    return clipped


def rolling_sigma_normalize(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Normalize a series by rolling historical mean and sigma."""
    numeric = pd.to_numeric(series, errors="coerce")
    history = numeric.shift(1)
    mean = history.rolling(window, min_periods=window).mean()
    std = history.rolling(window, min_periods=window).std().replace(0, np.nan)
    return (numeric - mean) / std


def add_rolling_winsorized_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Add rolling historical winsorized versions of return-like features."""
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        out[f"{column}_winsor_{window}"] = rolling_winsorize_series(
            df[column],
            window=window,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
    return out


def _rolling_percentile_rank_last(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    last = valid[-1]
    return float(np.sum(valid <= last) / len(valid))


def add_rolling_percentile_rank_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int,
) -> pd.DataFrame:
    """Add rolling percentile ranks for selected regime-style features."""
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        rolling = pd.to_numeric(df[column], errors="coerce").rolling(window, min_periods=window)
        try:
            out[f"{column}_pct_rank_{window}"] = rolling.rank(pct=True)
        except AttributeError:
            out[f"{column}_pct_rank_{window}"] = rolling.apply(
                _rolling_percentile_rank_last,
                raw=True,
            )
    return out


def add_atr_normalized_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    atr_column: str = "atr_14",
) -> pd.DataFrame:
    """Normalize selected price-unit features by ATR."""
    out = pd.DataFrame(index=df.index)
    if atr_column in df.columns:
        atr = df[atr_column]
    elif {"high", "low", "close"}.issubset(df.columns):
        atr = calculate_atr(df)
    else:
        return out

    for column in columns:
        if column not in df.columns:
            continue
        out[f"{column}_atr_norm"] = safe_divide(df[column], atr)
    return out


def add_sigma_normalized_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int,
) -> pd.DataFrame:
    """Normalize selected features by rolling historical sigma."""
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        out[f"{column}_sigma_norm_{window}"] = rolling_sigma_normalize(df[column], window)
    return out


def add_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    periods: Sequence[int],
) -> pd.DataFrame:
    """Create lag features for the selected columns."""
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        for period in periods:
            out[f"{column}_lag_{period}"] = df[column].shift(period)
    return out


def add_rolling_statistics(
    df: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    """Add rolling means and standard deviations."""
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        for window in windows:
            out[f"{column}_roll_mean_{window}"] = df[column].rolling(window).mean()
            out[f"{column}_roll_std_{window}"] = df[column].rolling(window).std()
    return out


def add_rolling_zscores(
    df: pd.DataFrame,
    columns: Sequence[str],
    window: int,
) -> pd.DataFrame:
    """Rolling z-scores using past data only."""
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        mean = df[column].rolling(window).mean()
        std = df[column].rolling(window).std().replace(0, np.nan)
        out[f"{column}_zscore_{window}"] = (df[column] - mean) / std
    return out


def _combine_max(series_list: Sequence[pd.Series]) -> pd.Series | None:
    if not series_list:
        return None
    return pd.concat(series_list, axis=1).max(axis=1)


def _combine_mean(series_list: Sequence[pd.Series]) -> pd.Series | None:
    if not series_list:
        return None
    return pd.concat(series_list, axis=1).mean(axis=1)


def proximity_score(
    distance: pd.Series | np.ndarray,
    threshold: float = 2.0,
) -> pd.Series:
    """Convert a distance feature into a bounded proximity score in [0, 1]."""
    distance_series = pd.to_numeric(pd.Series(distance), errors="coerce")
    bounded = distance_series.clip(lower=0, upper=threshold)
    return (1.0 - (bounded / threshold)).clip(lower=0).fillna(0.0)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Curated interaction features inspired by the pipeline docs."""
    out = pd.DataFrame(index=df.index)

    if {"ema_alignment", "roc_5"}.issubset(df.columns):
        out["interaction_trend_momentum"] = df["ema_alignment"] * df["roc_5"]

    if {"ema_alignment", "rsi_from_mid"}.issubset(df.columns):
        out["interaction_trend_rsi"] = df["ema_alignment"] * df["rsi_from_mid"]

    if {"roc_5", "atr_ratio_14_50"}.issubset(df.columns):
        out["interaction_vol_adjusted_momentum"] = safe_divide(
            df["roc_5"],
            df["atr_ratio_14_50"],
        )

    if {"in_london_session", "roc_5"}.issubset(df.columns):
        out["interaction_london_momentum"] = df["in_london_session"] * df["roc_5"]

    if {"in_newyork_session", "roc_5"}.issubset(df.columns):
        out["interaction_newyork_momentum"] = df["in_newyork_session"] * df["roc_5"]

    if {"sweep_low_20", "in_discount_zone", "ema_alignment"}.issubset(df.columns):
        out["interaction_bull_reversal_context"] = (
            df["sweep_low_20"]
            * (df["in_discount_zone"] + (df["ema_alignment"] > 0).astype(int))
        )

    if {"sweep_high_20", "in_premium_zone", "ema_alignment"}.issubset(df.columns):
        out["interaction_bear_reversal_context"] = (
            df["sweep_high_20"]
            * (df["in_premium_zone"] + (df["ema_alignment"] < 0).astype(int))
        )

    if {"volume_imbalance_10", "close_return_1"}.issubset(df.columns):
        out["interaction_orderflow_return"] = df["volume_imbalance_10"] * df["close_return_1"]

    htf_trend_score = None
    if "htf_alignment_score" in df.columns:
        htf_trend_score = df["htf_alignment_score"]
    elif {"htf_30m_ema_alignment", "htf_1h_ema_alignment"}.issubset(df.columns):
        htf_trend_score = df["htf_30m_ema_alignment"] + df["htf_1h_ema_alignment"]
    elif "htf_1h_ema_alignment" in df.columns:
        htf_trend_score = df["htf_1h_ema_alignment"]

    if htf_trend_score is not None and "roc_5" in df.columns:
        out["interaction_htf_trend_momentum"] = htf_trend_score * df["roc_5"]

    if htf_trend_score is not None and "rsi_from_mid" in df.columns:
        out["interaction_htf_trend_rsi"] = htf_trend_score * df["rsi_from_mid"]

    if htf_trend_score is not None and "ema_alignment" in df.columns:
        out["interaction_htf_ltf_alignment"] = htf_trend_score * df["ema_alignment"]

    bull_proximity_parts = [
        proximity_score(df[column])
        for column in (
            "dist_to_bull_fvg_atr",
            "dist_to_bull_order_block_atr",
            "dist_to_equal_low_pool_atr",
            "ict_dist_to_swing_low_atr",
            "htf_30m_dist_to_swing_low_atr",
            "htf_1h_dist_to_swing_low_atr",
            "htf_dist_to_prev_day_low_atr",
            "htf_dist_to_rolling_daily_low_atr",
        )
        if column in df.columns
    ]
    bear_proximity_parts = [
        proximity_score(df[column])
        for column in (
            "dist_to_bear_fvg_atr",
            "dist_to_bear_order_block_atr",
            "dist_to_equal_high_pool_atr",
            "ict_dist_to_swing_high_atr",
            "htf_30m_dist_to_swing_high_atr",
            "htf_1h_dist_to_swing_high_atr",
            "htf_dist_to_prev_day_high_atr",
            "htf_dist_to_rolling_daily_high_atr",
        )
        if column in df.columns
    ]
    bull_structure_proximity = _combine_max(bull_proximity_parts)
    bear_structure_proximity = _combine_max(bear_proximity_parts)

    if bull_structure_proximity is not None:
        out["interaction_bull_structure_proximity"] = bull_structure_proximity
    if bear_structure_proximity is not None:
        out["interaction_bear_structure_proximity"] = bear_structure_proximity

    if bull_structure_proximity is not None and "bullish_rejection" in df.columns:
        out["interaction_bull_rejection_structure"] = bull_structure_proximity * df["bullish_rejection"]
    if bear_structure_proximity is not None and "bearish_rejection" in df.columns:
        out["interaction_bear_rejection_structure"] = bear_structure_proximity * df["bearish_rejection"]

    if bull_structure_proximity is not None and "sweep_low_20" in df.columns:
        out["interaction_bull_sweep_structure"] = bull_structure_proximity * df["sweep_low_20"]
    if bear_structure_proximity is not None and "sweep_high_20" in df.columns:
        out["interaction_bear_sweep_structure"] = bear_structure_proximity * df["sweep_high_20"]

    if bull_structure_proximity is not None and "in_discount_zone" in df.columns:
        out["interaction_bull_discount_structure"] = bull_structure_proximity * df["in_discount_zone"]
    if bear_structure_proximity is not None and "in_premium_zone" in df.columns:
        out["interaction_bear_premium_structure"] = bear_structure_proximity * df["in_premium_zone"]

    bull_reversion_strength = _combine_mean(
        [
            df[column]
            for column in ("bullish_divergence_strength", "bull_volume_exhaustion")
            if column in df.columns
        ]
        + ([df["failed_breakout_bear"].astype(float)] if "failed_breakout_bear" in df.columns else [])
        + ([bull_structure_proximity] if bull_structure_proximity is not None else [])
    )
    bear_reversion_strength = _combine_mean(
        [
            df[column]
            for column in ("bearish_divergence_strength", "bear_volume_exhaustion")
            if column in df.columns
        ]
        + ([df["failed_breakout_bull"].astype(float)] if "failed_breakout_bull" in df.columns else [])
        + ([bear_structure_proximity] if bear_structure_proximity is not None else [])
    )

    if "compression_regime" in df.columns and bull_reversion_strength is not None:
        out["interaction_compression_bull_reversion"] = df["compression_regime"] * bull_reversion_strength
    if "compression_regime" in df.columns and bear_reversion_strength is not None:
        out["interaction_compression_bear_reversion"] = df["compression_regime"] * bear_reversion_strength

    if "rolling_vol_20_pct_rank_60" in df.columns and bull_reversion_strength is not None:
        out["interaction_lowvol_bull_reversion"] = (1.0 - df["rolling_vol_20_pct_rank_60"]) * bull_reversion_strength
    if "rolling_vol_20_pct_rank_60" in df.columns and bear_reversion_strength is not None:
        out["interaction_lowvol_bear_reversion"] = (1.0 - df["rolling_vol_20_pct_rank_60"]) * bear_reversion_strength

    bull_setup_score = _combine_mean(
        ([bull_structure_proximity] if bull_structure_proximity is not None else [])
        + ([df["bullish_rejection"].astype(float)] if "bullish_rejection" in df.columns else [])
        + ([df["bull_sequence_sweep_displacement_retrace"].astype(float)] if "bull_sequence_sweep_displacement_retrace" in df.columns else [])
        + ([safe_divide(df["ict_bull_confluence_1atr"], 3.0, fill_value=0.0)] if "ict_bull_confluence_1atr" in df.columns else [])
        + ([((htf_trend_score > 0).astype(float))] if htf_trend_score is not None else [])
    )
    bear_setup_score = _combine_mean(
        ([bear_structure_proximity] if bear_structure_proximity is not None else [])
        + ([df["bearish_rejection"].astype(float)] if "bearish_rejection" in df.columns else [])
        + ([df["bear_sequence_sweep_displacement_retrace"].astype(float)] if "bear_sequence_sweep_displacement_retrace" in df.columns else [])
        + ([safe_divide(df["ict_bear_confluence_1atr"], 3.0, fill_value=0.0)] if "ict_bear_confluence_1atr" in df.columns else [])
        + ([((htf_trend_score < 0).astype(float))] if htf_trend_score is not None else [])
    )

    if bull_setup_score is not None and "in_london_killzone" in df.columns:
        out["interaction_london_bull_setup"] = df["in_london_killzone"] * bull_setup_score
    if bear_setup_score is not None and "in_london_killzone" in df.columns:
        out["interaction_london_bear_setup"] = df["in_london_killzone"] * bear_setup_score
    if bull_setup_score is not None and "in_newyork_killzone" in df.columns:
        out["interaction_newyork_bull_setup"] = df["in_newyork_killzone"] * bull_setup_score
    if bear_setup_score is not None and "in_newyork_killzone" in df.columns:
        out["interaction_newyork_bear_setup"] = df["in_newyork_killzone"] * bear_setup_score
    if bull_setup_score is not None and "in_london_ny_overlap" in df.columns:
        out["interaction_overlap_bull_setup"] = df["in_london_ny_overlap"] * bull_setup_score
    if bear_setup_score is not None and "in_london_ny_overlap" in df.columns:
        out["interaction_overlap_bear_setup"] = df["in_london_ny_overlap"] * bear_setup_score

    return out
