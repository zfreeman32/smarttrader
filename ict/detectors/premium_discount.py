from __future__ import annotations

import numpy as np
import pandas as pd

from features.transforms import safe_divide

from ..common import cfg_value, get_atr_like


def _candidate_level_frame(
    index: pd.Index,
    columns: tuple[str, ...],
    *frames: pd.DataFrame | None,
) -> pd.DataFrame:
    candidates: list[pd.Series] = []
    seen: set[str] = set()

    for column in columns:
        if column in seen:
            continue
        for frame in frames:
            if frame is None or frame.empty or column not in frame.columns:
                continue
            candidates.append(pd.to_numeric(frame[column], errors="coerce").rename(column))
            seen.add(column)
            break

    if not candidates:
        return pd.DataFrame(index=index)
    return pd.concat(candidates, axis=1)


def detect_ict_premium_discount(
    df: pd.DataFrame,
    config: object,
    *,
    swing_features: pd.DataFrame | None = None,
    liquidity_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build premium/discount, OTE, and DOL context from causal anchors."""

    out = pd.DataFrame(index=df.index)
    if swing_features is None or swing_features.empty or "close" not in df.columns:
        return out

    close = pd.to_numeric(df["close"], errors="coerce")
    atr = get_atr_like(df)
    latest_high = pd.to_numeric(swing_features.get("ict_latest_swing_high"), errors="coerce")
    latest_low = pd.to_numeric(swing_features.get("ict_latest_swing_low"), errors="coerce")
    latest_high_idx = pd.to_numeric(swing_features.get("ict_latest_swing_high_index"), errors="coerce")
    latest_low_idx = pd.to_numeric(swing_features.get("ict_latest_swing_low_index"), errors="coerce")

    range_low = pd.concat([latest_high, latest_low], axis=1).min(axis=1)
    range_high = pd.concat([latest_high, latest_low], axis=1).max(axis=1)
    range_size = range_high - range_low

    percentile = safe_divide(close - range_low, range_size, fill_value=np.nan)
    out["ict_dealing_range_low"] = range_low
    out["ict_dealing_range_high"] = range_high
    out["ict_price_percentile_in_range"] = percentile
    out["ict_discount_zone"] = percentile.lt(0.5).astype(np.int8)
    out["ict_premium_zone"] = percentile.gt(0.5).astype(np.int8)
    out["ict_equilibrium_distance"] = percentile - 0.5

    ote_lower = float(cfg_value(config, "ict_ote_lower", default=0.62))
    ote_upper = float(cfg_value(config, "ict_ote_upper", default=0.79))
    ote_mid = float(cfg_value(config, "ict_ote_mid", default=0.705))

    bullish_impulse = latest_high_idx.gt(latest_low_idx)
    bearish_impulse = latest_low_idx.gt(latest_high_idx)
    bullish_retrace = safe_divide(latest_high - close, range_size, fill_value=np.nan)
    bearish_retrace = safe_divide(close - latest_low, range_size, fill_value=np.nan)
    retracement = bullish_retrace.where(bullish_impulse, bearish_retrace.where(bearish_impulse))

    out["ict_impulse_direction"] = np.where(bullish_impulse, 1, np.where(bearish_impulse, -1, 0))
    out["ict_ote_retracement"] = retracement
    out["ict_in_ote_band"] = retracement.between(ote_lower, ote_upper, inclusive="both").astype(np.int8)
    out["ict_dist_to_ote_mid"] = (retracement - ote_mid).abs()
    out["ict_ote_bucket_code"] = np.where(
        retracement < ote_lower,
        -1,
        np.where(retracement > ote_upper, 1, 0),
    )

    if liquidity_features is None or liquidity_features.empty:
        return out

    candidate_up = _candidate_level_frame(
        df.index,
        (
            "ict_prior_rth_high",
            "ict_overnight_high",
            "ict_ib_high",
            "ict_prior_week_high",
            "ict_equal_high_pool_level",
            "ict_latest_swing_high",
        ),
        liquidity_features,
        swing_features,
    )

    candidate_down = _candidate_level_frame(
        df.index,
        (
            "ict_prior_rth_low",
            "ict_overnight_low",
            "ict_ib_low",
            "ict_prior_week_low",
            "ict_equal_low_pool_level",
            "ict_latest_swing_low",
        ),
        liquidity_features,
        swing_features,
    )

    nearest_up = (
        candidate_up.where(candidate_up.gt(close, axis=0)).min(axis=1)
        if not candidate_up.empty
        else pd.Series(np.nan, index=df.index)
    )
    nearest_down = (
        candidate_down.where(candidate_down.lt(close, axis=0)).max(axis=1)
        if not candidate_down.empty
        else pd.Series(np.nan, index=df.index)
    )
    out["ict_dol_level_up"] = nearest_up
    out["ict_dol_level_down"] = nearest_down
    out["ict_dol_distance_up_atr"] = safe_divide(nearest_up - close, atr)
    out["ict_dol_distance_down_atr"] = safe_divide(close - nearest_down, atr)
    return out
