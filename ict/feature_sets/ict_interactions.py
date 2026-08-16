from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureBuilderConfig


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float) > 0.0


def build_ict_interaction_features(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """Lightweight ICT interaction terms for the Phase 1 scaffold."""

    out = pd.DataFrame(index=df.index)
    recent_sweep_window = max(1, int(config.swing_window))

    recent_sweep = _numeric_series(df, "ict_bars_since_sweep_any").le(recent_sweep_window).fillna(False)
    bull_fvg_near = _numeric_series(df, "dist_to_bull_fvg_atr").le(1.0).fillna(False)
    bear_fvg_near = _numeric_series(df, "dist_to_bear_fvg_atr").le(1.0).fillna(False)
    bull_ob_near = _numeric_series(df, "dist_to_bull_order_block_atr").le(1.0).fillna(False)
    bear_ob_near = _numeric_series(df, "dist_to_bear_order_block_atr").le(1.0).fillna(False)
    choch_bull = _bool_series(df, "ict_choch_bull")
    choch_bear = _bool_series(df, "ict_choch_bear")
    displacement_bull = _bool_series(df, "displacement_bullish")
    displacement_bear = _bool_series(df, "displacement_bearish")
    price_position = _numeric_series(df, "price_position_50")
    discount = price_position.le(0.5).fillna(False)
    premium = price_position.ge(0.5).fillna(False)

    out["ict_bull_sweep_plus_fvg"] = (recent_sweep & bull_fvg_near).astype(np.int8)
    out["ict_bear_sweep_plus_fvg"] = (recent_sweep & bear_fvg_near).astype(np.int8)
    out["ict_bull_sweep_plus_ob"] = (recent_sweep & bull_ob_near).astype(np.int8)
    out["ict_bear_sweep_plus_ob"] = (recent_sweep & bear_ob_near).astype(np.int8)
    out["ict_bull_choch_after_sweep"] = (recent_sweep & choch_bull).astype(np.int8)
    out["ict_bear_choch_after_sweep"] = (recent_sweep & choch_bear).astype(np.int8)
    out["ict_sweep_plus_choch"] = (recent_sweep & (choch_bull | choch_bear)).astype(np.int8)
    out["ict_displacement_after_sweep"] = (recent_sweep & (displacement_bull | displacement_bear)).astype(np.int8)
    out["ict_fvg_retrace_after_displacement"] = (
        (displacement_bull & bull_fvg_near) | (displacement_bear & bear_fvg_near)
    ).astype(np.int8)
    out["ict_bull_zone_in_discount"] = (discount & (bull_fvg_near | bull_ob_near)).astype(np.int8)
    out["ict_bear_zone_in_premium"] = (premium & (bear_fvg_near | bear_ob_near)).astype(np.int8)

    return out
