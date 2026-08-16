from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureBuilderConfig
from ict.detectors import (
    detect_ict_displacement,
    detect_ict_fvg,
    detect_ict_market_structure,
    detect_ict_order_blocks,
    detect_ict_premium_discount,
    detect_ict_sweeps,
)
from ict.structure import build_reference_level_features, detect_ict_swings


def _merge_feature_frames(index: pd.Index, *frames: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for frame in frames:
        if frame is None or frame.empty:
            continue
        fresh_columns = [column for column in frame.columns if column not in out.columns]
        if fresh_columns:
            out = pd.concat([out, frame.loc[:, fresh_columns]], axis=1)
    return out


def _near_one_atr(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").le(1.0).fillna(False)


def build_ict_context_features(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """Compose the Phase 2 ICT structural detector stack."""

    swings = detect_ict_swings(df, config)
    liquidity = build_reference_level_features(df, config)
    sweeps = detect_ict_sweeps(df, config, swing_features=swings, liquidity_features=liquidity)
    displacement = detect_ict_displacement(df, config, liquidity_features=liquidity)
    structure = detect_ict_market_structure(df, config, swing_features=swings, sweep_features=sweeps)
    fvg = detect_ict_fvg(df, config, displacement_features=displacement)
    order_blocks = detect_ict_order_blocks(
        df,
        config,
        displacement_features=displacement,
        structure_features=structure,
    )
    premium_discount = detect_ict_premium_discount(
        df,
        config,
        swing_features=swings,
        liquidity_features=_merge_feature_frames(df.index, swings, liquidity),
    )

    out = _merge_feature_frames(
        df.index,
        swings,
        liquidity,
        sweeps,
        displacement,
        structure,
        fvg,
        order_blocks,
        premium_discount,
    )

    bull_confluence = (
        _near_one_atr(out, "dist_to_bull_fvg_atr").astype(np.int8)
        + _near_one_atr(out, "dist_to_bull_order_block_atr").astype(np.int8)
        + _near_one_atr(out, "dist_to_equal_low_pool_atr").astype(np.int8)
    )
    bear_confluence = (
        _near_one_atr(out, "dist_to_bear_fvg_atr").astype(np.int8)
        + _near_one_atr(out, "dist_to_bear_order_block_atr").astype(np.int8)
        + _near_one_atr(out, "dist_to_equal_high_pool_atr").astype(np.int8)
    )
    confluence = pd.DataFrame(
        {
            "ict_bull_confluence_1atr": bull_confluence,
            "ict_bear_confluence_1atr": bear_confluence,
            "ict_total_confluence_1atr": bull_confluence + bear_confluence,
            "ict_zone_balance_1atr": bull_confluence - bear_confluence,
        },
        index=df.index,
    )
    out = pd.concat([out, confluence], axis=1)

    out.attrs["ict_component_frames"] = {
        "swings": swings,
        "liquidity": liquidity,
        "sweeps": sweeps,
        "displacement": displacement,
        "structure": structure,
        "fvg": fvg,
        "order_blocks": order_blocks,
        "premium_discount": premium_discount,
    }
    out.attrs["fvg_zones"] = fvg.attrs.get("fvg_zones", pd.DataFrame())
    out.attrs["sweep_events"] = sweeps.attrs.get("sweep_events", pd.DataFrame())
    out.attrs["order_blocks"] = order_blocks.attrs.get("order_blocks", pd.DataFrame())
    return out
