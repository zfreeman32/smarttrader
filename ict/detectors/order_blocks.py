from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common import cfg_value, get_atr_like, get_tick_size


@dataclass
class _OrderBlock:
    block_id: int
    direction: int
    lower: float
    upper: float
    body_lower: float
    body_upper: float
    formed_index: int
    displacement_score: float
    retest_count: int = 0
    invalidated: bool = False


def _distance_to_zone(price: float, lower: float, upper: float) -> float:
    if np.isnan(lower) or np.isnan(upper):
        return np.nan
    if lower <= price <= upper:
        return 0.0
    if price < lower:
        return lower - price
    return price - upper


def detect_ict_order_blocks(
    df: pd.DataFrame,
    config: object,
    *,
    displacement_features: pd.DataFrame | None = None,
    structure_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect order blocks confirmed by displacement and structure breaks."""

    out = pd.DataFrame(index=df.index)
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return out

    max_age = int(cfg_value(config, "ict_order_block_max_age", default=120))
    use_wicks = bool(cfg_value(config, "ict_order_block_use_wicks", default=False))
    tick_size = get_tick_size(config)
    atr = get_atr_like(df)
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")

    displacement_bull = (
        pd.to_numeric(displacement_features.get("displacement_bullish"), errors="coerce").fillna(0).astype(bool).to_numpy()
        if displacement_features is not None and "displacement_bullish" in displacement_features.columns
        else np.zeros(len(df), dtype=bool)
    )
    displacement_bear = (
        pd.to_numeric(displacement_features.get("displacement_bearish"), errors="coerce").fillna(0).astype(bool).to_numpy()
        if displacement_features is not None and "displacement_bearish" in displacement_features.columns
        else np.zeros(len(df), dtype=bool)
    )
    displacement_score = (
        pd.to_numeric(displacement_features.get("ict_displacement_score"), errors="coerce").fillna(0.0).to_numpy()
        if displacement_features is not None and "ict_displacement_score" in displacement_features.columns
        else np.zeros(len(df), dtype=float)
    )
    bull_break = (
        (
            pd.to_numeric(structure_features.get("ict_bos_bull"), errors="coerce").fillna(0).astype(bool)
            | pd.to_numeric(structure_features.get("ict_choch_bull"), errors="coerce").fillna(0).astype(bool)
        ).to_numpy()
        if structure_features is not None and not structure_features.empty
        else np.zeros(len(df), dtype=bool)
    )
    bear_break = (
        (
            pd.to_numeric(structure_features.get("ict_bos_bear"), errors="coerce").fillna(0).astype(bool)
            | pd.to_numeric(structure_features.get("ict_choch_bear"), errors="coerce").fillna(0).astype(bool)
        ).to_numpy()
        if structure_features is not None and not structure_features.empty
        else np.zeros(len(df), dtype=bool)
    )

    bull_distance = np.full(len(df), np.nan)
    bear_distance = np.full(len(df), np.nan)
    bull_age = np.full(len(df), np.nan)
    bear_age = np.full(len(df), np.nan)
    bull_retests = np.full(len(df), np.nan)
    bear_retests = np.full(len(df), np.nan)
    active_bull = np.zeros(len(df), dtype=float)
    active_bear = np.zeros(len(df), dtype=float)
    bull_id = np.full(len(df), np.nan)
    bear_id = np.full(len(df), np.nan)
    bull_lower = np.full(len(df), np.nan)
    bull_upper = np.full(len(df), np.nan)
    bear_lower = np.full(len(df), np.nan)
    bear_upper = np.full(len(df), np.nan)
    bull_formed_index = np.full(len(df), np.nan)
    bear_formed_index = np.full(len(df), np.nan)
    bull_retest_event = np.zeros(len(df), dtype=np.int8)
    bear_retest_event = np.zeros(len(df), dtype=np.int8)

    bull_blocks: list[_OrderBlock] = []
    bear_blocks: list[_OrderBlock] = []
    records: list[dict[str, object]] = []
    block_id = 1

    closes = close.to_numpy(dtype=float, copy=False)
    opens = open_.to_numpy(dtype=float, copy=False)
    highs = high.to_numpy(dtype=float, copy=False)
    lows = low.to_numpy(dtype=float, copy=False)
    atr_values = atr.ffill().fillna(0.0).to_numpy(dtype=float, copy=False)

    for i in range(len(df)):
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]
        current_atr = atr_values[i] if atr_values[i] > 0 else 1.0
        retested_bull_ids: set[int] = set()
        retested_bear_ids: set[int] = set()

        for blocks, direction in ((bull_blocks, 1), (bear_blocks, -1)):
            for block in blocks:
                if block.invalidated:
                    continue
                if (i - block.formed_index) > max_age:
                    block.invalidated = True
                    continue
                if direction == 1 and current_close < block.lower - tick_size:
                    block.invalidated = True
                elif direction == -1 and current_close > block.upper + tick_size:
                    block.invalidated = True
                elif i > block.formed_index and current_high >= block.lower and current_low <= block.upper:
                    block.retest_count += 1
                    if direction == 1:
                        retested_bull_ids.add(block.block_id)
                    else:
                        retested_bear_ids.add(block.block_id)

        if i >= 1 and displacement_bull[i] and bull_break[i] and closes[i - 1] < opens[i - 1]:
            body_lower = min(opens[i - 1], closes[i - 1])
            body_upper = max(opens[i - 1], closes[i - 1])
            lower = lows[i - 1] if use_wicks else body_lower
            upper = highs[i - 1] if use_wicks else body_upper
            block = _OrderBlock(
                block_id=block_id,
                direction=1,
                lower=float(lower),
                upper=float(upper),
                body_lower=float(body_lower),
                body_upper=float(body_upper),
                formed_index=i,
                displacement_score=float(displacement_score[i]),
            )
            bull_blocks.append(block)
            records.append(
                {
                    "order_block_id": block.block_id,
                    "direction": "bullish",
                    "formed_index": i,
                    "lower": block.lower,
                    "upper": block.upper,
                    "body_lower": block.body_lower,
                    "body_upper": block.body_upper,
                    "displacement_score": block.displacement_score,
                }
            )
            block_id += 1

        if i >= 1 and displacement_bear[i] and bear_break[i] and closes[i - 1] > opens[i - 1]:
            body_lower = min(opens[i - 1], closes[i - 1])
            body_upper = max(opens[i - 1], closes[i - 1])
            lower = lows[i - 1] if use_wicks else body_lower
            upper = highs[i - 1] if use_wicks else body_upper
            block = _OrderBlock(
                block_id=block_id,
                direction=-1,
                lower=float(lower),
                upper=float(upper),
                body_lower=float(body_lower),
                body_upper=float(body_upper),
                formed_index=i,
                displacement_score=float(displacement_score[i]),
            )
            bear_blocks.append(block)
            records.append(
                {
                    "order_block_id": block.block_id,
                    "direction": "bearish",
                    "formed_index": i,
                    "lower": block.lower,
                    "upper": block.upper,
                    "body_lower": block.body_lower,
                    "body_upper": block.body_upper,
                    "displacement_score": block.displacement_score,
                }
            )
            block_id += 1

        live_bull = [block for block in bull_blocks if not block.invalidated]
        live_bear = [block for block in bear_blocks if not block.invalidated]
        active_bull[i] = len(live_bull)
        active_bear[i] = len(live_bear)
        if live_bull:
            nearest = min(live_bull, key=lambda block: _distance_to_zone(current_close, block.lower, block.upper))
            bull_distance[i] = _distance_to_zone(current_close, nearest.lower, nearest.upper) / current_atr
            bull_age[i] = i - nearest.formed_index
            bull_retests[i] = nearest.retest_count
            bull_id[i] = nearest.block_id
            bull_lower[i] = nearest.lower
            bull_upper[i] = nearest.upper
            bull_formed_index[i] = nearest.formed_index
            bull_retest_event[i] = int(nearest.block_id in retested_bull_ids)
        if live_bear:
            nearest = min(live_bear, key=lambda block: _distance_to_zone(current_close, block.lower, block.upper))
            bear_distance[i] = _distance_to_zone(current_close, nearest.lower, nearest.upper) / current_atr
            bear_age[i] = i - nearest.formed_index
            bear_retests[i] = nearest.retest_count
            bear_id[i] = nearest.block_id
            bear_lower[i] = nearest.lower
            bear_upper[i] = nearest.upper
            bear_formed_index[i] = nearest.formed_index
            bear_retest_event[i] = int(nearest.block_id in retested_bear_ids)

    out["dist_to_bull_order_block_atr"] = bull_distance
    out["dist_to_bear_order_block_atr"] = bear_distance
    out["bull_order_block_age_bars"] = bull_age
    out["bear_order_block_age_bars"] = bear_age
    out["ict_bull_order_block_retest_count"] = bull_retests
    out["ict_bear_order_block_retest_count"] = bear_retests
    out["ict_active_bull_order_block_count"] = active_bull
    out["ict_active_bear_order_block_count"] = active_bear
    out["ict_nearest_bull_order_block_id"] = pd.Series(bull_id, index=df.index, dtype="Int64")
    out["ict_nearest_bear_order_block_id"] = pd.Series(bear_id, index=df.index, dtype="Int64")
    out["ict_nearest_bull_order_block_lower"] = bull_lower
    out["ict_nearest_bull_order_block_upper"] = bull_upper
    out["ict_nearest_bear_order_block_lower"] = bear_lower
    out["ict_nearest_bear_order_block_upper"] = bear_upper
    out["ict_nearest_bull_order_block_formed_index"] = pd.Series(bull_formed_index, index=df.index, dtype="Int64")
    out["ict_nearest_bear_order_block_formed_index"] = pd.Series(bear_formed_index, index=df.index, dtype="Int64")
    out["ict_bull_order_block_retest_event"] = bull_retest_event
    out["ict_bear_order_block_retest_event"] = bear_retest_event
    out.attrs["order_blocks"] = pd.DataFrame(records)
    return out
