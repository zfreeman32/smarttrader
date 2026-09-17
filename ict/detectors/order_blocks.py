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


@dataclass
class _BreakerBlock:
    breaker_id: int
    direction: int
    source_order_block_id: int
    source_direction: int
    lower: float
    upper: float
    body_lower: float
    body_upper: float
    source_formed_index: int
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


def _bool_feature(frame: pd.DataFrame | None, column: str, length: int) -> np.ndarray:
    if frame is not None and column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(bool).to_numpy()
    return np.zeros(length, dtype=bool)


def _recent_feature(frame: pd.DataFrame | None, column: str, length: int, maximum: int) -> np.ndarray:
    if frame is not None and column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").le(maximum).fillna(False).to_numpy(dtype=bool)
    return np.zeros(length, dtype=bool)


def detect_ict_order_blocks(
    df: pd.DataFrame,
    config: object,
    *,
    displacement_features: pd.DataFrame | None = None,
    structure_features: pd.DataFrame | None = None,
    sweep_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect order blocks plus causal failed-OB breaker state."""

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

    displacement_bull = _bool_feature(displacement_features, "displacement_bullish", len(df))
    displacement_bear = _bool_feature(displacement_features, "displacement_bearish", len(df))
    displacement_score = (
        pd.to_numeric(displacement_features.get("ict_displacement_score"), errors="coerce").fillna(0.0).to_numpy()
        if displacement_features is not None and "ict_displacement_score" in displacement_features.columns
        else np.zeros(len(df), dtype=float)
    )
    bos_bull = _bool_feature(structure_features, "ict_bos_bull", len(df))
    choch_bull = _bool_feature(structure_features, "ict_choch_bull", len(df))
    mss_bull = _bool_feature(structure_features, "ict_mss_bull", len(df))
    bos_bear = _bool_feature(structure_features, "ict_bos_bear", len(df))
    choch_bear = _bool_feature(structure_features, "ict_choch_bear", len(df))
    mss_bear = _bool_feature(structure_features, "ict_mss_bear", len(df))
    bull_break = bos_bull | choch_bull
    bear_break = bos_bear | choch_bear
    bull_shift = choch_bull | mss_bull
    bear_shift = choch_bear | mss_bear
    breaker_sweep_window = max(
        1,
        int(
            cfg_value(
                config,
                "breaker_sweep_window_bars",
                "ict_breaker_sweep_window_bars",
                default=5,
            )
        ),
    )
    recent_sell_sweep = _recent_feature(
        sweep_features,
        "ict_bars_since_sell_side_sweep",
        len(df),
        breaker_sweep_window,
    )
    recent_buy_sweep = _recent_feature(
        sweep_features,
        "ict_bars_since_buy_side_sweep",
        len(df),
        breaker_sweep_window,
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
    dist_bull_breaker = np.full(len(df), np.nan)
    dist_bear_breaker = np.full(len(df), np.nan)
    bull_breaker_age = np.full(len(df), np.nan)
    bear_breaker_age = np.full(len(df), np.nan)
    bull_breaker_retests = np.full(len(df), np.nan)
    bear_breaker_retests = np.full(len(df), np.nan)
    active_bull_breaker = np.zeros(len(df), dtype=float)
    active_bear_breaker = np.zeros(len(df), dtype=float)
    bull_breaker_id = np.full(len(df), np.nan)
    bear_breaker_id = np.full(len(df), np.nan)
    bull_breaker_source_id = np.full(len(df), np.nan)
    bear_breaker_source_id = np.full(len(df), np.nan)
    bull_breaker_lower = np.full(len(df), np.nan)
    bull_breaker_upper = np.full(len(df), np.nan)
    bear_breaker_lower = np.full(len(df), np.nan)
    bear_breaker_upper = np.full(len(df), np.nan)
    bull_breaker_formed_index = np.full(len(df), np.nan)
    bear_breaker_formed_index = np.full(len(df), np.nan)
    bull_breaker_source_formed_index = np.full(len(df), np.nan)
    bear_breaker_source_formed_index = np.full(len(df), np.nan)
    bull_breaker_retest_event = np.zeros(len(df), dtype=np.int8)
    bear_breaker_retest_event = np.zeros(len(df), dtype=np.int8)
    bull_breaker_created_event = np.zeros(len(df), dtype=np.int8)
    bear_breaker_created_event = np.zeros(len(df), dtype=np.int8)

    bull_blocks: list[_OrderBlock] = []
    bear_blocks: list[_OrderBlock] = []
    bull_breakers: list[_BreakerBlock] = []
    bear_breakers: list[_BreakerBlock] = []
    records: list[dict[str, object]] = []
    breaker_records: list[dict[str, object]] = []
    block_id = 1
    breaker_id = 1

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
        retested_bull_breaker_ids: set[int] = set()
        retested_bear_breaker_ids: set[int] = set()

        for breakers, direction in ((bull_breakers, 1), (bear_breakers, -1)):
            for breaker in breakers:
                if breaker.invalidated:
                    continue
                if (i - breaker.formed_index) > max_age:
                    breaker.invalidated = True
                    continue
                if direction == 1 and current_close < breaker.lower - tick_size:
                    breaker.invalidated = True
                elif direction == -1 and current_close > breaker.upper + tick_size:
                    breaker.invalidated = True
                elif i > breaker.formed_index and current_high >= breaker.lower and current_low <= breaker.upper:
                    breaker.retest_count += 1
                    if direction == 1:
                        retested_bull_breaker_ids.add(breaker.breaker_id)
                    else:
                        retested_bear_breaker_ids.add(breaker.breaker_id)

        for blocks, direction in ((bull_blocks, 1), (bear_blocks, -1)):
            for block in blocks:
                if block.invalidated:
                    continue
                if (i - block.formed_index) > max_age:
                    block.invalidated = True
                    continue
                if direction == 1 and current_close < block.lower - tick_size:
                    block.invalidated = True
                    if displacement_bear[i] and bear_shift[i] and recent_buy_sweep[i]:
                        breaker = _BreakerBlock(
                            breaker_id=breaker_id,
                            direction=-1,
                            source_order_block_id=block.block_id,
                            source_direction=block.direction,
                            lower=block.lower,
                            upper=block.upper,
                            body_lower=block.body_lower,
                            body_upper=block.body_upper,
                            source_formed_index=block.formed_index,
                            formed_index=i,
                            displacement_score=float(displacement_score[i]),
                        )
                        bear_breakers.append(breaker)
                        bear_breaker_created_event[i] = 1
                        breaker_records.append(
                            {
                                "breaker_id": breaker.breaker_id,
                                "direction": "bearish",
                                "source_order_block_id": block.block_id,
                                "source_direction": "bullish",
                                "source_formed_index": block.formed_index,
                                "formed_index": i,
                                "lower": breaker.lower,
                                "upper": breaker.upper,
                                "body_lower": breaker.body_lower,
                                "body_upper": breaker.body_upper,
                                "displacement_score": breaker.displacement_score,
                            }
                        )
                        breaker_id += 1
                elif direction == -1 and current_close > block.upper + tick_size:
                    block.invalidated = True
                    if displacement_bull[i] and bull_shift[i] and recent_sell_sweep[i]:
                        breaker = _BreakerBlock(
                            breaker_id=breaker_id,
                            direction=1,
                            source_order_block_id=block.block_id,
                            source_direction=block.direction,
                            lower=block.lower,
                            upper=block.upper,
                            body_lower=block.body_lower,
                            body_upper=block.body_upper,
                            source_formed_index=block.formed_index,
                            formed_index=i,
                            displacement_score=float(displacement_score[i]),
                        )
                        bull_breakers.append(breaker)
                        bull_breaker_created_event[i] = 1
                        breaker_records.append(
                            {
                                "breaker_id": breaker.breaker_id,
                                "direction": "bullish",
                                "source_order_block_id": block.block_id,
                                "source_direction": "bearish",
                                "source_formed_index": block.formed_index,
                                "formed_index": i,
                                "lower": breaker.lower,
                                "upper": breaker.upper,
                                "body_lower": breaker.body_lower,
                                "body_upper": breaker.body_upper,
                                "displacement_score": breaker.displacement_score,
                            }
                        )
                        breaker_id += 1
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
        live_bull_breakers = [breaker for breaker in bull_breakers if not breaker.invalidated]
        live_bear_breakers = [breaker for breaker in bear_breakers if not breaker.invalidated]
        bull_blocks = live_bull
        bear_blocks = live_bear
        bull_breakers = live_bull_breakers
        bear_breakers = live_bear_breakers
        active_bull[i] = len(live_bull)
        active_bear[i] = len(live_bear)
        active_bull_breaker[i] = len(live_bull_breakers)
        active_bear_breaker[i] = len(live_bear_breakers)
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
        if live_bull_breakers:
            nearest = min(
                live_bull_breakers,
                key=lambda breaker: _distance_to_zone(current_close, breaker.lower, breaker.upper),
            )
            dist_bull_breaker[i] = _distance_to_zone(current_close, nearest.lower, nearest.upper) / current_atr
            bull_breaker_age[i] = i - nearest.formed_index
            bull_breaker_retests[i] = nearest.retest_count
            bull_breaker_id[i] = nearest.breaker_id
            bull_breaker_source_id[i] = nearest.source_order_block_id
            bull_breaker_lower[i] = nearest.lower
            bull_breaker_upper[i] = nearest.upper
            bull_breaker_formed_index[i] = nearest.formed_index
            bull_breaker_source_formed_index[i] = nearest.source_formed_index
            bull_breaker_retest_event[i] = int(nearest.breaker_id in retested_bull_breaker_ids)
        if live_bear_breakers:
            nearest = min(
                live_bear_breakers,
                key=lambda breaker: _distance_to_zone(current_close, breaker.lower, breaker.upper),
            )
            dist_bear_breaker[i] = _distance_to_zone(current_close, nearest.lower, nearest.upper) / current_atr
            bear_breaker_age[i] = i - nearest.formed_index
            bear_breaker_retests[i] = nearest.retest_count
            bear_breaker_id[i] = nearest.breaker_id
            bear_breaker_source_id[i] = nearest.source_order_block_id
            bear_breaker_lower[i] = nearest.lower
            bear_breaker_upper[i] = nearest.upper
            bear_breaker_formed_index[i] = nearest.formed_index
            bear_breaker_source_formed_index[i] = nearest.source_formed_index
            bear_breaker_retest_event[i] = int(nearest.breaker_id in retested_bear_breaker_ids)

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
    out["dist_to_bull_breaker_atr"] = dist_bull_breaker
    out["dist_to_bear_breaker_atr"] = dist_bear_breaker
    out["bull_breaker_age_bars"] = bull_breaker_age
    out["bear_breaker_age_bars"] = bear_breaker_age
    out["ict_bull_breaker_retest_count"] = bull_breaker_retests
    out["ict_bear_breaker_retest_count"] = bear_breaker_retests
    out["ict_active_bull_breaker_count"] = active_bull_breaker
    out["ict_active_bear_breaker_count"] = active_bear_breaker
    out["ict_nearest_bull_breaker_id"] = pd.Series(bull_breaker_id, index=df.index, dtype="Int64")
    out["ict_nearest_bear_breaker_id"] = pd.Series(bear_breaker_id, index=df.index, dtype="Int64")
    out["ict_nearest_bull_breaker_source_order_block_id"] = pd.Series(
        bull_breaker_source_id,
        index=df.index,
        dtype="Int64",
    )
    out["ict_nearest_bear_breaker_source_order_block_id"] = pd.Series(
        bear_breaker_source_id,
        index=df.index,
        dtype="Int64",
    )
    out["ict_nearest_bull_breaker_lower"] = bull_breaker_lower
    out["ict_nearest_bull_breaker_upper"] = bull_breaker_upper
    out["ict_nearest_bear_breaker_lower"] = bear_breaker_lower
    out["ict_nearest_bear_breaker_upper"] = bear_breaker_upper
    out["ict_nearest_bull_breaker_formed_index"] = pd.Series(
        bull_breaker_formed_index,
        index=df.index,
        dtype="Int64",
    )
    out["ict_nearest_bear_breaker_formed_index"] = pd.Series(
        bear_breaker_formed_index,
        index=df.index,
        dtype="Int64",
    )
    out["ict_nearest_bull_breaker_source_order_block_formed_index"] = pd.Series(
        bull_breaker_source_formed_index,
        index=df.index,
        dtype="Int64",
    )
    out["ict_nearest_bear_breaker_source_order_block_formed_index"] = pd.Series(
        bear_breaker_source_formed_index,
        index=df.index,
        dtype="Int64",
    )
    out["ict_bull_breaker_retest_event"] = bull_breaker_retest_event
    out["ict_bear_breaker_retest_event"] = bear_breaker_retest_event
    out["ict_bull_breaker_created_event"] = bull_breaker_created_event
    out["ict_bear_breaker_created_event"] = bear_breaker_created_event
    out.attrs["order_blocks"] = pd.DataFrame(records)
    out.attrs["breaker_blocks"] = pd.DataFrame(breaker_records)
    return out
