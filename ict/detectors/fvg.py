from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.transforms import safe_divide

from ..common import cfg_value, get_atr_like, get_tick_size


@dataclass
class _GapZone:
    zone_id: int
    direction: int
    original_direction: int
    lower: float
    upper: float
    ce: float
    formed_index: int
    formed_time: object
    source_timeframe: str
    created_by_displacement: bool
    inverted: bool = False
    inversion_index: int | None = None
    ce_tapped: bool = False
    max_mitigation: float = 0.0
    invalidated: bool = False


def _distance_to_zone(price: float, lower: float, upper: float) -> float:
    if np.isnan(lower) or np.isnan(upper):
        return np.nan
    if lower <= price <= upper:
        return 0.0
    if price < lower:
        return lower - price
    return price - upper


def detect_ict_fvg(
    df: pd.DataFrame,
    config: object,
    *,
    displacement_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect active FVG and IFVG zones with CE tracking."""

    out = pd.DataFrame(index=df.index)
    if not {"high", "low", "close"}.issubset(df.columns):
        return out

    atr = get_atr_like(df)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    event_time = pd.to_datetime(df.get("datetime", pd.Series(pd.NaT, index=df.index)), errors="coerce")

    min_gap_atr = float(cfg_value(config, "ict_fvg_min_gap_atr", default=0.15))
    tick_size = get_tick_size(config)
    min_gap = np.maximum(atr.fillna(0.0).to_numpy(dtype=float, copy=False) * min_gap_atr, tick_size)
    max_age = int(cfg_value(config, "ict_fvg_max_age", "ict_zone_max_age", default=120))

    bull_form = (low - high.shift(2)).to_numpy(dtype=float, copy=False)
    bear_form = (low.shift(2) - high).to_numpy(dtype=float, copy=False)
    bull_disp = (
        pd.to_numeric(displacement_features.get("displacement_bullish"), errors="coerce").fillna(0).astype(bool).to_numpy()
        if displacement_features is not None and "displacement_bullish" in displacement_features.columns
        else np.zeros(len(df), dtype=bool)
    )
    bear_disp = (
        pd.to_numeric(displacement_features.get("displacement_bearish"), errors="coerce").fillna(0).astype(bool).to_numpy()
        if displacement_features is not None and "displacement_bearish" in displacement_features.columns
        else np.zeros(len(df), dtype=bool)
    )

    active_bull_count = np.zeros(len(df), dtype=float)
    active_bear_count = np.zeros(len(df), dtype=float)
    active_bull_ifvg_count = np.zeros(len(df), dtype=float)
    active_bear_ifvg_count = np.zeros(len(df), dtype=float)
    dist_bull = np.full(len(df), np.nan)
    dist_bear = np.full(len(df), np.nan)
    dist_bull_ce = np.full(len(df), np.nan)
    dist_bear_ce = np.full(len(df), np.nan)
    inside_bull = np.zeros(len(df), dtype=np.int8)
    inside_bear = np.zeros(len(df), dtype=np.int8)
    bull_ce_tap = np.zeros(len(df), dtype=np.int8)
    bear_ce_tap = np.zeros(len(df), dtype=np.int8)
    nearest_bull_ifvg = np.zeros(len(df), dtype=np.int8)
    nearest_bear_ifvg = np.zeros(len(df), dtype=np.int8)
    bull_mitigation = np.full(len(df), np.nan)
    bear_mitigation = np.full(len(df), np.nan)
    bull_id = np.full(len(df), np.nan)
    bear_id = np.full(len(df), np.nan)
    bull_lower = np.full(len(df), np.nan)
    bull_upper = np.full(len(df), np.nan)
    bear_lower = np.full(len(df), np.nan)
    bear_upper = np.full(len(df), np.nan)
    bull_ce_price = np.full(len(df), np.nan)
    bear_ce_price = np.full(len(df), np.nan)
    bull_formed_index = np.full(len(df), np.nan)
    bear_formed_index = np.full(len(df), np.nan)
    bull_inversion_index = np.full(len(df), np.nan)
    bear_inversion_index = np.full(len(df), np.nan)
    bull_created_by_disp = np.zeros(len(df), dtype=np.int8)
    bear_created_by_disp = np.zeros(len(df), dtype=np.int8)

    zones: list[_GapZone] = []
    emitted_records: list[dict[str, object]] = []
    zone_id_counter = itertools.count(1)

    highs = high.to_numpy(dtype=float, copy=False)
    lows = low.to_numpy(dtype=float, copy=False)
    closes = close.to_numpy(dtype=float, copy=False)
    atr_values = atr.ffill().fillna(0.0).to_numpy(dtype=float, copy=False)

    for i in range(len(df)):
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]
        current_atr = atr_values[i] if atr_values[i] > 0 else 1.0

        for zone in zones:
            if zone.invalidated:
                continue
            if (i - zone.formed_index) > max_age:
                zone.invalidated = True
                continue

            overlap = max(0.0, min(current_high, zone.upper) - max(current_low, zone.lower))
            width = max(zone.upper - zone.lower, tick_size)
            if overlap > 0:
                zone.max_mitigation = max(zone.max_mitigation, overlap / width)
                if current_low <= zone.ce <= current_high:
                    zone.ce_tapped = True

            if zone.direction == 1 and current_close < zone.lower:
                zone.direction = -1
                zone.inverted = True
                zone.inversion_index = i
            elif zone.direction == -1 and current_close > zone.upper:
                zone.direction = 1
                zone.inverted = True
                zone.inversion_index = i

        if i >= 2 and np.isfinite(highs[i - 2]) and bull_form[i] > min_gap[i]:
            lower = float(highs[i - 2])
            upper = float(lows[i])
            zone = _GapZone(
                zone_id=next(zone_id_counter),
                direction=1,
                original_direction=1,
                lower=lower,
                upper=upper,
                ce=(lower + upper) / 2.0,
                formed_index=i,
                formed_time=event_time.iloc[i] if i < len(event_time) else pd.NaT,
                source_timeframe="5m",
                created_by_displacement=bool(bull_disp[i]),
            )
            zones.append(zone)
            emitted_records.append(
                {
                    "fvg_id": zone.zone_id,
                    "formed_index": i,
                    "formed_time": zone.formed_time,
                    "direction": zone.direction,
                    "lower": zone.lower,
                    "upper": zone.upper,
                    "ce": zone.ce,
                    "created_by_displacement": zone.created_by_displacement,
                }
            )

        if i >= 2 and np.isfinite(lows[i - 2]) and bear_form[i] > min_gap[i]:
            lower = float(highs[i])
            upper = float(lows[i - 2])
            zone = _GapZone(
                zone_id=next(zone_id_counter),
                direction=-1,
                original_direction=-1,
                lower=lower,
                upper=upper,
                ce=(lower + upper) / 2.0,
                formed_index=i,
                formed_time=event_time.iloc[i] if i < len(event_time) else pd.NaT,
                source_timeframe="5m",
                created_by_displacement=bool(bear_disp[i]),
            )
            zones.append(zone)
            emitted_records.append(
                {
                    "fvg_id": zone.zone_id,
                    "formed_index": i,
                    "formed_time": zone.formed_time,
                    "direction": zone.direction,
                    "lower": zone.lower,
                    "upper": zone.upper,
                    "ce": zone.ce,
                    "created_by_displacement": zone.created_by_displacement,
                }
            )

        active_bull = [zone for zone in zones if not zone.invalidated and zone.direction == 1]
        active_bear = [zone for zone in zones if not zone.invalidated and zone.direction == -1]
        active_bull_count[i] = len(active_bull)
        active_bear_count[i] = len(active_bear)
        active_bull_ifvg_count[i] = sum(1 for zone in active_bull if zone.inverted)
        active_bear_ifvg_count[i] = sum(1 for zone in active_bear if zone.inverted)

        if active_bull:
            nearest = min(active_bull, key=lambda zone: _distance_to_zone(current_close, zone.lower, zone.upper))
            dist_bull[i] = _distance_to_zone(current_close, nearest.lower, nearest.upper) / current_atr
            dist_bull_ce[i] = abs(current_close - nearest.ce) / current_atr
            inside_bull[i] = int(nearest.lower <= current_close <= nearest.upper)
            bull_ce_tap[i] = int(nearest.ce_tapped)
            nearest_bull_ifvg[i] = int(nearest.inverted)
            bull_mitigation[i] = nearest.max_mitigation
            bull_id[i] = nearest.zone_id
            bull_lower[i] = nearest.lower
            bull_upper[i] = nearest.upper
            bull_ce_price[i] = nearest.ce
            bull_formed_index[i] = nearest.formed_index
            bull_inversion_index[i] = float(nearest.inversion_index) if nearest.inversion_index is not None else np.nan
            bull_created_by_disp[i] = int(nearest.created_by_displacement)

        if active_bear:
            nearest = min(active_bear, key=lambda zone: _distance_to_zone(current_close, zone.lower, zone.upper))
            dist_bear[i] = _distance_to_zone(current_close, nearest.lower, nearest.upper) / current_atr
            dist_bear_ce[i] = abs(current_close - nearest.ce) / current_atr
            inside_bear[i] = int(nearest.lower <= current_close <= nearest.upper)
            bear_ce_tap[i] = int(nearest.ce_tapped)
            nearest_bear_ifvg[i] = int(nearest.inverted)
            bear_mitigation[i] = nearest.max_mitigation
            bear_id[i] = nearest.zone_id
            bear_lower[i] = nearest.lower
            bear_upper[i] = nearest.upper
            bear_ce_price[i] = nearest.ce
            bear_formed_index[i] = nearest.formed_index
            bear_inversion_index[i] = float(nearest.inversion_index) if nearest.inversion_index is not None else np.nan
            bear_created_by_disp[i] = int(nearest.created_by_displacement)

    out["dist_to_bull_fvg_atr"] = dist_bull
    out["dist_to_bear_fvg_atr"] = dist_bear
    out["dist_to_bull_fvg_ce_atr"] = dist_bull_ce
    out["dist_to_bear_fvg_ce_atr"] = dist_bear_ce
    out["active_bull_fvg_count"] = active_bull_count
    out["active_bear_fvg_count"] = active_bear_count
    out["active_bull_ifvg_count"] = active_bull_ifvg_count
    out["active_bear_ifvg_count"] = active_bear_ifvg_count
    out["ict_inside_bull_fvg"] = inside_bull
    out["ict_inside_bear_fvg"] = inside_bear
    out["ict_bull_fvg_ce_tapped"] = bull_ce_tap
    out["ict_bear_fvg_ce_tapped"] = bear_ce_tap
    out["ict_nearest_bull_fvg_is_ifvg"] = nearest_bull_ifvg
    out["ict_nearest_bear_fvg_is_ifvg"] = nearest_bear_ifvg
    out["ict_bull_fvg_mitigation_pct"] = bull_mitigation
    out["ict_bear_fvg_mitigation_pct"] = bear_mitigation
    out["ict_nearest_bull_fvg_id"] = pd.Series(bull_id, index=df.index, dtype="Int64")
    out["ict_nearest_bear_fvg_id"] = pd.Series(bear_id, index=df.index, dtype="Int64")
    out["ict_nearest_bull_fvg_lower"] = bull_lower
    out["ict_nearest_bull_fvg_upper"] = bull_upper
    out["ict_nearest_bear_fvg_lower"] = bear_lower
    out["ict_nearest_bear_fvg_upper"] = bear_upper
    out["ict_nearest_bull_fvg_ce"] = bull_ce_price
    out["ict_nearest_bear_fvg_ce"] = bear_ce_price
    out["ict_nearest_bull_fvg_formed_index"] = pd.Series(bull_formed_index, index=df.index, dtype="Int64")
    out["ict_nearest_bear_fvg_formed_index"] = pd.Series(bear_formed_index, index=df.index, dtype="Int64")
    out["ict_nearest_bull_fvg_inversion_index"] = pd.Series(bull_inversion_index, index=df.index, dtype="Int64")
    out["ict_nearest_bear_fvg_inversion_index"] = pd.Series(bear_inversion_index, index=df.index, dtype="Int64")
    out["ict_nearest_bull_fvg_created_by_displacement"] = bull_created_by_disp
    out["ict_nearest_bear_fvg_created_by_displacement"] = bear_created_by_disp
    out.attrs["fvg_zones"] = pd.DataFrame(
        [
            {
                "fvg_id": zone.zone_id,
                "direction": zone.direction,
                "original_direction": zone.original_direction,
                "lower": zone.lower,
                "upper": zone.upper,
                "ce": zone.ce,
                "formed_index": zone.formed_index,
                "formed_time": zone.formed_time,
                "source_timeframe": zone.source_timeframe,
                "created_by_displacement": zone.created_by_displacement,
                "inverted": zone.inverted,
                "inversion_index": zone.inversion_index,
                "ce_tapped": zone.ce_tapped,
                "mitigated_pct": zone.max_mitigation,
                "invalidated": zone.invalidated,
            }
            for zone in zones
        ]
    )
    return out
