from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.transforms import bars_since_event, safe_divide

from ..common import cfg_value, get_atr_like, sweep_buffer_price


@dataclass(frozen=True)
class _LevelSpec:
    name: str
    column: str
    direction: int
    code: int


_LEVEL_SPECS = (
    _LevelSpec("swing_high", "ict_latest_swing_high", 1, 1),
    _LevelSpec("equal_high", "ict_equal_high_pool_level", 1, 2),
    _LevelSpec("prior_rth_high", "ict_prior_rth_high", 1, 3),
    _LevelSpec("overnight_high", "ict_overnight_high", 1, 4),
    _LevelSpec("ib_high", "ict_ib_high", 1, 5),
    _LevelSpec("prior_week_high", "ict_prior_week_high", 1, 6),
    _LevelSpec("swing_low", "ict_latest_swing_low", -1, 11),
    _LevelSpec("equal_low", "ict_equal_low_pool_level", -1, 12),
    _LevelSpec("prior_rth_low", "ict_prior_rth_low", -1, 13),
    _LevelSpec("overnight_low", "ict_overnight_low", -1, 14),
    _LevelSpec("ib_low", "ict_ib_low", -1, 15),
    _LevelSpec("prior_week_low", "ict_prior_week_low", -1, 16),
)


def detect_ict_sweeps(
    df: pd.DataFrame,
    config: object,
    *,
    swing_features: pd.DataFrame | None = None,
    liquidity_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect canonical sweep-and-reclaim events across ICT liquidity levels."""

    out = pd.DataFrame(index=df.index)
    if not {"high", "low", "close"}.issubset(df.columns):
        return out

    reference = pd.DataFrame(index=df.index)
    for frame in (swing_features, liquidity_features):
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            if column not in reference.columns:
                reference[column] = frame[column]

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    atr = get_atr_like(df)
    buffer_values = np.asarray(sweep_buffer_price(atr, config), dtype=float)
    reclaim_bars = int(cfg_value(config, "sweep_close_back_bars", "ict_sweep_close_back_bars", default=1))

    buy_event = np.zeros(len(df), dtype=np.int8)
    sell_event = np.zeros(len(df), dtype=np.int8)
    failed_buy = np.zeros(len(df), dtype=np.int8)
    failed_sell = np.zeros(len(df), dtype=np.int8)
    level_code = np.zeros(len(df), dtype=np.int16)
    penetration = np.full(len(df), np.nan)
    reclaim_delay = np.full(len(df), np.nan)
    level_value = np.full(len(df), np.nan)
    sweep_direction = np.zeros(len(df), dtype=np.int8)
    event_records: list[dict[str, object]] = []

    highs = high.to_numpy(dtype=float, copy=False)
    lows = low.to_numpy(dtype=float, copy=False)
    closes = close.to_numpy(dtype=float, copy=False)
    atr_values = atr.ffill().fillna(0.0).to_numpy(dtype=float, copy=False)

    for spec in _LEVEL_SPECS:
        if spec.column not in reference.columns:
            continue
        levels = pd.to_numeric(reference[spec.column], errors="coerce").to_numpy(dtype=float, copy=False)
        for i in range(len(df)):
            level = levels[i]
            if not np.isfinite(level):
                continue

            buffer_value = buffer_values[i] if i < len(buffer_values) else 0.0
            if spec.direction == 1:
                pierced = highs[i] >= level + buffer_value
            else:
                pierced = lows[i] <= level - buffer_value
            if not pierced:
                continue

            found_reclaim = False
            end_index = min(len(df), i + reclaim_bars + 1)
            for j in range(i, end_index):
                if spec.direction == 1 and closes[j] < level:
                    buy_event[j] = 1
                    level_code[j] = spec.code
                    penetration[j] = max((highs[i] - level) / max(atr_values[i], 1e-6), 0.0)
                    reclaim_delay[j] = float(j - i)
                    level_value[j] = level
                    sweep_direction[j] = 1
                    event_records.append(
                        {
                            "event_index": j,
                            "pierce_index": i,
                            "direction": "buy_side",
                            "level_type": spec.name,
                            "level_code": spec.code,
                            "level_value": level,
                            "penetration_atr": penetration[j],
                            "bars_to_reclaim": reclaim_delay[j],
                            "failed_sweep": False,
                        }
                    )
                    found_reclaim = True
                    break
                if spec.direction == -1 and closes[j] > level:
                    sell_event[j] = 1
                    level_code[j] = spec.code
                    penetration[j] = max((level - lows[i]) / max(atr_values[i], 1e-6), 0.0)
                    reclaim_delay[j] = float(j - i)
                    level_value[j] = level
                    sweep_direction[j] = -1
                    event_records.append(
                        {
                            "event_index": j,
                            "pierce_index": i,
                            "direction": "sell_side",
                            "level_type": spec.name,
                            "level_code": spec.code,
                            "level_value": level,
                            "penetration_atr": penetration[j],
                            "bars_to_reclaim": reclaim_delay[j],
                            "failed_sweep": False,
                        }
                    )
                    found_reclaim = True
                    break

            if not found_reclaim:
                decision_index = min(len(df) - 1, i + reclaim_bars)
                if spec.direction == 1:
                    failed_buy[decision_index] = 1
                else:
                    failed_sell[decision_index] = 1
                event_records.append(
                    {
                        "event_index": decision_index,
                        "pierce_index": i,
                        "direction": "buy_side" if spec.direction == 1 else "sell_side",
                        "level_type": spec.name,
                        "level_code": spec.code,
                        "level_value": level,
                        "penetration_atr": max(
                            (highs[i] - level) / max(atr_values[i], 1e-6),
                            0.0,
                        )
                        if spec.direction == 1
                        else max((level - lows[i]) / max(atr_values[i], 1e-6), 0.0),
                        "bars_to_reclaim": np.nan,
                        "failed_sweep": True,
                    }
                )

    buy_event_series = pd.Series(buy_event > 0, index=df.index)
    sell_event_series = pd.Series(sell_event > 0, index=df.index)
    sweep_any = buy_event_series | sell_event_series
    out["ict_buy_side_sweep"] = buy_event
    out["ict_sell_side_sweep"] = sell_event
    out["ict_sweep_any"] = sweep_any.astype(np.int8)
    out["ict_failed_buy_side_sweep"] = failed_buy
    out["ict_failed_sell_side_sweep"] = failed_sell
    out["ict_sweep_level_code"] = pd.Series(level_code, index=df.index, dtype="Int64")
    out["ict_sweep_penetration_atr"] = penetration
    out["ict_sweep_reclaim_bars"] = reclaim_delay
    out["ict_sweep_level_value"] = level_value
    out["ict_sweep_direction"] = pd.Series(sweep_direction, index=df.index, dtype="Int64")
    out["ict_bars_since_buy_side_sweep"] = bars_since_event(buy_event_series)
    out["ict_bars_since_sell_side_sweep"] = bars_since_event(sell_event_series)
    out["ict_bars_since_sweep_any"] = bars_since_event(sweep_any)
    out.attrs["sweep_events"] = pd.DataFrame(event_records)
    return out
