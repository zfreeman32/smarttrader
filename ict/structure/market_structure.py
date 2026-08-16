from __future__ import annotations

import numpy as np
import pandas as pd

from features.transforms import bars_since_event

from ..common import get_atr_like, structure_break_buffer_price


def detect_ict_market_structure(
    df: pd.DataFrame,
    config: object,
    *,
    swing_features: pd.DataFrame | None = None,
    sweep_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect causal BOS / CHoCH / MSS using confirmed swings only."""

    out = pd.DataFrame(index=df.index)
    if swing_features is None or swing_features.empty:
        return out

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr = get_atr_like(df)
    latest_swing_high = pd.to_numeric(swing_features.get("ict_latest_swing_high"), errors="coerce")
    latest_swing_low = pd.to_numeric(swing_features.get("ict_latest_swing_low"), errors="coerce")
    break_buffer = structure_break_buffer_price(atr, config)

    recent_sell_sweep = (
        pd.to_numeric(sweep_features.get("ict_bars_since_sell_side_sweep"), errors="coerce").le(3).fillna(False)
        if sweep_features is not None and not sweep_features.empty
        else pd.Series(False, index=df.index)
    )
    recent_buy_sweep = (
        pd.to_numeric(sweep_features.get("ict_bars_since_buy_side_sweep"), errors="coerce").le(3).fillna(False)
        if sweep_features is not None and not sweep_features.empty
        else pd.Series(False, index=df.index)
    )

    bos_bull = np.zeros(len(df), dtype=np.int8)
    bos_bear = np.zeros(len(df), dtype=np.int8)
    choch_bull = np.zeros(len(df), dtype=np.int8)
    choch_bear = np.zeros(len(df), dtype=np.int8)
    mss_bull = np.zeros(len(df), dtype=np.int8)
    mss_bear = np.zeros(len(df), dtype=np.int8)
    state = np.zeros(len(df), dtype=np.int8)
    break_level = np.full(len(df), np.nan)

    last_broken_high = np.nan
    last_broken_low = np.nan
    trend_state = 0

    highs = high.to_numpy(dtype=float, copy=False)
    lows = low.to_numpy(dtype=float, copy=False)
    latest_swing_highs = latest_swing_high.to_numpy(dtype=float, copy=False)
    latest_swing_lows = latest_swing_low.to_numpy(dtype=float, copy=False)
    recent_sell = recent_sell_sweep.to_numpy(dtype=bool, copy=False)
    recent_buy = recent_buy_sweep.to_numpy(dtype=bool, copy=False)
    break_buffers = np.asarray(break_buffer, dtype=float)

    for i in range(len(df)):
        latest_high = latest_swing_highs[i]
        latest_low = latest_swing_lows[i]
        buffer_value = break_buffers[i] if i < len(break_buffers) else 0.0

        bull_break = (
            np.isfinite(latest_high)
            and highs[i] > latest_high + buffer_value
            and (not np.isfinite(last_broken_high) or abs(latest_high - last_broken_high) > buffer_value / 2.0)
        )
        bear_break = (
            np.isfinite(latest_low)
            and lows[i] < latest_low - buffer_value
            and (not np.isfinite(last_broken_low) or abs(latest_low - last_broken_low) > buffer_value / 2.0)
        )

        if bull_break:
            break_level[i] = latest_high
            if trend_state == -1:
                choch_bull[i] = 1
                if recent_sell[i]:
                    mss_bull[i] = 1
            else:
                bos_bull[i] = 1
            trend_state = 1
            last_broken_high = latest_high
        elif bear_break:
            break_level[i] = latest_low
            if trend_state == 1:
                choch_bear[i] = 1
                if recent_buy[i]:
                    mss_bear[i] = 1
            else:
                bos_bear[i] = 1
            trend_state = -1
            last_broken_low = latest_low

        state[i] = trend_state

    out["ict_bos_bull"] = bos_bull
    out["ict_bos_bear"] = bos_bear
    out["ict_choch_bull"] = choch_bull
    out["ict_choch_bear"] = choch_bear
    out["ict_mss_bull"] = mss_bull
    out["ict_mss_bear"] = mss_bear
    out["ict_structure_state"] = state
    out["ict_structure_break_level"] = break_level

    bos_any = pd.Series((bos_bull + bos_bear) > 0, index=df.index)
    choch_any = pd.Series((choch_bull + choch_bear) > 0, index=df.index)
    mss_any = pd.Series((mss_bull + mss_bear) > 0, index=df.index)
    out["ict_bars_since_bos_bull"] = bars_since_event(pd.Series(bos_bull > 0, index=df.index))
    out["ict_bars_since_bos_bear"] = bars_since_event(pd.Series(bos_bear > 0, index=df.index))
    out["ict_bars_since_choch_bull"] = bars_since_event(pd.Series(choch_bull > 0, index=df.index))
    out["ict_bars_since_choch_bear"] = bars_since_event(pd.Series(choch_bear > 0, index=df.index))
    out["ict_bars_since_mss_bull"] = bars_since_event(pd.Series(mss_bull > 0, index=df.index))
    out["ict_bars_since_mss_bear"] = bars_since_event(pd.Series(mss_bear > 0, index=df.index))
    out["ict_bars_since_bos_any"] = bars_since_event(bos_any)
    out["ict_bars_since_choch_any"] = bars_since_event(choch_any)
    out["ict_bars_since_mss_any"] = bars_since_event(mss_any)
    return out
