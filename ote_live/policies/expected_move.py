"""Causal ES next-bar absolute-movement estimate; no labels or target distances."""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT = "es_abs_close_move_20x5m_v1"
LOOKBACK = 20
TICK_SIZE = 0.25
SPREAD_TICKS = {"overlap": 1.0, "london": 1.0, "new_york": 1.0, "asia": 1.5, "off_hours": 2.0}


def build_expected_move_frame(market: pd.DataFrame) -> pd.DataFrame:
    """At bar t close, mean(|close[j]-close[j-1]|)/.25 for j=t-19..t.

    Requires 21 valid completed same-contract bars at uninterrupted 5m intervals.
    Any unknown identity/completion, invalid OHLC, gap or roll resets warmup.
    Closed-market intervals also reset: no interpolation or overnight jump sample.
    Raw market input only; callers must not pass imputed model features here.
    """
    frame = market.reset_index(drop=True)
    count = len(frame)
    out = pd.DataFrame(index=frame.index)
    out["expected_move_contract"] = CONTRACT
    out["expected_move_unit"] = "ticks"
    out["expected_move_ticks"] = np.nan
    out["expected_move_status"] = "missing_market_columns"
    out["expected_move_sample_count"] = 0
    out["expected_move_source_close_ts"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    time_col = next((c for c in ("timestamp", "open_ts", "datetime") if c in frame), None)
    identity_col = next((c for c in ("instrument_id", "coordinate_id", "contract_symbol") if c in frame and frame[c].notna().any()), None)
    required = {"open", "high", "low", "close", "is_complete"}
    if not count or not required.issubset(frame) or time_col is None or identity_col is None:
        return out
    times = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    identity = frame[identity_col].astype("string")
    valid = (times.notna() & frame.is_complete.eq(True).fillna(False) & identity.notna()
             & ~identity.isin(["", "nan", "None", "<NA>"]) & np.isfinite(prices).all(axis=1)
             & prices.gt(0).all(axis=1)
             & prices.high.ge(prices[["open", "close", "low"]].max(axis=1))
             & prices.low.le(prices[["open", "close", "high"]].min(axis=1)))
    if "asset" in frame:
        valid &= frame.asset.eq("ES")
    if "timeframe" in frame:
        valid &= frame.timeframe.eq("5m")
    consecutive = times.diff().eq(pd.Timedelta(minutes=5))
    same_contract = identity.eq(identity.shift()).fillna(False)
    pair = valid & valid.shift(fill_value=False) & consecutive & same_contract
    movements = (prices.close.diff().abs() / TICK_SIZE).where(pair)
    samples = movements.notna().astype(int).groupby((~pair).astype("int64").cumsum()).cumsum().clip(upper=LOOKBACK)
    values = movements.rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    out["expected_move_ticks"] = values
    out["expected_move_status"] = np.where(values.notna(), "ready", "warmup_after_gap_roll_or_invalid_bar")
    out.loc[~valid, "expected_move_status"] = "invalid_or_unverified_source_bar"
    out["expected_move_sample_count"] = samples
    out["expected_move_source_close_ts"] = times + pd.Timedelta(minutes=5)
    return out


def expected_move_spread_check(policy, row: dict) -> dict:
    """Strict opt-in tick contract. Legacy pips values never satisfy this gate."""
    check = {"status": "unverified", "reasons": [], "inputs": {}, "unit": "ticks",
             "contract": CONTRACT, "spread_source": "es_session_cost_v1"}
    keys = ["expected_move_ticks", "expected_move_status", "expected_move_contract", "expected_move_unit",
            "expected_move_sample_count", "expected_move_source_close_ts", "session_regime"]
    check["inputs"] = {key: _serial(row.get(key)) for key in keys}
    reasons = check["reasons"]
    if row.get("expected_move_contract") != CONTRACT or row.get("expected_move_unit") != "ticks":
        reasons.append("expected_move_contract_or_unit_mismatch")
    if row.get("expected_move_status") != "ready" or row.get("expected_move_sample_count") != LOOKBACK:
        reasons.append("expected_move_not_ready")
    value = row.get("expected_move_ticks")
    if not _finite(value) or float(value) < 0:
        reasons.append("expected_move_missing_or_invalid")
    source_close = pd.to_datetime(row.get("expected_move_source_close_ts"), utc=True, errors="coerce")
    decision_bar = pd.to_datetime(row.get("timestamp", row.get("datetime", row.get("open_ts"))), utc=True, errors="coerce")
    if pd.isna(source_close) or pd.isna(decision_bar) or source_close != decision_bar + pd.Timedelta(minutes=5):
        reasons.append("expected_move_source_timestamp_mismatch")
    spreads = policy.cost_assumptions.session_spread_ticks
    if spreads != SPREAD_TICKS or policy.cost_assumptions.session_spread_pips:
        reasons.append("expected_move_spread_unit_contract_mismatch")
    session = row.get("session_regime")
    if session not in SPREAD_TICKS:
        reasons.append("policy_input_missing_or_unknown:session_regime")
    if reasons:
        return check
    spread = spreads[session]
    required = policy.abstain_policy.minimum_expected_move_to_spread * spread
    if not _finite(required) or required < 0:
        reasons.append("expected_move_spread_multiplier_invalid")
        return check
    check.update(spread_ticks=spread, required_move_ticks=required,
                 status="passed" if value >= required else "rejected")
    if value < required:
        reasons.append("expected_move_below_spread")
    return check


def _finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _serial(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value
