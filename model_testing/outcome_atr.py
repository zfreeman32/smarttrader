"""Causal structural ATR in price units; never imports historical label outputs."""
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from model_testing.outcome_contracts import FrozenModel, content_hash


class StructuralATREvidence(FrozenModel):
    definition: Literal["completed_1h_structural_atr"] = "completed_1h_structural_atr"
    producer_id: Literal["b1_structural_1h_wilder14_v1"] = "b1_structural_1h_wilder14_v1"
    timeframe_minutes: Literal[60] = 60
    period: Literal[14] = 14
    smoothing: Literal["wilder_alpha_1_over_14_adjust_false"] = "wilder_alpha_1_over_14_adjust_false"
    units: Literal["price"] = "price"
    value: float = Field(gt=0)
    source_close_ts: datetime
    history_start_ts: datetime
    available_at_ts: datetime
    coordinate_id: str | None
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_close_ts", "history_start_ts", "available_at_ts")
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None:
            raise ValueError("ATR provenance timestamps require an explicit timezone")
        return value


class Structural30mATREvidence(StructuralATREvidence):
    definition: Literal["completed_30m_structural_atr"] = "completed_30m_structural_atr"
    producer_id: Literal["b1_structural_30m_wilder14_v1"] = "b1_structural_30m_wilder14_v1"
    timeframe_minutes: Literal[30] = 30


STRUCTURAL_ATR_MINUTES = {"completed_1h_structural_atr": 60, "completed_30m_structural_atr": 30}


def completed_structural_atr(tape, *, timeframe_minutes: int = 60,
                             instrument: str | None = None) -> pd.DataFrame:
    """One record per scheduled structural period, published only at its close.

    Twelve exact 5m slots (or sixty 1m slots) are required. Missing/corrupt,
    partial-session and mixed-roll hours reset the Wilder state/warmup. Planned
    fully closed hours are absent from the schedule and do not fabricate bars.
    The output can be as-of mapped to decisions, never mapped by the period open.
    OTE continuation/breakout use six exact 5m slots per 30m period. Spot EURUSD
    may omit futures coordinate IDs; an explicitly supplied series ID must still
    be consistent. Futures and callers without an instrument retain strict IDs.
    """
    from model_testing.outcome_engine import valid_ohlc, utc

    if timeframe_minutes not in (30, 60):
        raise ValueError("Structural ATR supports only the frozen 30m/1h definitions")
    if instrument not in (None, "ES", "6E", "EURUSD"):
        raise ValueError("Unknown ATR instrument")
    delta = tape.delta
    if pd.Timedelta(minutes=timeframe_minutes) % delta:
        raise ValueError("Execution bars must divide the structural ATR period")
    evidence_type = StructuralATREvidence if timeframe_minutes == 60 else Structural30mATREvidence
    period_name = "hour" if timeframe_minutes == 60 else "30m_period"
    records, history = [], []
    previous_close = state = coordinate = available = None
    for start in tape.schedule.index.floor(f"{timeframe_minutes}min").unique():
        end = start + pd.Timedelta(minutes=timeframe_minutes)
        expected = pd.date_range(start, end, freq=delta, inclusive="left")
        scheduled = tape.schedule.index[(tape.schedule.index >= start) & (tape.schedule.index < end)]
        status = "OK"
        if not scheduled.equals(expected) or not expected.isin(tape.bars.index).all():
            status = f"incomplete_{period_name}"
            frame = None
        else:
            frame = tape.bars.loc[expected]
            if not all(valid_ohlc(row) for _, row in frame.iterrows()):
                status = f"corrupt_{period_name}"
            elif "is_complete" in frame and not frame.is_complete.eq(True).all():
                status = f"provisional_{period_name}"
            elif ("coordinate_id" not in frame and instrument != "EURUSD") or (
                    "coordinate_id" in frame and (frame.coordinate_id.isna().any() or frame.coordinate_id.nunique() != 1)):
                status = f"invalid_roll_{period_name}"
            elif "invalid_roll" in frame and frame.invalid_roll.fillna(False).astype(bool).any():
                status = f"invalid_roll_{period_name}"
        if status != "OK":
            records.append(dict(source_close_ts=end, available_at_ts=end, status=status, evidence=None))
            previous_close = state = coordinate = available = None
            history = []
            continue
        current_coordinate = str(frame.coordinate_id.iloc[0]) if "coordinate_id" in frame else None
        if current_coordinate != coordinate:
            previous_close = state = available = None
            history = []
            coordinate = current_coordinate
        high, low, close = float(frame.high.max()), float(frame.low.min()), float(frame.close.iloc[-1])
        tr = high - low if previous_close is None else max(high - low, abs(high - previous_close), abs(low - previous_close))
        state = tr if state is None else state + (tr - state) / 14
        previous_close = close
        observed = end
        if "observed_at_ts" in frame:
            if frame.observed_at_ts.isna().any():
                raise ValueError("Partial ATR arrival provenance")
            observed = max(end, *(utc(t) for t in frame.observed_at_ts))
        available = max(observed, available or observed)
        history.append(dict(hour_open=start.isoformat(), hour_close=end.isoformat(),
                            coordinate_id=coordinate, high=high, low=low, close=close,
                            observed_at=observed.isoformat(),
                            bars=[[ts.isoformat(), *[float(row[k]) for k in ("open", "high", "low", "close")]]
                                  for ts, row in frame.iterrows()]))
        evidence = None
        if len(history) >= 14 and np.isfinite(state) and state > 0:
            evidence = evidence_type(value=state, source_close_ts=end,
                history_start_ts=pd.Timestamp(history[0]["hour_open"]), available_at_ts=available,
                coordinate_id=coordinate, source_hash=content_hash(history))
        records.append(dict(source_close_ts=end, available_at_ts=available,
                            status="OK" if evidence else "warmup", evidence=evidence))
    return pd.DataFrame(records, columns=["source_close_ts", "available_at_ts", "status", "evidence"])


def structural_atr_at(hourly: pd.DataFrame, known_at, *, coordinate_id: str | None) -> StructuralATREvidence | None:
    from model_testing.outcome_engine import utc

    known = utc(known_at)
    eligible = hourly.loc[(hourly.source_close_ts <= known) & (hourly.available_at_ts <= known)]
    if eligible.empty:
        return None
    evidence = eligible.iloc[-1].evidence
    return evidence if evidence is not None and evidence.coordinate_id == coordinate_id else None
