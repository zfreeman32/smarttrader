from __future__ import annotations

import numpy as np
import pandas as pd

from features.transforms import safe_divide

from ..common import get_atr_like, resolve_event_time, resolve_instrument, session_phase_codes_from_frame
from ..sessions.equity import build_ict_equity_session_frame


def _empty_output(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index)


def build_reference_level_features(
    df: pd.DataFrame,
    config: object,
) -> pd.DataFrame:
    """Build ES-specific reference levels used by ICT liquidity logic."""

    out = _empty_output(df.index)
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return out

    event_time = resolve_event_time(df)
    if event_time is None:
        return out

    instrument = resolve_instrument(config)
    source_timezone = getattr(config, "source_timezone", "UTC")
    canonical_timezone = getattr(config, "canonical_timezone", "UTC")

    session = build_ict_equity_session_frame(
        event_time,
        instrument=instrument,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    atr = get_atr_like(df)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df.get("volume", pd.Series(np.nan, index=df.index)), errors="coerce")

    working = pd.DataFrame(index=df.index)
    working["session_date"] = session["session_date"]
    working["is_rth"] = session["is_rth"].astype(bool)
    working["is_overnight"] = session["is_overnight"].astype(bool)
    working["is_ib"] = session["is_ib"].astype(bool)
    working["ib_complete"] = session["ib_complete"].astype(bool)
    working["equity_datetime"] = session["equity_datetime"]
    working["minutes_since_rth_open"] = pd.to_numeric(session["minutes_since_rth_open"], errors="coerce")
    working["minutes_until_rth_close"] = pd.to_numeric(session["minutes_until_rth_close"], errors="coerce")
    working["open"] = open_
    working["high"] = high
    working["low"] = low
    working["close"] = close
    working["volume"] = volume

    session_phase_code = session_phase_codes_from_frame(session)
    out["ict_session_phase_code"] = session_phase_code
    out["ict_is_rth"] = working["is_rth"].astype(np.int8)
    out["ict_is_overnight"] = working["is_overnight"].astype(np.int8)
    out["ict_is_ib"] = working["is_ib"].astype(np.int8)
    out["ict_ib_complete"] = working["ib_complete"].astype(np.int8)
    out["ict_open_drive_flag"] = (
        working["is_rth"] & working["minutes_since_rth_open"].between(0, 30, inclusive="left")
    ).astype(np.int8)
    local_dt = pd.to_datetime(working["equity_datetime"], errors="coerce")
    local_minutes = (local_dt.dt.hour * 60) + local_dt.dt.minute
    out["ict_lunch_lull_flag"] = local_minutes.between(12 * 60, 13 * 60 + 30, inclusive="left").astype(np.int8)
    out["ict_close_ramp_flag"] = (
        working["is_rth"] & working["minutes_until_rth_close"].le(60)
    ).astype(np.int8)

    rth_rows = working.loc[working["is_rth"]].copy()
    if not rth_rows.empty:
        session_rth = (
            rth_rows.groupby("session_date")
            .agg(
                rth_high=("high", "max"),
                rth_low=("low", "min"),
                rth_close=("close", "last"),
                rth_open=("open", "first"),
            )
            .sort_index()
        )
        prior_rth = session_rth.shift(1)
        out["ict_prior_rth_high"] = working["session_date"].map(prior_rth["rth_high"])
        out["ict_prior_rth_low"] = working["session_date"].map(prior_rth["rth_low"])
        out["ict_prior_rth_close"] = working["session_date"].map(prior_rth["rth_close"])
        out["ict_rth_open"] = working["session_date"].map(session_rth["rth_open"])

        week_key = pd.to_datetime(session_rth.index).to_series(index=session_rth.index).dt.to_period("W-FRI")
        weekly = (
            session_rth.assign(week_key=week_key.values)
            .groupby("week_key")
            .agg(week_high=("rth_high", "max"), week_low=("rth_low", "min"))
            .sort_index()
        )
        prior_weekly = weekly.shift(1)
        session_week_map = week_key.to_dict()
        out["ict_prior_week_high"] = working["session_date"].map(
            lambda value: prior_weekly["week_high"].get(session_week_map.get(value))
        )
        out["ict_prior_week_low"] = working["session_date"].map(
            lambda value: prior_weekly["week_low"].get(session_week_map.get(value))
        )
    else:
        out["ict_prior_rth_high"] = np.nan
        out["ict_prior_rth_low"] = np.nan
        out["ict_prior_rth_close"] = np.nan
        out["ict_rth_open"] = np.nan
        out["ict_prior_week_high"] = np.nan
        out["ict_prior_week_low"] = np.nan

    overnight_high_dev = working["high"].where(working["is_overnight"]).groupby(working["session_date"]).cummax()
    overnight_low_dev = working["low"].where(working["is_overnight"]).groupby(working["session_date"]).cummin()
    overnight_final_high = (
        working.loc[working["is_overnight"]].groupby("session_date")["high"].max() if working["is_overnight"].any() else pd.Series(dtype=float)
    )
    overnight_final_low = (
        working.loc[working["is_overnight"]].groupby("session_date")["low"].min() if working["is_overnight"].any() else pd.Series(dtype=float)
    )
    out["ict_overnight_high"] = overnight_high_dev.where(
        working["is_overnight"],
        working["session_date"].map(overnight_final_high),
    )
    out["ict_overnight_low"] = overnight_low_dev.where(
        working["is_overnight"],
        working["session_date"].map(overnight_final_low),
    )

    ib_high_dev = working["high"].where(working["is_ib"]).groupby(working["session_date"]).cummax()
    ib_low_dev = working["low"].where(working["is_ib"]).groupby(working["session_date"]).cummin()
    ib_final_high = (
        working.loc[working["is_ib"]].groupby("session_date")["high"].max() if working["is_ib"].any() else pd.Series(dtype=float)
    )
    ib_final_low = (
        working.loc[working["is_ib"]].groupby("session_date")["low"].min() if working["is_ib"].any() else pd.Series(dtype=float)
    )
    out["ict_ib_high"] = working["session_date"].map(ib_final_high).where(working["ib_complete"])
    out["ict_ib_low"] = working["session_date"].map(ib_final_low).where(working["ib_complete"])
    out["ict_ib_high_developing"] = ib_high_dev
    out["ict_ib_low_developing"] = ib_low_dev

    local_date = local_dt.dt.tz_localize(None).dt.normalize()
    out["ict_midnight_open"] = local_date.map(
        working.loc[local_minutes.eq(0)].groupby(local_date[local_minutes.eq(0)])["open"].first()
    )
    out["ict_open_0830"] = local_date.map(
        working.loc[local_minutes.eq(8 * 60 + 30)].groupby(local_date[local_minutes.eq(8 * 60 + 30)])["open"].first()
    )

    typical_price = (working["high"] + working["low"] + working["close"]) / 3.0
    if volume.notna().any():
        session_vwap_num = (typical_price * volume.fillna(0.0)).groupby(working["session_date"]).cumsum()
        session_vwap_den = volume.fillna(0.0).groupby(working["session_date"]).cumsum().replace(0, np.nan)
        out["ict_session_vwap"] = session_vwap_num / session_vwap_den

        rth_volume = volume.where(working["is_rth"]).fillna(0.0)
        rth_num = (typical_price * rth_volume).groupby(working["session_date"]).cumsum()
        rth_den = rth_volume.groupby(working["session_date"]).cumsum().replace(0, np.nan)
        out["ict_rth_vwap"] = (rth_num / rth_den).where(working["is_rth"]).groupby(working["session_date"]).ffill()
    else:
        out["ict_session_vwap"] = np.nan
        out["ict_rth_vwap"] = np.nan

    out["ict_rth_gap_open"] = out["ict_rth_open"] - out["ict_prior_rth_close"]

    reference_level_columns = [
        "ict_prior_rth_high",
        "ict_prior_rth_low",
        "ict_prior_rth_close",
        "ict_overnight_high",
        "ict_overnight_low",
        "ict_rth_open",
        "ict_ib_high",
        "ict_ib_low",
        "ict_prior_week_high",
        "ict_prior_week_low",
        "ict_midnight_open",
        "ict_open_0830",
        "ict_session_vwap",
        "ict_rth_vwap",
    ]
    for column in reference_level_columns:
        out[f"{column}_dist_atr"] = safe_divide(pd.to_numeric(out[column], errors="coerce") - close, atr)

    return out
