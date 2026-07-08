from __future__ import annotations

import pandas as pd

from features.fx_calendar import build_market_day_close_labels, normalize_datetime_series, resolve_timezone

from frvp.calendars.equity import session_calendar_overrides
from frvp.config.instruments import InstrumentConfig, get_instrument_config


DEFAULT_MARKET_CLOSE_TIMEZONE = "America/New_York"
DEFAULT_MARKET_CLOSE_HOUR = 17
DEFAULT_MARKET_CLOSE_MINUTE = 0


def build_equity_market_day_labels(
    datetime_values,
    *,
    source_timezone: str = "UTC",
    canonical_timezone: str = "UTC",
    market_close_timezone: str = DEFAULT_MARKET_CLOSE_TIMEZONE,
    market_close_hour: int = DEFAULT_MARKET_CLOSE_HOUR,
    market_close_minute: int = DEFAULT_MARKET_CLOSE_MINUTE,
) -> pd.Series:
    """Mirror the fx_calendar market-day pattern for futures continuity.

    Design references:
    - Section 4: daily causal lead assignment and roll handling
    - Section 8.1: equity session calendar requirement
    """

    return build_market_day_close_labels(
        datetime_values,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
        market_close_timezone=market_close_timezone,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
    )


def build_equity_session_frame(*args, **kwargs) -> pd.DataFrame:
    """Build RTH/overnight/IB boundaries for FRVP anchors and features.

    Timestamps are treated as left-labeled intraday bars, which matches the
    tagged futures files already used by the repo. A 09:30 timestamp is the
    first RTH bar, and 16:00 is the first post-RTH overnight bar.
    """

    datetime_values = args[0] if args else kwargs.pop("datetime_values")
    instrument = kwargs.pop("instrument", "es")
    source_timezone = kwargs.pop("source_timezone", "UTC")
    canonical_timezone = kwargs.pop("canonical_timezone", "UTC")
    feature_clock_timezone = kwargs.pop("feature_clock_timezone", None)
    rth_timezone = kwargs.pop("rth_timezone", None)
    rth_start_hour = kwargs.pop("rth_start_hour", None)
    rth_start_minute = kwargs.pop("rth_start_minute", None)
    rth_end_hour = kwargs.pop("rth_end_hour", None)
    rth_end_minute = kwargs.pop("rth_end_minute", None)
    overnight_start_hour = kwargs.pop("overnight_start_hour", None)
    overnight_start_minute = kwargs.pop("overnight_start_minute", None)
    ib_minutes = kwargs.pop("ib_minutes", None)
    market_close_timezone = kwargs.pop("market_close_timezone", None)
    market_close_hour = kwargs.pop("market_close_hour", None)
    market_close_minute = kwargs.pop("market_close_minute", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    config = get_instrument_config(instrument)
    feature_clock_timezone = feature_clock_timezone or config.rth_timezone
    rth_timezone = rth_timezone or config.rth_timezone
    rth_start_hour = config.rth_start_hour if rth_start_hour is None else int(rth_start_hour)
    rth_start_minute = config.rth_start_minute if rth_start_minute is None else int(rth_start_minute)
    rth_end_hour = config.rth_end_hour if rth_end_hour is None else int(rth_end_hour)
    rth_end_minute = config.rth_end_minute if rth_end_minute is None else int(rth_end_minute)
    overnight_start_hour = (
        config.overnight_start_hour if overnight_start_hour is None else int(overnight_start_hour)
    )
    overnight_start_minute = (
        config.overnight_start_minute if overnight_start_minute is None else int(overnight_start_minute)
    )
    ib_minutes = config.ib_minutes if ib_minutes is None else int(ib_minutes)
    market_close_timezone = market_close_timezone or config.market_close_timezone
    market_close_hour = config.market_close_hour if market_close_hour is None else int(market_close_hour)
    market_close_minute = config.market_close_minute if market_close_minute is None else int(market_close_minute)

    datetime_utc = normalize_datetime_series(
        datetime_values,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    feature_clock = datetime_utc.dt.tz_convert(resolve_timezone(feature_clock_timezone))
    rth_clock = datetime_utc.dt.tz_convert(resolve_timezone(rth_timezone))
    minutes = (rth_clock.dt.hour * 60) + rth_clock.dt.minute

    rth_start_total = (rth_start_hour * 60) + rth_start_minute
    rth_end_total = (rth_end_hour * 60) + rth_end_minute
    overnight_start_total = (overnight_start_hour * 60) + overnight_start_minute
    rth_start_offset = pd.to_timedelta(rth_start_total, unit="m")
    overnight_start_offset = pd.to_timedelta(overnight_start_total, unit="m")
    ib_offset = pd.to_timedelta(ib_minutes, unit="m")

    session_date_local = rth_clock.dt.normalize()
    session_date_local = session_date_local.where(minutes < overnight_start_total, session_date_local + pd.Timedelta(days=1))
    session_date = session_date_local.dt.tz_localize(None).dt.normalize()
    overrides = session_calendar_overrides(session_date)
    rth_end_total_series = (
        (overrides["rth_end_hour"].astype(int) * 60) + overrides["rth_end_minute"].astype(int)
    )
    rth_end_offset = pd.to_timedelta(rth_end_total_series.to_numpy(dtype="int64"), unit="m")
    holiday_mask = overrides["equity_holiday_flag"].fillna(False).astype(bool)
    rth_start_local = session_date_local + rth_start_offset
    rth_end_local = session_date_local + rth_end_offset
    overnight_start_local = session_date_local - pd.Timedelta(days=1) + overnight_start_offset
    overnight_end_local = rth_start_local
    ib_start_local = rth_start_local
    ib_end_local = (rth_start_local + ib_offset).where(~holiday_mask, rth_start_local)

    is_rth = (~holiday_mask) & (rth_clock >= rth_start_local) & (rth_clock < rth_end_local)
    is_overnight = (rth_clock >= overnight_start_local) & (rth_clock < overnight_end_local)
    is_ib = is_rth & (rth_clock < ib_end_local)
    ib_complete = is_rth & (rth_clock >= ib_end_local)

    out = pd.DataFrame(index=datetime_utc.index)
    out["datetime_utc"] = datetime_utc
    out["equity_datetime"] = feature_clock
    out["session_date"] = session_date
    out["market_day_close"] = build_equity_market_day_labels(
        datetime_utc,
        source_timezone=canonical_timezone,
        canonical_timezone=canonical_timezone,
        market_close_timezone=market_close_timezone,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
    )
    out["is_rth"] = is_rth.astype(bool)
    out["is_overnight"] = is_overnight.astype(bool)
    out["is_ib"] = is_ib.astype(bool)
    out["ib_complete"] = ib_complete.astype(bool)
    out["rth_start"] = rth_start_local.dt.tz_convert(resolve_timezone(canonical_timezone))
    out["rth_end"] = rth_end_local.dt.tz_convert(resolve_timezone(canonical_timezone))
    out["overnight_start"] = overnight_start_local.dt.tz_convert(resolve_timezone(canonical_timezone))
    out["overnight_end"] = overnight_end_local.dt.tz_convert(resolve_timezone(canonical_timezone))
    out["ib_start"] = ib_start_local.dt.tz_convert(resolve_timezone(canonical_timezone))
    out["ib_end"] = ib_end_local.dt.tz_convert(resolve_timezone(canonical_timezone))
    out["minutes_since_rth_open"] = ((rth_clock - rth_start_local).dt.total_seconds() / 60.0).where(is_rth)
    out["minutes_until_rth_close"] = ((rth_end_local - rth_clock).dt.total_seconds() / 60.0).where(is_rth)
    out["equity_holiday_flag"] = holiday_mask.to_numpy(dtype=bool, copy=False)
    out["equity_half_day_flag"] = overrides["equity_half_day_flag"].fillna(False).astype(bool).to_numpy(copy=False)
    out["equity_early_close_flag"] = overrides["equity_early_close_flag"].fillna(False).astype(bool).to_numpy(copy=False)
    return out
