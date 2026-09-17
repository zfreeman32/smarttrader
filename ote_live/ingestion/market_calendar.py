from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ote_live.ingestion.base import canonical_asset_symbol, ensure_utc


_NEW_YORK_TZ = ZoneInfo("America/New_York")
_FX_CURRENCY_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "CNH",
    "EUR",
    "GBP",
    "JPY",
    "MXN",
    "NOK",
    "NZD",
    "SEK",
    "USD",
    "ZAR",
}
_ES_SPECIAL_CLOSURES_ET = {
    # CME Globex equity-index futures Labor Day halt:
    # Monday 2026-09-07 12:00-17:00 CT, expressed in New York wall time.
    date(2026, 9, 7): ((13 * 60, 18 * 60),),
}
MARKET_CALENDAR_VERSION = "es-dated-broker-hours-v2+weekly-diagnostic-fallback"
MARKET_CALENDAR_SOURCE = "https://www.cmegroup.com/trading-hours.html"


def is_forex_asset(asset: str) -> bool:
    symbol = canonical_asset_symbol(asset)
    if len(symbol) != 6 or not symbol.isalpha():
        return False
    return symbol[:3] in _FX_CURRENCY_CODES and symbol[3:] in _FX_CURRENCY_CODES


def is_expected_market_bar_timestamp(
    timestamp: datetime,
    *,
    asset: str,
    feature_context: dict[str, Any] | None = None,
) -> bool:
    if canonical_asset_symbol(asset) == "ES":
        context = feature_context or {}
        reported = broker_schedule_market_open(
            timestamp, trading_hours=context.get("ibkr_trading_hours"),
            timezone=context.get("ibkr_timezone"),
        )
        if reported is not None:
            return reported
        return is_es_futures_market_open(timestamp)
    if is_forex_asset(asset):
        return is_forex_market_open(timestamp)
    return True


def broker_schedule_market_open(timestamp: datetime, *, trading_hours: str | None, timezone: str | None) -> bool | None:
    """Use dated broker holiday hours only where the received schedule has coverage.

    IBKR encodes each dated session as start-end, possibly overnight and with
    comma-separated intervals. Missing/stale schedules fall back to the versioned
    weekly calendar; they are never treated as proof of a holiday closure.
    """
    if not isinstance(trading_hours, str) or not trading_hours or not isinstance(timezone, str) or not timezone:
        return None
    try:
        local = ensure_utc(timestamp).astimezone(ZoneInfo(timezone)).replace(tzinfo=None)
    except (ValueError, ZoneInfoNotFoundError):
        return None
    covered = False
    opened = False
    for segment in trading_hours.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        try:
            day, intervals = segment.split(":", 1)
            session_day = datetime.strptime(day, "%Y%m%d").date()
            if intervals.upper() == "CLOSED":
                covered = covered or local.date() == session_day
                continue
            for interval in intervals.split(","):
                start_text, end_text = interval.split("-", 1)
                start = datetime.strptime(start_text if ":" in start_text else f"{day}:{start_text}", "%Y%m%d:%H%M")
                end = datetime.strptime(end_text if ":" in end_text else f"{day}:{end_text}", "%Y%m%d:%H%M")
                if end <= start and ":" not in end_text:
                    end += timedelta(days=1)
                if end <= start:
                    return None
                covered = covered or start.date() <= local.date() <= end.date()
                if start <= local < end:
                    opened = True
        except (ValueError, TypeError):
            # A partially parsed schedule cannot attest either openings or gaps.
            return None
    return opened if covered else None


def is_es_futures_market_open(timestamp: datetime) -> bool:
    """Return whether an ES bar may open at this timestamp."""

    local = ensure_utc(timestamp).astimezone(_NEW_YORK_TZ)
    weekday = local.weekday()
    minutes = (local.hour * 60) + local.minute
    maintenance_start = 17 * 60
    session_reopen = 18 * 60

    if _is_es_special_closure(local):
        return False
    if weekday == 5:
        return False
    if weekday == 6:
        return minutes >= session_reopen
    if weekday == 4 and minutes >= maintenance_start:
        return False
    return not (maintenance_start <= minutes < session_reopen)


def _is_es_special_closure(local_timestamp: datetime) -> bool:
    minutes = (local_timestamp.hour * 60) + local_timestamp.minute
    for start_minute, end_minute in _ES_SPECIAL_CLOSURES_ET.get(local_timestamp.date(), ()):
        if start_minute <= minutes < end_minute:
            return True
    return False


def is_forex_market_open(timestamp: datetime) -> bool:
    local = ensure_utc(timestamp).astimezone(_NEW_YORK_TZ)
    weekday = local.weekday()
    minutes = (local.hour * 60) + local.minute
    weekly_boundary_minutes = 17 * 60

    if weekday == 6:
        return minutes > weekly_boundary_minutes
    if 0 <= weekday <= 3:
        return True
    if weekday == 4:
        return minutes < weekly_boundary_minutes
    return False


def filter_expected_market_bar_timestamps(
    timestamps: list[datetime] | tuple[datetime, ...] | set[datetime],
    *,
    asset: str,
    feature_context: dict[str, Any] | None = None,
) -> list[datetime]:
    return [
        ensure_utc(timestamp)
        for timestamp in timestamps
        if is_expected_market_bar_timestamp(timestamp, asset=asset, feature_context=feature_context)
    ]
