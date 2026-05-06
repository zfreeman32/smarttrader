from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

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


def is_forex_asset(asset: str) -> bool:
    symbol = canonical_asset_symbol(asset)
    if len(symbol) != 6 or not symbol.isalpha():
        return False
    return symbol[:3] in _FX_CURRENCY_CODES and symbol[3:] in _FX_CURRENCY_CODES


def is_expected_market_bar_timestamp(
    timestamp: datetime,
    *,
    asset: str,
) -> bool:
    if is_forex_asset(asset):
        return is_forex_market_open(timestamp)
    return True


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
) -> list[datetime]:
    return [
        ensure_utc(timestamp)
        for timestamp in timestamps
        if is_expected_market_bar_timestamp(timestamp, asset=asset)
    ]
