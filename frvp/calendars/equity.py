from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

import pandas as pd


EARLY_CLOSE_HOUR = 13
EARLY_CLOSE_MINUTE = 0


@dataclass(frozen=True)
class EquitySessionCalendarEntry:
    session_date: pd.Timestamp
    equity_holiday_flag: bool
    equity_half_day_flag: bool
    equity_early_close_flag: bool
    rth_end_hour: int
    rth_end_minute: int


def session_calendar_overrides(session_dates: Iterable[object]) -> pd.DataFrame:
    raw_values = session_dates.copy() if isinstance(session_dates, pd.Series) else pd.Series(session_dates)
    values = pd.to_datetime(raw_values, errors="coerce").dt.normalize()
    if values.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "equity_holiday_flag",
                "equity_half_day_flag",
                "equity_early_close_flag",
                "rth_end_hour",
                "rth_end_minute",
            ]
        )

    valid = values.dropna()
    if valid.empty:
        return pd.DataFrame(
            {
                "session_date": values,
                "equity_holiday_flag": False,
                "equity_half_day_flag": False,
                "equity_early_close_flag": False,
                "rth_end_hour": 16,
                "rth_end_minute": 0,
            }
        )

    calendar = build_us_equity_session_calendar(valid.min(), valid.max())
    merged = pd.DataFrame({"session_date": values}).merge(calendar, on="session_date", how="left")
    merged.index = values.index
    merged["equity_holiday_flag"] = merged["equity_holiday_flag"].fillna(False).astype(bool)
    merged["equity_half_day_flag"] = merged["equity_half_day_flag"].fillna(False).astype(bool)
    merged["equity_early_close_flag"] = merged["equity_early_close_flag"].fillna(False).astype(bool)
    merged["rth_end_hour"] = merged["rth_end_hour"].fillna(16).astype(int)
    merged["rth_end_minute"] = merged["rth_end_minute"].fillna(0).astype(int)
    return merged


def build_us_equity_session_calendar(
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Session calendar bounds must be valid dates.")
    if end < start:
        start, end = end, start

    rows: list[dict[str, object]] = []
    for day in pd.date_range(start, end, freq="D"):
        normalized = pd.Timestamp(day).normalize()
        holiday = _is_us_equity_holiday(normalized.date())
        early_close = _is_us_equity_early_close(normalized.date())
        rows.append(
            {
                "session_date": normalized,
                "equity_holiday_flag": bool(holiday),
                "equity_half_day_flag": bool(early_close),
                "equity_early_close_flag": bool(early_close),
                "rth_end_hour": EARLY_CLOSE_HOUR if early_close else 16,
                "rth_end_minute": EARLY_CLOSE_MINUTE if early_close else 0,
            }
        )
    return pd.DataFrame(rows)


def _is_us_equity_holiday(value: date) -> bool:
    return value in _holiday_dates_for_year(value.year)


def _is_us_equity_early_close(value: date) -> bool:
    return value in _early_close_dates_for_year(value.year)


@lru_cache(maxsize=None)
def _holiday_dates_for_year(year: int) -> frozenset[date]:
    holidays = {
        _observed_fixed_holiday(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),   # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),   # Presidents Day
        _good_friday(year),
        _last_weekday(year, 5, 0),     # Memorial Day
        _observed_fixed_holiday(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed_fixed_holiday(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(date(year, 6, 19)))
    return frozenset(holidays)


@lru_cache(maxsize=None)
def _early_close_dates_for_year(year: int) -> frozenset[date]:
    dates: set[date] = set()

    independence = date(year, 7, 4)
    if independence.weekday() in {1, 2, 3, 4}:
        candidate = independence - timedelta(days=1)
        if candidate.weekday() < 5 and candidate not in _holiday_dates_for_year(year):
            dates.add(candidate)

    black_friday = _nth_weekday(year, 11, 4, 4) + timedelta(days=1)
    if black_friday.weekday() < 5:
        dates.add(black_friday)

    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in _holiday_dates_for_year(year):
        dates.add(christmas_eve)

    return frozenset(dates)


def _observed_fixed_holiday(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + timedelta(days=delta + (occurrence - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _good_friday(year: int) -> date:
    easter = _western_easter(year)
    return easter - timedelta(days=2)


def _western_easter(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
