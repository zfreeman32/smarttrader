from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from features.fx_calendar import normalize_datetime_series, resolve_timezone


_NY_TZ = resolve_timezone("America/New_York")
_CPI_WINDOW = (8 * 60 + 15, 9 * 60)
_NFP_WINDOW = (8 * 60 + 15, 9 * 60)
_FOMC_WINDOW = (13 * 60 + 45, 15 * 60 + 15)
_STATEMENT_WINDOW = (13 * 60 + 55, 14 * 60 + 10)
_PRESSER_WINDOW = (14 * 60 + 25, 15 * 60 + 15)
_CPI_ARCHIVE_PATH = Path(__file__).resolve().parents[2] / "data" / "futures_data" / "CPI_release_dates.txt"


def macro_calendar_contract() -> dict[str, object]:
    archive_years = sorted(_monthly_cpi_release_dates_from_archive())
    training_window_years = range(2015, 2024)
    historical_backfill_complete = all(
        len(_cpi_release_dates_for_year(year)) == 12
        for year in training_window_years
    )
    return {
        "timezone": "America/New_York",
        "fomc_source": "curated_fed_schedule_and_statement_dates",
        "nfp_source": "rule_based_bls_employment_situation_release_schedule",
        "cpi_source": "archived_alfred_monthly_history_plus_curated_gap_fill",
        "cpi_archive_path": str(Path("data") / "futures_data" / "CPI_release_dates.txt"),
        "cpi_archive_available": _CPI_ARCHIVE_PATH.exists(),
        "cpi_historical_backfill_complete": historical_backfill_complete,
        "cpi_historical_years": {
            "start": archive_years[0] if archive_years else None,
            "end": archive_years[-1] if archive_years else None,
        },
        "cpi_historical_validation_window": {
            "start": min(training_window_years),
            "end": max(training_window_years),
        },
        "cpi_curated_recent_years": sorted(_CURATED_CPI_RELEASES),
        "window_minutes": {
            "cpi": {"start": _CPI_WINDOW[0], "end": _CPI_WINDOW[1]},
            "nfp": {"start": _NFP_WINDOW[0], "end": _NFP_WINDOW[1]},
            "fomc": {"start": _FOMC_WINDOW[0], "end": _FOMC_WINDOW[1]},
            "statement": {"start": _STATEMENT_WINDOW[0], "end": _STATEMENT_WINDOW[1]},
            "presser": {"start": _PRESSER_WINDOW[0], "end": _PRESSER_WINDOW[1]},
        },
        "caveats": [
            "Archived CPI dates are collapsed to the latest extracted release date in each calendar month and are preferred whenever the offline export contains that month.",
            "Curated recent monthly CPI dates remain in-code as a gap-fill only for archive-missing months so the runtime schedule stays one-date-per-month and network-free.",
            "FOMC dates are curated offline so the runtime enrichment stays network-free.",
        ],
    }


def annotate_us_macro_event_flags(
    datetime_values: Iterable[object],
    *,
    source_timezone: str = "UTC",
    canonical_timezone: str = "UTC",
) -> pd.DataFrame:
    datetime_utc = normalize_datetime_series(
        datetime_values,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    local = datetime_utc.dt.tz_convert(_NY_TZ)
    local_dates = local.dt.tz_localize(None).dt.normalize()
    minutes = (local.dt.hour * 60) + local.dt.minute

    unique_years = sorted({int(value.year) for value in local_dates.dropna().tolist()})
    cpi_dates = {pd.Timestamp(value).normalize() for year in unique_years for value in _cpi_release_dates_for_year(year)}
    nfp_dates = {pd.Timestamp(value).normalize() for year in unique_years for value in _nfp_release_dates_for_year(year)}
    fomc_dates = {pd.Timestamp(value).normalize() for year in unique_years for value in _fomc_dates_for_year(year)}
    presser_dates = {pd.Timestamp(value).normalize() for year in unique_years for value in _fomc_presser_dates_for_year(year)}
    statement_dates = set(fomc_dates)

    cpi_flag = local_dates.isin(cpi_dates) & minutes.between(_CPI_WINDOW[0], _CPI_WINDOW[1], inclusive="both")
    nfp_flag = local_dates.isin(nfp_dates) & minutes.between(_NFP_WINDOW[0], _NFP_WINDOW[1], inclusive="both")
    fomc_flag = local_dates.isin(fomc_dates) & minutes.between(_FOMC_WINDOW[0], _FOMC_WINDOW[1], inclusive="both")
    statement_flag = local_dates.isin(statement_dates) & minutes.between(
        _STATEMENT_WINDOW[0],
        _STATEMENT_WINDOW[1],
        inclusive="both",
    )
    presser_flag = local_dates.isin(presser_dates) & minutes.between(
        _PRESSER_WINDOW[0],
        _PRESSER_WINDOW[1],
        inclusive="both",
    )
    major_flag = cpi_flag | nfp_flag | fomc_flag | statement_flag | presser_flag

    return pd.DataFrame(
        {
            "cpi_flag": cpi_flag.astype(bool),
            "nfp_flag": nfp_flag.astype(bool),
            "fomc_flag": fomc_flag.astype(bool),
            "fed_statement_flag": statement_flag.astype(bool),
            "fed_presser_flag": presser_flag.astype(bool),
            "macro_event_flag": major_flag.astype(bool),
            "major_macro_event_flag": major_flag.astype(bool),
        },
        index=datetime_utc.index,
    )


def _business_day_on_or_after(year: int, month: int, day: int) -> date:
    current = date(year, month, day)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _first_friday(year: int, month: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != 4:
        current += timedelta(days=1)
    return current


def _next_friday(current: date) -> date:
    while current.weekday() != 4:
        current += timedelta(days=1)
    return current


def _previous_business_day(current: date) -> date:
    current -= timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _us_federal_holidays(year: int) -> frozenset[date]:
    from .equity import _holiday_dates_for_year

    return _holiday_dates_for_year(year)


def _nfp_release_dates_for_year(year: int) -> tuple[date, ...]:
    dates: list[date] = []
    for month in range(1, 13):
        release = _first_friday(year, month)
        holidays = _us_federal_holidays(release.year)
        if release in holidays:
            if release.month == 1 and release.day <= 2:
                release = _next_friday(release + timedelta(days=1))
            else:
                release = _previous_business_day(release)
        dates.append(_NFP_OVERRIDES.get((year, month), release))
    return tuple(dates)


def _cpi_release_dates_for_year(year: int) -> tuple[date, ...]:
    archive_dates = _monthly_cpi_release_date_map_from_archive().get(year, {})
    curated_dates = {
        release.month: release
        for release in _CURATED_CPI_RELEASES.get(year, ())
    }
    rule_based_dates = {
        release.month: release
        for release in _rule_based_cpi_release_dates_for_year(year)
    }

    resolved: list[date] = []
    for month in range(1, 13):
        resolved_date = archive_dates.get(month)
        if resolved_date is None:
            resolved_date = curated_dates.get(month, rule_based_dates[month])
        resolved.append(resolved_date)
    return tuple(resolved)


@lru_cache(maxsize=1)
def _archived_cpi_release_dates() -> tuple[date, ...]:
    if not _CPI_ARCHIVE_PATH.exists():
        return ()

    releases: list[date] = []
    for line in _CPI_ARCHIVE_PATH.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if len(value) != 10:
            continue
        try:
            releases.append(date.fromisoformat(value))
        except ValueError:
            continue
    return tuple(releases)


@lru_cache(maxsize=1)
def _monthly_cpi_release_dates_from_archive() -> dict[int, tuple[date, ...]]:
    monthly_map = _monthly_cpi_release_date_map_from_archive()
    return {
        year: tuple(monthly_map[year][month] for month in sorted(monthly_map[year]))
        for year in sorted(monthly_map)
    }


@lru_cache(maxsize=1)
def _monthly_cpi_release_date_map_from_archive() -> dict[int, dict[int, date]]:
    monthly_latest: dict[tuple[int, int], date] = {}
    for release_date in _archived_cpi_release_dates():
        key = (release_date.year, release_date.month)
        current = monthly_latest.get(key)
        if current is None or release_date > current:
            monthly_latest[key] = release_date

    grouped: dict[int, dict[int, date]] = {}
    for (year, month), release_date in sorted(monthly_latest.items()):
        grouped.setdefault(year, {})[month] = release_date
    return grouped


@lru_cache(maxsize=1)
def _historical_cpi_release_dates_by_year() -> dict[int, tuple[date, ...]]:
    return dict(_monthly_cpi_release_dates_from_archive())


def _rule_based_cpi_release_dates_for_year(year: int) -> tuple[date, ...]:
    dates: list[date] = []
    for month in range(1, 13):
        candidate = _business_day_on_or_after(year, month, 10)
        while candidate in _us_federal_holidays(candidate.year):
            candidate += timedelta(days=1)
        dates.append(candidate)
    return tuple(dates)


def _fomc_dates_for_year(year: int) -> tuple[date, ...]:
    return tuple(_FOMC_STATEMENT_DATES.get(year, ()))


def _fomc_presser_dates_for_year(year: int) -> tuple[date, ...]:
    return tuple(_FOMC_PRESSER_DATES.get(year, ()))


_FOMC_STATEMENT_DATES: dict[int, tuple[date, ...]] = {
    2017: (
        date(2017, 2, 1),
        date(2017, 3, 15),
        date(2017, 5, 3),
        date(2017, 6, 14),
        date(2017, 7, 26),
        date(2017, 9, 20),
        date(2017, 11, 1),
        date(2017, 12, 13),
    ),
    2018: (
        date(2018, 1, 31),
        date(2018, 3, 21),
        date(2018, 5, 2),
        date(2018, 6, 13),
        date(2018, 8, 1),
        date(2018, 9, 26),
        date(2018, 11, 8),
        date(2018, 12, 19),
    ),
    2019: (
        date(2019, 1, 30),
        date(2019, 3, 20),
        date(2019, 5, 1),
        date(2019, 6, 19),
        date(2019, 7, 31),
        date(2019, 9, 18),
        date(2019, 10, 30),
        date(2019, 12, 11),
    ),
    2020: (
        date(2020, 1, 29),
        date(2020, 3, 3),
        date(2020, 3, 15),
        date(2020, 4, 29),
        date(2020, 6, 10),
        date(2020, 7, 29),
        date(2020, 9, 16),
        date(2020, 11, 5),
        date(2020, 12, 16),
    ),
    2021: (
        date(2021, 1, 27),
        date(2021, 3, 17),
        date(2021, 4, 28),
        date(2021, 6, 16),
        date(2021, 7, 28),
        date(2021, 9, 22),
        date(2021, 11, 3),
        date(2021, 12, 15),
    ),
    2022: (
        date(2022, 1, 26),
        date(2022, 3, 16),
        date(2022, 5, 4),
        date(2022, 6, 15),
        date(2022, 7, 27),
        date(2022, 9, 21),
        date(2022, 11, 2),
        date(2022, 12, 14),
    ),
    2023: (
        date(2023, 2, 1),
        date(2023, 3, 22),
        date(2023, 5, 3),
        date(2023, 6, 14),
        date(2023, 7, 26),
        date(2023, 9, 20),
        date(2023, 11, 1),
        date(2023, 12, 13),
    ),
    2024: (
        date(2024, 1, 31),
        date(2024, 3, 20),
        date(2024, 5, 1),
        date(2024, 6, 12),
        date(2024, 7, 31),
        date(2024, 9, 18),
        date(2024, 11, 7),
        date(2024, 12, 18),
    ),
    2025: (
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
    ),
    2026: (
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ),
}

_FOMC_PRESSER_DATES: dict[int, tuple[date, ...]] = {
    2017: (
        date(2017, 3, 15),
        date(2017, 6, 14),
        date(2017, 9, 20),
        date(2017, 12, 13),
    ),
    2018: (
        date(2018, 3, 21),
        date(2018, 6, 13),
        date(2018, 9, 26),
        date(2018, 12, 19),
    ),
    2019: _FOMC_STATEMENT_DATES[2019],
    2020: (
        date(2020, 1, 29),
        date(2020, 3, 15),
        date(2020, 4, 29),
        date(2020, 6, 10),
        date(2020, 7, 29),
        date(2020, 9, 16),
        date(2020, 11, 5),
        date(2020, 12, 16),
    ),
    2021: _FOMC_STATEMENT_DATES[2021],
    2022: _FOMC_STATEMENT_DATES[2022],
    2023: _FOMC_STATEMENT_DATES[2023],
    2024: _FOMC_STATEMENT_DATES[2024],
    2025: _FOMC_STATEMENT_DATES[2025],
    2026: _FOMC_STATEMENT_DATES[2026],
}

_NFP_OVERRIDES: dict[tuple[int, int], date] = {
    (2020, 1): date(2020, 1, 10),
    (2021, 1): date(2021, 1, 8),
    (2025, 7): date(2025, 7, 3),
}

_CURATED_CPI_RELEASES: dict[int, tuple[date, ...]] = {
    2024: (
        date(2024, 1, 11),
        date(2024, 2, 13),
        date(2024, 3, 12),
        date(2024, 4, 10),
        date(2024, 5, 15),
        date(2024, 6, 12),
        date(2024, 7, 11),
        date(2024, 8, 14),
        date(2024, 9, 11),
        date(2024, 10, 10),
        date(2024, 11, 13),
        date(2024, 12, 11),
    ),
    2025: (
        date(2025, 1, 15),
        date(2025, 2, 12),
        date(2025, 3, 12),
        date(2025, 4, 10),
        date(2025, 5, 13),
        date(2025, 6, 11),
        date(2025, 7, 15),
        date(2025, 8, 12),
        date(2025, 9, 11),
        date(2025, 10, 15),
        date(2025, 11, 13),
        date(2025, 12, 10),
    ),
    2026: (
        date(2026, 1, 14),
        date(2026, 2, 18),
        date(2026, 3, 12),
        date(2026, 4, 10),
        date(2026, 5, 12),
        date(2026, 6, 10),
        date(2026, 7, 15),
        date(2026, 8, 12),
        date(2026, 9, 11),
        date(2026, 10, 15),
        date(2026, 11, 13),
        date(2026, 12, 10),
    ),
}
