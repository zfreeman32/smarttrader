from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frvp.calendars.macro import (  # noqa: E402
    _cpi_release_dates_for_year,
    annotate_us_macro_event_flags,
    macro_calendar_contract,
)
from frvp.sessions.equity import build_equity_session_frame  # noqa: E402


def _utc(local_value: str) -> pd.Timestamp:
    return pd.Timestamp(local_value, tz="America/New_York").tz_convert("UTC")


def test_equity_session_frame_marks_early_close_and_holiday() -> None:
    datetimes = pd.Series(
        [
            _utc("2025-07-03 12:55:00"),
            _utc("2025-07-03 13:00:00"),
            _utc("2025-07-04 10:00:00"),
        ]
    )

    session = build_equity_session_frame(datetimes, instrument="es")

    early_close_open = session.iloc[0]
    early_close_post = session.iloc[1]
    holiday_row = session.iloc[2]

    assert bool(early_close_open["equity_early_close_flag"]) is True
    assert bool(early_close_open["is_rth"]) is True
    assert bool(early_close_post["equity_early_close_flag"]) is True
    assert bool(early_close_post["is_rth"]) is False

    assert bool(holiday_row["equity_holiday_flag"]) is True
    assert bool(holiday_row["is_rth"]) is False
    assert bool(holiday_row["is_ib"]) is False


def test_equity_session_frame_preserves_timestamp_index_without_alignment_warnings() -> None:
    datetimes = pd.Series(
        [
            _utc("2025-07-03 09:30:00"),
            _utc("2025-07-03 15:55:00"),
            _utc("2025-07-04 10:00:00"),
        ],
        index=pd.DatetimeIndex(
            [
                _utc("2025-07-03 09:30:00"),
                _utc("2025-07-03 15:55:00"),
                _utc("2025-07-04 10:00:00"),
            ]
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = build_equity_session_frame(datetimes, instrument="es")

    assert session.index.equals(datetimes.index)
    assert not any("sort order is undefined" in str(w.message) for w in caught)
    assert bool(session.iloc[0]["is_rth"]) is True
    assert bool(session.iloc[2]["equity_holiday_flag"]) is True


def test_macro_event_flags_mark_known_release_windows() -> None:
    datetimes = pd.Series(
        [
            _utc("2025-06-06 08:20:00"),  # NFP window
            _utc("2025-06-11 08:35:00"),  # CPI window
            _utc("2025-06-18 14:00:00"),  # FOMC statement window
            _utc("2025-06-18 14:35:00"),  # FOMC presser window
        ]
    )

    flags = annotate_us_macro_event_flags(datetimes)

    assert bool(flags.iloc[0]["nfp_flag"]) is True
    assert bool(flags.iloc[0]["macro_event_flag"]) is True
    assert bool(flags.iloc[1]["cpi_flag"]) is True
    assert bool(flags.iloc[2]["fomc_flag"]) is True
    assert bool(flags.iloc[2]["fed_statement_flag"]) is True
    assert bool(flags.iloc[3]["fed_presser_flag"]) is True


def test_macro_calendar_contract_marks_historical_cpi_backfill_complete() -> None:
    contract = macro_calendar_contract()

    assert contract["cpi_source"] == "archived_alfred_monthly_history_plus_curated_gap_fill"
    assert bool(contract["cpi_archive_available"]) is True
    assert bool(contract["cpi_historical_backfill_complete"]) is True
    assert contract["cpi_historical_years"]["start"] == 1949
    assert contract["cpi_historical_years"]["end"] == 2026


def test_historical_cpi_flags_use_primary_monthly_release_dates() -> None:
    datetimes = pd.Series(
        [
            _utc("2017-02-13 08:35:00"),  # archive contains an extra CPI-related release date
            _utc("2017-02-15 08:35:00"),  # primary monthly CPI release
            _utc("2019-02-11 08:35:00"),  # archive contains an extra CPI-related release date
            _utc("2019-02-13 08:35:00"),  # primary monthly CPI release
        ]
    )

    flags = annotate_us_macro_event_flags(datetimes)

    assert bool(flags.iloc[0]["cpi_flag"]) is False
    assert bool(flags.iloc[1]["cpi_flag"]) is True
    assert bool(flags.iloc[2]["cpi_flag"]) is False
    assert bool(flags.iloc[3]["cpi_flag"]) is True


def test_post_2022_cpi_dates_prefer_archive_but_keep_gap_fill() -> None:
    releases_2025 = _cpi_release_dates_for_year(2025)

    assert len(releases_2025) == 12
    assert date(2025, 10, 24) in releases_2025
    assert date(2025, 11, 13) in releases_2025
    assert date(2025, 10, 15) not in releases_2025
