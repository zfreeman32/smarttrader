from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ict.structure.liquidity import build_reference_level_features


CONFIG = SimpleNamespace(instrument="es", source_timezone="UTC", canonical_timezone="UTC")


def bars(times: list[str | None]) -> pd.DataFrame:
    # Deliberately preserve non-default labels to catch positional alignment bugs.
    close = 5000. + np.arange(len(times)) * 3
    return pd.DataFrame({
        "datetime": pd.to_datetime(times, utc=True), "open": close - 1,
        "high": close + 2, "low": close - 2, "close": close,
        "volume": 100., "atr_14": 4.,
    }, index=pd.Index(np.arange(len(times)) * 7 + 10, name="source_row"))


@pytest.mark.parametrize("times", [
    # Post-close, midnight, 08:30, RTH open and initial-balance completion.
    ["2026-09-10 13:30Z", "2026-09-10 19:55Z", "2026-09-10 20:00Z",
     "2026-09-11 03:55Z", "2026-09-11 04:00Z", "2026-09-11 12:25Z",
     "2026-09-11 12:30Z", "2026-09-11 13:25Z", "2026-09-11 13:30Z",
     "2026-09-11 14:25Z", "2026-09-11 14:30Z", "2026-09-11 20:00Z"],
    # Missing entire sessions/weeks and missing the next RTH opening bar.
    ["2026-08-28 13:30Z", "2026-08-28 19:55Z", "2026-08-30 22:00Z",
     "2026-09-14 12:00Z", "2026-09-14 13:35Z", "2026-09-14 19:55Z"],
    # Cold start: future history must never supply absent prior references.
    ["2026-09-10 12:00Z", "2026-09-10 13:35Z", "2026-09-10 19:55Z",
     "2026-09-11 12:00Z", "2026-09-11 13:30Z"],
    # DST in both directions, plus the year/week boundary.
    ["2026-03-06 14:30Z", "2026-03-06 20:55Z", "2026-03-08 22:00Z",
     "2026-03-09 13:25Z", "2026-03-09 13:30Z"],
    ["2026-10-30 13:30Z", "2026-10-30 19:55Z", "2026-11-01 23:00Z",
     "2026-11-02 14:25Z", "2026-11-02 14:30Z"],
    ["2025-12-31 14:30Z", "2025-12-31 20:55Z", "2026-01-01 00:00Z",
     "2026-01-02 14:30Z", "2026-01-02 20:55Z", "2026-01-05 13:00Z"],
])
def test_entire_reference_surface_is_unchanged_by_future_bars(times: list[str]) -> None:
    frame = bars(times)
    full = build_reference_level_features(frame, CONFIG)
    for count in range(1, len(frame) + 1):
        pd.testing.assert_frame_equal(
            build_reference_level_features(frame.iloc[:count], CONFIG), full.iloc[:count],
        )


def test_known_references_survive_overnight_and_missing_weeks() -> None:
    frame = bars(["2026-08-28 13:30Z", "2026-08-28 19:55Z", "2026-08-28 20:00Z",
                  "2026-08-30 22:00Z", "2026-09-14 12:00Z", "2026-09-14 13:30Z"])
    result = build_reference_level_features(frame, CONFIG).reset_index(drop=True)
    assert result.loc[2:, "ict_prior_rth_high"].eq(5005.).all()
    assert result.loc[2:, "ict_prior_rth_low"].eq(4998.).all()
    assert result.loc[2:, "ict_prior_rth_close"].eq(5003.).all()
    assert result.loc[2:, "ict_prior_week_high"].eq(5005.).all()
    assert result.loc[2:, "ict_prior_week_low"].eq(4998.).all()
    assert result.loc[2:4, ["ict_rth_open", "ict_rth_gap_open"]].isna().all().all()
    assert result.loc[5, "ict_rth_open"] == 5014.
    assert result.loc[5, "ict_rth_gap_open"] == 11.
    assert result.loc[5, "ict_prior_rth_high_dist_atr"] == (5005. - 5015.) / 4.


def test_missing_opening_bar_is_not_replaced_by_later_open_or_next_session() -> None:
    frame = bars(["2026-09-10 12:35Z", "2026-09-10 13:35Z", "2026-09-10 19:55Z",
                  "2026-09-10 20:00Z", "2026-09-11 13:30Z"])
    result = build_reference_level_features(frame, CONFIG).reset_index(drop=True)
    assert result.loc[:3, ["ict_rth_open", "ict_rth_gap_open", "ict_open_0830",
                           "ict_midnight_open"]].isna().all().all()
    assert result.loc[:2, ["ict_prior_rth_high", "ict_prior_week_high"]].isna().all().all()
    assert result.loc[4, "ict_rth_open"] == 5011.
    assert result.loc[4, "ict_prior_rth_close"] == 5006.


def test_unknown_timestamp_cannot_select_future_reference_aggregates() -> None:
    frame = bars(["2026-09-10 13:30Z", None, "2026-09-11 13:30Z"])
    full = build_reference_level_features(frame, CONFIG)
    columns = ["ict_prior_rth_high", "ict_prior_rth_low", "ict_prior_rth_close",
               "ict_prior_week_high", "ict_prior_week_low"]
    assert full.iloc[1][columns].isna().all()
    pd.testing.assert_frame_equal(
        build_reference_level_features(frame.iloc[:2], CONFIG)[columns], full.iloc[:2][columns],
    )
