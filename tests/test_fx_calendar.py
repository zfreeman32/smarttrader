from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.fx_calendar import build_intraday_calendar_frame, build_market_day_close_labels


def test_intraday_calendar_uses_named_timezones_for_session_overlap() -> None:
    datetime_utc = pd.Series(pd.to_datetime(["2024-03-15 12:30:00"], utc=True))

    calendar = build_intraday_calendar_frame(datetime_utc)

    assert int(calendar["in_london_session"].iloc[0]) == 1
    assert int(calendar["in_newyork_session"].iloc[0]) == 1
    assert int(calendar["in_london_ny_overlap"].iloc[0]) == 1
    assert calendar["session_regime"].iloc[0] == "overlap"


def test_market_day_close_labels_roll_at_new_york_1700_close() -> None:
    datetime_utc = pd.Series(
        pd.to_datetime(
            [
                "2024-01-02 21:55:00+00:00",
                "2024-01-02 22:05:00+00:00",
            ]
        )
    )

    close_labels = build_market_day_close_labels(datetime_utc)

    assert str(close_labels.iloc[0]) == "2024-01-02 22:00:00+00:00"
    assert str(close_labels.iloc[1]) == "2024-01-03 22:00:00+00:00"
