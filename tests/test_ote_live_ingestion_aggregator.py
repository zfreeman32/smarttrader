from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator, aggregate_bar_sequence
from ote_live.ingestion.normalizer import load_local_csv_bars


def test_incremental_aggregator_finalizes_five_minute_bar_when_bucket_rolls() -> None:
    aggregator = MultiTimeframeBarAggregator(target_timeframes=("5m",))
    bars = [
        _bar(minute, open_price, high, low, close, volume)
        for minute, open_price, high, low, close, volume in [
            (0, 1.1000, 1.1002, 1.0998, 1.1001, 10),
            (1, 1.1001, 1.1003, 1.1000, 1.1002, 11),
            (2, 1.1002, 1.1005, 1.1001, 1.1004, 9),
            (3, 1.1004, 1.1006, 1.1003, 1.1005, 12),
            (4, 1.1005, 1.1007, 1.1002, 1.1003, 8),
            (5, 1.1003, 1.1008, 1.1001, 1.1007, 15),
        ]
    ]

    finalized = []
    for bar in bars:
        finalized.extend(aggregator.ingest_bar(bar))

    assert len(finalized) == 1
    five_minute_bar = finalized[0]
    assert five_minute_bar.timeframe == "5m"
    assert five_minute_bar.timestamp == datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    assert five_minute_bar.open == pytest.approx(1.1000)
    assert five_minute_bar.high == pytest.approx(1.1007)
    assert five_minute_bar.low == pytest.approx(1.0998)
    assert five_minute_bar.close == pytest.approx(1.1003)
    assert five_minute_bar.volume == pytest.approx(50.0)


def test_one_minute_replay_matches_canonical_five_minute_reference_window() -> None:
    reference_five_minute = load_local_csv_bars(
        ROOT / "data" / "currency_data" / "eurusd-5m.csv",
        timeframe="5m",
        source_timezone="UTC",
        canonical_timezone="UTC",
        nrows=100,
    )
    start_timestamp = reference_five_minute[0].timestamp
    lookup_text = start_timestamp.strftime("%d/%m/%Y;%H:%M:%S")
    skiprows = _find_line_number(
        ROOT / "data" / "currency_data" / "eurusd-1m.csv",
        lookup_text,
    ) - 1

    one_minute_bars = load_local_csv_bars(
        ROOT / "data" / "currency_data" / "eurusd-1m.csv",
        timeframe="1m",
        source_timezone="UTC",
        canonical_timezone="UTC",
        skiprows=skiprows,
        nrows=(len(reference_five_minute) * 5) + 10,
    )
    end_timestamp = reference_five_minute[-1].timestamp + timedelta(minutes=5)
    one_minute_window = [bar for bar in one_minute_bars if start_timestamp <= bar.timestamp < end_timestamp]

    aggregated = aggregate_bar_sequence(one_minute_window, target_timeframe="5m")

    assert len(aggregated) >= len(reference_five_minute)
    for actual, expected in zip(aggregated[: len(reference_five_minute)], reference_five_minute):
        assert actual.timestamp == expected.timestamp
        assert actual.open == pytest.approx(expected.open)
        assert actual.high == pytest.approx(expected.high)
        assert actual.low == pytest.approx(expected.low)
        assert actual.close == pytest.approx(expected.close)
        assert actual.volume == pytest.approx(expected.volume)


def _bar(
    minute: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> MarketBar:
    return MarketBar(
        asset="EURUSD",
        timeframe="1m",
        timestamp=datetime(2024, 1, 2, 10, minute, tzinfo=timezone.utc),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _find_line_number(path: Path, needle: str) -> int:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith(needle):
                return line_number
    raise AssertionError(f"Could not locate {needle!r} in {path}")
