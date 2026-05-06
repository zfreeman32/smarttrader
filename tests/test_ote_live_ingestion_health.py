from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor


def test_gap_detector_flags_missing_duplicate_and_out_of_order_bars() -> None:
    detector = GapDetector(timeframe="1m")

    first = _bar(0)
    second = _bar(2)
    duplicate = _bar(2)
    out_of_order = _bar(1)

    initial = detector.observe(first)
    missing = detector.observe(second)
    dup = detector.observe(duplicate)
    old = detector.observe(out_of_order)

    assert not initial.has_issue
    assert missing.gap is not None
    assert missing.gap.gap_size == 1
    assert missing.gap.missing_timestamps == [datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc)]
    assert dup.duplicate is True
    assert old.out_of_order is True


def test_gap_detector_ignores_fx_weekend_market_closure() -> None:
    detector = GapDetector(asset="EURUSD", timeframe="5m")

    friday_last_bar = _bar_at(datetime(2026, 4, 10, 20, 55, tzinfo=timezone.utc), timeframe="5m")
    sunday_first_provider_bar = _bar_at(datetime(2026, 4, 12, 21, 5, tzinfo=timezone.utc), timeframe="5m")

    initial = detector.observe(friday_last_bar)
    after_weekend = detector.observe(sunday_first_provider_bar)

    assert not initial.has_issue
    assert after_weekend.gap is None


def test_gap_detector_flags_missing_after_sunday_reopen_boundary() -> None:
    detector = GapDetector(asset="EURUSD", timeframe="5m")

    friday_last_bar = _bar_at(datetime(2026, 4, 10, 20, 55, tzinfo=timezone.utc), timeframe="5m")
    sunday_later_bar = _bar_at(datetime(2026, 4, 12, 21, 10, tzinfo=timezone.utc), timeframe="5m")

    detector.observe(friday_last_bar)
    after_weekend = detector.observe(sunday_later_bar)

    assert after_weekend.gap is not None
    assert after_weekend.gap.gap_size == 1
    assert after_weekend.gap.expected_timestamp == datetime(2026, 4, 12, 21, 5, tzinfo=timezone.utc)
    assert after_weekend.gap.missing_timestamps == [datetime(2026, 4, 12, 21, 5, tzinfo=timezone.utc)]


def test_heartbeat_monitor_marks_source_stale_after_threshold() -> None:
    monitor = HeartbeatMonitor(source="fmp.polling", stale_after=timedelta(seconds=30))
    beat_at = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)

    healthy = monitor.beat(observed_at_utc=beat_at)
    stale = monitor.snapshot(now=beat_at + timedelta(seconds=31))

    assert healthy.is_stale is False
    assert healthy.lag_seconds == 0.0
    assert stale.is_stale is True
    assert stale.lag_seconds == 31.0


def _bar(minute: int) -> MarketBar:
    return _bar_at(datetime(2024, 1, 2, 10, minute, tzinfo=timezone.utc))


def _bar_at(timestamp: datetime, *, timeframe: str = "1m") -> MarketBar:
    return MarketBar(
        asset="EURUSD",
        timeframe=timeframe,
        timestamp=timestamp,
        open=1.1,
        high=1.1005,
        low=1.0995,
        close=1.1001,
        volume=10.0,
    )
