from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import HeartbeatStatus, IngestionGap
from ote_live.storage.db import SQLiteLiveDataStore


def test_sqlite_live_data_store_round_trips_bars_and_events() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_storage_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    with SQLiteLiveDataStore(db_path) as store:
        bar = MarketBar(
            asset="EURUSD",
            timeframe="5m",
            timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
            open=1.1010,
            high=1.1015,
            low=1.1008,
            close=1.1012,
            volume=42.0,
            source="unit-test",
        )
        store.upsert_bar(bar)
        store.record_raw_event(
            provider="fmp",
            event_type="time_series",
            payload={"ok": True},
            asset="EURUSD",
            timeframe="1m",
            event_timestamp_utc=bar.timestamp,
        )
        store.record_gap(
            IngestionGap(
                asset="EURUSD",
                timeframe="1m",
                expected_timestamp=datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc),
                observed_timestamp=datetime(2024, 1, 2, 10, 3, tzinfo=timezone.utc),
                missing_timestamps=[
                    datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, 10, 2, tzinfo=timezone.utc),
                ],
                gap_size=2,
            )
        )
        unresolved_gaps = store.fetch_gaps(asset="EURUSD", timeframe="1m", unresolved_only=True)
        store.mark_gap_resolved(unresolved_gaps[0].gap_id)
        resolved_gaps = store.fetch_gaps(asset="EURUSD", timeframe="1m", unresolved_only=False)
        store.record_heartbeat(
            HeartbeatStatus(
                source="fmp.polling",
                observed_at_utc=bar.timestamp,
                stale_after_seconds=30.0,
                lag_seconds=0.0,
                is_stale=False,
            ),
            metadata={"asset": "EURUSD"},
        )
        store.upsert_runtime_state(
            scope="feature_seed_offsets",
            state_key="EURUSD:5m:test",
            payload={"latest_timestamp": bar.timestamp.isoformat(), "feature_seed_offsets": {"foo": 1.5}},
        )
        runtime_state = store.get_runtime_state(scope="feature_seed_offsets", state_key="EURUSD:5m:test")
        store.delete_runtime_state(scope="feature_seed_offsets", state_key="EURUSD:5m:test")
        deleted_runtime_state = store.get_runtime_state(scope="feature_seed_offsets", state_key="EURUSD:5m:test")

        bars = store.fetch_bars(asset="EURUSD", timeframe="5m")

    assert len(bars) == 1
    assert bars[0].timestamp == bar.timestamp
    assert bars[0].close == bar.close
    assert runtime_state == {
        "feature_seed_offsets": {"foo": 1.5},
        "latest_timestamp": bar.timestamp.isoformat(),
    }
    assert len(unresolved_gaps) == 1
    assert unresolved_gaps[0].gap_size == 2
    assert unresolved_gaps[0].resolved_at_utc is None
    assert len(resolved_gaps) == 1
    assert resolved_gaps[0].resolved_at_utc is not None
    assert deleted_runtime_state is None
