from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.runtime_state import FeatureRuntimeState
from ote_live.ingestion.setup_revisions import (
    initialize_setup_revision_cursor, mark_setup_evaluated, reconcile_setup_revisions,
)
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.setup_events import SetupEventRepository


NOW = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)


def _bar(offset, close=101, version="v1"):
    return MarketBar(asset="ES", timeframe="5m", timestamp=NOW + timedelta(minutes=offset),
                     open=100, high=102, low=98, close=close, volume=50,
                     bar_version=version, is_complete=True, feed_type="recorded", observation_kind="backfill")


def _processor(store):
    bars = store.fetch_bars(asset="ES", timeframe="5m")
    state = FeatureRuntimeState(max_history_bars=20)
    state.extend(bars)
    prefixes = []

    def build(frame, **kwargs):
        prefixes.append(frame.copy())
        return pd.DataFrame({"example_feature": frame["close"]})

    return SimpleNamespace(
        audit_repository=SimpleNamespace(store=store), asset="ES", timeframe="5m", group_name="FRVP",
        collection_version="v1", setup_event_repository=SetupEventRepository(store),
        last_processed_timestamp=bars[-1].timestamp if bars else None,
        feature_engine=SimpleNamespace(state=state, build_feature_frame=build),
        prefixes=prefixes,
        _build_policy_frame_from_market_frame=lambda features, market_frame: market_frame.copy(),
    )


def _detector(frame, **kwargs):
    if frame.iloc[-1]["close"] <= 100 and (len(frame) == 1 or frame.iloc[0]["close"] >= 100):
        return ()
    return ({"setup_type": "3", "setup_side": 1, "selected": True, "confidence": 0.7,
             "geometry": {"anchor_level": float(frame.iloc[0]["close"])}},)


def _collect(processor, bar):
    prefix = processor.audit_repository.store.fetch_bars(asset="ES", timeframe="5m", end=bar.timestamp)
    processor.setup_event_repository.collect_from_frame(
        IncrementalFeatureEngine.market_frame_from_bars(prefix), bar=bar,
        collection_version=processor.collection_version, strategies=("FRVP",), observed_at=NOW,
    )
    mark_setup_evaluated(processor, bar)


def test_correction_replays_causal_prefixes_including_prior_zero_fire_and_resumes_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", _detector)
    with SQLiteLiveDataStore(tmp_path / "source.sqlite") as store:
        bars = [_bar(0), _bar(5, close=99), _bar(10)]
        store.upsert_bars(bars)
        processor = _processor(store)
        initialize_setup_revision_cursor(processor)
        for bar in bars:
            _collect(processor, bar)
        first, second = processor.setup_event_repository.list_events()
        assert first.source_timestamp == NOW
        assert second.source_timestamp == NOW + timedelta(minutes=10)
        store.upsert_bar(_bar(0, close=99, version="v2"))
        assert reconcile_setup_revisions(processor, max_evaluations=1) == 1
        assert processor.setup_event_repository.history(first.event_key)[-1].event_kind == "invalidated"
        assert processor.feature_engine.state.to_frame().iloc[0]["close"] == 99
        restarted = _processor(store)
        assert reconcile_setup_revisions(restarted) == 2
        assert [frame.iloc[-1]["timestamp"] for frame in restarted.prefixes] == [
            NOW + timedelta(minutes=5), NOW + timedelta(minutes=10)]
        events = restarted.setup_event_repository.list_events(latest_only=True)
        new_event = next(event for event in events if event.source_timestamp == NOW + timedelta(minutes=5))
        assert new_event.event_kind == "observed"
        assert new_event.observed_at > new_event.source_timestamp
        assert restarted.setup_event_repository.history(second.event_key)[-1].event_kind == "revised"
        assert reconcile_setup_revisions(restarted) == 0
        assert store.connection.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == 0
        assert store.connection.execute("SELECT COUNT(*) FROM signal_decisions").fetchone()[0] == 0


def test_successfully_evaluated_ordinary_bar_is_not_replayed_even_when_no_setup_fired(tmp_path, monkeypatch):
    monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", _detector)
    with SQLiteLiveDataStore(tmp_path / "source.sqlite") as store:
        store.upsert_bar(_bar(0))
        processor = _processor(store)
        initialize_setup_revision_cursor(processor)
        ordinary = _bar(5, close=99)
        store.upsert_bar(ordinary)
        _collect(processor, ordinary)
        processor.last_processed_timestamp = ordinary.timestamp
        assert reconcile_setup_revisions(processor) == 0
        assert processor.prefixes == []


def test_new_earlier_backfilled_bar_is_observed_at_actual_reconciliation_time(tmp_path, monkeypatch):
    monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", _detector)
    with SQLiteLiveDataStore(tmp_path / "source.sqlite") as store:
        store.upsert_bars([_bar(0), _bar(10)])
        processor = _processor(store)
        initialize_setup_revision_cursor(processor)
        for bar in store.fetch_bars(asset="ES", timeframe="5m"):
            _collect(processor, bar)
        store.upsert_bar(_bar(5))
        assert reconcile_setup_revisions(processor) == 2
        backfilled = next(event for event in processor.setup_event_repository.list_events()
                          if event.source_timestamp == NOW + timedelta(minutes=5))
        assert backfilled.observed_at > backfilled.source_timestamp
        assert processor.setup_event_repository.prediction_links(backfilled.event_id) == ()


def test_unpublished_source_revision_waits_for_canonical_update_without_blocking_other_history(tmp_path, monkeypatch):
    monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", _detector)
    with SQLiteLiveDataStore(tmp_path / "source.sqlite") as store:
        store.upsert_bars([_bar(0), _bar(5)])
        processor = _processor(store)
        initialize_setup_revision_cursor(processor)
        revised = _bar(0, close=99, version="v2")
        store.record_bar_observation(revised)
        assert reconcile_setup_revisions(processor) == 0
        store.upsert_bar(revised)
        assert reconcile_setup_revisions(processor) == 2


def test_failed_revision_retries_after_database_reopen_without_duplicate_events(tmp_path, monkeypatch):
    path = tmp_path / "retry.sqlite"
    monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", _detector)
    with SQLiteLiveDataStore(path) as store:
        store.upsert_bars([_bar(0), _bar(5), _bar(10)])
        processor = _processor(store)
        initialize_setup_revision_cursor(processor)
        for bar in store.fetch_bars(asset="ES", timeframe="5m"):
            _collect(processor, bar)
        store.upsert_bar(_bar(0, close=99, version="v2"))

        def failing(frame, **kwargs):
            if frame.iloc[-1]["timestamp"] == NOW + timedelta(minutes=5):
                raise ValueError("producer unavailable")
            return _detector(frame, **kwargs)

        monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", failing)
        with pytest.raises(ValueError, match="producer unavailable"):
            reconcile_setup_revisions(processor)
        assert len(processor.setup_event_repository.list_events()) == 4
    with SQLiteLiveDataStore(path) as reopened:
        processor = _processor(reopened)
        monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", _detector)
        assert reconcile_setup_revisions(processor) == 2
        assert len(processor.setup_event_repository.list_events()) == 6
        assert reconcile_setup_revisions(processor) == 0
