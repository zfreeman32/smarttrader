from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator, aggregate_bar_sequence
from ote_live.ingestion.base import filter_finalized_bars
from ote_live.ingestion.base import IngestionGap
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.ibkr.store import IBKRBar, IBKRMarketDataStore
from ote_live.ingestion.market_calendar import broker_schedule_market_open, is_expected_market_bar_timestamp
from ote_live.ingestion.normalizer import bars_to_dataframe, dataframe_to_market_bars
from ote_live.ingestion.provenance import assess_shadow_bar_eligibility, observe_bar
from ote_live.storage.db import SQLiteLiveDataStore


START = datetime(2026, 9, 11, 14, 0, tzinfo=UTC)
CALENDAR = {
    "ibkr_trading_hours": "20260909:1700-20260910:1600;20260910:1700-20260911:1600",
    "ibkr_timezone": "America/Chicago",
}


def _bar(start=START, *, timeframe="5m", **updates):
    step = timedelta(minutes=1 if timeframe == "1m" else 5)
    context = {**CALENDAR, **updates.pop("feature_context", {})}
    bar = MarketBar(asset="ES", timeframe=timeframe, timestamp=start,
                    open=6000, high=6002, low=5999, close=6001, volume=5,
                    is_complete=True, feed_type="live", observation_kind="live",
                    first_observed_at=start + step + timedelta(seconds=1),
                    last_observed_at=start + step + timedelta(seconds=1),
                    feature_context=context,
                    **updates)
    return observe_bar(bar)


def test_dataframe_round_trip_preserves_source_provenance_and_feature_context():
    bar = _bar(feature_context={"contract": "ESU6"})
    actual = dataframe_to_market_bars(bars_to_dataframe([bar]), asset="ES", timeframe="5m")[0]
    assert actual == bar


def test_store_retains_first_observation_and_immutable_revised_payload(tmp_path):
    first = _bar()
    revised = observe_bar(first.model_copy(update={"close": 6002, "bar_version": None,
                            "first_observed_at": first.first_observed_at + timedelta(minutes=5),
                            "last_observed_at": first.last_observed_at + timedelta(minutes=5)}))
    with SQLiteLiveDataStore(tmp_path / "bars.sqlite") as store:
        store.upsert_bar(first)
        store.upsert_bar(revised)
        store.upsert_bar(revised)
        actual = store.fetch_bars(asset="ES", timeframe="5m")[0]
        assert actual.close == 6002
        assert actual.first_observed_at == first.first_observed_at
        assert actual.bar_version != first.bar_version
        history = store.connection.execute("SELECT * FROM source_bar_history ORDER BY id").fetchall()
        assert len(history) == 2
        assert json.loads(history[0]["payload_json"])["close"] == 6001
        for statement in ["UPDATE source_bar_history SET event_type='changed'", "DELETE FROM source_bar_history"]:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                store.connection.execute(statement)


def test_legacy_bar_is_snapshotted_before_first_corrective_overwrite(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "legacy.sqlite") as store:
        store.connection.execute(
            "INSERT INTO canonical_bars (asset,timeframe,timestamp_utc,open,high,low,close,volume,inserted_at_utc,updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("ES", "5m", START.isoformat(), 5998, 6000, 5997, 5999, 3, START.isoformat(), START.isoformat()),
        )
        store.connection.commit()
        store.upsert_bar(_bar())
        history = store.connection.execute("SELECT * FROM source_bar_history ORDER BY id").fetchall()
        assert history[0]["event_type"] == "legacy_baseline"
        assert json.loads(history[0]["payload_json"])["close"] == 5999
        assert json.loads(history[0]["payload_json"])["is_complete"] is None


def test_first_provisional_receipt_survives_reconnect_before_canonical_publication(tmp_path):
    path = tmp_path / "reconnect.sqlite"
    first = observe_bar(_bar().model_copy(update={"bar_version": None, "is_complete": False, "first_observed_at": START, "last_observed_at": START}))
    with SQLiteLiveDataStore(path) as store:
        store.record_bar_observation(first)
    with SQLiteLiveDataStore(path) as store:
        store.upsert_bar(_bar())
        actual = store.fetch_bars(asset="ES", timeframe="5m")[0]
        assert actual.first_observed_at == START
        assert actual.last_observed_at == _bar().last_observed_at
        payload = json.loads(store.connection.execute("SELECT payload_json FROM source_bar_history ORDER BY id DESC LIMIT 1").fetchone()[0])
        assert datetime.fromisoformat(payload["first_observed_at"]) == START


def test_qualified_bar_uses_actual_recording_time_and_never_supplies_a_historical_fill():
    recorded = START + timedelta(minutes=5, seconds=10)
    result = assess_shadow_bar_eligibility(_bar(), recorded_at=recorded)
    assert result.eligible
    assert result.earliest_entry_at == recorded
    assert result.prediction_recorded_at == recorded
    assert result.executable_entry_price is None
    assert result.delay_seconds == 10
    assert result.to_payload()["source_bar_timestamp"] == START.isoformat()


@pytest.mark.parametrize(("updates", "elapsed", "reason", "kind"), [
    ({"is_complete": False}, 10, "bar_incomplete", "provisional"),
    ({"is_complete": None}, 10, "bar_completion_unknown", "diagnostic"),
    ({"feed_type": "delayed"}, 10, "feed_not_live", "delayed"),
    ({"observation_kind": "backfill"}, 10, "backfilled_evaluation", "backfilled"),
    ({}, 91, "bar_stale", "delayed"),
    ({"first_observed_at": None}, 10, "first_observation_unknown", "diagnostic"),
    ({"last_observed_at": None}, 10, "last_observation_unknown", "diagnostic"),
    ({"feature_context": {}}, 10, "market_calendar_unverified", "diagnostic"),
    ({"source_timestamp": START - timedelta(minutes=5)}, 10, "source_timestamp_mismatch", "diagnostic"),
    ({"last_observed_at": START + timedelta(seconds=1)}, 10, "completion_observed_before_bar_close", "diagnostic"),
])
def test_disqualified_observations_cannot_acquire_executable_entry(updates, elapsed, reason, kind):
    bar = _bar().model_copy(update=updates)
    result = assess_shadow_bar_eligibility(bar, recorded_at=START + timedelta(minutes=5, seconds=elapsed))
    assert not result.eligible
    assert reason in result.reasons
    assert result.evaluation_kind == kind
    assert result.earliest_entry_at is None
    assert result.executable_entry_price is None


def test_latency_is_checked_again_for_each_model_and_explicit_incomplete_bars_stay_incomplete():
    bar = _bar()
    assert assess_shadow_bar_eligibility(bar, recorded_at=START + timedelta(minutes=6, seconds=30)).eligible
    assert not assess_shadow_bar_eligibility(bar, recorded_at=START + timedelta(minutes=6, seconds=31)).eligible
    provisional = bar.model_copy(update={"is_complete": False})
    assert filter_finalized_bars([provisional], now=START + timedelta(days=1)) == []


@pytest.mark.parametrize("hours", [
    "20260910:1700-20260911:1600,20260911:1700-20260912:1600",
    "20260911:0000-1600,1700-2300",
    "20260911:0000-1600,1700-2300;",
])
def test_broker_schedule_supports_multiple_dated_and_legacy_intervals(hours):
    assert broker_schedule_market_open(START, trading_hours=hours, timezone="America/Chicago") is True
    assert broker_schedule_market_open(START.replace(hour=21, minute=30), trading_hours=hours, timezone="America/Chicago") is False


@pytest.mark.parametrize("context", [
    {"ibkr_trading_hours": "20260910:1700-20260911:1600;broken", "ibkr_timezone": "America/Chicago"},
    {"ibkr_trading_hours": "20260910:1700-20260911:1600", "ibkr_timezone": "bad-zone"},
    {"ibkr_trading_hours": "20260909:1700-20260910:1600", "ibkr_timezone": "America/Chicago"},
])
def test_malformed_or_expired_calendar_does_not_certify_a_fresh_bar(context):
    result = assess_shadow_bar_eligibility(_bar(feature_context=context), recorded_at=START + timedelta(minutes=5, seconds=1))
    assert not result.eligible
    assert "market_calendar_unverified" in result.reasons


def test_intrabar_halt_is_not_hidden_by_open_endpoints():
    context = {"ibkr_trading_hours": "20260911:0800-0902,0903-1600", "ibkr_timezone": "America/Chicago"}
    result = assess_shadow_bar_eligibility(_bar(feature_context=context), recorded_at=START + timedelta(minutes=5, seconds=1))
    assert "source_bar_crosses_market_closure" in result.reasons
    assert not result.eligible


def test_unverified_gap_calendar_and_invalid_sequence_fail_closed():
    bar = _bar()
    for previous, reason in [
        (START, "source_bar_sequence_invalid"),
        (START - timedelta(minutes=6), "source_bar_sequence_invalid"),
        (START - timedelta(days=7), "gap_calendar_unverified"),
    ]:
        result = assess_shadow_bar_eligibility(bar, recorded_at=START + timedelta(minutes=5, seconds=1), previous_bar_timestamp=previous)
        assert not result.eligible
        assert reason in result.reasons


@pytest.mark.parametrize(("timestamp", "hours"), [
    ("2026-03-08T22:00:00+00:00", "20260308:1700-20260309:1600"),
    ("2026-11-01T23:00:00+00:00", "20261101:1700-20261102:1600"),
])
def test_dated_broker_schedule_uses_dst_at_sunday_reopen(timestamp, hours):
    reopened = datetime.fromisoformat(timestamp)
    assert broker_schedule_market_open(reopened, trading_hours=hours, timezone="America/Chicago") is True
    assert broker_schedule_market_open(reopened - timedelta(seconds=1), trading_hours=hours, timezone="America/Chicago") is False


def test_prediction_at_early_close_cannot_enter_even_if_source_bar_was_complete():
    bar = _bar(datetime(2026, 9, 7, 16, 55, tzinfo=UTC), feature_context={
        "ibkr_trading_hours": "20260906:1700-20260907:1200;20260907:1700-20260908:1600",
    })
    result = assess_shadow_bar_eligibility(bar, recorded_at=bar.last_observed_at)
    assert "prediction_market_closed" in result.reasons
    assert not result.eligible
    assert result.earliest_entry_at is None


def test_ibkr_preserves_first_receipt_versions_and_original_delayed_feed():
    store = IBKRMarketDataStore()
    callback = IBKRBar(timestamp_utc=START, open=6000, high=6002, low=5999, close=6001,
                       volume=5, wap=None, trade_count=2, conid=123, local_symbol="ESU6",
                       contract_month="202609", bar_size="5 mins", market_data_type="delayed",
                       received_at_utc=START + timedelta(seconds=10), observation_kind="live")
    store.upsert_bar(callback)
    first = store.get_bars()[0].to_market_bar()
    store.set_market_data_type(123, "live")
    assert store.get_bars()[0].market_data_type == "delayed"
    store.upsert_bar(replace(callback, close=6002, received_at_utc=START + timedelta(seconds=20)))
    revised = store.get_bars()[0].to_market_bar()
    assert first.bar_version != revised.bar_version
    assert revised.first_observed_at == first.first_observed_at
    store.upsert_bar(replace(callback, timestamp_utc=START + timedelta(minutes=5), received_at_utc=START + timedelta(minutes=5, seconds=2)))
    completed = store.get_bars()[0].to_market_bar()
    assert completed.is_complete
    assert completed.bar_version != revised.bar_version
    assert completed.first_observed_at == first.first_observed_at


def test_partial_callbacks_are_persisted_even_when_snapshot_publication_is_throttled(tmp_path):
    from ote_live.ingestion.ibkr.service import SQLiteIBKRSnapshotSink
    path = tmp_path / "callbacks.sqlite"
    sink = SQLiteIBKRSnapshotSink(path, minimum_write_interval_seconds=1000)
    store = IBKRMarketDataStore()
    store.add_listener(sink)
    callback = IBKRBar(timestamp_utc=START, open=6000, high=6002, low=5999, close=6001,
                       volume=5, wap=None, trade_count=2, conid=123, local_symbol="ESU6",
                       contract_month="202609", bar_size="5 mins", market_data_type="live",
                       received_at_utc=START, observation_kind="live")
    store.upsert_bar(callback)
    store.upsert_bar(replace(callback, close=6002, received_at_utc=START + timedelta(seconds=1)))
    sink.close()
    with SQLiteLiveDataStore(path) as db:
        assert db.connection.execute("SELECT COUNT(*) FROM source_bar_history").fetchone()[0] == 2
        assert not db.fetch_bars(asset="ES", timeframe="5m")


def test_out_of_order_history_end_keeps_newest_source_bucket_provisional():
    store = IBKRMarketDataStore()
    callback = IBKRBar(timestamp_utc=START, open=6000, high=6002, low=5999, close=6001,
                       volume=5, wap=None, trade_count=2, conid=123, local_symbol="ESU6",
                       contract_month="202609", bar_size="5 mins", received_at_utc=START)
    store.upsert_bar(replace(callback, timestamp_utc=START + timedelta(minutes=5)))
    store.upsert_bar(callback)
    store.mark_request_history_complete(123, observed_at_utc=START + timedelta(minutes=7))
    older, newest = store.get_bars()
    assert older.is_complete
    assert not newest.is_complete
    assert older.first_observed_at_utc == START
    assert older.received_at_utc == START + timedelta(minutes=7)


def test_aggregation_requires_complete_source_coverage_and_retains_es_on_flush():
    sources = [_bar(START + timedelta(minutes=i), timeframe="1m") for i in range(5)]
    complete = aggregate_bar_sequence(sources, target_timeframe="5m")[0]
    assert complete.is_complete
    assert complete.feed_type == "live"
    assert len(complete.feature_context["source_bars"]) == 5
    partial = aggregate_bar_sequence(sources[:2] + sources[3:], target_timeframe="5m")[0]
    assert partial.is_complete is False
    assert partial.bar_version != complete.bar_version
    revised_sources = sources[:4] + [sources[4].model_copy(update={"is_complete": False})]
    assert aggregate_bar_sequence(revised_sources, target_timeframe="5m")[0].is_complete is False
    aggregator = MultiTimeframeBarAggregator(target_timeframes=("5m",))
    aggregator.ingest_bar(sources[0])
    flushed = aggregator.flush()[0]
    assert flushed.asset == "ES"
    assert flushed.is_complete is False


def test_market_calendar_handles_dst_weekend_and_labor_day_boundaries():
    for timestamp, expected in [
        ("2026-03-08T21:59:00+00:00", False), ("2026-03-08T22:00:00+00:00", True),
        ("2026-11-01T22:59:00+00:00", False), ("2026-11-01T23:00:00+00:00", True),
        ("2026-09-07T16:55:00+00:00", True), ("2026-09-07T17:00:00+00:00", False),
        ("2026-09-07T22:00:00+00:00", True),
    ]:
        assert is_expected_market_bar_timestamp(datetime.fromisoformat(timestamp), asset="ES") is expected
    detector = GapDetector(asset="ES", timeframe="5m")
    detector.observe(_bar(datetime(2026, 9, 7, 16, 55, tzinfo=UTC)))
    assert detector.observe(_bar(datetime(2026, 9, 7, 22, 0, tzinfo=UTC))).gap is None


def test_dated_broker_holiday_calendar_and_gap_gate():
    context = {"ibkr_trading_hours": "20261125:1700-20261126:1200;20261126:1700-20261127:1215", "ibkr_timezone": "America/Chicago"}
    assert not is_expected_market_bar_timestamp(datetime(2026, 11, 26, 18, 30, tzinfo=UTC), asset="ES", feature_context=context)
    assert is_expected_market_bar_timestamp(datetime(2026, 11, 26, 23, 0, tzinfo=UTC), asset="ES", feature_context=context)
    assert broker_schedule_market_open(START, trading_hours=context["ibkr_trading_hours"], timezone=context["ibkr_timezone"]) is None
    result = assess_shadow_bar_eligibility(_bar(), recorded_at=START + timedelta(minutes=5, seconds=10),
                                           previous_bar=_bar(START - timedelta(minutes=10)))
    assert "source_bar_gap" in result.reasons
    after_break = _bar(datetime(2026, 9, 10, 22, 0, tzinfo=UTC))
    assert assess_shadow_bar_eligibility(after_break, recorded_at=after_break.timestamp + timedelta(minutes=5, seconds=10),
                                        previous_bar=_bar(datetime(2026, 9, 10, 20, 55, tzinfo=UTC))).eligible


@pytest.mark.parametrize("invalid", [{"is_complete": False}, {"is_complete": None}, {"asset": "NQ"}, {"timeframe": "1m"}])
def test_reconciliation_retains_unchanged_live_classification_and_rejects_invalid_gap_recovery(tmp_path, invalid):
    from ote_live.ingestion.service import LiveBarIngestionService
    from ote_live.ingestion.heartbeat import HeartbeatMonitor
    bar = _bar()
    with SQLiteLiveDataStore(tmp_path / "reconciliation.sqlite") as db:
        db.upsert_bar(bar)
        service = LiveBarIngestionService(
            stream=None, backfill=None, store=db, gap_detector=GapDetector(asset="ES", timeframe="5m"),
            aggregator=MultiTimeframeBarAggregator(target_timeframes=()), heartbeat_monitor=HeartbeatMonitor(source="test"),
        )
        service.repair_source_bars([bar])
        unchanged = db.fetch_bars(asset="ES", timeframe="5m")[0]
        assert unchanged.observation_kind == "live"
        corrected = observe_bar(bar.model_copy(update={"close": 6002, "bar_version": None}))
        service.repair_source_bars([corrected])
        assert db.fetch_bars(asset="ES", timeframe="5m")[0].observation_kind == "repair"
        gap = IngestionGap(asset="ES", timeframe="5m", expected_timestamp=START,
                           observed_timestamp=START + timedelta(minutes=5), missing_timestamps=[START], gap_size=1)
        invalid_bar = bar.model_copy(update=invalid)
        assert not service._has_full_gap_coverage(gap, (invalid_bar,))
        with pytest.raises(RuntimeError, match="Incomplete gap recovery"):
            service._partition_gap_recovery_bars(
                gap=gap, recovered_bars=[invalid_bar],
                existing_bars=(), component="test", gap_id=1,
            )


@pytest.mark.parametrize(("updates", "previous_minutes", "expected_reason"), [
    ({}, 5, None),
    ({}, 10, "source_bar_gap"),
    ({"observation_kind": "backfill"}, 5, "backfilled_evaluation"),
    ({"observation_kind": "replay"}, 5, "backfilled_evaluation"),
    ({"feed_type": "delayed"}, 5, "feed_not_live"),
    ({"is_complete": False}, 5, "bar_incomplete"),
])
def test_runtime_records_each_model_after_inference_and_preserves_frozen_gap_context(
    tmp_path, monkeypatch, updates, previous_minutes, expected_reason,
):
    from ote_live.contracts.prediction import ModelPrediction
    from ote_live.features.incremental_engine import IncrementalFeatureEngine
    from ote_live.features.input_contract import INPUT_CONTRACT_VERSION
    from ote_live.features.manifest import LiveRuntimeManifest
    from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
    from ote_live.models.loaders import LoadedRuntimeModel
    from ote_live.storage import LiveAuditRepository

    path = ROOT / "ote_live/runtime_manifests/frvp_es_shadow_20260715/frvp_short_meta_xgb_v1/live_runtime_manifest.json"
    original = LiveRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    manifests = [original.model_copy(update={
        "model_id": f"a4_test_{index}",
        "feature_manifest": original.feature_manifest.model_copy(update={
            "selected_feature_names": ["close"], "selected_feature_count": 1,
            "lag_steps": [0], "input_contract_version": INPUT_CONTRACT_VERSION,
        }),
        "context_requirements": original.context_requirements.model_copy(update={
            "window_size": 1, "minimum_runtime_history_bars": 1,
        }),
    }) for index in range(2)]
    engine = IncrementalFeatureEngine(manifests)
    monkeypatch.setattr(engine.builder, "build", lambda frame: (
        frame.assign(**{name: 0. for name in engine.plan.built_feature_names if name not in frame}), {},
    ))
    bindings = [SignalRuntimeModelBinding(loaded_model=LoadedRuntimeModel(
        manifest=manifest, model=None, scaler=None, calibrator=None,
        training_summary={}, scale_clip=0., batch_size=1, use_amp=False,
    ), shadow_mode=True) for manifest in manifests]
    clock = {"now": START + timedelta(minutes=5, seconds=1)}
    monkeypatch.setattr("ote_live.ingestion.signals.utc_now", lambda: clock["now"])
    # SQLite's actual insertion time is separate and slightly later.
    monkeypatch.setattr("ote_live.storage.repositories.utc_now", lambda: clock["now"] + timedelta(milliseconds=20))
    bar = _bar().model_copy(update=updates)
    previous = _bar(START - timedelta(minutes=previous_minutes))
    with SQLiteLiveDataStore(tmp_path / "runtime.sqlite") as store:
        audit = LiveAuditRepository(store)
        processor = LiveSignalProcessor(bindings=bindings, audit_repository=audit, feature_engine=engine,
                                        enable_ict_paper_signal_ledger=False, force_shadow_mode=True)
        engine.ingest_bar(previous)
        processor.ingest_bar_for_evaluation(bar)
        prepared = processor.prepare_ingested_bar_snapshot(bar)
        # Later mutable state must not conceal a gap in the evaluated snapshot.
        engine.state.clear()
        engine.state.extend([_bar(START - timedelta(minutes=5)), bar])
        for index, manifest in enumerate(manifests):
            def predict(frame, *, model=manifest, elapsed=(10 if index == 0 else 91), **kwargs):
                clock["now"] = START + timedelta(minutes=5, seconds=elapsed)
                return ModelPrediction(model_id=model.model_id, direction=model.direction,
                    backend="xgboost", calibrated_probability=0.9, raw_score=0.9, **kwargs)
            monkeypatch.setattr(processor._runner_by_model_id[manifest.model_id], "predict_latest", predict)
        results = processor.publish_prepared_snapshot(prepared, emit_operator_artifacts=False, emit_notifications=False)
        assert len(results) == 2
        for index, result in enumerate(results):
            trail = audit.reconstruct_signal(result.signal_decision_id, include_bars=False)
            metadata = trail.feature_snapshot.observation_metadata
            contract = metadata["bar_eligibility"]
            expected_time = START + timedelta(minutes=5, seconds=10 if index == 0 else 91)
            assert trail.prediction.timestamp == START
            assert trail.prediction.prediction_recorded_at_utc == expected_time
            assert contract["prediction_recorded_at"] == expected_time.isoformat()
            assert contract["previous_bar_timestamp"] == previous.timestamp.isoformat()
            assert contract["source_bar_version"] == bar.bar_version
            assert contract["first_observed_at"] == bar.first_observed_at.isoformat()
            if expected_reason:
                assert expected_reason in contract["reasons"]
            if index == 1:
                assert "bar_stale" in contract["reasons"]
            assert contract["eligible"] == (index == 0 and expected_reason is None)
            assert contract["earliest_entry_at"] == (expected_time.isoformat() if contract["eligible"] else None)
            assert metadata["source_bar"]["close"] == bar.close
            assert not metadata["qualified_shadow_entry"]
            assert metadata["executable_entry_at"] is None
            assert metadata["executable_entry_price"] is None
            row = store.connection.execute("SELECT recorded_at_utc FROM model_predictions WHERE model_id=?", (result.model_id,)).fetchone()
            assert datetime.fromisoformat(row[0]) == expected_time + timedelta(milliseconds=20)
