"""A5 exercises real processors and SQLite with deterministic detector/scorer inputs."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.ingestion.signals import LiveSignalProcessor, MultiGroupLiveSignalProcessor, SignalRuntimeModelBinding
from ote_live.models.loaders import LoadedRuntimeModel
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore


ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)


def _bar(offset, *, close=101., version="v1"):
    return MarketBar(asset="ES", timeframe="5m", timestamp=START + timedelta(minutes=5 * offset),
                     open=100., high=102., low=98., close=close, volume=50,
                     is_complete=True, feed_type="recorded", observation_kind="backfill", bar_version=version)


def _detector(frame, *, strategy):
    if frame.iloc[-1]["close"] < 100:
        return ()
    return ({"setup_type": "3" if strategy == "FRVP" else "premium_discount_continuation",
             "setup_side": 1, "selected": False, "confidence": 0.73,
             "rejection_reasons": ["already_active"], "geometry": {"anchor_level": 100.}},)


def _processor(audit, strategy, monkeypatch, *, model_id=None):
    path = ROOT / "ote_live/runtime_manifests/frvp_es_shadow_20260715/frvp_short_meta_xgb_v1/live_runtime_manifest.json"
    original = LiveRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    manifest = original.model_copy(update={
        "model_id": model_id or f"{strategy.lower()}_a5_test",
        "feature_manifest": original.feature_manifest.model_copy(update={
            "selected_feature_names": ["close"], "selected_feature_count": 1, "lag_steps": [0],
        }),
        "context_requirements": original.context_requirements.model_copy(update={
            "window_size": 1, "minimum_runtime_history_bars": 1,
        }),
    })
    engine = IncrementalFeatureEngine([manifest])
    monkeypatch.setattr(engine.builder, "build", lambda frame: (
        frame.assign(**{name: 0. for name in engine.plan.built_feature_names if name not in frame}), {},
    ))
    model = LoadedRuntimeModel(manifest=manifest, model=None, scaler=None, calibrator=None,
                               training_summary={}, scale_clip=0., batch_size=1, use_amp=False)
    processor = LiveSignalProcessor(bindings=[SignalRuntimeModelBinding(loaded_model=model, shadow_mode=True)],
                                    audit_repository=audit, feature_engine=engine, group_name=strategy,
                                    enable_ict_paper_signal_ledger=False, force_shadow_mode=True)
    monkeypatch.setattr(processor, "_persist_dashboard_state", lambda **kwargs: None)
    monkeypatch.setattr(processor._runner_by_model_id[manifest.model_id], "predict_latest", lambda frame, **kwargs:
                        ModelPrediction(model_id=manifest.model_id, direction=manifest.direction,
                                        backend="xgboost", calibrated_probability=0.2, **kwargs))
    return processor


@pytest.mark.parametrize("parallel", [False, True])
def test_live_history_retains_both_strategies_links_rejections_and_reconciles_without_new_bars(tmp_path, monkeypatch, parallel):
    monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", _detector)
    path = tmp_path / "runtime.sqlite"
    with SQLiteLiveDataStore(path) as store:
        audit = LiveAuditRepository(store)
        processors = [_processor(audit, strategy, monkeypatch) for strategy in ("FRVP", "ICT")]
        multi = MultiGroupLiveSignalProcessor(processors) if parallel else None
        try:
            for offset in range(43):
                bar = _bar(offset)
                store.upsert_bar(bar)
                bar = store.fetch_bars(asset="ES", timeframe="5m", start=bar.timestamp, end=bar.timestamp)[0]
                if multi:
                    results = multi.process_bars([bar], emit_operator_artifacts=False)
                else:
                    results = tuple(result for processor in processors for result in
                                    processor.process_bars([bar], emit_operator_artifacts=False))
                assert len(results) == 2
            for processor in processors:
                repo = processor.setup_event_repository
                events = repo.list_events(strategy=processor.group_name)
                assert len(events) == 43
                assert all(not event.selected for event in events)
                assert all(len(repo.prediction_links(event.event_id)) == 1 for event in events)
                link = repo.prediction_links(events[0].event_id)[0]
                assert json.loads(link["payload_json"])["qualified_shadow_entry"] is False
                shadow = json.loads(link["payload_json"])["shadow_evaluation"]
                assert shadow["contract_version"] == "es-selected-setup-full-policy-v1"
                assert not shadow["setup_matched_decision"]
                assert not shadow["qualified_shadow_entry"]
                assert not json.loads(link["payload_json"])["shadow_setup_matched"]
                assert repo.outcomes(events[0].event_id)
            prediction_count = store.connection.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0]
            # A correction on the preceding bar must be consumed even on an idle poll.
            store.upsert_bar(_bar(41, close=99., version="v2"))
            if multi:
                assert multi.process_new_bars_from_store(emit_operator_artifacts=False, max_timestamp=None) == ()
            else:
                for processor in processors:
                    assert processor.process_new_bars_from_store(emit_operator_artifacts=False) == ()
            for processor in processors:
                events = processor.setup_event_repository.list_events(strategy=processor.group_name, latest_only=True)
                revised = next(event for event in events if event.source_timestamp == _bar(41).timestamp)
                assert revised.event_kind == "invalidated"
                assert revised.source_bar_version == "v2"
                assert revised.observed_at > revised.source_timestamp
            assert store.connection.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == prediction_count
        finally:
            if multi:
                multi.close()
    with SQLiteLiveDataStore(path) as reopened:
        from ote_live.storage.setup_events import SetupEventRepository
        assert len(SetupEventRepository(reopened).list_events()) == 88


@pytest.mark.parametrize("parallel", [False, True])
def test_a6_runtime_links_only_the_matching_selected_setup(tmp_path, monkeypatch, parallel):
    def detector(frame, *, strategy):
        return ({"setup_type": "3" if strategy == "FRVP" else "premium_discount_continuation",
                 "setup_side": -1, "setup_family": "continuation", "selected": True, "confidence": .73},
                {"setup_type": "5" if strategy == "FRVP" else "ifvg",
                 "setup_side": -1, "setup_family": "continuation", "selected": False, "confidence": .8})

    monkeypatch.setattr("ote_live.storage.setup_events.extract_setup_observations", detector)
    with SQLiteLiveDataStore(tmp_path / "matching.sqlite") as store:
        audit = LiveAuditRepository(store)
        processors = [_processor(audit, strategy, monkeypatch, model_id=f"{strategy.lower()}_short_continuation_xgb_v1")
                      for strategy in ("FRVP", "ICT")]
        for processor in processors:
            model_id = processor.bindings[0].loaded_model.model_id
            monkeypatch.setattr(processor._runner_by_model_id[model_id], "predict_latest",
                                lambda frame, _model_id=model_id, **kwargs: ModelPrediction(
                                    model_id=_model_id, direction="short", backend="xgboost",
                                    calibrated_probability=1., **kwargs))
        multi = MultiGroupLiveSignalProcessor(processors) if parallel else None
        try:
            bar = _bar(0)
            store.upsert_bar(bar)
            bar = store.fetch_bars(asset="ES", timeframe="5m")[0]
            if multi:
                results = multi.process_bars([bar], emit_operator_artifacts=False)
            else:
                results = tuple(result for processor in processors for result in
                                processor.process_bars([bar], emit_operator_artifacts=False))
            assert len(results) == 2
            for processor in processors:
                events = processor.setup_event_repository.list_events(strategy=processor.group_name)
                assert len(events) == 2
                for event in events:
                    link = processor.setup_event_repository.prediction_links(event.event_id)[0]
                    payload = json.loads(link["payload_json"])
                    assert payload["shadow_setup_matched"] is event.selected
                    assert payload["shadow_evaluation"]["setup_matched_decision"]
                    assert not payload["shadow_evaluation"]["qualified_shadow_entry"]
                    assert not payload["shadow_evaluation"]["policy_candidate_eligible"]  # Recorded/backfilled bar.
        finally:
            if multi:
                multi.close()
