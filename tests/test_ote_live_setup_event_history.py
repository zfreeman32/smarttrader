from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.repositories import LiveAuditRepository
from ote_live.storage.setup_events import SetupEventRepository, extract_setup_observations
from test_frvp_setups import _setup_fixture
from test_ict_setup_detector_phase3 import _setup_input


NOW = datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)


def _bar(timestamp=NOW, **updates):
    return MarketBar(asset="ES", timeframe="5m", timestamp=timestamp, open=100, high=102,
                     low=99, close=101, volume=50, **updates)


def _observation(strategy="FRVP", **updates):
    return {"setup_type": "3" if strategy == "FRVP" else "premium_discount_continuation",
            "setup_side": 1, "confidence": 0.73, "selected": True,
            "geometry": {"anchor_level": 100.0, "stop_reference": 99.0}, **updates}


@pytest.mark.parametrize("strategy", ["FRVP", "ICT"])
def test_history_preserves_revisions_invalidations_and_collection_boundaries(tmp_path, strategy):
    with SQLiteLiveDataStore(tmp_path / "setups.sqlite") as store:
        repo = SetupEventRepository(store)
        kwargs = dict(bar=_bar(), strategy=strategy, collection_version="corrected-v1")
        first, = repo.record_observations([_observation(strategy)], source_bar_version="1", observed_at=NOW, **kwargs)
        repeat, = repo.record_observations([_observation(strategy)], source_bar_version="1",
                                           observed_at=NOW + timedelta(seconds=2), **kwargs)
        assert repeat.event_id == first.event_id
        revised, = repo.record_observations([_observation(strategy, confidence=0.51)], source_bar_version="2",
                                            observed_at=NOW + timedelta(minutes=1), **kwargs)
        assert revised.first_observed_at == NOW
        assert revised.observed_at == NOW + timedelta(minutes=1)
        repo.record_observations([], source_bar_version="3", **kwargs)
        history = repo.history(first.event_key)
        assert [event.event_kind for event in history] == ["observed", "revised", "invalidated"]
        assert history[0].payload["confidence"] == 0.73
        assert history[1].payload["confidence"] == 0.51
        assert history[2].payload["invalidation_reason"] == "absent_from_recomputed_source_bar"
        assert history[0].payload["geometry"]["stop_reference"] == 99
        other, = repo.record_observations([_observation(strategy)], bar=_bar(), strategy=strategy,
                                          collection_version="replacement-v2")
        assert other.event_key != first.event_key
        assert len(repo.list_events(collection_version="corrected-v1")) == 3
        for operation in ("UPDATE setup_events SET rule_confidence=0", "DELETE FROM setup_events"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                store.connection.execute(operation)


def test_more_than_dashboard_limit_survives_restart_and_rejected_outcomes_are_retained(tmp_path):
    db_path = tmp_path / "setups.sqlite"
    with SQLiteLiveDataStore(db_path) as store:
        repo = SetupEventRepository(store)
        for offset in range(51):
            strategy = "FRVP" if offset % 2 else "ICT"
            event, = repo.record_observations([
                _observation(strategy, selected=False, rejection_reasons=["detector_priority"])
            ], bar=_bar(NOW + timedelta(minutes=5 * offset)), strategy=strategy, collection_version="v1")
        audit = LiveAuditRepository(store)
        snapshot_id = audit.record_feature_snapshot(FeatureSnapshot(asset="ES", timeframe="5m", timestamp=event.source_timestamp,
                                                                     feature_values={"x": 1.0}))
        prediction_id = audit.record_prediction(ModelPrediction(
            model_id="ict_long_continuation_xgb_v1", direction="long", backend="xgboost",
            timestamp=event.source_timestamp, collection_version="v1", calibrated_probability=0.2), feature_snapshot_id=snapshot_id)
        link_id = repo.link_prediction(event.event_id, prediction_id, model_id="ict_long_continuation_xgb_v1",
                                      decision="hold", rejection_reasons=["threshold_not_met"])
        assert link_id == repo.link_prediction(event.event_id, prediction_id,
                model_id="ict_long_continuation_xgb_v1", decision="hold", rejection_reasons=["threshold_not_met"])
        future = _bar(event.source_timestamp + timedelta(minutes=5), is_complete=True, bar_version="bar-v1")
        repo.record_followup_bar(future, collection_version="v1")
        repo.record_followup_bar(future, collection_version="v1")
        assert len(repo.outcomes(event.event_id)) == 1
        outcome = json.loads(repo.outcomes(event.event_id)[0]["payload_json"])
        assert outcome["executable_entry"] is False
        assert outcome["elapsed_seconds"] == 300
        assert len(repo.prediction_links(event.event_id)) == 1
        link = repo.prediction_links(event.event_id)[0]
        assert json.loads(link["rejection_reasons_json"]) == ["threshold_not_met"]
        assert json.loads(link["payload_json"])["prediction"]["calibrated_probability"] == 0.2
        for table in ("setup_prediction_links", "setup_event_outcomes"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                store.connection.execute(f"DELETE FROM {table}")
        with pytest.raises(ValueError, match="existing prediction"):
            repo.link_prediction(event.event_id, 99999, model_id="missing", decision="hold")
    with SQLiteLiveDataStore(db_path) as reopened:
        repo = SetupEventRepository(reopened)
        assert len(repo.list_events()) == 51
        assert all(not event.selected for event in repo.list_events())
        assert repo.outcomes(event.event_id)


def test_frvp_detector_candidates_include_priority_and_repeat_rejections():
    frame = _setup_fixture()
    # Keep an active candidate for another bar to exercise detector suppression.
    frame = pd.concat([frame.iloc[:2], frame.iloc[[1]]], ignore_index=True)
    observations = extract_setup_observations(frame, strategy="FRVP")
    assert observations
    assert any(not item["selected"] for item in observations)
    assert any("already_active" in item.get("rejection_reasons", []) for item in observations)
    assert all(item["detector_version"].startswith("sha256:") for item in observations)
    assert all("frvp_dist_vah_atr" in item["geometry"] for item in observations)


def test_ict_collector_uses_all_candidates_and_the_selected_events_attribute(monkeypatch):
    import ote_live.storage.setup_events as module

    def detector(frame):
        output = pd.DataFrame({"fired": [True]})
        output.attrs["candidate_events"] = pd.DataFrame([
            {"bar_index": 0, "setup_type": "sweep_reclaim", "setup_side": 1,
             "confidence": 0.9, "eligible": True, "selected": False, "anchor_level": 99},
            {"bar_index": 0, "setup_type": "ifvg_reversal", "setup_side": -1,
             "confidence": 0.8, "eligible": True, "selected": False, "stop_reference": 102},
        ])
        output.attrs["fired_events"] = output.attrs["candidate_events"].iloc[[0]].copy()
        return output

    monkeypatch.setattr(module, "detect_ict_setups", detector)
    observations = extract_setup_observations(pd.DataFrame({"close": [101]}), strategy="ICT")
    assert len(observations) == 2
    assert observations[0]["selected"] is True
    assert observations[1]["selected"] is False
    assert observations[1]["rejection_reasons"] == ["detector_priority"]
    assert observations[1]["geometry"]["stop_reference"] == 102


def test_missing_detector_inputs_raise_without_erasing_observed_history(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "setups.sqlite") as store:
        repo = SetupEventRepository(store)
        first, = repo.record_observations([_observation()], bar=_bar(), strategy="FRVP", collection_version="v1")
        with pytest.raises(ValueError, match="missing required columns"):
            repo.collect_from_frame(pd.DataFrame({"close": [101]}), bar=_bar(), collection_version="v1",
                                    strategies=("FRVP",))
        assert len(repo.history(first.event_key)) == 1


def test_ict_no_candidate_result_is_empty_without_optional_geometry():
    assert extract_setup_observations(_setup_input(), strategy="ICT") == ()


def test_reobserving_same_source_values_does_not_manufacture_a_revision(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "setups.sqlite") as store:
        repo = SetupEventRepository(store)
        first, = repo.record_observations([_observation()],
            bar=_bar(first_observed_at=NOW, last_observed_at=NOW), strategy="FRVP", collection_version="v1")
        again, = repo.record_observations([_observation()],
            bar=_bar(first_observed_at=NOW, last_observed_at=NOW + timedelta(seconds=2)),
            strategy="FRVP", collection_version="v1")
        assert first.event_id == again.event_id
        assert len(repo.history(first.event_key)) == 1


def test_failed_detector_result_rolls_back_all_appends(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "atomic.sqlite") as store:
        repo = SetupEventRepository(store)
        kwargs = dict(bar=_bar(), strategy="FRVP", collection_version="v1")
        first, = repo.record_observations([_observation()], **kwargs)
        for bad in (_observation(setup_side=0), _observation(confidence=0.4)):
            with pytest.raises(ValueError):
                repo.record_observations([_observation(confidence=0.4), bad], **kwargs)
            store.connection.commit()  # Subsequent unrelated persistence cannot save a partial result.
            assert repo.history(first.event_key) == (first,)


@pytest.mark.parametrize("mismatch", ["collection", "timestamp", "asset", "timeframe"])
def test_prediction_links_reject_unrelated_observations(tmp_path, mismatch):
    with SQLiteLiveDataStore(tmp_path / "links.sqlite") as store:
        repo = SetupEventRepository(store)
        event, = repo.record_observations([_observation()], bar=_bar(), strategy="FRVP", collection_version="v1")
        audit = LiveAuditRepository(store)
        snapshot_id = audit.record_feature_snapshot(FeatureSnapshot(
            asset="NQ" if mismatch == "asset" else "ES",
            timeframe="1m" if mismatch == "timeframe" else "5m", timestamp=NOW, feature_values={"x": 1.},
        ))
        prediction_id = audit.record_prediction(ModelPrediction(
            model_id="frvp_test", direction="long", backend="xgboost", calibrated_probability=0.2,
            timestamp=NOW + timedelta(minutes=5 if mismatch == "timestamp" else 0),
            collection_version="other" if mismatch == "collection" else "v1",
        ), feature_snapshot_id=snapshot_id)
        with pytest.raises(ValueError, match="must share"):
            repo.link_prediction(event.event_id, prediction_id, model_id="frvp_test", decision="hold")
        assert repo.prediction_links(event.event_id) == ()


def test_replace_cannot_overwrite_history_even_from_another_connection(tmp_path):
    path = tmp_path / "immutable.sqlite"
    with SQLiteLiveDataStore(path) as store:
        repo = SetupEventRepository(store)
        event, = repo.record_observations([_observation()], bar=_bar(), strategy="FRVP", collection_version="v1")
        repo.record_outcome(event.event_id, "example", {"value": 1})
        original = dict(store.connection.execute("SELECT * FROM setup_events").fetchone())
        with sqlite3.connect(path) as other:
            changed = {**original, "rule_confidence": 0.01}
            other.execute(f"INSERT OR REPLACE INTO setup_events ({','.join(changed)}) "
                          f"VALUES ({','.join('?' for _ in changed)})", tuple(changed.values()))
            other.execute("INSERT OR REPLACE INTO setup_event_outcomes "
                          "SELECT id,setup_event_id,outcome_type,observed_at_utc,'{}',content_hash FROM setup_event_outcomes")
        assert dict(store.connection.execute("SELECT * FROM setup_events").fetchone()) == original
        assert json.loads(repo.outcomes(event.event_id)[0]["payload_json"]) == {"value": 1}
