from __future__ import annotations

import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard.queries import compute_signal_markouts, summarize_signal_markouts
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from ote_live.storage.collection import (
    LEGACY_COLLECTION, MixedCollectionError, collection_identity,
    fetch_collection_predictions, register_collection,
)
from scripts.preserve_es_collection_baseline import preserve, verify

NOW = datetime(2026, 9, 11, 14, tzinfo=UTC)


def seed(repo, version, timestamp=NOW):
    snapshot = FeatureSnapshot(asset="ES", timeframe="5m", direction="long", timestamp=timestamp,
                               feature_values={"close": 6000}, collection_version=version)
    snapshot_id = repo.record_feature_snapshot(snapshot)
    prediction_id = repo.record_prediction(ModelPrediction(
        model_id="test", direction="long", backend="xgboost", timestamp=timestamp,
        calibrated_probability=.8, collection_version=version), feature_snapshot_id=snapshot_id)
    repo.record_signal_decision(SignalDecision(
        model_id="test", direction="long", timestamp=timestamp, decision="shadow",
        probability=.8, collection_version=version), prediction_id=prediction_id)


def test_report_rejects_mixed_collections_before_limit_and_allows_explicit_partition(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "audit.sqlite") as store:
        repo = LiveAuditRepository(store)
        for i in range(5):
            store.upsert_bar(MarketBar(asset="ES", timeframe="5m", timestamp=NOW + timedelta(minutes=5*i),
                                      open=6000, high=6002, low=5999, close=6000+i/4, volume=10))
        seed(repo, LEGACY_COLLECTION)
        seed(repo, "corrected", NOW + timedelta(minutes=5))
        with pytest.raises(MixedCollectionError):
            fetch_collection_predictions(store)
        with pytest.raises(MixedCollectionError):
            compute_signal_markouts(store, repo, decisions=("shadow",), limit=1)
        frame = compute_signal_markouts(store, repo, decisions=("shadow",), collection_version="corrected")
        assert frame.collection_version.tolist() == ["corrected"]
        assert summarize_signal_markouts(frame).completed_count == 1
        assert len(fetch_collection_predictions(store, collection_version=LEGACY_COLLECTION)) == 1
        combined = pd.concat([frame, frame.assign(collection_version=LEGACY_COLLECTION)])
        with pytest.raises(MixedCollectionError):
            summarize_signal_markouts(combined)


def test_collection_registry_is_immutable(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "audit.sqlite") as store:
        register_collection(store, "v1", {"research_only": True})
        register_collection(store, "v1", {"research_only": True})
        with pytest.raises(ValueError, match="immutable"):
            register_collection(store, "v1", {"research_only": False})
        for sql in ("UPDATE collection_versions SET contract_json='{}'", "DELETE FROM collection_versions"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                store.connection.execute(sql)


@pytest.mark.parametrize("reference", ["features/recipes/input.json", "ict/feature_sets/context.py",
    "ote_live/contracts/prediction.py", "scripts/ote_targeted_filter_presets.py", "model.bin",
    "ote_live/models/ict_research.py"])
def test_content_changes_rotate_collection_even_with_same_model_name(tmp_path, reference):
    path = tmp_path / reference
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original")
    payload = {"model_id": "same-model", "artifact_references": {"model_file": "model.bin"},
               "live_policy": {"threshold": .5}}
    manifest = SimpleNamespace(model_id="same-model", model_dump=lambda **kwargs: payload)
    first, contract = collection_identity([manifest], root=tmp_path)
    assert collection_identity([manifest], root=tmp_path)[0] == first
    assert contract["research_only"] and not contract["qualified_period_started"]
    assert contract["ict_research_roster_version"] == "es-ict-research-priorities-v1"
    path.write_text("replacement content")
    second, _ = collection_identity([manifest], root=tmp_path)
    assert first != second
    payload["live_policy"]["threshold"] = .7
    assert collection_identity([manifest], root=tmp_path)[0] != second


def test_legacy_migration_preserves_payload_and_marks_old_rows(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript((ROOT / "ote_live/storage/schema.sql").read_text())
    for migration in sorted((ROOT / "ote_live/storage/migrations").glob("*.sql")):
        if migration.name >= "0006":
            break
        connection.executescript(migration.read_text())
        connection.execute("INSERT INTO schema_migrations VALUES (?, ?)", (migration.name, NOW.isoformat()))
    payload = '{"original": true}'
    connection.execute("INSERT INTO feature_snapshots (asset,timeframe,timestamp_utc,feature_values_json,snapshot_json,recorded_at_utc) VALUES (?,?,?,?,?,?)",
                       ("ES", "5m", NOW.isoformat(), "{}", payload, NOW.isoformat()))
    connection.commit()
    connection.close()
    with SQLiteLiveDataStore(path) as store:
        row = store.connection.execute("SELECT * FROM feature_snapshots").fetchone()
        assert row["collection_version"] == LEGACY_COLLECTION
        assert row["snapshot_json"] == payload


def test_snapshot_captures_nested_evidence_policy_and_verifies_independently(tmp_path):
    audit = tmp_path / "audit"
    (audit / "appendix").mkdir(parents=True)
    (audit / "appendix/evidence.txt").write_text("evidence")
    (audit / "current_model_semantics.csv").write_text("model_id,manifest_path\ntest,manifest.json\n")
    (tmp_path / "weights.bin").write_bytes(b"original model")
    (tmp_path / "policy.csv").write_text("frozen threshold")
    manifest = {"model_id": "test", "artifact_references": {"model_file": "weights.bin"},
                "live_policy": {"lineage": {"policy_table_path": "policy.csv"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "frozen"
    result = preserve(audit, destination, root=tmp_path)
    assert "audit/appendix/evidence.txt" in result["files"]
    assert "policy_lineage/test/policy_table_path" in result["files"]
    (tmp_path / "weights.bin").write_bytes(b"replacement")
    assert verify(destination)["missing"] == []
    with pytest.raises(FileExistsError):
        preserve(audit, destination, root=tmp_path)
    artifact = result["files"]["artifact/test/model_file"]
    (destination / artifact["object"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity failure"):
        verify(destination)
