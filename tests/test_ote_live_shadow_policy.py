"""A6: shadow policy gates, durable state, setup identity and legacy isolation."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np
import pytest

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.policies.shadow import match_shadow_setups
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from ote_live.storage.setup_events import SetupEventRecord
from ote_live.models.ict_research import ICT_RESEARCH_PRIORITIES, ICT_LINEAGE_BLOCK_REASON, ICT_RESEARCH_ROSTER_VERSION

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 9, 14, 14, 30, tzinfo=timezone.utc)
CONTEXT = {"trend_regime": "flat", "vol_regime": "normal", "session_regime": "new_york",
           "stress_regime": "normal", "composite_regime": "flat_normal", "expected_move_pips": 10.}


@pytest.fixture
def manifest():
    path = ROOT / "ote_live/runtime_manifests/frvp_es_shadow_20260715/frvp_short_meta_xgb_v1/live_runtime_manifest.json"
    base = LiveRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    policy = base.live_policy.model_copy(update={
        "policy_status": "complete",
        "thresholds": base.live_policy.thresholds.model_copy(update={"global_threshold": .5, "regime_thresholds": {}}),
        "abstain_policy": base.live_policy.abstain_policy.model_copy(update={
            "enabled": True, "abstain_high_stress": True, "abstain_off_hours": True, "cooldown_bars": 2,
            "minimum_expected_move_to_spread": 2., "abstain_session_regimes": [],
            "abstain_composite_regimes": [], "abstain_composite_session_pairs": [],
            "abstain_composite_stress_pairs": [], "minimum_probability_quantile": None,
        }),
        "cost_assumptions": base.live_policy.cost_assumptions.model_copy(update={"session_spread_pips": {"new_york": 1., "off_hours": 1.}}),
    })
    return base.model_copy(update={"live_policy": policy})


def inputs(manifest, idx=10, probability=.8, collection="a6-test"):
    timestamp = START + timedelta(minutes=5 * idx)
    prediction = ModelPrediction(model_id=manifest.model_id, direction="short", backend="xgboost",
                                 timestamp=timestamp, source_row_idx=idx, calibrated_probability=probability,
                                 raw_score=.79, collection_version=collection,
                                 prediction_recorded_at_utc=timestamp + timedelta(minutes=5, seconds=1))
    snapshot = FeatureSnapshot(asset="ES", timeframe="5m", direction="short", timestamp=timestamp,
                               source_row_idx=idx, collection_version=collection, feature_values={"close": 100.},
                               valid_feature_count=1, observation_metadata={
                                   "bar_eligibility": {"eligible": True},
                                   "model_input_contract": {"diagnostic_only": False},
                                   "diagnostic_reasons": [], "source_bar": {"bar_version": "v1"}})
    event = SetupEventRecord(event_id=42, event_key="test", revision=1, event_kind="observed", strategy="FRVP",
                             setup_type="3", setup_side=-1, setup_family="continuation", selected=True,
                             source_timestamp=timestamp, source_bar_version="v1", observed_at=prediction.prediction_recorded_at_utc,
                             first_observed_at=prediction.prediction_recorded_at_utc, collection_version=collection,
                             payload={"source_bar": {"asset": "ES", "timeframe": "5m"}})
    return prediction, snapshot, event


def evaluate(engine, manifest, *, idx=10, probability=.8, collection="a6-test", context=None,
             event_changes=None, metadata=None, **kwargs):
    prediction, snapshot, event = inputs(manifest, idx, probability, collection)
    if event_changes:
        event = replace(event, **event_changes)
    if metadata:
        snapshot = snapshot.model_copy(update={"observation_metadata": {**snapshot.observation_metadata, **metadata}})
    return engine.evaluate_prediction(prediction, live_policy=manifest.live_policy,
                                      policy_context=CONTEXT if context is None else context,
                                      feature_snapshot=snapshot, runtime_manifest=manifest,
                                      shadow_mode=True, shadow_setup_match=match_shadow_setups(prediction, snapshot, [event]),
                                      **kwargs)


def test_four_stages_and_all_rejections_are_persisted(manifest, tmp_path):
    with SQLiteLiveDataStore(tmp_path / "audit.sqlite") as store:
        audit = LiveAuditRepository(store)
        # ES diagnostics persist even with the legacy emit-only default.
        engine = LiveDecisionEngine(audit_repository=audit)
        first = evaluate(engine, manifest)
        result = first.signal.shadow_evaluation
        assert result["threshold_crossed"] and result["setup_matched_decision"] and result["full_policy_passed"]
        assert result["policy_candidate_eligible"] and not result["qualified_shadow_entry"]
        assert result["executable_entry_at"] is None and first.signal.decision == "shadow"
        rejected = evaluate(engine, manifest, idx=11, context={**CONTEXT, "stress_regime": "high",
                            "session_regime": "off_hours", "expected_move_pips": .1})
        result = rejected.signal.shadow_evaluation
        assert {"high_stress", "off_hours", "expected_move_below_spread", "cooldown"} <= set(result["rejection_reasons"])
        assert result["policy_checks"]["cooldown"]["bars_remaining"] == 2
        assert not result["full_policy_passed"]
        assert audit.get_signal_decision(rejected.audit_record.signal_decision_id) == rejected.signal
        metadata = json.loads(store.connection.execute("SELECT metadata_json FROM model_predictions ORDER BY id DESC LIMIT 1").fetchone()[0])
        assert metadata["shadow_evaluation"] == result
        raw = evaluate(engine, manifest, idx=12, probability=.2)
        assert not raw.signal.shadow_evaluation["threshold_crossed"]
        off_setup = evaluate(engine, manifest, idx=13, event_changes={"selected": False})
        assert off_setup.signal.shadow_evaluation["threshold_crossed"]
        assert not off_setup.signal.shadow_evaluation["setup_matched_decision"]
        assert store.connection.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == 4
        assert len(audit.list_shadow_evaluations(stage="raw")) == 4
        assert len(audit.list_shadow_evaluations(stage="threshold")) == 3
        assert len(audit.list_shadow_evaluations(stage="setup")) == 2
        assert len(audit.list_shadow_evaluations(stage="policy")) == 1
        assert audit.list_shadow_evaluations(stage="qualified") == []


@pytest.mark.parametrize("changes,reason", [
    ({"selected": False}, "setup_not_selected"), ({"setup_side": 1}, "setup_side_mismatch"),
    ({"source_bar_version": "v2"}, "setup_source_mismatch"),
    ({"collection_version": "old"}, "setup_source_mismatch"),
    ({"event_kind": "invalidated"}, "setup_not_selected"),
    ({"setup_family": "unknown"}, "setup_family_mismatch"),
    ({"observed_at": START + timedelta(days=1)}, "setup_not_known_at_prediction"),
])
def test_setup_match_rejects_unavailable_or_wrong_identity(manifest, changes, reason):
    path = evaluate(LiveDecisionEngine(), manifest, event_changes=changes)
    match = path.signal.shadow_evaluation["setup_match"]
    assert not match["matched"]
    assert reason in match["candidates"][0]["reasons"]


@pytest.mark.parametrize("model_id,setup_type,family,matched", [
    ("frvp_short_continuation_xgb_v1", "3", "continuation", True),
    ("frvp_short_reversal_xgb_v1", "3", "continuation", False),
    ("frvp_short_continuation_setup5_xgb_v1", "3", "continuation", False),
    ("frvp_short_continuation_setup5_xgb_v1", "5", "continuation", True),
    ("ict_short_continuation_xgb_v1", "premium_discount_continuation", "continuation", True),
    ("ict_short_continuation_ifvg_xgb_v1", "premium_discount_continuation", "continuation", False),
    ("ict_short_meta_xgb_v1", "liquidity_sweep_reclaim", "reversal", True),
    ("frvp_unknown", "3", "continuation", False),
])
def test_pooled_and_specialist_routes(manifest, model_id, setup_type, family, matched):
    manifest = manifest.model_copy(update={"model_id": model_id})
    path = evaluate(LiveDecisionEngine(), manifest, event_changes={"strategy": model_id.split("_")[0].upper(),
                    "setup_type": setup_type, "setup_family": family})
    assert path.signal.shadow_evaluation["setup_match"]["matched"] is matched


def test_missing_policy_inputs_fail_closed_but_legacy_execution_is_preserved(manifest):
    engine = LiveDecisionEngine()
    context = {key: value for key, value in CONTEXT.items() if key != "expected_move_pips"}
    shadow = evaluate(engine, manifest, context=context)
    assert "policy_input_missing:expected_move_pips" in shadow.signal.shadow_evaluation["rejection_reasons"]
    prediction, _, _ = inputs(manifest)
    live = engine.evaluate_prediction(prediction, live_policy=manifest.live_policy, policy_context=context)
    assert live.signal.decision == "emit"
    assert live.signal.shadow_evaluation is None


@pytest.mark.parametrize("metadata", [
    {"bar_eligibility": {"eligible": False}, "diagnostic_reasons": ["backfilled_evaluation"]},
    {"model_input_contract": {"diagnostic_only": True}},
    {"diagnostic_reasons": ["setup_revision_pending"]},
])
def test_diagnostic_observations_do_not_consume_cooldown_or_quantile_history(manifest, metadata):
    engine = LiveDecisionEngine()
    diagnostic = evaluate(engine, manifest, metadata=metadata).signal.shadow_evaluation
    assert not diagnostic["policy_candidate_eligible"]
    assert diagnostic["state_before"] == diagnostic["state_after"]
    next_result = evaluate(engine, manifest, idx=11).signal.shadow_evaluation
    assert next_result["policy_candidate_eligible"]


def test_state_survives_database_reopen_and_partitions_collection_and_policy(manifest, tmp_path):
    db = tmp_path / "restart.sqlite"
    with SQLiteLiveDataStore(db) as store:
        first = evaluate(LiveDecisionEngine(audit_repository=LiveAuditRepository(store)), manifest)
        assert first.signal.shadow_evaluation["policy_candidate_eligible"]
    with SQLiteLiveDataStore(db) as store:
        engine = LiveDecisionEngine(audit_repository=LiveAuditRepository(store))
        second = evaluate(engine, manifest, idx=11)
        result = second.signal.shadow_evaluation
        assert "cooldown" in result["rejection_reasons"]
        assert result["state_before"] == first.signal.shadow_evaluation["state_after"]
        replay = evaluate(LiveDecisionEngine(), manifest, idx=11, shadow_state_before=result["state_before"])
        assert replay.signal == second.signal
        assert evaluate(engine, manifest, idx=13).signal.shadow_evaluation["policy_candidate_eligible"]
        assert evaluate(engine, manifest, idx=14, collection="new").signal.shadow_evaluation["policy_candidate_eligible"]
        changed = manifest.model_copy(update={"live_policy": manifest.live_policy.model_copy(update={"schema_version": "new"})})
        assert evaluate(engine, changed, idx=14).signal.shadow_evaluation["policy_candidate_eligible"]
        from ote_live.storage.collection import MixedCollectionError
        with pytest.raises(MixedCollectionError):
            LiveAuditRepository(store).list_shadow_evaluations()
        assert len(LiveAuditRepository(store).list_shadow_evaluations(collection_version="new")) == 1


def test_quantile_history_uses_matching_candidates_only_and_execution_state_is_isolated(manifest):
    manifest = manifest.model_copy(update={"live_policy": manifest.live_policy.model_copy(update={
        "abstain_policy": manifest.live_policy.abstain_policy.model_copy(update={"minimum_probability_quantile": .5})})})
    engine = LiveDecisionEngine()
    evaluate(engine, manifest, probability=.99, event_changes={"selected": False})
    first = evaluate(engine, manifest, idx=11, probability=.8)
    assert first.signal.shadow_evaluation["policy_candidate_eligible"]
    low = evaluate(engine, manifest, idx=14, probability=.6)
    assert "probability_quantile_filter" in low.signal.shadow_evaluation["rejection_reasons"]
    assert low.signal.shadow_evaluation["state_after"]["candidate_probabilities"] == [.8, .6]
    prediction, _, _ = inputs(manifest, idx=11)
    assert engine.evaluate_prediction(prediction, live_policy=manifest.live_policy, policy_context=CONTEXT).signal.decision == "emit"


def test_missing_numeric_context_is_valid_json_and_survives_restart(manifest, tmp_path):
    db = tmp_path / "missing.sqlite"
    with SQLiteLiveDataStore(db) as store:
        result = evaluate(LiveDecisionEngine(audit_repository=LiveAuditRepository(store)), manifest,
                          context={**CONTEXT, "expected_move_pips": np.float64("nan")})
        assert result.signal.shadow_evaluation["policy_checks"]["expected_move_spread"]["status"] == "unverified"
        assert store.connection.execute("SELECT json_valid(signal_json) FROM signal_decisions").fetchone()[0] == 1
    with SQLiteLiveDataStore(db) as store:
        assert evaluate(LiveDecisionEngine(audit_repository=LiveAuditRepository(store)), manifest, idx=11).signal.shadow_evaluation["policy_candidate_eligible"]


@pytest.mark.parametrize("model_id,setup_type,family,direction,tier,blocked", [
    ("frvp_short_continuation_tcn_v1", "3", "continuation", "short", "priority", False),
    ("frvp_long_continuation_xgb_v1", "3", "continuation", "long", "priority", False),
    ("frvp_long_continuation_setup3_xgb_v1", "3", "continuation", "long", "exploratory", False),
    ("frvp_long_continuation_setup5_xgb_v1", "5", "continuation", "long", "collect_matches", False),
    ("frvp_long_reversal_xgb_v1", "1", "reversal", "long", "background", False),
    ("frvp_long_meta_xgb_v1", "3", "continuation", "long", "background", False),
    ("frvp_short_meta_xgb_v1", "3", "continuation", "short", "background", False),
    ("frvp_long_reversal_setup1_xgb_v1", "1", "reversal", "long", "background", False),
    ("frvp_long_continuation_setup2_xgb_v1", "2", "continuation", "long", "background", False),
    ("frvp_long_reversal_setup6_xgb_v1", "6", "reversal", "long", "background", False),
    ("frvp_long_reversal_setup4_xgb_v1", "4", "reversal", "long", "paused", True),
    ("frvp_short_reversal_xgb_v1", "1", "reversal", "short", "retired", True),
])
def test_a9_a10_roster_persists_diagnostics_and_preserves_qualification_gates(
        manifest, tmp_path, model_id, setup_type, family, direction, tier, blocked):
    policy = manifest.live_policy.model_copy(update={"model_id": model_id, "direction": direction})
    manifest = manifest.model_copy(update={"model_id": model_id, "direction": direction, "live_policy": policy})
    prediction, snapshot, event = inputs(manifest)
    prediction = prediction.model_copy(update={"direction": direction})
    snapshot = snapshot.model_copy(update={"direction": direction})
    event = replace(event, setup_type=setup_type, setup_family=family, setup_side=1 if direction == "long" else -1)
    original_policy = policy.model_dump(mode="json")
    with SQLiteLiveDataStore(tmp_path / "roster.sqlite") as store:
        audit = LiveAuditRepository(store)
        engine = LiveDecisionEngine(audit_repository=audit)
        result = engine.evaluate_prediction(
            prediction, live_policy=policy, policy_context=CONTEXT, shadow_mode=True,
            feature_snapshot=snapshot, runtime_manifest=manifest,
            shadow_setup_match=match_shadow_setups(prediction, snapshot, [event]))
        evaluation = result.signal.shadow_evaluation
        assert evaluation["research_priority"]["tier"] == tier
        assert evaluation["research_priority"]["promotion_authorized"] is False
        assert evaluation["threshold_crossed"] and evaluation["setup_matched_decision"]
        assert evaluation["policy_candidate_eligible"] is not blocked
        assert not evaluation["qualified_shadow_entry"]
        assert evaluation["prerequisite_reasons"] and evaluation["executable_entry_price"] is None
        assert result.signal.decision == "shadow"
        if blocked:
            assert evaluation["research_priority"]["candidate_block_reason"] in evaluation["rejection_reasons"]
            assert evaluation["state_before"] == evaluation["state_after"]
        assert policy.model_dump(mode="json") == original_policy
        assert audit.get_signal_decision(result.audit_record.signal_decision_id) == result.signal
        assert len(audit.list_shadow_evaluations(stage="raw")) == 1
        # Long S5 may never inherit the short-S5 evidence.
        if tier == "collect_matches":
            opposite = replace(event, setup_side=-1)
            assert not match_shadow_setups(prediction, snapshot, [opposite])["matched"]


@pytest.mark.parametrize("setting,value,reason", [
    ("abstain_session_regimes", ["new_york"], "session_filter"),
    ("abstain_composite_regimes", ["flat_normal"], "composite_filter"),
    ("abstain_composite_session_pairs", [("flat_normal", "new_york")], "composite_session_filter"),
    ("abstain_composite_stress_pairs", [("flat_normal", "normal")], "composite_stress_filter"),
])
def test_regime_filters_reuse_execution_predicates(manifest, setting, value, reason):
    manifest = manifest.model_copy(update={"live_policy": manifest.live_policy.model_copy(update={
        "abstain_policy": manifest.live_policy.abstain_policy.model_copy(update={setting: value})})})
    path = evaluate(LiveDecisionEngine(), manifest)
    assert reason in path.signal.shadow_evaluation["rejection_reasons"]
    prediction, _, _ = inputs(manifest)
    legacy = LiveDecisionEngine().evaluate_prediction(prediction, live_policy=manifest.live_policy, policy_context=CONTEXT)
    assert legacy.signal.decision == "abstain" and reason in legacy.signal.reasons


def test_regime_threshold_and_disabled_gates_are_preserved(manifest):
    policy = manifest.live_policy.model_copy(update={
        "thresholds": manifest.live_policy.thresholds.model_copy(update={"regime_thresholds": {"flat_normal": .9}}),
        "abstain_policy": manifest.live_policy.abstain_policy.model_copy(update={"enabled": False}),
    })
    manifest = manifest.model_copy(update={"live_policy": policy})
    result = evaluate(LiveDecisionEngine(), manifest, context={"composite_regime": "flat_normal"}).signal.shadow_evaluation
    assert not result["threshold_crossed"] and result["threshold_source"] == "regime"
    assert all(check["status"] == "disabled" for check in result["policy_checks"].values())
    result = evaluate(LiveDecisionEngine(), manifest, context={"composite_regime": "other"}).signal.shadow_evaluation
    assert result["threshold_crossed"] and result["threshold_source"] == "global"


@pytest.mark.parametrize("model_id", [*ICT_RESEARCH_PRIORITIES, "ict_short_meta_xgb_v99"])
def test_a11_entire_ict_roster_is_quarantined_even_with_passing_inputs(manifest, tmp_path, model_id):
    from ote_live.features.input_contract import INPUT_CONTRACT_VERSION
    from ote_live.models.setup_family import infer_setup_model_route

    direction = model_id.split("_")[1]
    route = infer_setup_model_route(model_id)
    family = "reversal" if "reversal" in model_id else "continuation"
    setup_type = route.setup_type if route else "premium_discount_continuation" if family == "continuation" else "sweep_reclaim"
    policy = manifest.live_policy.model_copy(update={"model_id": model_id, "direction": direction})
    manifest = manifest.model_copy(update={"model_id": model_id, "direction": direction, "live_policy": policy,
        "feature_manifest": manifest.feature_manifest.model_copy(update={"input_contract_version": INPUT_CONTRACT_VERSION})})
    prediction, snapshot, event = inputs(manifest)
    prediction = prediction.model_copy(update={"direction": direction})
    snapshot = snapshot.model_copy(update={"direction": direction})
    event = replace(event, strategy="ICT", setup_type=setup_type, setup_family=family,
                    setup_side=1 if direction == "long" else -1)
    original = manifest.model_dump(mode="json")
    with SQLiteLiveDataStore(tmp_path / "ict.sqlite") as store:
        audit = LiveAuditRepository(store)
        result = LiveDecisionEngine(audit_repository=audit).evaluate_prediction(
            prediction, live_policy=policy, policy_context=CONTEXT, shadow_mode=True,
            feature_snapshot=snapshot, runtime_manifest=manifest,
            shadow_setup_match=match_shadow_setups(prediction, snapshot, [event]))
        evaluation = result.signal.shadow_evaluation
        assert evaluation["threshold_crossed"] and evaluation["setup_matched_decision"] and evaluation["full_policy_passed"]
        assert not evaluation["policy_candidate_eligible"] and not evaluation["qualified_shadow_entry"]
        assert ICT_LINEAGE_BLOCK_REASON in evaluation["rejection_reasons"]
        assert evaluation["state_before"] == evaluation["state_after"]
        priority = evaluation["research_priority"]
        assert priority["contract_version"] == ICT_RESEARCH_ROSTER_VERSION
        assert priority["artifact_review_status"] == "unverified"
        assert priority["research_only"] and not priority["promotion_authorized"]
        assert evaluation["executable_entry_at"] is None and evaluation["executable_entry_price"] is None
        assert result.signal.decision == "shadow"
        assert manifest.model_dump(mode="json") == original
        assert audit.get_signal_decision(result.audit_record.signal_decision_id) == result.signal
        assert len(audit.list_shadow_evaluations(stage="raw")) == 1
        assert audit.list_shadow_evaluations(stage="qualified") == []


def test_a6_audit_replay_rebuilds_features_and_restores_shadow_policy_state(manifest, tmp_path, monkeypatch):
    import pandas as pd
    from ote_live.contracts.market_data import MarketBar
    from ote_live.features.incremental_engine import IncrementalFeatureEngine
    from ote_live.models.loaders import LoadedRuntimeModel
    from ote_live.models.runners import RuntimeModelRunner
    from ote_live.storage import replay_audited_signal

    manifest = manifest.model_copy(update={
        "feature_manifest": manifest.feature_manifest.model_copy(update={
            "selected_feature_names": ["close"], "selected_feature_count": 1, "lag_steps": [0], "delta_feature_count": 0}),
        "context_requirements": manifest.context_requirements.model_copy(update={
            "window_size": 1, "minimum_runtime_history_bars": 1, "context_rows": 0}),
    })
    loader = lambda manifest: LoadedRuntimeModel(manifest=manifest, model=None, scaler=None, calibrator=None,
                                                 training_summary={}, scale_clip=0., batch_size=1, use_amp=False)
    monkeypatch.setattr(RuntimeModelRunner, "_predict_latest_raw_probability", lambda self, frame: .8)
    with SQLiteLiveDataStore(tmp_path / "replay.sqlite") as store:
        audit = LiveAuditRepository(store)
        engine = LiveDecisionEngine(audit_repository=audit)
        # Establish a preceding acceptance so the replay must restore cooldown state.
        evaluate(engine, manifest, idx=10)
        prediction, snapshot, event = inputs(manifest, idx=11)
        for offset in (1, 0):
            store.upsert_bar(MarketBar(asset="ES", timeframe="5m", timestamp=prediction.timestamp - timedelta(minutes=5 * offset),
                                      open=100., high=101., low=99., close=100., volume=50.))
        features = IncrementalFeatureEngine([manifest])
        features.extend(store.fetch_bars(asset="ES", timeframe="5m"))
        frame = features.build_feature_frame()
        rebuilt_snapshot = features.compute_latest_snapshots(require_ready=True)[manifest.model_id]
        snapshot = snapshot.model_copy(update={"feature_values": rebuilt_snapshot.feature_values})
        prediction = RuntimeModelRunner(loader(manifest)).predict_latest(
            frame, timestamp=prediction.timestamp, source_row_idx=11).model_copy(update={
                "prediction_recorded_at_utc": prediction.prediction_recorded_at_utc, "collection_version": "a6-test"})
        original = engine.evaluate_prediction(
            prediction, live_policy=manifest.live_policy, policy_context=pd.DataFrame([CONTEXT]),
            feature_snapshot=snapshot, runtime_manifest=manifest, shadow_mode=True,
            shadow_setup_match=match_shadow_setups(prediction, snapshot, [event]))
        assert "cooldown" in original.signal.shadow_evaluation["rejection_reasons"]
        replay = replay_audited_signal(audit, original.audit_record.signal_decision_id, model_loader=loader)
        assert replay.matches
        assert replay.replayed_signal == original.signal
