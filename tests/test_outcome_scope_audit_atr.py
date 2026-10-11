from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from test_outcome_contracts import signal, tape, simulator, contract, ROOT
from model_testing.outcome_artifacts import replay_outcome_run, run_outcome_evaluation
from model_testing.outcome_atr import completed_structural_atr, structural_atr_at
from model_testing.outcome_contracts import load_contracts, file_hash
from model_testing.outcome_diagnostics import ModelSetupContext, SetupObservation
from model_testing.outcome_engine import EvaluationTape, evaluate_signal, simulate_positions
from model_testing.outcome_scope import load_scope


@pytest.mark.parametrize("strategy,family,setup,atr_definition", [
    ("frvp", "reversal", "setup1", "completed_1h_structural_atr"),
    ("frvp", "continuation", "setup3", "completed_1h_structural_atr"),
    ("frvp", "meta", "setup4", "completed_1h_structural_atr"),
    ("ict", "reversal", "sweep_reclaim", "signal_bar_atr14"),
    ("ict", "continuation", "premium_discount_continuation", "signal_bar_atr14"),
    ("ict", "meta", "ifvg_reversal", "signal_bar_atr14"),
    ("ote", "ote", "reversal", "UNRESOLVED"),
    ("ote", "meta", "reversal", "UNRESOLVED"),
    ("ote", "union", "reversal", "UNRESOLVED"),
])
def test_every_pooled_family_is_diagnostics_only_even_if_policy_passes(strategy, family, setup, atr_definition):
    from model_testing.outcome_contracts import resolve_contract
    instrument = "EURUSD" if strategy == "ote" else "ES"
    c = resolve_contract(instrument=instrument, strategy=strategy, model_family=family, setup_family=setup, side="long")
    s = signal(instrument=instrument, strategy=strategy, model_family=family, setup_family=setup,
               atr_definition=atr_definition, policy_eligible=True)
    result = evaluate_signal(s, tape("target"), c)
    assert result["baseline_scope_reason"] == "pooled_model_diagnostics_only"
    assert result["paper_contract_outcome"] == "NOT_EVALUATED"
    assert result["forward_return_1_status"] == "OK"
    assert result["stop_price"] is None and not result["paper_resolved"]
    p, = simulate_positions([s], [result], {c.contract_id: c}, simulator(accounting_unit=c.costs.unit))
    assert p["no_trade_reason"] == "pooled_model_diagnostics_only"
    assert not p["position_taken"] and p["net_pnl"] is None


def test_breaker_excluded_but_causal_diagnostics_retained():
    c = load_contracts()["es_ict_classic_breaker_eval_v1"]
    s = signal(strategy="ict", model_family="classic_breaker", setup_family="classic_breaker",
        atr_definition="signal_bar_atr14", source_structure_formed_ts="2026-09-14T13:00:00Z",
        structure_activation_ts="2026-09-14T13:30:00Z", setup_retest_ts="2026-09-14T13:55:00Z")
    result = evaluate_signal(s, tape(), c)
    assert result["baseline_scope_reason"] == "classic_breaker_excluded_from_baseline"
    assert result["paper_contract_outcome"] == "NOT_EVALUATED"
    assert result["forward_return_48_status"] == "OK"


def test_unmatched_remains_distinct_from_matching_pooled_prediction():
    c = load_contracts()["es_frvp_unmatched_eval_v1"]
    result = evaluate_signal(signal(model_family="meta", setup_family="unmatched"), tape(), c)
    assert result["baseline_scope_reason"] == "unmatched_prediction_diagnostics_only"
    assert result["forward_return_48_status"] == "OK"


def test_scope_is_locked_separately_from_existing_risk_contracts(tmp_path):
    path = ROOT / "configs/outcome_contracts/baseline_scope_v1.json"
    edited = json.loads(path.read_text())
    edited["pooled_families"] = []
    (tmp_path / "scope.json").write_text(json.dumps(edited))
    (tmp_path / "scope.lock.json").write_bytes(path.with_suffix(".lock.json").read_bytes())
    with pytest.raises(ValueError, match="publish a new version"):
        load_scope(tmp_path / "scope.json")


def audit_context(**updates):
    values = dict(collection_id="synthetic-v1", opportunity_id="bar-1", prediction_id="prediction-1",
        available_at_ts="2026-09-14T14:00:00Z", coverage="all_model_observations", setup_coverage="complete",
        expected_model_ids=("pooled", "specialist", "missing"), raw_score=-2, calibrated_probability=0.2,
        threshold=0.7, threshold_passed=False, trend_regime="up", volatility_regime="high",
        context_source_hash="a" * 64,
        setups=[dict(setup_event_id=str(i), strategy="frvp", setup_family=f"setup{i}", side=side,
                     known_at_ts="2026-09-14T14:00:00Z", selected=selected, detector_version="synthetic-detector-v1")
                for i, side, selected in ((1, "long", True), (2, "long", False), (5, "short", True))])
    values.update(updates)
    return ModelSetupContext.model_validate(values)


def test_audit_retains_all_setups_rejected_scores_missing_models_and_scope_in_replay(tmp_path):
    signals = [signal(event_id="pooled-row", model_id="pooled", model_family="reversal", audit_context=audit_context()),
               signal(event_id="specialist-row", model_id="specialist", audit_context=audit_context(prediction_id="prediction-2"))]
    summary = run_outcome_evaluation(signals, tape("target"), output_dir=tmp_path / "run", simulator=simulator())
    assert summary["diagnostics_only_count"] == 1
    assert summary["paper_resolved_count"] == 1
    assert summary["position_taken_count"] == 1
    rows = pd.read_csv(tmp_path / "run/model_setup_observations.csv")
    assert len(rows) == 6
    assert set(rows.relationship) == {"selected_same_side", "other_same_side", "opposite_side"}
    assert rows.calibrated_probability.eq(0.2).all() and rows.threshold_passed.eq(False).all()
    coverage = pd.read_csv(tmp_path / "run/model_observation_coverage.csv")
    assert coverage.loc[coverage.model_id.eq("missing"), "observed"].eq(False).all()
    assert replay_outcome_run(tmp_path / "run", tmp_path / "replay") == summary
    for namespace in ("training_label_outcomes", "paper_contract_outcomes", "forward_returns", "simulated_positions"):
        saved = pd.read_csv(tmp_path / f"run/{namespace}.csv")
        assert saved.baseline_scope_hash.eq(load_scope().sha256).all()


def test_missing_setup_context_is_not_claimed_to_be_no_setup_and_future_setups_excluded(tmp_path):
    context = audit_context()
    future = context.setups[0].model_copy(update={"known_at_ts": pd.Timestamp("2026-09-14T14:05Z")})
    signals = [signal(event_id="unknown", model_family="reversal"),
               signal(event_id="empty", model_family="reversal", audit_context=audit_context(setups=[])),
               signal(event_id="future", model_family="reversal", audit_context=audit_context(setups=[future]))]
    run_outcome_evaluation(signals, tape("target"), output_dir=tmp_path / "audit")
    rows = pd.read_csv(tmp_path / "audit/model_setup_observations.csv")
    assert rows.relationship.tolist() == ["unknown_setup_coverage", "no_setup", "not_known_at_decision"]
    summary = pd.read_csv(tmp_path / "audit/model_setup_response_summary.csv")
    assert summary.loc[summary.relationship.eq("not_known_at_decision"), "forward_return_1_valid_count"].eq(0).all()


def history_tape(*, remove=(), roll=None, spike=True, instrument="ES"):
    times = pd.date_range("2026-09-14T00:00Z", periods=32 * 12, freq="5min")
    frame = pd.DataFrame(dict(open_ts=times, open=100., high=101., low=99., close=100., coordinate_id="ESU6"))
    if spike:
        frame.loc[15 * 12 + 11, "high"] = 141
    if roll is not None:
        frame.loc[roll:, "coordinate_id"] = "ESZ6"
    if instrument == "6E":
        for col in ("open", "high", "low", "close"):
            frame[col] = 1 + (frame[col] - 100) * 0.00005
        frame["coordinate_id"] = "6EU6"
    schedule = pd.DataFrame(dict(open_ts=times, session_id="synthetic"))
    return EvaluationTape(frame.drop(index=list(remove)), schedule, timeframe_minutes=5,
        calendar_id="synthetic", calendar_source="explicit_test_schedule", coverage_start=times[0],
        coverage_end=times[-1] + pd.Timedelta(minutes=5))


def test_completed_hour_atr_cannot_see_later_constituents_and_uses_wilder14():
    t = history_tape()
    hourly = completed_structural_atr(t)
    proof = structural_atr_at(hourly, "2026-09-14T15:05Z", coordinate_id="ESU6")
    assert proof.value == 2
    assert proof.source_close_ts == pd.Timestamp("2026-09-14T15:00Z")
    completed = structural_atr_at(hourly, "2026-09-14T16:00Z", coordinate_id="ESU6")
    assert completed.value == pytest.approx(2 + 40 / 14)
    t.bars.loc[t.bars.index >= pd.Timestamp("2026-09-14T15:10Z"), "high"] = 199
    same = structural_atr_at(completed_structural_atr(t), "2026-09-14T15:05Z", coordinate_id="ESU6")
    assert same == proof  # Includes the source hash; no future bytes enter its identity.


@pytest.mark.parametrize("instrument,ticks", [("ES", 8), ("6E", 2)])
def test_structural_atr_is_price_units_not_ticks_or_five_minute_atr(instrument, ticks):
    hourly = completed_structural_atr(history_tape(instrument=instrument, spike=False))
    proof = structural_atr_at(hourly, "2026-09-14T14:00Z", coordinate_id=f"{instrument}U6")
    increment = 0.25 if instrument == "ES" else 0.00005
    assert proof.value / increment == pytest.approx(ticks)
    assert proof.units == "price" and proof.timeframe_minutes == 60 and proof.period == 14


@pytest.mark.parametrize("kwargs", [dict(remove=(14 * 12 + 2,)), dict(roll=14 * 12 + 2)])
def test_structural_atr_does_not_bridge_missing_or_invalid_roll_hour(kwargs):
    hourly = completed_structural_atr(history_tape(**kwargs))
    assert structural_atr_at(hourly, "2026-09-14T15:00Z", coordinate_id="ESU6") is None
    assert structural_atr_at(hourly, "2026-09-14T16:00Z", coordinate_id="ESZ6") is None


def test_legacy_frvp_hour_open_mapping_exposes_future_range_without_changing_labels():
    from data.labeling.frvp_labeling_engine import _resample_ohlcv
    from data.labeling.reversal_labeling_engine import compute_atr, map_htf_atr_to_ltf
    t = history_tape()
    bars = t.bars.copy()
    bars["volume"] = 1
    legacy = map_htf_atr_to_ltf(compute_atr(_resample_ohlcv(bars, "1h")), bars.index)
    # The legacy mapping assigns the 15:55 spike to 15:00. Preserve old labels,
    # but prevent this path from entering a causally attested B1 ATR contract.
    assert legacy.loc[pd.Timestamp("2026-09-14T15:00Z")] > 2
    proof = structural_atr_at(completed_structural_atr(t), "2026-09-14T15:00Z", coordinate_id="ESU6")
    assert proof.value == 2


def test_missing_or_forged_atr_provenance_blocks_paper_and_atr_markouts():
    s = signal(atr_provenance=None)
    result = evaluate_signal(s, tape(), contract())
    assert result["paper_outcome_reason"] == "structural_atr_provenance_missing"
    assert result["forward_return_1_status"] == "OK" and result["forward_return_1_atr"] is None
    proof = signal().atr_provenance.model_copy(update={"source_close_ts": pd.Timestamp("2026-09-14T15:00Z")})
    result = evaluate_signal(signal(atr_provenance=proof), tape(), contract())
    assert result["paper_outcome_reason"] == "structural_atr_not_available"


def test_derive_mode_produces_verified_atr_and_replays_without_relabeling(tmp_path):
    s = signal(atr=999, atr_provenance=None)
    before = s.model_dump()
    result = run_outcome_evaluation([s], history_tape(spike=False), output_dir=tmp_path / "derived",
                                    derive_structural_atr=True)
    saved = pd.read_csv(tmp_path / "derived/evaluated_signals.csv")
    assert saved.atr_price.iloc[0] == 2
    assert saved.atr_status.iloc[0] == "verified_definition"
    assert saved.training_label_outcome.iloc[0] == 1
    assert s.model_dump() == before
    assert replay_outcome_run(tmp_path / "derived", tmp_path / "replay") == result


def test_position_simulator_independently_enforces_scope_even_on_imported_outcomes():
    s = signal(model_family="reversal")
    result = evaluate_signal(signal(), tape("target"), contract())
    assert result["paper_resolved"]
    position, = simulate_positions([s], [result], {contract().contract_id: contract()}, simulator())
    assert not position["position_taken"]
    assert position["no_trade_reason"] == "pooled_model_diagnostics_only"


def test_mixed_collection_audit_cannot_be_merged_into_one_baseline(tmp_path):
    with pytest.raises(ValueError, match="different collection baselines"):
        run_outcome_evaluation([signal(event_id="a", audit_context=audit_context()),
            signal(event_id="b", audit_context=audit_context(collection_id="other"))],
            tape(), output_dir=tmp_path / "mixed")


def test_partial_hour_and_delayed_source_cannot_publish_atr_early():
    t = history_tape(spike=False)
    t.bars["observed_at_ts"] = t.bars.index + pd.Timedelta(minutes=5)
    t.bars.loc[t.bars.index == pd.Timestamp("2026-09-14T13:55Z"), "observed_at_ts"] = pd.Timestamp("2026-09-14T14:03Z")
    hourly = completed_structural_atr(t)
    assert structural_atr_at(hourly, "2026-09-14T14:00Z", coordinate_id="ESU6") is None
    assert structural_atr_at(hourly, "2026-09-14T14:03Z", coordinate_id="ESU6").value == 2
    t.bars["is_complete"] = True
    t.bars.loc[t.bars.index == pd.Timestamp("2026-09-14T13:55Z"), "is_complete"] = False
    assert structural_atr_at(completed_structural_atr(t), "2026-09-14T14:03Z", coordinate_id="ESU6") is None
