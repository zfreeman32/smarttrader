from __future__ import annotations

import json

import pandas as pd

from model_testing.outcome_scope import load_research_scope, load_scope
from model_testing.outcome_engine import evaluate_signal, simulate_positions
from test_outcome_contracts import ROOT, contract, signal, simulator, tape
from test_outcome_scope_audit_atr import audit_context


def test_research_scope_is_locked_and_baseline_bytes_are_unchanged():
    baseline_path = ROOT / "configs/outcome_contracts/baseline_scope_v1.json"
    baseline_bytes = baseline_path.read_bytes()
    baseline_lock_bytes = baseline_path.with_suffix(".lock.json").read_bytes()

    scope = load_research_scope()

    assert scope.scope_id == "b2_research_scope_v1"
    assert scope.simulated_outcomes_only is True
    assert scope.broker_order_authority is False
    assert baseline_path.read_bytes() == baseline_bytes
    assert baseline_path.with_suffix(".lock.json").read_bytes() == baseline_lock_bytes
    assert load_scope(baseline_path).scope_id == "b1_baseline_scope_v1"


def _pooled_signal(**changes):
    values = dict(model_family="reversal", setup_family="setup1", audit_context=audit_context())
    values.update(changes)
    return signal(**values)


def test_selected_causal_same_side_pooled_route_can_simulate():
    scope = load_research_scope()
    s = _pooled_signal()
    result = evaluate_signal(s, tape("target"), contract(), scope=scope)

    assert result["baseline_scope"] == "included"
    assert result["baseline_scope_reason"] == ""
    assert result["paper_contract_outcome"] == "TARGET"
    position, = simulate_positions([s], [result], {contract().contract_id: contract()},
                                    simulator(), scope=scope)
    assert position["position_taken"]


def test_pooled_route_fails_closed_without_causal_selected_setup():
    scope = load_research_scope()
    c = contract()
    cases = [
        (None, "pooled_setup_context_missing"),
        (audit_context(setups=[]), "pooled_setup_not_known_at_decision"),
        (audit_context(setups=[audit_context().setups[0].model_copy(update={"selected": False})]),
         "pooled_counterfactual_setup_non_executable"),
        (audit_context(setups=[audit_context().setups[0].model_copy(update={
            "known_at_ts": pd.Timestamp("2026-09-14T14:05Z")
        })]), "pooled_setup_not_known_at_decision"),
    ]
    for context, reason in cases:
        s = _pooled_signal(audit_context=context)
        result = evaluate_signal(s, tape("target"), c, scope=scope)
        assert result["baseline_scope"] == "diagnostics_only"
        assert result["baseline_scope_reason"] == reason
        assert result["paper_contract_outcome"] == "NOT_EVALUATED"
        position, = simulate_positions([s], [result], {c.contract_id: c}, simulator(), scope=scope)
        assert not position["position_taken"]
        assert position["no_trade_reason"] == reason


def test_research_scope_rejects_late_context_and_preserves_counterfactual_audit():
    scope = load_research_scope()
    context = audit_context(available_at_ts="2026-09-14T14:01:00Z")
    s = _pooled_signal(audit_context=context)
    s = s.model_copy(update={"known_at_ts": pd.Timestamp("2026-09-14T14:00:00Z")})
    result = evaluate_signal(s, tape("target"), contract(), scope=scope)
    assert result["baseline_scope_reason"] == "pooled_setup_context_not_available_at_decision"

    # The locked scope has an explicit, machine-readable counterfactual rule.
    payload = json.loads((ROOT / "configs/outcome_contracts/b2_research_scope_v1.json").read_text())
    assert payload["pooled_counterfactual"] == "other_same_side_markout_only"
