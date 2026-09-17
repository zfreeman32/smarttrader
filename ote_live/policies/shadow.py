"""Versioned prospective ES policy diagnostics; never an execution ledger.

Historical execution keeps its existing specialist gate. This experiment also
requires selected same-bar, same-side setups for pooled family and meta models.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from numbers import Number

from model_testing.ote_abstain_policy import DEFAULT_SESSION_SPREAD_PIPS
from ote_live.models.setup_family import infer_setup_model_route
from ote_live.models.frvp_research import frvp_research_priority
from ote_live.models.ict_research import ict_research_priority
from ote_live.policies.abstain import LiveAbstainState, evaluate_live_abstain
from ote_live.storage.collection import canonical_json

SHADOW_POLICY_CONTRACT = "es-selected-setup-full-policy-v1"
SETUP_MATCH_CONTRACT = "selected-same-bar-side-family-or-specialist-v1"
PREREQUISITE_REASONS = ["outcome_contract_B1_pending", "artifact_baseline_B2_unverified",
                        "comparison_contract_B3_pending", "research_ledger_A7_pending"]
_POOLED = re.compile(r"^(frvp|ict)_(long|short)_(meta|reversal|continuation)_(xgb|tcn|lstm)_v\d+$")


def match_shadow_setups(prediction, snapshot, events) -> dict:
    specialist = infer_setup_model_route(prediction.model_id)
    pooled = _POOLED.fullmatch(prediction.model_id.lower())
    strategy = specialist.strategy if specialist else pooled[1].upper() if pooled else None
    family = specialist.family if specialist else pooled[3] if pooled else None
    direction = specialist.direction if specialist else pooled[2] if pooled else None
    candidates = []
    for event in events:
        if event.strategy != strategy:
            continue
        reasons = []
        if not event.selected or event.payload.get("research") or event.event_kind == "invalidated":
            reasons.append("setup_not_selected")
        if event.setup_side != (1 if prediction.direction == "long" else -1):
            reasons.append("setup_side_mismatch")
        if event.setup_family not in {"reversal", "continuation"} or (family != "meta" and event.setup_family != family):
            reasons.append("setup_family_mismatch")
        if specialist and event.setup_type != specialist.setup_type:
            reasons.append("setup_type_mismatch")
        source = event.payload.get("source_bar", {})
        if (event.collection_version != snapshot.collection_version
                or event.source_timestamp != prediction.timestamp
                or source.get("asset") != snapshot.asset or source.get("timeframe") != snapshot.timeframe
                or not event.source_bar_version
                or event.source_bar_version != snapshot.observation_metadata.get("source_bar", {}).get("bar_version")):
            reasons.append("setup_source_mismatch")
        if prediction.prediction_recorded_at_utc is None or event.observed_at > prediction.prediction_recorded_at_utc:
            reasons.append("setup_not_known_at_prediction")
        candidates.append({"event_id": event.event_id, "matched": not reasons, "reasons": reasons})
    matched_ids = [item["event_id"] for item in candidates if item["matched"]]
    reasons = []
    if strategy is None or direction != prediction.direction:
        reasons.append("setup_model_route_unverified")
        matched_ids = []
    elif not matched_ids:
        reasons.append("setup_no_selected_match")
    return {"contract_version": SETUP_MATCH_CONTRACT, "matched": bool(matched_ids),
            "matched_event_ids": matched_ids, "candidates": candidates, "reasons": reasons,
            "historical_execution_requirement": "specialist_exact_setup" if specialist else "no_pooled_setup_gate",
            "experimental_pooled_requirement": specialist is None}


def policy_identity(policy) -> str:
    return hashlib.sha256(canonical_json(policy.model_dump(mode="json")).encode()).hexdigest()


def state_payload(state: LiveAbstainState) -> dict:
    return {"last_emitted_source_row_idx": state.last_emitted_source_row_idx,
            "candidate_probabilities": list(state.candidate_probabilities)}


def restore_state(payload: dict) -> LiveAbstainState:
    return LiveAbstainState(last_emitted_source_row_idx=payload.get("last_emitted_source_row_idx"),
                            candidate_probabilities=deque(payload.get("candidate_probabilities", [])))


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _known(value) -> bool:
    return value is not None and str(value) not in {"", "nan", "None", "<NA>"}


def _scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if not _known(value) or isinstance(value, Number) and not _finite(value):
        return None
    return value


def evaluate_shadow_policy(*, prediction, live_policy, policy_context, snapshot,
                           setup_match: dict | None, threshold_passed: bool,
                           state: LiveAbstainState, force_hold_reasons=()) -> dict:
    """Evaluate every exclusion using the existing execution predicate in isolation.

    Missing inputs fail this versioned experiment closed; legacy execution's
    optional-input behavior is untouched. State advances only for fresh, verified,
    setup-matched threshold candidates, independently of execution state.
    """
    row = {} if policy_context is None or policy_context.empty else policy_context.iloc[-1].to_dict()
    abstain = live_policy.abstain_policy
    checks = {}
    rejection_reasons = []
    blank = {"enabled": True, "abstain_high_stress": False, "abstain_off_hours": False,
             "abstain_session_regimes": [], "abstain_composite_regimes": [],
             "abstain_composite_session_pairs": [], "abstain_composite_stress_pairs": [],
             "minimum_probability_quantile": None, "minimum_expected_move_to_spread": 0., "cooldown_bars": -1}
    specs = [
        ("stress", bool(abstain.abstain_high_stress), ["stress_regime"], ["abstain_high_stress"]),
        ("session", bool(abstain.abstain_off_hours or abstain.abstain_session_regimes), ["session_regime"],
         ["abstain_off_hours", "abstain_session_regimes"]),
        ("composite", bool(abstain.abstain_composite_regimes), ["composite_regime"], ["abstain_composite_regimes"]),
        ("composite_session", bool(abstain.abstain_composite_session_pairs), ["composite_regime", "session_regime"],
         ["abstain_composite_session_pairs"]),
        ("composite_stress", bool(abstain.abstain_composite_stress_pairs), ["composite_regime", "stress_regime"],
         ["abstain_composite_stress_pairs"]),
        ("probability_quantile", abstain.minimum_probability_quantile is not None, [], ["minimum_probability_quantile"]),
        ("expected_move_spread", True, ["expected_move_pips", "session_regime"], ["minimum_expected_move_to_spread"]),
        ("cooldown", True, [], ["cooldown_bars"]),
    ]
    for name, enabled, columns, settings in specs:
        values = {key: _scalar(row.get(key)) for key in columns}
        check = {"status": "disabled", "reasons": [], "inputs": values}
        checks[name] = check
        if not abstain.enabled or not enabled:
            continue
        missing = [key for key, value in values.items() if not (
            _finite(value) if key == "expected_move_pips" else _known(value))]
        if name == "cooldown" and prediction.source_row_idx is None:
            missing.append("source_row_idx")
        if missing:
            check.update(status="unverified", reasons=[f"policy_input_missing:{key}" for key in missing])
        else:
            isolated = abstain.model_copy(update={**blank, **{key: getattr(abstain, key) for key in settings}})
            result = evaluate_live_abstain(
                live_policy.model_copy(update={"abstain_policy": isolated}),
                probability=prediction.calibrated_probability,
                source_row_idx=prediction.source_row_idx if name == "cooldown" else None,
                policy_context=values, state=state if name in {"cooldown", "probability_quantile"} else None,
            )
            check.update(status="rejected" if result.abstain else "passed",
                         reasons=[result.reason] if result.reason else [])
            if name == "cooldown":
                check.update(bars_remaining=result.cooldown_bars_remaining,
                             last_policy_candidate_row=state.last_emitted_source_row_idx)
            if name == "expected_move_spread":
                spreads = {**DEFAULT_SESSION_SPREAD_PIPS, **live_policy.cost_assumptions.session_spread_pips}
                spread = spreads.get(values["session_regime"], spreads["off_hours"])
                check.update(spread_pips=_scalar(spread), required_move_pips=_scalar(abstain.minimum_expected_move_to_spread * spread),
                             spread_source="existing_policy_session_assumption")
                if not _finite(spread) or float(spread) < 0:
                    check.update(status="unverified", reasons=["policy_spread_invalid"])
        rejection_reasons.extend(check["reasons"])

    setup_match = setup_match or {"matched": False, "matched_event_ids": [], "reasons": ["setup_matching_unavailable"],
                                 "contract_version": SETUP_MATCH_CONTRACT}
    metadata = snapshot.observation_metadata
    input_ok = metadata.get("model_input_contract", {}).get("diagnostic_only") is False
    bar_ok = metadata.get("bar_eligibility", {}).get("eligible") is True
    observation_reasons = list(metadata.get("diagnostic_reasons", []))
    priority = frvp_research_priority(prediction.model_id) or ict_research_priority(prediction.model_id)
    if priority is not None and priority.candidate_block_reason:
        observation_reasons.append(priority.candidate_block_reason)
    if not input_ok:
        observation_reasons.append("model_input_contract_unverified")
    if not bar_ok:
        observation_reasons.append("bar_not_eligible")
    if not _finite(prediction.calibrated_probability) or not 0 <= prediction.calibrated_probability <= 1:
        threshold_passed = False
        observation_reasons.append("probability_invalid")
    if live_policy.policy_status != "complete":
        rejection_reasons.append("policy_provisional")
    rejection_reasons.extend(force_hold_reasons)
    before = state_payload(state)
    setup_passed = bool(threshold_passed and setup_match["matched"])
    policy_passed = bool(setup_passed and not rejection_reasons)
    eligible = bool(policy_passed and not observation_reasons)
    if setup_passed and not observation_reasons and not force_hold_reasons:
        state.remember_candidate(prediction.calibrated_probability)
    if eligible:
        state.record_emit(prediction.source_row_idx)
    reasons = [*observation_reasons, *setup_match["reasons"], *rejection_reasons]
    if not threshold_passed:
        reasons.append("probability_below_threshold")
    return {"contract_version": SHADOW_POLICY_CONTRACT, "policy_sha256": policy_identity(live_policy),
            "raw_score": _scalar(prediction.raw_score), "calibrated_probability": _scalar(prediction.calibrated_probability),
            "threshold": prediction.threshold_applied, "threshold_source": prediction.threshold_source,
            "threshold_crossed": bool(threshold_passed), "setup_match": setup_match,
            "setup_matched_decision": setup_passed, "policy_checks": checks,
            "full_policy_passed": policy_passed, "policy_candidate_eligible": eligible,
            "research_priority": priority.payload() if priority is not None else None,
            "qualified_shadow_entry": False, "qualification_status": "prerequisites_pending",
            "prerequisite_reasons": list(PREREQUISITE_REASONS),
            "force_hold_reasons": list(force_hold_reasons),
            "rejection_reasons": list(dict.fromkeys(reasons)),
            "executable_entry_at": None, "executable_entry_price": None,
            "state_before": before, "state_after": state_payload(state),
            "state_semantics": "hypothetical_policy_candidates_not_positions"}
