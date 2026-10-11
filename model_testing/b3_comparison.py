"""Predeclared B3 comparison planning and full-policy result guards.

The B3 roster is sourced from the frozen broad-selection record rather than a
hand-maintained pair of candidate IDs.  Every selected recipe receives the
same three branch declarations on one common opportunity cohort:

* setup-only;
* setup plus the development-selected, frozen threshold; and
* the intended full policy, if an exact policy binding exists.

An absent exact policy binding is ``not_defined``.  It is not a zero-trade
ledger, a zero-P&L result, or a passing full-policy evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


B3_BRANCHES = ("setup_only", "model_threshold", "full_policy")
FULL_POLICY_METRIC_FIELDS = frozenset({
    "gross_pnl", "net_pnl", "cost", "total_cost", "trades", "closed",
    "admitted", "win_rate", "profit_factor", "max_drawdown", "passed",
    "pass",
})


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_candidates(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("The frozen selection must contain a non-empty candidates list")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Every selected candidate must be an object")
        candidate_id = candidate.get("queue_id")
        if not isinstance(candidate_id, int):
            raise ValueError("Every selected candidate must have an integer queue_id")
        if candidate_id in seen:
            raise ValueError(f"Duplicate selected queue_id: {candidate_id}")
        seen.add(candidate_id)
        result.append(candidate)
    return result


def _full_policy_status(candidate: Mapping[str, Any]) -> str:
    binding = candidate.get("full_policy_binding_at_v0")
    if not binding or str(binding).startswith("not_defined"):
        return "not_defined"
    return "predeclared_exact_binding"


def build_candidate_branch_plan(candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = candidate["queue_id"]
    full_policy_status = _full_policy_status(candidate)
    return {
        "candidate_id": candidate_id,
        "target": candidate.get("target"),
        "model_id": candidate.get("model_id"),
        "order_group": candidate.get("order_group"),
        "branches": {
            "setup_only": {
                "status": "predeclared",
                "common_opportunity_universe": "shared_candidate_prediction_setup_matches",
                "admission": "all_common_opportunities",
                "pnl_reportable": True,
            },
            "model_threshold": {
                "status": "predeclared",
                "common_opportunity_universe": "shared_candidate_prediction_setup_matches",
                "admission": "frozen_development_selected_threshold",
                "threshold_rule": "selected during development and frozen before evaluation; never selected from evaluation outcomes",
                "pnl_reportable": True,
            },
            "full_policy": {
                "status": full_policy_status,
                "common_opportunity_universe": "shared_candidate_prediction_setup_matches",
                "admission": "frozen_threshold_plus_exact_candidate_policy_binding",
                "pnl_reportable": full_policy_status != "not_defined",
                "undefined_rule": (
                    "No ledger, P&L, zero-trade result, pass, or promotion claim may be emitted"
                    if full_policy_status == "not_defined" else None
                ),
            },
        },
    }


def build_b3_comparison_plan(
    protocol: Mapping[str, Any], selection: Mapping[str, Any],
    *, selection_path: Path | None = None,
) -> dict[str, Any]:
    candidates = selected_candidates(selection)
    scope = protocol.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("B3 protocol is missing scope")
    roster = scope.get("selected_recipe_roster")
    if not isinstance(roster, dict):
        raise ValueError("B3 protocol is missing selected_recipe_roster")

    ids = [candidate["queue_id"] for candidate in candidates]
    declared_ids = roster.get("candidate_ids")
    if declared_ids != ids:
        raise ValueError(f"B3 roster IDs do not match frozen selection: {declared_ids!r} != {ids!r}")
    if roster.get("candidate_count") != len(candidates):
        raise ValueError("B3 roster candidate_count does not match frozen selection")

    branches = protocol.get("comparison_branches")
    if not isinstance(branches, dict) or set(branches) != set(B3_BRANCHES):
        raise ValueError(f"B3 must declare exactly these branches: {B3_BRANCHES}")
    full_policy = branches["full_policy"]
    if not isinstance(full_policy, dict):
        raise ValueError("B3 full_policy branch must be an object")
    if full_policy.get("unbound_status") != "not_defined":
        raise ValueError("An unbound full-policy branch must be recorded as not_defined")
    if not full_policy.get("common_opportunity_universe"):
        raise ValueError("All B3 branches must name the common opportunity universe")

    streams = validate_streams(protocol, selection, candidates) if "streams" in protocol else None

    plan = {
        "protocol_id": protocol.get("protocol_id"),
        "selection_id": selection.get("selection_id"),
        "selection_sha256": None if selection_path is None else sha256_file(selection_path),
        "candidate_count": len(candidates),
        "candidate_ids": ids,
        "branches": list(B3_BRANCHES),
        "common_opportunity_rule": scope["common_opportunity_universe"]["rule"],
        "candidates": [build_candidate_branch_plan(candidate) for candidate in candidates],
    }
    if streams is not None:
        plan["streams"] = streams
        for candidate, candidate_plan in zip(candidates, plan["candidates"]):
            instrument = candidate["instrument"]
            candidate_plan["stream"] = instrument
            if protocol["streams"][instrument]["expected_move"]["status"] == "not_defined":
                # No frozen expected-move contract: an exact full policy cannot exist in this stream.
                full_policy_plan = candidate_plan["branches"]["full_policy"]
                full_policy_plan.update(build_candidate_branch_plan(
                    {**candidate, "full_policy_binding_at_v0": None})["branches"]["full_policy"])
    return plan


def validate_streams(
    protocol: Mapping[str, Any], selection: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one self-contained assumption set per instrument and no cross-stream mixing."""
    streams = protocol["streams"]
    by_instrument: dict[str, list[int]] = {}
    for candidate in candidates:
        by_instrument.setdefault(str(candidate.get("instrument")), []).append(candidate["queue_id"])
    if not isinstance(streams, dict) or set(streams) != set(by_instrument):
        raise ValueError("B3 streams must match the selected instruments exactly")
    if protocol.get("cross_stream_rules", {}).get("aggregate_across_streams") is not False:
        raise ValueError("B3 must forbid aggregation across streams")
    cost_bindings = selection.get("collection_controls", {}).get("cost_bindings", {})
    required = (
        "prospective_boundary", "execution_assumptions", "freshness_and_eligibility",
        "cost_cases", "session_reporting", "expected_move", "units", "outcome_contract_ids",
    )
    summary: dict[str, Any] = {}
    units_seen: set[str] = set()
    for instrument, stream in streams.items():
        missing = [name for name in required if not stream.get(name)]
        if missing:
            raise ValueError(f"B3 {instrument} stream is missing: {', '.join(missing)}")
        if stream.get("instrument") != instrument or stream.get("candidate_ids") != by_instrument[instrument]:
            raise ValueError(f"B3 {instrument} stream candidates do not match the frozen selection")
        cost_model = stream["cost_cases"].get("instrument_cost_model")
        if cost_model != cost_bindings.get(instrument) or cost_model != stream.get("cost_model_id"):
            raise ValueError(f"B3 {instrument} stream does not use its frozen cost contract")
        prefix = instrument.lower() + "_"
        if not all(str(contract_id).startswith(prefix) for contract_id in stream["outcome_contract_ids"]):
            raise ValueError(f"B3 {instrument} stream names another instrument's outcome contract")
        unit = stream["units"].get("markout_and_pnl")
        if not unit or unit in units_seen:
            raise ValueError("Each B3 stream must report in its own unit")
        units_seen.add(unit)
        if stream["expected_move"].get("status") not in {"predeclared", "not_defined"}:
            raise ValueError(f"B3 {instrument} expected-move status must be predeclared or not_defined")
        if (stream["expected_move"]["status"] == "predeclared") != bool(stream["expected_move"].get("expected_move_contract")):
            raise ValueError(f"B3 {instrument} expected-move contract and status disagree")
        summary[instrument] = {
            "candidate_ids": by_instrument[instrument],
            "cost_model_id": cost_model,
            "unit": unit,
            "expected_move_status": stream["expected_move"]["status"],
        }
    return summary


def not_defined_full_policy_result(reason: str) -> dict[str, Any]:
    """Return the only valid shape for an unbound full-policy result."""
    return {
        "status": "not_defined",
        "reason": reason,
        "pnl_reportable": False,
        "ledger_status": "not_created",
        "promotion_eligible": False,
    }


def validate_full_policy_result(result: Mapping[str, Any]) -> None:
    """Reject metric-bearing or passing output for a not-defined policy."""
    if result.get("status") != "not_defined":
        return
    forbidden = sorted(field for field in FULL_POLICY_METRIC_FIELDS if field in result)
    if forbidden:
        raise ValueError(
            "A not_defined full-policy result cannot contain metrics or pass fields: "
            + ", ".join(forbidden)
        )
    if result.get("pnl_reportable") is not False or result.get("ledger_status") != "not_created":
        raise ValueError("A not_defined full-policy result must be non-reportable and have no ledger")

