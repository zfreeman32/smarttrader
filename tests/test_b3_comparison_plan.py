import pytest
from pathlib import Path

from model_testing.b3_comparison import (
    B3_BRANCHES,
    build_b3_comparison_plan,
    load_json,
    not_defined_full_policy_result,
    validate_full_policy_result,
)


ROOT = Path(__file__).resolve().parents[1]


def test_b3_declares_all_frozen_selection_recipes_and_three_branches():
    protocol_path = ROOT / "configs/b3_b4_candidate_review_v1_20260929.json"
    selection_path = ROOT / "configs/broad_full_session_shadow_selection_v0.json"
    plan = build_b3_comparison_plan(load_json(protocol_path), load_json(selection_path), selection_path=selection_path)

    assert plan["candidate_count"] == 22
    assert plan["branches"] == list(B3_BRANCHES)
    assert len(plan["candidates"]) == 22
    for candidate in plan["candidates"]:
        assert set(candidate["branches"]) == set(B3_BRANCHES)
        assert candidate["branches"]["setup_only"]["common_opportunity_universe"] == "shared_candidate_prediction_setup_matches"
        assert candidate["branches"]["model_threshold"]["common_opportunity_universe"] == "shared_candidate_prediction_setup_matches"
        assert candidate["branches"]["full_policy"]["status"] == "not_defined"
        assert candidate["branches"]["full_policy"]["pnl_reportable"] is False


def _v2():
    protocol_path = ROOT / "configs/b3_b4_candidate_review_v2_20261002.json"
    selection_path = ROOT / "configs/broad_full_session_shadow_selection_v1.json"
    return protocol_path, selection_path, load_json(protocol_path), load_json(selection_path)


def test_b3_v2_declares_separate_es_and_eurusd_streams_for_the_16_model_roster():
    from model_testing.b3_comparison import sha256_file
    from model_testing.outcome_routes import ROUTE_BINDINGS_V2, load_route_bindings, route_binding_audit

    protocol_path, selection_path, protocol, selection = _v2()
    lock = load_json(protocol_path.with_suffix(".lock.json"))
    assert lock[protocol["protocol_id"]] == sha256_file(protocol_path)
    assert lock[selection["selection_id"]] == sha256_file(selection_path)
    assert lock["broad_full_session_shadow_route_bindings_v2"] == sha256_file(ROUTE_BINDINGS_V2)
    v1_path = ROOT / protocol["parent_protocol"]
    assert protocol["parent_protocol_sha256"] == sha256_file(v1_path) == lock["b3_b4_candidate_review_v1_20260929"]

    plan = build_b3_comparison_plan(protocol, selection, selection_path=selection_path)
    assert plan["candidate_count"] == 16
    assert plan["streams"]["ES"]["candidate_ids"] == [2, 10, 12, 13, 15, 16, 3, 19, 5, 18, 14]
    assert plan["streams"]["EURUSD"]["candidate_ids"] == [22, 26, 21, 25, 23]
    assert (plan["streams"]["ES"]["cost_model_id"], plan["streams"]["ES"]["unit"]) == ("es_session_cost_v1", "ES ticks")
    assert (plan["streams"]["EURUSD"]["cost_model_id"], plan["streams"]["EURUSD"]["unit"]) == ("eurusd_session_cost_v1", "pips")
    assert plan["streams"]["EURUSD"]["expected_move_status"] == "not_defined"
    for candidate in plan["candidates"]:
        assert set(candidate["branches"]) == set(B3_BRANCHES)
        assert candidate["branches"]["full_policy"]["status"] == "not_defined"

    # ES assumptions are the v1 text; each stream binds exactly its routed frozen contracts.
    v1, es, fx = load_json(v1_path), protocol["streams"]["ES"], protocol["streams"]["EURUSD"]
    for section in ("execution_assumptions", "freshness_and_eligibility", "cost_cases", "session_reporting"):
        assert es[section] == v1[section]
    rows = route_binding_audit(load_route_bindings(ROUTE_BINDINGS_V2))["rows"]
    for instrument, stream in protocol["streams"].items():
        routed = sorted({row["outcome_contract_id"] for row in rows if row["instrument"] == instrument})
        assert stream["outcome_contract_ids"] == routed
    assert fx["execution_assumptions"]["portfolio"]["accounting_unit"] == "pips"
    assert fx["execution_assumptions"]["portfolio"]["starting_equity_usd"] is None
    assert "ES" not in fx["expected_move"]["expected_move_units"]


def test_b3_v2_rejects_cross_stream_mixing():
    _, _, protocol, selection = _v2()

    def changed(mutate):
        value = load_json(ROOT / "configs/b3_b4_candidate_review_v2_20261002.json")
        mutate(value)
        return value

    cases = {
        "frozen cost contract": lambda p: p["streams"]["EURUSD"]["cost_cases"].update(instrument_cost_model="es_session_cost_v1"),
        "forbid aggregation": lambda p: p["cross_stream_rules"].update(aggregate_across_streams=True),
        "candidates do not match": lambda p: p["streams"]["ES"]["candidate_ids"].append(23),
        "another instrument's outcome contract": lambda p: p["streams"]["EURUSD"]["outcome_contract_ids"].append("es_frvp_setup2_eval_v2"),
        "its own unit": lambda p: p["streams"]["EURUSD"]["units"].update(markout_and_pnl="ES ticks"),
        "contract and status disagree": lambda p: p["streams"]["EURUSD"]["expected_move"].update(expected_move_contract="es_abs_close_move_20x5m_v1"),
        "match the selected instruments": lambda p: p["streams"].pop("EURUSD"),
    }
    for message, mutate in cases.items():
        with pytest.raises(ValueError, match=message):
            build_b3_comparison_plan(changed(mutate), selection)


def test_not_defined_full_policy_cannot_be_reported_as_zero_or_pass():
    result = not_defined_full_policy_result("missing exact binding")
    validate_full_policy_result(result)

    with pytest.raises(ValueError, match="metrics or pass fields"):
        validate_full_policy_result({**result, "net_pnl": 0.0})

    with pytest.raises(ValueError, match="no ledger"):
        validate_full_policy_result({**result, "ledger_status": "empty"})
