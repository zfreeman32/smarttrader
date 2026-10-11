from __future__ import annotations

import json

import pytest

from model_testing.outcome_routes import (
    ROUTE_BINDINGS_V1,
    ROUTE_BINDINGS_V2,
    load_route_bindings,
    resolve_model_setup_route,
    route_binding_audit,
)


FRVP_CONTINUATION = {"setup2", "setup3", "setup5"}
FRVP_REVERSAL = {"setup1", "setup4", "setup6"}
FRVP_META = FRVP_CONTINUATION | FRVP_REVERSAL
ICT_REVERSAL = {
    "sweep_reclaim",
    "sweep_displacement_fvg",
    "ob_retest_after_mss",
    "ifvg_reversal",
    "session_open_manipulation_pre_ib",
    "session_open_manipulation_post_ib",
}
ICT_CONTINUATION = {
    "premium_discount_continuation",
    "displacement_continuation_after_raid",
}
OTE_FAMILIES = {"reversal", "continuation", "breakout", "breakout_sustained"}


# path -> (selected models, routes, ES routes)
EXPECTED_COUNTS = {ROUTE_BINDINGS_V1: (12, 51, 41), ROUTE_BINDINGS_V2: (16, 59, 45)}


@pytest.fixture(params=[ROUTE_BINDINGS_V1, ROUTE_BINDINGS_V2], ids=["v1", "v2"])
def route_path(request):
    return request.param


@pytest.fixture
def manifest(route_path):
    return load_route_bindings(route_path)


def _bindings(manifest):
    return {binding.model_id: binding for binding in manifest.bindings}


def test_selected_roster_has_complete_exact_setup_routes(manifest):
    bindings = _bindings(manifest)
    assert set(bindings["frvp_long_continuation_xgb_v1"].setup_bindings) == FRVP_CONTINUATION
    assert set(bindings["frvp_long_reversal_xgb_v1"].setup_bindings) == FRVP_REVERSAL
    assert set(bindings["frvp_long_meta_xgb_v1"].setup_bindings) == FRVP_META
    for model_id in ("ict_long_meta_xgb_v1", "ict_short_meta_xgb_v1"):
        assert set(bindings[model_id].setup_bindings) == ICT_REVERSAL | ICT_CONTINUATION
    for model_id in ("ict_long_reversal_xgb_v1", "ict_short_reversal_xgb_v1"):
        assert set(bindings[model_id].setup_bindings) == ICT_REVERSAL
    assert bindings["ict_short_reversal_sweep_reclaim_xgb_v1"].setup_bindings == {
        "sweep_reclaim": "es_ict_sweep_reclaim_eval_v1"
    }
    assert all("classic_breaker" not in binding.setup_bindings for binding in bindings.values())


def test_every_es_route_uses_existing_b1_strategy_and_es_cost_contract(route_path, manifest):
    audit = route_binding_audit(manifest)
    es_rows = [row for row in audit["rows"] if row["instrument"] == "ES"]
    assert (audit["selected_model_count"], audit["route_count"], len(es_rows)) == EXPECTED_COUNTS[route_path]
    assert {row["strategy"] for row in es_rows} == {"frvp", "ict"}
    assert {row["cost_model_id"] for row in es_rows} == {"es_session_cost_v1"}
    assert all(row["outcome_contract_id"].startswith("es_") for row in es_rows)


def test_ote_reversal_and_meta_reuse_family_v2_contracts_and_eurusd_costs(manifest):
    bindings = _bindings(manifest)
    for model_id in (
        "long_reversal_tcn_v2_20260525_narrow48",
        "short_reversal_xgb_v2_20260525",
    ):
        assert bindings[model_id].setup_bindings == {
            "reversal": "eurusd_ote_reversal_eval_v2"
        }
    pooled = [model_id for model_id in (
        "long_ote_meta_tcn_champion", "short_ote_meta_tcn_champion", "long_union_tcn_champion",
    ) if model_id in bindings]
    assert len(pooled) == (3 if manifest.version == 2 else 2)
    for model_id in pooled:
        assert set(bindings[model_id].setup_bindings) == OTE_FAMILIES
        for setup_family, contract_id in bindings[model_id].setup_bindings.items():
            assert contract_id == f"eurusd_ote_{setup_family}_eval_v2"
            assert "meta" not in contract_id and "union" not in contract_id
            contract = resolve_model_setup_route(
                model_id=model_id,
                setup_family=setup_family,
                instrument="EURUSD",
                side=bindings[model_id].side,
                manifest=manifest,
            )
            assert contract.setup_family == setup_family
            assert contract.costs.cost_model_id == "eurusd_session_cost_v1"


def test_v2_carries_v1_routes_forward_and_adds_extension_routes():
    v1, v2 = load_route_bindings(ROUTE_BINDINGS_V1), load_route_bindings(ROUTE_BINDINGS_V2)
    assert v2.bindings[:len(v1.bindings)] == v1.bindings
    added = {binding.queue_id: binding for binding in v2.bindings[len(v1.bindings):]}
    assert set(added) == {5, 23, 18, 14}
    assert (added[5].route_selection, added[5].setup_bindings) == (
        "exact_contemporaneous_same_side", {"setup2": "es_frvp_setup2_eval_v2"})
    assert (added[18].route_selection, added[18].side, added[18].setup_bindings) == (
        "exact_contemporaneous_same_side", "short", {"ifvg_reversal": "es_ict_ifvg_reversal_eval_v1"})
    assert added[14].side == "short" and set(added[14].setup_bindings) == ICT_CONTINUATION
    long_meta = _bindings(v1)["long_ote_meta_tcn_champion"]
    assert (added[23].instrument, added[23].side, added[23].model_family) == ("EURUSD", "long", "union")
    assert added[23].route_selection == long_meta.route_selection
    assert added[23].setup_bindings == long_meta.setup_bindings
    with pytest.raises(ValueError, match="Undeclared setup route"):
        resolve_model_setup_route(
            model_id="ict_short_reversal_ifvg_reversal_xgb_v1",
            setup_family="sweep_reclaim",
            instrument="ES",
            side="short",
            manifest=v2,
        )
    with pytest.raises(ValueError, match="instrument/side mismatch"):
        resolve_model_setup_route(
            model_id="long_union_tcn_champion",
            setup_family="reversal",
            instrument="ES",
            side="long",
            manifest=v2,
        )
    with pytest.raises(ValueError, match="No unique frozen model route"):
        resolve_model_setup_route(
            model_id="long_union_tcn_champion",
            setup_family="reversal",
            instrument="EURUSD",
            side="long",
            manifest=v1,
        )


def test_route_resolution_fails_closed_on_undeclared_setup_or_identity(manifest):
    with pytest.raises(ValueError, match="Undeclared setup route"):
        resolve_model_setup_route(
            model_id="long_reversal_tcn_v2_20260525_narrow48",
            setup_family="continuation",
            instrument="EURUSD",
            side="long",
            manifest=manifest,
        )
    with pytest.raises(ValueError, match="instrument/side mismatch"):
        resolve_model_setup_route(
            model_id="frvp_long_continuation_xgb_v1",
            setup_family="setup2",
            instrument="EURUSD",
            side="long",
            manifest=manifest,
        )
    with pytest.raises(ValueError, match="No unique frozen model route"):
        resolve_model_setup_route(
            model_id="unknown_model",
            setup_family="setup2",
            instrument="ES",
            side="long",
            manifest=manifest,
        )


def test_route_manifest_is_lock_protected(tmp_path, route_path):
    changed = json.loads(route_path.read_text(encoding="utf-8"))
    changed["bindings"][0]["setup_bindings"]["setup2"] = "es_frvp_setup1_eval_v1"
    candidate = tmp_path / route_path.name
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    candidate.with_suffix(".lock.json").write_text(
        route_path.with_suffix(".lock.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Frozen route manifest changed"):
        load_route_bindings(candidate)
