"""Immutable model/setup-to-outcome bindings for the broad shadow roster."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from model_testing.outcome_contracts import (
    CATALOG,
    ROOT,
    FrozenModel,
    OutcomeContract,
    file_hash,
    load_contracts,
    resolve_contract,
)
from model_testing.outcome_diagnostics import POOLED_ROUTING_CONTRACT


ROUTE_BINDINGS_V1 = ROOT / "configs" / "broad_full_session_shadow_route_bindings_v1.json"
ROUTE_BINDINGS_V2 = ROOT / "configs" / "broad_full_session_shadow_route_bindings_v2.json"
ROUTE_BINDINGS = ROUTE_BINDINGS_V2


class ModelRouteBinding(FrozenModel):
    queue_id: int = Field(gt=0)
    model_id: str
    instrument: Literal["ES", "EURUSD"]
    strategy: Literal["frvp", "ict", "ote"]
    side: Literal["long", "short"]
    model_family: str
    route_selection: Literal[
        "detector_selected_contemporaneous_same_side",
        "exact_contemporaneous_same_side",
    ]
    setup_bindings: dict[str, str]

    @model_validator(mode="after")
    def complete(self):
        if not self.setup_bindings:
            raise ValueError("A selected model must bind at least one setup route")
        if any(not setup or not contract_id for setup, contract_id in self.setup_bindings.items()):
            raise ValueError("Setup route names and contract IDs cannot be empty")
        if self.instrument == "ES" and self.strategy not in {"frvp", "ict"}:
            raise ValueError("Every ES route must use the FRVP or ICT contract family")
        if self.instrument == "EURUSD" and self.strategy != "ote":
            raise ValueError("Every EURUSD route in this roster must use the OTE contract family")
        return self


class OutcomeRouteManifest(FrozenModel):
    binding_id: str
    version: int = Field(ge=1)
    selected_collection_id: str
    selection_path: str
    selection_sha256: str = Field(min_length=64, max_length=64)
    catalog_path: str
    catalog_sha256: str = Field(min_length=64, max_length=64)
    catalog_lock_sha256: str = Field(min_length=64, max_length=64)
    pooled_routing_contract: Literal["pooled_detector_selected_same_side_v1"]
    bindings: tuple[ModelRouteBinding, ...]

    @model_validator(mode="after")
    def identity(self):
        if not self.binding_id.endswith(f"_v{self.version}"):
            raise ValueError("Route binding ID must include its version")
        model_ids = [binding.model_id for binding in self.bindings]
        queue_ids = [binding.queue_id for binding in self.bindings]
        if len(model_ids) != len(set(model_ids)) or len(queue_ids) != len(set(queue_ids)):
            raise ValueError("Model and queue IDs must be unique")
        return self


def _root_path(relative: str) -> Path:
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Manifest path escapes repository root: {relative}")
    return path


def load_route_bindings(path: str | Path = ROUTE_BINDINGS) -> OutcomeRouteManifest:
    """Load the frozen route manifest and verify every transitive identity pin."""
    path = Path(path)
    manifest = OutcomeRouteManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    pins = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    if pins != {manifest.binding_id: file_hash(path)}:
        raise ValueError("Frozen route manifest changed; publish a new version")
    selection_path = _root_path(manifest.selection_path)
    catalog_path = _root_path(manifest.catalog_path)
    if file_hash(selection_path) != manifest.selection_sha256:
        raise ValueError("Frozen selected roster changed")
    if file_hash(catalog_path) != manifest.catalog_sha256:
        raise ValueError("Frozen outcome catalogue changed")
    if file_hash(catalog_path.with_suffix(".lock.json")) != manifest.catalog_lock_sha256:
        raise ValueError("Frozen outcome catalogue lock changed")
    if catalog_path != CATALOG:
        raise ValueError("Broad shadow routes must bind the active B1 catalogue")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_id") != manifest.selected_collection_id:
        raise ValueError("Selected collection ID does not match its pinned roster")
    selected = {
        row["model_id"]: (row["queue_id"], row["instrument"])
        for row in selection.get("replacement_bindings", [])
    }
    bound = {row.model_id: (row.queue_id, row.instrument) for row in manifest.bindings}
    if selected != bound:
        raise ValueError("Route bindings must cover the selected roster exactly")

    contracts = load_contracts(catalog_path)
    for binding in manifest.bindings:
        expected_cost = "es_session_cost_v1" if binding.instrument == "ES" else "eurusd_session_cost_v1"
        for setup_family, contract_id in binding.setup_bindings.items():
            contract = resolve_contract(
                instrument=binding.instrument,
                strategy=binding.strategy,
                model_family=binding.model_family,
                setup_family=setup_family,
                side=binding.side,
                contracts=contracts,
            )
            if contract.contract_id != contract_id:
                raise ValueError(
                    f"Route {binding.model_id}/{setup_family} resolves to {contract.contract_id}, "
                    f"not frozen {contract_id}"
                )
            if contract.costs.cost_model_id != expected_cost:
                raise ValueError(f"Wrong cost contract for {binding.model_id}/{setup_family}")
            if (binding.strategy == "ote" and binding.model_family in {"meta", "union"}
                    and binding.model_family in contract.contract_id):
                raise ValueError("OTE meta/union must reuse the causal setup-family exit contract")
    return manifest


def resolve_model_setup_route(
    *,
    model_id: str,
    setup_family: str,
    instrument: str,
    side: Literal["long", "short"],
    manifest: OutcomeRouteManifest | None = None,
    contracts: dict[str, OutcomeContract] | None = None,
) -> OutcomeContract:
    """Resolve one selected route and fail closed on any undeclared setup."""
    manifest = manifest or load_route_bindings()
    matches = [binding for binding in manifest.bindings if binding.model_id == model_id]
    if len(matches) != 1:
        raise ValueError(f"No unique frozen model route for {model_id}")
    binding = matches[0]
    if (binding.instrument, binding.side) != (instrument, side):
        raise ValueError(f"Frozen instrument/side mismatch for {model_id}")
    try:
        frozen_contract_id = binding.setup_bindings[setup_family]
    except KeyError as exc:
        raise ValueError(f"Undeclared setup route for {model_id}/{setup_family}") from exc
    contract_set = contracts or load_contracts()
    contract = resolve_contract(
        instrument=instrument,
        strategy=binding.strategy,
        model_family=binding.model_family,
        setup_family=setup_family,
        side=side,
        contracts=contract_set,
    )
    if contract.contract_id != frozen_contract_id:
        raise ValueError(f"Resolved contract does not match frozen route for {model_id}/{setup_family}")
    return contract


def route_binding_audit(manifest: OutcomeRouteManifest | None = None) -> dict:
    manifest = manifest or load_route_bindings()
    contracts = load_contracts()
    rows = []
    for binding in manifest.bindings:
        for setup_family in binding.setup_bindings:
            contract = resolve_model_setup_route(
                model_id=binding.model_id,
                setup_family=setup_family,
                instrument=binding.instrument,
                side=binding.side,
                manifest=manifest,
                contracts=contracts,
            )
            rows.append({
                "queue_id": binding.queue_id,
                "model_id": binding.model_id,
                "instrument": binding.instrument,
                "side": binding.side,
                "strategy": binding.strategy,
                "model_family": binding.model_family,
                "setup_family": setup_family,
                "route_selection": binding.route_selection,
                "outcome_contract_id": contract.contract_id,
                "outcome_contract_sha256": contract.sha256,
                "cost_model_id": contract.costs.cost_model_id,
                "cost_contract_sha256": contract.costs.sha256,
            })
    return {
        "binding_id": manifest.binding_id,
        "binding_sha256": manifest.sha256,
        "selected_model_count": len(manifest.bindings),
        "route_count": len(rows),
        "pooled_routing_contract": POOLED_ROUTING_CONTRACT,
        "all_routes_valid": True,
        "rows": rows,
    }
