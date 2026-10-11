"""Section 3 surfaces and ledgers (B3 v3): one schema for offline replay and the app.

Event markouts (surface 1 and the per-opportunity paper outcome, including pooled
counterfactuals flagged non-executable) live in markout stores. Positions live only
in ledgers: one independent ledger per model x variant x cost case, and one
separately named instrument-capacity ledger per stream x cost case. ES and EURUSD
are built, persisted and reported separately. Surfaces 1-3 and every ledger apply
no hour or session filter; a frozen full-policy exclusion is only ever a recorded
rejection reason in surface 4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from model_testing.b3_comparison import not_defined_full_policy_result, validate_full_policy_result
from model_testing.outcome_contracts import PositionContract, content_hash, file_hash, load_contracts
from model_testing.outcome_diagnostics import build_model_setup_audit, resolve_pooled_routing
from model_testing.outcome_engine import (
    COST_CASES, ENGINE_VERSION, EvaluationTape, SignalEvidence, evaluate_signal, reprice_positions,
    session_at, simulate_positions, utc,
)
from model_testing.outcome_routes import OutcomeRouteManifest, load_route_bindings, resolve_model_setup_route
from model_testing.outcome_scope import BaselineScope, load_research_scope


SURFACE_SCHEMA_VERSION = "section3-surfaces-v1"
SURFACE_1 = "raw_score_gross_forward_response"
POSITION_VARIANTS = ("setup_only", "setup_threshold", "full_policy")
SURFACES = (SURFACE_1,) + POSITION_VARIANTS
LEDGER_KINDS = ("independent_model_variant", "instrument_capacity")
STORE_KINDS = ("markout_store", "position_ledger")
STREAM_UNITS = {"ES": ("ticks", "cash"), "EURUSD": ("pips",)}
ATTRIBUTION_FIELDS = (
    "stream", "queue_id", "model_id", "model_version_hash", "route_id", "setup_event_id",
    "outcome_contract_id", "outcome_contract_hash", "cost_contract_id", "cost_case",
    "surface", "ledger_kind", "ledger_id", "engine_version",
)
REPORT_SLICE_FIELDS = ("decision_session", "decision_hour_new_york")
APP_VARIANT_ALIASES = {"setup_full_policy": "full_policy"}
NOT_DEFINED = "not_defined"


class SurfaceReportError(ValueError):
    """A report tried to mix stores, ledger kinds, streams or cost cases, or to total not_defined."""


@dataclass(frozen=True)
class FullPolicyBinding:
    """An exact frozen full policy: identity plus recorded rejection reasons per event ID."""
    policy_id: str
    policy_hash: str
    rejections: Mapping[str, tuple[str, ...]]


def surface_schema() -> dict:
    """The versioned schema both the offline replay and the app report."""
    return dict(schema_version=SURFACE_SCHEMA_VERSION, engine_version=ENGINE_VERSION, surfaces=list(SURFACES),
                position_variants=list(POSITION_VARIANTS), ledger_kinds=list(LEDGER_KINDS),
                store_kinds=list(STORE_KINDS), attribution_fields=list(ATTRIBUTION_FIELDS),
                report_slice_fields=list(REPORT_SLICE_FIELDS), cost_cases=list(COST_CASES),
                stream_units={stream: list(units) for stream, units in STREAM_UNITS.items()},
                app_variant_aliases=dict(APP_VARIANT_ALIASES), scored_collection_enabled=False)


def _checked(stream, cost_case=None, variant=None):
    if stream not in STREAM_UNITS or (cost_case is not None and cost_case not in COST_CASES) or (
            variant is not None and variant not in POSITION_VARIANTS):
        raise ValueError("Unknown stream, cost case or variant")


def markout_store_id(stream: str, queue_id: int, surface: str = SURFACE_1) -> str:
    _checked(stream)
    return f"mk.{stream}.{queue_id}.{surface}"


def independent_ledger_id(stream: str, queue_id: int, variant: str, cost_case: str) -> str:
    _checked(stream, cost_case, variant)
    return f"ind.{stream}.{queue_id}.{variant}.{cost_case}"


def capacity_ledger_id(stream: str, cost_case: str) -> str:
    _checked(stream, cost_case)
    return f"cap.{stream}.instrument_policy.{cost_case}"


def report_slices(known_at) -> dict:
    """Session and hour are report-slice fields only; nothing here admits or rejects on them."""
    known = utc(known_at)
    return {"decision_session": session_at(known),
            "decision_hour_new_york": known.tz_convert("America/New_York").hour}


def _eligibility(signal, variant, threshold, policy):
    if variant == "setup_only":
        return True, ""
    context = signal.audit_context
    if context is None or context.calibrated_probability is None:
        return False, "calibrated_probability_missing"
    if context.calibrated_probability < threshold:  # equality is accepted
        return False, "probability_below_frozen_threshold"
    if variant == "setup_threshold":
        return True, ""
    reasons = tuple(policy.rejections.get(signal.event_id, ()))
    return not reasons, "|".join(reasons)


def _setup_event_id(signal):
    context = signal.audit_context
    if context is None:
        return None
    return resolve_pooled_routing(context, strategy=signal.strategy, setup_family=signal.setup_family,
                                  side=signal.side, decision_at=signal.known_at_ts).primary_setup_event_id


def build_stream_surfaces(stream: str, signals: list[SignalEvidence], tape: EvaluationTape, *,
                          thresholds: Mapping[str, float], model_hashes: Mapping[str, str],
                          selection_order: list[int], protocol_sha256: str,
                          full_policy: Mapping[str, FullPolicyBinding | None] | None = None,
                          unit: str | None = None, manifest: OutcomeRouteManifest | None = None,
                          scope: BaselineScope | None = None, contracts: dict | None = None) -> dict:
    """Route -> four surfaces -> both ledger kinds for one stream. Never call with two streams."""
    unit = unit or STREAM_UNITS[stream][0]
    if unit not in STREAM_UNITS[stream]:
        raise ValueError(f"{unit} is not a {stream} unit")
    if not tape.require_completion:
        raise ValueError("Section 3 surfaces require a completion-enforcing tape")
    if len({s.event_id for s in signals}) != len(signals):
        raise ValueError("Event IDs must be unique within a stream")
    manifest = manifest or load_route_bindings()
    contracts = contracts or load_contracts()
    scope = scope or load_research_scope()
    full_policy = dict(full_policy or {})
    bindings = {b.model_id: b for b in manifest.bindings if b.instrument == stream}
    rank = {queue_id: index for index, queue_id in enumerate(selection_order)}
    used, outcomes, attribution, by_model = {}, {}, {}, {}
    for signal in signals:
        if signal.instrument != stream or signal.model_id not in bindings:
            raise ValueError(f"{signal.model_id} is not a selected {stream} model")
        binding = bindings[signal.model_id]
        contract = resolve_model_setup_route(model_id=signal.model_id, setup_family=signal.setup_family,
            instrument=stream, side=signal.side, manifest=manifest, contracts=contracts)
        used[contract.contract_id] = contract
        outcomes[signal.event_id] = evaluate_signal(signal, tape, contract, scope=scope)
        attribution[signal.event_id] = dict(stream=stream, queue_id=binding.queue_id, model_id=signal.model_id,
            model_version_hash=model_hashes[signal.model_id],
            route_id=f"{binding.queue_id}:{signal.setup_family}:{contract.contract_id}",
            setup_event_id=_setup_event_id(signal), outcome_contract_id=contract.contract_id,
            outcome_contract_hash=contract.sha256, cost_contract_id=contract.costs.cost_model_id,
            engine_version=ENGINE_VERSION)
        by_model.setdefault(signal.model_id, []).append(signal)

    def variant_signals(model_signals, variant):
        result = []
        for signal in model_signals:
            allowed, reason = _eligibility(signal, variant, thresholds[signal.model_id],
                                           full_policy.get(signal.model_id))
            result.append(signal.model_copy(update={"policy_eligible": allowed, "policy_rejection_reason": reason}))
        return result

    def simulator(ledger_id, kind):
        return PositionContract(simulator_id=f"{ledger_id}_v1", version=1, policy_id=ledger_id,
            policy_hash=content_hash({"protocol_sha256": protocol_sha256, "ledger_id": ledger_id}),
            max_positions=1, max_quantity=1.0, allow_opposite=False, cooldown_minutes=0, starting_equity=100000.0,
            concurrency_scope="model" if kind == LEDGER_KINDS[0] else "instrument", accounting_unit=unit)

    def ledger_frame(positions, *, kind, ledger_id, variants):
        rows = []
        for sequence, position in enumerate(positions):
            event_id, paper = position["event_id"], outcomes[position["event_id"]]
            rows.append({**attribution[event_id], "cost_case": position["cost_case"], "surface": variants[event_id],
                "ledger_kind": kind, "ledger_id": ledger_id, "store_kind": "position_ledger", "unit": unit,
                "schema_version": SURFACE_SCHEMA_VERSION, "admission_sequence": sequence, "side": paper["side"],
                "known_at_ts": paper["known_at_ts"], "entry_ts": paper["entry_ts"], "entry_price": paper["entry_price"],
                "paper_contract_outcome": paper["paper_contract_outcome"], "exit_price": paper["paper_exit_price"],
                **report_slices(paper["known_at_ts"]),
                **{key: value for key, value in position.items() if key != "cost_case"}})
        return pd.DataFrame(rows)

    def both_cost_cases(positions, ledger_of, **frame):
        return {ledger_of(case): ledger_frame(positions if case == "base" else reprice_positions(positions, case),
                                               ledger_id=ledger_of(case), **frame) for case in COST_CASES}

    markouts, ledgers, surface_4, feed = {}, {}, {}, []
    for model_id, model_signals in sorted(by_model.items(), key=lambda item: rank[bindings[item[0]].queue_id]):
        queue_id = bindings[model_id].queue_id
        model_outcomes = [outcomes[s.event_id] for s in model_signals]
        store_id = markout_store_id(stream, queue_id)
        store = dict(surface=SURFACE_1, ledger_kind="not_a_ledger", ledger_id=store_id, cost_case="none",
                     store_kind="markout_store", unit=unit, schema_version=SURFACE_SCHEMA_VERSION)
        forward, _ = build_model_setup_audit(model_signals, model_outcomes)
        forward = forward.assign(**{key: forward["event_id"].map(lambda e, k=key: attribution[e][k])
                                    for key in attribution[model_signals[0].event_id]}, **store)
        # Row grain is prediction x observed setup: the row's setup is the observed one.
        forward["setup_event_id"] = forward["observed_setup_id"]
        paper = pd.DataFrame([{**attribution[r["event_id"]], **store,
                               "executable": r["baseline_scope"] == "included",
                               **{k: v for k, v in r.items() if not k.startswith(("forward_", "training_label_"))}}
                              for r in model_outcomes])
        markouts[store_id] = {"forward_response": forward, "paper_contract_outcome": paper}
        policy = full_policy.get(model_id)
        for variant in POSITION_VARIANTS:
            if variant == "full_policy" and policy is None:
                result = not_defined_full_policy_result("No exact full-policy binding is frozen for this model.")
                validate_full_policy_result(result)
                surface_4[queue_id] = result
                continue
            if variant == "full_policy":
                surface_4[queue_id] = dict(status="exact", policy_id=policy.policy_id, policy_hash=policy.policy_hash)
            eligible = variant_signals(model_signals, variant)
            ledger_of = lambda case, v=variant: independent_ledger_id(stream, queue_id, v, case)
            positions = simulate_positions(eligible, model_outcomes, used, simulator(ledger_of("base"), LEDGER_KINDS[0]),
                                           scope=scope)
            ledgers.update(both_cost_cases(positions, ledger_of, kind=LEDGER_KINDS[0],
                                           variants={s.event_id: variant for s in eligible}))
        # Capacity feed: the full policy where exact, otherwise setup_threshold, labeled on every entry.
        feed_variant = "setup_threshold" if policy is None else "full_policy"
        feed += [(signal, feed_variant) for signal in variant_signals(model_signals, feed_variant)]
    if feed:
        ledger_of = lambda case: capacity_ledger_id(stream, case)
        positions = simulate_positions([s for s, _ in feed], [outcomes[s.event_id] for s, _ in feed], used,
            simulator(ledger_of("base"), LEDGER_KINDS[1]), scope=scope,
            priority={model_id: rank[binding.queue_id] for model_id, binding in bindings.items()
                      if binding.queue_id in rank})
        ledgers.update(both_cost_cases(positions, ledger_of, kind=LEDGER_KINDS[1],
                                       variants={s.event_id: variant for s, variant in feed}))
    return dict(schema_version=SURFACE_SCHEMA_VERSION, engine_version=ENGINE_VERSION, stream=stream, unit=unit,
                protocol_sha256=protocol_sha256, route_binding_id=manifest.binding_id, scope_id=scope.scope_id,
                calendar_id=tape.calendar_id, calendar_hash=tape.schedule_hash,
                markouts=markouts, ledgers=ledgers, full_policy=surface_4)


def guard_report_table(table) -> pd.DataFrame:
    """Fail closed unless the table is one store kind, one ledger kind, one stream and one cost case."""
    if isinstance(table, Mapping):
        if table.get("status") == NOT_DEFINED:
            raise SurfaceReportError("not_defined has no ledger and is never rendered as zero")
        raise SurfaceReportError("A report table must be one markout store or one ledger")
    for column, message in (("store_kind", "markouts and positions are never placed in one table"),
                            ("stream", "ES and EURUSD are never summed, ranked or pooled"),
                            ("unit", "ES and EURUSD are never summed, ranked or pooled"),
                            ("ledger_kind", "independent and instrument-capacity ledgers are never mixed"),
                            ("cost_case", "cost cases are never mixed")):
        if column not in table:
            raise SurfaceReportError(f"Report table has no {column} column")
        if table[column].nunique(dropna=False) > 1:
            raise SurfaceReportError(message)
    return table


def ledger_summary(table) -> dict:
    """Totals for exactly one ledger. Unresolved positions are open exposure, never zero P&L."""
    table = guard_report_table(table)
    if len(table) and (table["store_kind"].iloc[0] != "position_ledger" or table["ledger_id"].nunique() != 1):
        raise SurfaceReportError("Totals exist only for a single position ledger")
    closed = table.loc[table["position_status"].eq("CLOSED")] if len(table) else table
    unresolved = table.loc[table["position_status"].eq("UNRESOLVED")] if len(table) else table
    first = lambda column: table[column].iloc[0] if len(table) else None
    total = lambda frame, column: float(pd.to_numeric(frame[column]).sum()) if len(frame) else 0.0
    return dict(ledger_id=first("ledger_id"), ledger_kind=first("ledger_kind"), stream=first("stream"),
                unit=first("unit"), cost_case=first("cost_case"), opportunities=len(table),
                admitted=int(table["position_taken"].sum()) if len(table) else 0, closed=len(closed),
                unresolved_open_exposure=len(unresolved), gross_pnl=total(closed, "gross_pnl"),
                total_cost=total(closed, "total_cost"), net_pnl=total(closed, "net_pnl"),
                open_unrealized_gross_pnl=total(unresolved, "unrealized_gross_pnl"),
                no_trade_reasons=({str(reason): int(count) for reason, count in table.loc[
                    ~table["position_taken"].astype(bool), "no_trade_reason"].value_counts().sort_index().items()}
                                  if len(table) else {}))


def persist_stream_surfaces(result: dict, output_dir: str | Path) -> dict:
    """Write one stream's immutable surfaces. Markout stores and ledgers get separate folders."""
    root = Path(output_dir) / result["stream"]
    root.mkdir(parents=True, exist_ok=False)
    files = {}

    def write(relative, frame):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        files[relative] = file_hash(path)

    for store_id, tables in result["markouts"].items():
        for name, frame in tables.items():
            write(f"markouts/{store_id}/{name}.csv", guard_report_table(frame))
    summaries = {}
    for ledger_id, frame in result["ledgers"].items():
        summaries[ledger_id] = ledger_summary(frame)
        write(f"ledgers/{ledger_id}.csv", frame)
    manifest = {key: value for key, value in result.items() if key not in {"markouts", "ledgers", "full_policy"}}
    # A not_defined surface 4 is persisted as its status only: no ledger file, no rows, no totals.
    manifest.update(schema=surface_schema(), attribution_fields=list(ATTRIBUTION_FIELDS),
                    full_policy={str(queue_id): value for queue_id, value in sorted(result["full_policy"].items())},
                    markout_stores=sorted(result["markouts"]), ledger_summaries=summaries, files=files,
                    scored_collection_enabled=False, promotion_eligible=False)
    manifest["surface_hash"] = content_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False, default=str) + "\n",
                                        encoding="utf-8")
    return manifest
