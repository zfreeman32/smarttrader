"""Saved B1 namespaces and immutable replay inputs; no model or label writes."""
from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path

import pandas as pd

from model_testing.outcome_contracts import (
    CATALOG, PositionContract, content_hash, file_hash, load_contracts, resolve_contract,
)
from model_testing.outcome_engine import (
    ENGINE_VERSION, EvaluationTape, SignalEvidence, evaluate_signal, simulate_positions,
)
from model_testing.outcome_scope import BaselineScope, load_scope
from model_testing.outcome_diagnostics import build_model_setup_audit, summarize_model_setup_audit
from model_testing.outcome_atr import completed_structural_atr, structural_atr_at, STRUCTURAL_ATR_MINUTES


def run_outcome_evaluation(signals: list[SignalEvidence], tape: EvaluationTape, *,
                           output_dir: str | Path, simulator: PositionContract | None = None,
                           catalog_path: str | Path = CATALOG,
                           scope: BaselineScope | None = None,
                           derive_structural_atr: bool = False,
                           input_lineage: dict | None = None) -> dict:
    """One dataset/instrument/timeframe per run. Policy decisions must already exist."""
    if not signals or len({s.event_id for s in signals}) != len(signals):
        raise ValueError("Nonempty, unique event IDs required")
    if len({(s.instrument, s.timeframe_minutes) for s in signals}) != 1:
        raise ValueError("Use separate tapes/runs for different instruments or timeframes")
    if len({s.audit_context.collection_id for s in signals if s.audit_context is not None}) > 1:
        raise ValueError("Use separate runs for different collection baselines")
    if derive_structural_atr:
        periods = {s.atr_definition for s in signals} & STRUCTURAL_ATR_MINUTES.keys()
        histories = {definition: completed_structural_atr(tape,
                     timeframe_minutes=STRUCTURAL_ATR_MINUTES[definition], instrument=signals[0].instrument)
                     for definition in periods}
        derived = []
        for signal in signals:
            if signal.atr_definition in histories:
                proof = structural_atr_at(histories[signal.atr_definition], signal.known_at_ts, coordinate_id=signal.coordinate_id)
                signal = signal.model_copy(update=dict(atr=None if proof is None else proof.value,
                    atr_provenance=proof, atr_available_at=signal.atr_available_at if proof is None else proof.available_at_ts))
            derived.append(signal)
        signals = derived
    contracts = load_contracts(catalog_path)
    scope = scope or load_scope()
    used, outcomes = {}, []
    for signal in signals:
        contract = resolve_contract(instrument=signal.instrument, strategy=signal.strategy,
                                    model_family=signal.model_family, setup_family=signal.setup_family,
                                    side=signal.side,
                                    timeframe_minutes=signal.timeframe_minutes, contracts=contracts)
        used[contract.contract_id] = contract
        outcomes.append(evaluate_signal(signal, tape, contract, scope=scope))
    if simulator is not None:
        positions = simulate_positions(signals, outcomes, used, simulator, scope=scope)
    else:
        positions = [dict(event_id=s.event_id, position_taken=False,
                          no_trade_reason="position_simulation_not_requested", position_status="NOT_SIMULATED",
                          gross_pnl=None, spread_cost=None, slippage_cost=None, commission_cost=None,
                          total_cost=None, net_pnl=None) for s in signals]
    combined = pd.DataFrame(outcomes).merge(pd.DataFrame(positions), on="event_id", validate="one_to_one")
    # Identity travels with every namespace; the joined table is a convenience.
    identity = ["event_id", "model_id", "model_version", "strategy", "model_family", "setup_family", "instrument", "side",
                "baseline_scope_id", "baseline_scope_hash", "baseline_scope", "baseline_scope_reason",
                "evaluation_contract_id", "evaluation_contract_version", "evaluation_contract_hash"]
    training = identity + [c for c in combined if c.startswith("training_label_")]
    forward = identity + ["event_anchor_ts", "known_at_ts", "entry_ts", "entry_price"] + [c for c in combined if c.startswith(("forward_", "atr_"))]
    paper = identity + [c for c in outcomes[0] if c not in identity and not c.startswith(("forward_", "training_label_"))]
    economic = identity + ["entry_ts", "entry_price", "paper_exit_ts", "paper_exit_price"] + [c for c in positions[0] if c != "event_id"]
    # An output directory is an immutable run; never overwrite old backtests.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, columns in (("training_label_outcomes", training), ("paper_contract_outcomes", paper),
                          ("forward_returns", forward), ("simulated_positions", economic)):
        combined.loc[:, columns].to_csv(output_dir / f"{name}.csv", index=False)
    combined.to_csv(output_dir / "evaluated_signals.csv", index=False)
    observations, coverage = build_model_setup_audit(signals, outcomes)
    observations.to_csv(output_dir / "model_setup_observations.csv", index=False)
    coverage.to_csv(output_dir / "model_observation_coverage.csv", index=False)
    summarize_model_setup_audit(observations).to_csv(output_dir / "model_setup_response_summary.csv", index=False)
    closed = combined.loc[combined["net_pnl"].notna()].copy()
    if simulator is not None:
        closed = closed.sort_values(["paper_exit_ts", "event_id"])
        equity = closed[["event_id", "paper_exit_ts", "net_pnl"]].copy()
        equity["equity_curve"] = simulator.starting_equity + pd.to_numeric(equity["net_pnl"]).cumsum()
        equity.to_csv(output_dir / "simulated_equity_curve.csv", index=False)
    (output_dir / "signal_evidence.jsonl").write_text(
        "".join(s.model_dump_json() + "\n" for s in signals), encoding="utf-8")
    for name, frame in (("bars", tape.bars), ("schedule", tape.schedule), ("finer_bars", tape.finer)):
        if frame is not None:
            frame.rename_axis("open_ts").reset_index().to_csv(output_dir / f"{name}.csv", index=False)
    calendar = dict(calendar_id=tape.calendar_id, calendar_source=tape.calendar_source,
                    coverage_start=tape.coverage_start.isoformat(), coverage_end=tape.coverage_end.isoformat(),
                    timeframe_minutes=int(tape.delta.total_seconds() / 60), calendar_hash=tape.schedule_hash,
                    require_completion=tape.require_completion)
    (output_dir / "calendar.json").write_text(json.dumps(calendar, indent=2) + "\n", encoding="utf-8")
    frozen_contracts = {key: c.model_dump(mode="json") for key, c in sorted(used.items())}
    (output_dir / "outcome_contracts.json").write_text(json.dumps(frozen_contracts, indent=2) + "\n", encoding="utf-8")
    (output_dir / "catalog.json").write_text(json.dumps({"schema_version": 1, "contracts": list(frozen_contracts.values())}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "catalog.lock.json").write_text(json.dumps({key: c.sha256 for key, c in sorted(used.items())}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "baseline_scope.json").write_text(scope.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "baseline_scope.lock.json").write_text(json.dumps({scope.scope_id: scope.sha256}) + "\n", encoding="utf-8")
    if simulator is not None:
        (output_dir / "position_contract.json").write_text(simulator.model_dump_json(indent=2) + "\n", encoding="utf-8")
    snapshot_dir = output_dir / "engine_source"
    snapshot_dir.mkdir()
    for name in ("outcome_contracts.py", "outcome_engine.py", "outcome_artifacts.py", "outcome_scope.py", "outcome_atr.py", "outcome_diagnostics.py"):
        source = Path(__file__).with_name(name)
        (snapshot_dir / name).write_bytes(source.read_bytes())
    counts = Counter(r["paper_contract_outcome"] for r in outcomes)
    resolved = sum(bool(r["paper_resolved"]) for r in outcomes)
    summary = dict(schema_version="b1-outcome-artifacts-v2", engine_version=ENGINE_VERSION,
        baseline_scope_id=scope.scope_id, baseline_scope_hash=scope.sha256,
        structural_atr_mode="derive_from_tape" if derive_structural_atr else "supplied_evidence",
        diagnostics_only_count=sum(r["baseline_scope"] == "diagnostics_only" for r in outcomes),
        model_setup_context_missing_count=int(observations.loc[observations.context_status.ne("OK"), "event_id"].nunique()),
        outcome_surfaces=["training_label_outcome", "paper_contract_outcome", "forward_returns", "simulated_position_pnl"],
        evaluation_contracts={key: c.sha256 for key, c in sorted(used.items())},
        signal_count=len(signals), paper_outcome_counts=dict(sorted(counts.items())), paper_resolved_count=resolved,
        paper_win_rate=(counts["TARGET"] + counts["TIMEOUT_WIN"]) / resolved if resolved else None,
        paper_target_rate=counts["TARGET"] / resolved if resolved else None,
        paper_unresolved_count=sum(not r["paper_resolved"] and r["baseline_scope"] == "included" for r in outcomes),
        paper_not_evaluated_count=counts["NOT_EVALUATED"],
        position_taken_count=sum(p["position_taken"] for p in positions),
        position_unresolved_count=sum(p["position_status"] == "UNRESOLVED" for p in positions),
        simulator_id=None if simulator is None else simulator.simulator_id,
        simulator_hash=None if simulator is None else simulator.sha256,
        calendar_hash=tape.schedule_hash, input_lineage=input_lineage or {},
        # B1 never clears B2/B3/B4/B5 or changes the paper-trial activation gate.
        scored_collection_enabled=False, promotion_eligible=False,
        python_version=platform.python_version(), pandas_version=pd.__version__)
    summary["artifact_hashes"] = {str(p.relative_to(output_dir)): file_hash(p)
                                   for p in sorted(output_dir.rglob("*")) if p.is_file()}
    summary["run_hash"] = content_hash(summary)
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def replay_outcome_run(saved_dir: str | Path, output_dir: str | Path) -> dict:
    """Check all saved bytes and engine identity before replaying a frozen run."""
    saved_dir = Path(saved_dir)
    manifest = json.loads((saved_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_run_hash = manifest.pop("run_hash")
    if content_hash(manifest) != expected_run_hash:
        raise ValueError("Saved manifest hash mismatch")
    for relative, digest in manifest["artifact_hashes"].items():
        path = (saved_dir / relative).resolve()
        if not path.is_relative_to(saved_dir.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Saved artifact hash mismatch: {relative}")
    for source in (saved_dir / "engine_source").glob("*.py"):
        if file_hash(source) != file_hash(Path(__file__).with_name(source.name)):
            raise ValueError("Engine changed; replay with the archived engine source in an isolated checkout")
    calendar = json.loads((saved_dir / "calendar.json").read_text(encoding="utf-8"))
    calendar.pop("calendar_hash")
    finer = saved_dir / "finer_bars.csv"
    tape = EvaluationTape(pd.read_csv(saved_dir / "bars.csv"), pd.read_csv(saved_dir / "schedule.csv"),
                          finer_bars=pd.read_csv(finer) if finer.exists() else None, **calendar)
    signals = [SignalEvidence.model_validate_json(line)
               for line in (saved_dir / "signal_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    position_path = saved_dir / "position_contract.json"
    simulator = PositionContract.model_validate_json(position_path.read_text(encoding="utf-8")) if position_path.exists() else None
    return run_outcome_evaluation(signals, tape, output_dir=output_dir, simulator=simulator,
                                  catalog_path=saved_dir / "catalog.json", scope=load_scope(saved_dir / "baseline_scope.json"),
                                  derive_structural_atr=manifest["structural_atr_mode"] == "derive_from_tape",
                                  input_lineage=manifest["input_lineage"])
