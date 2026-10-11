"""Run the frozen B1 outcome review for the remaining named candidates.

This is deliberately a review tool, not a trainer or promotion tool.  It uses
only the replacement-model held-out endpoints, the frozen B1 contracts, and a
predeclared row-for-row comparison.  Pooled candidates are recorded as
excluded by the B1 scope and are never silently treated as specialists.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp/b2_s3_calendar_dependencies"))

import numpy as np
import pandas as pd
import pandas_market_calendars as calendars

from model_testing.outcome_artifacts import run_outcome_evaluation, replay_outcome_run
from model_testing.outcome_atr import completed_structural_atr, structural_atr_at
from model_testing.b3_comparison import (
    B3_BRANCHES,
    build_b3_comparison_plan,
    load_json,
    not_defined_full_policy_result,
    validate_full_policy_result,
)
from model_testing.outcome_contracts import PositionContract, content_hash, file_hash, resolve_contract
from model_testing.outcome_diagnostics import ModelSetupContext, SetupObservation
from model_testing.outcome_engine import EvaluationTape, SignalEvidence, TIME_FIELDS


CANDIDATES = {
    7: dict(
        target="long_frvp_continuation_setup5",
        strategy="frvp", side="long", setup_family="setup5",
        model_family="continuation_setup5", backend="xgboost",
        prepared="artifacts/b2_frvp_setups_prepared_20260921_v1/dataset.parquet",
        model="models/b2_candidate_history_20260928_v1/07_long_frvp_continuation_setup5/long_frvp_continuation_setup5",
        output_model_id="frvp_long_continuation_setup5_candidate_20260928_v1",
        threshold=0.62,
    ),
    17: dict(
        target="short_ict_continuation_premium_discount_continuation",
        strategy="ict", side="short", setup_family="premium_discount_continuation",
        model_family="continuation_premium_discount_continuation", backend="xgboost",
        prepared="artifacts/b2_ict_specialists_prepared_20260921_v1/dataset.parquet",
        events="artifacts/b2_ict_labels_20260921_v1/ict_es_events.csv",
        model="models/b2_candidate_history_20260928_v1/17_short_ict_continuation_premium_discount_continuation/short_ict_continuation_premium_discount_continuation",
        output_model_id="ict_short_continuation_premium_discount_continuation_candidate_20260928_v1",
        threshold=0.53,
    ),
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def build_schedule(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if calendars.__version__ != "5.1.3":
        raise ValueError("The B1 calendar dependency must remain pinned at 5.1.3")
    sessions = calendars.get_calendar("CME Globex Equity").schedule(
        (start - pd.Timedelta(days=1)).date(), (end + pd.Timedelta(days=1)).date())
    if pd.Timestamp("2025-01-09") in sessions.index:
        sessions.loc[pd.Timestamp("2025-01-09"), "market_close"] = pd.Timestamp("2025-01-09 14:30:00+00:00")
    frames = []
    for day, row in sessions.iterrows():
        opens = pd.date_range(row.market_open, row.market_close, freq="5min", inclusive="left")
        opens = opens[(opens >= start) & (opens < end)]
        if len(opens):
            frames.append(pd.DataFrame({"open_ts": opens, "session_id": str(day.date())}))
    return pd.concat(frames, ignore_index=True)


def source_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ROOT / "data/futures_data/ES-5m-tagged.csv")
    raw["open_ts"] = pd.to_datetime(raw.ts_event, utc=True)
    raw["coordinate_id"] = "ES:" + raw.instrument_id.astype("int64").astype(str)
    raw["is_complete"] = True
    bars = raw[["open_ts", "open", "high", "low", "close", "coordinate_id", "is_complete"]].copy()
    return raw, bars


def artifact_hash(model_dir: Path) -> str:
    name = "model.json" if (model_dir / "model.json").exists() else "model.pt"
    return file_hash(model_dir / name)


def load_candidates(candidate: dict, raw: pd.DataFrame):
    model_dir = ROOT / candidate["model"]
    pred = pd.read_csv(model_dir / "test_predictions.csv")
    prepared = pd.read_parquet(ROOT / candidate["prepared"], columns=["source_row_idx", "datetime", "atr_14", "frvp_setup_type", "frvp_setup_side"] if candidate["strategy"] == "frvp" else ["source_row_idx", "datetime", "atr_14"])
    prepared["datetime"] = pd.to_datetime(prepared.datetime, utc=True)
    rows = prepared.set_index("source_row_idx", drop=False).loc[pred.source_row_idx].copy()
    rows = rows.reset_index(drop=True).join(pred.set_index("source_row_idx"), on="source_row_idx", rsuffix="_prediction")
    excluded = pd.Series(False, index=rows.index)
    if candidate["strategy"] == "frvp":
        # The specialist's repaired target contains two historical endpoints
        # whose contemporaneous detector row is setup 3.  They are not silently
        # relabeled as setup 5 opportunities.
        excluded = ~((rows.frvp_setup_type == 5) & (rows.frvp_setup_side == 1))
    else:
        events = pd.read_csv(ROOT / candidate["events"])
        events = events[(events.setup_type == candidate["setup_family"]) & (events.setup_side == -1)]
        events = events.drop_duplicates("signal_index").set_index("signal_index")
        rows = rows.join(events, on="source_row_idx", rsuffix="_event")
        excluded = rows.event_time.isna()
    rows["review_excluded"] = excluded
    rows["threshold_passed"] = rows.raw_probability >= candidate["threshold"]
    return rows.loc[~rows.review_excluded].copy(), rows.loc[rows.review_excluded].copy(), pred


def make_signal(row, candidate, tape, hourly, policy_hash: str) -> SignalEvidence:
    anchor = pd.Timestamp(row.datetime).to_pydatetime()
    close = pd.Timestamp(row.datetime) + pd.Timedelta(minutes=5)
    known = (pd.Timestamp(row.event_time) if candidate["strategy"] == "ict" else pd.Timestamp(row.datetime)) + pd.Timedelta(minutes=5, seconds=1)
    known = known.to_pydatetime()
    coordinate = str(tape.bars.loc[tape.bars.index == pd.Timestamp(row.datetime), "coordinate_id"].iloc[0])
    contract = resolve_contract(instrument="ES", strategy=candidate["strategy"],
        setup_family=candidate["setup_family"], model_family=candidate["model_family"],
        side=candidate["side"])
    proof = None
    atr = float(row.atr_14) if candidate["strategy"] == "ict" else None
    if candidate["strategy"] == "frvp":
        proof = structural_atr_at(hourly, known, coordinate_id=coordinate)
        atr = None if proof is None else proof.value
    event_id = f"b1-candidate-{candidate['output_model_id']}-{int(row.source_row_idx)}"
    clocks = {field: known for field in TIME_FIELDS}
    clocks.update(event_anchor_ts=anchor, signal_bar_close_ts=close.to_pydatetime(), confirmation_ts=close.to_pydatetime(),
                  feature_max_source_close_ts=close.to_pydatetime(), atr_available_at=known if proof is None else proof.available_at_ts)
    audit = ModelSetupContext(collection_id="b1_candidate_outcome_review_20260929",
        opportunity_id=event_id, prediction_id=event_id + "-prediction", available_at_ts=known,
        coverage="all_model_observations", setup_coverage="complete",
        expected_model_ids=(candidate["output_model_id"],), model_artifact_hash=artifact_hash(ROOT / candidate["model"]),
        policy_hash=policy_hash, raw_score=float(row.raw_probability), calibrated_probability=float(row.raw_probability),
        threshold=candidate["threshold"], threshold_source="frozen_b2_candidate_oof", threshold_passed=bool(row.threshold_passed),
        context_source_hash=file_hash(ROOT / candidate["model"] / "test_predictions.csv"),
        setups=(SetupObservation(setup_event_id=event_id, strategy=candidate["strategy"], setup_family=candidate["setup_family"],
            side=candidate["side"], known_at_ts=known, selected=True, detector_version="b1_candidate_review_frozen_event_source"),))
    extra = {}
    if candidate["strategy"] == "ict":
        extra.update(stop_reference=float(row.stop_reference) if pd.notna(row.stop_reference) else None,
                     target_reference=float(row.target_reference) if pd.notna(row.target_reference) else None)
    return SignalEvidence(event_id=event_id, model_id=candidate["output_model_id"],
        model_version="b2_candidate_history_20260928_v1", model_family=candidate["model_family"],
        strategy=candidate["strategy"], setup_family=candidate["setup_family"], instrument="ES", side=candidate["side"],
        training_label_outcome=None, training_label_version="not_used_for_admission", training_label_provenance="unknown_legacy",
        **clocks, availability_status="verified",
        availability_evidence_refs=("historical B1 tape and predeclared bar-close-plus-one-second offline timing; production arrival not certified",),
        atr=atr, atr_definition=contract.risk.atr_definition, atr_provenance=proof,
        coordinate_id=coordinate, policy_eligible=True, policy_rejection_reason="", quantity=1.0,
        audit_context=audit, **extra)


def summary(frame: pd.DataFrame, threshold_passes: int) -> dict:
    closed = frame.loc[frame.position_status.eq("CLOSED")]
    pnl = closed.net_pnl.astype(float).to_numpy()
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    curve = np.r_[0.0, np.cumsum(pnl)]
    return dict(opportunities=len(frame), threshold_passes=threshold_passes,
        admitted=int(frame.position_taken.sum()), closed=len(closed), unresolved=int(frame.position_status.eq("UNRESOLVED").sum()),
        net_pnl=float(pnl.sum()) if len(pnl) else 0.0, gross_pnl=float(closed.gross_pnl.sum()) if len(pnl) else 0.0,
        cost=float(closed.total_cost.sum()) if len(pnl) else 0.0,
        win_rate=float((pnl > 0).mean()) if len(pnl) else None,
        profit_factor=float(wins / losses) if losses else None,
        max_drawdown=float(np.max(np.maximum.accumulate(curve) - curve)) if len(curve) else 0.0)


def run_branch(signals, tape, candidate, output: Path, thresholded: bool, policy_hash: str):
    branch_signals = []
    for signal in signals:
        allowed = True if not thresholded else bool(signal.audit_context.threshold_passed)
        branch_signals.append(signal.model_copy(update={"policy_eligible": allowed,
            "policy_rejection_reason": "" if allowed else "probability_below_frozen_threshold"}))
    simulator = PositionContract(simulator_id=f"b1_candidate_{candidate['output_model_id']}_{'model' if thresholded else 'setup'}_v1",
        version=1, policy_id=f"b3b4_frozen_candidate_review_{'model' if thresholded else 'setup'}",
        policy_hash=policy_hash, max_positions=1, max_quantity=1.0, concurrency_scope="instrument",
        allow_opposite=False, cooldown_minutes=0, starting_equity=100000.0, accounting_unit="cash")
    manifest = run_outcome_evaluation(branch_signals, tape, output_dir=output, simulator=simulator,
        input_lineage={"candidate_test_predictions": file_hash(ROOT / candidate["model"] / "test_predictions.csv"),
                       "b1_catalog": file_hash(ROOT / "configs/outcome_contracts/b1_v3.json"),
                       "b1_scope": file_hash(ROOT / "configs/outcome_contracts/baseline_scope_v1.json")})
    replay = replay_outcome_run(output, output.parent / (output.name + "_replay"))
    if manifest != replay:
        raise ValueError("B1 candidate outcome replay was not identical")
    return summary(pd.read_csv(output / "evaluated_signals.csv"),
                   sum(bool(signal.audit_context.threshold_passed) for signal in branch_signals if signal.policy_eligible)), manifest


def main(output_dir: Path):
    if output_dir.exists():
        raise ValueError("Use a new output directory")
    output_dir.mkdir(parents=True)
    raw, bars = source_bars()
    candidate_results = {}
    protocol_path = ROOT / "configs/b3_b4_candidate_review_v1_20260929.json"
    selection_path = ROOT / "configs/broad_full_session_shadow_selection_v0.json"
    protocol = load_json(protocol_path)
    selection = load_json(selection_path)
    comparison_plan = build_b3_comparison_plan(protocol, selection, selection_path=selection_path)
    predeclared = {
        "comparison_id": "b3_b4_candidate_review_v1_20260929",
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_recipe_roster": comparison_plan,
        "common_population": "For every selected recipe, exact replacement-model held-out endpoints that have a verified contemporaneous same-side named setup match; setup-only and frozen-threshold legs reuse the identical event IDs.",
        "comparisons": list(B3_BRANCHES),
        "thresholds": "Candidate-specific development-selected thresholds, frozen before evaluation; no test outcomes, AUC, tolerance precision or artifact replay selects them.",
        "b1_contract": "b1_v3; exact named setup contract; full B1 costs; causal entry and missing-bar/roll censoring.",
        "position_assumptions": "one ES contract, one instrument position, no opposite position, no cooldown, 100000 USD nominal accounting.",
        "full_policy_rule": "Assess only with an exact replacement-version policy binding. Missing binding is not_defined: no full-policy ledger, P&L, zero-trade result, pass, or substituted historical policy.",
        "clean_evaluation": {
            "historical": "Held-out endpoints were untouched by fitting and outcomes are evaluated once here; technical replay inspection is not treated as economic evidence.",
            "prospective": "Reserve C1 from the first healthy collection timestamp after B7; freeze the roster, policy, session coverage and review target before observing outcomes.",
            "future_outcomes_required_to_start": False,
        },
    }
    policy_hash = content_hash(predeclared)
    dump(output_dir / "predeclaration.json", predeclared)
    for number, candidate in CANDIDATES.items():
        rows, excluded, pred = load_candidates(candidate, raw)
        if rows.empty:
            raise ValueError(f"No usable common opportunities for {number}")
        start = pd.to_datetime(rows.datetime.min(), utc=True) - pd.Timedelta(days=30)
        end = min(raw.open_ts.max() + pd.Timedelta(minutes=5), pd.to_datetime(rows.datetime.max(), utc=True) + pd.Timedelta(days=7))
        schedule = build_schedule(start, end)
        bar_frame = bars.loc[bars.open_ts.ge(start) & bars.open_ts.lt(end)].copy()
        tape = EvaluationTape(bar_frame, schedule, timeframe_minutes=5,
            calendar_id="pmc_5_1_3_cme_globex_equity_plus_carter_v1",
            calendar_source="pandas_market_calendars 5.1.3 CME Globex Equity; B1 review frozen calendar",
            coverage_start=start, coverage_end=end)
        hourly = completed_structural_atr(tape, instrument="ES") if candidate["strategy"] == "frvp" else None
        signals = [make_signal(row, candidate, tape, hourly, policy_hash) for row in rows.itertuples()]
        branch_dir = output_dir / str(number)
        setup_metrics, setup_manifest = run_branch(signals, tape, candidate, branch_dir / "setup_only", False, policy_hash)
        model_metrics, model_manifest = run_branch(signals, tape, candidate, branch_dir / "model_threshold", True, policy_hash)
        full_policy_result = not_defined_full_policy_result("No exact replacement-version full-policy binding was frozen.")
        validate_full_policy_result(full_policy_result)
        candidate_results[str(number)] = {
            "model_id": candidate["output_model_id"], "target": candidate["target"],
            "status": "reviewed_not_promotion",
            "common_opportunities": len(signals), "excluded_endpoint_rows": len(excluded),
            "excluded_reasons": ("named specialist event mismatch: test endpoint is not contemporaneous setup5" if number == 7 else "no matching named ICT event record"),
            "setup_only": setup_metrics, "model_threshold": model_metrics,
            "full_policy": full_policy_result,
            "replay_identical": True, "setup_run_hash": setup_manifest["run_hash"], "model_run_hash": model_manifest["run_hash"],
            "clean_evaluation": "historical held-out endpoint review; prospective C1 remains separately reserved",
        }
    result = {
        "status": "b1_candidate_outcome_review_complete",
        "comparison_id": predeclared["comparison_id"],
        "predeclaration_sha256": file_hash(output_dir / "predeclaration.json"),
        "selected_recipe_count": comparison_plan["candidate_count"],
        "selected_recipe_ids": comparison_plan["candidate_ids"],
        "historical_reviewed_candidate_ids": sorted(int(key) for key in candidate_results),
        "selected_recipes_not_historically_reviewed": [
            candidate_id for candidate_id in comparison_plan["candidate_ids"]
            if candidate_id not in candidate_results
        ],
        "comparison_plan": comparison_plan,
        "pooled_exclusions": {
            "1": {"target": "short_frvp_continuation", "status": "excluded", "reason": "B1 baseline_scope_v1: pooled FRVP model diagnostics-only, including setup matches."},
            "2": {"target": "long_frvp_continuation", "status": "excluded", "reason": "B1 baseline_scope_v1: pooled FRVP model diagnostics-only, including setup matches."},
            "11": {"target": "long_ict_continuation", "status": "excluded", "reason": "B1 baseline_scope_v1: pooled ICT model diagnostics-only, including setup matches."},
        },
        "specialist_reviews": candidate_results,
        "not_evidence": ["artifact replay", "AUC", "tolerance-based event precision", "training completion", "future C1 outcomes"],
        "research_collection_rule": "An otherwise eligible named specialist does not wait for future C1 outcomes; it still requires live feature parity, B5-B7, A12/A13 and all existing restrictions.",
        "promotion_eligible": False,
    }
    dump(output_dir / "comparison.json", result)
    dump(ROOT / "docs/audits/b1_candidate_outcome_review_20260929.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.output_dir)
