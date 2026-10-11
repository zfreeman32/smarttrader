"""Completed expected-move S3 full-policy historical diagnostic: all matching setups, no label-based admission.

Uses B1 outcomes and the runtime numerical inference path. Does not register a
model, fit anything, or certify live feature/latency parity. Outputs are immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp/b2_s3_calendar_dependencies"))

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as calendars
import xgboost as xgb

from model_testing.outcome_artifacts import run_outcome_evaluation, replay_outcome_run
from model_testing.outcome_atr import completed_structural_atr, structural_atr_at
from model_testing.outcome_contracts import PositionContract, content_hash, file_hash, resolve_contract
from model_testing.outcome_diagnostics import ModelSetupContext, SetupObservation
from model_testing.outcome_engine import EvaluationTape, SignalEvidence, TIME_FIELDS, position_costs
from ote_live.models.runners import RuntimeModelRunner
from model_training.source_history import impute_matrix
from scripts.run_b2_source_history import runtime_model
from ote_live.features.manifest import LivePolicy
from ote_live.policies.regime import label_policy_frame
from ote_live.policies.abstain import LiveAbstainState
from ote_live.policies.shadow import evaluate_shadow_policy, SETUP_MATCH_CONTRACT
from ote_live.policies.expected_move import build_expected_move_frame, CONTRACT, SPREAD_TICKS


def full_policy_decision(policy, row, *, probability, threshold_passed, source_row_idx, ready, state):
    """Use production shadow gates; missing inputs remain unknown, never zero-filled."""
    prediction = SimpleNamespace(model_id=policy.model_id, calibrated_probability=probability,
        raw_score=probability, source_row_idx=source_row_idx, collection_version="offline_corrected_s3",
        timestamp=pd.Timestamp(row["datetime"]).to_pydatetime(), prediction_recorded_at_utc=None,
        threshold_applied=policy.thresholds.global_threshold, threshold_source="frozen_training_oof")
    snapshot = SimpleNamespace(observation_metadata={
        "model_input_contract": {"diagnostic_only": not ready}, "bar_eligibility": {"eligible": ready},
        "diagnostic_reasons": [] if ready else ["offline_common_readiness_failed"]})
    return evaluate_shadow_policy(prediction=prediction, live_policy=policy,
        policy_context=pd.DataFrame([row]), snapshot=snapshot, threshold_passed=threshold_passed,
        setup_match={"matched": True, "primary_event_id": f"s3-long-{source_row_idx}",
                     "matched_event_ids": [f"s3-long-{source_row_idx}"], "candidates": [],
                     "counterfactual_event_ids": [], "later_event_ids": [], "reasons": [],
                     "contract_version": SETUP_MATCH_CONTRACT}, state=state)



def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def verify_files(pins):
    for name, expected in pins.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Input changed: {name}")


def build_schedule(start, end, overrides):
    if calendars.__version__ != "5.1.3":
        raise ValueError("Calendar dependency must be pinned at 5.1.3")
    sessions = calendars.get_calendar("CME Globex Equity").schedule(
        (start - pd.Timedelta(days=1)).date(), (end + pd.Timedelta(days=1)).date())
    for day, close in overrides.items():
        sessions.loc[pd.Timestamp(day), "market_close"] = pd.Timestamp(close)
    frames = []
    for day, row in sessions.iterrows():
        idx = pd.date_range(row.market_open, row.market_close, freq="5min", inclusive="left")
        idx = idx[(idx >= start) & (idx < end)]
        if len(idx):
            frames.append(pd.DataFrame(dict(open_ts=idx, session_id=str(day.date()))))
    return pd.concat(frames, ignore_index=True), sessions


def summarize_positions(frame):
    closed = frame.loc[frame.position_status.eq("CLOSED")].sort_values(["paper_exit_ts", "event_id"])
    pnl = closed.net_pnl.to_numpy(float)
    curve = np.r_[0., np.cumsum(pnl)]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return dict(opportunities=len(frame), admitted=int(frame.position_taken.sum()),
                closed=len(closed), unresolved=int(frame.position_status.eq("UNRESOLVED").sum()),
                closed_net_usd=float(pnl.sum()), closed_gross_usd=float(closed.gross_pnl.sum()),
                closed_cost_usd=float(closed.total_cost.sum()),
                closed_net_win_rate=float((pnl > 0).mean()) if len(pnl) else None,
                closed_profit_factor=float(wins / losses) if losses else None,
                closed_max_drawdown_usd=float(np.max(np.maximum.accumulate(curve) - curve)),
                net_usd_per_closed_trade=float(pnl.mean()) if len(pnl) else None,
                complete_portfolio_pnl=not frame.position_status.eq("UNRESOLVED").any(),
                no_trade_reasons=frame.loc[~frame.position_taken, "no_trade_reason"].value_counts().to_dict())


def main(config_path, output):
    cfg = json.loads(config_path.read_text())
    if output.exists() or not output.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("Use a new directory under artifacts")
    verify_files(cfg["inputs_sha256"])
    output.mkdir(parents=True)
    # Persist the entire design and hashes BEFORE generating scores or outcomes.
    declaration = dict(cfg, declared_at_utc=datetime.now(timezone.utc).isoformat(),
                       script_sha256=file_hash(Path(__file__)))
    dump(output / "predeclaration.json", declaration)
    # Preserve the precise policy producer/gates used by this research version.
    for name in cfg["inputs_sha256"]:
        if name.endswith(".py"):
            archived = output / "source_code" / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes((ROOT / name).read_bytes())
    run = json.loads((ROOT / cfg["training_manifest"]).read_text())
    model_dir = ROOT / cfg["model_dir"]
    prepared = ROOT / cfg["prepared_dir"]
    if run["status"] != "trained_pending_source_history_review":
        raise ValueError("Incomplete corrected model")
    review = json.loads((ROOT / cfg["model_review"]).read_text())
    if review["status"] != "source_history_artifact_review_passed" or review["model_outputs_sha256"] != run["outputs_sha256"]:
        raise ValueError("Model review does not match this bundle")
    preparation = ROOT / cfg["prepared_manifest"]
    if file_hash(preparation) != run["prepared_manifest_sha256"]:
        raise ValueError("Training preparation pin changed")
    verify_files(run["code_sha256"])
    for name, sha in run["outputs_sha256"].items():
        if file_hash(model_dir / name) != sha:
            raise ValueError(f"Changed model artifact: {name}")
    for name, sha in json.loads(preparation.read_text())["prepared_files_sha256"].items():
        if file_hash(preparation.parent / name) != sha:
            raise ValueError(f"Changed prepared artifact: {name}")
    summary = json.loads((model_dir / "training_summary.json").read_text())
    if not np.isclose(summary["threshold"], cfg["threshold"], rtol=0, atol=1e-15) or summary["calibration"]["resolved_method"] != "none":
        raise ValueError("Frozen threshold/calibration mismatch")
    names, context = summary["selected_feature_names"], summary["context_rows"]
    print("Loading frozen predictors and source bars", flush=True)
    columns = list(dict.fromkeys(["datetime", "source_row_idx", "frvp_setup_type", "frvp_setup_side", "ema_alignment", "atr_14", "range_shock_20", *names]))
    full = pd.read_parquet(ROOT / cfg["features"], columns=columns).set_index("source_row_idx", drop=False)
    full.datetime = pd.to_datetime(full.datetime, utc=True)
    raw = pd.read_csv(ROOT / cfg["bars"])
    raw["open_ts"] = pd.to_datetime(raw.ts_event, utc=True)
    np.testing.assert_array_equal(raw.open_ts.to_numpy(), full.datetime.to_numpy())
    raw["coordinate_id"] = "ES:" + raw.instrument_id.astype("int64").astype(str)
    raw["is_complete"] = True
    saved = pd.read_csv(model_dir / "test_predictions.csv")
    first, last = int(saved.source_row_idx.min()), int(saved.source_row_idx.max())
    assert (first, last) == tuple(cfg["source_row_bounds"])
    contract_history = summary["model_config"]["source_history_contract"]
    if contract_history != json.loads((prepared / "source_history_contract.json").read_text()):
        raise ValueError("Source-history contract differs from preparation")
    matrix = full[names].copy()  # RAW predictors: runtime applies its frozen fills.
    dump(output / "frozen_imputation.json", contract_history)
    prepared_test = pd.read_csv(prepared / "test.csv", usecols=["source_row_idx", *names])
    np.testing.assert_allclose(impute_matrix(matrix.loc[prepared_test.source_row_idx].to_numpy(), contract_history),
                               prepared_test[names], rtol=1e-6, atol=1e-6)
    model = xgb.Booster(); model.load_model(model_dir / "model.json")
    runtime = runtime_model(model, joblib.load(model_dir / "scaler.joblib"), summary)
    replay = np.array([runtime.predict_latest(matrix.loc[i-context:i],
        timestamp=full.loc[i, "datetime"].to_pydatetime()).raw_score for i in saved.source_row_idx])
    np.testing.assert_allclose(replay, saved.raw_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(replay >= cfg["threshold"], saved.predicted_label)
    policy = LivePolicy.model_validate(cfg["full_policy"])
    if policy.thresholds.global_threshold != cfg["threshold"] or policy.thresholds.regime_thresholds:
        raise ValueError("Unexpected full-policy threshold")
    policy_inputs = label_policy_frame(full[["datetime", "ema_alignment", "atr_14", "range_shock_20"]],
                                       timezone_contract=summary["timezone_contract"])
    # Regime defaults must not hide genuinely unavailable stress inputs.
    policy_inputs.loc[~np.isfinite(full.range_shock_20), "stress_regime"] = None
    if policy.expected_move_contract != CONTRACT or policy.cost_assumptions.session_spread_ticks != SPREAD_TICKS:
        raise ValueError("Frozen ES expected-move/spread contract required")
    movement = build_expected_move_frame(raw)
    for column in movement:
        policy_inputs[column] = movement[column].to_numpy()
    dump(output / "expected_move_definition.json", json.loads((ROOT / cfg["expected_move_definition"]).read_text()))
    candidates = full.loc[(full.index >= first) & (full.index <= last)
                          & full.frvp_setup_type.eq(3) & full.frvp_setup_side.eq(1)].copy()
    # Runtime history is chronological source bars; no target/exclusion/weight enters admission.
    probabilities = [runtime.predict_latest(matrix.loc[i-context:i, names], timestamp=full.loc[i, "datetime"].to_pydatetime()).raw_score for i in candidates.index]
    candidates = candidates[["datetime", "source_row_idx", "frvp_setup_type", "frvp_setup_side"]]
    candidates["probability"] = probabilities
    candidates["threshold_passed"] = candidates.probability.ge(cfg["threshold"])
    candidates["saved_test_probability"] = saved.set_index("source_row_idx").raw_probability.reindex(candidates.index)
    candidates.to_csv(output / "candidate_predictions.csv", index=False)
    del matrix, full
    start = candidates.datetime.min().floor("D") - pd.Timedelta(days=30)
    end = min(raw.open_ts.max() + pd.Timedelta(minutes=5), candidates.datetime.max().ceil("D") + pd.Timedelta(days=7))
    schedule, sessions = build_schedule(start, end, cfg["calendar_close_overrides"])
    bars = raw.loc[raw.open_ts.ge(start) & raw.open_ts.lt(end),
                   ["open_ts", "open", "high", "low", "close", "coordinate_id", "is_complete"]].copy()
    outside = bars.loc[~bars.open_ts.isin(schedule.open_ts)]
    outside.to_csv(output / "bars_outside_calendar.csv", index=False)
    if len(outside):
        raise ValueError("Market bars outside independent schedule; resolve before outcomes")
    missing = schedule.loc[~schedule.open_ts.isin(bars.open_ts)]
    missing.to_csv(output / "missing_scheduled_bars.csv", index=False)
    sessions.to_csv(output / "calendar_sessions.csv")
    tape = EvaluationTape(bars, schedule, timeframe_minutes=5, calendar_id=cfg["calendar_id"],
        calendar_source=cfg["calendar_source"], coverage_start=start, coverage_end=end)
    print(f"{len(candidates)} matching setups; {candidates.threshold_passed.sum()} threshold passes; deriving completed-hour ATR", flush=True)
    hourly = completed_structural_atr(tape, instrument="ES")
    hourly[["source_close_ts", "available_at_ts", "status"]].to_csv(output / "atr_coverage.csv", index=False)
    contract = resolve_contract(instrument="ES", strategy="frvp", setup_family="setup3",
                                model_family="continuation_setup3", side="long")
    policy_sha = content_hash(declaration)
    signals = []
    policy_decisions = {}
    policy_state = LiveAbstainState()
    for row in candidates.itertuples():
        anchor = row.datetime
        close = anchor + pd.Timedelta(minutes=5)
        known = close + pd.Timedelta(seconds=cfg["assumed_processing_seconds"])
        coordinate = raw.loc[row.source_row_idx, "coordinate_id"]
        proof = structural_atr_at(hourly, known, coordinate_id=coordinate)
        context_opens = raw.loc[row.source_row_idx-context:row.source_row_idx, "open_ts"]
        expected_context = tape.schedule.index[tape.schedule.index <= anchor][-context-1:]
        contiguous = np.array_equal(context_opens.to_numpy(), expected_context.to_numpy())
        same_coordinate = raw.loc[row.source_row_idx-context:row.source_row_idx, "coordinate_id"].eq(coordinate).all()
        ready = bool(contiguous and same_coordinate and proof is not None)
        rejection = "" if ready else "context_gap_or_roll_or_atr_unavailable_at_decision"
        event_id = f"s3-long-{row.source_row_idx}"
        decision = full_policy_decision(policy, policy_inputs.loc[row.source_row_idx].to_dict(),
            probability=row.probability, threshold_passed=row.threshold_passed,
            source_row_idx=row.source_row_idx, ready=ready, state=policy_state)
        policy_decisions[event_id] = decision
        clocks = {field: known for field in TIME_FIELDS}
        clocks.update(event_anchor_ts=anchor, signal_bar_close_ts=close, confirmation_ts=close,
                      feature_max_source_close_ts=close, atr_available_at=known if proof is None else proof.available_at_ts)
        audit = ModelSetupContext(collection_id=cfg["experiment_id"], opportunity_id=event_id,
            prediction_id=event_id+"-score", available_at_ts=known,
            coverage="all_model_observations", setup_coverage="complete",
            expected_model_ids=(cfg["model_id"],), model_artifact_hash=run["outputs_sha256"]["model.json"],
            policy_hash=policy_sha, raw_score=row.probability, calibrated_probability=row.probability,
            threshold=cfg["threshold"], threshold_source="frozen_training_oof", threshold_passed=row.threshold_passed,
            context_source_hash=file_hash(output / "candidate_predictions.csv"),
            setups=(SetupObservation(setup_event_id=event_id, strategy="frvp", setup_family="setup3", side="long",
                    known_at_ts=close, selected=True, detector_version="b2_corrected_saved_frvp_features"),))
        signals.append(SignalEvidence(event_id=event_id, model_id=cfg["model_id"], model_version=cfg["model_version"],
            model_family="continuation_setup3", strategy="frvp", setup_family="setup3", instrument="ES", side="long",
            **clocks, availability_status="verified", availability_evidence_refs=(
                "predeclaration.json: offline bar-close-plus-one-second assumption, not observed production arrival", 
                "candidate_predictions.csv: frozen causal source-row context", "frozen_imputation.json"),
            atr=None if proof is None else proof.value, atr_definition=contract.risk.atr_definition,
            atr_provenance=proof, coordinate_id=coordinate, policy_eligible=ready,
            policy_rejection_reason=rejection, quantity=1., audit_context=audit))
    dump(output / "full_policy_decisions.json", policy_decisions)
    policy_inputs.loc[candidates.source_row_idx].to_csv(output / "policy_inputs.csv", index=False)
    evidence = dict(cfg["inputs_sha256"])
    for p in [config_path, Path(__file__), output / "predeclaration.json", output / "frozen_imputation.json",
              output / "candidate_predictions.csv", output / "full_policy_decisions.json", output / "policy_inputs.csv"]:
        evidence[str(p.resolve().relative_to(ROOT))] = file_hash(p)
    outcomes = {}
    for branch in ("setup_only", "model_threshold", "full_policy"):
        branch_signals = []
        for signal in signals:
            admitted = signal.policy_eligible
            reasons = [signal.policy_rejection_reason] if signal.policy_rejection_reason else []
            if branch != "setup_only" and not signal.audit_context.threshold_passed:
                admitted = False
                reasons.append("probability_below_frozen_threshold")
            if branch == "full_policy":
                decision = policy_decisions[signal.event_id]
                admitted = admitted and decision["policy_candidate_eligible"]
                reasons.extend(decision["rejection_reasons"])
            branch_signals.append(signal.model_copy(update=dict(policy_eligible=admitted,
                policy_rejection_reason=";".join(dict.fromkeys(reasons)))))
        simulator = PositionContract(simulator_id=f"b2_s3_{branch}_v1", version=1,
            policy_id=f"{cfg['experiment_id']}_{branch}", policy_hash=policy_sha,
            max_positions=1, max_quantity=1., concurrency_scope="instrument", allow_opposite=False,
            cooldown_minutes=0, starting_equity=100000., accounting_unit="cash")
        print("Evaluating", branch, flush=True)
        manifest = run_outcome_evaluation(branch_signals, tape, output_dir=output / branch,
                                         simulator=simulator, input_lineage=evidence)
        replay_manifest = replay_outcome_run(output / branch, output / (branch + "_replay"))
        assert manifest == replay_manifest, "Immutable replay differs"
        frame = pd.read_csv(output / branch / "evaluated_signals.csv")
        metrics = summarize_positions(frame)
        closed = frame.loc[frame.position_status.eq("CLOSED")].copy()
        closed["month"] = pd.to_datetime(closed.paper_exit_ts, utc=True).dt.strftime("%Y-%m")
        closed.groupby("month").agg(trades=("event_id", "size"), net_usd=("net_pnl", "sum")).to_csv(output / f"{branch}_monthly.csv")
        # Stress fixed admitted trades; costs do not influence this policy's admission.
        metrics["closed_net_double_cost_usd"] = float((closed.gross_pnl - 2 * closed.total_cost).sum())
        metrics["paper_outcome_counts"] = manifest["paper_outcome_counts"]
        metrics["replay_identical"] = True
        metrics["run_hash"] = manifest["run_hash"]
        # Independent signals are reported separately; their sum is NOT portfolio P&L.
        independent = frame.loc[frame.policy_eligible if "policy_eligible" in frame else
            frame.event_id.isin([s.event_id for s in branch_signals if s.policy_eligible])].copy()
        independent = independent.loc[independent.paper_resolved]
        net = []
        for r in independent.itertuples():
            cost = position_costs(entry_ts=r.entry_ts, exit_ts=r.paper_exit_ts, cost=contract.costs,
                                  quantity=1, accounting_unit="cash")["total_cost"]
            net.append((r.paper_exit_price-r.entry_price) / .25 * 12.5 - cost)
        metrics["independent_resolved_signals"] = len(net)
        metrics["independent_mean_net_usd"] = float(np.mean(net)) if net else None
        outcomes[branch] = metrics
    overlapping = candidates.dropna(subset=["saved_test_probability"])
    result = dict(status="historical_diagnostic_complete", independent_fresh_validation=False,
        experiment_id=cfg["experiment_id"], period_start=str(candidates.datetime.min()),
        period_end=str(candidates.datetime.max()), source_data_end=str(raw.open_ts.max()),
        candidates=len(candidates), threshold_passes=int(candidates.threshold_passed.sum()),
        common_ready_candidates=sum(s.policy_eligible for s in signals), missing_scheduled_bars=len(missing),
        source_history_vs_saved_test=dict(compared=len(overlapping), mean_abs_probability_difference=float(
            (overlapping.probability-overlapping.saved_test_probability).abs().mean()),
            threshold_disagreements=int((overlapping.threshold_passed != overlapping.saved_test_probability.ge(cfg["threshold"])).sum())),
        saved_prediction_replay_max_abs_error=float(np.max(np.abs(replay - saved.raw_probability))),
        full_policy_status="evaluated_with_causal_expected_move_ticks_v1",
        full_policy_economics_assessable=outcomes["full_policy"]["closed"] > 0,
        expected_move_status_counts=policy_inputs.loc[candidates.source_row_idx, "expected_move_status"].value_counts().to_dict(),
        expected_move_contract=CONTRACT,
        full_policy_rejection_counts=pd.Series([r for d in policy_decisions.values() for r in d["rejection_reasons"]]).value_counts().to_dict(),
        promotion_eligible=False, live_registry_changed=False, models_fitted=False,
        branches=outcomes, input_lineage=evidence)
    verify_files(cfg["inputs_sha256"])
    result["artifacts_sha256"] = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    dump(output / "comparison.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_lineage", "artifacts_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/b2_s3_full_policy_backtest_20260927_v2.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    main(args.config, args.output_dir)
