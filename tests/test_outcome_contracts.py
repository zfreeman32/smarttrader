from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_testing.outcome_artifacts import run_outcome_evaluation, replay_outcome_run
from model_testing.outcome_contracts import (
    CATALOG, OutcomeContract, PositionContract, file_hash, load_contracts, resolve_contract,
)
from model_testing.outcome_engine import (
    EvaluationTape, SignalEvidence, evaluate_signal, position_costs, simulate_positions,
)
from model_testing.outcome_lineage import registry_inventory, signal_from_available_inputs
from pruned_archive import pruned_registrations


def contract(timeout=3, **changes):
    base = load_contracts()["es_frvp_setup1_eval_v1"].model_dump(mode="json")
    base["risk"]["timeout_bars"] = timeout
    base.update(changes)
    return OutcomeContract.model_validate(base)


def signal(**changes):
    known = "2026-09-14T14:00:00Z"
    values = dict(event_id="one", model_id="frvp_long_reversal_test", model_version="synthetic_v1",
        model_family="reversal_setup1", strategy="frvp", setup_family="setup1", instrument="ES", side="long",
        training_label_outcome=1, training_label_version="historical_label_v7", training_label_hash="a" * 64,
        training_label_provenance="verified", event_anchor_ts="2026-09-14T13:50:00Z", known_at_ts=known,
        signal_bar_close_ts=known, confirmation_ts=known, features_available_at=known,
        feature_max_source_close_ts=known, regime_available_at=known, prediction_available_at=known,
        calibration_available_at=known, policy_available_at=known, geometry_available_at=known,
        atr_available_at=known, availability_status="verified", availability_evidence_refs=("synthetic-fixture-v1",),
        atr=1, atr_definition="completed_1h_structural_atr", coordinate_id="ESU6", policy_eligible=True)
    values.update(changes)
    if "atr_provenance" not in changes and values["atr_definition"] == "completed_1h_structural_atr":
        values["atr_provenance"] = dict(value=values["atr"] or 1,
            source_close_ts=known, history_start_ts="2026-09-13T00:00:00Z", available_at_ts=known,
            coordinate_id=values["coordinate_id"] or "unknown", source_hash="c" * 64)
    return SignalEvidence.model_validate(values)


def tape(case="timeout_flat", *, n=60, remove=(), corrupt=None, finer=None, times=None, sessions=None, roll=None):
    fixture = pd.read_csv(Path(__file__).parent / "fixtures/outcome_b1_paths.csv")
    data = fixture.loc[fixture.case.eq(case), ["open", "high", "low", "close"]].to_dict("records")
    data += [dict(open=100, high=100.5, low=99.5, close=100)] * (n - len(data))
    times = pd.date_range("2026-09-14T14:05:00Z", periods=n, freq="5min") if times is None else pd.DatetimeIndex(times)
    frame = pd.DataFrame(data[:len(times)])
    frame["open_ts"] = times
    frame["coordinate_id"] = "ESU6"
    if roll is not None:
        frame.loc[roll:, "coordinate_id"] = "ESZ6"
    if corrupt is not None:
        frame.loc[corrupt, "high"] = float("nan")
    schedule = pd.DataFrame(dict(open_ts=times, session_id=sessions or ["session1"] * len(times)))
    frame = frame.drop(index=list(remove)).reset_index(drop=True)
    return EvaluationTape(frame, schedule, timeframe_minutes=5, calendar_id="synthetic_calendar_v1",
        calendar_source="synthetic_fixture", coverage_start=times[0] - pd.Timedelta(minutes=10),
        coverage_end=times[-1] + pd.Timedelta(minutes=5), finer_bars=finer)


def simulator(**changes):
    values = dict(simulator_id="test_sim_v1", version=1, policy_id="synthetic_policy_v1", policy_hash="b" * 64,
        max_positions=1, max_quantity=5, concurrency_scope="instrument", allow_opposite=False,
        cooldown_minutes=0, starting_equity=1000, accounting_unit="ticks")
    values.update(changes)
    return PositionContract.model_validate(values)


@pytest.mark.parametrize("case,side,expected", [
    ("target", "long", "TARGET"), ("stop", "long", "STOP"),
    ("short_target", "short", "TARGET"), ("short_stop", "short", "STOP"),
    ("collision", "long", "STOP"), ("short_collision", "short", "STOP"),
    ("timeout_profit", "long", "TIMEOUT_WIN"), ("timeout_loss", "long", "TIMEOUT_LOSS"),
    ("timeout_flat", "long", "TIMEOUT_FLAT"), ("short_timeout_profit", "short", "TIMEOUT_WIN"),
])
def test_synthetic_paths(case, side, expected):
    result = evaluate_signal(signal(side=side), tape(case), contract())
    assert result["paper_contract_outcome"] == expected
    assert result["entry_ts"] > result["known_at_ts"] >= result["event_anchor_ts"]
    assert result["entry_price"] == 100
    assert result["timeout_ts"] == pd.Timestamp("2026-09-14T14:20:00Z")
    if "collision" in case:
        assert result["barrier_resolution_method"] == "adverse_first"
        assert result["ambiguous_bar_resolution_required"]


@pytest.mark.parametrize("field", ["event_anchor_ts", "confirmation_ts", "signal_bar_close_ts",
    "features_available_at", "feature_max_source_close_ts", "regime_available_at", "prediction_available_at",
    "calibration_available_at", "policy_available_at", "geometry_available_at", "atr_available_at"])
def test_future_inputs_cannot_be_used(field):
    s = signal(**{field: "2026-09-14T14:06:00Z"})
    result = evaluate_signal(s, tape(), contract())
    assert result["paper_contract_outcome"] == "INVALID"
    assert result["entry_ts"] is None


def test_close_boundary_is_strict_not_same_clock_open():
    times = pd.date_range("2026-09-14T14:00:00Z", periods=60, freq="5min")
    result = evaluate_signal(signal(), tape(times=times), contract())
    assert result["entry_ts"] == pd.Timestamp("2026-09-14T14:05:00Z")


@pytest.mark.parametrize("kwargs,reason", [({"remove": (0,)}, "entry_missing_or_corrupt_bar"),
    ({"remove": (1,)}, "missing_or_corrupt_bar"), ({"corrupt": 1}, "missing_or_corrupt_bar"),
    ({"roll": 1}, "roll_coordinate_change_or_unknown")])
def test_missing_bars_are_never_filled_or_searched_forward(kwargs, reason):
    result = evaluate_signal(signal(), tape(**kwargs), contract())
    assert result["paper_contract_outcome"] == "DATA_GAP"
    assert reason in result["data_quality_flags"]
    assert not result["paper_resolved"]


def test_no_entry_at_end_of_calendar():
    s = signal(known_at_ts="2026-09-14T19:00:00Z")
    result = evaluate_signal(s, tape(), contract())
    assert result["paper_contract_outcome"] == "NO_ENTRY"


def test_timeout_does_not_start_from_retrospective_anchor():
    result = evaluate_signal(signal(event_anchor_ts="2026-09-14T10:00:00Z"), tape(), contract())
    assert result["timeout_ts"] == result["entry_ts"] + pd.Timedelta(minutes=15)


def test_planned_weekend_and_forbidden_overnight():
    times = pd.DatetimeIndex(["2026-09-18T20:55:00Z", "2026-09-20T22:00:00Z", "2026-09-20T22:05:00Z"])
    t = tape(times=times, sessions=["fri", "sun", "sun"])
    s = signal(known_at_ts="2026-09-18T20:50:00Z")
    result = evaluate_signal(s, t, contract())
    assert result["paper_contract_outcome"] == "TIMEOUT_FLAT"
    assert result["paper_exit_ts"] == times[-1] + pd.Timedelta(minutes=5)
    result = evaluate_signal(s, t, contract(overnight="exit_before_session_boundary"))
    assert result["paper_exit_ts"] == times[0] + pd.Timedelta(minutes=5)
    assert result["barrier_resolution_method"] == "session_boundary_close"


def fine_path():
    return pd.DataFrame(dict(open_ts=pd.date_range("2026-09-14T14:05:00Z", periods=5, freq="1min"),
        open=[100, 100.5, 100.2, 99.5, 99.5], high=[101, 100.6, 100.4, 100, 100],
        low=[99.8, 100, 99, 98.5, 99], close=[100.5, 100.2, 99.5, 99.5, 100], coordinate_id=["ESU6"] * 5))


def test_complete_causal_finer_path_resolves_first_touch():
    result = evaluate_signal(signal(), tape("collision", finer=fine_path()), contract())
    assert result["paper_contract_outcome"] == "TARGET"
    assert result["barrier_resolution_method"] == "1m_first_touch"
    assert result["paper_exit_ts"] > result["entry_ts"] > result["known_at_ts"]


@pytest.mark.parametrize("invalid", ["missing_minute", "future_coordinate", "different_high", "predecision"])
def test_invalid_finer_path_falls_back_adverse_first(invalid):
    finer = fine_path()
    if invalid == "missing_minute":
        finer = finer.drop(index=2)
    elif invalid == "future_coordinate":
        finer["coordinate_id"] = "ESZ6"
    elif invalid == "different_high":
        finer["high"] += 1
    else:
        finer["open_ts"] -= pd.Timedelta(minutes=5)
    result = evaluate_signal(signal(), tape("collision", finer=finer), contract())
    assert result["paper_contract_outcome"] == "STOP"
    assert result["barrier_resolution_method"] == "adverse_first"


def test_stop_gap_pays_open_not_stop():
    t = tape()
    t.bars.loc[t.bars.index[1], ["open", "high", "low", "close"]] = [98, 98.5, 97.5, 98]
    result = evaluate_signal(signal(), t, contract())
    assert result["paper_exit_price"] == 98
    assert result["barrier_resolution_method"] == "adverse_gap_open"


def test_four_surfaces_are_independent():
    s, t, c = signal(), tape("target"), contract()
    original = s.model_dump()
    a = evaluate_signal(s, t, c)
    b = evaluate_signal(signal(quantity=100, policy_eligible=False, training_label_outcome=0), t, c)
    assert {k: v for k, v in a.items() if not k.startswith("training_label_")} == {
        k: v for k, v in b.items() if not k.startswith("training_label_")}
    changed = c.model_dump(mode="json")
    changed["risk"]["target_atr"] = 20
    d = evaluate_signal(s, t, OutcomeContract.model_validate(changed))
    assert {k: v for k, v in a.items() if k.startswith("forward_")} == {k: v for k, v in d.items() if k.startswith("forward_")}
    assert s.model_dump() == original
    short = evaluate_signal(signal(side="short"), tape("short_target"), c)
    assert short["forward_return_1"] > 0


def test_diagnostics_survive_unresolved_contract_without_inventing_paper_outcome():
    c = load_contracts(ROOT / "configs/outcome_contracts/b1_v1.json")["es_frvp_setup2_eval_v1"]
    s = signal(model_family="continuation_setup2", setup_family="setup2")
    result = evaluate_signal(s, tape(), c)
    assert result["paper_contract_outcome"] == "INVALID"
    assert result["contract_status"] == "UNRESOLVED"
    assert result["forward_return_48_status"] == "OK"


def test_overlap_paper_target_can_be_position_rejected():
    signals = [signal(event_id="a"), signal(event_id="b")]
    c = contract()
    outcomes = [evaluate_signal(s, tape("target"), c) for s in signals]
    positions = simulate_positions(signals, outcomes, {c.contract_id: c}, simulator())
    assert all(o["paper_contract_outcome"] == "TARGET" for o in outcomes)
    assert positions[0]["position_taken"] and not positions[1]["position_taken"]
    assert positions[1]["no_trade_reason"] == "concurrent_position_limit"
    p = positions[0]
    assert p["net_pnl"] == p["gross_pnl"] - p["total_cost"]
    assert p["total_cost"] == pytest.approx(p["spread_cost"] + p["slippage_cost"] + p["commission_cost"])


def test_target_hit_can_have_negative_net_pnl():
    s, c = signal(atr=0.1), contract()
    t = tape("target")
    t.bars.loc[t.bars.index[0], "low"] = 99.95
    out = evaluate_signal(s, t, c)
    p = simulate_positions([s], [out], {c.contract_id: c}, simulator())[0]
    assert out["paper_contract_outcome"] == "TARGET"
    assert p["net_pnl"] < 0


def test_unknown_exit_consumes_capacity_without_future_selection():
    signals = [signal(event_id="a"), signal(event_id="b", known_at_ts="2026-09-14T14:15:00Z")]
    c, t = contract(), tape(remove=(1,))
    outcomes = [evaluate_signal(s, t, c) for s in signals]
    positions = simulate_positions(signals, outcomes, {c.contract_id: c}, simulator())
    assert positions[0]["position_taken"]
    assert positions[0]["position_status"] == "UNRESOLVED"
    assert positions[0]["net_pnl"] is None
    assert positions[1]["no_trade_reason"] == "concurrent_position_limit"


@pytest.mark.parametrize("instrument", ["ES", "6E", "EURUSD"])
def test_frozen_cost_units_and_round_trip_reconcile(instrument):
    key = instrument.lower() + ("_ote_reversal_eval_v2" if instrument == "EURUSD" else "_frvp_setup1_eval_v1")
    cost = load_contracts()[key].costs
    values = position_costs(entry_ts="2026-09-14T14:05:00Z", exit_ts="2026-09-14T14:10:00Z",
                            cost=cost, quantity=2, accounting_unit=cost.unit)
    assert values["spread_cost"] == 4
    assert values["commission_cost"] == 2 * (0.35 if instrument == "EURUSD" else 0.4)
    assert values["total_cost"] == sum(values[k] for k in ("spread_cost", "slippage_cost", "commission_cost"))
    assert cost.spread_mode == "session_schedule"


def test_contract_immutable_hash_and_version_enforced(tmp_path):
    c = contract()
    with pytest.raises(Exception):
        c.risk.timeout_bars = 100
    changed = c.model_dump(mode="json")
    changed["risk"]["timeout_bars"] += 1
    assert OutcomeContract.model_validate(changed).sha256 != c.sha256
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    data["contracts"][0]["risk"]["timeout_bars"] += 1
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    path.with_suffix(".lock.json").write_bytes(CATALOG.with_suffix(".lock.json").read_bytes())
    with pytest.raises(ValueError, match="Frozen contract changed"):
        load_contracts(path)
    changed["contract_version"] = 2
    with pytest.raises(ValueError, match="ID must include version"):
        OutcomeContract.model_validate(changed)


def test_saved_artifacts_contain_all_surfaces_hashes_and_deterministic_inputs(tmp_path):
    label_sources = [ROOT / "data/labeling/frvp_labeling_engine.py", ROOT / "ict/labeling/ict_labeling_engine.py"]
    before = {p: file_hash(p) for p in label_sources}
    s, t = signal(), tape("target")
    one = run_outcome_evaluation([s], t, output_dir=tmp_path / "one", simulator=simulator())
    two = run_outcome_evaluation([s], t, output_dir=tmp_path / "two", simulator=simulator())
    assert one == two
    replay = replay_outcome_run(tmp_path / "one", tmp_path / "replay")
    assert replay == one
    for name in ("training_label_outcomes", "paper_contract_outcomes", "forward_returns", "simulated_positions"):
        frame = pd.read_csv(tmp_path / "one" / f"{name}.csv")
        assert frame.evaluation_contract_hash.iloc[0] == load_contracts()["es_frvp_setup1_eval_v1"].sha256
    assert {p: file_hash(p) for p in label_sources} == before
    assert pd.read_csv(tmp_path / "one/training_label_outcomes.csv").training_label_outcome.iloc[0] == 1
    assert one["paper_resolved_count"] == 1
    with pytest.raises(FileExistsError):
        run_outcome_evaluation([s], t, output_dir=tmp_path / "one")


def test_data_gaps_excluded_from_paper_win_rate_and_kept_in_artifacts(tmp_path):
    summary = run_outcome_evaluation([signal()], tape(remove=(1,)), output_dir=tmp_path / "gap")
    assert summary["paper_win_rate"] is None
    assert summary["paper_outcome_counts"] == {"DATA_GAP": 1}
    assert summary["paper_unresolved_count"] == 1


def test_pooled_setup_dispatch_and_unknown_family_fail_closed():
    assert resolve_contract(instrument="ES", strategy="ict", model_family="meta", setup_family="ifvg_reversal").risk.timeout_bars == 18
    assert resolve_contract(instrument="ES", strategy="ict", model_family="meta", setup_family="premium_discount_continuation").risk.timeout_bars == 24
    with pytest.raises(ValueError):
        resolve_contract(instrument="ES", strategy="ict", model_family="reversal", setup_family="classic_breaker")
    report = registry_inventory()
    assert report["registrations"] + pruned_registrations() >= 120
    assert not [row for row in report["rows"] if row["binding_error"]]


def test_adapter_does_not_invent_availability_from_training_label():
    with pytest.raises(KeyError):
        signal_from_available_inputs(dict(target=1, datetime="2026-09-14T14:00:00Z"))
    payload = signal().model_dump(mode="json")
    payload["prediction_available_at"] = "2026-09-14T14:07:00Z"
    payload["calibration_available_at"] = "2026-09-14T14:07:01Z"
    payload["policy_available_at"] = "2026-09-14T14:07:02Z"
    s = signal_from_available_inputs(payload)
    assert s.known_at_ts == pd.Timestamp("2026-09-14T14:07:02Z")
    assert evaluate_signal(s, tape(), contract())["entry_ts"] == pd.Timestamp("2026-09-14T14:10:00Z")


def test_legacy_evaluators_require_explicit_opt_in():
    from scripts.run_ote_policy_backtest import run_policy_backtest
    from scripts.run_ote_threshold_policy_search import run_threshold_policy_search
    for runner in (run_policy_backtest, run_threshold_policy_search):
        with pytest.raises(ValueError, match="pre-B1"):
            runner(regime_report_root="missing", output_root="unused", registry_path="missing")


def test_classic_breaker_requires_formation_activation_retest_order():
    c = load_contracts()["es_ict_classic_breaker_eval_v1"]
    s = signal(strategy="ict", model_family="classic_breaker", setup_family="classic_breaker",
               atr_definition="signal_bar_atr14")
    assert evaluate_signal(s, tape(), c)["paper_outcome_reason"] == "missing_breaker_causal_lineage"
    s = signal(strategy="ict", model_family="classic_breaker", setup_family="classic_breaker",
        atr_definition="signal_bar_atr14", source_structure_formed_ts="2026-09-14T13:30:00Z",
        structure_activation_ts="2026-09-14T13:45:00Z", setup_retest_ts="2026-09-14T14:05:00Z")
    assert evaluate_signal(s, tape(), c)["paper_outcome_reason"] == "invalid_breaker_causal_lineage"


def test_structural_barriers_freeze_initial_references_for_both_sides():
    c = load_contracts()["es_ict_sweep_reclaim_eval_v1"]
    for side, stop, target in (("long", 98, 104), ("short", 102, 96)):
        s = signal(strategy="ict", model_family="reversal_sweep_reclaim", setup_family="sweep_reclaim", side=side,
                   atr_definition="signal_bar_atr14", stop_reference=stop, target_reference=target)
        result = evaluate_signal(s, tape(), c)
        assert result["stop_price"] == stop
        assert result["target_price"] == target
        assert result["timeout_ts"] == result["entry_ts"] + pd.Timedelta(minutes=90)


def test_tampered_replay_is_rejected(tmp_path):
    run_outcome_evaluation([signal()], tape(), output_dir=tmp_path / "run")
    with (tmp_path / "run/bars.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_outcome_run(tmp_path / "run", tmp_path / "replay")


def test_cli_replays_saved_run(tmp_path, capsys):
    from scripts.run_outcome_evaluation import main
    run_outcome_evaluation([signal()], tape("target"), output_dir=tmp_path / "source")
    assert main(["--replay-run", str(tmp_path / "source"), "--output-dir", str(tmp_path / "result")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["paper_outcome_counts"] == {"TARGET": 1}


def test_shared_phase7_runner_dispatches_b1_without_legacy_economics(tmp_path):
    from scripts.run_ote_policy_backtest import run_policy_backtest
    result = run_policy_backtest(regime_report_root=None, registry_path=None,
        output_root=tmp_path / "phase7", b1_input_manifest=ROOT / "model_testing/examples/outcome_b1/input_manifest.json")
    assert result["schema_version"] == "b1-outcome-artifacts-v2"
    assert result["paper_outcome_counts"] == {"NOT_EVALUATED": 3}
    assert result["position_taken_count"] == 0


# Explicit frozen expectations: do not derive these from the mutable labeler.
FRVP_RESOLVED_RISK = [
    ("setup2", "long", 1.0, 1.0, 96, "setup2"),
    ("setup2", "short", 1.0, 1.0, 96, "setup2"),
    ("setup3", "long", 1.0, 0.95, 96, "setup3_long"),
    ("setup3", "short", 1.0, 1.0, 96, "setup3_short"),
    ("setup4", "long", 0.9, 1.0, 48, "setup4"),
    ("setup4", "short", 0.9, 1.0, 48, "setup4"),
    ("setup5", "long", 1.0, 0.9, 96, "setup5_long"),
    ("setup5", "short", 1.0, 0.85, 96, "setup5_short"),
]


@pytest.mark.parametrize("instrument", ["ES", "6E"])
@pytest.mark.parametrize("model_kind", ["pooled", "meta", "specialist"])
@pytest.mark.parametrize("setup,side,stop,target,horizon,stem", FRVP_RESOLVED_RISK)
def test_frvp_v2_risk_is_frozen_per_setup_side_and_instrument(
        instrument, model_kind, setup, side, stop, target, horizon, stem):
    family = "reversal" if setup == "setup4" else "continuation"
    model_family = {"pooled": family, "meta": "meta", "specialist": f"{family}_{setup}"}[model_kind]
    c = resolve_contract(instrument=instrument, strategy="frvp", model_family=model_family,
                         setup_family=setup, side=side)
    assert c.contract_id == f"{instrument.lower()}_frvp_{stem}_eval_v2"
    assert (c.risk.stop_atr, c.risk.target_atr, c.risk.timeout_bars) == (stop, target, horizon)
    assert c.risk.rule == "atr"
    assert c.risk.atr_definition == "completed_1h_structural_atr"
    assert c.paper_management == "fixed_barriers"
    assert c.costs.cost_model_id == f"{instrument.lower()}_session_cost_v1"


@pytest.mark.parametrize("setup,side,stop,target,horizon,stem", FRVP_RESOLVED_RISK)
def test_frvp_v2_geometry_and_timeout_start_at_causal_entry(setup, side, stop, target, horizon, stem):
    family = "reversal" if setup == "setup4" else "continuation"
    c = load_contracts()[f"es_frvp_{stem}_eval_v2"]
    s = signal(setup_family=setup, side=side, model_family=f"{family}_{setup}", atr=10)
    result = evaluate_signal(s, tape(n=100), c)
    sign = 1 if side == "long" else -1
    assert result["paper_contract_outcome"] == "TIMEOUT_FLAT"
    assert result["stop_price"] == pytest.approx(100 - sign * stop * 10)
    assert result["target_price"] == pytest.approx(100 + sign * target * 10)
    assert result["timeout_ts"] == result["entry_ts"] + pd.Timedelta(minutes=5 * horizon)
    assert result["paper_exit_ts"] == result["timeout_ts"]
    assert result["entry_ts"] > result["known_at_ts"] > result["event_anchor_ts"]


@pytest.mark.parametrize("setup", ["setup3", "setup5"])
def test_side_specific_contract_requires_side_even_in_single_contract_archive(setup):
    long_contract = load_contracts()[f"es_frvp_{setup}_long_eval_v2"]
    for catalog in (None, {long_contract.contract_id: long_contract}):
        with pytest.raises(ValueError, match="side is required"):
            resolve_contract(instrument="ES", strategy="frvp", model_family="continuation",
                             setup_family=setup, contracts=catalog)
    wrong = evaluate_signal(signal(setup_family=setup, model_family="continuation", side="short"),
                            tape(), long_contract)
    assert wrong["paper_contract_outcome"] == "INVALID"
    assert wrong["paper_outcome_reason"] == "contract_side_mismatch"
    assert wrong["entry_ts"] is None


def test_frvp_v2_archived_contracts_and_costs_remain_unchanged():
    old = load_contracts(ROOT / "configs/outcome_contracts/b1_v1.json")
    active = load_contracts(ROOT / "configs/outcome_contracts/b1_v2.json")
    affected = [c for c in old.values() if c.strategy == "frvp" and c.setup_family in {"setup2", "setup3", "setup4", "setup5"}]
    assert len(affected) == 8
    assert len(active) == 45
    assert sum(c.risk.rule == "UNRESOLVED" for c in active.values()) == 13
    for c in old.values():
        if c in affected:
            assert c.risk.rule == "UNRESOLVED"
            assert c.contract_id not in active
            for side in ("long", "short"):
                new = resolve_contract(instrument=c.instrument, strategy=c.strategy, model_family="meta",
                                       setup_family=c.setup_family, side=side)
                assert new.contract_version == c.contract_version + 1
                assert new.sha256 != c.sha256
                assert new.costs == c.costs
        else:
            assert active[c.contract_id] == c
    evidence = json.loads((ROOT / "docs/audits/b1_frvp_risk_evidence_20260917.json").read_text())
    for name in ("b1_v1.json", "b1_v1.lock.json"):
        relative = f"configs/outcome_contracts/{name}"
        assert file_hash(ROOT / relative) == evidence["source_hashes"][relative]


def test_v2_side_specific_artifacts_preserve_all_surfaces_and_replay(tmp_path):
    signals = [signal(event_id=f"{setup}-{side}", setup_family=setup, side=side,
                      model_family=f"continuation_{setup}", atr=10, training_label_outcome=label, quantity=quantity)
               for setup, side, label, quantity in (("setup3", "long", 0, 1), ("setup3", "short", 1, 2),
                                                    ("setup5", "long", 1, 3), ("setup5", "short", 0, 4))]
    original = [s.model_dump() for s in signals]
    first = run_outcome_evaluation(signals, tape(n=100), output_dir=tmp_path / "run", simulator=simulator())
    second = replay_outcome_run(tmp_path / "run", tmp_path / "replay")
    assert first == second
    assert first["engine_version"] == "b1-v5"
    assert len(first["evaluation_contracts"]) == 4
    labels = pd.read_csv(tmp_path / "run/training_label_outcomes.csv")
    assert labels.training_label_outcome.tolist() == [0, 1, 1, 0]
    assert [s.model_dump() for s in signals] == original
    for namespace in ("training_label_outcomes", "paper_contract_outcomes", "forward_returns", "simulated_positions"):
        frame = pd.read_csv(tmp_path / f"run/{namespace}.csv")
        assert frame.evaluation_contract_version.eq(2).all()
        for _, row in frame.iterrows():
            assert row.evaluation_contract_hash == load_contracts()[row.evaluation_contract_id].sha256
    positions = pd.read_csv(tmp_path / "run/simulated_positions.csv")
    taken = positions.loc[positions.position_taken]
    assert len(taken) == 1
    assert taken.net_pnl.iloc[0] == taken.gross_pnl.iloc[0] - taken.total_cost.iloc[0]
    assert positions.loc[~positions.position_taken, "net_pnl"].isna().all()
    paper = pd.read_csv(tmp_path / "run/paper_contract_outcomes.csv")
    assert paper.paper_contract_outcome.eq("TIMEOUT_FLAT").all()


def test_same_side_frvp_targets_change_paper_outcomes_not_forward_response():
    s3 = signal(setup_family="setup3", model_family="continuation_setup3", atr=10)
    s2 = signal(setup_family="setup2", model_family="continuation_setup2", atr=10)
    t = tape(n=100)
    t.bars.loc[t.bars.index[0], "high"] = 109.5
    results = [evaluate_signal(s, t, resolve_contract(instrument="ES", strategy="frvp",
               model_family="continuation", setup_family=s.setup_family, side="long")) for s in (s3, s2)]
    assert [r["paper_contract_outcome"] for r in results] == ["TARGET", "TIMEOUT_FLAT"]
    assert {k: v for k, v in results[0].items() if k.startswith("forward_")} == {
        k: v for k, v in results[1].items() if k.startswith("forward_")}


def test_frvp_registry_inventory_does_not_bind_opposite_side_contracts():
    contracts = load_contracts()
    rows = [r for r in registry_inventory()["rows"] if r["strategy"] == "frvp"]
    assert rows
    for row in rows:
        assert row["contract_ids"]
        assert all(contracts[key].side in {"long_or_short", row["side"]} for key in row["contract_ids"])
