"""Section 3 R1/R2/R5: four surfaces, both ledger kinds, reporting guard, full-session coverage.

Synthetic two-model fixture per stream on the frozen route bindings v2 and B1 v3
contracts. ES (ticks) and EURUSD (pips) are built and checked separately, never pooled.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_testing import outcome_surfaces as surfaces
from model_testing.outcome_contracts import file_hash, load_contracts
from model_testing.outcome_diagnostics import ModelSetupContext, SetupObservation
from model_testing.outcome_engine import (
    ENGINE_VERSION, TIME_FIELDS, EvaluationTape, evaluate_signal, position_costs, simulate_positions,
)
from model_testing.outcome_routes import load_route_bindings
from model_testing.outcome_surfaces import (
    ATTRIBUTION_FIELDS, FullPolicyBinding, SurfaceReportError, build_stream_surfaces, capacity_ledger_id,
    guard_report_table, independent_ledger_id, ledger_summary, markout_store_id, persist_stream_surfaces,
)
from test_outcome_contracts import signal, simulator
from test_outcome_ote_contracts import ote_signal

STREAMS = ["ES", "EURUSD"]
PX = {"ES": 100.0, "EURUSD": 1.1}
ATR = {"ES": 1.0, "EURUSD": 0.002}
UNIT = {"ES": "ticks", "EURUSD": "pips"}
PROTOCOL = ROOT / "configs/b3_b4_candidate_review_v3_20261002.json"
SELECTION_ORDER = json.loads(PROTOCOL.read_text(encoding="utf-8"))["ledgers"]["instrument_capacity_rule"]["selection_order"]
# M1 precedes M2 in the frozen selection order. ES: pooled then exact. EURUSD: exact then pooled.
MODELS = {
    "ES": [dict(queue_id=10, model_id="frvp_long_reversal_xgb_v1", model_family="reversal", setup="setup1"),
           dict(queue_id=5, model_id="frvp_long_continuation_setup2_xgb_v1", model_family="continuation_setup2", setup="setup2")],
    "EURUSD": [dict(queue_id=22, model_id="long_reversal_tcn_v2_20260525_narrow48", model_family="reversal", setup="reversal"),
               dict(queue_id=21, model_id="long_ote_meta_tcn_champion", model_family="meta", setup="reversal")],
}
POOLED = {"ES": 0, "EURUSD": 1}
N = 60
# name -> (model index, entry bar, seconds before that bar's open, calibrated probability)
EVENTS = {
    "b01-m1": (0, 2, 1, 0.9), "a02-m2": (1, 2, 1, 0.9),      # tie: selection order decides capacity
    "c03-m2": (1, 10, 1, 0.3), "c04-m1": (0, 10, 1, 0.5),    # below threshold; equal to threshold
    "c05-m1": (0, 11, 1, 0.9),                               # own ledger is busy
    "c11-m2": (1, 12, 1, 0.9),                               # enters at the open the stop gap exits on
    "d06-m2": (1, 20, 1, 0.9),
    "b09-m2": (1, 40, 120, 0.9), "a10-m1": (0, 40, 1, 0.9),  # earlier decision beats selection order
    "e07-m2": (1, 57, 1, 0.9),                               # still open at the end of the tape
}


def bars(stream, start):
    times = pd.date_range(start, periods=N, freq="5min")
    px, atr = PX[stream], ATR[stream]
    opens = [px + i * 0.001 * atr for i in range(N)]
    frame = pd.DataFrame(dict(open_ts=times, open=opens, high=[o + 0.1 * atr for o in opens],
                              low=[o - 0.1 * atr for o in opens], close=opens, is_complete=True))
    for i in (4, 22, 42):  # targets
        frame.loc[i, "high"] = frame.loc[i, "open"] + 3 * atr
    gap = px - 3 * atr     # bar 12 gaps through every open stop
    frame.loc[12, ["open", "high", "low", "close"]] = [gap, gap + 0.1 * atr, gap - 0.1 * atr, gap]
    if stream == "ES":
        frame["coordinate_id"] = "ESU6"
    return frame


def make_signal(stream, start, name, *, model, bar, lead, probability, selected=True, counterfactual=False):
    base = pd.Timestamp(start) - pd.Timedelta(minutes=5)  # on the hour: the structural ATR close
    known = pd.Timestamp(start) + pd.Timedelta(minutes=5 * bar) - pd.Timedelta(seconds=lead)
    event_id = f"{stream}-{name}"
    setup = dict(strategy="frvp" if stream == "ES" else "ote", side="long", known_at_ts=known, detector_version="fixture_v1")
    setups = [SetupObservation(setup_event_id=event_id + "-setup", setup_family=model["setup"], selected=selected, **setup)]
    if counterfactual:
        setups.append(SetupObservation(setup_event_id=event_id + "-other", setup_family="other_family", selected=False, **setup))
    context = ModelSetupContext(collection_id="section3_fixture", opportunity_id=event_id, prediction_id=event_id + "-p",
        available_at_ts=known, coverage="all_model_observations", setup_coverage="complete",
        expected_model_ids=(model["model_id"],), setups=tuple(setups), model_artifact_hash="e" * 64,
        raw_score=probability, calibrated_probability=probability, threshold=0.5, threshold_source="fixture",
        threshold_passed=probability >= 0.5, context_source_hash="c" * 64)
    proof = dict(value=ATR[stream], source_close_ts=base, history_start_ts="2026-09-13T00:00:00Z",
                 available_at_ts=base, coordinate_id="ESU6" if stream == "ES" else None, source_hash="c" * 64)
    values = dict({field: known for field in TIME_FIELDS}, event_id=event_id, model_id=model["model_id"],
                  model_family=model["model_family"], setup_family=model["setup"], side="long",
                  event_anchor_ts=base, atr_available_at=base, atr=ATR[stream], audit_context=context)
    if stream == "ES":
        return signal(atr_provenance=proof, **values)
    proof.update(definition="completed_1h_structural_atr", producer_id="b1_structural_1h_wilder14_v1", timeframe_minutes=60)
    return ote_signal("reversal", atr_provenance=proof, **values)


def build(stream, start="2026-09-14T14:05:00Z", full_policy=None):
    models = MODELS[stream]
    signals = [make_signal(stream, start, name, model=models[m], bar=bar, lead=lead, probability=p,
                           counterfactual=name == "d06-m2")
               for name, (m, bar, lead, p) in EVENTS.items()]
    # A pooled prediction whose only contemporaneous same-side setup was not detector-selected.
    signals.append(make_signal(stream, start, "f08-pooled", model=models[POOLED[stream]], bar=30, lead=1,
                               probability=0.9, selected=False))
    frame = bars(stream, start)
    tape = EvaluationTape(frame, pd.DataFrame(dict(open_ts=frame["open_ts"], session_id="s1")), timeframe_minutes=5,
        calendar_id=f"synthetic_{stream}_v1", calendar_source="synthetic fixture",
        coverage_start=frame["open_ts"].iloc[0] - pd.Timedelta(days=1),
        coverage_end=frame["open_ts"].iloc[-1] + pd.Timedelta(minutes=5), require_completion=True)
    return build_stream_surfaces(stream, signals, tape, thresholds={m["model_id"]: 0.5 for m in models},
        model_hashes={m["model_id"]: "e" * 64 for m in models}, selection_order=SELECTION_ORDER,
        protocol_sha256=file_hash(PROTOCOL), full_policy=full_policy)


@pytest.fixture(scope="module")
def built():
    return {stream: build(stream) for stream in STREAMS}


def taken(result, ledger_id):
    frame = result["ledgers"][ledger_id]
    return {event.split("-", 1)[1] for event in frame.loc[frame.position_taken, "event_id"]}


def reasons(result, ledger_id):
    frame = result["ledgers"][ledger_id]
    return {event.split("-", 1)[1]: reason for event, reason in zip(frame.event_id, frame.no_trade_reason) if reason}


def test_schema_matches_the_frozen_b3_protocol():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    schema = surfaces.surface_schema()
    assert tuple(protocol["surfaces"]["attribution_fields"]) == ATTRIBUTION_FIELDS
    assert schema["position_variants"] == protocol["ledgers"]["naming"]["variant_values"]
    assert schema["cost_cases"] == protocol["ledgers"]["naming"]["cost_case_values"]
    assert schema["ledger_kinds"] == list(protocol["ledgers"]["kinds"])
    assert schema["schema_version"] == "section3-surfaces-v1" and schema["engine_version"] == ENGINE_VERSION == "b1-v5"
    assert schema["scored_collection_enabled"] is False
    assert markout_store_id("ES", 5) == "mk.ES.5.raw_score_gross_forward_response"
    assert independent_ledger_id("EURUSD", 22, "setup_only", "base") == "ind.EURUSD.22.setup_only.base"
    assert capacity_ledger_id("ES", "double_all_costs") == "cap.ES.instrument_policy.double_all_costs"
    with pytest.raises(ValueError):
        capacity_ledger_id("6E", "base")


@pytest.mark.parametrize("stream", STREAMS)
def test_route_to_surfaces_to_both_ledger_kinds_end_to_end(stream, built, tmp_path):
    result = built[stream]
    m1, m2 = (m["queue_id"] for m in MODELS[stream])
    expected = {independent_ledger_id(stream, q, v, c) for q in (m1, m2) for v in ("setup_only", "setup_threshold")
                for c in ("base", "double_all_costs")} | {capacity_ledger_id(stream, c) for c in ("base", "double_all_costs")}
    assert set(result["ledgers"]) == expected
    assert set(result["markouts"]) == {markout_store_id(stream, m1), markout_store_id(stream, m2)}

    # Surface 4: not_defined is a status, never a ledger, rows or a number.
    assert {q: r["status"] for q, r in result["full_policy"].items()} == {m1: "not_defined", m2: "not_defined"}
    assert not any("full_policy" in ledger_id for ledger_id in result["ledgers"])
    with pytest.raises(SurfaceReportError, match="never rendered as zero"):
        ledger_summary(result["full_policy"][m1])

    # Independent ledgers: no model or variant consumes another's capacity.
    pooled_m1 = POOLED[stream] == 0
    assert taken(result, independent_ledger_id(stream, m1, "setup_only", "base")) == {"b01-m1", "c04-m1", "a10-m1"}
    assert taken(result, independent_ledger_id(stream, m1, "setup_threshold", "base")) == {"b01-m1", "c04-m1", "a10-m1"}
    assert taken(result, independent_ledger_id(stream, m2, "setup_only", "base")) == {
        "a02-m2", "c03-m2", "c11-m2", "d06-m2", "b09-m2", "e07-m2"}
    assert taken(result, independent_ledger_id(stream, m2, "setup_threshold", "base")) == {
        "a02-m2", "c11-m2", "d06-m2", "b09-m2", "e07-m2"}
    assert reasons(result, independent_ledger_id(stream, m1, "setup_only", "base"))["c05-m1"] == "concurrent_position_limit"
    assert reasons(result, independent_ledger_id(stream, m2, "setup_threshold", "base"))["c03-m2"] == "probability_below_frozen_threshold"

    # Instrument capacity: decision time, then selection order; a gap-open exit does not free its own open.
    cap = capacity_ledger_id(stream, "base")
    assert taken(result, cap) == {"b01-m1", "c04-m1", "d06-m2", "b09-m2", "e07-m2"}
    cap_reasons = reasons(result, cap)
    assert {cap_reasons[e] for e in ("a02-m2", "c05-m1", "c11-m2", "a10-m1")} == {"concurrent_position_limit"}
    assert cap_reasons["c03-m2"] == "probability_below_frozen_threshold"
    assert set(result["ledgers"][cap]["surface"]) == {"setup_threshold"}
    assert list(result["ledgers"][cap]["admission_sequence"]) == list(range(len(result["ledgers"][cap])))

    # Pooled counterfactual: markout-only, flagged non-executable, never a position.
    pooled_queue = m1 if pooled_m1 else m2
    forward = result["markouts"][markout_store_id(stream, pooled_queue)]["forward_response"]
    row = forward.loc[forward.event_id.eq(f"{stream}-f08-pooled")].iloc[0]
    assert row["counterfactual_markout_only"] and not row["executable_route"] and row["forward_return_1_status"] == "OK"
    paper = result["markouts"][markout_store_id(stream, pooled_queue)]["paper_contract_outcome"]
    assert not paper.loc[paper.event_id.eq(f"{stream}-f08-pooled"), "executable"].item()
    other = result["markouts"][markout_store_id(stream, m2)]["forward_response"]
    other = other.loc[other.observed_setup_id.eq(f"{stream}-d06-m2-other")]
    assert len(other) == 1 and not other["executable_route"].item()
    for ledger_id, frame in result["ledgers"].items():
        assert not frame.loc[frame.event_id.eq(f"{stream}-f08-pooled"), "position_taken"].any()
        assert not frame["setup_event_id"].astype(str).str.endswith("-other").any()
        assert reasons(result, ledger_id).get("f08-pooled", "pooled_counterfactual_setup_non_executable") == \
            "pooled_counterfactual_setup_non_executable"

    # Attribution travels with every row of every store.
    tables = list(result["ledgers"].values()) + [t for store in result["markouts"].values() for t in store.values()]
    for table in tables:
        assert set(ATTRIBUTION_FIELDS) <= set(table.columns)
        assert set(table["stream"]) == {stream} and set(table["unit"]) == {UNIT[stream]}
        assert set(table["engine_version"]) == {"b1-v5"}
    for store in result["markouts"].values():
        for table in store.values():
            assert not {"net_pnl", "gross_pnl", "total_cost", "position_taken"} & set(table.columns)
            assert set(table["cost_case"]) == {"none"} and set(table["store_kind"]) == {"markout_store"}
    for frame in result["ledgers"].values():
        admitted = frame.loc[frame.position_taken]
        assert admitted[list(ATTRIBUTION_FIELDS)].notna().all().all()
        assert (admitted["route_id"] == admitted["queue_id"].astype(str) + ":" + admitted["route_id"].str.split(":").str[1]
                + ":" + admitted["outcome_contract_id"]).all()

    # Persist, read back: per-trade sums equal ledger totals and costs are applied exactly once.
    manifest = persist_stream_surfaces(result, tmp_path)
    assert manifest["full_policy"] == {str(q): result["full_policy"][q] for q in sorted((m1, m2))}
    assert not list((tmp_path / stream).rglob("*full_policy*"))
    assert len(list((tmp_path / stream / "ledgers").glob("*.csv"))) == len(result["ledgers"])
    assert not list((tmp_path / stream / "ledgers").glob("mk.*")) and not list((tmp_path / stream / "markouts").rglob("ind.*"))
    contracts = load_contracts()
    for ledger_id, summary in manifest["ledger_summaries"].items():
        frame = pd.read_csv(tmp_path / stream / "ledgers" / f"{ledger_id}.csv", float_precision="round_trip")
        closed = frame.loc[frame.position_status.eq("CLOSED")]
        assert summary == ledger_summary(frame)
        assert summary["gross_pnl"] == pytest.approx(closed.gross_pnl.sum())
        assert summary["total_cost"] == pytest.approx(closed.total_cost.sum())
        assert summary["net_pnl"] == pytest.approx(closed.net_pnl.sum())
        assert summary["net_pnl"] == pytest.approx(summary["gross_pnl"] - summary["total_cost"])
        factor = 2.0 if summary["cost_case"] == "double_all_costs" else 1.0
        for row in closed.itertuples():
            once = position_costs(entry_ts=row.entry_ts, exit_ts=row.position_exit_ts,
                cost=contracts[row.outcome_contract_id].costs, quantity=1.0, accounting_unit=UNIT[stream])
            assert row.total_cost == pytest.approx(factor * once["total_cost"]) and once["total_cost"] > 0
            assert row.net_pnl == pytest.approx(row.gross_pnl - row.total_cost)
        open_rows = frame.loc[frame.position_status.eq("UNRESOLVED")]
        assert open_rows[["gross_pnl", "net_pnl", "total_cost"]].isna().all().all()
        assert summary["unresolved_open_exposure"] == len(open_rows)
    base = pd.read_csv(tmp_path / stream / "ledgers" / f"{cap}.csv")
    double = pd.read_csv(tmp_path / stream / "ledgers" / f"{capacity_ledger_id(stream, 'double_all_costs')}.csv")
    assert list(base.event_id) == list(double.event_id) and list(base.position_taken) == list(double.position_taken)
    assert base.gross_pnl.dropna().tolist() == double.gross_pnl.dropna().tolist()
    assert manifest["ledger_summaries"][cap]["unresolved_open_exposure"] == 1
    assert manifest["scored_collection_enabled"] is False and manifest["promotion_eligible"] is False
    with pytest.raises(FileExistsError):
        persist_stream_surfaces(result, tmp_path)


@pytest.mark.parametrize("stream", STREAMS)
def test_surfaces_are_deterministic(stream, built, tmp_path):
    first = persist_stream_surfaces(built[stream], tmp_path / "a")
    second = persist_stream_surfaces(build(stream), tmp_path / "b")
    assert first["surface_hash"] == second["surface_hash"] and first["files"] == second["files"]


def test_reporting_guard_fails_closed(built):
    es, fx = built["ES"], built["EURUSD"]
    ind = es["ledgers"][independent_ledger_id("ES", 10, "setup_only", "base")]
    cap = es["ledgers"][capacity_ledger_id("ES", "base")]
    markout = es["markouts"][markout_store_id("ES", 10)]["paper_contract_outcome"]
    fx_ind = fx["ledgers"][independent_ledger_id("EURUSD", 22, "setup_only", "base")]
    assert guard_report_table(ind) is ind
    for mixed, message in (
            ([ind, cap], "ledgers are never mixed"),
            ([ind, markout], "markouts and positions"),
            ([ind, fx_ind], "ES and EURUSD"),
            ([ind, es["ledgers"][independent_ledger_id("ES", 10, "setup_only", "double_all_costs")]], "cost cases")):
        with pytest.raises(SurfaceReportError, match=message):
            ledger_summary(pd.concat(mixed, ignore_index=True))
    with pytest.raises(SurfaceReportError, match="single position ledger"):
        ledger_summary(markout)
    with pytest.raises(SurfaceReportError, match="single position ledger"):
        ledger_summary(pd.concat([ind, es["ledgers"][independent_ledger_id("ES", 5, "setup_only", "base")]]))
    with pytest.raises(SurfaceReportError, match="never rendered as zero"):
        guard_report_table(es["full_policy"][10])
    with pytest.raises(SurfaceReportError):
        guard_report_table(ind.drop(columns=["ledger_kind"]))


@pytest.mark.parametrize("stream", STREAMS)
def test_streams_models_and_units_fail_closed(stream, built):
    other = "EURUSD" if stream == "ES" else "ES"
    signals = [make_signal(other, "2026-09-14T14:05:00Z", "x", model=MODELS[other][0], bar=2, lead=1, probability=0.9)]
    frame = bars(stream, "2026-09-14T14:05:00Z")
    kwargs = dict(timeframe_minutes=5, calendar_id="synthetic_v1", calendar_source="synthetic fixture",
                  coverage_start=frame["open_ts"].iloc[0], coverage_end=frame["open_ts"].iloc[-1] + pd.Timedelta(minutes=5))
    schedule = pd.DataFrame(dict(open_ts=frame["open_ts"], session_id="s1"))
    common = dict(thresholds={}, model_hashes={}, selection_order=SELECTION_ORDER, protocol_sha256="p" * 64)
    with pytest.raises(ValueError, match="not a selected"):
        build_stream_surfaces(stream, signals, EvaluationTape(frame, schedule, require_completion=True, **kwargs), **common)
    with pytest.raises(ValueError, match="completion-enforcing"):
        build_stream_surfaces(stream, [], EvaluationTape(frame, schedule, **kwargs), **common)
    with pytest.raises(ValueError, match="unit"):
        build_stream_surfaces(stream, [], EvaluationTape(frame, schedule, require_completion=True, **kwargs),
                              unit=UNIT[other], **common)


@pytest.mark.parametrize("stream", STREAMS)
def test_default_ordering_is_unchanged_without_the_capacity_key(stream):
    """b1-v4 ordering (entry time, event ID) still applies when no priority is supplied."""
    models = MODELS[stream]
    pair = [make_signal(stream, "2026-09-14T14:05:00Z", name, model=models[m], bar=40, lead=lead, probability=0.9)
            for name, m, lead in (("b09-m2", 1, 120), ("a10-m1", 0, 1))]
    frame = bars(stream, "2026-09-14T14:05:00Z")
    tape = EvaluationTape(frame, pd.DataFrame(dict(open_ts=frame["open_ts"], session_id="s1")), timeframe_minutes=5,
        calendar_id="synthetic_v1", calendar_source="synthetic fixture", coverage_start=frame["open_ts"].iloc[0] - pd.Timedelta(days=1),
        coverage_end=frame["open_ts"].iloc[-1] + pd.Timedelta(minutes=5), require_completion=True)
    manifest, contracts, scope = load_route_bindings(), load_contracts(), surfaces.load_research_scope()
    used = {}
    outcomes = []
    for s in pair:
        c = surfaces.resolve_model_setup_route(model_id=s.model_id, setup_family=s.setup_family, instrument=stream,
                                               side="long", manifest=manifest, contracts=contracts)
        used[c.contract_id] = c
        outcomes.append(evaluate_signal(s, tape, c, scope=scope))
    sim = simulator(accounting_unit=UNIT[stream], max_quantity=1)
    default = {r["event_id"]: r["position_taken"] for r in simulate_positions(pair, outcomes, used, sim, scope=scope)}
    keyed = {r["event_id"]: r["position_taken"] for r in simulate_positions(
        pair, outcomes, used, sim, scope=scope, priority={models[0]["model_id"]: 0, models[1]["model_id"]: 1})}
    assert default == {f"{stream}-a10-m1": True, f"{stream}-b09-m2": False}
    assert keyed == {f"{stream}-a10-m1": False, f"{stream}-b09-m2": True}


# R5: full eligible session coverage.

SESSION_REASON_WORDS = ("session", "hour", "weekday", "asia", "london", "new_york", "overlap", "off_hours")


@pytest.mark.parametrize("stream", STREAMS)
def test_surfaces_1_to_3_and_all_ledgers_ignore_session_and_hour(stream, built):
    """Moving the identical fixture through every session changes report slices only."""
    reference, seen = built[stream], set()
    for hours in (0, 5, 10, 15, 20):
        shifted = build(stream, str(pd.Timestamp("2026-09-14T14:05:00Z") + pd.Timedelta(hours=hours)))
        assert set(shifted["ledgers"]) == set(reference["ledgers"])
        for ledger_id, frame in shifted["ledgers"].items():
            expected = reference["ledgers"][ledger_id]
            for column in ("event_id", "position_taken", "no_trade_reason", "position_status", "surface"):
                assert frame[column].tolist() == expected[column].tolist(), (ledger_id, column)
            assert pd.to_numeric(frame["gross_pnl"]).fillna(0).tolist() == pytest.approx(
                pd.to_numeric(expected["gross_pnl"]).fillna(0).tolist()), ledger_id
            assert not any(word in reason for reason in frame["no_trade_reason"] for word in SESSION_REASON_WORDS)
            seen |= set(frame.loc[frame.position_taken, "decision_session"])
        for store_id, tables in shifted["markouts"].items():
            for name, table in tables.items():
                assert len(table) == len(reference["markouts"][store_id][name])
    assert seen == {"asia", "london", "overlap", "new_york", "off_hours"}


@pytest.mark.parametrize("stream", STREAMS)
def test_frozen_full_policy_exclusion_is_only_a_surface_4_rejection_reason(stream, built):
    m1, m2 = MODELS[stream]
    policy = FullPolicyBinding(policy_id="fixture_policy_v1", policy_hash="f" * 64,
                               rejections={f"{stream}-d06-m2": ("frozen_policy_session_excluded_overlap",)})
    result = build(stream, full_policy={m2["model_id"]: policy})
    assert result["full_policy"][m2["queue_id"]] == dict(status="exact", policy_id="fixture_policy_v1", policy_hash="f" * 64)
    assert result["full_policy"][m1["queue_id"]]["status"] == "not_defined"
    assert not any(f".{m1['queue_id']}.full_policy." in ledger_id for ledger_id in result["ledgers"])
    surface_4 = independent_ledger_id(stream, m2["queue_id"], "full_policy", "base")
    assert reasons(result, surface_4)["d06-m2"] == "frozen_policy_session_excluded_overlap"
    assert reasons(result, surface_4)["c03-m2"] == "probability_below_frozen_threshold"
    for variant in ("setup_only", "setup_threshold"):
        ledger_id = independent_ledger_id(stream, m2["queue_id"], variant, "base")
        assert result["ledgers"][ledger_id].equals(built[stream]["ledgers"][ledger_id])
    for frame in result["ledgers"].values():
        flagged = frame.loc[frame["no_trade_reason"].str.contains("session")]
        assert set(flagged["surface"]) <= {"full_policy"}
    # The capacity ledger is fed by the exact full policy and labels the feed on every entry.
    cap = result["ledgers"][capacity_ledger_id(stream, "base")]
    assert set(cap.loc[cap.model_id.eq(m2["model_id"]), "surface"]) == {"full_policy"}
    assert set(cap.loc[cap.model_id.eq(m1["model_id"]), "surface"]) == {"setup_threshold"}


def test_no_session_or_hour_filter_exists_outside_surface_4():
    """Fails if session/hour logic appears anywhere but the report-slice helper."""
    banned_names = {"session_at", "hour", "weekday", "dayofweek", "day_of_week", "day_name", "tz_convert", "time"}
    tree = ast.parse(Path(surfaces.__file__).read_text(encoding="utf-8"))
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "report_slices":
            allowed |= {id(child) for child in ast.walk(node)}
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REPORT_SLICE_FIELDS" for t in node.targets):
            allowed |= {id(child) for child in ast.walk(node)}
    offenders = []
    for node in ast.walk(tree):
        if id(node) in allowed:
            continue
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in banned_names:
            offenders.append((name, node.lineno))
        if isinstance(node, ast.Constant) and node.value in surfaces.REPORT_SLICE_FIELDS:
            offenders.append((node.value, node.lineno))
    assert not offenders, f"session/hour logic outside report_slices: {offenders}"
    # The helper is called only to add slice columns, never inside a condition.
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.comprehension, ast.Compare)):
            calls = [c for c in ast.walk(node) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "report_slices"]
            assert not calls, "report_slices used in a condition"
    for function in (simulate_positions, evaluate_signal, surfaces._eligibility):
        source = inspect.getsource(function)
        for token in ("session_at", ".hour", "weekday", "dayofweek", "decision_session", "session_regime"):
            assert token not in source, f"{function.__name__} references {token}"


def test_app_reports_the_shared_schema_and_keeps_scored_registration_disabled():
    from ote_live.storage import research_execution

    assert research_execution.surface_schema is surfaces.surface_schema
    source = inspect.getsource(research_execution.ResearchExecutionLedger.report)
    assert '"surface_schema": surface_schema()' in source and '"scored_collection_enabled": False' in source
    assert set(surfaces.APP_VARIANT_ALIASES) <= set(research_execution.VARIANTS)
