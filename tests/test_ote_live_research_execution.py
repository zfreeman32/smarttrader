"""A7: executable-time simulation, independent books and fail-closed accounting."""
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.research_execution import ResearchContract, ResearchOpportunity
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.research_execution import ResearchExecutionLedger


START = datetime(2026, 9, 14, 14, 30, tzinfo=timezone.utc)
MODEL = "frvp_long_continuation_xgb_v1"


def contract(**updates):
    return ResearchContract(**{ "collection_version": "a7-test", "outcome_contract_ref": "draft-test-B1",
        "comparison_contract_ref": "draft-test-B3", "timeframe": "5m", "model_ids": (MODEL,),
        "entry_latency_seconds": 0, "entry_wait_seconds": 900, "max_bar_delay_seconds": 90,
        "tick_size": .25, "tick_value": 12.5, "stop_ticks": 4, "target_ticks": 8, "timeout_bars": 3,
        "ambiguity": "stop_first", "quantity": 1, "max_positions": 1, "max_contracts": 1,
        "concurrency_scope": "per_model", "allow_opposite_positions": False,
        "spread_ticks": 1, "slippage_ticks_per_side": .25, "commission_ticks_round_trip": .4,
        "cost_multipliers": (1., 2.), **updates})


def opportunity(key="one", **updates):
    return ResearchOpportunity(**{"opportunity_key": key, "collection_version": "a7-test", "model_id": MODEL,
        "setup_event_id": 1, "prediction_id": 1, "signal_decision_id": 1, "source_bar_version": "source-v1",
        "setup_type": "3", "setup_family": "continuation", "detector_version": "test-v1", "policy_sha256": "test-policy",
        "source_timestamp": START, "setup_observed_at": START + timedelta(minutes=5),
        "prediction_recorded_at": START + timedelta(minutes=5, seconds=1), "side": 1,
        "setup_matched": True, "threshold_crossed": True, "full_policy_passed": True, **updates})


def bar(minutes, **updates):
    timestamp = START + timedelta(minutes=minutes)
    observed = timestamp + timedelta(minutes=5, seconds=1)
    return MarketBar(**{"asset": "ES", "timeframe": "5m", "timestamp": timestamp,
        "source_timestamp": timestamp, "bar_version": f"bar-{minutes}", "is_complete": True,
        "feed_type": "live", "observation_kind": "live", "first_observed_at": observed,
        "last_observed_at": observed, "open": 100, "high": 100.5, "low": 99.5, "close": 100,
        "feature_context": {"ibkr_trading_hours": "20260914:0000-20260914:2359", "ibkr_timezone": "UTC"}, **updates})


@pytest.fixture
def ledger(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "research.sqlite") as store:
        repo = ResearchExecutionLedger(store)
        repo.register("test", contract(), started_at=START)
        yield repo


def advance(ledger, minutes, **updates):
    item = bar(minutes, **updates)
    ledger.process_bar("test", item, observed_at=item.last_observed_at)


def test_future_entry_costs_books_and_restart(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 5, open=90, high=95, low=89, close=91)
    assert len(ledger.positions("test", status="pending")) == 6  # decision was one second AFTER this open
    advance(ledger, 10)
    report = ledger.report("test")
    assert len(report["open_exposure"]) == 6 and not report["realized_outcomes"]
    assert {p["entry_price"] for p in report["open_exposure"]} == {100}
    assert {p["entry_at"] for p in report["open_exposure"]} == {(START + timedelta(minutes=10)).isoformat()}
    # Resume from persisted transitions rather than in-memory exposure.
    repo = ResearchExecutionLedger(ledger.store)
    advance(repo, 15, high=103, close=102)
    result = repo.report("test")
    assert not result["open_exposure"] and len(result["realized_outcomes"]) == 6
    for p in result["realized_outcomes"]:
        assert p["gross_pnl_ticks"] == 8
        assert p["costs_ticks"] == pytest.approx(1.9 * p["cost_multiplier"])
        assert p["net_pnl_dollars"] == pytest.approx((8 - 1.9 * p["cost_multiplier"]) * 12.5)
    assert result["scored_collection_enabled"] is False


@pytest.mark.parametrize("changes,counts", [
    ({"threshold_crossed": False, "full_policy_passed": False}, (2, 4)),
    ({"full_policy_passed": False, "policy_rejection_reasons": ("cooldown",)}, (4, 2)),
    ({"setup_matched": False}, (0, 6)),
    ({"common_rejection_reasons": ("backfilled_evaluation",)}, (0, 6)),
])
def test_comparison_uses_same_opportunities_without_changing_economics(ledger, changes, counts):
    ledger.submit("test", opportunity(**changes))
    assert len(ledger.positions("test", status="pending")) == counts[0]
    assert len(ledger.report("test")["rejections"]) == counts[1]


def test_same_bar_exits_do_not_free_capacity_at_earlier_open(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 10)
    ledger.submit("test", opportunity("two", source_timestamp=START + timedelta(minutes=10),
        setup_observed_at=START + timedelta(minutes=15, seconds=1),
        prediction_recorded_at=START + timedelta(minutes=15, seconds=1)))
    advance(ledger, 15)
    advance(ledger, 20, high=103, close=102)
    assert len(ledger.report("test")["realized_outcomes"]) == 6
    assert {p["reasons"][0] for p in ledger.report("test")["rejections"]} == {"position_limit"}


@pytest.mark.parametrize("side,updates,reason,gross", [
    (1, {"high": 103, "low": 98}, "stop", -4),
    (-1, {"high": 103, "low": 98}, "stop", -4),
    (-1, {"high": 100.5, "low": 97, "close": 98}, "target", 8),
])
def test_stop_target_and_short_accounting(ledger, side, updates, reason, gross):
    ledger.submit("test", opportunity(side=side))
    advance(ledger, 5)
    advance(ledger, 10, **updates)
    for state in ledger.report("test")["realized_outcomes"]:
        assert state["exit_reason"] == reason and state["gross_pnl_ticks"] == gross


def test_gap_stop_uses_worse_open_and_timeout_uses_close(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 10)
    advance(ledger, 15, open=97, high=98, low=96, close=97)
    assert {p["gross_pnl_ticks"] for p in ledger.report("test")["realized_outcomes"]} == {-12}
    ledger.register("timeout", contract(timeout_bars=1), started_at=START)
    ledger.submit("timeout", opportunity())
    for minute in (5, 10):
        item = bar(minute, close=100.5)
        ledger.process_bar("timeout", item, observed_at=item.last_observed_at)
    assert {p["exit_reason"] for p in ledger.report("timeout")["realized_outcomes"]} == {"timeout"}
    assert {p["gross_pnl_ticks"] for p in ledger.report("timeout")["realized_outcomes"]} == {2}


@pytest.mark.parametrize("updates", [
    {"is_complete": False}, {"feed_type": "delayed"}, {"observation_kind": "backfill"},
    {"feature_context": {}}, {"last_observed_at": START + timedelta(hours=1)},
])
def test_bad_bar_censors_exposure_without_realized_zero(ledger, updates):
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 10)
    advance(ledger, 15, **updates)
    report = ledger.report("test")
    assert len(report["censored_outcomes"]) == len(report["unresolved_exposure"]) == 6
    assert not report["realized_outcomes"]
    assert all("net_pnl_ticks" not in s for s in report["censored_outcomes"])


def test_gap_censor_preserves_capacity_reservation(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 10)
    advance(ledger, 20)  # missing expected 15-minute bar
    ledger.submit("test", opportunity("after_gap", source_timestamp=START + timedelta(minutes=20),
        setup_observed_at=START + timedelta(minutes=25, seconds=1),
        prediction_recorded_at=START + timedelta(minutes=25, seconds=1)))
    advance(ledger, 25)
    advance(ledger, 30)
    assert len(ledger.report("test")["unresolved_exposure"]) == 6
    assert {p["reasons"][0] for p in ledger.report("test")["rejections"]} == {"position_limit"}


def test_ambiguity_and_end_of_data_censor(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "ambiguity.sqlite") as store:
        ledger = ResearchExecutionLedger(store)
        ledger.register("test", contract(ambiguity="censor"), started_at=START)
        ledger.submit("test", opportunity())
        advance(ledger, 5)
        advance(ledger, 10, high=103, low=98)
        assert {p["reasons"][0] for p in ledger.report("test")["censored_outcomes"]} == {"ambiguous_stop_target"}
        ledger.finish("test", observed_at=START + timedelta(minutes=16))
        with pytest.raises(ValueError, match="finished"):
            advance(ledger, 15)


def test_eof_keeps_unresolved_exposure_separate(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 10)
    ledger.finish("test", observed_at=START + timedelta(minutes=16))
    assert len(ledger.report("test")["unresolved_exposure"]) == 6
    assert not ledger.report("test")["realized_outcomes"]


def test_idempotence_immutability_and_transaction_rollback(ledger):
    ledger.submit("test", opportunity())
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 5)
    with pytest.raises(ValueError, match="different evidence"):
        ledger.submit("test", opportunity(side=-1))
    with pytest.raises(ValueError, match="retroactive"):
        ledger.submit("test", opportunity("late"))
    assert ledger.db.execute("SELECT COUNT(*) FROM research_opportunities").fetchone()[0] == 1
    assert len(ledger.positions("test")) == 6
    with pytest.raises(ValueError, match="immutable"):
        ledger.register("test", contract(stop_ticks=10), started_at=START)
    for query in ("DELETE FROM research_runs", "UPDATE research_transitions SET status='resolved'",
                  "INSERT OR REPLACE INTO research_runs SELECT * FROM research_runs"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger.db.execute(query)
        ledger.db.rollback()


def test_revisions_preserve_original_outcome_but_remove_from_clean_report(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 10, high=103, close=102)
    original = ledger.report("test")["realized_outcomes"]
    revision = bar(10, bar_version="revised", high=100.5, last_observed_at=START + timedelta(minutes=16))
    ledger.process_bar("test", revision, observed_at=revision.last_observed_at)
    report = ledger.report("test")
    assert not report["realized_outcomes"]
    assert len(report["revision_tainted_outcomes"]) == 6
    assert [p["net_pnl_ticks"] for p in original] == [p["net_pnl_ticks"] for p in report["revision_tainted_outcomes"]]


def test_source_revision_taints_resolved_positions(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 5)
    advance(ledger, 10, high=103, close=102)
    revised = bar(0, bar_version="revised-source", last_observed_at=START + timedelta(minutes=16))
    ledger.process_bar("test", revised, observed_at=revised.last_observed_at)
    assert len(ledger.report("test")["revision_tainted_outcomes"]) == 6
    assert not ledger.report("test")["realized_outcomes"]


def test_pending_orders_with_unknown_fills_reserve_exposure(ledger):
    ledger.submit("test", opportunity())
    advance(ledger, 15)  # potential 14:40 fill is now unknowable
    report = ledger.report("test")
    assert len(report["unresolved_exposure"]) == 6
    assert all("entry_price" not in p for p in report["unresolved_exposure"])


def test_mixed_collections_and_future_dated_evidence_are_rejected(ledger):
    with pytest.raises(ValueError, match="collection/model"):
        ledger.submit("test", opportunity(collection_version="old"))
    ledger.submit("test", opportunity(prediction_recorded_at=START + timedelta(minutes=7)))
    assert len(ledger.report("test")["rejections"]) == 6


@pytest.mark.parametrize("updates", [{"mode": "scored"}, {"stop_ticks": float("nan")},
    {"cost_multipliers": (1., float("inf"))}, {"entry_latency_seconds": -1}, {"cost_multipliers": (2.,)},
    {"quantity": 2}, {"model_ids": ("ote_long",)}])
def test_contract_fails_closed(updates):
    with pytest.raises(ValueError):
        contract(**updates)


def test_cli_replay_and_frozen_ledger_isolation(tmp_path):
    from ote_live.scripts.replay_research_execution import main
    config, events, database, report = (tmp_path / name for name in ("contract.json", "events.jsonl", "replay.sqlite", "report.json"))
    config.write_text(contract().model_dump_json(), encoding="utf-8")
    records = [{"kind": "opportunity", "payload": opportunity().model_dump(mode="json")}]
    for minute in (5, 10):
        item = bar(minute)
        records.append({"kind": "bar", "observed_at": item.last_observed_at.isoformat(), "payload": item.model_dump(mode="json")})
    events.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    main(["--contract", str(config), "--events", str(events), "--database", str(database),
          "--report", str(report), "--run-id", "test", "--started-at", START.isoformat()])
    assert len(json.loads(report.read_text())["open_exposure"]) == 6
    with SQLiteLiveDataStore(database) as store:
        repo = ResearchExecutionLedger(store)
        assert len(repo.report("test")["open_exposure"]) == 6
        tables = [row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            if "paper" in table:
                assert store.connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0


@pytest.mark.parametrize("scope,opposite,limit,expected", [
    ("all_models", False, 1, "position_limit"),
    ("all_models", False, 2, "opposite_exposure"),
    ("all_models", True, 2, None),
    ("per_model", False, 1, None),
])
def test_model_concurrency_and_opposite_positions(tmp_path, scope, opposite, limit, expected):
    other = "ict_short_continuation_xgb_v1"
    with SQLiteLiveDataStore(tmp_path / "limits.sqlite") as store:
        ledger = ResearchExecutionLedger(store)
        ledger.register("test", contract(model_ids=(MODEL, other), concurrency_scope=scope,
            max_positions=limit, max_contracts=2, allow_opposite_positions=opposite), started_at=START)
        ledger.submit("test", opportunity("a"))
        ledger.submit("test", opportunity("b", model_id=other, side=-1))
        advance(ledger, 5)
        advance(ledger, 10)
        report = ledger.report("test")
        assert len(report["open_exposure"]) == (6 if expected else 12)
        assert {p["reasons"][0] for p in report["rejections"]} == ({expected} if expected else set())


def test_quantity_limit_and_costs_are_per_contract(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "quantity.sqlite") as store:
        ledger = ResearchExecutionLedger(store)
        ledger.register("test", contract(quantity=2, max_positions=3, max_contracts=3), started_at=START)
        ledger.submit("test", opportunity("a"))
        ledger.submit("test", opportunity("b"))
        advance(ledger, 5)
        advance(ledger, 10, high=103, close=102)
        report = ledger.report("test")
        assert {p["reasons"][0] for p in report["rejections"]} == {"contract_limit"}
        assert {p["gross_pnl_ticks"] for p in report["realized_outcomes"]} == {16}
        for p in report["realized_outcomes"]:
            assert p["costs_ticks"] == pytest.approx(3.8 * p["cost_multiplier"])


def test_latency_expiry_and_initial_missing_bar(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "availability.sqlite") as store:
        ledger = ResearchExecutionLedger(store)
        ledger.register("test", contract(entry_latency_seconds=300, entry_wait_seconds=1), started_at=START)
        ledger.submit("test", opportunity())
        for minute in (5, 10, 15):
            advance(ledger, minute)
        assert {p["reasons"][0] for p in ledger.report("test")["rejections"]} == {"entry_wait_expired"}
        ledger.register("missing", contract(), started_at=START)
        ledger.submit("missing", opportunity())
        item = bar(10)  # Cannot infer what happened in the absent 5-minute bar.
        ledger.process_bar("missing", item, observed_at=item.last_observed_at)
        assert len(ledger.report("missing")["censored_outcomes"]) == 6


def test_verified_market_closure_is_not_a_missing_bar(tmp_path):
    start = datetime(2026, 9, 14, 20, 50, tzinfo=timezone.utc)
    context = {"ibkr_trading_hours": "20260914:0000-20260914:2100,20260914:2200-20260914:2359",
               "ibkr_timezone": "UTC"}
    with SQLiteLiveDataStore(tmp_path / "closure.sqlite") as store:
        ledger = ResearchExecutionLedger(store)
        ledger.register("test", contract(timeout_bars=10), started_at=start)
        ledger.submit("test", opportunity(source_timestamp=start,
            setup_observed_at=start + timedelta(minutes=5), prediction_recorded_at=start + timedelta(minutes=5)))
        for stamp in (start + timedelta(minutes=5), start + timedelta(minutes=70)):
            observed = stamp + timedelta(minutes=5)
            item = bar(0, timestamp=stamp, source_timestamp=stamp, bar_version=stamp.isoformat(),
                       first_observed_at=observed, last_observed_at=observed, feature_context=context)
            ledger.process_bar("test", item, observed_at=observed)
        assert len(ledger.report("test")["open_exposure"]) == 6
        assert not ledger.report("test")["censored_outcomes"]


@pytest.mark.parametrize("strategy", ["FRVP", "ICT"])
def test_a5_a6_audit_adapter_uses_persisted_links_and_common_cohort(tmp_path, strategy):
    from test_ote_live_shadow_policy import manifest as manifest_fixture, inputs, CONTEXT
    from ote_live.storage.setup_events import SetupEventRepository
    from ote_live.storage.repositories import LiveAuditRepository
    from ote_live.policies.decision_engine import LiveDecisionEngine
    from ote_live.policies.shadow import match_shadow_setups

    manifest = manifest_fixture.__wrapped__()
    if strategy == "ICT":
        manifest = manifest.model_copy(update={"model_id": "ict_short_continuation_xgb_v1"})
    prediction, snapshot, _ = inputs(manifest, idx=0, probability=.2, collection="a7-test")
    with SQLiteLiveDataStore(tmp_path / "linked.sqlite") as store:
        history = SetupEventRepository(store)
        setup, = history.record_observations([{
            "setup_type": "3" if strategy == "FRVP" else "premium_discount_continuation",
            "setup_side": -1, "selected": True,
        }], bar=bar(0), strategy=strategy, collection_version="a7-test", source_bar_version="v1",
            observed_at=prediction.prediction_recorded_at_utc)
        audit = LiveAuditRepository(store)
        result = LiveDecisionEngine(audit_repository=audit).evaluate_prediction(
            prediction, feature_snapshot=snapshot, runtime_manifest=manifest, live_policy=manifest.live_policy,
            policy_context=CONTEXT, shadow_mode=True, shadow_setup_match=match_shadow_setups(prediction, snapshot, [setup]))
        ledger = ResearchExecutionLedger(store)
        ledger.register("test", contract(model_ids=(manifest.model_id,)), started_at=START)
        with pytest.raises(ValueError, match="association"):
            ledger.submit_audit("test", signal_decision_id=result.audit_record.signal_decision_id,
                                setup_event_id=setup.event_id)
        history.link_prediction(setup.event_id, result.audit_record.prediction_id, model_id=manifest.model_id,
                                decision="shadow", observed_at=prediction.prediction_recorded_at_utc)
        ledger.submit_audit("test", signal_decision_id=result.audit_record.signal_decision_id, setup_event_id=setup.event_id)
        assert len(ledger.positions("test", status="pending")) == 2
        assert len(ledger.report("test")["rejections"]) == 4
        assert {p["variant"] for p in ledger.positions("test", status="pending")} == {"setup_only"}
        advance(ledger, 5)
        advance(ledger, 10, low=97, close=98)
        outcomes = ledger.report("test")["realized_outcomes"]
        assert len(outcomes) == 2
        assert {p["prediction_id"] for p in outcomes} == {result.audit_record.prediction_id}
