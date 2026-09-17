"""Execution-aware diagnostic research ledger, isolated from paper/broker paths.

Each variant/cost case owns an independent book. Immutable lifecycle snapshots
are authoritative; open positions and realized outcomes are separate queries.
Scored registration is deliberately unsupported until B1/B3 are finalized.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from datetime import datetime, timedelta

from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.research_execution import ResearchContract, ResearchOpportunity
from ote_live.ingestion.base import ensure_utc, timeframe_to_timedelta
from ote_live.ingestion.provenance import assess_shadow_bar_eligibility
from ote_live.storage.collection import canonical_json


VARIANTS = ("setup_only", "setup_threshold", "setup_full_policy")
TABLES = ("research_runs", "research_opportunities", "research_bars", "research_transitions")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY, contract_json TEXT NOT NULL, started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_opportunities (
    run_id TEXT NOT NULL, opportunity_key TEXT NOT NULL, payload_json TEXT NOT NULL,
    PRIMARY KEY(run_id, opportunity_key)
);
CREATE TABLE IF NOT EXISTS research_bars (
    run_id TEXT NOT NULL, source_timestamp TEXT NOT NULL, bar_version TEXT NOT NULL,
    observed_at TEXT NOT NULL, payload_json TEXT NOT NULL,
    PRIMARY KEY(run_id, source_timestamp, bar_version)
);
CREATE TABLE IF NOT EXISTS research_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    position_key TEXT NOT NULL, observed_at TEXT NOT NULL,
    status TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS research_latest ON research_transitions(run_id, position_key, id);
CREATE TABLE IF NOT EXISTS research_cursors (
    run_id TEXT PRIMARY KEY, clock_at TEXT NOT NULL,
    last_source_at TEXT, finished INTEGER NOT NULL DEFAULT 0
);
"""


def _time(value):
    return ensure_utc(datetime.fromisoformat(value))


def _stamp(value):
    if value.tzinfo is None:
        raise ValueError("Research times must have an explicit timezone")
    return ensure_utc(value).isoformat()


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class ResearchExecutionLedger:
    def __init__(self, store):
        self.store = store
        self.db = store.connection
        self.db.executescript(_SCHEMA)
        for table in TABLES:
            for operation in ("UPDATE", "DELETE"):
                self.db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} "
                    f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'research ledger is append-only'); END"
                )
            # Prevent INSERT OR REPLACE from bypassing UPDATE/DELETE protection.
            keys = {"research_runs": ("run_id",), "research_opportunities": ("run_id", "opportunity_key"),
                    "research_bars": ("run_id", "source_timestamp", "bar_version"),
                    "research_transitions": ("id",)}[table]
            condition = " AND ".join(f"{key}=NEW.{key}" for key in keys)
            self.db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_replace BEFORE INSERT ON {table} "
                            f"WHEN EXISTS (SELECT 1 FROM {table} WHERE {condition}) "
                            "BEGIN SELECT RAISE(ABORT, 'research ledger is append-only'); END")
        self.db.commit()

    @contextmanager
    def _transaction(self):
        # Serialize read/decide/append across separate replay connections.
        if self.db.in_transaction:
            raise RuntimeError("Research operations require their own transaction")
        self.db.execute("BEGIN IMMEDIATE")
        with self.db:
            yield

    def register(self, run_id: str, contract: ResearchContract, *, started_at: datetime):
        """Explicit diagnostic run; never toggles a collector or paper registry."""
        contract = ResearchContract.model_validate(contract.model_dump())
        if not run_id.strip():
            raise ValueError("run_id must be explicit")
        payload = canonical_json(contract.model_dump(mode="json"))
        start = _stamp(started_at)
        existing = self.db.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
        if existing:
            if existing["contract_json"] != payload or existing["started_at"] != start:
                raise ValueError("Run identity already binds different immutable assumptions")
            return
        with self._transaction():
            self.db.execute("INSERT INTO research_runs VALUES (?,?,?)", (run_id, payload, start))
            self.db.execute("INSERT INTO research_cursors(run_id,clock_at) VALUES (?,?)", (run_id, start))

    def contract(self, run_id):
        row = self.db.execute("SELECT contract_json FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return ResearchContract.model_validate_json(row[0])

    def _clock(self, run_id, observed_at):
        cursor = self.db.execute("SELECT * FROM research_cursors WHERE run_id=?", (run_id,)).fetchone()
        if cursor is None:
            raise KeyError(run_id)
        if cursor["finished"]:
            raise ValueError("Research run is finished")
        if _time(cursor["clock_at"]) > observed_at:
            raise ValueError("Replay must follow observation time; retroactive submissions are forbidden")
        self.db.execute("UPDATE research_cursors SET clock_at=? WHERE run_id=?", (_stamp(observed_at), run_id))
        return cursor

    def _append(self, run_id, state, observed_at, **changes):
        state = {**state, **changes}
        self.db.execute("INSERT INTO research_transitions(run_id,position_key,observed_at,status,payload_json) "
                        "VALUES (?,?,?,?,?)", (run_id, state["position_key"], _stamp(observed_at),
                                              state["status"], canonical_json(state)))
        return state

    def positions(self, run_id, *, status=None):
        self.contract(run_id)
        rows = self.db.execute(
            "SELECT t.payload_json FROM research_transitions t WHERE run_id=? AND "
            "id=(SELECT MAX(id) FROM research_transitions WHERE run_id=t.run_id AND position_key=t.position_key) "
            "ORDER BY t.id", (run_id,))
        states = [json.loads(row[0]) for row in rows]
        return [state for state in states if status is None or state["status"] == status]

    def submit(self, run_id, opportunity: ResearchOpportunity):
        """Submit explicit diagnostic input to all predeclared variant/cost books."""
        c = self.contract(run_id)
        o = ResearchOpportunity.model_validate(opportunity.model_dump())
        if o.collection_version != c.collection_version or o.model_id not in c.model_ids:
            raise ValueError("Opportunity does not belong to this collection/model contract")
        payload = canonical_json(o.model_dump(mode="json"))
        previous = self.db.execute("SELECT payload_json FROM research_opportunities WHERE run_id=? AND opportunity_key=?",
                                   (run_id, o.opportunity_key)).fetchone()
        if previous:
            if previous[0] != payload:
                raise ValueError("Opportunity identity already has different evidence")
            return
        available = o.prediction_recorded_at + timedelta(seconds=c.entry_latency_seconds)
        with self._transaction():
            self._clock(run_id, o.prediction_recorded_at)
            self.db.execute("INSERT INTO research_opportunities VALUES (?,?,?)", (run_id, o.opportunity_key, payload))
            for variant in VARIANTS:
                reasons = list(o.common_rejection_reasons)
                delay = (o.prediction_recorded_at - o.source_timestamp - timeframe_to_timedelta(c.timeframe)).total_seconds()
                if not 0 <= delay <= c.max_bar_delay_seconds:
                    reasons.append("prediction_outside_entry_freshness_contract")
                if not o.setup_matched:
                    reasons.append("setup_not_matched")
                if variant != "setup_only" and not o.threshold_crossed:
                    reasons.append("below_threshold")
                if variant == "setup_full_policy" and not o.full_policy_passed:
                    reasons.extend(o.policy_rejection_reasons or ("full_policy_rejected",))
                for multiplier in c.cost_multipliers:
                    self._append(run_id, {
                        "position_key": _hash([o.opportunity_key, variant, multiplier]),
                        "opportunity_key": o.opportunity_key, "model_id": o.model_id,
                        "setup_event_id": o.setup_event_id, "prediction_id": o.prediction_id,
                        "signal_decision_id": o.signal_decision_id, "source_bar_version": o.source_bar_version,
                        "source_timestamp": _stamp(o.source_timestamp),
                        "setup_type": o.setup_type, "setup_family": o.setup_family,
                        "detector_version": o.detector_version, "policy_sha256": o.policy_sha256,
                        "variant": variant, "cost_multiplier": multiplier,
                        "status": "rejected" if reasons else "pending", "reasons": reasons,
                        "available_at": available.isoformat(),
                        "expires_at": (available + timedelta(seconds=c.entry_wait_seconds)).isoformat(),
                        "side": o.side, "quantity": c.quantity,
                    }, o.prediction_recorded_at)

    def submit_audit(self, run_id, *, signal_decision_id: int, setup_event_id: int):
        """Consume immutable A5 association and A6 decisions without re-inference.

        The setup-only control uses the same model-observation availability as
        the filtered variants. It is explicitly a matched-cohort control.
        """
        from ote_live.policies.shadow import SHADOW_POLICY_CONTRACT
        row = self.db.execute(
            "SELECT sd.signal_json, sd.prediction_id, mp.prediction_json, fs.snapshot_json "
            "FROM signal_decisions sd JOIN model_predictions mp ON mp.id=sd.prediction_id "
            "JOIN feature_snapshots fs ON fs.id=mp.feature_snapshot_id WHERE sd.id=?", (signal_decision_id,)).fetchone()
        event = self.db.execute("SELECT * FROM setup_events WHERE id=?", (setup_event_id,)).fetchone()
        if row is None or event is None:
            raise ValueError("Missing persisted signal/setup evidence")
        link = self.db.execute("SELECT id FROM setup_prediction_links WHERE setup_event_id=? AND prediction_id=?",
                               (setup_event_id, row["prediction_id"])).fetchone()
        if link is None:
            raise ValueError("Missing immutable setup-prediction association")
        signal, prediction, snapshot = (json.loads(row[name]) for name in
                                       ("signal_json", "prediction_json", "snapshot_json"))
        evaluation = signal.get("shadow_evaluation") or {}
        if evaluation.get("contract_version") != SHADOW_POLICY_CONTRACT or signal["decision"] != "shadow":
            raise ValueError("Requires versioned A6 shadow evidence")
        c = self.contract(run_id)
        if any(obj.get("collection_version") != c.collection_version for obj in (signal, prediction, snapshot)) or (
                event["collection_version"] != c.collection_version or snapshot["asset"] != c.asset
                or snapshot["timeframe"] != c.timeframe):
            raise ValueError("Mismatched source/collection contract")
        metadata = snapshot.get("observation_metadata", {})
        reasons = list(metadata.get("diagnostic_reasons", []))
        if metadata.get("model_input_contract", {}).get("diagnostic_only") is not False:
            reasons.append("model_input_contract_unverified")
        if metadata.get("bar_eligibility", {}).get("eligible") is not True:
            reasons.append("bar_not_eligible")
        recorded = prediction.get("prediction_recorded_at_utc")
        if recorded is None:
            raise ValueError("Prediction recording time is required")
        source = metadata.get("source_bar", {})
        if (source.get("bar_version") != event["source_bar_version"]
                or _time(prediction["timestamp"]) != _time(event["source_timestamp_utc"])):
            raise ValueError("Mismatched setup source bar")
        delay = (_time(recorded) - _time(event["source_timestamp_utc"])
                 - timeframe_to_timedelta(c.timeframe)).total_seconds()
        if not 0 <= delay <= c.max_bar_delay_seconds:
            reasons.append("prediction_outside_entry_freshness_contract")
        if event["setup_side"] != (1 if prediction["direction"] == "long" else -1):
            reasons.append("setup_side_mismatch")
        # A later revision cannot silently stand in for the originally linked event.
        prior_revision = self.db.execute(
            "SELECT MAX(revision) FROM setup_events WHERE event_key=? AND observed_at_utc<=?",
            (event["event_key"], _stamp(_time(recorded)))).fetchone()[0]
        matched = (event["revision"] == prior_revision and bool(event["selected"])
                   and event["event_kind"] != "invalidated"
                   and setup_event_id in evaluation.get("setup_match", {}).get("matched_event_ids", []))
        self.submit(run_id, ResearchOpportunity(
            opportunity_key=_hash([event["event_key"], prediction["model_id"]]),
            collection_version=c.collection_version, model_id=prediction["model_id"],
            setup_event_id=setup_event_id, prediction_id=row["prediction_id"], signal_decision_id=signal_decision_id,
            setup_type=event["setup_type"], setup_family=event["setup_family"],
            detector_version=event["detector_version"], policy_sha256=evaluation["policy_sha256"],
            source_bar_version=event["source_bar_version"], source_timestamp=_time(event["source_timestamp_utc"]),
            setup_observed_at=_time(event["observed_at_utc"]), prediction_recorded_at=_time(recorded),
            side=event["setup_side"], setup_matched=matched,
            threshold_crossed=evaluation.get("threshold_crossed") is True,
            full_policy_passed=evaluation.get("full_policy_passed") is True,
            common_rejection_reasons=tuple(reasons),
            policy_rejection_reasons=tuple(evaluation.get("rejection_reasons", [])),
            source_evidence={"setup": json.loads(event["payload_json"]), "prediction": prediction,
                             "signal": signal, "observation_metadata": metadata},
        ))

    def process_bar(self, run_id, bar: MarketBar, *, observed_at: datetime):
        """Resolve resting hypothetical orders only from completed, fresh bars.

        An entry is at a *future* bar open relative to opportunity availability.
        Its fill becomes observable at this completed bar's observation time.
        OHLC exits expose an interval, not an invented intrabar timestamp.
        """
        c = self.contract(run_id)
        observed_at = _time(_stamp(observed_at))
        if bar.asset != c.asset or bar.timeframe != c.timeframe or not bar.bar_version:
            raise ValueError("Bar source identity is required and must match the run")
        # Revalidate model_copy inputs, including non-finite prices rejected below.
        bar = MarketBar.model_validate(bar.model_dump())
        if not all(math.isfinite(value) for value in (bar.open, bar.high, bar.low, bar.close)):
            raise ValueError("Finite OHLC prices required")
        payload = canonical_json(bar.model_dump(mode="json"))
        prior = self.db.execute("SELECT payload_json FROM research_bars WHERE run_id=? AND source_timestamp=? AND bar_version=?",
                                (run_id, _stamp(bar.timestamp), bar.bar_version)).fetchone()
        if prior:
            if prior[0] != payload:
                raise ValueError("Bar version reused with different evidence")
            return
        with self._transaction():
            cursor = self._clock(run_id, observed_at)
            previous = _time(cursor["last_source_at"]) if cursor["last_source_at"] else None
            self.db.execute("INSERT INTO research_bars VALUES (?,?,?,?,?)",
                            (run_id, _stamp(bar.timestamp), bar.bar_version, _stamp(observed_at), payload))
            if previous is not None and bar.timestamp <= previous:
                # Preserve original fills/outcomes. Mark affected resolved trades
                # separately and censor current uncertain exposure.
                for state in self.positions(run_id):
                    if state["status"] in {"pending", "open"}:
                        self._censor(run_id, state, observed_at, "out_of_order_or_revised_bar")
                    elif state["status"] == "resolved" and (
                            _time(state["source_timestamp"]) == bar.timestamp or
                            _time(state["entry_at"]) <= bar.timestamp <= _time(state["exit_bar_at"])):
                        self._append(run_id, state, observed_at, revision_tainted=True)
                return
            gap_anchor = previous
            if gap_anchor is None:
                sources = [ResearchOpportunity.model_validate_json(row[0]).source_timestamp for row in
                           self.db.execute("SELECT payload_json FROM research_opportunities WHERE run_id=?", (run_id,))]
                gap_anchor = min(sources) if sources else None
                if gap_anchor is not None and gap_anchor >= bar.timestamp:
                    gap_anchor = None
            eligibility = assess_shadow_bar_eligibility(bar, recorded_at=observed_at,
                max_age_seconds=c.max_bar_delay_seconds, previous_bar_timestamp=gap_anchor)
            self.db.execute("UPDATE research_cursors SET last_source_at=? WHERE run_id=?", (_stamp(bar.timestamp), run_id))
            # This is evidence for orders already knowable before the bar open,
            # not a new decision at its close. A session's final completed bar
            # may be observed during the scheduled closure immediately after it.
            bar_reasons = [reason for reason in eligibility.reasons if reason != "prediction_market_closed"]
            if bar_reasons:
                for state in self.positions(run_id):
                    if state["status"] in {"pending", "open"}:
                        self._censor(run_id, state, observed_at, ",".join(bar_reasons))
                return
            states = self.positions(run_id)
            opened = [state for state in states if state["status"] == "open"]
            uncertain = [state for state in states if state.get("exposure_unresolved")]
            # Allocate against exposure at the bar OPEN. Intrabar exits cannot
            # release capacity for another entry at the same historical open.
            pending = sorted((state for state in states if state["status"] == "pending"),
                             key=lambda state: (state["available_at"], state["opportunity_key"], state["position_key"]))
            for state in pending:
                if bar.timestamp > _time(state["expires_at"]):
                    self._append(run_id, state, observed_at, status="expired", reasons=["entry_wait_expired"])
                    continue
                if bar.timestamp < _time(state["available_at"]):
                    continue
                book = [p for p in opened + uncertain if p["variant"] == state["variant"]
                        and p["cost_multiplier"] == state["cost_multiplier"]
                        and (c.concurrency_scope == "all_models" or p["model_id"] == state["model_id"])]
                reason = None
                if len(book) >= c.max_positions:
                    reason = "position_limit"
                elif sum(p["quantity"] for p in book) + c.quantity > c.max_contracts:
                    reason = "contract_limit"
                elif not c.allow_opposite_positions and any(p["side"] != state["side"] for p in book):
                    reason = "opposite_exposure"
                if reason:
                    self._append(run_id, state, observed_at, status="rejected", reasons=[reason])
                    continue
                state = self._append(run_id, state, observed_at, status="open", entry_at=_stamp(bar.timestamp),
                    entry_observed_at=_stamp(observed_at), entry_price=bar.open, entry_bar_version=bar.bar_version,
                    stop_price=bar.open - state["side"] * c.stop_ticks * c.tick_size,
                    target_price=bar.open + state["side"] * c.target_ticks * c.tick_size, holding_bars=0)
                opened.append(state)
            for state in opened:
                self._advance(run_id, c, state, bar, observed_at)

    def _advance(self, run_id, c, state, bar, observed_at):
        side, stop, target = state["side"], state["stop_price"], state["target_price"]
        stop_gap = side * (bar.open - stop) <= 0
        target_gap = side * (bar.open - target) >= 0
        stop_hit = bar.low <= stop if side == 1 else bar.high >= stop
        target_hit = bar.high >= target if side == 1 else bar.low <= target
        held = state["holding_bars"] + 1
        price, reason = None, None
        if stop_gap:
            price, reason = bar.open, "stop_gap"
        elif target_gap:
            price, reason = target, "target_gap"  # conservative limit fill; no gap improvement
        elif stop_hit and target_hit and c.ambiguity == "censor":
            self._censor(run_id, state, observed_at, "ambiguous_stop_target")
            return
        elif stop_hit:
            price, reason = stop, "stop"
        elif target_hit:
            price, reason = target, "target"
        elif held >= c.timeout_bars:
            price, reason = bar.close, "timeout"
        if reason is None:
            self._append(run_id, state, observed_at, holding_bars=held,
                         last_mark_price=bar.close, last_mark_at=_stamp(bar.timestamp))
            return
        gross = side * (price - state["entry_price"]) / c.tick_size * state["quantity"]
        costs = (c.spread_ticks + 2 * c.slippage_ticks_per_side + c.commission_ticks_round_trip
                 ) * state["cost_multiplier"] * state["quantity"]
        self._append(run_id, state, observed_at, status="resolved", holding_bars=held,
            exit_reason=reason, exit_price=price, exit_bar_at=_stamp(bar.timestamp), exit_bar_version=bar.bar_version,
            exit_interval_start=_stamp(bar.timestamp),
            exit_interval_end=_stamp(bar.timestamp + timeframe_to_timedelta(bar.timeframe)),
            exit_observed_at=_stamp(observed_at), gross_pnl_ticks=gross, costs_ticks=costs,
            net_pnl_ticks=gross - costs, net_pnl_dollars=(gross - costs) * c.tick_value,
            ambiguous_bar=bool(stop_hit and target_hit and not (stop_gap or target_gap)), revision_tainted=False)

    def _censor(self, run_id, state, observed_at, reason):
        self._append(run_id, state, observed_at, status="censored", reasons=[reason],
                     exposure_unresolved=(state["status"] == "open" or
                         state["status"] == "pending" and observed_at >= _time(state["available_at"])))

    def finish(self, run_id, *, observed_at: datetime):
        """Right-censor open exposure; never fabricate a closing trade at EOF."""
        observed_at = _time(_stamp(observed_at))
        with self._transaction():
            self._clock(run_id, observed_at)
            for state in self.positions(run_id):
                if state["status"] in {"pending", "open"}:
                    self._censor(run_id, state, observed_at, "end_of_observation")
            self.db.execute("UPDATE research_cursors SET finished=1 WHERE run_id=?", (run_id,))

    def report(self, run_id):
        c = self.contract(run_id)
        states = self.positions(run_id)
        return {"run_id": run_id, "contract_sha256": _hash(c.model_dump(mode="json")),
                "mode": "diagnostic", "scored_collection_enabled": False,
                "contract": c.model_dump(mode="json"),
                "outcome_kind": "position_level_simulated_pnl",
                "open_exposure": [s for s in states if s["status"] == "open"],
                "realized_outcomes": [s for s in states if s["status"] == "resolved" and not s["revision_tainted"]],
                "revision_tainted_outcomes": [s for s in states if s["status"] == "resolved" and s["revision_tainted"]],
                "censored_outcomes": [s for s in states if s["status"] == "censored"],
                "unresolved_exposure": [s for s in states if s.get("exposure_unresolved")],
                "pending_entries": [s for s in states if s["status"] == "pending"],
                "rejections": [s for s in states if s["status"] in {"rejected", "expired"}]}
