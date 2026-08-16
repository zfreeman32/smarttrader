from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.signal import SignalDecision
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.policies.decision_engine import PersistedAuditRecord
from ote_live.storage.repositories import LiveAuditRepository


ICT_PAPER_SIGNAL_BUNDLE_ID = "ict_es_paper_signal_20260813"
ICT_PAPER_SIGNAL_MODEL_ID = "ict_long_meta_xgb_v1"
ICT_PAPER_SIGNAL_REGISTRY_PATH = "models/ict_es_paper_signal_registry_20260813.json"
ICT_PAPER_SIGNAL_HOLDING_BARS = 20
ICT_PAPER_SIGNAL_TICK_SIZE = 0.25
ICT_PAPER_SIGNAL_TICK_VALUE = 12.5
ICT_PAPER_SIGNAL_SLIPPAGE_TICKS = 0.25
ICT_PAPER_SIGNAL_COMMISSION_TICKS = 0.40
ICT_PAPER_SIGNAL_SPREAD_TICKS = {
    "overlap": 1.0,
    "london": 1.0,
    "new_york": 1.0,
    "asia": 1.5,
    "off_hours": 2.0,
}


@dataclass(frozen=True)
class IctPaperSignalEvent:
    event_id: int
    event_key: str
    lifecycle_status: str
    source_timestamp: datetime
    source_row_idx: int
    entry_price: float
    total_cost_ticks: float
    exit_timestamp: datetime | None
    exit_price: float | None
    gross_pnl_ticks: float | None
    net_pnl_ticks: float | None
    outcome: str | None


class IctPaperSignalLedgerRepository:
    """Persistent independent-event markouts; this API has no broker concepts."""

    def __init__(self, audit_repository: LiveAuditRepository) -> None:
        self.audit_repository = audit_repository
        self.store = audit_repository.store

    def open_event(
        self,
        *,
        manifest: LiveRuntimeManifest,
        audit_record: PersistedAuditRecord,
        signal: SignalDecision,
        bar: MarketBar,
        session_regime: str | None,
    ) -> IctPaperSignalEvent:
        _validate_contract_manifest(manifest)
        persisted_manifest = self.audit_repository.get_runtime_manifest(
            audit_record.runtime_manifest_id
        )
        if persisted_manifest is None:
            raise KeyError(f"Missing runtime manifest id={audit_record.runtime_manifest_id}.")
        if signal.decision != "emit" or signal.model_id != ICT_PAPER_SIGNAL_MODEL_ID:
            raise ValueError("Only emitted active ICT long-meta signals belong in this ledger.")
        if signal.source_row_idx is None or signal.threshold is None:
            raise ValueError("Paper-signal identity requires source_row_idx and threshold.")

        resolved_session = _session_regime(session_regime)
        spread_ticks = ICT_PAPER_SIGNAL_SPREAD_TICKS[resolved_session]
        total_cost_ticks = (
            spread_ticks
            + spread_ticks
            + ICT_PAPER_SIGNAL_SLIPPAGE_TICKS
            + ICT_PAPER_SIGNAL_COMMISSION_TICKS
        )
        identity = "|".join(
            (
                ICT_PAPER_SIGNAL_BUNDLE_ID,
                persisted_manifest.manifest_hash,
                signal.model_id,
                ensure_utc(signal.timestamp).isoformat(),
                str(int(signal.source_row_idx)),
            )
        )
        event_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        now = ensure_utc(utc_now()).isoformat()
        self.store.connection.execute(
            """
            INSERT INTO ict_paper_signal_events (
                event_key, bundle_id, runtime_manifest_id, manifest_hash, signal_decision_id,
                model_id, direction, asset, timeframe, source_timestamp_utc, source_row_idx,
                probability, threshold, composite_regime, session_regime,
                entry_timing, entry_timestamp_utc, entry_source_row_idx, entry_price,
                holding_period_bars, exit_timing, stop_price, target_price, stop_target_semantics,
                tick_size, tick_value, spread_cost_mode, entry_spread_ticks, exit_spread_ticks,
                fixed_slippage_ticks, commission_ticks, total_cost_ticks, lifecycle_status,
                metadata_json, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'signal_close', ?, ?, ?,
                20, 'close_after_20_completed_bars', NULL, NULL, 'not_applicable', ?, ?,
                'session_schedule', ?, ?, ?, ?, ?, 'open', ?, ?, ?)
            ON CONFLICT(event_key) DO NOTHING
            """,
            (
                event_key,
                ICT_PAPER_SIGNAL_BUNDLE_ID,
                audit_record.runtime_manifest_id,
                persisted_manifest.manifest_hash,
                audit_record.signal_decision_id,
                signal.model_id,
                signal.direction,
                bar.asset,
                bar.timeframe,
                ensure_utc(signal.timestamp).isoformat(),
                int(signal.source_row_idx),
                float(signal.probability),
                float(signal.threshold),
                signal.regime,
                resolved_session,
                ensure_utc(bar.timestamp).isoformat(),
                int(signal.source_row_idx),
                float(bar.close),
                ICT_PAPER_SIGNAL_TICK_SIZE,
                ICT_PAPER_SIGNAL_TICK_VALUE,
                spread_ticks,
                spread_ticks,
                ICT_PAPER_SIGNAL_SLIPPAGE_TICKS,
                ICT_PAPER_SIGNAL_COMMISSION_TICKS,
                total_cost_ticks,
                json.dumps(
                    {
                        "event_semantics": "independent_overlapping_markout",
                        "source": "accepted_20260730_walk_forward_backtest",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                now,
                now,
            ),
        )
        self.store.connection.commit()
        row = self.store.connection.execute(
            "SELECT * FROM ict_paper_signal_events WHERE event_key = ?", (event_key,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to persist ICT paper-signal event.")
        return _row_to_event(row)

    def settle_completed_events(self, through_bar: MarketBar) -> tuple[IctPaperSignalEvent, ...]:
        if through_bar.asset != "ES" or through_bar.timeframe != "5m":
            return ()
        pending = self.store.connection.execute(
            """
            SELECT * FROM ict_paper_signal_events
            WHERE lifecycle_status = 'open' AND asset = ? AND timeframe = ?
              AND entry_timestamp_utc < ?
            ORDER BY entry_timestamp_utc, id
            """,
            (through_bar.asset, through_bar.timeframe, ensure_utc(through_bar.timestamp).isoformat()),
        ).fetchall()
        settled: list[IctPaperSignalEvent] = []
        for event in pending:
            exit_row = self.store.connection.execute(
                """
                SELECT timestamp_utc, close FROM canonical_bars
                WHERE asset = ? AND timeframe = ? AND timestamp_utc > ? AND timestamp_utc <= ?
                ORDER BY timestamp_utc ASC LIMIT 1 OFFSET 19
                """,
                (
                    event["asset"],
                    event["timeframe"],
                    event["entry_timestamp_utc"],
                    ensure_utc(through_bar.timestamp).isoformat(),
                ),
            ).fetchone()
            if exit_row is None:
                continue
            exit_price = float(exit_row["close"])
            gross_ticks = (exit_price - float(event["entry_price"])) / float(event["tick_size"])
            net_ticks = gross_ticks - float(event["total_cost_ticks"])
            outcome = "win" if net_ticks > 0.0 else "loss" if net_ticks < 0.0 else "flat"
            now = ensure_utc(utc_now()).isoformat()
            self.store.connection.execute(
                """
                UPDATE ict_paper_signal_events SET
                    lifecycle_status = 'settled', exit_timestamp_utc = ?, exit_source_row_idx = ?,
                    exit_price = ?, gross_pnl_ticks = ?, net_pnl_ticks = ?,
                    gross_pnl_dollars = ?, net_pnl_dollars = ?, outcome = ?,
                    updated_at_utc = ?, settled_at_utc = ?
                WHERE id = ? AND lifecycle_status = 'open'
                """,
                (
                    str(exit_row["timestamp_utc"]),
                    int(event["entry_source_row_idx"]) + ICT_PAPER_SIGNAL_HOLDING_BARS,
                    exit_price,
                    gross_ticks,
                    net_ticks,
                    gross_ticks * float(event["tick_value"]),
                    net_ticks * float(event["tick_value"]),
                    outcome,
                    now,
                    now,
                    int(event["id"]),
                ),
            )
            current = self.store.connection.execute(
                "SELECT * FROM ict_paper_signal_events WHERE id = ?", (int(event["id"]),)
            ).fetchone()
            settled.append(_row_to_event(current))
        self.store.connection.commit()
        return tuple(settled)

    def fetch_events(self, *, lifecycle_status: str | None = None) -> tuple[IctPaperSignalEvent, ...]:
        query = "SELECT * FROM ict_paper_signal_events"
        params: tuple[Any, ...] = ()
        if lifecycle_status is not None:
            query += " WHERE lifecycle_status = ?"
            params = (lifecycle_status,)
        query += " ORDER BY source_timestamp_utc, id"
        return tuple(_row_to_event(row) for row in self.store.connection.execute(query, params))


def supports_ict_paper_signal_manifest(manifest: LiveRuntimeManifest) -> bool:
    return (
        manifest.model_id == ICT_PAPER_SIGNAL_MODEL_ID
        and manifest.asset == "ES"
        and manifest.timeframe == "5m"
        and manifest.direction == "long"
        and manifest.status == "active"
        and manifest.registry_path.replace("\\", "/") == ICT_PAPER_SIGNAL_REGISTRY_PATH
    )


def _validate_contract_manifest(manifest: LiveRuntimeManifest) -> None:
    if not supports_ict_paper_signal_manifest(manifest):
        raise ValueError("Manifest is not the sole active ICT paper-signal meta contract.")
    horizon = manifest.context_requirements.label_horizon_assumptions.label_max_holding_bars
    costs = manifest.live_policy.cost_assumptions
    schedule = {str(key): float(value) for key, value in costs.session_spread_pips.items()}
    if horizon != ICT_PAPER_SIGNAL_HOLDING_BARS:
        raise ValueError("ICT paper-signal manifest no longer specifies a 20-bar horizon.")
    if schedule != ICT_PAPER_SIGNAL_SPREAD_TICKS:
        raise ValueError("ICT paper-signal spread schedule differs from accepted economics.")
    if costs.fixed_slippage_pips_per_trade != ICT_PAPER_SIGNAL_SLIPPAGE_TICKS:
        raise ValueError("ICT paper-signal slippage differs from accepted economics.")
    if costs.commission_pips_per_trade != ICT_PAPER_SIGNAL_COMMISSION_TICKS:
        raise ValueError("ICT paper-signal commission differs from accepted economics.")


def _session_regime(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ICT_PAPER_SIGNAL_SPREAD_TICKS else "off_hours"


def _row_to_event(row: Mapping[str, Any]) -> IctPaperSignalEvent:
    return IctPaperSignalEvent(
        event_id=int(row["id"]),
        event_key=str(row["event_key"]),
        lifecycle_status=str(row["lifecycle_status"]),
        source_timestamp=ensure_utc(datetime.fromisoformat(str(row["source_timestamp_utc"]))),
        source_row_idx=int(row["source_row_idx"]),
        entry_price=float(row["entry_price"]),
        total_cost_ticks=float(row["total_cost_ticks"]),
        exit_timestamp=(
            ensure_utc(datetime.fromisoformat(str(row["exit_timestamp_utc"])))
            if row["exit_timestamp_utc"] is not None
            else None
        ),
        exit_price=float(row["exit_price"]) if row["exit_price"] is not None else None,
        gross_pnl_ticks=(
            float(row["gross_pnl_ticks"]) if row["gross_pnl_ticks"] is not None else None
        ),
        net_pnl_ticks=float(row["net_pnl_ticks"]) if row["net_pnl_ticks"] is not None else None,
        outcome=str(row["outcome"]) if row["outcome"] is not None else None,
    )
