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
from ote_live.models.loaders import (
    FROZEN_FRVP_ACTIVE_MANIFEST_SHA256,
    FROZEN_FRVP_ACTIVE_MODEL_ID,
    runtime_manifest_sha256,
    validate_frozen_frvp_active_manifest,
)
from ote_live.policies.decision_engine import PersistedAuditRecord
from ote_live.storage.repositories import LiveAuditRepository


FRVP_PAPER_SIGNAL_BUNDLE_ID = "frvp_es_paper_signal_20260816"
FRVP_PAPER_SIGNAL_REGISTRY_PATH = "models/frvp_es_paper_signal_registry_20260816.json"
FRVP_PAPER_SIGNAL_MANIFEST_HASHES = {
    FROZEN_FRVP_ACTIVE_MODEL_ID: FROZEN_FRVP_ACTIVE_MANIFEST_SHA256,
}
FRVP_PAPER_SIGNAL_MODEL_IDS = frozenset(FRVP_PAPER_SIGNAL_MANIFEST_HASHES)
FRVP_PAPER_SIGNAL_HOLDING_BARS = 120
FRVP_PAPER_SIGNAL_TICK_SIZE = 0.25
FRVP_PAPER_SIGNAL_TICK_VALUE = 12.5
FRVP_PAPER_SIGNAL_SLIPPAGE_TICKS = 0.25
FRVP_PAPER_SIGNAL_COMMISSION_TICKS = 0.40
FRVP_PAPER_SIGNAL_SPREAD_TICKS = {
    "overlap": 1.0,
    "london": 1.0,
    "new_york": 1.0,
    "asia": 1.5,
    "off_hours": 2.0,
}


@dataclass(frozen=True)
class FrvpPaperSignalEvent:
    event_id: int
    event_key: str
    model_id: str
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


class FrvpPaperSignalLedgerRepository:
    """Persistent FRVP event markouts, intentionally independent of broker positions."""

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
    ) -> FrvpPaperSignalEvent:
        _validate_contract_manifest(manifest)
        persisted_manifest = self.audit_repository.get_runtime_manifest(
            audit_record.runtime_manifest_id
        )
        if persisted_manifest is None:
            raise KeyError(f"Missing runtime manifest id={audit_record.runtime_manifest_id}.")
        expected_manifest_hash = FRVP_PAPER_SIGNAL_MANIFEST_HASHES[manifest.model_id]
        if (
            persisted_manifest.manifest_hash != expected_manifest_hash
            or _manifest_hash(persisted_manifest.manifest) != expected_manifest_hash
        ):
            raise ValueError("Persisted runtime manifest is not the frozen FRVP reversal contract.")
        if signal.decision != "emit" or signal.model_id not in FRVP_PAPER_SIGNAL_MODEL_IDS:
            raise ValueError("Only emitted active FRVP paper-signal models belong in this ledger.")
        if signal.model_id != manifest.model_id:
            raise ValueError("Signal model identity does not match its runtime manifest.")
        if signal.source_row_idx is None or signal.threshold is None:
            raise ValueError("Paper-signal identity requires source_row_idx and threshold.")
        if bar.asset != manifest.asset or bar.timeframe != manifest.timeframe:
            raise ValueError("Entry bar asset/timeframe does not match its runtime manifest.")
        if ensure_utc(signal.timestamp) != ensure_utc(bar.timestamp):
            raise ValueError("Signal and signal-close entry timestamps must match.")

        resolved_session = _session_regime(session_regime)
        spread_ticks = FRVP_PAPER_SIGNAL_SPREAD_TICKS[resolved_session]
        total_cost_ticks = (
            spread_ticks
            + spread_ticks
            + FRVP_PAPER_SIGNAL_SLIPPAGE_TICKS
            + FRVP_PAPER_SIGNAL_COMMISSION_TICKS
        )
        identity = "|".join(
            (
                FRVP_PAPER_SIGNAL_BUNDLE_ID,
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
            INSERT INTO frvp_paper_signal_events (
                event_key, bundle_id, runtime_manifest_id, manifest_hash, signal_decision_id,
                model_id, direction, asset, timeframe, source_timestamp_utc, source_row_idx,
                probability, threshold, composite_regime, session_regime,
                entry_timing, entry_timestamp_utc, entry_source_row_idx, entry_price,
                holding_period_bars, exit_timing, stop_price, target_price, stop_target_semantics,
                tick_size, tick_value, spread_cost_mode, entry_spread_ticks, exit_spread_ticks,
                fixed_slippage_ticks, commission_ticks, total_cost_ticks, lifecycle_status,
                metadata_json, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'signal_close', ?, ?, ?,
                120, 'close_after_120_completed_bars', NULL, NULL, 'not_applicable', ?, ?,
                'session_schedule', ?, ?, ?, ?, ?, 'open', ?, ?, ?)
            ON CONFLICT(event_key) DO NOTHING
            """,
            (
                event_key,
                FRVP_PAPER_SIGNAL_BUNDLE_ID,
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
                FRVP_PAPER_SIGNAL_TICK_SIZE,
                FRVP_PAPER_SIGNAL_TICK_VALUE,
                spread_ticks,
                spread_ticks,
                FRVP_PAPER_SIGNAL_SLIPPAGE_TICKS,
                FRVP_PAPER_SIGNAL_COMMISSION_TICKS,
                total_cost_ticks,
                json.dumps(
                    {
                        "event_semantics": "independent_overlapping_markout",
                        "source": "accepted_frvp_walk_forward_backtest",
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
            "SELECT * FROM frvp_paper_signal_events WHERE event_key = ?", (event_key,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to persist FRVP paper-signal event.")
        return _row_to_event(row)

    def settle_completed_events(self, through_bar: MarketBar) -> tuple[FrvpPaperSignalEvent, ...]:
        if through_bar.asset != "ES" or through_bar.timeframe != "5m":
            return ()
        pending = self.store.connection.execute(
            """
            SELECT * FROM frvp_paper_signal_events
            WHERE lifecycle_status = 'open' AND asset = ? AND timeframe = ?
              AND entry_timestamp_utc < ?
            ORDER BY entry_timestamp_utc, id
            """,
            (through_bar.asset, through_bar.timeframe, ensure_utc(through_bar.timestamp).isoformat()),
        ).fetchall()
        settled: list[FrvpPaperSignalEvent] = []
        for event in pending:
            exit_row = self.store.connection.execute(
                """
                SELECT timestamp_utc, close FROM canonical_bars
                WHERE asset = ? AND timeframe = ? AND timestamp_utc > ? AND timestamp_utc <= ?
                ORDER BY timestamp_utc ASC LIMIT 1 OFFSET 119
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
                UPDATE frvp_paper_signal_events SET
                    lifecycle_status = 'settled', exit_timestamp_utc = ?, exit_source_row_idx = ?,
                    exit_price = ?, gross_pnl_ticks = ?, net_pnl_ticks = ?,
                    gross_pnl_dollars = ?, net_pnl_dollars = ?, outcome = ?,
                    updated_at_utc = ?, settled_at_utc = ?
                WHERE id = ? AND lifecycle_status = 'open'
                """,
                (
                    str(exit_row["timestamp_utc"]),
                    int(event["entry_source_row_idx"]) + FRVP_PAPER_SIGNAL_HOLDING_BARS,
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
                "SELECT * FROM frvp_paper_signal_events WHERE id = ?", (int(event["id"]),)
            ).fetchone()
            settled.append(_row_to_event(current))
        self.store.connection.commit()
        return tuple(settled)

    def reconcile_missing_events(self) -> tuple[FrvpPaperSignalEvent, ...]:
        """Recover eligible emitted decisions whose ledger insert did not complete."""

        rows = self.store.connection.execute(
            """
            SELECT
                sd.id AS signal_decision_id,
                sd.runtime_manifest_id,
                sd.prediction_id,
                sd.signal_json,
                sd.metadata_json,
                p.feature_snapshot_id
            FROM signal_decisions AS sd
            JOIN model_predictions AS p ON p.id = sd.prediction_id
            LEFT JOIN frvp_paper_signal_events AS event
                ON event.signal_decision_id = sd.id
            WHERE sd.model_id = ? AND sd.decision = 'emit' AND event.id IS NULL
            ORDER BY sd.timestamp_utc, sd.id
            """,
            (next(iter(FRVP_PAPER_SIGNAL_MODEL_IDS)),),
        ).fetchall()
        recovered: list[FrvpPaperSignalEvent] = []
        for row in rows:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
            if not isinstance(metadata, dict) or metadata.get("paper_signal_event_eligible") is not True:
                continue
            runtime_manifest_id = row["runtime_manifest_id"]
            if runtime_manifest_id is None:
                raise RuntimeError("Eligible FRVP emit is missing its runtime manifest identity.")
            persisted_manifest = self.audit_repository.get_runtime_manifest(
                int(runtime_manifest_id)
            )
            if persisted_manifest is None:
                raise RuntimeError("Eligible FRVP emit references a missing runtime manifest.")
            manifest = persisted_manifest.manifest
            _validate_contract_manifest(manifest)
            signal = SignalDecision.model_validate(json.loads(str(row["signal_json"])))
            entry_bars = self.store.fetch_bars(
                asset=manifest.asset,
                timeframe=manifest.timeframe,
                start=signal.timestamp,
                end=signal.timestamp,
            )
            if len(entry_bars) != 1:
                raise RuntimeError(
                    "Eligible FRVP emit does not resolve to exactly one canonical entry bar."
                )
            policy_context = metadata.get("policy_context")
            policy_context = policy_context if isinstance(policy_context, dict) else {}
            recovered.append(
                self.open_event(
                    manifest=manifest,
                    audit_record=PersistedAuditRecord(
                        runtime_manifest_id=int(runtime_manifest_id),
                        feature_snapshot_id=int(row["feature_snapshot_id"]),
                        prediction_id=int(row["prediction_id"]),
                        signal_decision_id=int(row["signal_decision_id"]),
                    ),
                    signal=signal,
                    bar=entry_bars[0],
                    session_regime=(
                        str(policy_context["session_regime"])
                        if policy_context.get("session_regime") is not None
                        else None
                    ),
                )
            )
        return tuple(recovered)

    def latest_emitted_source_row_idx(
        self,
        manifest: LiveRuntimeManifest,
    ) -> int | None:
        """Return the latest canonical emit for this exact frozen manifest."""

        _validate_contract_manifest(manifest)
        expected_hash = FRVP_PAPER_SIGNAL_MANIFEST_HASHES[manifest.model_id]
        row = self.store.connection.execute(
            """
            SELECT
                sd.id AS signal_decision_id,
                sd.source_row_idx,
                sd.signal_json,
                sd.runtime_manifest_id
            FROM signal_decisions AS sd
            JOIN runtime_manifests AS rm ON rm.id = sd.runtime_manifest_id
            WHERE sd.model_id = ? AND sd.decision = 'emit'
              AND sd.source_row_idx IS NOT NULL AND rm.manifest_hash = ?
            ORDER BY sd.source_row_idx DESC, sd.id DESC
            LIMIT 1
            """,
            (manifest.model_id, expected_hash),
        ).fetchone()
        if row is None:
            return None
        persisted_manifest = self.audit_repository.get_runtime_manifest(
            int(row["runtime_manifest_id"])
        )
        if (
            persisted_manifest is None
            or persisted_manifest.manifest_hash != expected_hash
            or runtime_manifest_sha256(persisted_manifest.manifest) != expected_hash
        ):
            raise ValueError("Persisted cooldown emit has a non-frozen manifest identity.")
        signal = SignalDecision.model_validate(json.loads(str(row["signal_json"])))
        source_row_idx = int(row["source_row_idx"])
        if (
            signal.model_id != manifest.model_id
            or signal.decision != "emit"
            or signal.source_row_idx != source_row_idx
        ):
            raise ValueError("Persisted cooldown emit is not a canonical signal decision.")
        return source_row_idx

    def fetch_events(
        self,
        *,
        lifecycle_status: str | None = None,
        model_id: str | None = None,
    ) -> tuple[FrvpPaperSignalEvent, ...]:
        query = "SELECT * FROM frvp_paper_signal_events WHERE 1 = 1"
        params: list[Any] = []
        if lifecycle_status is not None:
            query += " AND lifecycle_status = ?"
            params.append(lifecycle_status)
        if model_id is not None:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY source_timestamp_utc, id"
        return tuple(_row_to_event(row) for row in self.store.connection.execute(query, params))


def supports_frvp_paper_signal_manifest(manifest: LiveRuntimeManifest) -> bool:
    return (
        is_frvp_paper_signal_manifest_identity(manifest)
        and manifest.status == "active"
        and _manifest_hash(manifest)
        == FRVP_PAPER_SIGNAL_MANIFEST_HASHES.get(manifest.model_id)
    )


def is_frvp_paper_signal_manifest_identity(manifest: LiveRuntimeManifest) -> bool:
    """Identify the controlled model before applying integrity/hash checks.

    This deliberately does not inspect the manifest hash.  Callers use it to
    fail closed when a controlled manifest has been altered instead of silently
    treating the altered model as an unrelated FRVP runtime.
    """

    return (
        manifest.model_id in FRVP_PAPER_SIGNAL_MODEL_IDS
        and manifest.asset == "ES"
        and manifest.timeframe == "5m"
        and manifest.direction == "long"
        and manifest.registry_path.replace("\\", "/") == FRVP_PAPER_SIGNAL_REGISTRY_PATH
    )


def validate_frvp_paper_signal_manifest(manifest: LiveRuntimeManifest) -> None:
    _validate_contract_manifest(manifest)


def _validate_contract_manifest(manifest: LiveRuntimeManifest) -> None:
    if not supports_frvp_paper_signal_manifest(manifest):
        raise ValueError("Manifest is not an active FRVP paper-signal contract.")
    validate_frozen_frvp_active_manifest(manifest)
    horizon = manifest.context_requirements.label_horizon_assumptions.label_max_holding_bars
    costs = manifest.live_policy.cost_assumptions
    schedule = {str(key): float(value) for key, value in costs.session_spread_pips.items()}
    if horizon != FRVP_PAPER_SIGNAL_HOLDING_BARS:
        raise ValueError("FRVP paper-signal manifest no longer specifies a 120-bar horizon.")
    if schedule != FRVP_PAPER_SIGNAL_SPREAD_TICKS:
        raise ValueError("FRVP paper-signal spread schedule differs from accepted economics.")
    if costs.fixed_slippage_pips_per_trade != FRVP_PAPER_SIGNAL_SLIPPAGE_TICKS:
        raise ValueError("FRVP paper-signal slippage differs from accepted economics.")
    if costs.commission_pips_per_trade != FRVP_PAPER_SIGNAL_COMMISSION_TICKS:
        raise ValueError("FRVP paper-signal commission differs from accepted economics.")


def _session_regime(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in FRVP_PAPER_SIGNAL_SPREAD_TICKS else "off_hours"


def _manifest_hash(manifest: LiveRuntimeManifest) -> str:
    return runtime_manifest_sha256(manifest)


def _row_to_event(row: Mapping[str, Any]) -> FrvpPaperSignalEvent:
    return FrvpPaperSignalEvent(
        event_id=int(row["id"]),
        event_key=str(row["event_key"]),
        model_id=str(row["model_id"]),
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
