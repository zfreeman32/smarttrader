from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage.db import SQLiteLiveDataStore


@dataclass(frozen=True)
class AuditTableSpec:
    table_name: str
    timestamp_expression: str
    restore_order: int


@dataclass(frozen=True)
class PersistedRuntimeManifest:
    runtime_manifest_id: int
    manifest_hash: str
    manifest_path: str | None
    policy_path: str | None
    created_at_utc: datetime
    manifest: LiveRuntimeManifest


@dataclass(frozen=True)
class NotificationRecord:
    notification_id: int
    signal_decision_id: int | None
    channel: str
    status: str
    dedupe_key: str | None
    payload: dict[str, Any]
    metadata: dict[str, Any]
    error_message: str | None
    sent_at_utc: datetime


@dataclass(frozen=True)
class MediaArtifactRecord:
    artifact_id: int
    signal_decision_id: int | None
    artifact_type: str
    file_path: str
    file_hash: str | None
    metadata: dict[str, Any]
    captured_at_utc: datetime


@dataclass(frozen=True)
class HealthEventRecord:
    health_event_id: int
    component: str
    event_type: str
    severity: str
    message: str
    payload: dict[str, Any]
    event_timestamp_utc: datetime
    recorded_at_utc: datetime


@dataclass(frozen=True)
class SignalAuditTrail:
    signal_decision_id: int
    feature_snapshot_id: int
    prediction_id: int
    runtime_manifest: PersistedRuntimeManifest | None
    feature_snapshot: FeatureSnapshot
    prediction: ModelPrediction
    signal: SignalDecision
    feature_snapshot_metadata: dict[str, Any] = field(default_factory=dict)
    prediction_metadata: dict[str, Any] = field(default_factory=dict)
    signal_metadata: dict[str, Any] = field(default_factory=dict)
    bars: tuple[MarketBar, ...] = ()
    notifications: tuple[NotificationRecord, ...] = ()
    media_artifacts: tuple[MediaArtifactRecord, ...] = ()


AUDIT_TABLE_SPECS: tuple[AuditTableSpec, ...] = (
    AuditTableSpec("raw_market_events", "COALESCE(event_timestamp_utc, ingested_at_utc)", 10),
    AuditTableSpec("canonical_bars", "timestamp_utc", 20),
    AuditTableSpec("ingestion_gaps", "detected_at_utc", 30),
    AuditTableSpec("heartbeat_log", "observed_at_utc", 40),
    AuditTableSpec("runtime_state", "updated_at_utc", 50),
    AuditTableSpec("runtime_manifests", "created_at_utc", 60),
    AuditTableSpec("feature_snapshots", "timestamp_utc", 70),
    AuditTableSpec("model_predictions", "timestamp_utc", 80),
    AuditTableSpec("signal_decisions", "timestamp_utc", 90),
    AuditTableSpec("notifications", "sent_at_utc", 100),
    AuditTableSpec("media_artifacts", "captured_at_utc", 110),
    AuditTableSpec("health_events", "event_timestamp_utc", 120),
)
AUDIT_TABLE_SPECS_BY_NAME = {spec.table_name: spec for spec in AUDIT_TABLE_SPECS}


class LiveAuditRepository:
    def __init__(self, store: SQLiteLiveDataStore) -> None:
        self.store = store

    def record_runtime_manifest(
        self,
        manifest: LiveRuntimeManifest,
        *,
        manifest_path: str | Path | None = None,
        policy_path: str | Path | None = None,
    ) -> PersistedRuntimeManifest:
        manifest_payload = manifest.model_dump(mode="json")
        manifest_json = _json_dumps(manifest_payload)
        manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        now = _isoformat(utc_now())
        self.store.connection.execute(
            """
            INSERT INTO runtime_manifests (
                manifest_hash, model_id, direction, backend, role, status, manifest_version,
                generated_at_utc, manifest_path, policy_path, artifact_references_json,
                live_policy_json, manifest_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(manifest_hash) DO UPDATE SET
                manifest_path = COALESCE(excluded.manifest_path, runtime_manifests.manifest_path),
                policy_path = COALESCE(excluded.policy_path, runtime_manifests.policy_path),
                artifact_references_json = excluded.artifact_references_json,
                live_policy_json = excluded.live_policy_json,
                manifest_json = excluded.manifest_json
            """,
            (
                manifest_hash,
                manifest.model_id,
                manifest.direction,
                manifest.backend,
                manifest.role,
                manifest.status,
                manifest.manifest_version,
                _isoformat(manifest.generated_at_utc),
                _coerce_optional_path(manifest_path),
                _coerce_optional_path(policy_path),
                _json_dumps(manifest_payload["artifact_references"]),
                _json_dumps(manifest_payload["live_policy"]),
                manifest_json,
                now,
            ),
        )
        self.store.connection.commit()
        row = self.store.connection.execute(
            """
            SELECT id, manifest_hash, manifest_path, policy_path, created_at_utc, manifest_json
            FROM runtime_manifests
            WHERE manifest_hash = ?
            LIMIT 1
            """,
            (manifest_hash,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to persist runtime manifest for {manifest.model_id}.")
        return _runtime_manifest_row_to_record(row)

    def get_runtime_manifest(self, runtime_manifest_id: int) -> PersistedRuntimeManifest | None:
        row = self.store.connection.execute(
            """
            SELECT id, manifest_hash, manifest_path, policy_path, created_at_utc, manifest_json
            FROM runtime_manifests
            WHERE id = ?
            LIMIT 1
            """,
            (int(runtime_manifest_id),),
        ).fetchone()
        if row is None:
            return None
        return _runtime_manifest_row_to_record(row)

    def record_feature_snapshot(
        self,
        snapshot: FeatureSnapshot,
        *,
        runtime_manifest_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        snapshot_payload = snapshot.model_dump(mode="json")
        cursor = self.store.connection.execute(
            """
            INSERT INTO feature_snapshots (
                runtime_manifest_id, asset, timeframe, direction, timestamp_utc, source_row_idx,
                feature_values_json, valid_feature_count, metadata_json, snapshot_json, recorded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                runtime_manifest_id,
                snapshot.asset,
                snapshot.timeframe,
                snapshot.direction,
                _isoformat(snapshot.timestamp),
                snapshot.source_row_idx,
                _json_dumps(snapshot_payload["feature_values"]),
                snapshot.valid_feature_count,
                _json_dumps(dict(metadata or {})),
                _json_dumps(snapshot_payload),
                _isoformat(utc_now()),
            ),
        )
        self.store.connection.commit()
        return int(cursor.lastrowid)

    def get_feature_snapshot(self, feature_snapshot_id: int) -> FeatureSnapshot | None:
        row = self.store.connection.execute(
            """
            SELECT snapshot_json
            FROM feature_snapshots
            WHERE id = ?
            LIMIT 1
            """,
            (int(feature_snapshot_id),),
        ).fetchone()
        if row is None:
            return None
        return FeatureSnapshot.model_validate(json.loads(row["snapshot_json"]))

    def record_prediction(
        self,
        prediction: ModelPrediction,
        *,
        feature_snapshot_id: int,
        runtime_manifest_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        resolved_manifest_id = runtime_manifest_id or self._linked_runtime_manifest_id(
            table_name="feature_snapshots",
            row_id=feature_snapshot_id,
        )
        prediction_payload = prediction.model_dump(mode="json")
        cursor = self.store.connection.execute(
            """
            INSERT INTO model_predictions (
                runtime_manifest_id, feature_snapshot_id, model_id, direction, backend, timestamp_utc,
                source_row_idx, regime, raw_score, calibrated_probability, threshold_applied,
                threshold_source, metadata_json, prediction_json, recorded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_manifest_id,
                int(feature_snapshot_id),
                prediction.model_id,
                prediction.direction,
                prediction.backend,
                _isoformat(prediction.timestamp),
                prediction.source_row_idx,
                prediction.regime,
                prediction.raw_score,
                prediction.calibrated_probability,
                prediction.threshold_applied,
                prediction.threshold_source,
                _json_dumps(dict(metadata or {})),
                _json_dumps(prediction_payload),
                _isoformat(utc_now()),
            ),
        )
        self.store.connection.commit()
        return int(cursor.lastrowid)

    def get_prediction(self, prediction_id: int) -> ModelPrediction | None:
        row = self.store.connection.execute(
            """
            SELECT prediction_json
            FROM model_predictions
            WHERE id = ?
            LIMIT 1
            """,
            (int(prediction_id),),
        ).fetchone()
        if row is None:
            return None
        return ModelPrediction.model_validate(json.loads(row["prediction_json"]))

    def record_signal_decision(
        self,
        signal: SignalDecision,
        *,
        prediction_id: int,
        runtime_manifest_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        resolved_manifest_id = runtime_manifest_id or self._linked_runtime_manifest_id(
            table_name="model_predictions",
            row_id=prediction_id,
        )
        signal_payload = signal.model_dump(mode="json")
        cursor = self.store.connection.execute(
            """
            INSERT INTO signal_decisions (
                runtime_manifest_id, prediction_id, model_id, direction, timestamp_utc, source_row_idx,
                decision, probability, threshold, regime, reasons_json, cooldown_bars_remaining,
                metadata_json, signal_json, recorded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_manifest_id,
                int(prediction_id),
                signal.model_id,
                signal.direction,
                _isoformat(signal.timestamp),
                signal.source_row_idx,
                signal.decision,
                signal.probability,
                signal.threshold,
                signal.regime,
                _json_dumps(list(signal.reasons)),
                signal.cooldown_bars_remaining,
                _json_dumps(dict(metadata or {})),
                _json_dumps(signal_payload),
                _isoformat(utc_now()),
            ),
        )
        self.store.connection.commit()
        return int(cursor.lastrowid)

    def get_signal_decision(self, signal_decision_id: int) -> SignalDecision | None:
        row = self.store.connection.execute(
            """
            SELECT signal_json
            FROM signal_decisions
            WHERE id = ?
            LIMIT 1
            """,
            (int(signal_decision_id),),
        ).fetchone()
        if row is None:
            return None
        return SignalDecision.model_validate(json.loads(row["signal_json"]))

    def record_notification(
        self,
        *,
        channel: str,
        status: str,
        payload: Mapping[str, Any],
        signal_decision_id: int | None = None,
        dedupe_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        error_message: str | None = None,
        sent_at_utc: datetime | None = None,
    ) -> int:
        cursor = self.store.connection.execute(
            """
            INSERT INTO notifications (
                signal_decision_id, channel, status, dedupe_key, payload_json, metadata_json,
                error_message, sent_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_decision_id,
                channel,
                status,
                dedupe_key,
                _json_dumps(dict(payload)),
                _json_dumps(dict(metadata or {})),
                error_message,
                _isoformat(sent_at_utc or utc_now()),
            ),
        )
        self.store.connection.commit()
        return int(cursor.lastrowid)

    def record_media_artifact(
        self,
        *,
        artifact_type: str,
        file_path: str | Path,
        signal_decision_id: int | None = None,
        file_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        captured_at_utc: datetime | None = None,
    ) -> int:
        cursor = self.store.connection.execute(
            """
            INSERT INTO media_artifacts (
                signal_decision_id, artifact_type, file_path, file_hash, metadata_json, captured_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                signal_decision_id,
                artifact_type,
                str(file_path),
                file_hash,
                _json_dumps(dict(metadata or {})),
                _isoformat(captured_at_utc or utc_now()),
            ),
        )
        self.store.connection.commit()
        return int(cursor.lastrowid)

    def record_health_event(
        self,
        *,
        component: str,
        event_type: str,
        severity: str,
        message: str,
        payload: Mapping[str, Any] | None = None,
        event_timestamp_utc: datetime | None = None,
    ) -> int:
        resolved_event_timestamp = event_timestamp_utc or utc_now()
        cursor = self.store.connection.execute(
            """
            INSERT INTO health_events (
                component, event_type, severity, message, event_timestamp_utc, payload_json, recorded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                component,
                event_type,
                severity,
                message,
                _isoformat(resolved_event_timestamp),
                _json_dumps(dict(payload or {})),
                _isoformat(utc_now()),
            ),
        )
        self.store.connection.commit()
        return int(cursor.lastrowid)

    def get_health_event(self, health_event_id: int) -> HealthEventRecord | None:
        row = self.store.connection.execute(
            """
            SELECT id, component, event_type, severity, message, event_timestamp_utc, payload_json, recorded_at_utc
            FROM health_events
            WHERE id = ?
            LIMIT 1
            """,
            (int(health_event_id),),
        ).fetchone()
        if row is None:
            return None
        return _health_event_row_to_record(row)

    def fetch_health_events(
        self,
        *,
        component: str | None = None,
        event_type: str | None = None,
        severity: str | None = None,
    ) -> tuple[HealthEventRecord, ...]:
        query = """
            SELECT id, component, event_type, severity, message, event_timestamp_utc, payload_json, recorded_at_utc
            FROM health_events
            WHERE 1 = 1
        """
        params: list[Any] = []
        if component is not None:
            query += " AND component = ?"
            params.append(component)
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        if severity is not None:
            query += " AND severity = ?"
            params.append(severity)
        query += " ORDER BY event_timestamp_utc ASC, id ASC"
        rows = self.store.connection.execute(query, params).fetchall()
        return tuple(_health_event_row_to_record(row) for row in rows)

    def reconstruct_signal(
        self,
        signal_decision_id: int,
        *,
        include_bars: bool = True,
        lookback_bars: int | None = None,
    ) -> SignalAuditTrail:
        row = self.store.connection.execute(
            """
            SELECT
                sd.id AS signal_decision_id,
                sd.signal_json,
                sd.metadata_json AS signal_metadata_json,
                sd.prediction_id,
                p.prediction_json,
                p.metadata_json AS prediction_metadata_json,
                p.feature_snapshot_id,
                fs.snapshot_json,
                fs.metadata_json AS feature_snapshot_metadata_json,
                COALESCE(sd.runtime_manifest_id, p.runtime_manifest_id, fs.runtime_manifest_id) AS runtime_manifest_id
            FROM signal_decisions AS sd
            INNER JOIN model_predictions AS p
                ON p.id = sd.prediction_id
            INNER JOIN feature_snapshots AS fs
                ON fs.id = p.feature_snapshot_id
            WHERE sd.id = ?
            LIMIT 1
            """,
            (int(signal_decision_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"No signal decision found for id={signal_decision_id}.")

        feature_snapshot = FeatureSnapshot.model_validate(json.loads(row["snapshot_json"]))
        prediction = ModelPrediction.model_validate(json.loads(row["prediction_json"]))
        signal = SignalDecision.model_validate(json.loads(row["signal_json"]))
        runtime_manifest = (
            self.get_runtime_manifest(int(row["runtime_manifest_id"]))
            if row["runtime_manifest_id"] is not None
            else None
        )

        bars: tuple[MarketBar, ...] = ()
        if include_bars:
            resolved_bars = self.store.fetch_bars(
                asset=feature_snapshot.asset,
                timeframe=feature_snapshot.timeframe,
                end=feature_snapshot.timestamp,
            )
            if lookback_bars is not None:
                resolved_bars = resolved_bars[-int(lookback_bars) :]
            bars = tuple(resolved_bars)

        return SignalAuditTrail(
            signal_decision_id=int(row["signal_decision_id"]),
            feature_snapshot_id=int(row["feature_snapshot_id"]),
            prediction_id=int(row["prediction_id"]),
            runtime_manifest=runtime_manifest,
            feature_snapshot=feature_snapshot,
            prediction=prediction,
            signal=signal,
            feature_snapshot_metadata=_json_loads(row["feature_snapshot_metadata_json"]),
            prediction_metadata=_json_loads(row["prediction_metadata_json"]),
            signal_metadata=_json_loads(row["signal_metadata_json"]),
            bars=bars,
            notifications=self.fetch_notifications(int(row["signal_decision_id"])),
            media_artifacts=self.fetch_media_artifacts(int(row["signal_decision_id"])),
        )

    def fetch_notifications(self, signal_decision_id: int) -> tuple[NotificationRecord, ...]:
        rows = self.store.connection.execute(
            """
            SELECT id, signal_decision_id, channel, status, dedupe_key, payload_json, metadata_json,
                   error_message, sent_at_utc
            FROM notifications
            WHERE signal_decision_id = ?
            ORDER BY sent_at_utc ASC, id ASC
            """,
            (int(signal_decision_id),),
        ).fetchall()
        return tuple(_notification_row_to_record(row) for row in rows)

    def fetch_media_artifacts(self, signal_decision_id: int) -> tuple[MediaArtifactRecord, ...]:
        rows = self.store.connection.execute(
            """
            SELECT id, signal_decision_id, artifact_type, file_path, file_hash, metadata_json, captured_at_utc
            FROM media_artifacts
            WHERE signal_decision_id = ?
            ORDER BY captured_at_utc ASC, id ASC
            """,
            (int(signal_decision_id),),
        ).fetchall()
        return tuple(_media_artifact_row_to_record(row) for row in rows)

    def _linked_runtime_manifest_id(self, *, table_name: str, row_id: int) -> int | None:
        row = self.store.connection.execute(
            f"""
            SELECT runtime_manifest_id
            FROM {table_name}
            WHERE id = ?
            LIMIT 1
            """,
            (int(row_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"No row found in {table_name} for id={row_id}.")
        return int(row["runtime_manifest_id"]) if row["runtime_manifest_id"] is not None else None


def _runtime_manifest_row_to_record(row) -> PersistedRuntimeManifest:
    return PersistedRuntimeManifest(
        runtime_manifest_id=int(row["id"]),
        manifest_hash=str(row["manifest_hash"]),
        manifest_path=row["manifest_path"],
        policy_path=row["policy_path"],
        created_at_utc=_parse_datetime(row["created_at_utc"]),
        manifest=LiveRuntimeManifest.model_validate(json.loads(row["manifest_json"])),
    )


def _notification_row_to_record(row) -> NotificationRecord:
    return NotificationRecord(
        notification_id=int(row["id"]),
        signal_decision_id=int(row["signal_decision_id"]) if row["signal_decision_id"] is not None else None,
        channel=str(row["channel"]),
        status=str(row["status"]),
        dedupe_key=row["dedupe_key"],
        payload=_json_loads(row["payload_json"]),
        metadata=_json_loads(row["metadata_json"]),
        error_message=row["error_message"],
        sent_at_utc=_parse_datetime(row["sent_at_utc"]),
    )


def _media_artifact_row_to_record(row) -> MediaArtifactRecord:
    return MediaArtifactRecord(
        artifact_id=int(row["id"]),
        signal_decision_id=int(row["signal_decision_id"]) if row["signal_decision_id"] is not None else None,
        artifact_type=str(row["artifact_type"]),
        file_path=str(row["file_path"]),
        file_hash=row["file_hash"],
        metadata=_json_loads(row["metadata_json"]),
        captured_at_utc=_parse_datetime(row["captured_at_utc"]),
    )


def _health_event_row_to_record(row) -> HealthEventRecord:
    return HealthEventRecord(
        health_event_id=int(row["id"]),
        component=str(row["component"]),
        event_type=str(row["event_type"]),
        severity=str(row["severity"]),
        message=str(row["message"]),
        payload=_json_loads(row["payload_json"]),
        event_timestamp_utc=_parse_datetime(row["event_timestamp_utc"]),
        recorded_at_utc=_parse_datetime(row["recorded_at_utc"]),
    )


def _coerce_optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload)
    return loaded if isinstance(loaded, dict) else {}


def _isoformat(value: datetime) -> str:
    return ensure_utc(value).isoformat()


def _parse_datetime(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value))
