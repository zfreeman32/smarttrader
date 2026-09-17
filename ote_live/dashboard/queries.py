from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

import pandas as pd

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from ote_live.storage.collection import LEGACY_COLLECTION, MixedCollectionError


@dataclass(frozen=True)
class DashboardHealthSummary:
    asset: str
    timeframe: str
    latest_bar_timestamp: datetime | None
    latest_prediction_timestamp: datetime | None
    latest_signal_timestamp: datetime | None
    latest_heartbeat_source: str | None
    latest_heartbeat_observed_at: datetime | None
    heartbeat_lag_seconds: float | None
    heartbeat_is_stale: bool | None
    unresolved_gap_count: int
    alerts_sent_last_24h: int
    media_artifacts_last_24h: int
    recent_health_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DashboardPerformanceSummary:
    signal_count: int
    completed_count: int
    open_count: int
    win_rate: float | None
    avg_markout_pips: float | None
    cumulative_markout_pips: float


@dataclass(frozen=True)
class DashboardFrvpPaperSignalSummary:
    event_count: int
    completed_count: int
    open_count: int
    win_rate: float | None
    avg_net_ticks: float | None
    cumulative_net_ticks: float


def fetch_recent_bars(
    store: SQLiteLiveDataStore,
    *,
    asset: str,
    timeframe: str,
    limit: int = 300,
) -> pd.DataFrame:
    rows = store.connection.execute(
        """
        SELECT
            asset,
            timeframe,
            timestamp_utc,
            open,
            high,
            low,
            close,
            volume,
            bid,
            ask,
            spread,
            source,
            symbol,
            contract_symbol,
            instrument_id,
            is_complete, bar_version, feed_type, observation_kind,
            first_observed_at_utc, last_observed_at_utc
        FROM canonical_bars
        WHERE asset = ? AND timeframe = ?
        ORDER BY timestamp_utc DESC
        LIMIT ?
        """,
        (asset, timeframe, int(limit)),
    ).fetchall()
    payload = [
        {
            "asset": row["asset"],
            "timeframe": row["timeframe"],
            "timestamp": _parse_datetime(row["timestamp_utc"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "bid": float(row["bid"]) if row["bid"] is not None else None,
            "ask": float(row["ask"]) if row["ask"] is not None else None,
            "spread": float(row["spread"]) if row["spread"] is not None else None,
            "source": row["source"],
            "symbol": row["symbol"],
            "contract_symbol": row["contract_symbol"],
            "instrument_id": int(row["instrument_id"]) if row["instrument_id"] is not None else None,
            "is_complete": bool(row["is_complete"]) if row["is_complete"] is not None else None,
            "bar_version": row["bar_version"],
            "feed_type": row["feed_type"],
            "observation_kind": row["observation_kind"],
            "first_observed_at_utc": row["first_observed_at_utc"],
            "last_observed_at_utc": row["last_observed_at_utc"],
        }
        for row in reversed(rows)
    ]
    return pd.DataFrame(payload)


def fetch_recent_signals(
    audit_repository: LiveAuditRepository,
    *,
    model_ids: Sequence[str] | None = None,
    decisions: Sequence[str] | None = None,
    runtime_manifest_hashes: Sequence[str] | None = None,
    collection_version: str | None = None,
    asset: str | None = None,
    timeframe: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    filters: list[str] = []
    params: list[Any] = []
    if asset is not None:
        filters.append("fs.asset = ?")
        params.append(asset)
    if timeframe is not None:
        filters.append("fs.timeframe = ?")
        params.append(timeframe)
    if collection_version is not None:
        filters.append("sd.collection_version = ?")
        params.append(collection_version)
    if model_ids:
        placeholders = ", ".join("?" for _ in model_ids)
        filters.append(f"sd.model_id IN ({placeholders})")
        params.extend(str(item) for item in model_ids)
    if decisions:
        placeholders = ", ".join("?" for _ in decisions)
        filters.append(f"sd.decision IN ({placeholders})")
        params.extend(str(item) for item in decisions)
    if runtime_manifest_hashes is not None:
        resolved_hashes = tuple(str(item) for item in runtime_manifest_hashes)
        if not resolved_hashes:
            filters.append("1 = 0")
        else:
            placeholders = ", ".join("?" for _ in resolved_hashes)
            filters.append(f"rm.manifest_hash IN ({placeholders})")
            params.extend(resolved_hashes)
    where_clause = ""
    if filters:
        where_clause = "AND " + " AND ".join(filters)

    rows = audit_repository.store.connection.execute(
        f"""
        SELECT
            sd.id AS signal_decision_id,
            sd.collection_version,
            sd.model_id,
            sd.direction,
            sd.timestamp_utc,
            sd.source_row_idx,
            sd.decision,
            sd.probability,
            mp.raw_score,
            sd.threshold,
            sd.regime,
            sd.reasons_json,
            COALESCE(
                sd.runtime_manifest_id,
                mp.runtime_manifest_id,
                fs.runtime_manifest_id
            ) AS runtime_manifest_id,
            rm.manifest_hash,
            fs.asset,
            fs.timeframe,
            b.open AS bar_open,
            b.high AS bar_high,
            b.low AS bar_low,
            b.close AS bar_close,
            COALESCE(n.notification_count, 0) AS notification_count,
            COALESCE(m.media_artifact_count, 0) AS media_artifact_count
        FROM signal_decisions AS sd
        LEFT JOIN model_predictions AS mp
            ON mp.id = sd.prediction_id
        LEFT JOIN feature_snapshots AS fs
            ON fs.id = mp.feature_snapshot_id
        LEFT JOIN runtime_manifests AS rm
            ON rm.id = COALESCE(
                sd.runtime_manifest_id,
                mp.runtime_manifest_id,
                fs.runtime_manifest_id
            )
        LEFT JOIN canonical_bars AS b
            ON b.asset = fs.asset
           AND b.timeframe = fs.timeframe
           AND b.timestamp_utc = sd.timestamp_utc
        LEFT JOIN (
            SELECT signal_decision_id, COUNT(*) AS notification_count
            FROM notifications
            GROUP BY signal_decision_id
        ) AS n
            ON n.signal_decision_id = sd.id
        LEFT JOIN (
            SELECT signal_decision_id, COUNT(*) AS media_artifact_count
            FROM media_artifacts
            GROUP BY signal_decision_id
        ) AS m
            ON m.signal_decision_id = sd.id
        WHERE 1 = 1
            {where_clause}
        ORDER BY sd.timestamp_utc DESC, sd.id DESC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    payload = [
        {
            "signal_decision_id": int(row["signal_decision_id"]),
            "collection_version": row["collection_version"],
            "model_id": row["model_id"],
            "direction": row["direction"],
            "timestamp": _parse_datetime(row["timestamp_utc"]),
            "source_row_idx": int(row["source_row_idx"]) if row["source_row_idx"] is not None else None,
            "decision": row["decision"],
            "probability": float(row["probability"]),
            "raw_score": float(row["raw_score"]) if row["raw_score"] is not None else None,
            "threshold": float(row["threshold"]) if row["threshold"] is not None else None,
            "regime": row["regime"],
            "reasons": _parse_json_list(row["reasons_json"]),
            "runtime_manifest_id": (
                int(row["runtime_manifest_id"])
                if row["runtime_manifest_id"] is not None
                else None
            ),
            "manifest_hash": row["manifest_hash"],
            "asset": row["asset"],
            "timeframe": row["timeframe"],
            "bar_open": float(row["bar_open"]) if row["bar_open"] is not None else None,
            "bar_high": float(row["bar_high"]) if row["bar_high"] is not None else None,
            "bar_low": float(row["bar_low"]) if row["bar_low"] is not None else None,
            "bar_close": float(row["bar_close"]) if row["bar_close"] is not None else None,
            "notification_count": int(row["notification_count"]),
            "media_artifact_count": int(row["media_artifact_count"]),
        }
        for row in reversed(rows)
    ]
    return pd.DataFrame(payload)


def fetch_confidence_history(
    audit_repository: LiveAuditRepository,
    *,
    model_id: str | None = None,
    direction: str | None = None,
    runtime_manifest_hashes: Sequence[str] | None = None,
    collection_version: str | None = None,
    asset: str | None = None,
    timeframe: str | None = None,
    limit: int = 300,
) -> pd.DataFrame:
    filters = []
    params: list[Any] = []
    if collection_version is not None:
        filters.append("mp.collection_version = ?")
        params.append(collection_version)
    if asset is not None:
        filters.append("fs.asset = ?")
        params.append(asset)
    if timeframe is not None:
        filters.append("fs.timeframe = ?")
        params.append(timeframe)
    if model_id is not None:
        filters.append("mp.model_id = ?")
        params.append(model_id)
    if direction is not None:
        filters.append("mp.direction = ?")
        params.append(direction)
    if runtime_manifest_hashes is not None:
        resolved_hashes = tuple(str(item) for item in runtime_manifest_hashes)
        if not resolved_hashes:
            filters.append("1 = 0")
        else:
            placeholders = ", ".join("?" for _ in resolved_hashes)
            filters.append(f"rm.manifest_hash IN ({placeholders})")
            params.extend(resolved_hashes)
    where_clause = " AND ".join(filters)
    if where_clause:
        where_clause = "WHERE " + where_clause

    rows = audit_repository.store.connection.execute(
        f"""
        SELECT
            mp.id AS prediction_id,
            sd.id AS signal_decision_id,
            mp.model_id,
            mp.direction,
            mp.timestamp_utc,
            mp.regime,
            mp.raw_score,
            mp.calibrated_probability,
            mp.threshold_applied,
            mp.threshold_source,
            mp.collection_version,
            mp.prediction_json,
            fs.timeframe,
            COALESCE(mp.runtime_manifest_id, sd.runtime_manifest_id) AS runtime_manifest_id,
            rm.manifest_hash,
            sd.decision
        FROM model_predictions AS mp
        LEFT JOIN feature_snapshots AS fs ON fs.id = mp.feature_snapshot_id
        LEFT JOIN signal_decisions AS sd
            ON sd.prediction_id = mp.id
        LEFT JOIN runtime_manifests AS rm
            ON rm.id = COALESCE(mp.runtime_manifest_id, sd.runtime_manifest_id)
        {where_clause}
        ORDER BY mp.timestamp_utc DESC, mp.id DESC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    payload = [
        {
            "prediction_id": int(row["prediction_id"]),
            "collection_version": row["collection_version"],
            "prediction_recorded_at_utc": json.loads(row["prediction_json"]).get("prediction_recorded_at_utc"),
            "timeframe": row["timeframe"],
            "signal_decision_id": int(row["signal_decision_id"]) if row["signal_decision_id"] is not None else None,
            "model_id": row["model_id"],
            "direction": row["direction"],
            "timestamp": _parse_datetime(row["timestamp_utc"]),
            "regime": row["regime"],
            "raw_score": float(row["raw_score"]) if row["raw_score"] is not None else None,
            "calibrated_probability": float(row["calibrated_probability"]),
            "threshold_applied": float(row["threshold_applied"]) if row["threshold_applied"] is not None else None,
            "threshold_source": row["threshold_source"],
            "runtime_manifest_id": (
                int(row["runtime_manifest_id"])
                if row["runtime_manifest_id"] is not None
                else None
            ),
            "manifest_hash": row["manifest_hash"],
            "decision": row["decision"],
        }
        for row in reversed(rows)
    ]
    return pd.DataFrame(payload)


def fetch_recent_health_events(
    audit_repository: LiveAuditRepository,
    *,
    severity: str | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    params: list[Any] = []
    where_clause = ""
    if severity is not None:
        where_clause = "WHERE severity = ?"
        params.append(severity)

    rows = audit_repository.store.connection.execute(
        f"""
        SELECT id, component, event_type, severity, message, payload_json, event_timestamp_utc, recorded_at_utc
        FROM health_events
        {where_clause}
        ORDER BY event_timestamp_utc DESC, id DESC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    payload = [
        {
            "health_event_id": int(row["id"]),
            "component": row["component"],
            "event_type": row["event_type"],
            "severity": row["severity"],
            "message": row["message"],
            "payload": _parse_json_dict(row["payload_json"]),
            "event_timestamp": _parse_datetime(row["event_timestamp_utc"]),
            "recorded_at": _parse_datetime(row["recorded_at_utc"]),
        }
        for row in reversed(rows)
    ]
    return pd.DataFrame(payload)


def build_health_summary(
    store: SQLiteLiveDataStore,
    audit_repository: LiveAuditRepository,
    *,
    asset: str,
    timeframe: str,
    now: datetime | None = None,
    runtime_manifest_hashes: Sequence[str] | None = None,
) -> DashboardHealthSummary:
    resolved_now = ensure_utc(now or utc_now())
    latest_bar_timestamp = store.get_latest_bar_timestamp(
        asset=asset,
        timeframe=timeframe,
    )
    latest_prediction_timestamp = _fetch_latest_timestamp(
        store,
        table_name="model_predictions",
        runtime_manifest_hashes=runtime_manifest_hashes,
    )
    latest_signal_timestamp = _fetch_latest_timestamp(
        store,
        table_name="signal_decisions",
        runtime_manifest_hashes=runtime_manifest_hashes,
    )
    heartbeat_row = store.connection.execute(
        """
        SELECT source, observed_at_utc, lag_seconds, is_stale
        FROM heartbeat_log
        ORDER BY observed_at_utc DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    gap_row = store.connection.execute(
        """
        SELECT COUNT(*) AS unresolved_gap_count
        FROM ingestion_gaps
        WHERE resolved_at_utc IS NULL
        """
    ).fetchone()
    cutoff = resolved_now - timedelta(hours=24)
    alerts_row = store.connection.execute(
        """
        SELECT COUNT(*) AS notification_count
        FROM notifications
        WHERE sent_at_utc >= ?
        """,
        (cutoff.isoformat(),),
    ).fetchone()
    media_row = store.connection.execute(
        """
        SELECT COUNT(*) AS artifact_count
        FROM media_artifacts
        WHERE captured_at_utc >= ?
        """,
        (cutoff.isoformat(),),
    ).fetchone()
    recent_health_df = fetch_recent_health_events(
        audit_repository,
        limit=5,
    )

    recent_health_events = tuple(
        {
            "health_event_id": int(row.health_event_id),
            "component": row.component,
            "event_type": row.event_type,
            "severity": row.severity,
            "message": row.message,
            "event_timestamp": row.event_timestamp.isoformat(),
        }
        for row in recent_health_df.itertuples(index=False)
    )
    return DashboardHealthSummary(
        asset=asset,
        timeframe=timeframe,
        latest_bar_timestamp=latest_bar_timestamp,
        latest_prediction_timestamp=latest_prediction_timestamp,
        latest_signal_timestamp=latest_signal_timestamp,
        latest_heartbeat_source=heartbeat_row["source"] if heartbeat_row is not None else None,
        latest_heartbeat_observed_at=(
            _parse_datetime(heartbeat_row["observed_at_utc"])
            if heartbeat_row is not None
            else None
        ),
        heartbeat_lag_seconds=(
            float(heartbeat_row["lag_seconds"])
            if heartbeat_row is not None
            else None
        ),
        heartbeat_is_stale=(
            bool(heartbeat_row["is_stale"])
            if heartbeat_row is not None
            else None
        ),
        unresolved_gap_count=int(gap_row["unresolved_gap_count"]) if gap_row is not None else 0,
        alerts_sent_last_24h=int(alerts_row["notification_count"]) if alerts_row is not None else 0,
        media_artifacts_last_24h=int(media_row["artifact_count"]) if media_row is not None else 0,
        recent_health_events=recent_health_events,
    )


def compute_signal_markouts(
    store: SQLiteLiveDataStore,
    audit_repository: LiveAuditRepository,
    *,
    horizon_bars: int = 3,
    limit: int = 100,
    decisions: Sequence[str] = ("emit",),
    model_ids: Sequence[str] | None = None,
    collection_version: str | None = None,
) -> pd.DataFrame:
    # Check the entire requested population before LIMIT can hide an older cohort.
    filters: list[str] = []
    params: list[Any] = []
    for column, values in (("model_id", model_ids), ("decision", decisions)):
        if values:
            filters.append(f"{column} IN ({', '.join('?' for _ in values)})")
            params.extend(values)
    query = "SELECT DISTINCT collection_version FROM signal_decisions"
    if filters:
        query += " WHERE " + " AND ".join(filters)
    versions = {row[0] for row in store.connection.execute(query, params)}
    if collection_version is None:
        if len(versions) > 1:
            raise MixedCollectionError("Mixed collection versions: choose an explicit collection_version for this report")
        collection_version = next(iter(versions), LEGACY_COLLECTION)
    signals = fetch_recent_signals(
        audit_repository,
        model_ids=model_ids,
        decisions=decisions,
        collection_version=collection_version,
        limit=limit,
    )
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "signal_decision_id",
                "timestamp",
                "direction",
                "decision",
                "entry_price",
                "exit_price",
                "markout_pips",
                "status",
            ]
        )

    rows: list[dict[str, Any]] = []
    grouped = signals.groupby(["asset", "timeframe"], dropna=False)
    for (asset, timeframe), group in grouped:
        if asset is None or timeframe is None:
            continue
        bars = store.fetch_bars(asset=asset, timeframe=timeframe)
        if not bars:
            continue
        index_by_timestamp = {bar.timestamp: idx for idx, bar in enumerate(bars)}
        for row in group.itertuples(index=False):
            bar_index = index_by_timestamp.get(row.timestamp)
            if bar_index is None:
                continue
            entry_price = float(row.bar_close) if row.bar_close is not None else float(bars[bar_index].close)
            exit_index = bar_index + int(horizon_bars)
            exit_price = None
            status = "open"
            markout_pips = None
            if exit_index < len(bars):
                exit_price = float(bars[exit_index].close)
                direction_factor = 1.0 if row.direction == "long" else -1.0
                markout_pips = (exit_price - entry_price) * 10_000.0 * direction_factor
                status = "complete"

            rows.append(
                {
                    "signal_decision_id": int(row.signal_decision_id),
                    "collection_version": row.collection_version,
                    "timestamp": row.timestamp,
                    "direction": row.direction,
                    "decision": row.decision,
                    "probability": float(row.probability),
                    "threshold": float(row.threshold) if row.threshold is not None else None,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "markout_pips": markout_pips,
                    "status": status,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_decision_id",
                "timestamp",
                "direction",
                "decision",
                "entry_price",
                "exit_price",
                "markout_pips",
                "status",
            ]
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def summarize_signal_markouts(markouts: pd.DataFrame) -> DashboardPerformanceSummary:
    if "collection_version" in markouts and markouts["collection_version"].fillna(LEGACY_COLLECTION).nunique() > 1:
        raise MixedCollectionError("Cannot summarize mixed collection versions")
    if markouts.empty:
        return DashboardPerformanceSummary(
            signal_count=0,
            completed_count=0,
            open_count=0,
            win_rate=None,
            avg_markout_pips=None,
            cumulative_markout_pips=0.0,
        )

    completed = markouts.loc[markouts["status"] == "complete"].copy()
    open_count = int((markouts["status"] != "complete").sum())
    if completed.empty:
        return DashboardPerformanceSummary(
            signal_count=int(len(markouts)),
            completed_count=0,
            open_count=open_count,
            win_rate=None,
            avg_markout_pips=None,
            cumulative_markout_pips=0.0,
        )

    markout_values = pd.to_numeric(completed["markout_pips"], errors="coerce").dropna()
    win_rate = float((markout_values > 0).mean()) if not markout_values.empty else None
    avg_markout = float(markout_values.mean()) if not markout_values.empty else None
    cumulative = float(markout_values.sum()) if not markout_values.empty else 0.0
    return DashboardPerformanceSummary(
        signal_count=int(len(markouts)),
        completed_count=int(len(completed)),
        open_count=open_count,
        win_rate=win_rate,
        avg_markout_pips=avg_markout,
        cumulative_markout_pips=cumulative,
    )


def fetch_frvp_paper_signal_markouts(
    store: SQLiteLiveDataStore,
    *,
    bundle_id: str,
    runtime_manifest_hashes: Sequence[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read the controlled FRVP 120-bar, friction-adjusted ES markout ledger."""

    filters = [
        "bundle_id = ?",
        "asset = 'ES'",
        "timeframe = '5m'",
        "holding_period_bars = 120",
    ]
    params: list[Any] = [str(bundle_id)]
    if runtime_manifest_hashes is not None:
        resolved_hashes = tuple(str(item) for item in runtime_manifest_hashes)
        if not resolved_hashes:
            filters.append("1 = 0")
        else:
            placeholders = ", ".join("?" for _ in resolved_hashes)
            filters.append(f"manifest_hash IN ({placeholders})")
            params.extend(resolved_hashes)
    where_clause = " AND ".join(filters)
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(int(limit))
    rows = store.connection.execute(
        f"""
        SELECT
            id,
            signal_decision_id,
            bundle_id,
            runtime_manifest_id,
            manifest_hash,
            model_id,
            source_timestamp_utc,
            entry_price,
            holding_period_bars,
            total_cost_ticks,
            lifecycle_status,
            exit_timestamp_utc,
            exit_price,
            gross_pnl_ticks,
            net_pnl_ticks,
            outcome
        FROM frvp_paper_signal_events
        WHERE {where_clause}
        ORDER BY source_timestamp_utc DESC, id DESC
        {limit_clause}
        """,
        params,
    ).fetchall()
    payload = [
        {
            "event_id": int(row["id"]),
            "signal_decision_id": int(row["signal_decision_id"]),
            "bundle_id": row["bundle_id"],
            "runtime_manifest_id": int(row["runtime_manifest_id"]),
            "manifest_hash": row["manifest_hash"],
            "model_id": row["model_id"],
            "timestamp": _parse_datetime(row["source_timestamp_utc"]),
            "entry_price": float(row["entry_price"]),
            "holding_period_bars": int(row["holding_period_bars"]),
            "total_cost_ticks": float(row["total_cost_ticks"]),
            "status": row["lifecycle_status"],
            "exit_timestamp": (
                _parse_datetime(row["exit_timestamp_utc"])
                if row["exit_timestamp_utc"] is not None
                else None
            ),
            "exit_price": (
                float(row["exit_price"])
                if row["exit_price"] is not None
                else None
            ),
            "gross_markout_ticks": (
                float(row["gross_pnl_ticks"])
                if row["gross_pnl_ticks"] is not None
                else None
            ),
            "net_markout_ticks": (
                float(row["net_pnl_ticks"])
                if row["net_pnl_ticks"] is not None
                else None
            ),
            "outcome": row["outcome"],
        }
        for row in reversed(rows)
    ]
    return pd.DataFrame(payload)


def summarize_frvp_paper_signal_markouts(
    markouts: pd.DataFrame,
) -> DashboardFrvpPaperSignalSummary:
    if markouts.empty:
        return DashboardFrvpPaperSignalSummary(
            event_count=0,
            completed_count=0,
            open_count=0,
            win_rate=None,
            avg_net_ticks=None,
            cumulative_net_ticks=0.0,
        )

    completed = markouts.loc[markouts["status"] == "settled"].copy()
    open_count = int((markouts["status"] != "settled").sum())
    net_ticks = pd.to_numeric(
        completed["net_markout_ticks"], errors="coerce"
    ).dropna()
    return DashboardFrvpPaperSignalSummary(
        event_count=int(len(markouts)),
        completed_count=int(len(completed)),
        open_count=open_count,
        win_rate=float((net_ticks > 0.0).mean()) if not net_ticks.empty else None,
        avg_net_ticks=float(net_ticks.mean()) if not net_ticks.empty else None,
        cumulative_net_ticks=float(net_ticks.sum()) if not net_ticks.empty else 0.0,
    )


def _fetch_latest_timestamp(
    store: SQLiteLiveDataStore,
    *,
    table_name: str,
    runtime_manifest_hashes: Sequence[str] | None = None,
) -> datetime | None:
    if table_name not in {"model_predictions", "signal_decisions"}:
        raise ValueError(f"Unsupported timestamp table: {table_name}")
    params: list[Any] = []
    where_clause = ""
    if runtime_manifest_hashes is not None:
        resolved_hashes = tuple(str(item) for item in runtime_manifest_hashes)
        if not resolved_hashes:
            where_clause = "WHERE 1 = 0"
        else:
            placeholders = ", ".join("?" for _ in resolved_hashes)
            where_clause = f"WHERE rm.manifest_hash IN ({placeholders})"
            params.extend(resolved_hashes)
    row = store.connection.execute(
        f"""
        SELECT source.timestamp_utc
        FROM {table_name} AS source
        LEFT JOIN runtime_manifests AS rm
            ON rm.id = source.runtime_manifest_id
        {where_clause}
        ORDER BY source.timestamp_utc DESC, source.id DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        return None
    return _parse_datetime(row["timestamp_utc"])


def _parse_json_list(payload: str | None) -> list[Any]:
    if not payload:
        return []
    loaded = json.loads(payload)
    return loaded if isinstance(loaded, list) else []


def _parse_json_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload)
    return loaded if isinstance(loaded, dict) else {}


def _parse_datetime(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value))
