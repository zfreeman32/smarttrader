from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

import pandas as pd

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore


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
            instrument_id
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
        }
        for row in reversed(rows)
    ]
    return pd.DataFrame(payload)


def fetch_recent_signals(
    audit_repository: LiveAuditRepository,
    *,
    model_ids: Sequence[str] | None = None,
    decisions: Sequence[str] | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    filters: list[str] = []
    params: list[Any] = []
    if model_ids:
        placeholders = ", ".join("?" for _ in model_ids)
        filters.append(f"sd.model_id IN ({placeholders})")
        params.extend(str(item) for item in model_ids)
    if decisions:
        placeholders = ", ".join("?" for _ in decisions)
        filters.append(f"sd.decision IN ({placeholders})")
        params.extend(str(item) for item in decisions)
    where_clause = ""
    if filters:
        where_clause = "AND " + " AND ".join(filters)

    rows = audit_repository.store.connection.execute(
        f"""
        SELECT
            sd.id AS signal_decision_id,
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
    limit: int = 300,
) -> pd.DataFrame:
    filters = []
    params: list[Any] = []
    if model_id is not None:
        filters.append("mp.model_id = ?")
        params.append(model_id)
    if direction is not None:
        filters.append("mp.direction = ?")
        params.append(direction)
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
            sd.decision
        FROM model_predictions AS mp
        LEFT JOIN signal_decisions AS sd
            ON sd.prediction_id = mp.id
        {where_clause}
        ORDER BY mp.timestamp_utc DESC, mp.id DESC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    payload = [
        {
            "prediction_id": int(row["prediction_id"]),
            "signal_decision_id": int(row["signal_decision_id"]) if row["signal_decision_id"] is not None else None,
            "model_id": row["model_id"],
            "direction": row["direction"],
            "timestamp": _parse_datetime(row["timestamp_utc"]),
            "regime": row["regime"],
            "raw_score": float(row["raw_score"]) if row["raw_score"] is not None else None,
            "calibrated_probability": float(row["calibrated_probability"]),
            "threshold_applied": float(row["threshold_applied"]) if row["threshold_applied"] is not None else None,
            "threshold_source": row["threshold_source"],
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
) -> DashboardHealthSummary:
    resolved_now = ensure_utc(now or utc_now())
    latest_bar_timestamp = store.get_latest_bar_timestamp(
        asset=asset,
        timeframe=timeframe,
    )
    latest_prediction_timestamp = _fetch_latest_timestamp(
        store,
        table_name="model_predictions",
    )
    latest_signal_timestamp = _fetch_latest_timestamp(
        store,
        table_name="signal_decisions",
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
) -> pd.DataFrame:
    signals = fetch_recent_signals(
        audit_repository,
        model_ids=model_ids,
        decisions=decisions,
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


def _fetch_latest_timestamp(store: SQLiteLiveDataStore, *, table_name: str) -> datetime | None:
    row = store.connection.execute(
        f"""
        SELECT timestamp_utc
        FROM {table_name}
        ORDER BY timestamp_utc DESC, id DESC
        LIMIT 1
        """
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
