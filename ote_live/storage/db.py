from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import HeartbeatStatus, IngestionGap, ensure_utc, utc_now

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SQLiteLiveDataStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL;")
        self.initialize()

    def initialize(self) -> None:
        self._connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteLiveDataStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def upsert_bar(self, bar: MarketBar) -> None:
        now = _isoformat(utc_now())
        self._connection.execute(
            """
            INSERT INTO canonical_bars (
                asset, timeframe, timestamp_utc, open, high, low, close, volume,
                bid, ask, spread, source, inserted_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset, timeframe, timestamp_utc) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                bid = excluded.bid,
                ask = excluded.ask,
                spread = excluded.spread,
                source = excluded.source,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                bar.asset,
                bar.timeframe,
                _isoformat(bar.timestamp),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.bid,
                bar.ask,
                bar.spread,
                bar.source,
                now,
                now,
            ),
        )
        self._connection.commit()

    def upsert_bars(self, bars: list[MarketBar]) -> None:
        for bar in bars:
            self.upsert_bar(bar)

    def record_raw_event(
        self,
        *,
        provider: str,
        event_type: str,
        payload: dict[str, Any],
        asset: str | None = None,
        timeframe: str | None = None,
        event_timestamp_utc: datetime | None = None,
    ) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO raw_market_events (
                provider, event_type, asset, timeframe, event_timestamp_utc, payload_json, ingested_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                provider,
                event_type,
                asset,
                timeframe,
                _isoformat(event_timestamp_utc) if event_timestamp_utc is not None else None,
                json.dumps(payload, sort_keys=True, default=str),
                _isoformat(utc_now()),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def record_gap(self, gap: IngestionGap) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO ingestion_gaps (
                asset, timeframe, expected_timestamp_utc, observed_timestamp_utc,
                missing_timestamps_json, gap_size, detected_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gap.asset,
                gap.timeframe,
                _isoformat(gap.expected_timestamp),
                _isoformat(gap.observed_timestamp),
                json.dumps([_isoformat(item) for item in gap.missing_timestamps]),
                gap.gap_size,
                _isoformat(gap.detected_at_utc),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def record_heartbeat(self, status: HeartbeatStatus, *, metadata: dict[str, Any] | None = None) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO heartbeat_log (
                source, observed_at_utc, stale_after_seconds, lag_seconds, is_stale, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                status.source,
                _isoformat(status.observed_at_utc),
                status.stale_after_seconds,
                status.lag_seconds,
                int(status.is_stale),
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def upsert_runtime_state(
        self,
        *,
        scope: str,
        state_key: str,
        payload: dict[str, Any],
    ) -> None:
        now = _isoformat(utc_now())
        self._connection.execute(
            """
            INSERT INTO runtime_state (
                scope, state_key, payload_json, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scope, state_key) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                scope,
                state_key,
                json.dumps(payload, sort_keys=True, default=str),
                now,
                now,
            ),
        )
        self._connection.commit()

    def get_runtime_state(
        self,
        *,
        scope: str,
        state_key: str,
    ) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT payload_json
            FROM runtime_state
            WHERE scope = ? AND state_key = ?
            LIMIT 1
            """,
            (scope, state_key),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def delete_runtime_state(
        self,
        *,
        scope: str,
        state_key: str,
    ) -> None:
        self._connection.execute(
            """
            DELETE FROM runtime_state
            WHERE scope = ? AND state_key = ?
            """,
            (scope, state_key),
        )
        self._connection.commit()

    def fetch_bars(
        self,
        *,
        asset: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[MarketBar]:
        query = """
            SELECT asset, timeframe, timestamp_utc, open, high, low, close, volume, bid, ask, spread, source
            FROM canonical_bars
            WHERE asset = ? AND timeframe = ?
        """
        params: list[Any] = [asset, timeframe]
        if start is not None:
            query += " AND timestamp_utc >= ?"
            params.append(_isoformat(start))
        if end is not None:
            query += " AND timestamp_utc <= ?"
            params.append(_isoformat(end))
        query += " ORDER BY timestamp_utc ASC"

        rows = self._connection.execute(query, params).fetchall()
        return [
            MarketBar(
                asset=row["asset"],
                timeframe=row["timeframe"],
                timestamp=_parse_datetime(row["timestamp_utc"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                bid=float(row["bid"]) if row["bid"] is not None else None,
                ask=float(row["ask"]) if row["ask"] is not None else None,
                spread=float(row["spread"]) if row["spread"] is not None else None,
                source=row["source"],
            )
            for row in rows
        ]

    def get_latest_bar_timestamp(
        self,
        *,
        asset: str,
        timeframe: str,
    ) -> datetime | None:
        row = self._connection.execute(
            """
            SELECT timestamp_utc
            FROM canonical_bars
            WHERE asset = ? AND timeframe = ?
            ORDER BY timestamp_utc DESC
            LIMIT 1
            """,
            (asset, timeframe),
        ).fetchone()
        if row is None:
            return None
        return _parse_datetime(row["timestamp_utc"])


def _isoformat(value: datetime) -> str:
    return ensure_utc(value).isoformat()


def _parse_datetime(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value))
