from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import BackfillWindow, HeartbeatStatus, IngestionGap, ensure_utc, utc_now

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
MIGRATIONS_DIR = Path(__file__).with_name("migrations")


@dataclass(frozen=True)
class StoredIngestionGap:
    gap_id: int
    asset: str
    timeframe: str
    expected_timestamp: datetime
    observed_timestamp: datetime
    missing_timestamps: tuple[datetime, ...]
    gap_size: int
    detected_at_utc: datetime
    resolved_at_utc: datetime | None = None

    def to_ingestion_gap(self) -> IngestionGap:
        return IngestionGap(
            asset=self.asset,
            timeframe=self.timeframe,  # type: ignore[arg-type]
            expected_timestamp=self.expected_timestamp,
            observed_timestamp=self.observed_timestamp,
            missing_timestamps=list(self.missing_timestamps),
            gap_size=self.gap_size,
            detected_at_utc=self.detected_at_utc,
        )

    def to_backfill_window(self) -> BackfillWindow:
        return self.to_ingestion_gap().to_backfill_window()


class SQLiteLiveDataStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The dashboard serves requests on worker threads, so its read path must
        # be able to reuse the same SQLite connection safely across threads.
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL;")
        self._connection.execute("PRAGMA busy_timeout=30000;")
        self.initialize()

    def initialize(self) -> None:
        self._connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._apply_pending_migrations()
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteLiveDataStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def upsert_bar(self, bar: MarketBar) -> None:
        now = _isoformat(utc_now())
        self._connection.execute(
            """
            INSERT INTO canonical_bars (
                asset, timeframe, timestamp_utc, open, high, low, close, volume,
                bid, ask, spread, source, symbol, contract_symbol, instrument_id,
                inserted_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                symbol = excluded.symbol,
                contract_symbol = excluded.contract_symbol,
                instrument_id = excluded.instrument_id,
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
                bar.symbol,
                bar.contract_symbol,
                bar.instrument_id,
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

    def fetch_gaps(
        self,
        *,
        asset: str | None = None,
        timeframe: str | None = None,
        unresolved_only: bool = False,
    ) -> tuple[StoredIngestionGap, ...]:
        query = """
            SELECT id, asset, timeframe, expected_timestamp_utc, observed_timestamp_utc,
                   missing_timestamps_json, gap_size, detected_at_utc, resolved_at_utc
            FROM ingestion_gaps
            WHERE 1 = 1
        """
        params: list[Any] = []
        if asset is not None:
            query += " AND asset = ?"
            params.append(asset)
        if timeframe is not None:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if unresolved_only:
            query += " AND resolved_at_utc IS NULL"
        query += " ORDER BY expected_timestamp_utc ASC, id ASC"
        rows = self._connection.execute(query, params).fetchall()
        return tuple(_gap_row_to_record(row) for row in rows)

    def mark_gap_resolved(
        self,
        gap_id: int,
        *,
        resolved_at_utc: datetime | None = None,
    ) -> None:
        self._connection.execute(
            """
            UPDATE ingestion_gaps
            SET resolved_at_utc = ?
            WHERE id = ?
            """,
            (_isoformat(resolved_at_utc or utc_now()), int(gap_id)),
        )
        self._connection.commit()

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
                symbol=row["symbol"],
                contract_symbol=row["contract_symbol"],
                instrument_id=int(row["instrument_id"]) if row["instrument_id"] is not None else None,
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

    def _apply_pending_migrations(self) -> None:
        if not MIGRATIONS_DIR.exists():
            return

        applied = {
            row["migration_id"]
            for row in self._connection.execute("SELECT migration_id FROM schema_migrations").fetchall()
        }
        for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            migration_id = migration_path.name
            if migration_id in applied:
                continue
            self._connection.executescript(migration_path.read_text(encoding="utf-8"))
            self._connection.execute(
                """
                INSERT INTO schema_migrations (migration_id, applied_at_utc)
                VALUES (?, ?)
                """,
                (migration_id, _isoformat(utc_now())),
            )


def _isoformat(value: datetime) -> str:
    return ensure_utc(value).isoformat()


def _parse_datetime(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value))


def _gap_row_to_record(row) -> StoredIngestionGap:
    missing_payload = json.loads(row["missing_timestamps_json"])
    missing_timestamps = tuple(_parse_datetime(str(item)) for item in missing_payload)
    return StoredIngestionGap(
        gap_id=int(row["id"]),
        asset=str(row["asset"]),
        timeframe=str(row["timeframe"]),
        expected_timestamp=_parse_datetime(row["expected_timestamp_utc"]),
        observed_timestamp=_parse_datetime(row["observed_timestamp_utc"]),
        missing_timestamps=missing_timestamps,
        gap_size=int(row["gap_size"]),
        detected_at_utc=_parse_datetime(row["detected_at_utc"]),
        resolved_at_utc=(
            _parse_datetime(row["resolved_at_utc"])
            if row["resolved_at_utc"] is not None
            else None
        ),
    )
