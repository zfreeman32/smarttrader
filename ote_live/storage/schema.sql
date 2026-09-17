PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_market_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    event_type TEXT NOT NULL,
    asset TEXT,
    timeframe TEXT,
    event_timestamp_utc TEXT,
    payload_json TEXT NOT NULL,
    ingested_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_bars (
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL DEFAULT 0,
    bid REAL,
    ask REAL,
    spread REAL,
    source TEXT,
    inserted_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (asset, timeframe, timestamp_utc)
);

CREATE TABLE IF NOT EXISTS ingestion_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    expected_timestamp_utc TEXT NOT NULL,
    observed_timestamp_utc TEXT NOT NULL,
    missing_timestamps_json TEXT NOT NULL,
    gap_size INTEGER NOT NULL,
    detected_at_utc TEXT NOT NULL,
    resolved_at_utc TEXT
);

CREATE TABLE IF NOT EXISTS heartbeat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    stale_after_seconds REAL NOT NULL,
    lag_seconds REAL NOT NULL,
    is_stale INTEGER NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS runtime_state (
    scope TEXT NOT NULL,
    state_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (scope, state_key)
);

CREATE INDEX IF NOT EXISTS idx_raw_market_events_provider_time
    ON raw_market_events (provider, event_timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_canonical_bars_timeframe_timestamp
    ON canonical_bars (timeframe, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_ingestion_gaps_unresolved
    ON ingestion_gaps (asset, timeframe, resolved_at_utc);

CREATE INDEX IF NOT EXISTS idx_heartbeat_log_source_time
    ON heartbeat_log (source, observed_at_utc);

CREATE INDEX IF NOT EXISTS idx_runtime_state_scope_time
    ON runtime_state (scope, updated_at_utc);
