CREATE TABLE IF NOT EXISTS frvp_paper_signal_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL UNIQUE,
    bundle_id TEXT NOT NULL,
    runtime_manifest_id INTEGER NOT NULL,
    manifest_hash TEXT NOT NULL,
    signal_decision_id INTEGER NOT NULL UNIQUE,
    model_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction = 'long'),
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    source_timestamp_utc TEXT NOT NULL,
    source_row_idx INTEGER NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,
    composite_regime TEXT,
    session_regime TEXT NOT NULL,
    entry_timing TEXT NOT NULL CHECK (entry_timing = 'signal_close'),
    entry_timestamp_utc TEXT NOT NULL,
    entry_source_row_idx INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    holding_period_bars INTEGER NOT NULL CHECK (holding_period_bars = 120),
    exit_timing TEXT NOT NULL CHECK (exit_timing = 'close_after_120_completed_bars'),
    stop_price REAL,
    target_price REAL,
    stop_target_semantics TEXT NOT NULL CHECK (stop_target_semantics = 'not_applicable'),
    tick_size REAL NOT NULL,
    tick_value REAL NOT NULL,
    spread_cost_mode TEXT NOT NULL CHECK (spread_cost_mode = 'session_schedule'),
    entry_spread_ticks REAL NOT NULL,
    exit_spread_ticks REAL NOT NULL,
    fixed_slippage_ticks REAL NOT NULL,
    commission_ticks REAL NOT NULL,
    total_cost_ticks REAL NOT NULL,
    lifecycle_status TEXT NOT NULL CHECK (lifecycle_status IN ('open', 'settled')),
    exit_timestamp_utc TEXT,
    exit_source_row_idx INTEGER,
    exit_price REAL,
    gross_pnl_ticks REAL,
    net_pnl_ticks REAL,
    gross_pnl_dollars REAL,
    net_pnl_dollars REAL,
    outcome TEXT CHECK (outcome IS NULL OR outcome IN ('win', 'loss', 'flat')),
    metadata_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    settled_at_utc TEXT,
    FOREIGN KEY (runtime_manifest_id) REFERENCES runtime_manifests(id),
    FOREIGN KEY (signal_decision_id) REFERENCES signal_decisions(id) ON DELETE RESTRICT,
    CHECK (stop_price IS NULL AND target_price IS NULL),
    CHECK (
        (lifecycle_status = 'open' AND exit_timestamp_utc IS NULL AND exit_price IS NULL
            AND gross_pnl_ticks IS NULL AND net_pnl_ticks IS NULL AND outcome IS NULL)
        OR
        (lifecycle_status = 'settled' AND exit_timestamp_utc IS NOT NULL AND exit_price IS NOT NULL
            AND gross_pnl_ticks IS NOT NULL AND net_pnl_ticks IS NOT NULL AND outcome IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_frvp_paper_signal_events_pending
    ON frvp_paper_signal_events (lifecycle_status, asset, timeframe, entry_timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_frvp_paper_signal_events_source
    ON frvp_paper_signal_events (bundle_id, model_id, source_timestamp_utc, source_row_idx);
