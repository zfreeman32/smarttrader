CREATE TABLE IF NOT EXISTS runtime_manifests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_hash TEXT NOT NULL UNIQUE,
    model_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    backend TEXT NOT NULL,
    role TEXT,
    status TEXT,
    manifest_version TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    manifest_path TEXT,
    policy_path TEXT,
    artifact_references_json TEXT NOT NULL,
    live_policy_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    runtime_manifest_id INTEGER,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT,
    timestamp_utc TEXT NOT NULL,
    source_row_idx INTEGER,
    feature_values_json TEXT NOT NULL,
    valid_feature_count INTEGER,
    metadata_json TEXT,
    snapshot_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    FOREIGN KEY (runtime_manifest_id) REFERENCES runtime_manifests(id)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    runtime_manifest_id INTEGER,
    feature_snapshot_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    backend TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    source_row_idx INTEGER,
    regime TEXT,
    raw_score REAL,
    calibrated_probability REAL NOT NULL,
    threshold_applied REAL,
    threshold_source TEXT NOT NULL,
    metadata_json TEXT,
    prediction_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    FOREIGN KEY (runtime_manifest_id) REFERENCES runtime_manifests(id),
    FOREIGN KEY (feature_snapshot_id) REFERENCES feature_snapshots(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS signal_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    runtime_manifest_id INTEGER,
    prediction_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    source_row_idx INTEGER,
    decision TEXT NOT NULL,
    probability REAL NOT NULL,
    threshold REAL,
    regime TEXT,
    reasons_json TEXT NOT NULL,
    cooldown_bars_remaining INTEGER,
    metadata_json TEXT,
    signal_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    FOREIGN KEY (runtime_manifest_id) REFERENCES runtime_manifests(id),
    FOREIGN KEY (prediction_id) REFERENCES model_predictions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_decision_id INTEGER,
    channel TEXT NOT NULL,
    status TEXT NOT NULL,
    dedupe_key TEXT,
    payload_json TEXT NOT NULL,
    metadata_json TEXT,
    error_message TEXT,
    sent_at_utc TEXT NOT NULL,
    FOREIGN KEY (signal_decision_id) REFERENCES signal_decisions(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS media_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_decision_id INTEGER,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    metadata_json TEXT,
    captured_at_utc TEXT NOT NULL,
    FOREIGN KEY (signal_decision_id) REFERENCES signal_decisions(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS health_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    event_timestamp_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runtime_manifests_model_id
    ON runtime_manifests (model_id, generated_at_utc);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_lookup
    ON feature_snapshots (asset, timeframe, timestamp_utc, direction);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_manifest
    ON feature_snapshots (runtime_manifest_id, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_model_predictions_lookup
    ON model_predictions (model_id, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_model_predictions_feature_snapshot
    ON model_predictions (feature_snapshot_id);

CREATE INDEX IF NOT EXISTS idx_signal_decisions_lookup
    ON signal_decisions (model_id, timestamp_utc, decision);

CREATE INDEX IF NOT EXISTS idx_signal_decisions_prediction
    ON signal_decisions (prediction_id);

CREATE INDEX IF NOT EXISTS idx_notifications_signal
    ON notifications (signal_decision_id, sent_at_utc);

CREATE INDEX IF NOT EXISTS idx_media_artifacts_signal
    ON media_artifacts (signal_decision_id, captured_at_utc);

CREATE INDEX IF NOT EXISTS idx_health_events_component_time
    ON health_events (component, event_timestamp_utc);
