-- A5: immutable FRVP/ICT setup observations, links and diagnostic outcomes.

CREATE TABLE IF NOT EXISTS setup_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    previous_event_id INTEGER REFERENCES setup_events(id),
    event_kind TEXT NOT NULL CHECK(event_kind IN ('observed','revised','invalidated')),
    collection_version TEXT NOT NULL,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy TEXT NOT NULL,
    source_timestamp_utc TEXT NOT NULL,
    source_bar_version TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    first_observed_at_utc TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    setup_side INTEGER NOT NULL,
    setup_family TEXT NOT NULL,
    rule_confidence REAL,
    detector_identity TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    selected INTEGER NOT NULL,
    geometry_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(event_key, revision)
);
CREATE INDEX IF NOT EXISTS idx_setup_events_source
ON setup_events(asset, timeframe, collection_version, source_timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_setup_events_key ON setup_events(event_key, revision);
CREATE TABLE IF NOT EXISTS setup_prediction_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_event_id INTEGER NOT NULL REFERENCES setup_events(id),
    prediction_id INTEGER NOT NULL REFERENCES model_predictions(id),
    model_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(setup_event_id, prediction_id)
);
CREATE TABLE IF NOT EXISTS setup_event_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_event_id INTEGER NOT NULL REFERENCES setup_events(id),
    outcome_type TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(setup_event_id, outcome_type, content_hash)
);

CREATE TRIGGER IF NOT EXISTS setup_events_no_update BEFORE UPDATE ON setup_events
BEGIN SELECT RAISE(ABORT, 'setup history is append-only'); END;

CREATE TRIGGER IF NOT EXISTS setup_events_no_delete BEFORE DELETE ON setup_events
BEGIN SELECT RAISE(ABORT, 'setup history is append-only'); END;

CREATE TRIGGER IF NOT EXISTS setup_prediction_links_no_update BEFORE UPDATE ON setup_prediction_links
BEGIN SELECT RAISE(ABORT, 'setup history is append-only'); END;

CREATE TRIGGER IF NOT EXISTS setup_prediction_links_no_delete BEFORE DELETE ON setup_prediction_links
BEGIN SELECT RAISE(ABORT, 'setup history is append-only'); END;

CREATE TRIGGER IF NOT EXISTS setup_event_outcomes_no_update BEFORE UPDATE ON setup_event_outcomes
BEGIN SELECT RAISE(ABORT, 'setup history is append-only'); END;

CREATE TRIGGER IF NOT EXISTS setup_event_outcomes_no_delete BEFORE DELETE ON setup_event_outcomes
BEGIN SELECT RAISE(ABORT, 'setup history is append-only'); END;
