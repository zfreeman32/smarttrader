ALTER TABLE canonical_bars ADD COLUMN source_timestamp_utc TEXT;
ALTER TABLE canonical_bars ADD COLUMN bar_version TEXT;
ALTER TABLE canonical_bars ADD COLUMN is_complete INTEGER;
ALTER TABLE canonical_bars ADD COLUMN feed_type TEXT;
ALTER TABLE canonical_bars ADD COLUMN first_observed_at_utc TEXT;
ALTER TABLE canonical_bars ADD COLUMN last_observed_at_utc TEXT;
ALTER TABLE canonical_bars ADD COLUMN observation_kind TEXT NOT NULL DEFAULT 'unknown';

CREATE TABLE IF NOT EXISTS source_bar_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    bar_version TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_bar_history_bar
ON source_bar_history(asset, timeframe, timestamp_utc, id);
CREATE TRIGGER IF NOT EXISTS source_bar_history_no_update
BEFORE UPDATE ON source_bar_history BEGIN SELECT RAISE(ABORT, 'source bar history is immutable'); END;
CREATE TRIGGER IF NOT EXISTS source_bar_history_no_delete
BEFORE DELETE ON source_bar_history BEGIN SELECT RAISE(ABORT, 'source bar history is immutable'); END;
