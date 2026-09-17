CREATE TABLE IF NOT EXISTS collection_versions (
    collection_version TEXT PRIMARY KEY,
    contract_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS collection_versions_no_update BEFORE UPDATE ON collection_versions
BEGIN SELECT RAISE(ABORT, 'collection contracts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS collection_versions_no_delete BEFORE DELETE ON collection_versions
BEGIN SELECT RAISE(ABORT, 'collection contracts are immutable'); END;
ALTER TABLE feature_snapshots ADD COLUMN collection_version TEXT NOT NULL DEFAULT 'legacy-unversioned';
ALTER TABLE model_predictions ADD COLUMN collection_version TEXT NOT NULL DEFAULT 'legacy-unversioned';
ALTER TABLE signal_decisions ADD COLUMN collection_version TEXT NOT NULL DEFAULT 'legacy-unversioned';
CREATE INDEX IF NOT EXISTS predictions_collection_model_time
ON model_predictions(collection_version, model_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS decisions_collection_model_time
ON signal_decisions(collection_version, model_id, timestamp_utc);
