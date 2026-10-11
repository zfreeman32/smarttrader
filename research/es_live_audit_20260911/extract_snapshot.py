"""Read-only ES audit extraction. Never imports or invokes live services."""
from pathlib import Path
import datetime as dt
import json
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
source = ROOT / 'ote_live/runtime_data/live_market_data.sqlite3'
dest = OUT / 'es_audit_snapshot.sqlite3'
if dest.exists():
    raise SystemExit('Snapshot already exists; preserve audit cutoff.')
src = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
src.execute('PRAGMA query_only=ON')
src.execute('BEGIN')
dst = sqlite3.connect(dest)
filters = {
    'canonical_bars': "asset='ES'",
    'runtime_manifests': "model_id LIKE 'frvp_%' OR model_id LIKE 'ict_%'",
    'model_predictions': "model_id LIKE 'frvp_%' OR model_id LIKE 'ict_%'",
    'signal_decisions': "model_id LIKE 'frvp_%' OR model_id LIKE 'ict_%'",
    'feature_snapshots': "asset='ES'",
    'ict_paper_signal_events': "asset='ES'",
    'frvp_paper_signal_events': "asset='ES'",
    'ingestion_gaps': "asset='ES'",
    'runtime_state': "scope='dashboard_view_state' OR (scope='ibkr.market_data' AND state_key='ES')",
}
manifest = {'extracted_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'source': str(source), 'method': 'SQLite read-only BEGIN transaction, ES-only table extraction', 'tables': {}}
for name, where in filters.items():
    schema = src.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()[0]
    dst.execute(schema)
    cur = src.execute(f'SELECT * FROM {name} WHERE {where}')
    n = 0
    for_batch = f'INSERT INTO {name} VALUES ({",".join("?" for _ in cur.description)})'
    while batch := cur.fetchmany(500):
        dst.executemany(for_batch, batch)
        n += len(batch)
    manifest['tables'][name] = n
    print(name, n, flush=True)
manifest['raw_es_events'] = [dict(zip(['provider','event_type','n','first_event','last_event','first_ingestion','last_ingestion'], r)) for r in src.execute("SELECT provider,event_type,count(*),min(event_timestamp_utc),max(event_timestamp_utc),min(ingested_at_utc),max(ingested_at_utc) FROM raw_market_events WHERE asset='ES' GROUP BY provider,event_type")]
manifest['health_summary'] = [dict(zip(['component','event_type','severity','n'], r)) for r in src.execute("SELECT component,event_type,severity,count(*) FROM health_events WHERE lower(component) LIKE '%es%' OR lower(component) LIKE '%frvp%' OR lower(component) LIKE '%ict%' OR lower(component) LIKE '%ibkr%' GROUP BY component,event_type,severity")]
dst.commit()
src.rollback()
src.close()
for table in ['model_predictions','signal_decisions']:
    dst.execute(f'CREATE INDEX audit_{table}_model_time ON {table}(model_id,timestamp_utc)')
dst.commit()
pd.read_sql_query("SELECT * FROM canonical_bars WHERE timeframe='5m' ORDER BY timestamp_utc", dst).to_csv(OUT/'es_bars_5m.csv', index=False)
dst.close()
(OUT/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest['tables']), flush=True)
