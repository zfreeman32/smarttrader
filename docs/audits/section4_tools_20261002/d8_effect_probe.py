"""Section 4 prompt 2: feature effect of D8 (carrying canonical-row provenance onto fetched bars).

Reads the last runtime window of recorded canonical bars from the live DB (read-only URI) and builds the
group's IncrementalFeatureEngine twice: bars as the app fetches them today (no provenance) and with the
D8 provenance carried. Writes counts only. Usage: python d8_effect_probe.py <ES|EURUSD> <ids> <out.json>
"""
import json, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from time_builds import staged
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.ingestion.signals import _bar_from_row

asset, ids, out = sys.argv[1], {int(x) for x in sys.argv[2].split(',')}, Path(sys.argv[3])
con = sqlite3.connect('file:ote_live/runtime_data/live_market_data.sqlite3?mode=ro', uri=True)
con.row_factory = sqlite3.Row
freeze = json.load(open('configs/b4_broad_full_session_shadow_freeze_v2.json'))
manifests = [staged(e) for e in freeze['frozen_replacements'] if e['queue_id'] in ids]
window = IncrementalFeatureEngine(manifests).plan.runtime_history_bars
rows = con.execute("SELECT * FROM canonical_bars WHERE asset=? AND timeframe='5m' ORDER BY timestamp_utc DESC LIMIT ?",
                   (asset, window)).fetchall()[::-1]


def build(carry):
    engine = IncrementalFeatureEngine(manifests)
    for row in rows:
        engine.state.ingest_bar(_bar_from_row(row, carry_provenance=carry))
    return engine.build_feature_frame()


off, on = build(False), build(True)
diff = {}
for column in off.columns:
    x, y = off[column].to_numpy(float), on[column].to_numpy(float)
    changed = ~((x == y) | (np.isnan(x) & np.isnan(y)))
    if changed.any():
        diff[column] = int(changed.sum())
doc = dict(probe='section4-d8-effect-v1', asset=asset, queue_ids=sorted(ids), rows=len(rows),
           first=str(rows[0]['timestamp_utc']), last=str(rows[-1]['timestamp_utc']),
           is_complete_counts={str(k): int(v) for k, v in pd.Series([r['is_complete'] for r in rows]).value_counts(dropna=False).items()},
           selected_features=int(off.shape[1]), changed_feature_count=len(diff),
           changed_feature_prefixes=sorted({c.split('_')[0] + '_' + c.split('_')[1] for c in diff}),
           changed_cells=int(sum(diff.values())), top_changed=sorted(diff.items(), key=lambda t: -t[1])[:10],
           endpoint_nan_off=int(off.iloc[-1].isna().sum()), endpoint_nan_on=int(on.iloc[-1].isna().sum()),
           non_htf_changes=sorted(c for c in diff if 'htf' not in c))
out.write_text(json.dumps(doc, indent=1) + '\n')
print('D8 PROBE COMPLETE', asset, len(diff), doc['endpoint_nan_off'], doc['endpoint_nan_on'], flush=True)
