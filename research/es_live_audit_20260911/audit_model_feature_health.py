"""Read-only selected-feature and policy-context diagnostics from the audit snapshot."""
import collections
import csv
import datetime
import json
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
db = sqlite3.connect('file:' + (OUT / 'es_audit_snapshot.sqlite3').as_posix() + '?mode=ro', uri=True)
def missing(v):
    return v is None or isinstance(v, float) and math.isnan(v)

rows = []
models = collections.defaultdict(list)
for model, timestamp, score, probability, features in db.execute('''
SELECT p.model_id,p.timestamp_utc,p.raw_score,p.calibrated_probability,f.feature_values_json
FROM model_predictions p JOIN feature_snapshots f ON f.id=p.feature_snapshot_id
WHERE f.asset='ES' AND p.id IN
(SELECT max(id) FROM model_predictions WHERE model_id LIKE 'frvp_%' OR model_id LIKE 'ict_%' GROUP BY model_id,timestamp_utc)
'''):
    values = json.loads(features)
    models[model].append((timestamp,score,probability,values))
for model, values in sorted(models.items()):
    nulls = collections.Counter()
    counts = collections.Counter()
    for _,_,_,features in values:
        counts.update(features.keys())
        nulls.update(k for k,v in features.items() if missing(v))
    for feature,total in counts.items():
        rows.append(dict(model_id=model, feature=feature, snapshots=total, missing=nulls[feature],missing_share=nulls[feature]/total))
    print('MODEL',model,'n',len(values),'raw_equals_calibrated',sum(s==p for _,s,p,_ in values),
          'top_nulls',nulls.most_common(8))
with (OUT/'model_selected_feature_missingness.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

context_rows=[]
counts=collections.defaultdict(collections.Counter)
for ts, metadata in db.execute('''SELECT p.timestamp_utc,p.metadata_json FROM model_predictions p
WHERE model_id='ict_long_meta_xgb_v1' AND id IN
(SELECT max(id) FROM model_predictions WHERE model_id='ict_long_meta_xgb_v1' GROUP BY timestamp_utc)
ORDER BY timestamp_utc'''):
    context=json.loads(metadata or '{}').get('policy_context',{})
    et=datetime.datetime.fromisoformat(ts).astimezone(ZoneInfo('America/New_York'))
    expected_rth=et.weekday()<5 and et.date()!=datetime.date(2026,9,7) and (9,30)<=(et.hour,et.minute)<(16,0)
    record=dict(timestamp=ts,et_time=et.isoformat(),expected_rth=expected_rth)
    for key in ('ict_is_rth','ict_session_phase_code','ict_prior_rth_high','ict_prior_rth_close','ict_rth_open',
                'ict_prior_week_high','ict_ib_complete','ict_ib_high','ict_midnight_open','ict_overnight_high','hour'):
        record[key]=context.get(key)
    record['setup_keys']=','.join(k for k in context if 'setup' in k)
    context_rows.append(record)
    bucket=counts[(et.hour,expected_rth)]
    bucket['rows']+=1
    bucket['is_rth_mismatch']+=int(context.get('ict_is_rth')!=int(expected_rth))
    for key in ('ict_prior_rth_high','ict_rth_open','ict_prior_week_high','ict_ib_high'):
        bucket[key+'_null']+=int(missing(context.get(key)))
with (OUT/'ict_policy_reference_health.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(context_rows[0]));w.writeheader();w.writerows(context_rows)
print('CONTEXT HOURS')
for key,value in sorted(counts.items()):print(key,dict(value))
print('LAST CONTEXT',context_rows[-1])

print('MANIFEST CALIBRATOR PATH EXISTENCE')
root=OUT.parents[1]
for model,body in db.execute('SELECT model_id,manifest_json FROM runtime_manifests ORDER BY id'):
    if model in models:
        m=json.loads(body);p=m['artifact_references'].get('calibrator_file')
        print(model,p, bool(p and (root/p).exists()))
