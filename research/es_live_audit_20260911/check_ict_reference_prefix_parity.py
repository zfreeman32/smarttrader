"""Demonstrate ICT reference-level prefix parity using frozen collected ES bars."""
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import json
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))
from ict.structure.liquidity import build_reference_level_features

db=sqlite3.connect('file:'+(OUT/'es_audit_snapshot.sqlite3').as_posix()+'?mode=ro',uri=True)
frame=pd.read_sql_query("SELECT timestamp_utc as datetime,open,high,low,close,volume FROM canonical_bars WHERE asset='ES' AND timeframe='5m' ORDER BY timestamp_utc",db)
frame['datetime']=pd.to_datetime(frame['datetime'],utc=True)
config=SimpleNamespace(instrument='es',source_timezone='UTC',canonical_timezone='UTC')
columns=['ict_is_rth','ict_prior_rth_high','ict_prior_rth_low','ict_prior_rth_close','ict_rth_open','ict_prior_week_high','ict_prior_week_low']
cases=[]
for timestamp in ['2026-09-10T20:00:00Z','2026-09-11T12:00:00Z','2026-09-11T13:25:00Z']:
    cut=pd.Timestamp(timestamp)
    before=frame.loc[frame['datetime']<=cut].reset_index(drop=True)
    extended=frame.loc[frame['datetime']<=pd.Timestamp('2026-09-11T15:00:00Z')].reset_index(drop=True)
    prefix_values=build_reference_level_features(before,config).iloc[-1]
    extended_values=build_reference_level_features(extended,config).iloc[len(before)-1]
    def normalize(row):return {k:None if pd.isna(row[k]) else float(row[k]) for k in columns}
    case={'timestamp':timestamp,'prefix_last_bar':str(before['datetime'].iloc[-1]),'prefix_rows':len(before),
          'extended_rows':len(extended),'prefix':normalize(prefix_values),'extended_same_bar':normalize(extended_values)}
    cases.append(case)
(OUT/'ict_reference_prefix_parity.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
print(json.dumps(cases,indent=2))
