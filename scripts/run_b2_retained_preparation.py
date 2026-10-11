"""Run the four ES replacement-data jobs sequentially as their parents finish."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
JOBS=[
    ('frvp_july','b2_frvp_july_features_20260921_v2','b2_frvp_labels_20260920_v1/frvp_es_primary_refresh_20260701','labels.csv.gz','events.csv.gz','b2_frvp_july_prepared_20260921_v1'),
    ('ict','b2_ict_features_20260920_v2','b2_ict_labels_20260921_v1','ict_es_labels.csv','ict_es_events.csv','b2_ict_prepared_20260921_v1'),
    ('ict_specialists','b2_ict_features_20260920_v2','b2_ict_labels_20260921_v1','ict_es_labels.csv','ict_es_events.csv','b2_ict_specialists_prepared_20260921_v1'),
    ('frvp_setups','b2_frvp_setups_features_20260921_v1','b2_frvp_labels_20260920_v1/frvp_es_primary_setup_targets_20260829','labels.csv.gz','events.csv.gz','b2_frvp_setups_prepared_20260921_v1'),
]


def main():
    os.chdir(ROOT)
    env=dict(os.environ, LOKY_MAX_CPU_COUNT='2')
    for group,features,label_root,labels,events,output in JOBS:
        parents=[ROOT/'artifacts'/features/'features.metadata.json', ROOT/'artifacts'/label_root/'comparison.json']
        if (ROOT/'artifacts'/output/'manifest.json').exists():
            print('Already complete',group,flush=True)
            continue
        print('Waiting for verified parents',group,flush=True)
        while not all(p.exists() for p in parents):
            time.sleep(15)
        print('Starting retained preparation',group,flush=True)
        subprocess.run([sys.executable,'-B','scripts/rebuild_b2_prepared.py','--group',group,
            '--features-root',f'artifacts/{features}','--labels',f'artifacts/{label_root}/{labels}',
            '--events',f'artifacts/{label_root}/{events}','--output-root',f'artifacts/{output}'],check=True,env=env)
    output=ROOT/'artifacts/b2_frvp_recency_prepared_20260921_v1/prepared'
    if not output.exists():
        subprocess.run([sys.executable,'-B','scripts/materialize_frvp_recency_prepared_root.py',
            '--base-prepared-root','artifacts/b2_frvp_july_prepared_20260921_v1/prepared',
            '--output-prepared-root',str(output),'--target','long_frvp_reversal',
            '--source-dataset','artifacts/b2_frvp_july_prepared_20260921_v1/dataset.parquet',
            '--half-life-days','730','--floor','0.20'],check=True,env=env)
    print('All retained ES preparation completed',flush=True)


if __name__=='__main__':
    main()
