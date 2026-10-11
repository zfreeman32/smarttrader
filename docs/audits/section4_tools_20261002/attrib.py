"""Section 4 inventory helper: exact column -> feature-set attribution on a small ES sample (read-only)."""
import json, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from features.config import FeatureBuilderConfig
from features.registry import FEATURE_REGISTRY
from features.io import standardize_market_frame
from features.builder import validate_ohlcv
out = Path(sys.argv[1])
meta = json.load(open('artifacts/b2_frvp_july_prepared_20260921_v1/dataset.metadata.json'))
ict = json.load(open('artifacts/b2_ict_prepared_20260921_v1/dataset.metadata.json'))
cfg = FeatureBuilderConfig.from_dict(meta['config'])
sets = list(dict.fromkeys(meta['feature_sets'] + ict['feature_sets']))
cfg.feature_sets = sets
raw = pd.read_csv('data/futures_data/ES-5m-tagged.csv', nrows=12000).rename(columns={'ts_event': 'datetime'})
raw = raw.tail(9000).reset_index(drop=True)
w = standardize_market_frame(raw, source_timezone=cfg.source_timezone, canonical_timezone=cfg.canonical_timezone)
w = validate_ohlcv(w, drop_invalid=True)
attrib = {}; timings = {}
for name in sets:
    before = set(w.columns); t = time.perf_counter()
    try:
        built = FEATURE_REGISTRY.build(name, w, cfg)
        new = [c for c in built.columns if c not in w.columns]
        w = pd.concat([w, built[new]], axis=1)
    except Exception as exc:
        attrib[name] = {'error': type(exc).__name__ + ': ' + str(exc)[:200]}; continue
    timings[name] = round(time.perf_counter() - t, 3)
    attrib[name] = sorted(set(w.columns) - before)
json.dump({'sets': sets, 'attribution': attrib, 'timings_s_9000_rows': timings}, open(out, 'w'))
print({k: (len(v) if isinstance(v, list) else v) for k, v in attrib.items()}); print(timings)
