"""Section 4 prompt 1: per-model selected-feature inventory from frozen B4 v2 metadata (read-only inputs)."""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from ote_live.features.incremental_engine import infer_required_history_bars

attr = json.load(open(sys.argv[1]))['attribution']
out = Path(sys.argv[2])
col2set = {c: s for s, cols in attr.items() if isinstance(cols, list) for c in cols}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


FREEZE = 'configs/b4_broad_full_session_shadow_freeze_v2.json'
freeze = json.load(open(FREEZE))
TRANSFORMS = [(re.compile(r'^(?P<b>.+)_lag_\d+$'), 'lag'), (re.compile(r'^(?P<b>.+)_roll_(mean|std)_\d+$'), 'rolling_stat'),
              (re.compile(r'^(?P<b>.+)_zscore_\d+$'), 'zscore'), (re.compile(r'^(?P<b>.+)_pct_rank_\d+$'), 'pct_rank'),
              (re.compile(r'^(?P<b>.+)_winsor_\d+$'), 'winsor'), (re.compile(r'^(?P<b>.+)_atr_norm$'), 'atr_norm'),
              (re.compile(r'^(?P<b>.+)_sigma_norm_\d+$'), 'sigma_norm')]
PRICE = re.compile(r'(^|_)(open|high|low|close|level|poc|vah|val|vwap|price|swing_high|swing_low|midnight_open|open_0830)(_|$)')
NOTPRICE = re.compile(r'(dist|atr|pct|ratio|_vs_|zscore|roll_std|spread|width|range|return|slope|vel|overlap|position|bars_since|count|code|direction|flag|^in_|is_|signal|strength|depth|imbalance|bps|rank|index)')


def base_of(name):
    chain = []
    while True:
        for rx, kind in TRANSFORMS:
            m = rx.match(name)
            if m and (m['b'] in col2set or any(r.match(m['b']) for r, _ in TRANSFORMS) or m['b'].startswith('strategy__')):
                chain.append(kind)
                name = m['b']
                break
        else:
            return name, chain


def producer(base):
    if base.startswith('strategy__'):
        return 'strategy_signals'
    if base in col2set:
        return col2set[base]
    if base.startswith('interaction_'):
        return 'transforms.interactions'
    return 'unattributed'


def native_tf(base):
    if base.startswith('htf_30m_'):
        return '30m', True
    if base.startswith('htf_1h_') or base == 'htf_alignment_score':
        return '1h', True
    if 'weekly' in base or 'prior_week' in base:
        return '1W', True
    if 'daily' in base or 'prev_day' in base or 'prior_rth' in base:
        return '1D/session', True
    if base.startswith('htf_'):
        return 'htf_other', True
    return '5m', False


def module(s):
    if (ROOT / f'features/feature_sets/{s}.py').exists():
        return f'features/feature_sets/{s}.py'
    if s == 'strategy_signals':
        return 'features/feature_sets/strategy_signals.py'
    if s.startswith('transforms'):
        return 'features/transforms.py'
    return None


models = []
for r in freeze['frozen_replacements']:
    root = Path(r['artifact_root'])
    ts = json.load(open(root / 'training_summary.json'))
    shc = r['source_data_pins']['source_history_contract']
    names, fills = shc['feature_names'], shc['fill_values']
    hist = Path(r['source_data_pins']['prepared_manifest']).parent / 'prepared' / r['target'] / 'source_history.parquet'
    tab = pq.read_table(hist, columns=names)
    feats = []
    for n in names:
        col = tab.column(n).to_numpy(zero_copy_only=False)
        dt = str(tab.schema.field(n).type)
        numeric = col.dtype.kind in 'fiub'
        as_f = col.astype('float64') if numeric else None
        fin = as_f[np.isfinite(as_f)] if numeric else None
        integer = numeric and (col.dtype.kind in 'iub' or (fin.size > 0 and bool(np.all(np.floor(fin) == fin))))
        b, chain = base_of(n)
        p = producer(b)
        tf, htf = native_tf(b)
        if integer:
            cls = 'C0_exact'
        elif 'volume' in b and 'relative' not in b and not chain:
            cls = 'C3_volume_scale'
        elif PRICE.search(b) and not NOTPRICE.search(b) and not chain:
            cls = 'C2_price_level'
        else:
            cls = 'C1_continuous'
        nan_frac = float(np.mean(~np.isfinite(as_f))) if numeric else None
        feats.append(dict(feature=n, base=b, transforms=chain, producer=p, producer_module=module(p), native_timeframe=tf,
                          htf_aligned=htf, frozen_dtype=dt, tolerance_class=cls, frozen_fill=fills[n],
                          frozen_nonfinite_fraction=None if nan_frac is None else round(nan_frac, 6),
                          fallback_path='htf_confluence_zero_default' if n.startswith('htf_confluence_') else None))
    mods = sorted({f['producer_module'] for f in feats if f['producer_module']})
    backend = ts['config'].get('model_type') if ts['config'].get('backend') == 'torch' else ts['config'].get('backend')
    models.append(dict(
        queue_id=r['queue_id'], model_id=r['model_id'], instrument=r['instrument'], side=r['side'], family=r['family'],
        backend=backend, artifact_root=r['artifact_root'], model_file=r['artifacts']['model_file'],
        scaler_file=r['artifacts']['scaler_file'], calibrator_file=r['artifacts']['calibrator_file'],
        calibration=r['calibration']['resolved_method'],
        imputer=dict(rule=shc['missing_value_rule'], nonzero_fills=sum(1 for v in fills.values() if v != 0), feature_count=len(fills)),
        scale_clip=ts['config'].get('scale_clip'), context_rows=ts['context_rows'], window_size=ts['window_size'],
        lag_steps=ts.get('lag_steps'), delta_feature_count=ts.get('delta_feature_count'), threshold=r['threshold']['value'],
        runtime_history_bars_inferred=infer_required_history_bars(names),
        source_history=str(hist).replace('\\', '/'), source_history_sha256=shc['source_history_sha256'],
        source_row_offset=shc.get('source_row_offset'), producer_module_sha256={m: sha(m) for m in mods},
        counts=dict(by_producer=pd.Series([f['producer'] for f in feats]).value_counts().to_dict(),
                    by_native_timeframe=pd.Series([f['native_timeframe'] for f in feats]).value_counts().to_dict(),
                    by_tolerance_class=pd.Series([f['tolerance_class'] for f in feats]).value_counts().to_dict(),
                    htf_aligned=sum(f['htf_aligned'] for f in feats), fallback=sum(1 for f in feats if f['fallback_path']),
                    unattributed=[f['feature'] for f in feats if f['producer'] == 'unattributed']),
        features=feats))

doc = dict(schema='section4-feature-inventory-v1', status='draft_unlocked', b4_freeze=FREEZE, b4_freeze_sha256=sha(FREEZE),
           attribution_method='exact for non-strategy sets: each feature set built once on 9,000 ES-5m-tagged rows and its new columns recorded; strategy__* -> strategy_signals by name; transforms stripped with the engine regexes',
           native_timeframe_method='name rule: htf_30m_* 30m, htf_1h_* 1h, *weekly*/prior_week 1W, *daily*/prev_day/prior_rth 1D/session, else 5m',
           producer_version_method='no per-producer version string exists; version = sha256 of the producer module in this checkout (A1 collection identity hashes the same sources)',
           tolerance_class_rule='C0 if every finite frozen value is integer-valued (or integer dtype); C3 raw volume scale; C2 untransformed price-unit level by name; else C1. Uses frozen reference data only, before any comparison.',
           models=models)
out.write_text(json.dumps(doc, indent=1, default=str))
for m in models:
    print(m['queue_id'], m['backend'], m['counts']['by_producer'], m['counts']['by_tolerance_class'],
          'htf', m['counts']['htf_aligned'], 'unattr', m['counts']['unattributed'][:4])
