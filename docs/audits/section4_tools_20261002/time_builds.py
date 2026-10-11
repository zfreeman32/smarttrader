"""Section 4 prompt 1b: per-build wall time of a stream-union IncrementalFeatureEngine (read-only timing).

Staged manifests = installed base manifest + B4 v2 selected features/lags/context (no model bytes loaded;
the feature engine never loads artifacts). Bars are training-source bars; timing only, no comparison.
Usage: python time_builds.py <ES|EURUSD> <out.json> [endpoints=3] [comma-separated queue ids]
"""
import json, sys, time
from datetime import timezone
from pathlib import Path
import pandas as pd
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.models.loaders import load_live_runtime_manifest

BASES = {2: 'frvp_es_paper_signal_20260816/frvp_long_continuation_xgb_v1', 3: 'frvp_es_paper_signal_20260816/frvp_long_meta_xgb_v1',
         5: 'frvp_es_setup_family_20260829/frvp_long_continuation_setup2_xgb_v1', 10: 'frvp_es_paper_signal_20260816/frvp_long_reversal_xgb_v1',
         12: 'ict_es_paper_signal_20260813/ict_long_meta_xgb_v1', 13: 'ict_es_paper_signal_20260813/ict_long_reversal_xgb_v1',
         14: 'ict_es_paper_signal_20260813/ict_short_continuation_xgb_v1', 15: 'ict_es_paper_signal_20260813/ict_short_meta_xgb_v1',
         16: 'ict_es_paper_signal_20260813/ict_short_reversal_xgb_v1', 18: 'ict_short_setup_family_20260811_audit/ict_short_reversal_ifvg_reversal_xgb_v1',
         19: 'ict_short_setup_family_20260811_audit/ict_short_reversal_sweep_reclaim_xgb_v1', 21: 'long_ote_meta_tcn_champion',
         22: 'long_reversal_tcn_v2_20260525_narrow48', 23: 'long_ote_union_tcn_candidate_20260523', 25: 'short_ote_meta_tcn_champion',
         26: 'short_reversal_xgb_v2_20260525'}


def staged(entry):
    base = load_live_runtime_manifest(ROOT / 'ote_live/runtime_manifests' / BASES[entry['queue_id']] / 'live_runtime_manifest.json')
    ts = json.load(open(ROOT / entry['artifact_root'] / 'training_summary.json'))
    sel = list(ts['selected_feature_names'])
    fm = base.feature_manifest.model_copy(update={'selected_feature_names': sel, 'selected_feature_count': len(sel),
                                                  'lag_steps': list(ts.get('lag_steps') or []),
                                                  'delta_feature_count': int(ts.get('delta_feature_count') or 0)})
    cr = base.context_requirements.model_copy(update={'context_rows': int(ts['context_rows']), 'window_size': int(ts['window_size'])})
    backend = ts['config']['model_type'] if ts['config']['backend'] == 'torch' else ts['config']['backend']
    return base.model_copy(update={'model_id': entry['model_id'], 'backend': backend, 'feature_manifest': fm, 'context_requirements': cr})


def bars(stream, n):
    if stream == 'ES':
        raw = pd.read_csv(ROOT / 'data/futures_data/ES-5m-tagged.csv').tail(n)
        ts = pd.to_datetime(raw.ts_event, utc=True)
        cols = dict(open=raw.open, high=raw.high, low=raw.low, close=raw.close, volume=raw.volume)
        extra = dict(symbol='ES', contract_symbol=raw.contract_symbol)
    else:
        raw = pd.read_csv(ROOT / 'data/currency_data/eurusd-5m.csv').tail(n)
        ts = pd.to_datetime(raw.Date.astype(str) + ' ' + raw.Time, utc=True)
        cols = dict(open=raw.Open, high=raw.High, low=raw.Low, close=raw.Close, volume=raw.Volume)
        extra = dict(symbol='EURUSD', contract_symbol=pd.Series([None] * len(raw), index=raw.index))
    out = []
    for i in range(len(raw)):
        out.append(MarketBar(asset=stream, timeframe='5m', timestamp=ts.iloc[i].to_pydatetime().astimezone(timezone.utc),
                             open=float(cols['open'].iloc[i]), high=float(cols['high'].iloc[i]), low=float(cols['low'].iloc[i]),
                             close=float(cols['close'].iloc[i]), volume=float(cols['volume'].iloc[i]),
                             symbol=extra['symbol'], contract_symbol=extra['contract_symbol'].iloc[i]))
    return out


def main():
    stream, out, k = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3
    freeze = json.load(open(ROOT / 'configs/b4_broad_full_session_shadow_freeze_v2.json'))
    ids = {int(x) for x in sys.argv[4].split(',')} if len(sys.argv) > 4 else None
    entries = [e for e in freeze['frozen_replacements'] if e['instrument'] == stream and (ids is None or e['queue_id'] in ids)]
    engine = IncrementalFeatureEngine([staged(e) for e in entries])
    window = engine.plan.runtime_history_bars + engine.plan.additive_seed_prefix_bars
    series = bars(stream, window + k)
    engine.extend(series[:window])
    builds = []
    for bar in series[window:]:
        engine.ingest_bar(bar)
        t = time.perf_counter()
        frame = engine.build_feature_frame()
        wall = time.perf_counter() - t
        top = sorted(engine.last_build_timings_seconds.items(), key=lambda x: -x[1])[:6]
        builds.append(dict(timestamp=bar.timestamp.isoformat(), wall_seconds=round(wall, 3), rows=len(frame), top_steps=top))
        print(stream, builds[-1]['wall_seconds'], top[:3], flush=True)
    doc = dict(stream=stream, model_ids=[e['queue_id'] for e in entries], runtime_history_bars=engine.plan.runtime_history_bars,
               built_feature_count=len(engine.plan.built_feature_names), builds=builds,
               median_wall_seconds=sorted(b['wall_seconds'] for b in builds)[len(builds) // 2])
    out.write_text(json.dumps(doc, indent=1))
    print('TIMING COMPLETE', stream, doc['median_wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
