"""Audit saved ES predictions; all outcomes are explicitly hypothetical markouts."""
from pathlib import Path
import json
import re
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
HORIZONS = (3, 6, 12, 20, 120)

class PriceOutcomes:
    def __init__(self):
        self.b = pd.read_csv(OUT/'es_bars_5m.csv')
        self.b['timestamp_utc'] = pd.to_datetime(self.b.timestamp_utc, utc=True)
        self.b['complete'] = self.b.feature_context_json.map(lambda s: json.loads(s or '{}').get('ibkr_is_complete') is not False)
        self.ts = self.b.timestamp_utc
        self.lookup = {str(t):i for i,t in enumerate(self.ts)}
        breaks = self.ts.diff().ne(pd.Timedelta(minutes=5)) | self.b.contract_symbol.ne(self.b.contract_symbol.shift())
        self.seg = breaks.cumsum().to_numpy()
        self.invalid = np.r_[0, (~self.b.complete).cumsum()]
        self.opens=self.b.open.to_numpy(); self.closes=self.b.close.to_numpy()
        self.highs=self.b.high.to_numpy(); self.lows=self.b.low.to_numpy()

    def index(self, timestamp):
        return self.lookup.get(str(pd.Timestamp(timestamp)))

    def valid(self, i, end):
        return i is not None and end<len(self.b) and self.seg[i]==self.seg[end] and self.invalid[end+1]==self.invalid[i]

    def mark(self, timestamp, side, cost_points=0.6625, input_close=None, recorded_at=None):
        i=self.index(timestamp)
        if i is None: return {'bar_found':False}
        d={'bar_found':True,'bar_index':i,'canonical_close':self.closes[i],'bar_complete':bool(self.b.complete.iloc[i])}
        for h in HORIZONS:
            if not self.valid(i,i+h):
                d[f'valid_{h}']=False
                continue
            entry=self.opens[i+1]; exit_price=self.closes[i+h]
            hi=np.max(self.highs[i+1:i+h+1]); lo=np.min(self.lows[i+1:i+h+1])
            gross=side*(exit_price-entry)
            d.update({f'valid_{h}':True,f'next_open_{h}':entry,f'exit_close_{h}':exit_price,
                      f'exit_timestamp_{h}':str(self.ts.iloc[i+h]+pd.Timedelta(minutes=5)),
                      f'gross_points_{h}':gross,f'net_points_{h}':gross-cost_points,
                      f'mfe_points_{h}':max(0, hi-entry if side>0 else entry-lo),
                      f'mae_points_{h}':max(0, entry-lo if side>0 else hi-entry),
                      f'canonical_close_gross_{h}':side*(exit_price-self.closes[i])})
            if input_close is not None and pd.notna(input_close): d[f'input_close_gross_{h}']=side*(exit_price-input_close)
        if recorded_at is not None:
            j=int(self.ts.searchsorted(pd.Timestamp(recorded_at)))
            end=j+11
            if j>i and self.valid(i,end):
                d.update({'availability_valid_12':True,'availability_entry_timestamp':str(self.ts.iloc[j]),
                          'availability_entry_price':self.opens[j],
                          'availability_exit_timestamp':str(self.ts.iloc[end]+pd.Timedelta(minutes=5)),
                          'availability_net_points_12':side*(self.closes[end]-self.opens[j])-cost_points})
            else: d['availability_valid_12']=False
        return d

def summarize(g, h=12, prefix=''):
    col=f'{prefix}net_points_{h}'
    if col not in g: return {'n':0}
    v=g[col].dropna(); valid=g.loc[v.index]
    if not len(v): return {'n':0}
    wins=v[v>0].sum(); losses=-v[v<0].sum()
    d={'n':len(v),'days':valid.timestamp_et.str[:10].nunique(),'win_rate':float((v>0).mean()),
       'net_points':float(v.sum()),'mean_points':float(v.mean()),'median_points':float(v.median()),
       'net_dollars_one_es_per_event':float(v.sum()*50),'profit_factor':float(wins/losses) if losses else np.nan,
       'best_points':float(v.max()),'worst_points':float(v.min())}
    daily=valid.assign(v=v).groupby(valid.timestamp_et.str[:10]).v.sum()
    d['positive_days']=int((daily>0).sum()); d['best_day_points']=float(daily.max())
    if 'calibrated_probability' in valid:
        p=valid.calibrated_probability
        d['mean_probability']=float(p.mean());d['probability_std']=float(p.std())
        d['auc_net_win_proxy']=float(roc_auc_score(v>0,p)) if (v>0).nunique()==2 else np.nan
        d['probability_return_spearman']=float(p.corr(v,method='spearman')) if p.nunique()>1 else np.nan
    return d

def run():
    c=sqlite3.connect(OUT/'es_audit_snapshot.sqlite3');c.row_factory=sqlite3.Row
    prices=PriceOutcomes(); rows=[]; manifests={r['id']:json.loads(r['live_policy_json']) for r in c.execute('select * from runtime_manifests')}
    query='''SELECT p.*,d.decision,d.reasons_json,d.metadata_json decision_metadata,
             f.valid_feature_count,f.feature_values_json
             FROM model_predictions p JOIN signal_decisions d ON d.prediction_id=p.id
             JOIN feature_snapshots f ON f.id=p.feature_snapshot_id ORDER BY p.timestamp_utc,p.id'''
    for r in c.execute(query):
        r=dict(r); meta=json.loads(r.pop('metadata_json') or '{}'); dm=json.loads(r.pop('decision_metadata') or '{}')
        context=meta.get('policy_context',{});fv=json.loads(r.pop('feature_values_json'))
        r.pop('prediction_json');r['strategy']=r['model_id'].split('_')[0].upper()
        r['model_family']=r['model_id'].split('_')[2]
        r['model_kind']='setup' if re.search(r'_setup[1-6]_',r['model_id']) or (r['strategy']=='ICT' and len(r['model_id'].split('_'))>5) else ('meta' if r['model_family']=='meta' else 'family')
        t=pd.Timestamp(r['timestamp_utc']); rt=pd.Timestamp(r['recorded_at_utc'])
        r['timestamp_et']=str(t.tz_convert('America/New_York'));r['hour_et']=t.tz_convert('America/New_York').hour
        r['lag_after_bar_close_minutes']=(rt-t).total_seconds()/60-5
        r['shadow_mode']=meta.get('shadow_mode'); r['threshold_pass']=r['calibrated_probability']>=r['threshold_applied'] if r['threshold_applied'] is not None else False
        r['selected_features']=len(fv);r['missing_features']=sum(v is None or (isinstance(v,float) and not np.isfinite(v)) for v in fv.values())
        r['missing_feature_fraction']=r['missing_features']/len(fv) if fv else np.nan
        for k in ['close','open','high','low','atr_14','session_regime','composite_regime','stress_regime','trend_regime','vol_regime',
                  'ibkr_is_complete','ibkr_market_data_type','frvp_setup_type','frvp_setup_side','frvp_setup_confidence_rule',
                  'ict_setup_type','ict_setup_side','ict_setup_family','ict_is_rth','ict_prior_rth_high','ict_prior_rth_low',
                  'ict_prior_week_high','ict_prior_week_low','ict_rth_open','frvp_dist_ib_high_atr','frvp_dist_ib_low_atr']:
            r['context_'+k]=context.get(k)
        session=context.get('session_regime','off_hours')
        spread={'london':1,'overlap':1,'new_york':1,'asia':1.5,'off_hours':2}.get(session,2)
        r['assumed_cost_points']=(2*spread+0.25+0.40)*0.25
        side=1 if r['direction']=='long' else -1
        r.update(prices.mark(t,side,r['assumed_cost_points'],context.get('close'),rt))
        r['input_close_minus_canonical']=context.get('close',np.nan)-r.get('canonical_close',np.nan)
        policy=manifests[r['runtime_manifest_id']]; abst=policy.get('abstain_policy',{})
        blocked=[]
        if abst.get('enabled'):
            composite=context.get('composite_regime');stress=context.get('stress_regime')
            if abst.get('abstain_high_stress') and stress=='high':blocked.append('high_stress')
            if abst.get('abstain_off_hours') and session=='off_hours':blocked.append('off_hours')
            if session in abst.get('abstain_session_regimes',[]):blocked.append('session')
            if composite in abst.get('abstain_composite_regimes',[]):blocked.append('composite')
            if [composite,session] in abst.get('abstain_composite_session_pairs',[]):blocked.append('composite_session')
            if [composite,stress] in abst.get('abstain_composite_stress_pairs',[]):blocked.append('composite_stress')
        r['static_policy_blocks']='|'.join(blocked)
        rows.append(r)
    p=pd.DataFrame(rows);p.to_csv(OUT/'prediction_outcomes.csv',index=False)
    summaries=[]; bins=[]
    for mid,g in p.groupby('model_id'):
        base={'model_id':mid,'model_kind':g.model_kind.iloc[0],'strategy':g.strategy.iloc[0],
              'predictions':len(g),'threshold_passes':int(g.threshold_pass.sum()),'first_et':g.timestamp_et.min(),'last_et':g.timestamp_et.max(),
              'mean_probability_all':g.calibrated_probability.mean(),'min_probability':g.calibrated_probability.min(),'max_probability':g.calibrated_probability.max(),
              'distinct_probabilities':g.calibrated_probability.nunique(),'raw_calibrated_max_difference':(g.raw_score-g.calibrated_probability).abs().max(),
              'median_delay_minutes':g.lag_after_bar_close_minutes.median(),'missing_features_mean':g.missing_feature_fraction.mean()}
        for scope,sub in [('all_bar_diagnostic',g),('threshold_only_diagnostic',g[g.threshold_pass])]:
            summaries.append(base|{'scope':scope}|summarize(sub))
            if len(sub):
                buckets=pd.cut(sub.calibrated_probability,[-0.001,.2,.3,.4,.5,.6,.7,.8,1.001])
                for bucket,bg in sub.groupby(buckets,observed=True): bins.append({'model_id':mid,'scope':scope,'probability_bin':str(bucket)}|summarize(bg))
    pd.DataFrame(summaries).to_csv(OUT/'model_all_bar_diagnostics.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'model_all_bar_probability_bins.csv',index=False)
    quality={'predictions':len(p),'unique_model_bar':len(p[['model_id','timestamp_utc']].drop_duplicates()),
             'unique_predicted_bars':p.timestamp_utc.nunique(),'decision_counts':p.decision.value_counts().to_dict(),
             'after_close_lag_minutes':p.lag_after_bar_close_minutes.describe(percentiles=[.05,.25,.5,.75,.95,.99]).to_dict(),
             'input_close_differs_from_final_bar_rows':int(p.input_close_minus_canonical.abs().gt(1e-8).sum()),
             'input_close_max_difference_points':p.input_close_minus_canonical.abs().max(),
             'input_complete_flag':p.context_ibkr_is_complete.astype(str).value_counts().to_dict(),
             'feed_flag':p.context_ibkr_market_data_type.fillna('unspecified').value_counts().to_dict()}
    (OUT/'prediction_quality.json').write_text(json.dumps(quality,indent=2,default=str))
    print(json.dumps(quality,indent=2))

if __name__=='__main__':run()
