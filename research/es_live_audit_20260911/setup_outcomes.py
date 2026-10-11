"""Comparable, explicitly hypothetical forward outcomes for every retained setup."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from model_outcomes import OUT, PriceOutcomes, summarize, HORIZONS

def bracket(prices,r,max_h=20):
    if r.strategy!='ICT':return {}
    i=prices.index(r.timestamp_utc);side=int(r.setup_side)
    entry=float(r.entry_price);stop=float(r.stop_price);target=float(r.target_price)
    if not all(np.isfinite([entry,stop,target])):return {'bracket_status':'missing_geometry'}
    if side*(entry-stop)<=0 or side*(target-entry)<=0:return {'bracket_status':'invalid_geometry'}
    for h in range(1,max_h+1):
        j=i+h
        if not prices.valid(i,j):return {'bracket_status':'censored_gap_or_incomplete'}
        op=prices.opens[j];hi=prices.highs[j];lo=prices.lows[j]
        stop_hit=lo<=stop if side>0 else hi>=stop
        target_hit=hi>=target if side>0 else lo<=target
        if stop_hit or target_hit:
            # Bar opening through a level determines precedence; else ambiguous => stop first.
            stop_open=side*(op-stop)<=0;target_open=side*(op-target)>=0
            if stop_open:exit_price=op;status='stop_gap'
            elif target_open:exit_price=target;status='target_gap_conservative_fill'
            elif stop_hit:exit_price=stop;status='stop_ambiguous' if target_hit else 'stop'
            else:exit_price=target;status='target'
            return {'bracket_status':status,'bracket_bars':h,'bracket_gross_points':side*(exit_price-entry),
                    'bracket_net_points':side*(exit_price-entry)-1,'bracket_exit_price':exit_price}
    return {'bracket_status':'timeout','bracket_bars':max_h,'bracket_gross_points':side*(prices.closes[i+max_h]-entry),
            'bracket_net_points':side*(prices.closes[i+max_h]-entry)-1,'bracket_exit_price':prices.closes[i+max_h]}

def group_summary(frame,cols,horizons=HORIZONS):
    records=[]
    for key,g in frame.groupby(cols,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for h in horizons:
            records.append(dict(zip(cols,key))|{'horizon_minutes':h*5,'marked_events':len(g)}|summarize(g,h))
    return pd.DataFrame(records)

def cluster_interval(g):
    g=g.dropna(subset=['net_points_12'])
    daily=g.groupby(g.timestamp_et.str[:10]).net_points_12.agg(['sum','count'])
    if len(daily)<2:return [None,None]
    rng=np.random.default_rng(20260911); samples=rng.integers(0,len(daily),(5000,len(daily)))
    means=daily['sum'].to_numpy()[samples].sum(axis=1)/daily['count'].to_numpy()[samples].sum(axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def run():
    prices=PriceOutcomes();s=pd.read_csv(OUT/'setup_events.csv'); rows=[]
    for r in s.itertuples(index=False):
        d=r._asdict(); t=pd.Timestamp(r.timestamp_utc)
        d['hour_et']=t.tz_convert('America/New_York').hour
        recorded=r.original_mark_recorded_at_utc if pd.notna(r.original_mark_recorded_at_utc) else r.first_model_recorded_at_utc
        d['availability_time_basis']='original_mark_record' if pd.notna(r.original_mark_recorded_at_utc) else 'same_bar_first_model_record_proxy'
        d.update(prices.mark(t,r.setup_side,1.,r.close,recorded if pd.notna(recorded) else None))
        d.update(bracket(prices,r));rows.append(d)
    s=pd.DataFrame(rows);s.to_csv(OUT/'setup_outcomes.csv',index=False)
    group_summary(s,['strategy']).to_csv(OUT/'setup_strategy_summary.csv',index=False)
    group_summary(s,['strategy','setup_type','setup_side','setup_name']).to_csv(OUT/'setup_type_summary.csv',index=False)
    group_summary(s,['strategy','hour_et'],(12,)).to_csv(OUT/'setup_hourly_summary.csv',index=False)
    group_summary(s,['hour_et'],(12,)).to_csv(OUT/'setup_hourly_combined.csv',index=False)
    group_summary(s,['strategy','setup_family'],(12,)).to_csv(OUT/'setup_family_summary.csv',index=False)
    p=pd.read_csv(OUT/'prediction_outcomes.csv',low_memory=False)
    m=pd.read_csv(OUT/'setup_model_probabilities.csv')
    # Agent's route_match requires side + family for pooled model, or exact type for specialist; meta accepts either family.
    matched=m[m.route_match].copy()
    extra=p[['id','model_kind','static_policy_blocks','context_session_regime','context_composite_regime','context_stress_regime','lag_after_bar_close_minutes']].rename(columns={'id':'prediction_id'})
    matched=matched.merge(extra,on='prediction_id',how='left')
    outcome_cols=[col for col in s if col not in matched or col=='setup_id']
    matched=matched.merge(s[outcome_cols],on='setup_id',how='left')
    # Model-specific availability uses that model's own recording time, not an earlier sibling model.
    for idx,r in matched.iterrows():
        delayed=prices.mark(r.timestamp_utc,int(r.setup_side),1.,r.close,r.recorded_at_utc)
        for key,value in delayed.items():
            if key.startswith('availability_'):matched.loc[idx,key]=value
        matched.loc[idx,'availability_time_basis']='this_model_recorded_at'
    matched['calibrated_probability']=matched.probability
    matched.to_csv(OUT/'matched_model_setup_outcomes.csv',index=False)
    summary=[]
    for model,g in matched.groupby('model_id'):
        for subset,frame in [('all_matching_setups',g),('passed_threshold',g[g.threshold_pass]),('below_threshold',g[~g.threshold_pass])]:
            for h in (3,6,12,20):
                summary.append({'model_id':model,'model_kind':g.model_kind.iloc[0],'strategy':g.strategy.iloc[0],
                                'subset':subset,'marked_events':len(frame),'horizon_minutes':h*5}|summarize(frame,h))
    pd.DataFrame(summary).to_csv(OUT/'matched_model_summary.csv',index=False)
    bins=[]
    for model,g in matched.groupby('model_id'):
        for bucket,bg in g.groupby(pd.cut(g.probability,[-.001,.2,.3,.4,.5,.6,.7,.8,1.001]),observed=True):
            bins.append({'model_id':model,'probability_bin':str(bucket)}|summarize(bg))
    pd.DataFrame(bins).to_csv(OUT/'matched_probability_bins.csv',index=False)
    filters=[]; filtered_events=[]
    for (strategy,kind),g in matched.groupby(['strategy','model_kind']):
        for rule,keep in [('threshold',g.threshold_pass),('threshold_and_static_policy',g.threshold_pass & g.static_policy_blocks.fillna('').eq(''))]:
            events=s[s.setup_id.isin(g.loc[keep,'setup_id'])].copy()
            events['model_kind']=kind;events['filter_rule']=rule;filtered_events.append(events)
            filters.append({'strategy':strategy,'model_kind':kind,'filter_rule':rule,'retained_marked_events':len(events)}|summarize(events))
    pd.DataFrame(filters).to_csv(OUT/'unique_setup_filter_summary.csv',index=False)
    pd.concat(filtered_events,ignore_index=True).to_csv(OUT/'filtered_setup_events.csv',index=False)
    nonoverlap=[]
    for strategy,g in s.groupby('strategy'):
        last_exit=None
        for _,r in g.sort_values('timestamp_utc').iterrows():
            if pd.isna(r.get('net_points_12')):continue
            entry=pd.Timestamp(r.timestamp_utc)+pd.Timedelta(minutes=5)
            if last_exit is not None and entry<last_exit:continue
            nonoverlap.append(r);last_exit=pd.Timestamp(r.exit_timestamp_12)
    no=pd.DataFrame(nonoverlap);no.to_csv(OUT/'nonoverlapping_setup_events_60m.csv',index=False)
    sensitivity=[]
    for strategy,g in s.groupby('strategy'):
        ci=cluster_interval(g)
        for cost in [0,.5,1,2]:
            v=g.gross_points_12.dropna()-cost
            sensitivity.append({'strategy':strategy,'cost_points':cost,'n':len(v),'net_points':v.sum(),'net_dollars':v.sum()*50,'mean_points':v.mean(),'win_rate':v.gt(0).mean(),
                                'day_cluster_95ci_mean_at_1pt_low':ci[0],'day_cluster_95ci_mean_at_1pt_high':ci[1]})
    pd.DataFrame(sensitivity).to_csv(OUT/'setup_cost_sensitivity.csv',index=False)
    day=s.copy();day['date_et']=day.timestamp_et.str[:10]
    group_summary(day,['strategy','date_et'],(12,)).to_csv(OUT/'setup_daily_summary.csv',index=False)
    print('STRATEGY',group_summary(s,['strategy'],(12,)).to_string(index=False))
    print('NONOVERLAP',group_summary(no,['strategy'],(12,)).to_string(index=False))
    print('FILTERS',pd.DataFrame(filters).to_string(index=False))
    print('HOURLY',group_summary(s,['hour_et'],(12,)).to_string(index=False))

if __name__=='__main__':run()
