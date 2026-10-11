import json,sys
sys.path.insert(0,'.')
from features.causal_contract import v2_predictor_excluded, V2_DROPPED_STRATEGIES
from features.feature_sets.strategy_signals import slugify_strategy_name
dropped=tuple(f'strategy__{slugify_strategy_name(s)}__' for s in V2_DROPPED_STRATEGIES)
pairs={'frvp_july':'b2_frvp_july_prepared_20260921_v1','ict':'b2_ict_prepared_20260921_v1','ict_specialists':'b2_ict_specialists_prepared_20260921_v1',
       'frvp_setups':'b2_frvp_setups_prepared_20260921_v1','ote':'b2_ote_prepared_20260921_v1'}
out={}; problems=[]
for g,old in pairs.items():
    new=f'b2_{g}_prepared_v2_20261006'
    so=json.load(open(f'artifacts/{old}/prepared/summary.json'))['targets']; sn=json.load(open(f'artifacts/{new}/prepared/summary.json'))['targets']
    mo=json.load(open(f'artifacts/{old}/manifest.json')); mn=json.load(open(f'artifacts/{new}/manifest.json'))
    print(f'== {g}: rows {mo.get("rows")} -> {mn.get("rows")}')
    for t in sn:
        a,b=so.get(t,{}),sn[t]
        feats=json.load(open(f'artifacts/{new}/prepared/{t}/features.json'))['features']
        oldf=set(json.load(open(f'artifacts/{old}/prepared/{t}/features.json'))['features'])
        bad=[f for f in feats if v2_predictor_excluded(f) or f.startswith(dropped) or f.startswith('htf_confluence_')]
        if bad: problems.append((g,t,bad[:5]))
        same=(a.get('usable_rows'),a.get('positive_count'))==(b.get('usable_rows'),b.get('positive_count'))
        if not same: problems.append((g,t,'rows/positives changed',a.get('usable_rows'),b.get('usable_rows'),a.get('positive_count'),b.get('positive_count')))
        print(f'  {t}: usable {a.get("usable_rows")}->{b.get("usable_rows")} pos {a.get("positive_count")}->{b.get("positive_count")} '
              f'sel {a.get("selected_features")}->{b.get("selected_features")} overlap {len(oldf & set(feats))} grade {b.get("readiness_grade")} bad {len(bad)}')
        out[f'{g}/{t}']=dict(old=a,new=b,overlap=len(oldf&set(feats)),excluded_selected=bad)
json.dump(out,open(sys.argv[1] if len(sys.argv) > 1 else "docs/audits/b2_v2_prepared_check_20261006.json",'w'),indent=1,default=str)
print('PROBLEMS', problems if problems else 'none')
