"""Research-only regime strategies and rolling ML with family-balanced mixtures.

Uses the first study's causal features and net-position accounting unchanged.
Select on 2021--2023 (six half-years), never on later historical reports.
Annual ML refits use preceding three calendar years with horizon purging.
"""
from __future__ import annotations
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training import search_meaningful_alpha_combinations as base

OUT=base.ROOT/'research/regime_diverse_combinations'
DESIGN={
    'seed':20260906,
    'selection':'six half-years in 2021--2023, fixed rule before run',
    'reports':'2024,2025,2026H1; previously exposed, no clean OOS claim',
    'ml':'annual refit on trailing three calendar years; maturity purge; no future row in fit; ridge/hgb/extra, 24h target',
    'models':base.DESIGN['models'],
    'mechanisms':['slow trend continuation','aligned-flow continuation','trend pullback reentry','range-bound reversal','funding carry with trend','regime switch trend/reversion','price-flow disagreement reversal'],
    'sizing':'raw max1x and 20% annual vol target using past24h vol clipped [.1,1]',
    'rebalances':[6,24],
    'side_modes':['both','long','short'],
    'rule_horizons':[168,720],
    'thresholds':{'trend':.75,'reversion_z':1.5,'flow_alignment':.02,'ml_return':.0012},
    'family_balance':'best two distinct selection paths per economic family and per ML family',
    'mixtures':'all cross-family pairs at weights .25,.5,.75; equal and inverse-vol aggregate; net before costs',
    'ranking':'mean six half-year annualized Sharpe minus .5*std(Sharpe), plus .25*min(Sharpe) plus .1*selection Calmar',
    'no_cost_ratio_or_frequency_gate':True,
    'finalists':'top five plus best pure-rule candidate, frozen before report-only replay',
    'risk':'net exposure cap1; five-minute conservative intrabar MDD; funding proxy caveat inherited',
    'no_live_changes':True,
}


def register():
    design={'design':DESIGN,'code_sha256':base.sha(__file__),'engine_sha256':base.sha(base.__file__),
            'sources':{'market':base.sha(base.MARKET),'funding':base.sha(base.FUNDING)}}
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=design:raise RuntimeError('Design drift')
    base.write_json(path,design);return design


def annual_masks(dates, horizon, year):
    dates=pd.DatetimeIndex(dates);start=pd.Timestamp(f'{year}-01-01');end=pd.Timestamp(f'{year+1}-01-01')
    train=(dates>=max(pd.Timestamp('2020-03-01'),pd.Timestamp(f'{year-3}-01-01')))&(dates+pd.Timedelta(hours=horizon,minutes=5)<start)
    test=(dates>=start)&(dates<end)
    return train,test


def candidates(x,data):
    signals={};specs={};notes=[]
    size=np.clip(np.divide(.20,x.vol24.to_numpy()*np.sqrt(365.25*24),out=np.ones(len(x)),where=x.vol24.to_numpy()>0),.1,1)
    def add(name,family,raw,rationale):
        raw=np.clip(np.nan_to_num(np.asarray(raw,float)),-1,1)
        for side in ['both','long','short']:
            directional=raw if side=='both' else np.maximum(raw,0) if side=='long' else np.minimum(raw,0)
            for reb in [6,24]:
                for sizing in ['raw','vol20']:
                    n=f'{name}__{side}__reb{reb}__{sizing}'
                    signals[n]=base.hold_signal(directional*(size if sizing=='vol20' else 1),reb,x.index)
                    specs[n]={'family':family,'side':side,'rebalance_hours':reb,'sizing':sizing,'rationale':rationale,'model':family.startswith('ml_')}
    z=x.z24.to_numpy();flow=x.flow6.to_numpy();fund=x.funding.to_numpy()
    for w in [168,720]:
        mom=x[f'mom{w}'].to_numpy();direction=np.sign(mom);trending=np.abs(mom)>.75
        add(f'trend{w}','trend',np.where(trending,direction,0),'slow trend persistence')
        add(f'flow_confirm{w}','flow',np.where(trending&(direction*flow>.02),direction,0),'aggressive volume aligns with slow trend')
        add(f'pullback{w}','pullback',np.where(trending&(direction*z<-.5),direction,0),'temporary displacement against established trend')
        add(f'carry{w}','carry',np.where(trending&(direction*fund<=0),direction,0),'trend with favorable funding; crowding avoided')
        reverse=np.where(np.abs(z)>1.5,-np.sign(z),0)
        add(f'range{w}','range',np.where(~trending,reverse,0),'mean reversion only in low-trend regime')
        add(f'switch{w}','switch',np.where(trending,direction,reverse),'trend continuation or range reversal selected by causal regime')
        add(f'flow_switch{w}','flow_switch',np.where(trending,np.where(direction*flow>0,direction,0),reverse),'flow-confirmed trend branch plus range reversal')
    add('absorption','absorption',np.where((np.abs(z)>1.5)&(np.sign(z)*flow<-.02),np.sign(flow),0),'flow-price disagreement at extreme displacement')
    horizon=24
    target=pd.Series(data['open']).shift(-horizon).to_numpy()/data['open']-1
    for kind in ['ridge','hgb','extra']:
        predicted=np.full(len(x),np.nan)
        for year in range(2021,2027):
            train,test=annual_masks(x.index,horizon,year);train &= np.isfinite(target)
            if not test.any():continue
            kw=DESIGN['models'][kind]
            if kind=='ridge':model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(**kw))
            elif kind=='hgb':model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(**kw,early_stopping=False,random_state=DESIGN['seed']))
            else:model=make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(**kw,n_jobs=2,random_state=DESIGN['seed']))
            model.fit(x.to_numpy()[train],target[train]);predicted[test]=model.predict(x.to_numpy()[test])
            notes.append({'model':kind,'prediction_year':year,'train_rows':int(train.sum()),'last_train_decision':str(x.index[train][-1]),'max_train_maturity':str(x.index[train][-1]+pd.Timedelta(hours=24,minutes=5))})
        add(f'rolling_{kind}','ml_'+kind,np.where(np.abs(predicted)>.0012,np.sign(predicted),0),'annually updated joint causal state model, trailing3y window')
    return signals,specs,notes


def rank(data,p):
    mask=base.window_mask(data,'2021-01-01','2024-01-01')
    overall=base.simulate(base.subset(data,mask),p[mask])
    half=[]
    for y in [2021,2022,2023]:
        for st,en in [(f'{y}-01-01',f'{y}-07-01'),(f'{y}-07-01',f'{y+1}-01-01')]:
            use=base.window_mask(data,st,en);half.append(base.simulate(base.subset(data,use),p[use])['sharpe'])
    sh=np.array(half)
    score=sh.mean(axis=0)-.5*sh.std(axis=0)+.25*sh.min(axis=0)+.1*overall['calmar']
    return score,overall,sh


def run():
    registered=json.loads((OUT/'design.json').read_text())
    if registered!=register():raise RuntimeError('Frozen design changed')
    m,f=base.load_sources();x=base.features(m,f);x,data,receipt=base.execution_blocks(m,f,x)
    signals,specs,notes=candidates(x,data)
    names=list(signals);p=np.column_stack(list(signals.values()));score,stats,half=rank(data,p)
    print('base candidates',len(names),flush=True)
    mask=base.window_mask(data,'2021-01-01','2024-01-01');counts={};seen=set();representatives=[]
    for i in np.argsort(-score,kind='stable'):
        family=specs[names[i]]['family'];fingerprint=hashlib.sha256(p[mask,i].tobytes()).hexdigest()
        if counts.get(family,0)>=2 or fingerprint in seen:continue
        counts[family]=counts.get(family,0)+1;seen.add(fingerprint);representatives.append(int(i))
    for a,i in enumerate(representatives):
        for j in representatives[a+1:]:
            if specs[names[i]]['family']==specs[names[j]]['family']:continue
            for w in [.25,.5,.75]:
                n=f'mixture_{i}_{j}_{w}';signals[n]=w*p[:,i]+(1-w)*p[:,j]
                specs[n]={'family':'portfolio','components':{names[i]:w,names[j]:1-w},'model':specs[names[i]]['model'] or specs[names[j]]['model']}
    # Allocation weights fit on the same declared selection period, never report years.
    representative_stats=base.simulate(base.subset(data,mask),p[mask][:,representatives])
    for mode in ['equal','inversevol']:
        weights=np.ones(len(representatives)) if mode=='equal' else 1/np.maximum(representative_stats['returns'].std(axis=0),1e-6)
        weights/=weights.sum();n='aggregate_'+mode;signals[n]=p[:,representatives]@weights
        specs[n]={'family':'portfolio','components':{names[i]:float(w) for i,w in zip(representatives,weights)},'model':True}
    names=list(signals);p=np.column_stack(list(signals.values()));score,stats,half=rank(data,p)
    order=np.argsort(-score,kind='stable');selected=list(map(int,order[:5]))
    pure=next(int(i) for i in order if not specs[names[i]]['model'])
    if pure not in selected:selected.append(pure)
    freeze={'selection':'2021--2023 only','report_reranking':False,'candidates':len(names),'top':[
        {'name':names[i],'spec':specs[names[i]],'rank_score':float(score[i]),'selection':base.stats_row(stats,i),'six_half_sharpe':half[:,i].tolist()} for i in selected]}
    base.write_json(OUT/'selection_freeze.json',freeze)
    finalist=np.column_stack([p[:,selected],np.ones(len(x)),np.zeros(len(x))]);report_names=[names[i] for i in selected]+['control_long','control_cash']
    reports={}
    for window,st,en in [('report2024','2024-01-01','2025-01-01'),('report2025','2025-01-01','2026-01-01'),('report2026','2026-01-01','2026-06-01'),('combined','2024-01-01','2026-06-01')]:
        use=base.window_mask(data,st,en);reports[window]={}
        for cost in [0.,.0006,.001]:
            metrics=base.simulate(base.subset(data,use),finalist[use],cost=cost,fine=True)
            reports[window][str(cost)]={name:base.stats_row(metrics,i) for i,name in enumerate(report_names)}
    result={'registered':registered,'data_receipt':receipt,'ml_fit_audit':notes,'finalist_freeze':freeze,'reports':reports,
            'inventory':[{'name':names[i],'spec':specs[names[i]],'rank_score':float(score[i]),'selection':base.stats_row(stats,i)} for i in order],
            'live_enabled':False,'limitations':['Prior historical exposure: no clean OOS claim.','All model refits are prequential; earlier report outcomes may enter next calendar-year training as declared, not same-year selection.','Funding marks missing use settlement open proxy.','Conservative intrabar MDD; no tick ordering, liquidation, or capacity model.']}
    base.write_json(OUT/'report.json',result)
    base.write_json(OUT/'research_config.json',{'live_enabled':False,'research_only':True,'winner':freeze['top'][0], 'net_exposure_cap':1.,'overlap_allowed':True,'offset_long_short':True,'cost_ratio_gate':False,'frequency_gate':False})
    for row in freeze['top']:
        name=row['name'];print(name,json.dumps(row['spec']),flush=True)
        for w in reports:print(w,json.dumps(reports[w]['0.0006'][name]),flush=True)
    print('candidates',len(names),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:register()
    else:run()
