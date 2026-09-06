"""Short overlays selected for portfolio complement, not standalone return."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from training import search_meaningful_alpha_combinations as base
from training import short_complement_ledger as engine

OUT=base.ROOT/'research/long_complement_shorts'
CONTEXT=base.ROOT/'research/short_complement_context/report.json'
DESIGN={'version':1,'parent':'G9 + macro coefficient1 (existing long and short sleeves retained)',
        'selection':'2024 only; 2025 and full2026H1 report without reranking',
        'objective':'portfolio complement: stress-Calmar improvement + .5 relative MDD reduction; standalone profitability not required',
        'families':['failed_rebound','sell_ignition','quiet_breakdown','crowded_break','distribution','currency_riskoff','regional_unwind','trend_acceleration'],
        'thresholds':[.5,1.],'hold_hours':[3,6,12],'hedge_coefficients':[.25,.5,1.],
        'execution':'completed-hour predictors, next5m open; short-only overlay capped by currently positive parent net units, native-event rebalances and parent barrier exits',
        'costs':[.0006,.001],'no_frequency_overlap_correlation_gate':True,'live_authorized':False,
        'controls':['no hedge','always-on hedge at each searched coefficient'],
        'report_finalists':'top5 distinct selection paths plus controls',
        'caveat':'all historical periods already exposed; not pristine OOS or guaranteed new independent alpha'}


def pulse_hold(active,hours):
    active=np.asarray(active,bool);p=np.zeros(len(active));event=np.zeros(len(active),bool);remaining=0
    for i,on in enumerate(active):
        if remaining==0 and on:
            remaining=hours;event[i]=True
        p[i]=-1. if remaining else 0.
        if i and p[i]!=p[i-1]:event[i]=True
        remaining=max(remaining-1,0)
    return p,event


def candidate_specs():
    specs=[{'name':'no_hedge','family':'control','coefficient':0.,'hold':0,'threshold':0.}]
    specs += [{'name':f'always_{c}','family':'always','coefficient':c,'hold':1,'threshold':0.} for c in DESIGN['hedge_coefficients']]
    for family in DESIGN['families']:
        for threshold in DESIGN['thresholds']:
            for hold in DESIGN['hold_hours']:
                for coefficient in DESIGN['hedge_coefficients']:
                    specs.append({'name':f'{family}_t{threshold}_h{hold}_w{coefficient}','family':family,'threshold':threshold,'hold':hold,'coefficient':coefficient})
    return specs


def raw_short(x,s):
    t=s['threshold'];f=s['family']
    if f=='control':return np.zeros(len(x),bool)
    if f=='always':return np.ones(len(x),bool)
    flow=x.flow6
    predicates={
        'failed_rebound':(x.mom168<-.5)&(x.z6>t)&(flow<-.01),
        'sell_ignition':(x.mom6<-t)&(x.volratio>1.2)&(x.volume_ratio>1.1)&(flow<-.02),
        'quiet_breakdown':(x.breakout<0)&(x.volratio<1.2)&(x.mom24<-t)&(flow<-.01),
        'crowded_break':(x.funding>0.00005)&(x.mom6<-t)&(flow<-.015),
        'distribution':(x.z24>t)&(x.mom6<0)&(flow<-.02),
        'currency_riskoff':(x.dxy_change6>0)&(x.mom24<-t)&(flow<-.01),
        'regional_unwind':(x.kimchi_premium_change6<0)&(x.mom6<-t)&(flow<-.01),
        'trend_acceleration':(x.mom168<-.5)&(x.mom24<-t)&(x.mom6<-.5)&(flow<-.01)}
    return predicates[f].fillna(False).to_numpy(bool)


def load_window(window):
    r=json.loads(CONTEXT.read_text());receipt=r['artifacts'][window];path=base.ROOT/receipt['path']
    if base.sha(path)!=receipt['sha256']:raise RuntimeError('Context file drift')
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    x=pd.DataFrame(a['features'],columns=a['feature_names'],index=pd.DatetimeIndex(a['feature_date']))
    d={k:a[k] for k in ['date','end_date','open','end','high','low','funding']}
    return a,x,d


def make_targets(a,x,specs):
    n=len(a['date']);p=np.zeros((n,len(specs)));events=np.zeros_like(p,bool)
    rowmap=a['feature_row_for_5m'];observed=np.flatnonzero(rowmap>=0)
    for j,s in enumerate(specs):
        if s['family']=='control':continue
        hp,he=pulse_hold(raw_short(x,s),s['hold'])
        sparse=np.full(n,np.nan);sparse[observed]=hp[rowmap[observed]]
        p[:,j]=pd.Series(sparse).ffill().fillna(0).to_numpy()
        events[observed,j]=he[rowmap[observed]]
    return p,events


def evaluate(window,specs,cost,long_only=False):
    a,x,d=load_window(window);hp,he=make_targets(a,x,specs)
    parent=np.maximum(a['targets'],0) if long_only else a['targets']
    p=np.column_stack([parent,np.zeros(len(d['date']))]);e=np.column_stack([a['events'],np.zeros(len(d['date']),bool)])
    b=np.column_stack([a['barriers'],np.full(len(d['date']),np.nan)])
    weights=np.column_stack([np.tile(a['weights'],(len(specs),1)),[s['coefficient'] for s in specs]])
    rows,paths=engine.simulate(d,p,e,b,weights,list(a['sleeve_names'])+['short_complement'],cost=cost,hedge_targets=hp,hedge_events=he)
    return rows,paths,hp


def register():
    context=json.loads(CONTEXT.read_text())
    payload={'design':DESIGN,'code_hash':base.sha(__file__),'engine_hash':base.sha(engine.__file__),
             'context_hash':base.sha(CONTEXT),'specs':candidate_specs(),'context_artifacts':context['artifacts']}
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('Frozen short search drift')
    base.write_json(path,payload);return payload


def run():
    registration=register();specs=candidate_specs();rows,paths,hp=evaluate('2024',specs,.001)
    control=rows[0]
    def score(row):
        if row['insolvent']:return -1e9
        return row['calmar']/max(abs(control['calmar']),1e-9)-1 + .5*(1-row['mdd_pct']/max(control['mdd_pct'],1e-9))
    order=sorted(range(4,len(rows)),key=lambda i:score(rows[i]),reverse=True)
    selected=[];seen=set()
    for i in order:
        key=hashlib.sha256(paths[:,i].tobytes()).hexdigest()
        if key in seen:continue
        seen.add(key);selected.append(i)
        if len(selected)==5:break
    freeze={'selection_window':'2024','top':[{'spec':specs[i],'score':score(rows[i]),'metrics':rows[i]} for i in selected],
            'control':control,'standalone_profit_gate':False,'report_reranking':False}
    base.write_json(OUT/'selection_freeze.json',freeze)
    chosen=list(range(4))+selected;final=[specs[i] for i in chosen]
    reports={}
    for window in ['2024','2025','2026H1']:
        reports[window]={}
        for cost in DESIGN['costs']:
            rr,pp,_=evaluate(window,final,cost);ref=rr[0]
            ref_returns=np.diff(np.r_[1.,pp[:,0]])/np.r_[1.,pp[:-1,0]]
            for j,row in enumerate(rr):
                row['return_delta_pp']=row['return_pct']-ref['return_pct']
                row['mdd_delta_pp']=row['mdd_pct']-ref['mdd_pct']
                row['calmar_delta']=row['calmar']-ref['calmar']
                returns=np.diff(np.r_[1.,pp[:,j]])/np.r_[1.,pp[:-1,j]]
                row['increment_on_parent_down_bars_bps']=float(np.sum((returns-ref_returns)[ref_returns<0])*10000)
            reports[window][str(cost)]={s['name']:row for s,row in zip(final,rr)}
    long_only_reports={}
    for window in ['2024','2025','2026H1']:
        long_only_reports[window]={}
        for cost in DESIGN['costs']:
            lr,_,_=evaluate(window,final,cost,long_only=True)
            long_only_reports[window][str(cost)]={s['name']:row for s,row in zip(final,lr)}
    result={'registration':registration,'count':len(specs),'selection_freeze':freeze,
            'inventory':[{'spec':specs[i],'score':score(rows[i]),'metrics':rows[i]} for i in order],
            'reports':reports,'long_only_reports':long_only_reports,'live_enabled':False,
            'interpretation':'Same-instrument net hedge is a dynamic reduction of net long exposure, not independent short alpha when flat.',
            'limitations':['Exposed periods, one-year allocation selection.', 'Bar/subset conservative MDD not tick ordering.',
                           'Down-bar increment is descriptive path attribution, not an independent additive cash PnL series.']}
    base.write_json(OUT/'report.json',result)
    for s in final:
        print(s['name'])
        for window in reports:
            r=reports[window]['0.0006'][s['name']]
            print(window,{k:r[k] for k in ['return_pct','mdd_pct','return_delta_pp','mdd_delta_pp','hedge_entries']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    register() if a.freeze else run()
