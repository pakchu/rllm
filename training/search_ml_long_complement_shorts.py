"""Prequential conditional-downside ML, judged as a long-portfolio complement."""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from training import search_long_complement_shorts as rules
from training import short_complement_ledger as engine
from training import search_meaningful_alpha_combinations as base

OUT=base.ROOT/'research/ml_long_complement_shorts'
DESIGN={'version':1,'fit':'2024H1 for2024H2;2024 for2025;2024-2025 for2026. Labels mature strictly before fit cutoff.',
        'selection':'2024H2 portfolio stress-Calmar +.5 relative MDD reduction; no later reranking',
        'target':'forward6/12h BTC return conditional on observed positive parent target exposure',
        'models':['ridge_alpha100','hgb_depth3_leaf80_iter100_l2_10'],
        'thresholds':'training prediction quantile .1/.2, upper bounded at -.0012',
        'weights':[.25,.5],'hold':'forecast horizon6/12h','seed':20260906,
        'standalone_profit_gate':False,'live_enabled':False,
        'caveat':'all data exposed; annual refits are prequential, not same-year tuning'}


def register():
    payload={'design':DESIGN,'code_hash':base.sha(__file__),'engine_hash':base.sha(engine.__file__),
             'rules_code_hash':base.sha(rules.__file__),'context_hash':base.sha(rules.CONTEXT),
             'predecessor_report':base.sha(rules.OUT/'report.json')}
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('ML design drift')
    base.write_json(path,payload);return payload


def fit_predictions():
    contexts={};parts=[];open_prices=[];exposure=[]
    for window in ['2024','2025','2026H1']:
        a,x,d=rules.load_window(window);contexts[window]=(a,x,d)
        obs=np.flatnonzero(a['feature_row_for_5m']>=0)
        if not np.array_equal(a['feature_row_for_5m'][obs],np.arange(len(x))):raise ValueError('Hourly feature mapping discontinuity')
        previous=np.maximum(obs-1,0)
        net=a['targets'][previous]@a['weights']
        x=x.copy();x['known_parent_net']=net
        parts.append(x);open_prices.extend(d['open'][obs]);exposure.extend(net)
    x=pd.concat(parts);prices=pd.Series(open_prices,index=x.index);exposure=np.array(exposure)
    specs=[{'name':'no_hedge','coefficient':0.,'hold':0}];signals={};notes=[]
    dates=x.index
    for kind in ['ridge','hgb']:
        for horizon in [6,12]:
            future=prices.reindex(dates+pd.Timedelta(hours=horizon)).to_numpy()/prices.to_numpy()-1
            predictions=np.full(len(x),np.nan);thresholds={q:np.full(len(x),np.nan) for q in [.1,.2]}
            for year,cutoff,finish in [(2024,'2024-07-01','2025-01-01'),(2025,'2025-01-01','2026-01-01'),(2026,'2026-01-01','2026-07-01')]:
                cutoff=pd.Timestamp(cutoff);fit=(dates+pd.Timedelta(hours=horizon,minutes=5)<cutoff)&np.isfinite(future)&(exposure>0)
                use=(dates>=cutoff)&(dates<pd.Timestamp(finish))
                if fit.sum()<100:raise ValueError('Insufficient mature long-state training rows')
                model=(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100.)) if kind=='ridge' else
                       make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=100,max_depth=3,min_samples_leaf=80,l2_regularization=10.,early_stopping=False,random_state=DESIGN['seed'])))
                model.fit(x.to_numpy()[fit],future[fit]);predictions[use]=model.predict(x.to_numpy()[use]);train_prediction=model.predict(x.to_numpy()[fit])
                selected_thresholds={}
                for q in [.1,.2]:
                    threshold=min(-.0012,float(np.quantile(train_prediction,q)));thresholds[q][use]=threshold;selected_thresholds[str(q)]=threshold
                notes.append({'kind':kind,'horizon':horizon,'year':year,'fit_rows':int(fit.sum()),'max_label_maturity':str((dates[fit]+pd.Timedelta(hours=horizon,minutes=5)).max()),'cutoff':str(cutoff),'thresholds':selected_thresholds})
            for q in [.1,.2]:
                active=(predictions<thresholds[q])&np.isfinite(predictions)
                for coefficient in [.25,.5]:
                    name=f'{kind}_h{horizon}_q{q}_w{coefficient}'
                    specs.append({'name':name,'kind':kind,'hold':horizon,'q':q,'coefficient':coefficient})
                    signals[name]=pd.Series(active,index=dates)
    return contexts,specs,signals,notes


def evaluate(context,window,specs,signals,cost,long_only=False):
    a,x,d=context;start='2024-07-01' if window=='2024' else str(pd.Timestamp(d['date'][0]))
    mask=d['date']>=np.datetime64(start);hp=np.zeros((len(d['date']),len(specs)));he=np.zeros_like(hp,bool)
    obs=np.flatnonzero(a['feature_row_for_5m']>=0)
    for j,s in enumerate(specs):
        if s['name']=='no_hedge':continue
        active=signals[s['name']].reindex(x.index).fillna(False).to_numpy(bool)
        hourly,event=rules.pulse_hold(active,s['hold'])
        sparse=np.full(len(d['date']),np.nan);sparse[obs]=hourly[a['feature_row_for_5m'][obs]]
        hp[:,j]=pd.Series(sparse).ffill().fillna(0).to_numpy();he[obs,j]=event[a['feature_row_for_5m'][obs]]
    parent=np.maximum(a['targets'],0) if long_only else a['targets']
    p=np.column_stack([parent,np.zeros(len(parent))])[mask]
    e=np.column_stack([a['events'],np.zeros(len(parent),bool)])[mask].copy();e[0]=True
    b=np.column_stack([a['barriers'],np.full(len(parent),np.nan)])[mask]
    w=np.column_stack([np.tile(a['weights'],(len(specs),1)),[s['coefficient'] for s in specs]])
    return engine.simulate(base.subset(d,mask),p,e,b,w,list(a['sleeve_names'])+['short_complement'],cost=cost,hedge_targets=hp[mask],hedge_events=he[mask])


def run():
    registration=register();contexts,specs,signals,notes=fit_predictions();rows,paths=evaluate(contexts['2024'],'2024',specs,signals,.001);control=rows[0]
    def score(r):return r['calmar']/max(abs(control['calmar']),1e-9)-1+.5*(1-r['mdd_pct']/max(control['mdd_pct'],1e-9)) if not r['insolvent'] else -1e9
    order=sorted(range(1,len(specs)),key=lambda i:score(rows[i]),reverse=True);selected=order[:3]
    freeze={'selection':'2024H2','top':[{'spec':specs[i],'score':score(rows[i]),'metrics':rows[i]} for i in selected],'control':control,'report_reranking':False}
    base.write_json(OUT/'selection_freeze.json',freeze);chosen=[specs[0]]+[specs[i] for i in selected]
    reports={};long_only={}
    for window in contexts:
        reports[window]={};long_only[window]={}
        for cost in [.0006,.001]:
            rr,_=evaluate(contexts[window],window,chosen,signals,cost)
            ll,_=evaluate(contexts[window],window,chosen,signals,cost,long_only=True)
            reports[window][str(cost)]=dict(zip([s['name'] for s in chosen],rr));long_only[window][str(cost)]=dict(zip([s['name'] for s in chosen],ll))
    base.write_json(OUT/'report.json',{'registration':registration,'fit_audit':notes,'count':len(specs),'selection_freeze':freeze,
                                    'reports':reports,'long_only_reports':long_only,'live_enabled':False})
    for window in reports:
        print(window)
        for n,row in reports[window]['0.0006'].items():print(n,{k:row[k] for k in ['return_pct','mdd_pct','calmar','hedge_entries']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
