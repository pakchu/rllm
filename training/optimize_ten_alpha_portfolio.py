"""Ten-sleeve portfolio search including two standalone shorts, research only."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from training import search_meaningful_alpha_combinations as base
from training import g9_joint_net_ledger as ledger
from training import optimize_g9_plus_added_alphas as old

OUT=base.ROOT/'research/ten_alpha_optimization'
CONTEXT=base.ROOT/'research/ten_alpha_context/report.json'
NAMES=old.NAMES+['dollar_rally_short','failed_rebound_short']
G9=np.r_[old.CONTROL_G9,0.,0.]
PARENT=G9.copy();PARENT[5]=1.
LIMITS=np.array([2.,1.5,2.,2.,2.,1.,1.,1.,1.,1.])
DESIGN={'version':1,'names':NAMES,'seed':20260907,'selection':'2024 only,10bp Calmar then return; reject insolvency',
        'reports':['2025','2026H1','recent_Jun1_Sep4','since_July','September1_4'],
        'controls':'G9, G9+macro1, all32 predeclared two-short addition cells around these parents',
        'samples':512,'sample_budgets_notional':[4.5,5.5,6.],'step':.25,'max_coefficients':LIMITS.tolist(),
        'capacity':'Rank7 coefficient<=1.5 (original weight3); new sleeves/shorts<=1x',
        'risk':'signed net cap4.5 after fees at execution events; no cross-sleeve overlap restriction',
        'no_frequency_cost_ratio_gate':True,'selection_finalists':5,'live_enabled':False,
        'caveat':'all windows previously exposed; bounded search, not a global/pristine-OOS optimum'}


def grid():
    rows=[G9.copy(),PARENT.copy()];labels=['g9','g9_macro1']
    for parent,p in [('g9',G9),('g9_macro1',PARENT)]:
        for dollar in [0.,.25,.5,1.]:
            for rebound in [0.,.25,.5,1.]:
                w=p.copy();w[-2:]=[dollar,rebound]
                rows.append(w);labels.append(f'{parent}_d{dollar}_r{rebound}')
    rng=np.random.default_rng(DESIGN['seed']);generated=0
    for attempt in range(50000):
        if generated==DESIGN['samples']:break
        active=rng.choice(10,size=int(rng.integers(2,11)),replace=False)
        w=np.zeros(10);w[active]=rng.multinomial(int(rng.choice([18,22,24])),rng.dirichlet(np.ones(len(active))))/4
        if (w>LIMITS+1e-12).any():continue
        rows.append(w);labels.append(f'seeded_{generated}');generated+=1
    if generated!=DESIGN['samples']:raise RuntimeError('Bounded sampler exhausted')
    seen=set();keep=[]
    for i,w in enumerate(rows):
        if tuple(w) in seen:continue
        seen.add(tuple(w));keep.append(i)
    return np.asarray(rows)[keep],[labels[i] for i in keep]


def register():
    w,labels=grid();paths=[__file__,ledger.__file__,old.__file__,CONTEXT,
                          base.ROOT/'research/legacy_short_september/report.json',base.ROOT/'research/independent_short_september/report.json']
    r={'design':DESIGN,'hashes':{str(p):base.sha(p) for p in paths},'count':len(w),'labels':labels,'weights_hash':hashlib.sha256(w.tobytes()).hexdigest()}
    p=OUT/'design.json'
    if p.exists() and json.loads(p.read_text())!=r:raise RuntimeError('Ten-alpha design drift')
    base.write_json(p,r);return r


def load(window):
    report=json.loads(CONTEXT.read_text());key='full2026H1' if window=='2026H1' else window
    meta=report['artifacts'][key];p=base.ROOT/meta['path']
    if base.sha(p)!=meta['sha256']:raise RuntimeError('Ten-alpha context drift')
    with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    if list(a['names'])!=NAMES:raise RuntimeError('Sleeve order mismatch')
    return ({k:a[k] for k in ['date','end_date','open','end','high','low','funding']},a['targets'],a['events'],a['barriers'])


def add_short_clock(d,trades):
    n=len(d['date']);p=np.zeros(n);e=np.zeros(n,bool);b=np.full(n,np.nan);idx=pd.DatetimeIndex(d['date']);last=-1
    for t in trades:
        a=idx.get_indexer([pd.Timestamp(t['entry_date'])])[0]
        exit_date=pd.Timestamp(t['exit_date']);z=idx.get_indexer([exit_date])[0]
        if z<0 and exit_date==pd.Timestamp(d['end_date'][-1]):z=n
        if a<0 or z<a or a<=last:raise ValueError('Invalid recent short clock')
        hit=bool(t['barrier']);p[a:min(z+int(hit),n)]=-1;e[a]=True
        if hit:
            if z>=n:raise ValueError('Barrier outside window')
            price=float(t['exit_price'])
            if not d['low'][z]<=price<=d['high'][z]:raise ValueError('Barrier outside OHLC')
            b[z]=price
            if z+1<n:e[z+1]=True
        elif z<n:e[z]=True
        last=z
    return p,e,b


def recent_context():
    d,p,e,b,receipt=old.context()
    dollar=json.loads((base.ROOT/'research/legacy_short_september/report.json').read_text())['reports']['recent']['original']['trades']
    rebound=json.loads((base.ROOT/'research/independent_short_september/report.json').read_text())['trades']['recent']
    for trades in [dollar,rebound]:
        pp,ee,bb=add_short_clock(d,trades);p=np.column_stack([p,pp]);e=np.column_stack([e,ee]);b=np.column_stack([b,bb])
    return d,p,e,b,receipt


def evaluate(context,weights,cost,start=None,end=None):
    d,p,e,b=context
    if start is not None:
        mask=(d['date']>=pd.Timestamp(start).to_datetime64())&(d['end_date']<=pd.Timestamp(end).to_datetime64())
        d=base.subset(d,mask);p=p[mask];e=e[mask].copy();b=b[mask];e[0]=True
    return ledger.simulate(d,p,e,b,weights,NAMES,cost=cost,net_cap=4.5)


def run():
    registration=register();w,labels=grid();selection=load('2024');rows,paths=evaluate(selection,w,.001)
    ranked=sorted(range(len(w)),key=lambda i:(not rows[i]['insolvent'],rows[i]['calmar'],rows[i]['return_pct']),reverse=True)
    selected=ranked[:5]
    freeze={'selection':'2024 only','top':[{'label':labels[i],'metrics':rows[i]} for i in selected],'report_reranking':False}
    base.write_json(OUT/'selection_freeze.json',freeze)
    # All anchored addition cells were part of the original candidate grid.
    final=list(dict.fromkeys([i for i,n in enumerate(labels) if not n.startswith('seeded_')]+selected))
    reports={}
    for window in ['2024','2025','2026H1']:
        ctx=selection if window=='2024' else load(window);reports[window]={}
        for cost in [.0006,.001]:
            rr,_=evaluate(ctx,w[final],cost);reports[window][str(cost)]=dict(zip([labels[i] for i in final],rr))
    d,p,e,b,receipt=recent_context();ctx=(d,p,e,b)
    for window,a,z in [('recent','2026-06-01','2026-09-05'),('since_july','2026-07-01','2026-09-05'),('september_only','2026-09-01','2026-09-05')]:
        reports[window]={}
        for cost in [.0006,.001]:
            rr,_=evaluate(ctx,w[final],cost,a,z);reports[window][str(cost)]=dict(zip([labels[i] for i in final],rr))
    result={'registration':registration,'selection_freeze':freeze,'inventory':[{'label':labels[i],**rows[i]} for i in ranked],
            'reports':reports,'recent_receipt':receipt,'live_enabled':False,
            'notes':['2026H1 and recent windows overlap, do not sum returns.','All periods exposed; no pristine OOS.',
                     'Ancillary anchored cells are descriptive; no replacement of the frozen rank1 based on reports.',
                     'Weights are notional/equity coefficients, not capital percentages.']}
    base.write_json(OUT/'report.json',result)
    base.write_json(OUT/'shadow_config.json',{'enabled':False,'live_authorized':False,'research_only':True,
                    'weights_notional':rows[selected[0]]['weights_notional'],'selection_label':labels[selected[0]],'net_cap':4.5,
                    'overlap_allowed':True,'long_short_offset':True,'fee_ratio_gate':False,'frequency_gate':False,
                    'report_hash':base.sha(OUT/'report.json')})
    for window in reports:
        print(window)
        for n in ['g9','g9_macro1',labels[selected[0]]]:print(n,json.dumps(reports[window]['0.0006'][n]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
