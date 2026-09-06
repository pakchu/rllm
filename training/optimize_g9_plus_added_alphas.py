"""Frozen eight-sleeve G9 plus additional-alpha net allocation research."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training import evaluate_added_alpha_september as new
from training import g9_joint_net_ledger as ledger
from training import search_meaningful_alpha_combinations as base

OUT=base.ROOT/'research/g9_added_alpha_optimization'
SOURCE=base.ROOT/'research/g9_september_inputs/report.json'
G9=['fresh_kimchi_fx','frozen_annual_rank7','rex_taker_low_range_position','cand_rex_veto_7','markov_transition_long']
NAMES=G9+['macro_flow','oi_pullback','regional_trend']
CONTROL_G9=np.array([1.,1.5,.2,.8,1.,0,0,0])
CONTROL_LIVE=np.array([1.,1.,.2,.8,1.,0,0,0])
DESIGN={'version':1,'names':NAMES,'seed':20260906,'random_candidates':512,
        'selection':['2026-06-01','2026-07-01'],'reports_end':new.END,
        'ranking':'June 10bp Calmar, reject insolvency only; no report reranking',
        'universe':'8 standalone, G8/G9 controls, fixed added-alpha mixes, G9+new local grid,512 seeded nonnegative allocations',
        'random_budgets_notional':[4.5,6.0],'random_step':.25,
        'risk':'net signed BTC cap4.5 at active events; gross budget not a risk measure; overlap allowed',
        'units':'weights are notional/equity; G9 original .5x sleeve convention converted exactly',
        'costs':[0.,.0006,.001], 'live_authorized':False,
        'limitation':'exposed June selection; no clean OOS, no capacity/liquidation/latency model'}


def allocation_grid():
    weights=[CONTROL_LIVE,CONTROL_G9]
    labels=['repository_live_g8','frozen_g9']
    for i,name in enumerate(NAMES):
        row=np.zeros(8);row[i]=1.;weights.append(row);labels.append('standalone_'+name)
    for a,b in [(.8,.2),(.6,.4),(.5,.5)]:
        row=np.zeros(8);row[5:7]=[a,b];weights.append(row);labels.append(f'new_macro{a}_oi{b}')
    for scaling in [.5,.75,1.]:
        for i in range(5,8):
            for weight in [.25,.5,1.]:
                row=CONTROL_G9*scaling;row[i]=weight;weights.append(row);labels.append(f'g9x{scaling}_{NAMES[i]}{weight}')
    rng=np.random.default_rng(DESIGN['seed'])
    for i in range(DESIGN['random_candidates']):
        active=rng.choice(8,size=int(rng.integers(2,9)),replace=False)
        shares=rng.dirichlet(np.ones(len(active)))
        row=np.zeros(8);row[active]=rng.multinomial(int(rng.choice([18,24])),shares)/4
        weights.append(row);labels.append(f'seeded_{i}')
    seen=set();keep=[]
    for i,w in enumerate(weights):
        key=tuple(w)
        if key not in seen:seen.add(key);keep.append(i)
    return np.array(weights)[keep],[labels[i] for i in keep]


def clock_arrays(report,dates):
    idx=pd.DatetimeIndex(dates);n=len(idx)
    targets=np.zeros((n,5));events=np.zeros((n,5),bool);barriers=np.full((n,5),np.nan)
    for j,name in enumerate(G9):
        trades=report['sleeves'][name]['trades']
        last=-1
        for t in trades:
            entry=pd.Timestamp(t['entry_date']);exit_=pd.Timestamp(t['exit_date'])
            if entry.tzinfo is not None:entry=entry.tz_convert(None)
            if exit_.tzinfo is not None:exit_=exit_.tz_convert(None)
            if entry<idx[0] or exit_>idx[-1]:continue
            a=idx.get_indexer([entry])[0];b=idx.get_indexer([exit_])[0]
            if a<0 or b<a or a<=last:raise ValueError(f'{name}: invalid/overlapping clock')
            if str(t['side']).upper() not in ['LONG','SHORT','1','-1']:raise ValueError('Unknown trade side')
            if t['exit_kind'] not in ['open','barrier']:raise ValueError('Unknown exit type')
            if not np.isfinite(float(t['exit_price'])) or float(t['exit_price'])<=0:raise ValueError('Invalid exit price')
            side=1 if str(t['side']).upper() in ['LONG','1'] else -1
            barrier=t['exit_kind']=='barrier'
            targets[a:b+int(barrier),j]=side;events[a,j]=True
            if barrier:
                barriers[b,j]=float(t['exit_price'])
                if b+1<n:events[b+1,j]=True
            else:events[b,j]=True
            last=b
    return targets,events,barriers


def register():
    w,labels=allocation_grid()
    paths=[__file__,ledger.__file__,new.__file__,SOURCE]
    payload={'design':DESIGN,'hashes':{str(p):base.sha(p) for p in paths},
             'weights_hash':hashlib.sha256(w.tobytes()).hexdigest(),'labels':labels,'count':len(w)}
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('Frozen allocation drift')
    base.write_json(path,payload);return payload


def context():
    report=json.loads(SOURCE.read_text())
    if not report.get('passed',False):raise RuntimeError('G9 source closure did not pass')
    d,p,e,receipt=new.context()
    path=Path(report['market_csv'])
    if base.sha(path)!=report['market_sha256']:raise RuntimeError('G9 market artifact drift')
    market=pd.read_csv(path);market['date']=pd.to_datetime(market.date,utc=True).dt.tz_convert(None)
    aligned=market.set_index('date').reindex(pd.DatetimeIndex(d['date']))
    for col in ['open','high','low']:
        if not np.allclose(aligned[col].to_numpy(float),d[col],rtol=0,atol=1e-8):raise RuntimeError(f'G9/new market {col} mismatch')
    for name in G9:
        for trade in report['sleeves'][name]['trades']:
            if trade['exit_kind']=='open':
                timestamp=pd.Timestamp(trade['exit_date'])
                if timestamp.tzinfo is not None:timestamp=timestamp.tz_convert(None)
                if timestamp in aligned.index and not np.isclose(float(trade['exit_price']),float(aligned.loc[timestamp,'open']),rtol=0,atol=1e-8):
                    raise RuntimeError('Fixed-hold exit price mismatch')
    oldp,olde,barriers=clock_arrays(report,d['date'])
    for j in range(5):
        hit=np.isfinite(barriers[:,j])
        if np.any((barriers[hit,j]<d['low'][hit])|(barriers[hit,j]>d['high'][hit])):raise RuntimeError('Barrier outside bar range')
    return d,np.column_stack([oldp,p]),np.column_stack([olde,e]),np.column_stack([barriers,np.full_like(p,np.nan)]),receipt


def run():
    reg=register();d,p,e,b,receipt=context();w,labels=allocation_grid()
    def evaluate(start,end,weights,cost):
        mask=(d['date']>=pd.Timestamp(start).tz_localize(None).to_datetime64())&(d['end_date']<=pd.Timestamp(end).tz_localize(None).to_datetime64())
        ee=e[mask].copy();ee[0]=True
        return ledger.simulate(base.subset(d,mask),p[mask],ee,b[mask],weights,NAMES,cost=cost)
    selection,_=evaluate(*DESIGN['selection'],w,.001)
    order=sorted(range(len(w)),key=lambda i:(not selection[i]['insolvent'],selection[i]['calmar'],selection[i]['return_pct']),reverse=True)
    selected=order[:5]
    freeze={'selected':[{'label':labels[i],**selection[i]} for i in selected],'indices':selected,'report_reranking':False}
    base.write_json(OUT/'selection_freeze.json',freeze)
    final=list(dict.fromkeys([0,1]+list(range(2,13))+selected))
    reports={}
    for window,(a,z) in {'common':(new.START,new.END),'july_to_september':('2026-07-01',new.END),'september_only':('2026-09-01',new.END)}.items():
        reports[window]={}
        for cost in DESIGN['costs']:
            rows,_=evaluate(a,z,w[final],cost)
            reports[window][str(cost)]=dict(zip([labels[i] for i in final],rows))
    result={'registration':reg,'source_receipt':receipt,'selection_freeze':freeze,'reports':reports,
            'inventory':[{'label':labels[i],**selection[i]} for i in order],'live_enabled':False,
            'limitations':['G9 runtime regeneration, not a frozen original historical REX event file.',
                          'All selection/report windows exposed; no pristine OOS.',
                          'Subwindow resets initialize carried target states from cash.',
                          'G9 net-cap4.5 comparison differs from legacy additive subaccount results.',
                          'Intrabar MDD includes all barrier-exit subsets; not observed tick ordering.']}
    base.write_json(OUT/'report.json',result)
    chosen=labels[selected[0]]
    base.write_json(OUT/'shadow_config.json',{'enabled':False,'live_authorized':False,'research_only':True,
                    'weights_notional':selection[selected[0]]['weights_notional'],'net_cap':4.5,
                    'overlap_allowed':True,'long_short_offset':True,'fee_ratio_gate':False,'frequency_gate':False,
                    'selection_label':chosen,'report_hash':base.sha(OUT/'report.json')})
    for window in reports:
        print(window)
        for name in ['repository_live_g8','frozen_g9',chosen]:print(name,json.dumps(reports[window]['0.0006'][name]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    register() if args.freeze else run()
