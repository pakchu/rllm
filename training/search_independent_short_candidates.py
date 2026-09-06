"""Independent short event search; no parent-position or hedge-only restriction."""
import argparse
import hashlib
import json
import numpy as np
from training import search_long_complement_shorts as source
from training import search_meaningful_alpha_combinations as base
from training import g9_joint_net_ledger as ledger

OUT=base.ROOT/'research/independent_short_candidates'
EXTRA=['failed_breakout','long_exhaustion','regional_top','rally_dollar']
EXITS=[(None,None),(.01,.01),(.02,.015),(.03,.02)]
DESIGN={'version':1,'selection':'2024 only; fixed finalists reported2025/full2026H1 without replacement',
        'objective':'standalone short net return/drawdown, independent of any parent long position',
        'families':source.DESIGN['families']+EXTRA,'thresholds':[.5,1.],'hold_hours':[3,6,12,24],
        'exits':[[a,b] for a,b in EXITS],'entry':'completed-hour features, next5m open, fixed1x entry notional',
        'execution':'internally non-overlapping; stop before take on ambiguous5m bar; force-close at window end',
        'costs':[.0006,.001],'ranking':'selection10bp proxy Calmar + .1*selection return percentage points',
        'finalists':'global top5 distinct schedules plus top1 per family, all before report',
        'risk':'exact five-minute ledger for finalists; no cost-ratio/frequency/correlation gates',
        'caveat':'historically exposed data; coarse trade-envelope drawdown used only for discovery ranking',
        'live_enabled':False}


def specs():
    return [{'name':f'{fam}_t{t}_h{h}_tp{tp}_sl{sl}','family':fam,'threshold':t,'hold':h,'tp':tp,'sl':sl}
            for fam in DESIGN['families'] for t in DESIGN['thresholds'] for h in DESIGN['hold_hours'] for tp,sl in EXITS]


def raw_signal(x,s):
    if s['family'] not in EXTRA:return source.raw_short(x,s)
    t=s['threshold'];f=s['family']
    masks={'failed_breakout':(x.z24>t)&(x.mom6<0)&(x.flow6<x.flow24)&(x.volume_ratio>1),
           'long_exhaustion':(x.funding>.0001*t)&(x.z24>1+t)&(x.mom6<.5),
           'regional_top':(x.kimchi_premium_change6>0)&(x.z24>1+t)&(x.flow6<.01),
           'rally_dollar':(x.mom168>0)&(x.z6>t)&(x.dxy_change24>0)&(x.flow6<-.005)}
    return masks[f].fillna(False).to_numpy(bool)


def potential_trades(d,entries,hold,tp,sl):
    n=len(d['open']);entries=np.asarray(entries,int);h=hold*12
    rows=entries[:,None]+np.arange(h);valid=rows<n;ix=np.minimum(rows,n-1)
    opening=d['open'][entries]
    stops=np.zeros_like(valid) if sl is None else valid&(d['high'][ix]>=opening[:,None]*(1+sl))
    takes=np.zeros_like(valid) if tp is None else valid&(d['low'][ix]<=opening[:,None]*(1-tp))
    first_stop=np.where(stops.any(axis=1),stops.argmax(axis=1),h)
    first_take=np.where(takes.any(axis=1),takes.argmax(axis=1),h)
    timeout=np.minimum(entries+h,n)
    hit_stop=(first_stop<=first_take)&(entries+first_stop<timeout)
    hit_take=(first_take<first_stop)&(entries+first_take<timeout)
    barrier=hit_stop|hit_take
    exit_i=np.where(barrier,entries+np.minimum(first_stop,first_take),timeout)
    price=np.where(exit_i<n,d['open'][np.minimum(exit_i,n-1)],d['end'][-1])
    if sl is not None:price=np.where(hit_stop,opening*(1+sl),price)
    if tp is not None:price=np.where(hit_take,opening*(1-tp),price)
    last=exit_i-np.where(barrier,0,1)
    held=valid&(rows<=last[:,None])
    prefix=np.r_[0.,np.cumsum(d['funding'])]
    funding=prefix[last+1]-prefix[entries]
    gross=2-price/opening+funding/opening
    within_funding=(prefix[ix+1]-prefix[entries,None])/opening[:,None]
    high=np.max(np.where(held,2-d['low'][ix]/opening[:,None]+within_funding,-np.inf),axis=1)
    low=np.min(np.where(held,2-d['high'][ix]/opening[:,None]+within_funding,np.inf),axis=1)
    return {'entry':entries,'exit':exit_i,'barrier':barrier,'exit_price':price,'gross_factor':gross,
            'high_factor':high,'low_factor':low,'exit_ratio':price/opening}


def schedule(potential,active):
    indices=[];next_entry=0
    for i in np.flatnonzero(active):
        if potential['entry'][i]<next_entry:continue
        indices.append(i);next_entry=int(potential['exit'][i])+1
    return {k:v[indices] for k,v in potential.items()}


def proxy(trades,years,cost):
    eq=peak=1.;mdd=0.;returns=[]
    for gross,high,low,ratio in zip(trades['gross_factor'],trades['high_factor'],trades['low_factor'],trades['exit_ratio']):
        end=gross-cost*(1+ratio);top=high-cost;bottom=min(low-cost,end)
        peak=max(peak,eq*top);mdd=max(mdd,1-eq*bottom/peak)
        eq*=max(end,0);peak=max(peak,eq);mdd=max(mdd,1-eq/peak);returns.append(end-1)
        if bottom<=0:eq=0.;mdd=1.;break
    cagr=eq**(1/years)-1
    return {'return_pct':(eq-1)*100,'mdd_pct':min(mdd,1)*100,'calmar':cagr/mdd if mdd>1e-12 else 0.,
            'trades':len(trades['entry']),'win_rate':float(np.mean(np.array(returns)>0)) if returns else 0.}


def exact(d,trades,cost):
    n=len(d['open']);p=np.zeros((n,1));e=np.zeros((n,1),bool);b=np.full((n,1),np.nan)
    for a,z,barrier,price in zip(trades['entry'],trades['exit'],trades['barrier'],trades['exit_price']):
        p[a:min(z+int(barrier),n),0]=-1;e[a,0]=True
        if barrier:
            b[z,0]=price
            if z+1<n:e[z+1,0]=True
        elif z<n:e[z,0]=True
    rows,_=ledger.simulate(d,p,e,b,np.ones((1,1)),['independent_short'],cost=cost,net_cap=4.5)
    row=rows[0];row['trades']=len(trades['entry'])
    row['win_rate']=float(np.mean(trades['gross_factor']-cost*(1+trades['exit_ratio'])>1)) if len(trades['entry']) else 0.
    return row


def register():
    r={'design':DESIGN,'code_hash':base.sha(__file__),'source_code_hash':base.sha(source.__file__),
       'ledger_hash':base.sha(ledger.__file__),'context_hash':base.sha(source.CONTEXT),'specs':specs()}
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=r:raise RuntimeError('Frozen independent-short design drift')
    base.write_json(path,r);return r


def run():
    registration=register();a,x,d=source.load_window('2024');entries=np.flatnonzero(a['feature_row_for_5m']>=0)
    candidates=specs();cache={};rows=[];schedules=[]
    for s in candidates:
        key=(s['hold'],s['tp'],s['sl'])
        if key not in cache:cache[key]=potential_trades(d,entries,*key)
        active=raw_signal(x,s)[a['feature_row_for_5m'][entries]]
        trades=schedule(cache[key],active);schedules.append(trades)
        metric=proxy(trades,366/365.25,.001);score=metric['calmar']+.1*metric['return_pct']
        rows.append({'spec':s,'selection_proxy':metric,'score':score})
    order=sorted(range(len(rows)),key=lambda i:rows[i]['score'],reverse=True)
    top=[];seen=set();families=set()
    for i in order:
        t=schedules[i];digest=hashlib.sha256(t['entry'].tobytes()+t['exit'].tobytes()+t['exit_price'].tobytes()).hexdigest()
        if len(top)<5 and digest not in seen:top.append(i);seen.add(digest)
    for i in order:
        family=candidates[i]['family']
        if family in families:continue
        families.add(family)
        if i not in top:top.append(i)
    freeze={'selection':'2024 only','top':[{**rows[i],'schedule_hash':hashlib.sha256(schedules[i]['entry'].tobytes()+schedules[i]['exit'].tobytes()).hexdigest()} for i in top],'report_reranking':False}
    base.write_json(OUT/'selection_freeze.json',freeze);reports={};export={}
    for window in ['2024','2025','2026H1']:
        a,x,d=source.load_window(window);entries=np.flatnonzero(a['feature_row_for_5m']>=0);cache={};reports[window]={};export[window]={}
        for i in top:
            s=candidates[i];key=(s['hold'],s['tp'],s['sl'])
            if key not in cache:cache[key]=potential_trades(d,entries,*key)
            trades=schedule(cache[key],raw_signal(x,s)[a['feature_row_for_5m'][entries]])
            reports[window][s['name']]={str(c):exact(d,trades,c) for c in DESIGN['costs']}
            export[window][s['name']]=[{k:(int(v[j]) if k in ['entry','exit'] else bool(v[j]) if k=='barrier' else float(v[j])) for k,v in trades.items()} for j in range(len(trades['entry']))]
    result={'registration':registration,'count':len(candidates),'freeze':freeze,'inventory':[rows[i] for i in order],
            'reports':reports,'live_enabled':False,'notes':['No parent long required.','Exact five-minute ledger for finalists; coarse proxy only ranks selection.',
                                                       'All historical windows previously exposed; no pristine OOS claim.']}
    base.write_json(OUT/'report.json',result);base.write_json(OUT/'finalist_trades.json',export)
    for i in top:
        name=candidates[i]['name'];print(name,flush=True)
        for window in reports:
            r=reports[window][name]['0.0006'];print(window,{k:r[k] for k in ['return_pct','mdd_pct','trades','win_rate']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
