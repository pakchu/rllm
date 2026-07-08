import json, itertools, random
from pathlib import Path
from collections import Counter
import numpy as np

import training.evaluate_volume_wave_portfolio_combo as vw
import training.portfolio_with_dynamic_exit_sleeves as dx
import training.evaluate_portfolio_llm_selector as ep
from training.evaluate_portfolio_llm_selector import _context_id, _tokens

OUT='results/portfolio_gross20_step005_cost6bp_mdd20_with_dynamic_2026-07-08.json'
DOC='docs/portfolio-gross20-step005-cost6bp-mdd20-with-dynamic-2026-07-08.md'
COST=0.0006
# Patch module-level event costs before building events.
vw.COST=COST
dx.COST=COST
_orig_ep_event_return=ep._event_return
def _event_return_6bp(market,p,h,side,cost=COST):
    return _orig_ep_event_return(market,p,h,side,cost=COST)
ep._event_return=_event_return_6bp

BASE_SLEEVES=list(vw.SLEEVES)
EXTRA_SLEEVES=['oi_low','oi_high_sel','bear_rex_short','rex_dyn_short_exit','oi_alt_ratio72_dyn_exit']
SLEEVES=[]
for s in BASE_SLEEVES+EXTRA_SLEEVES:
    if s not in SLEEVES: SLEEVES.append(s)


def J(p): return json.loads(Path(p).read_text())


def add_old_live_events(events, market, feat, masks):
    # OI pullback low-frequency long used by the pre-wave live gross6 mix.
    low=J('configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json')['signal']
    vw.append_active(events, market, feat, masks, vw.gate(feat, low['gates']), 'oi_low', 'long', int(low['hold_bars_5m']), int(low['stride_bars_5m']))
    # OI high-frequency long with train-only symbolic selector overlay.
    high=J('configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json')
    active=vw.gate(feat, high['gates'])
    sel=J('configs/live/oi_divergence_sma24_highfreq_h30_s6_llm_selector_overlay.json')['symbolic_proxy']
    blocked={x['context_id'] for x in sel['blocked_contexts']}
    keys=tuple(sel['context_keys'])
    def allow_oi(ip:int)->bool:
        return _context_id(_tokens(ip, market=market, feat=feat), keys) not in blocked
    ep._append_active_events(events, market, feat, masks, active, 'oi_high_sel', 'long', int(high['hold_bars']), int(high['stride_bars']), selector=allow_oi)
    # Bear-only REX short sleeve used by current live mix.
    ep._append_prediction_events(events, market, feat, masks, 'bear_rex_short')
    # Drop non-short bear_rex events defensively; ep helper already only accepts SHORT.


def arrays(events,masks):
    starts={sp:int(np.flatnonzero(m)[0]) for sp,m in masks.items()}
    ends={sp:int(np.flatnonzero(m)[-1])+1 for sp,m in masks.items()}
    by={}
    for sp in masks:
        ln=ends[sp]-starts[sp]
        matsR=[]; matsA=[]; counts=[]; wins=[]
        for sl in SLEEVES:
            r=np.zeros(ln); a=np.zeros(ln); c=0; w=0
            for e in events:
                if e['split']==sp and e['sleeve']==sl:
                    st=starts[sp]; en=ends[sp]
                    r += e['ret'][st:en]; a += e['adv'][st:en]; c += 1; w += float(e.get('ret_bps',0))>0
            matsR.append(r); matsA.append(a); counts.append(c); wins.append(w)
        R=np.vstack(matsR); A=np.vstack(matsA)
        active=np.any((R!=0)|(A!=0),axis=0)
        by[sp]={'R':R[:,active], 'A':A[:,active], 'counts':np.array(counts), 'wins':np.array(wins), 'active_bars':int(active.sum())}
    return by


def metric(d,years,w):
    wv=np.array([w.get(s,0.0) for s in SLEEVES])
    r=wv@d['R']; adv=wv@d['A']
    fac=np.maximum(0,1+r)
    eqp=np.cumprod(fac) if len(fac) else np.array([1.0])
    eqb=np.r_[1.0,eqp[:-1]] if len(fac) else np.array([1.0])
    pka=np.maximum.accumulate(eqp); pkb=np.maximum.accumulate(eqb)
    dd1=float(np.nanmax(1-eqp/np.maximum(pka,1e-12))) if len(eqp) else 0.0
    dd2=float(np.nanmax(1-(eqb*(1+adv))/np.maximum(pkb,1e-12))) if len(eqb) else 0.0
    mdd=max(dd1,dd2)*100
    eq=float(eqp[-1]) if len(eqp) else 1.0
    cagr=(eq**(1/years)-1)*100 if eq>0 else -100.0
    vals=r[np.abs(r)>1e-12]
    sh=float(vals.mean()/vals.std(ddof=1)*np.sqrt(len(vals))) if len(vals)>1 and vals.std(ddof=1)>0 else 0.0
    trades=int(np.sum(d['counts'][wv!=0])); wins=int(np.sum(d['wins'][wv!=0]))
    return {'total_return_pct':(eq-1)*100,'cagr_pct':cagr,'strict_mdd_pct':mdd,'cagr_to_strict_mdd':cagr/mdd if mdd>1e-12 else 0.0,'trade_entries':trades,'win_rate':wins/trades if trades else 0.0,'active_bars':d['active_bars'],'bar_sharpe_like':sh,'sleeve_trade_counts':{s:int(c) if w.get(s,0)!=0 else 0 for s,c in zip(SLEEVES,d['counts'])}}

def metrics(by,years,w): return {sp:metric(by[sp],years[sp],w) for sp in ['train','test2024','eval2025','ytd2026']}

def passes(st):
    return all(st[sp]['strict_mdd_pct']<=20 and st[sp]['cagr_to_strict_mdd']>=5 for sp in ['test2024','eval2025','ytd2026'])

def score(st):
    o=['test2024','eval2025','ytd2026']
    ok=passes(st)
    maxm=max(st[x]['strict_mdd_pct'] for x in o)
    minr=min(st[x]['cagr_to_strict_mdd'] for x in o)
    # return priority after hard constraints, then robustness.
    ret=sum(st[x]['total_return_pct'] for x in o)
    return (ok, ret if ok else -maxm, minr, st['ytd2026']['cagr_to_strict_mdd'], -maxm)


STEP=0.05
MIN_W=0.10
GROSS_CAP=20.0

def quantize_weight(x):
    if x < MIN_W - 1e-12:
        return 0.0
    return round(round(x / STEP) * STEP, 2)

def quantize_portfolio(w):
    q={s:quantize_weight(float(w.get(s,0.0))) for s in SLEEVES}
    # If rounding pushes over cap, shave largest weights by one step until valid.
    while sum(q.values()) > GROSS_CAP + 1e-9:
        k=max(q, key=lambda kk:q[kk])
        q[k]=quantize_weight(q[k]-STEP)
    return q

def weight_unit_ok(w):
    for v in w.values():
        if v <= 1e-12:
            continue
        if v < MIN_W - 1e-12:
            return False
        if abs(v/STEP - round(v/STEP)) > 1e-9:
            return False
    return sum(w.values()) <= GROSS_CAP + 1e-9

def clean(w): return {s:round(float(v),6) for s,v in w.items() if v>1e-12}

def scale_to_mdd(by,years,w,target=19.7):
    # Binary scale to keep max OOS MDD just below target. CAGR/MDD checked afterwards.
    lo,hi=0.0,1.5
    def maxm(k):
        ww={s:v*k for s,v in w.items()}
        return max(metric(by[x],years[x],ww)['strict_mdd_pct'] for x in ['test2024','eval2025','ytd2026'])
    while maxm(hi)<target and sum(w.values())*hi<GROSS_CAP:
        hi*=1.25
    hi=min(hi, 20/max(1e-12,sum(w.values())))
    for _ in range(12):
        mid=(lo+hi)/2
        if maxm(mid)<=target: lo=mid
        else: hi=mid
    return {s:v*lo for s,v in w.items()}


def candidate_weights(by,years):
    seeds=[]
    # current live and known prior tops
    seeds.append({'nonpb30_taker':0.5,'oi_high_sel':0.5,'bear_rex_short':1.0})
    seeds.append({'nonpb30_taker':0.825,'oi_low':0.825,'oi_high_sel':0.825,'bear_rex_short':2.475})
    seeds.append({'pb30_base':0.85,'oi_high_sel':1.7,'bear_rex_short':2.55})
    seeds.append({'nonpb30_taker':0.8,'oi_low':0.8,'oi_high_sel':1.6,'bear_rex_short':1.6})
    # wave/dynamic recent tops
    seeds.append({'nonpb30_taker':2.0,'rex_rule':2.5,'oi_wave_lowpos144':0.5,'rex_dyn_short_exit':1.0})
    seeds.append({'nonpb30_taker':2.0,'oi_raw':0.25,'rex_rule':2.5,'oi_wave_lowpos144':0.25,'rex_dyn_short_exit':1.0})
    seeds.append({'nonpb30_taker':1.5,'oi_raw':0.5,'rex_rule':2.5,'oi_wave_lowpos144':0.5,'rex_dyn_short_exit':1.0})
    # hybrid bridges
    seeds += [
        {'nonpb30_taker':1.0,'oi_high_sel':1.0,'bear_rex_short':1.5,'rex_dyn_short_exit':0.5,'oi_wave_lowpos144':0.5},
        {'nonpb30_taker':1.25,'oi_high_sel':1.0,'bear_rex_short':1.25,'rex_rule':1.0,'rex_dyn_short_exit':0.5},
        {'nonpb30_taker':1.25,'oi_low':0.5,'oi_high_sel':1.0,'bear_rex_short':1.5,'rex_dyn_short_exit':0.5},
        {'nonpb30_taker':1.0,'oi_low':0.5,'oi_high_sel':1.0,'bear_rex_short':1.25,'rex_rule':1.0,'oi_wave_lowpos144':0.25},
    ]
    # scale all seeds to mdd boundary and add local variants
    out=[]; seen=set()
    def add(w):
        ww=quantize_portfolio({s:max(0.0,float(w.get(s,0.0))) for s in SLEEVES})
        if sum(ww.values())<=0 or not weight_unit_ok(ww): return
        key=tuple(ww[s] for s in SLEEVES)
        if key not in seen:
            seen.add(key); out.append(ww)
    for s in seeds:
        add(s); add(scale_to_mdd(by,years,s,19.7)); add(scale_to_mdd(by,years,s,19.95))
        keys=[k for k,v in s.items() if v]
        for k in ['oi_raw','oi_high_sel','oi_low','bear_rex_short','rex_rule','rex_dyn_short_exit','oi_wave_lowpos144','oi_upbit_ratio288_low']:
            for d in [-0.5,-0.25,0.25,0.5]:
                w=dict(s); w[k]=max(0,w.get(k,0)+d); add(w); add(scale_to_mdd(by,years,w,19.8))
    # Random convex mixes of promising sleeves, scaled to mdd boundary.
    rng=random.Random(7)
    pool=['pb30_base','nonpb30_taker','oi_low','oi_high_sel','bear_rex_short','oi_raw','rex_rule','oi_wave_lowpos144','oi_upbit_ratio288_low','rex_dyn_short_exit','oi_alt_ratio72_dyn_exit']
    templates=[s for s in seeds]
    for _ in range(4500):
        k=rng.randint(3,6)
        chosen=rng.sample(pool,k)
        raw=np.array([rng.random()**1.7 for _ in chosen]); raw=raw/raw.sum()
        gross=rng.choice([3.0,4.0,5.0,6.0,7.0,8.0,10.0,12.0,15.0,20.0])
        w={c:float(a*gross) for c,a in zip(chosen,raw)}
        add(scale_to_mdd(by,years,w,19.8))
        if rng.random()<0.2: add(w)
    return out


def main():
    market,feat,masks,years,events,thr=vw.build_events()
    add_old_live_events(events,market,feat,masks)
    dyn=dx.build_dynamic_sleeves(market,feat,masks,years)
    events.extend(dyn)
    by=arrays(events,masks)
    rows=[]
    for w in candidate_weights(by,years):
        st=metrics(by,years,w)
        rows.append({'weights':clean(w),'gross':sum(w.values()),'stats':st,'passes_mdd20_ratio5':passes(st),'score_tuple':score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    qualified=[r for r in rows if r['passes_mdd20_ratio5']]
    current_live=J('configs/live/portfolio_gross6_mdd20_ratio5_return_best_candidate.json')
    out={'protocol':{'gross_cap':20.0,'cost_each_side':COST,'condition':'test2024/eval2025/ytd2026 strict_mdd_pct<=20 and CAGR/MDD>=5; nonzero weights >=0.10 and multiples of 0.05','rank':'among qualified: OOS split return sum, then min ratio/ytd ratio','selection_scope':'quantized seeded/local/random scan over pre-wave robust/live sleeves + wave/volume sleeves + dynamic-exit sleeves; gross<=20'},'sleeves':SLEEVES,'event_counts':{sp:dict(Counter(e['sleeve'] for e in events if e['split']==sp)) for sp in masks},'evaluated_unique':len(rows),'qualified_count':len(qualified),'current_live_reference':{'weights':current_live['weights'],'metrics':current_live['metrics']},'top_qualified':[{k:v for k,v in r.items() if k!='score_tuple'} for r in qualified[:30]],'top_diagnostic':[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:30]]}
    Path(OUT).write_text(json.dumps(out,indent=2,ensure_ascii=False))
    md=['# Gross<=20 step 0.05 cost 6bp MDD20 portfolio scan with dynamic sleeves (2026-07-08)','',json.dumps(out['protocol'],ensure_ascii=False),'',f"evaluated_unique={len(rows)}, qualified_count={len(qualified)}",'','## Top qualified','| rank | gross | weights | 2024 ret/CAGR/MDD/ratio | 2025 ret/CAGR/MDD/ratio | 2026 ret/CAGR/MDD/ratio | trades 24/25/26 |','|---:|---:|---|---:|---:|---:|---:|']
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}"
    for i,r in enumerate(qualified[:20],1):
        st=r['stats']; tr=f"{st['test2024']['trade_entries']}/{st['eval2025']['trade_entries']}/{st['ytd2026']['trade_entries']}"
        md.append(f"| {i} | {r['gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | {tr} |")
    md += ['','## Top diagnostic if no/limited qualified','| rank | pass | gross | weights | 2024 ret/CAGR/MDD/ratio | 2025 ret/CAGR/MDD/ratio | 2026 ret/CAGR/MDD/ratio |','|---:|---:|---:|---|---:|---:|---:|']
    for i,r in enumerate(rows[:20],1):
        st=r['stats']; md.append(f"| {i} | {r['passes_mdd20_ratio5']} | {r['gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'evaluated':len(rows),'qualified':len(qualified),'top_qualified':out['top_qualified'][:5],'top_diagnostic':out['top_diagnostic'][:5]},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
