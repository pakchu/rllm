"""Alpha101-derived dual gates on existing signal pool."""
from __future__ import annotations
import json, random
from collections import Counter
from pathlib import Path
import numpy as np
import training.search_portfolio_gross20_step005_mdd20_with_dynamic as base
from training.search_alpha101_derivative_alphas import add_features, qthr, active

OUT='results/alpha101_dual_gate_signal_pool_2026-07-09.json'
DOC='docs/alpha101-dual-gate-signal-pool-2026-07-09.md'
CORE=['nonpb30_taker','oi_raw','rex_rule','oi_upbit_ratio288_low','bear_rex_short','oi_alt_ratio72_dyn_exit','oi_low','oi_high_sel']
BASE_TOP={'nonpb30_taker':0.65,'oi_raw':0.55,'rex_rule':2.00,'oi_upbit_ratio288_low':2.25,'bear_rex_short':0.30,'oi_alt_ratio72_dyn_exit':0.35}
LIVE={'nonpb30_taker':0.95,'oi_low':0.95,'oi_high_sel':0.95,'bear_rex_short':2.90}

def masks_from_feat(m,feat,masks):
    f=add_features(m); tr=masks['train']; gates={}
    specs=[
      ('x101_not_vwap_exhaust_up','x101_vwap_exhaust_up','<=',0.80),
      ('x101_not_vwap_exhaust_down','x101_vwap_exhaust_down','<=',0.80),
      ('x101_low_vwap_exhaust_up','x101_vwap_exhaust_up','<=',0.60),
      ('x101_low_vwap_exhaust_down','x101_vwap_exhaust_down','<=',0.60),
      ('x101_clean_breakout_up','x101_clean_breakout_up','>=',0.75),
      ('x101_clean_breakout_down','x101_clean_breakout_down','>=',0.75),
      ('x101_not_clean_breakout_up','x101_clean_breakout_up','<=',0.75),
      ('x101_not_clean_breakout_down','x101_clean_breakout_down','<=',0.75),
      ('x101_lowvol_momo_up','x101_lowvol_intraday_momo_up','>=',0.75),
      ('x101_lowvol_momo_down','x101_lowvol_intraday_momo_down','>=',0.75),
      ('x101_not_volcorr_reversal_up','x101_volcorr_reversal_up','<=',0.80),
      ('x101_not_volcorr_reversal_down','x101_volcorr_reversal_down','<=',0.80),
      ('x101_vwap_revert_long_zone','a_vwap_gap_z','<=',0.20),
      ('x101_vwap_revert_short_zone','a_vwap_gap_z','>=',0.80),
      ('x101_low_absret_vol','a_absret_vol_rank','<=',0.70),
      ('x101_high_absret_vol','a_absret_vol_rank','>=',0.70),
    ]
    for name,col,op,qq in specs:
        thr=qthr(f,tr,col,qq)
        if thr is not None:
            gates[name]=active(f,[(col,op,thr)])
    return gates

def clone(events,gates):
    out=list(events); meta={}
    for sl in CORE:
        src=[e for e in events if e['sleeve']==sl]
        if not src: continue
        for gname,gmask in gates.items():
            kept=[]; name=f'{sl}__{gname}'
            for e in src:
                ip=int(e['signal_pos'])
                if 0<=ip<len(gmask) and bool(gmask[ip]):
                    ee=dict(e); ee['sleeve']=name; kept.append(ee)
            if kept:
                out.extend(kept); meta[name]={'source_sleeve':sl,'gate':gname,'kept':len(kept),'source':len(src),'keep_rate':len(kept)/max(1,len(src))}
    return out,meta

def clean(w): return {k:round(float(v),6) for k,v in w.items() if v>1e-12}

def cands(by,years,sleeves,gated):
    out=[]; seen=set()
    def add(w):
        ww=base.quantize_portfolio({s:max(0,float(w.get(s,0))) for s in sleeves})
        if sum(ww.values())<=0 or not base.weight_unit_ok(ww): return
        key=tuple(ww[s] for s in sleeves)
        if key not in seen: seen.add(key); out.append(ww)
    seeds=[BASE_TOP,LIVE,{'nonpb30_taker':0.9,'oi_raw':0.55,'rex_rule':2.0,'oi_upbit_ratio288_low':2.5,'bear_rex_short':0.3,'oi_alt_ratio72_dyn_exit':0.35}]
    for s in seeds:
        add(s); add(base.scale_to_mdd(by,years,s,19.8))
    bysrc={}
    for g in gated: bysrc.setdefault(g.split('__',1)[0],[]).append(g)
    for seed in seeds:
        for src,gs in bysrc.items():
            if seed.get(src,0)<=0: continue
            for g in gs:
                w=dict(seed); val=w.pop(src); w[g]=val
                add(w); add(base.scale_to_mdd(by,years,w,19.8))
    rng=random.Random(101101)
    interesting=[g for g in gated if any(x in g for x in ['not_','low_','revert','clean_breakout'])]
    for _ in range(1500):
        w=dict(rng.choice(seeds))
        for g in rng.sample(interesting,rng.randint(1,min(3,len(interesting)))):
            src=g.split('__',1)[0]
            if src in w:
                val=w.pop(src); w[g]=w.get(g,0)+val*rng.choice([0.5,0.75,1.0,1.25])
        for k in list(w):
            if rng.random()<0.2: w[k]=max(0,w[k]+rng.choice([-0.25,-0.15,0.15,0.25]))
        add(w); add(base.scale_to_mdd(by,years,w,19.8))
    return out

def main():
    m,feat,masks,years,events,_=base.vw.build_events(); base.add_old_live_events(events,m,feat,masks); events.extend(base.dx.build_dynamic_sleeves(m,feat,masks,years))
    gates=masks_from_feat(m,feat,masks); all_events,meta=clone(events,gates)
    sleeves=list(base.SLEEVES)
    for n in meta:
        if n not in sleeves: sleeves.append(n)
    base.SLEEVES=sleeves; by=base.arrays(all_events,masks)
    rows=[]
    for w in cands(by,years,sleeves,list(meta)):
        st=base.metrics(by,years,w); gg=sum(v for k,v in w.items() if '__x101_' in k)
        rows.append({'weights':clean(w),'gross':round(sum(w.values()),6),'gated_gross':round(gg,6),'stats':st,'passes_mdd20_ratio5':base.passes(st),'score_tuple':base.score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True); qual=[r for r in rows if r['passes_mdd20_ratio5']]; qg=[r for r in qual if r['gated_gross']>=0.1]
    rep={'protocol':'Alpha101-derived dual gates on existing signal pool; train-only gate thresholds; 6bp/side strict MDD; diagnostic.','gated_meta':meta,'evaluated_unique':len(rows),'qualified_count':len(qual),'qualified_gated_count':len(qg),'top_qualified_gated':[{k:v for k,v in r.items() if k!='score_tuple'} for r in qg[:30]],'top_qualified':[{k:v for k,v in r.items() if k!='score_tuple'} for r in qual[:30]]}
    Path(OUT).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"
    md=['# Alpha101 dual-gate signal-pool scan (2026-07-09)','',rep['protocol'],'',f"evaluated_unique={len(rows)}, qualified_count={len(qual)}, qualified_gated_count={len(qg)}",'','| rank | gross | gated | weights | 2024 | 2025 | 2026 |','|---:|---:|---:|---|---:|---:|---:|']
    for i,r in enumerate(rep['top_qualified_gated'][:20],1):
        st=r['stats']; md.append(f"| {i} | {r['gross']:.2f} | {r['gated_gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'evaluated':len(rows),'qualified':len(qual),'qualified_gated':len(qg),'top':rep['top_qualified_gated'][:8]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
