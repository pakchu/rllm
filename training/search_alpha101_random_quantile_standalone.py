"""Wide standalone Alpha101 primitive quantile-combo search.

Uses only Alpha101-style OHLCV/VWAP/rank/correlation primitives. Thresholds are
always train<2024 quantiles. This intentionally avoids portfolio/sleeve context.
"""
from __future__ import annotations
import json, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from training.search_alpha101_derivative_alphas import add_features
from training.search_vpin_formulaic_alpha import load_market_and_splits, stats, trade_arrays

OUT='results/alpha101_random_quantile_standalone_2026-07-09.json'
DOC='docs/alpha101-random-quantile-standalone-2026-07-09.md'
HOLDS=[24,48,72,96,144]
STRIDES=[12,24]


def qthr(f,train,col,q):
    vals=f.loc[train,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return None
    return float(np.quantile(vals,q))

def active(f,terms):
    a=np.ones(len(f),bool)
    for col,op,thr,q in terms:
        x=f[col].to_numpy(float); a &= np.isfinite(x) & ((x>=thr) if op=='>=' else (x<=thr))
    return a

def eval_rule(m,f,masks,years,terms,side,hold,stride):
    act=active(f,terms); n=len(m); smod=(np.arange(n)%stride)==0; fac,mn,rr=trade_arrays(m,hold,side)
    out={}
    for sp,mask in masks.items():
        loc=[]; nxt=0; idx=np.flatnonzero(act & mask & smod); idx=idx[(idx>=143)&(idx<n-hold-2)]
        for p in idx:
            p=int(p); xp=p+1+hold
            if p<nxt or xp>=n or not mask[xp] or not np.isfinite(fac[p]): continue
            loc.append((float(fac[p]),float(mn[p]),float(rr[p]))); nxt=xp
        out[sp]=stats(loc,years[sp])
    return out

def mat(f,train,spec):
    terms=[]
    for col,op,q in spec:
        if col=='a_early_session': thr=0.5
        elif col=='a_late_session': thr=0.5
        else:
            thr=qthr(f,train,col,q)
            if thr is None: return None
        terms.append((col,op,float(thr),float(q)))
    return terms

def score(st):
    t,e,y=st['test2024'],st['eval2025'],st['ytd2026']
    pos=t['cagr_pct']>0 and e['cagr_pct']>0 and y['cagr_pct']>0
    enough=t['trade_entries']>=20 and e['trade_entries']>=15 and y['trade_entries']>=8
    minr=min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd'],y['cagr_to_strict_mdd'])
    ret=t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct']
    return (pos and enough, minr, ret, y['cagr_to_strict_mdd'], t['trade_entries']+e['trade_entries']+y['trade_entries'])

def uniq_name(side,spec):
    raw=side+'|'+';'.join(f'{c}{o}{q:.2f}' for c,o,q in spec)
    return hashlib.md5(raw.encode()).hexdigest()[:10]

def gen_specs(seed=101, n=2500):
    rng=random.Random(seed)
    # Coherent building blocks. q is threshold quantile, not direction strength.
    long_blocks=[
      [('a_ret_z_24','<=',0.30),('a_vwap_gap_z','<=',0.30)],
      [('a_ret_z_72','<=',0.30),('a_pos_288','<=',0.35)],
      [('a_vwap_gap_z','<=',0.25),('a_clv_rank_288','<=',0.35)],
      [('a_break_high_288','>=',0.70),('a_range_compress_288','>=',0.60)],
      [('a_intr_rank_288','>=',0.65),('a_taker_72_z','>=',0.55)],
      [('a_ret_vol_corr_72','<=',0.30),('a_ret_z_72','<=',0.35)],
      [('a_spread_z_144','<=',0.35),('a_taker_72_z','>=',0.55)],
      [('a_absret_vol_rank','<=',0.60),('a_intr_rank_288','>=',0.60)],
      [('a_ret_z_12','<=',0.35),('a_early_session','>=',0.50)],
    ]
    short_blocks=[
      [('a_ret_z_24','>=',0.70),('a_vwap_gap_z','>=',0.70)],
      [('a_ret_z_72','>=',0.70),('a_pos_288','>=',0.65)],
      [('a_vwap_gap_z','>=',0.75),('a_clv_rank_288','>=',0.65)],
      [('a_break_low_288','>=',0.70),('a_range_compress_288','>=',0.60)],
      [('a_intr_rank_288','<=',0.35),('a_taker_72_z','<=',0.45)],
      [('a_ret_vol_corr_72','>=',0.70),('a_ret_z_72','>=',0.65)],
      [('a_spread_z_144','<=',0.35),('a_taker_72_z','<=',0.45)],
      [('a_absret_vol_rank','<=',0.60),('a_intr_rank_288','<=',0.40)],
      [('a_ret_z_12','>=',0.65),('a_early_session','>=',0.50)],
    ]
    optional_long=[('a_absret_vol_rank','>=',0.55),('a_absret_vol_rank','<=',0.70),('a_vol_z_288','>=',0.55),('a_vol_z_288','<=',0.55),('a_range_compress_288','>=',0.55),('a_spread_z_72','<=',0.45),('a_ret_vol_corr_288','<=',0.40),('a_taker_z','>=',0.50),('a_late_session','>=',0.50)]
    optional_short=[('a_absret_vol_rank','>=',0.55),('a_absret_vol_rank','<=',0.70),('a_vol_z_288','>=',0.55),('a_vol_z_288','<=',0.55),('a_range_compress_288','>=',0.55),('a_spread_z_72','<=',0.45),('a_ret_vol_corr_288','>=',0.60),('a_taker_z','<=',0.50),('a_late_session','>=',0.50)]
    yielded=set()
    def jitter(term):
        c,o,q=term
        if c in ('a_early_session','a_late_session'): return term
        return (c,o,max(0.05,min(0.95,q+rng.choice([-0.15,-0.10,-0.05,0,0.05,0.10,0.15]))))
    for side,blocks,opts in [('long',long_blocks,optional_long),('short',short_blocks,optional_short)]:
        for b in blocks:
            for _ in range(35):
                spec=[jitter(x) for x in b]
                for o in rng.sample(opts, rng.randint(0,2)): spec.append(jitter(o))
                # de-duplicate contradictory same feature by keeping first
                out=[]; seen=set()
                for t in spec:
                    if t[0] not in seen: seen.add(t[0]); out.append(t)
                key=(side,tuple(out))
                if key not in yielded:
                    yielded.add(key); yield side,out
    # pure random combos from primitives, but direction-coherent.
    for _ in range(n):
        side=rng.choice(['long','short'])
        if side=='long': blocks,opts=long_blocks,optional_long
        else: blocks,opts=short_blocks,optional_short
        spec=[]
        for b in rng.sample(blocks,rng.randint(1,2)): spec+= [jitter(x) for x in b]
        for o in rng.sample(opts,rng.randint(0,2)): spec.append(jitter(o))
        out=[]; seen=set()
        for t in spec:
            if t[0] not in seen: seen.add(t[0]); out.append(t)
        if 2<=len(out)<=5:
            key=(side,tuple(out))
            if key not in yielded:
                yielded.add(key); yield side,out

def main():
    m,_,masks,years=load_market_and_splits(); f=add_features(m); train=masks['train']
    rows=[]; tested=0
    for side,spec in gen_specs():
        terms=mat(f,train,spec)
        if terms is None: continue
        a=active(f,terms); ar=float(a[train].mean())
        if ar<0.0025 or ar>0.25: continue
        tested+=1
        for hold in HOLDS:
            for stride in STRIDES:
                st=eval_rule(m,f,masks,years,terms,side,hold,stride)
                rows.append({'name':'a101q_'+uniq_name(side,spec),'side':side,'active_rate_train':ar,'terms':[{'feature':c,'op':op,'threshold':thr,'train_q':q} for c,op,thr,q in terms],'hold':hold,'stride':stride,'stats':st,'score_tuple':score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    top=[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:200]]
    rep={'protocol':'Wide standalone Alpha101 primitive quantile-combo search. Thresholds train<2024 only. Cost 6bp/side. Strict MDD includes in-position adverse excursion. No existing signal/portfolio context.','tested_specs':tested,'all_count':len(rows),'top':top}
    Path(OUT).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}/{s['bar_sharpe_like']:.2f}"
    md=['# Wide Alpha101 standalone quantile search (2026-07-09)','',rep['protocol'],'',f"tested_specs={tested}, all_count={len(rows)}",'','| rank | name | side | active | hold/stride | train ratio/trades | 2024 ret/CAGR/MDD/ratio/trades/win/sh | 2025 | 2026 | terms |','|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    for i,r in enumerate(top[:80],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} q{t['train_q']:.2f}({t['threshold']:.4g})" for t in r['terms'])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['active_rate_train']:.3f} | {r['hold']}/{r['stride']} | {st['train']['cagr_to_strict_mdd']:.2f}/{st['train']['trade_entries']} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'tested_specs':tested,'all_count':len(rows),'top':top[:12]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
