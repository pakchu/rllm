"""Alpha101 primitive quantile-combo alpha search.

This mirrors the repo's successful alpha style: combine 2-4 causal primitive
features with train-fitted quantile hard gates, then evaluate OOS splits.
"""
from __future__ import annotations

import itertools, json, math, random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_alpha101_derivative_alphas import add_features
from training.search_vpin_formulaic_alpha import load_market_and_splits, stats, trade_arrays

OUT='results/alpha101_quantile_combo_scan_2026-07-09.json'
DOC='docs/alpha101-quantile-combo-scan-2026-07-09.md'

LONG_TEMPLATES = [
    # VWAP/CLV reversion after sell pressure but not pure crash continuation.
    ('vwap_clv_revert_long', [('a_vwap_gap_z','<=',0.20),('a_clv_rank_288','<=',0.35),('a_absret_vol_rank','>=',0.60)]),
    ('vwap_ret_revert_long', [('a_vwap_gap_z','<=',0.20),('a_ret_z_24','<=',0.35),('a_absret_vol_rank','>=',0.60)]),
    ('pullback_clean_long', [('a_ret_z_72','<=',0.35),('a_pos_288','<=',0.35),('a_range_compress_288','>=',0.60)]),
    ('compression_breakout_long', [('a_break_high_288','>=',0.70),('a_range_compress_288','>=',0.60),('a_taker_72_z','>=',0.55)]),
    ('lowvol_momo_long', [('a_intr_rank_288','>=',0.65),('a_spread_z_144','<=',0.40),('a_taker_72_z','>=',0.50)]),
    ('volcorr_revert_long', [('a_ret_vol_corr_72','<=',0.30),('a_ret_z_72','<=',0.35),('a_absret_vol_rank','>=',0.55)]),
    ('gap_revert_long', [('a_day_gap','<=',0.25),('a_early_session','>=',0.50),('a_ret_z_12','<=',0.40)]),
]
SHORT_TEMPLATES = [
    ('vwap_clv_revert_short', [('a_vwap_gap_z','>=',0.80),('a_clv_rank_288','>=',0.65),('a_absret_vol_rank','>=',0.60)]),
    ('vwap_ret_revert_short', [('a_vwap_gap_z','>=',0.80),('a_ret_z_24','>=',0.65),('a_absret_vol_rank','>=',0.60)]),
    ('rally_exhaust_short', [('a_ret_z_72','>=',0.65),('a_pos_288','>=',0.65),('a_range_expand_72','>=',0.60)]),
    ('compression_breakdown_short', [('a_break_low_288','>=',0.70),('a_range_compress_288','>=',0.60),('a_taker_72_z','<=',0.45)]),
    ('lowvol_momo_short', [('a_intr_rank_288','<=',0.35),('a_spread_z_144','<=',0.40),('a_taker_72_z','<=',0.50)]),
    ('volcorr_revert_short', [('a_ret_vol_corr_72','>=',0.70),('a_ret_z_72','>=',0.65),('a_absret_vol_rank','>=',0.55)]),
    ('gap_revert_short', [('a_day_gap','>=',0.75),('a_early_session','>=',0.50),('a_ret_z_12','>=',0.60)]),
]

# Nearby quantile perturbations around each hand-designed template.
Q_SHIFTS=[-0.10,-0.05,0.0,0.05,0.10]
HOLDS=[12,24,36,48,72,96,144,192]
STRIDES=[6,12,24]


def qthr(f, train, col, q):
    vals=f.loc[train,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return None
    return float(np.quantile(vals,q))


def clampq(q): return min(0.95,max(0.05,float(q)))


def materialize(f, train, spec):
    terms=[]
    for col,op,q in spec:
        if col=='a_early_session':
            terms.append((col,op,float(q),q)); continue
        qq=clampq(q); thr=qthr(f,train,col,qq)
        if thr is None: return None
        terms.append((col,op,thr,qq))
    return terms


def act(f, terms):
    a=np.ones(len(f),bool)
    for col,op,thr,_ in terms:
        x=f[col].to_numpy(float)
        a &= np.isfinite(x) & ((x>=thr) if op=='>=' else (x<=thr))
    return a


def eval_rule(m,f,masks,years,terms,side,hold,stride):
    active=act(f,terms); n=len(m); smod=(np.arange(n)%stride)==0
    fac,mn,rr=trade_arrays(m,hold,side)
    out={}
    for sp,mask in masks.items():
        loc=[]; nxt=0
        idx=np.flatnonzero(active & mask & smod); idx=idx[(idx>=143)&(idx<n-hold-2)]
        for p in idx:
            p=int(p); xp=p+1+hold
            if p<nxt or xp>=n or not mask[xp] or not np.isfinite(fac[p]): continue
            loc.append((float(fac[p]),float(mn[p]),float(rr[p]))); nxt=xp
        out[sp]=stats(loc,years[sp])
    return out


def score(st):
    t,e,y=st['test2024'],st['eval2025'],st['ytd2026']
    valid=t['trade_entries']>=20 and e['trade_entries']>=15 and y['trade_entries']>=5
    oos=min(t['cagr_to_strict_mdd'], e['cagr_to_strict_mdd'])
    positive=t['cagr_pct']>0 and e['cagr_pct']>0
    return (valid, positive, oos, y['cagr_to_strict_mdd'], t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct'], t['trade_entries']+e['trade_entries'])


def variants(template):
    name,spec=template
    # deterministic local variants: shift all non-session gates, and shift one gate at a time.
    yielded=set()
    def emit(suf, sp):
        key=tuple((c,o,round(q,3)) for c,o,q in sp)
        if key not in yielded:
            yielded.add(key); yield (name+suf, sp)
    for x in emit('', spec): yield x
    for sh in Q_SHIFTS:
        if abs(sh)<1e-12: continue
        sp=[(c,o,(q if c=='a_early_session' else clampq(q+sh if o=='>=' else q-sh))) for c,o,q in spec]
        for x in emit(f'_all{sh:+.2f}', sp): yield x
    for i in range(len(spec)):
        c,o,q=spec[i]
        if c=='a_early_session': continue
        for sh in Q_SHIFTS:
            if abs(sh)<1e-12: continue
            sp=list(spec); sp[i]=(c,o,clampq(q+sh if o=='>=' else q-sh))
            for x in emit(f'_g{i}{sh:+.2f}', sp): yield x
    # drop one non-session gate to avoid overfiltering.
    for i,(c,o,q) in enumerate(spec):
        if c=='a_early_session': continue
        sp=[x for j,x in enumerate(spec) if j!=i]
        for x in emit(f'_drop{i}', sp): yield x


def main():
    m,_,masks,years=load_market_and_splits(); f=add_features(m); train=masks['train']
    rows=[]
    for side,templates in [('long',LONG_TEMPLATES),('short',SHORT_TEMPLATES)]:
        for tmpl in templates:
            for name,spec in variants(tmpl):
                terms=materialize(f,train,spec)
                if terms is None: continue
                # Cheap train active-rate filter to avoid almost-empty or always-on rules.
                ar=float(act(f,terms)[train].mean())
                if ar<0.002 or ar>0.35: continue
                for hold in HOLDS:
                    for stride in STRIDES:
                        st=eval_rule(m,f,masks,years,terms,side,hold,stride)
                        rows.append({'name':name,'side':side,'active_rate_train':ar,'terms':[{'feature':c,'op':op,'threshold':thr,'train_q':qq} for c,op,thr,qq in terms],'hold':hold,'stride':stride,'stats':st,'score_tuple':score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    rep={'protocol':'Alpha101 primitive quantile-combo scan: 2-4 hard gates, thresholds fitted on train<2024 only, 6bp/side, strict MDD including in-position adverse excursion.','all_count':len(rows),'top':[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:150]]}
    Path(OUT).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}/{s['bar_sharpe_like']:.2f}"
    md=['# Alpha101 primitive quantile-combo scan (2026-07-09)','',rep['protocol'],'','| rank | name | side | active train | hold/stride | train ratio/trades | 2024 ret/CAGR/MDD/ratio/trades/win/sh | 2025 | 2026 | terms |','|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    for i,r in enumerate(rep['top'][:60],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} q{t['train_q']:.2f}({t['threshold']:.4g})" for t in r['terms'])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['active_rate_train']:.3f} | {r['hold']}/{r['stride']} | {st['train']['cagr_to_strict_mdd']:.2f}/{st['train']['trade_entries']} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'all_count':len(rows),'top':rep['top'][:12]},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
