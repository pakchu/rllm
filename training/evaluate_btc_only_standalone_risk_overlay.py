"""Risk overlay sweep for BTC-only standalone quantile candidates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from training.search_btc_only_quantile_standalone import add_features, active
from training.search_vpin_formulaic_alpha import load_market_and_splits, stats
SRC='results/btc_only_quantile_standalone_2026-07-09.json'
OUT='results/btc_only_standalone_risk_overlay_2026-07-09.json'
DOC='docs/btc-only-standalone-risk-overlay-2026-07-09.md'
COST=0.0006
TPS=[0.006,0.01,0.015,0.02,0.03]
SLS=[None,0.006,0.01,0.015]
MAX_HOLDS=[48,96,144]
STRIDES=[24]

def trade_overlay(m,p,max_hold,side,tp,sl):
    op=m.open.to_numpy(float); hi=m.high.to_numpy(float); lo=m.low.to_numpy(float)
    ep=int(p)+1; xp=min(ep+int(max_hold),len(m)-1)
    if ep>=len(m) or xp>=len(m): return None
    entry=op[ep]
    if not np.isfinite(entry) or entry<=0: return None
    eq=1-COST; min_rel=eq; exit_px=op[xp]
    for j in range(ep,xp):
        if side=='long':
            adverse=(lo[j]-entry)/entry; favorable=(hi[j]-entry)/entry
            if sl is not None and adverse<=-sl:
                exit_px=entry*(1-sl); min_rel=min(min_rel,eq*(1-sl)); break
            if tp is not None and favorable>=tp:
                exit_px=entry*(1+tp); min_rel=min(min_rel,eq*(1+tp)); break
            min_rel=min(min_rel,eq*max(0,1+adverse)); exit_px=op[j+1]
        else:
            adverse=(entry-hi[j])/entry; favorable=(entry-lo[j])/entry
            if sl is not None and adverse<=-sl:
                exit_px=entry*(1+sl); min_rel=min(min_rel,eq*(1-sl)); break
            if tp is not None and favorable>=tp:
                exit_px=entry*(1-tp); min_rel=min(min_rel,eq*(1+tp)); break
            min_rel=min(min_rel,eq*max(0,1+adverse)); exit_px=op[j+1]
    r=(exit_px-entry)/entry if side=='long' else (entry-exit_px)/entry
    fac=max(0,1+r)*(1-COST); min_rel=min(min_rel,fac)
    return fac,min_rel,fac-1

def eval_terms(m,f,masks,years,terms,side,max_hold,stride,tp,sl):
    act=active(f,[(t['feature'],t['op'],t['threshold'],t.get('train_q',0.5)) for t in terms])
    smod=(np.arange(len(m))%stride)==0; out={}
    for sp,mask in masks.items():
        loc=[]; nxt=0; idx=np.flatnonzero(act & mask & smod); idx=idx[(idx>=300)&(idx<len(m)-max_hold-2)]
        for p in idx:
            p=int(p); xp=p+1+max_hold
            if p<nxt or xp>=len(m) or not mask[xp]: continue
            tf=trade_overlay(m,p,max_hold,side,tp,sl)
            if tf is not None: loc.append(tf); nxt=xp
        out[sp]=stats(loc,years[sp])
    return out

def score(st):
    t,e,y,tr=st['test2024'],st['eval2025'],st['ytd2026'],st['train']
    pos=t['cagr_pct']>0 and e['cagr_pct']>0 and y['cagr_pct']>0
    enough=t['trade_entries']>=20 and e['trade_entries']>=15 and y['trade_entries']>=8
    train_ok=tr['cagr_pct']>-10 and tr['trade_entries']>=40
    minr=min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd'],y['cagr_to_strict_mdd'])
    oos=min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd'])
    ret=t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct']
    return (pos and enough and train_ok, minr, oos, ret, y['cagr_to_strict_mdd'])

def main():
    data=json.load(open(SRC))['top'][:12]
    seen=set(); rules=[]
    for r in data:
        key=(r['name'],r['side'],json.dumps(r['terms'],sort_keys=True))
        if key not in seen:
            seen.add(key); rules.append(r)
    m,_,masks,years=load_market_and_splits(); f=add_features(m)
    rows=[]
    for r in rules:
        for h in MAX_HOLDS:
            for stride in STRIDES:
                for tp in TPS:
                    for sl in SLS:
                        st=eval_terms(m,f,masks,years,r['terms'],r['side'],h,stride,tp,sl)
                        rows.append({'base_name':r['name'],'side':r['side'],'terms':r['terms'],'max_hold':h,'stride':stride,'take_profit':tp,'stop_loss':sl,'stats':st,'score_tuple':score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    top=[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:180]]
    rep={'protocol':'BTC-only standalone candidates with conservative intrabar TP/SL overlay. Inputs: BTC OHLCV/taker/OI/funding/premium only. Thresholds train<2024 only; overlay diagnostic; 6bp/side; strict MDD.','all_count':len(rows),'top':top}
    Path(OUT).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}/{s['bar_sharpe_like']:.2f}"
    md=['# BTC-only standalone risk overlay (2026-07-09)','',rep['protocol'],'','| rank | base | side | hold/stride | TP/SL | train | 2024 | 2025 | 2026 | terms |','|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    for i,r in enumerate(top[:80],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} q{t.get('train_q',0):.2f}({t['threshold']:.4g})" for t in r['terms'])
        md.append(f"| {i} | {r['base_name']} | {r['side']} | {r['max_hold']}/{r['stride']} | {r['take_profit']}/{r['stop_loss']} | {fmt(st['train'])} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'all_count':len(rows),'top':top[:12]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
