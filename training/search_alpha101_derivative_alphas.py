"""Creative Alpha101-inspired derivative alpha scan for BTCUSDT 5m.

Not a literal copy of the 101 formulas. It adapts their building blocks
(open/close/high/low/vwap/volume, ts-rank/correlation, reversion/momentum)
to BTC 5m bars with train-only threshold calibration.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_vpin_formulaic_alpha import load_market_and_splits, stats, trade_arrays

OUT = "results/alpha101_derivative_alpha_scan_2026-07-09.json"
DOC = "docs/alpha101-derivative-alpha-scan-2026-07-09.md"
COST = 0.0006


def z(s: pd.Series, n: int) -> pd.Series:
    mu=s.rolling(n,min_periods=min(n,max(20,n//4))).mean(); sd=s.rolling(n,min_periods=min(n,max(20,n//4))).std(ddof=0)
    return ((s-mu)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-8,8).fillna(0)


def ret(s: pd.Series, n: int) -> pd.Series:
    return (s/s.shift(n).replace(0,np.nan)-1).replace([np.inf,-np.inf],np.nan).clip(-10,10).fillna(0)


def rank_proxy(s: pd.Series, n: int) -> pd.Series:
    return (1/(1+np.exp(-z(s,n)))).fillna(0.5)


def corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n,min_periods=min(n,max(20,n//4))).corr(b).replace([np.inf,-np.inf],np.nan).fillna(0)


def add_features(m: pd.DataFrame) -> pd.DataFrame:
    o=m.open.astype(float); h=m.high.astype(float); l=m.low.astype(float); c=m.close.astype(float)
    v=m.volume.astype(float); qv=m.quote_asset_volume.astype(float) if 'quote_asset_volume' in m else v*c
    tbq=m.taker_buy_quote.astype(float) if 'taker_buy_quote' in m else qv*0.5
    vwap=(qv/v.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(c)
    spread=((h-l)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    lr=np.log(c/c.shift(1)).replace([np.inf,-np.inf],np.nan).fillna(0)
    clv=(((c-l)-(h-c))/(h-l).replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-3,3).fillna(0)
    intr=((c-o)/(h-l).replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-3,3).fillna(0)
    taker=((2*tbq-qv)/qv.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    out=pd.DataFrame(index=m.index)
    for n in [12,24,48,72,144,288,576,864]:
        out[f'a_ret_{n}']=ret(c,n)
        out[f'a_ret_z_{n}']=z(out[f'a_ret_{n}'], max(288,n*4))
        out[f'a_vol_z_{n}']=z(qv, max(288,n*4))
        out[f'a_spread_z_{n}']=z(spread, max(288,n*4))
        out[f'a_clv_rank_{n}']=rank_proxy(clv,n)
        out[f'a_intr_rank_{n}']=rank_proxy(intr,n)
    out['a_clv']=clv; out['a_intr']=intr
    out['a_vwap_gap']=((c-vwap)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-1,1).fillna(0)
    out['a_vwap_gap_z']=z(out['a_vwap_gap'],288)
    out['a_taker_z']=z(taker,288)
    out['a_taker_72']=taker.rolling(72,min_periods=18).mean().fillna(0)
    out['a_taker_72_z']=z(out['a_taker_72'],288)
    out['a_absret_vol']=abs(lr)*qv
    out['a_absret_vol_rank']=rank_proxy(out['a_absret_vol'],288)
    out['a_ret_vol_corr_72']=corr(lr,z(qv,288),72)
    out['a_ret_vol_corr_288']=corr(lr,z(qv,288),288)
    # Compression / expansion inspired by range ranks in Alpha101 operators.
    out['a_range_compress_288']=(-z(spread.rolling(72,min_periods=18).mean(),288)).clip(-8,8).fillna(0)
    out['a_range_expand_72']=z(spread.rolling(12,min_periods=6).mean(),72).fillna(0)
    rh=h.shift(1).rolling(288,min_periods=96).max(); rl=l.shift(1).rolling(288,min_periods=96).min(); rng=(rh-rl).replace(0,np.nan)
    out['a_pos_288']=((c-rl)/rng).replace([np.inf,-np.inf],np.nan).clip(-2,3).fillna(0.5)
    out['a_break_high_288']=((c-rh)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-1,1).fillna(0)
    out['a_break_low_288']=((rl-c)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-1,1).fillna(0)
    # Daily session gap/open reversion adapted from Alpha101 coarse examples.
    d=m.date.dt.floor('D')
    day_open=o.groupby(d).transform('first')
    daily_close=c.groupby(d).last().sort_index()
    prev_daily_close=d.map(daily_close.shift(1))
    day_minute=((m.date-d).dt.total_seconds()/300).astype(int)
    out['a_day_gap']=np.log(day_open/prev_daily_close.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0).clip(-1,1)
    out['a_early_session']=(day_minute<24).astype(float)
    out['a_late_session']=(day_minute>240).astype(float)
    # Creative composites: signs tested separately.
    out['x101_vwap_exhaust_up']=out['a_vwap_gap_z'].clip(lower=0)+out['a_clv'].clip(lower=0)+out['a_absret_vol_rank']
    out['x101_vwap_exhaust_down']=(-out['a_vwap_gap_z']).clip(lower=0)+(-out['a_clv']).clip(lower=0)+out['a_absret_vol_rank']
    out['x101_clean_breakout_up']=out['a_break_high_288'].clip(lower=0)*1000 + out['a_range_compress_288'].clip(lower=0) + out['a_taker_72_z'].clip(lower=0)
    out['x101_clean_breakout_down']=out['a_break_low_288'].clip(lower=0)*1000 + out['a_range_compress_288'].clip(lower=0) + (-out['a_taker_72_z']).clip(lower=0)
    out['x101_lowvol_intraday_momo_up']=(-out['a_spread_z_144']).clip(lower=0)+out['a_intr'].clip(lower=0)+out['a_ret_z_24'].clip(lower=0)
    out['x101_lowvol_intraday_momo_down']=(-out['a_spread_z_144']).clip(lower=0)+(-out['a_intr']).clip(lower=0)+(-out['a_ret_z_24']).clip(lower=0)
    out['x101_volcorr_reversal_up']=(-out['a_ret_vol_corr_72']).clip(lower=0)+(-out['a_ret_z_72']).clip(lower=0)+out['a_absret_vol_rank']
    out['x101_volcorr_reversal_down']=out['a_ret_vol_corr_72'].clip(lower=0)+out['a_ret_z_72'].clip(lower=0)+out['a_absret_vol_rank']
    out['x101_gap_revert_up']=(-out['a_day_gap']).clip(lower=0)*out['a_early_session']+(-out['a_ret_z_12']).clip(lower=0)
    out['x101_gap_revert_down']=out['a_day_gap'].clip(lower=0)*out['a_early_session']+out['a_ret_z_12'].clip(lower=0)
    return out.replace([np.inf,-np.inf],np.nan).fillna(0)


def qthr(feat, train, col, qq):
    vals=feat.loc[train,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return None
    return float(np.quantile(vals, qq))


def active(feat, terms):
    a=np.ones(len(feat),bool)
    for col,op,thr in terms:
        x=feat[col].to_numpy(float)
        a &= np.isfinite(x) & ((x>=thr) if op=='>=' else (x<=thr))
    return a


def eval_rule(m, feat, masks, years, terms, side, hold, stride):
    act=active(feat,terms); n=len(m); ar=np.arange(n); smod=(ar%stride)==0
    fac,mn,rr=trade_arrays(m,hold,side)
    out={}
    for sp,mask in masks.items():
        loc=[]; nxt=0
        idx=np.flatnonzero(act & mask & smod); idx=idx[(idx>=143)&(idx<n-hold-2)]
        for p in idx:
            p=int(p); xp=p+1+hold
            if p<nxt or xp>=n or not mask[xp] or not np.isfinite(fac[p]): continue
            loc.append((float(fac[p]),float(mn[p]),float(rr[p]))); nxt=xp
        out[sp]=stats(loc, years[sp])
    return out


def score(st):
    t,e,y=st['test2024'],st['eval2025'],st['ytd2026']
    valid=t['trade_entries']>=20 and e['trade_entries']>=15 and y['trade_entries']>=5
    oos=min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd'])
    return (valid, oos>=2 and t['cagr_pct']>0 and e['cagr_pct']>0, oos, y['cagr_to_strict_mdd'], t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct'], t['trade_entries']+e['trade_entries'])


def main():
    m,_,masks,years=load_market_and_splits()
    feat=add_features(m); train=masks['train']
    families=[
        ('vwap_exhaust_up_revert','short',[('x101_vwap_exhaust_up','>=',None)]),
        ('vwap_exhaust_up_cont','long',[('x101_vwap_exhaust_up','>=',None)]),
        ('vwap_exhaust_down_revert','long',[('x101_vwap_exhaust_down','>=',None)]),
        ('vwap_exhaust_down_cont','short',[('x101_vwap_exhaust_down','>=',None)]),
        ('clean_breakout_up','long',[('x101_clean_breakout_up','>=',None)]),
        ('clean_breakout_down','short',[('x101_clean_breakout_down','>=',None)]),
        ('lowvol_momo_up','long',[('x101_lowvol_intraday_momo_up','>=',None)]),
        ('lowvol_momo_down','short',[('x101_lowvol_intraday_momo_down','>=',None)]),
        ('volcorr_reversal_up','long',[('x101_volcorr_reversal_up','>=',None)]),
        ('volcorr_reversal_down','short',[('x101_volcorr_reversal_down','>=',None)]),
        ('gap_revert_up','long',[('x101_gap_revert_up','>=',None),('a_early_session','>=',0.5)]),
        ('gap_revert_down','short',[('x101_gap_revert_down','>=',None),('a_early_session','>=',0.5)]),
        ('alpha101_intraday_delay_long','long',[('a_intr_rank_288','>=',None),('a_absret_vol_rank','>=',None)]),
        ('alpha101_intraday_delay_short','short',[('a_intr_rank_288','<=',None),('a_absret_vol_rank','>=',None)]),
        ('alpha42_vwap_revert_long','long',[('a_vwap_gap_z','<=',None),('a_absret_vol_rank','>=',None)]),
        ('alpha42_vwap_revert_short','short',[('a_vwap_gap_z','>=',None),('a_absret_vol_rank','>=',None)]),
    ]
    rows=[]
    qs_main=[0.70,0.75,0.80,0.85,0.90,0.93]
    qs_low=[0.30,0.25,0.20,0.15,0.10,0.07]
    for name,side,base_terms in families:
        for qmain in qs_main:
            terms=[]; ok=True
            for col,op,qq in base_terms:
                if qq is None:
                    useq = qmain if op=='>=' else qs_low[qs_main.index(qmain)]
                    thr=qthr(feat,train,col,useq)
                    if thr is None: ok=False; break
                    terms.append((col,op,thr))
                else:
                    terms.append((col,op,float(qq)))
            if not ok: continue
            for hold in [12,24,36,48,72,96,144,192]:
                for stride in [6,12,24,36]:
                    st=eval_rule(m,feat,masks,years,terms,side,hold,stride)
                    rows.append({'name':name,'side':side,'q':qmain,'terms':[{'feature':c,'op':op,'threshold':thr} for c,op,thr in terms],'hold':hold,'stride':stride,'stats':st,'score_tuple':score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    report={'protocol':'Alpha101-inspired derivative scan; train-only quantile thresholds; 6bp/side; strict in-position MDD; diagnostic not live-promoted.','all_count':len(rows),'top':[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:120]]}
    Path(OUT).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}"
    md=['# Alpha101 derivative alpha scan (2026-07-09)','',report['protocol'],'','| rank | name | side | q | hold/stride | train | 2024 ret/CAGR/MDD/ratio/trades/win | 2025 | 2026 | terms |','|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    for i,r in enumerate(report['top'][:40],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} {t['threshold']:.4g}" for t in r['terms'])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['q']:.2f} | {r['hold']}/{r['stride']} | {st['train']['cagr_to_strict_mdd']:.2f}/{st['train']['trade_entries']} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'all_count':len(rows),'top':report['top'][:10]},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
