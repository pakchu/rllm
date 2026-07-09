"""BTC-only standalone quantile-combo alpha search.

No cross-asset inputs.  Uses BTC OHLCV/taker flow, BTC open interest, BTC
funding, and BTC premium/basis.  All thresholds are train<2024 quantiles.
Evaluation uses the repo strict event accounting with 6bp/side and period-contained exits.
"""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training.search_vpin_formulaic_alpha import load_market_and_splits, stats, trade_arrays

OUT='results/btc_only_quantile_standalone_2026-07-09.json'
DOC='docs/btc-only-quantile-standalone-2026-07-09.md'
HOLDS=[48,96,144]
STRIDES=[24]

def z(s: pd.Series, n: int) -> pd.Series:
    mu=s.rolling(n,min_periods=min(n,max(20,n//4))).mean(); sd=s.rolling(n,min_periods=min(n,max(20,n//4))).std(ddof=0)
    return ((s-mu)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-8,8).fillna(0)

def ret(s: pd.Series, n: int) -> pd.Series:
    return (s/s.shift(n).replace(0,np.nan)-1).replace([np.inf,-np.inf],np.nan).clip(-5,5).fillna(0)

def merge_asof_value(dates: pd.Series, path: str, value_col: str, out_col: str, tolerance: str) -> pd.Series:
    p=Path(path)
    if not p.exists(): return pd.Series(0.0,index=dates.index)
    raw=pd.read_csv(p)
    tcol='date' if 'date' in raw.columns else raw.columns[0]
    raw[tcol]=pd.to_datetime(raw[tcol],utc=True,errors='coerce').dt.tz_convert(None)
    if value_col not in raw.columns: return pd.Series(0.0,index=dates.index)
    r=raw[[tcol,value_col]].dropna().sort_values(tcol).rename(columns={tcol:'date',value_col:out_col})
    base=pd.DataFrame({'date':dates})
    x=pd.merge_asof(base.sort_values('date'),r,on='date',direction='backward',tolerance=pd.Timedelta(tolerance))
    return x[out_col].astype(float).fillna(0).reset_index(drop=True)

def add_features(m: pd.DataFrame) -> pd.DataFrame:
    o=m.open.astype(float).reset_index(drop=True); h=m.high.astype(float).reset_index(drop=True); l=m.low.astype(float).reset_index(drop=True); c=m.close.astype(float).reset_index(drop=True)
    v=m.volume.astype(float).reset_index(drop=True); qv=(m.quote_asset_volume if 'quote_asset_volume' in m else m.volume*m.close).astype(float).reset_index(drop=True)
    tbq=(m.taker_buy_quote if 'taker_buy_quote' in m else qv*0.5).astype(float).reset_index(drop=True)
    dates=m.date.reset_index(drop=True)
    vwap=(qv/v.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(c)
    spread=((h-l)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    lr=np.log(c/c.shift(1)).replace([np.inf,-np.inf],np.nan).fillna(0)
    taker=((2*tbq-qv)/qv.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    signed=(2*tbq-qv).fillna(0); cvd=signed.cumsum()
    f=pd.DataFrame(index=m.index)
    for n in [6,12,24,48,72,144,288,576]:
        f[f'px_ret_{n}']=ret(c,n)
        f[f'px_ret_z_{n}']=z(f[f'px_ret_{n}'],max(288,n*4))
        f[f'qv_z_{n}']=z(qv,max(288,n*4))
        f[f'spread_z_{n}']=z(spread,max(288,n*4))
        f[f'rvol_z_{n}']=z(lr.rolling(n,min_periods=max(6,n//4)).std(ddof=0).fillna(0),max(288,n*4))
        f[f'taker_mean_z_{n}']=z(taker.rolling(n,min_periods=max(6,n//4)).mean().fillna(0),max(288,n*4))
        f[f'cvd_ret_z_{n}']=z(ret(cvd.abs()+1,n)*np.sign(cvd.diff(n).fillna(0)),max(288,n*4))
    f['vwap_gap_z']=z(((c-vwap)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0),288)
    rh=h.shift(1).rolling(288,min_periods=96).max(); rl=l.shift(1).rolling(288,min_periods=96).min(); rng=(rh-rl).replace(0,np.nan)
    f['pos_288']=((c-rl)/rng).replace([np.inf,-np.inf],np.nan).clip(-2,3).fillna(0.5)
    f['range_compress_288']=(-z(spread.rolling(72,min_periods=18).mean(),288)).clip(-8,8).fillna(0)
    f['range_expand_72']=z(spread.rolling(12,min_periods=6).mean(),72).fillna(0)
    f['taker_div_72']=f['cvd_ret_z_72']-f['px_ret_z_72']
    f['taker_div_144']=f['cvd_ret_z_144']-f['px_ret_z_144']
    # BTC-only aux: OI, funding, premium.
    oi=merge_asof_value(dates,'/tmp/btcusdt_open_interest_5m_2020_2026.csv','open_interest','open_interest','7min')
    f['oi_available']=(oi>0).astype(float)
    oi=oi.replace(0,np.nan).ffill().fillna(0)
    for n in [24,48,72,144,288]:
        f[f'oi_ret_z_{n}']=z(ret(oi,n),max(288,n*4))
        f[f'oi_minus_px_z_{n}']=f[f'oi_ret_z_{n}']-f[f'px_ret_z_{n}']
        f[f'px_minus_oi_z_{n}']=f[f'px_ret_z_{n}']-f[f'oi_ret_z_{n}']
    funding=merge_asof_value(dates,'data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz','funding_rate','funding_rate','9h')
    premium=merge_asof_value(dates,'data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz','close','premium_close','2h')
    f['funding_z']=z(funding,288); f['premium_z']=z(premium,288); f['basis_stress_z']=z(premium-funding,288)
    # BTC-only composite contexts, used as normal quantile-gated features.
    f['btc_oi_unwind_long']=f['px_ret_z_72'].clip(lower=0)+(-f['oi_ret_z_72']).clip(lower=0)+(-f['funding_z']).clip(lower=0)+(-f['premium_z']).clip(lower=0)
    f['btc_oi_squeeze_short']=f['px_ret_z_72'].clip(lower=0)+f['oi_ret_z_72'].clip(lower=0)+(-f['taker_div_72']).clip(lower=0)
    f['btc_liq_revert_long']=(-f['px_ret_z_24']).clip(lower=0)+f['rvol_z_24'].clip(lower=0)+(-f['taker_mean_z_24']).clip(lower=0)+f['oi_ret_z_24'].clip(lower=0)
    f['btc_cvd_absorb_long']=(-f['px_ret_z_72']).clip(lower=0)+f['taker_div_72'].clip(lower=0)+f['pos_288'].rsub(1).clip(lower=0)
    f['btc_overheat_short']=f['px_ret_z_72'].clip(lower=0)+f['premium_z'].clip(lower=0)+f['funding_z'].clip(lower=0)+f['taker_mean_z_72'].clip(lower=0)
    hour=dates.dt.hour.to_numpy(); f['us_session']=((hour>=16)&(hour<24)).astype(float); f['asia_session']=((hour>=0)&(hour<8)).astype(float)
    return f.replace([np.inf,-np.inf],np.nan).fillna(0)

def qthr(f, train, col, q):
    use=train.copy()
    if col.startswith('oi_') or 'oi_' in col or col.startswith('btc_oi'):
        use = use & (f['oi_available'].to_numpy(float)>0.5)
    vals=f.loc[use,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return None
    return float(np.quantile(vals,q))

def materialize(f,train,spec):
    out=[]
    for col,op,q in spec:
        thr=0.5 if col.endswith('_session') else qthr(f,train,col,q)
        if thr is None: return None
        out.append((col,op,float(thr),float(q)))
    return out

def active(f,terms):
    a=np.ones(len(f),bool)
    for col,op,thr,_ in terms:
        x=f[col].to_numpy(float); a &= np.isfinite(x) & ((x>=thr) if op=='>=' else (x<=thr))
    return a

def eval_rule(m,f,masks,years,terms,side,hold,stride):
    act=active(f,terms); smod=(np.arange(len(m))%stride)==0; fac,mn,rr=trade_arrays(m,hold,side); out={}
    for sp,mask in masks.items():
        loc=[]; nxt=0; idx=np.flatnonzero(act & mask & smod); idx=idx[(idx>=300)&(idx<len(m)-hold-2)]
        for p in idx:
            p=int(p); xp=p+1+hold
            if p<nxt or xp>=len(m) or not mask[xp] or not np.isfinite(fac[p]): continue
            loc.append((float(fac[p]),float(mn[p]),float(rr[p]))); nxt=xp
        out[sp]=stats(loc,years[sp])
    return out

def score(st):
    t,e,y,tr=st['test2024'],st['eval2025'],st['ytd2026'],st['train']
    pos=t['cagr_pct']>0 and e['cagr_pct']>0 and y['cagr_pct']>0
    enough=t['trade_entries']>=25 and e['trade_entries']>=20 and y['trade_entries']>=8
    train_ok=tr['cagr_pct']>-10 and tr['trade_entries']>=40
    minr=min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd'],y['cagr_to_strict_mdd'])
    oos=min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd'])
    ret=t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct']
    return (pos and enough and train_ok, minr, oos, ret, y['cagr_to_strict_mdd'])

def name_for(side,spec):
    raw=side+'|'+';'.join(f'{c}{o}{q:.2f}' for c,o,q in spec)
    return 'btcq_'+hashlib.md5(raw.encode()).hexdigest()[:10]

def gen_specs(seed=907, n=800):
    rng=random.Random(seed)
    long_blocks=[
      [('btc_liq_revert_long','>=',0.75),('pos_288','<=',0.40)],
      [('btc_cvd_absorb_long','>=',0.75),('range_compress_288','>=',0.55)],
      [('btc_oi_unwind_long','>=',0.75),('premium_z','<=',0.45)],
      [('px_ret_z_72','<=',0.30),('oi_minus_px_z_72','>=',0.55),('taker_div_72','>=',0.55)],
      [('vwap_gap_z','<=',0.25),('rvol_z_24','>=',0.55),('taker_mean_z_24','<=',0.45)],
      [('px_ret_z_24','<=',0.35),('qv_z_24','>=',0.55),('premium_z','<=',0.45)],
    ]
    short_blocks=[
      [('btc_overheat_short','>=',0.75),('pos_288','>=',0.60)],
      [('btc_oi_squeeze_short','>=',0.75),('range_expand_72','>=',0.55)],
      [('px_ret_z_72','>=',0.70),('oi_minus_px_z_72','>=',0.55),('taker_div_72','<=',0.45)],
      [('vwap_gap_z','>=',0.75),('rvol_z_24','>=',0.55),('taker_mean_z_24','>=',0.55)],
      [('px_ret_z_24','>=',0.65),('qv_z_24','>=',0.55),('premium_z','>=',0.55)],
    ]
    opts_long=[('us_session','>=',0.5),('asia_session','>=',0.5),('qv_z_72','<=',0.60),('qv_z_72','>=',0.55),('funding_z','<=',0.50),('basis_stress_z','<=',0.45),('spread_z_72','<=',0.55),('oi_available','>=',0.5)]
    opts_short=[('us_session','>=',0.5),('asia_session','>=',0.5),('qv_z_72','<=',0.60),('qv_z_72','>=',0.55),('funding_z','>=',0.50),('basis_stress_z','>=',0.55),('spread_z_72','<=',0.55),('oi_available','>=',0.5)]
    seen=set()
    def jitter(t):
        c,o,q=t
        if c.endswith('_session') or c=='oi_available': return t
        return (c,o,max(0.05,min(0.95,q+rng.choice([-0.15,-0.10,-0.05,0,0.05,0.10,0.15]))))
    for side,blocks,opts in [('long',long_blocks,opts_long),('short',short_blocks,opts_short)]:
        for b in blocks:
            for _ in range(55):
                spec=[jitter(x) for x in b]+[jitter(x) for x in rng.sample(opts,rng.randint(0,2))]
                out=[]; used=set()
                for t in spec:
                    if t[0] not in used: used.add(t[0]); out.append(t)
                key=(side,tuple(out))
                if key not in seen: seen.add(key); yield side,out
    for _ in range(n):
        side=rng.choice(['long','short']); blocks=long_blocks if side=='long' else short_blocks; opts=opts_long if side=='long' else opts_short
        spec=[]
        for b in rng.sample(blocks,rng.randint(1,2)): spec += [jitter(x) for x in b]
        spec += [jitter(x) for x in rng.sample(opts,rng.randint(0,2))]
        out=[]; used=set()
        for t in spec:
            if t[0] not in used: used.add(t[0]); out.append(t)
        if 2<=len(out)<=5:
            key=(side,tuple(out))
            if key not in seen: seen.add(key); yield side,out

def main():
    m,_,masks,years=load_market_and_splits(); f=add_features(m); train=masks['train']
    rows=[]; tested=0
    for side,spec in gen_specs():
        terms=materialize(f,train,spec)
        if terms is None: continue
        a=active(f,terms); ar=float(a[train].mean())
        if ar<0.002 or ar>0.35: continue
        tested+=1
        for hold in HOLDS:
            for stride in STRIDES:
                st=eval_rule(m,f,masks,years,terms,side,hold,stride)
                rows.append({'name':name_for(side,spec),'side':side,'active_rate_train':ar,'terms':[{'feature':c,'op':op,'threshold':thr,'train_q':q} for c,op,thr,q in terms],'hold':hold,'stride':stride,'stats':st,'score_tuple':score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    top=[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:200]]
    rep={'protocol':'BTC-only standalone quantile-combo search. Inputs: BTC OHLCV/taker, BTC OI, BTC funding, BTC premium. No other coin/asset. Thresholds train<2024 only; cost 6bp/side; strict MDD includes in-position adverse excursion and forced period-contained exits.','tested_specs':tested,'all_count':len(rows),'top':top}
    Path(OUT).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}/{s['bar_sharpe_like']:.2f}"
    md=['# BTC-only standalone quantile search (2026-07-09)','',rep['protocol'],'',f"tested_specs={tested}, all_count={len(rows)}",'','| rank | name | side | active | hold/stride | train | 2024 | 2025 | 2026 | terms |','|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    for i,r in enumerate(top[:80],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} q{t['train_q']:.2f}({t['threshold']:.4g})" for t in r['terms'])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['active_rate_train']:.3f} | {r['hold']}/{r['stride']} | {fmt(st['train'])} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'tested_specs':tested,'all_count':len(rows),'top':top[:12]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
