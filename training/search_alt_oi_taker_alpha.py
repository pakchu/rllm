import json, glob, os, math
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from training.evaluate_portfolio_llm_selector import _prep, _event_return, SPLITS

OUT='results/alt_oi_taker_alpha_flip_search_2026-07-08.json'
DOC='docs/alt-oi-taker-alpha-flip-search-2026-07-08.md'
COST=0.0005

def z(s,n):
    mu=s.rolling(n,min_periods=max(20,n//4)).mean(); sd=s.rolling(n,min_periods=max(20,n//4)).std(ddof=0)
    return ((s-mu)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-8,8).fillna(0)

def ret(s,n):
    return (s/s.shift(n).replace(0,np.nan)-1).replace([np.inf,-np.inf],np.nan).clip(-10,10).fillna(0)

def add_alt_features(m, f, max_syms=18):
    # Pick liquid alt files by median quote volume after 2024 to keep scan bounded.
    paths=glob.glob('data/binance_um_pool_5m_2023_2026/*_5m_*.csv.gz')
    rows=[]
    for p in paths:
        try:
            a=pd.read_csv(p, usecols=['date','close','quote_asset_volume'])
            a['date']=pd.to_datetime(a.date)
            med=float(a.loc[a.date>=pd.Timestamp('2024-01-01'),'quote_asset_volume'].median())
            rows.append((med,p,os.path.basename(p).split('_')[0]))
        except Exception:
            pass
    rows=sorted(rows, reverse=True)[:max_syms]
    btc_ret_24=ret(m.close.astype(float),24)
    btc_ret_72=ret(m.close.astype(float),72)
    rets24=[]; rets72=[]; qvzs=[]; rel72=[]
    base_dates=m[['date']].sort_values('date')
    for _,p,sym in rows:
        a=pd.read_csv(p, usecols=['date','close','quote_asset_volume'])
        a['date']=pd.to_datetime(a.date)
        a=a.sort_values('date')
        joined=pd.merge_asof(base_dates, a, on='date', direction='backward', tolerance=pd.Timedelta('7min')).sort_index()
        close=joined.close.astype(float).ffill()
        qv=joined.quote_asset_volume.astype(float).fillna(0)
        r24=ret(close,24); r72=ret(close,72)
        rets24.append(r24); rets72.append(r72); rel72.append(r72-btc_ret_72); qvzs.append(z(qv,288))
        f[f'alt_{sym}_rel72']=r72-btc_ret_72
        f[f'alt_{sym}_qvz288']=z(qv,288)
    if rets24:
        R24=pd.concat(rets24,axis=1); R72=pd.concat(rets72,axis=1); REL72=pd.concat(rel72,axis=1); QVZ=pd.concat(qvzs,axis=1)
        f['alt_breadth24_pos']=(R24>0).mean(axis=1)
        f['alt_breadth72_pos']=(R72>0).mean(axis=1)
        f['alt_rel72_mean']=REL72.mean(axis=1).fillna(0)
        f['alt_rel72_topq']=REL72.quantile(.75,axis=1).fillna(0)
        f['alt_rel72_botq']=REL72.quantile(.25,axis=1).fillna(0)
        f['alt_disp72']=R72.std(axis=1).fillna(0)
        f['alt_qvz_breadth_hi']=(QVZ>1.0).mean(axis=1)
        f['alt_qvz_mean']=QVZ.mean(axis=1).fillna(0)
        f['alt_rot_riskon']=z(f['alt_rel72_mean'],288)+z(f['alt_breadth72_pos'],288)+0.5*z(f['alt_qvz_breadth_hi'],288)
        f['alt_rot_riskoff']=-z(f['alt_rel72_mean'],288)+z(f['alt_disp72'],288)+0.5*z(1-f['alt_breadth72_pos'],288)
    else:
        for c in ['alt_breadth24_pos','alt_breadth72_pos','alt_rel72_mean','alt_rel72_topq','alt_rel72_botq','alt_disp72','alt_qvz_breadth_hi','alt_qvz_mean','alt_rot_riskon','alt_rot_riskoff']:
            f[c]=0.0
    return f.replace([np.inf,-np.inf],np.nan).fillna(0), [x[2] for x in rows]

def add_oi_taker_features(m,f):
    c=m.close.astype(float); qv=m.quote_asset_volume.astype(float) if 'quote_asset_volume' in m else m.volume.astype(float)*c
    tbq=m.taker_buy_quote.astype(float) if 'taker_buy_quote' in m else qv*.5
    taker=(tbq/qv.replace(0,np.nan)*2-1).replace([np.inf,-np.inf],np.nan).fillna(0)
    signed=(2*tbq-qv).fillna(0)
    cvd=signed.cumsum()
    for n in [24,72,144,288]:
        f[f'taker_cvd_ret_{n}']=ret(cvd.abs()+1,n)*np.sign(cvd.diff(n).fillna(0))
        f[f'taker_cvd_z_{n}']=z(f[f'taker_cvd_ret_{n}'],288)
        f[f'px_ret_{n}']=ret(c,n)
        f[f'taker_div_{n}']=z(f[f'taker_cvd_ret_{n}'],288)-z(f[f'px_ret_{n}'],288)
    # OI squeeze/unwind primitives exist in _prep; robust fallback.
    for col in ['oi_ret_4h_z','oi_minus_px_4h_z','px_minus_oi_4h_z','oi_ret_8h_z','oi_minus_px_8h_z','px_minus_oi_8h_z','taker_imbalance','funding_zscore','premium_index_zscore']:
        if col not in f: f[col]=0.0
    f['oi_squeeze_long_ctx']=(-z(f['px_ret_72'],288)).clip(lower=0)+f['oi_ret_4h_z'].clip(lower=0)+f['taker_div_72'].clip(lower=0)
    f['oi_squeeze_short_ctx']=z(f['px_ret_72'],288).clip(lower=0)+f['oi_ret_4h_z'].clip(lower=0)+(-f['taker_div_72']).clip(lower=0)
    f['oi_unwind_long_ctx']=z(f['px_ret_72'],288).clip(lower=0)+(-f['oi_ret_4h_z']).clip(lower=0)+(-f['funding_zscore']).clip(lower=0)
    f['oi_unwind_short_ctx']=(-z(f['px_ret_72'],288)).clip(lower=0)+(-f['oi_ret_4h_z']).clip(lower=0)+f['funding_zscore'].clip(lower=0)
    return f.replace([np.inf,-np.inf],np.nan).fillna(0)

def qv(feat,mask,col,qq):
    vals=feat.loc[mask,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return None
    return float(np.quantile(vals,qq))

def active(feat,terms):
    a=np.ones(len(feat),bool)
    for c,op,thr in terms:
        x=feat[c].to_numpy(float); a &= np.isfinite(x)&((x>=thr) if op=='>=' else (x<=thr))
    return a

def trade_factor(m,p,hold,side):
    op=m.open.to_numpy(float); hi=m.high.to_numpy(float); lo=m.low.to_numpy(float)
    ep=int(p)+1; xp=ep+int(hold)
    if xp>=len(m): return None
    eq=1-COST; minf=eq
    for j in range(ep,xp):
        oj=op[j]
        if not np.isfinite(oj) or oj<=0: continue
        if side=='long': adverse=(lo[j]-oj)/oj; rr=(op[j+1]-oj)/oj
        else: adverse=(oj-hi[j])/oj; rr=(oj-op[j+1])/oj
        minf=min(minf, eq*max(0,1+adverse)); eq*=max(0,1+rr)
    eq*=1-COST; minf=min(minf,eq)
    return eq,minf,eq-1

def stats(local, years):
    if not local: return dict(total_return_pct=0,cagr_pct=0,strict_mdd_pct=0,cagr_to_strict_mdd=0,trade_entries=0,win_rate=0,bar_sharpe_like=0,mean_trade_ret_pct=0)
    eq=peak=1.0; mdd=0.0; rets=[]
    for fac,minfac,r in local:
        mdd=max(mdd,1-(eq*minfac)/max(peak,1e-12))
        eq*=fac; peak=max(peak,eq); mdd=max(mdd,1-eq/max(peak,1e-12)); rets.append(r)
    cagr=(eq**(1/years)-1)*100 if eq>0 else -100; md=mdd*100
    arr=np.array(rets,float)
    sh=float(arr.mean()/arr.std(ddof=1)*math.sqrt(len(arr)/max(years,1e-9))) if len(arr)>1 and arr.std(ddof=1)>0 else 0
    return dict(total_return_pct=(eq-1)*100,cagr_pct=cagr,strict_mdd_pct=md,cagr_to_strict_mdd=cagr/md if md>1e-12 else 0,trade_entries=len(local),win_rate=float((arr>0).mean()),bar_sharpe_like=sh,mean_trade_ret_pct=float(arr.mean()*100))

def eval_rule(m,feat,masks,years,terms,side,hold,stride):
    act=active(feat,terms); n=len(m); out={}
    ar=np.arange(n)
    stride_mask=(ar % int(stride))==0
    for sp,mask in masks.items():
        local=[]; nxt=0
        idx=np.flatnonzero(act & mask & stride_mask)
        idx=idx[(idx>=143) & (idx < n-hold-2)]
        for p in idx:
            xp=int(p)+1+int(hold)
            if p<nxt or xp>=n or not mask[xp]: continue
            tf=trade_factor(m,int(p),int(hold),side)
            if tf is None: continue
            local.append(tf); nxt=xp
        out[sp]=stats(local,years[sp])
    return out

def score(res):
    t,e,y=res['test2024'],res['eval2025'],res['ytd2026']
    ok=t['trade_entries']>=15 and e['trade_entries']>=10 and t['cagr_pct']>0 and e['cagr_pct']>0
    train_ok=res['train']['trade_entries']>=50
    return (ok, train_ok, min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd']), y['cagr_to_strict_mdd'], t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct'])

def main():
    m,f,masks,years=_prep(); f,syms=add_alt_features(m,f); f=add_oi_taker_features(m,f); train=masks['train']
    raw=[]
    families=[
      ('alt_rotation_long','long',[('alt_rot_riskon','>=',.80),('px_ret_72','<=',.70)]),
      ('alt_rotation_short','short',[('alt_rot_riskoff','>=',.80),('px_ret_72','>=',.30)]),
      ('alt_breadth_reversal_long','long',[('alt_breadth72_pos','<=',.20),('alt_qvz_breadth_hi','>=',.70)]),
      ('alt_breadth_reversal_short','short',[('alt_breadth72_pos','>=',.80),('alt_qvz_breadth_hi','>=',.70)]),
      ('oi_squeeze_long','long',[('oi_squeeze_long_ctx','>=',.85)]),
      ('oi_squeeze_short','short',[('oi_squeeze_short_ctx','>=',.85)]),
      ('oi_unwind_long','long',[('oi_unwind_long_ctx','>=',.85)]),
      ('oi_unwind_short','short',[('oi_unwind_short_ctx','>=',.85)]),
      ('cvd_bull_div_long','long',[('taker_div_72','>=',.85),('px_ret_72','<=',.35)]),
      ('cvd_bear_div_short','short',[('taker_div_72','<=',.15),('px_ret_72','>=',.65)]),
      ('cvd_flow_cont_long','long',[('taker_cvd_z_72','>=',.85),('px_ret_24','>=',.60)]),
      ('cvd_flow_cont_short','short',[('taker_cvd_z_72','<=',.15),('px_ret_24','<=',.40)]),
    ]
    for name,side,qs in families:
        terms=[]; bad=False
        for c,op,qq in qs:
            thr=qv(f,train,c,qq)
            if thr is None: bad=True; break
            terms.append((c,op,thr))
        if bad: continue
        for eval_name, eval_side in [(name, side), (name+'_flip', 'short' if side=='long' else 'long')]:
            for hold in [24,48,72,96]:
                for stride in [12,24]:
                    res=eval_rule(m,f,masks,years,terms,eval_side,hold,stride)
                    raw.append({'name':eval_name,'side':eval_side,'terms':[{'feature':c,'op':op,'threshold':thr} for c,op,thr in terms],'hold':hold,'stride':stride,'stats':res,'score_tuple':score(res)})
    raw.sort(key=lambda r:r['score_tuple'],reverse=True)
    report={'protocol':'alt breadth/rotation + OI squeeze/unwind + taker CVD event alpha scan. Thresholds fit train<2024 only; test/eval/ytd reported; 5bp per side strict MDD. Diagnostic not live-promoted.','alt_symbols':syms,'top':[{k:v for k,v in r.items() if k!='score_tuple'} for r in raw[:100]],'all_count':len(raw)}
    Path(OUT).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    md=['# Alt/OI/Taker orthogonal alpha search (2026-07-08)','',report['protocol'],'','Alt symbols: `'+', '.join(syms)+'`','','| rank | name | side | hold/stride | train ratio/trades | 2024 ratio/trades | 2025 ratio/trades | 2026 ratio/trades | terms |','|---:|---|---|---:|---:|---:|---:|---:|---|']
    for i,r in enumerate(report['top'][:30],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} {t['threshold']:.4g}" for t in r['terms'])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['hold']}/{r['stride']} | {st['train']['cagr_to_strict_mdd']:.2f}/{st['train']['trade_entries']} | {st['test2024']['cagr_to_strict_mdd']:.2f}/{st['test2024']['trade_entries']} | {st['eval2025']['cagr_to_strict_mdd']:.2f}/{st['eval2025']['trade_entries']} | {st['ytd2026']['cagr_to_strict_mdd']:.2f}/{st['ytd2026']['trade_entries']} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'top':report['top'][:12]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
