import json, math
from pathlib import Path
import numpy as np, pandas as pd
from training.evaluate_portfolio_llm_selector import _prep, _event_return, SPLITS

OUT='results/orthogonal_alpha_search_2026-07-08.json'
DOC='docs/orthogonal-alpha-search-2026-07-08.md'
COST=0.0005

def q(feat, mask, col, qq):
    vals=feat.loc[mask,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return None
    return float(np.quantile(vals,qq))

def add_extra(m, f):
    c=m.close.astype(float); o=m.open.astype(float); h=m.high.astype(float); l=m.low.astype(float)
    v=m.volume.astype(float); qv=m.quote_asset_volume.astype(float) if 'quote_asset_volume' in m else v*c
    tbq=m.taker_buy_quote.astype(float) if 'taker_buy_quote' in m else qv*.5
    def z(s,n):
        mu=s.rolling(n,min_periods=max(12,n//4)).mean(); sd=s.rolling(n,min_periods=max(12,n//4)).std(ddof=0)
        return ((s-mu)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-8,8).fillna(0)
    def ret(s,n): return (s/s.shift(n).replace(0,np.nan)-1).replace([np.inf,-np.inf],np.nan).clip(-10,10).fillna(0)
    # realized vol / range compression
    lr=np.log(c/c.shift(1)).replace([np.inf,-np.inf],np.nan).fillna(0)
    for n in [24,72,144,288,576]:
        f[f'x_ret_{n}']=ret(c,n)
        f[f'x_rvol_{n}']=lr.rolling(n,min_periods=max(12,n//4)).std(ddof=0).fillna(0)*np.sqrt(n)
        f[f'x_rvol_z_{n}']=z(f[f'x_rvol_{n}'], max(288,n*4))
    # intraday/session encodings (UTC; split-safe deterministic)
    dt=pd.to_datetime(m.date)
    hour=dt.dt.hour+dt.dt.minute/60
    f['x_hour']=hour
    f['x_asia']=((hour>=0)&(hour<8)).astype(float)
    f['x_eu']=((hour>=7)&(hour<15)).astype(float)
    f['x_us']=((hour>=13)&(hour<22)).astype(float)
    f['x_weekend']=(dt.dt.dayofweek>=5).astype(float)
    # dislocations / changes
    for col in ['funding_zscore','premium_index_zscore','kimchi_premium_zscore','dxy_zscore','dxy_momentum','taker_imbalance','oi_minus_px_4h_z','px_minus_oi_4h_z','oi_ret_4h_z','px_ret_4h_z']:
        if col not in f: f[col]=0.0
    f['x_funding_premium_spread']=f['funding_zscore']-f['premium_index_zscore']
    f['x_kimchi_dxy_spread']=f['kimchi_premium_zscore']-f['dxy_zscore']
    f['x_taker_oi_div']=f['taker_imbalance']-f['oi_ret_4h_z']/5.0
    f['x_down_vol_absorb']=(-f['x_ret_72']).clip(lower=0)*f['taker_imbalance'].clip(lower=0)
    f['x_up_oi_unwind']=f['x_ret_72'].clip(lower=0)*(-f['oi_ret_4h_z']).clip(lower=0)
    f['x_down_oi_build']=(-f['x_ret_72']).clip(lower=0)*f['oi_ret_4h_z'].clip(lower=0)
    return f.replace([np.inf,-np.inf],np.nan).fillna(0)

def active_from_terms(feat, terms):
    a=np.ones(len(feat),bool)
    for col,op,thr in terms:
        x=feat[col].to_numpy(float)
        a &= np.isfinite(x) & ((x>=thr) if op=='>=' else (x<=thr))
    return a

def stats(events, years):
    if not events: return {'total_return_pct':0,'cagr_pct':0,'strict_mdd_pct':0,'cagr_to_strict_mdd':0,'trade_entries':0,'win_rate':0,'bar_sharpe_like':0,'mean_trade_ret_pct':0}
    # events are (ret_path, adv_path, ret_bps)
    n=len(events[0][0]); r=np.zeros(n); a=np.zeros(n); b=[]
    for rr,aa,bps in events: r+=rr; a+=aa; b.append(bps/10000)
    fac=np.maximum(0,1+r); eqp=np.cumprod(fac); eqb=np.r_[1.0,eqp[:-1]]
    pka=np.maximum.accumulate(eqp); pkb=np.maximum.accumulate(eqb)
    mdd=max(float(np.nanmax(1-eqp/np.maximum(pka,1e-12))),float(np.nanmax(1-(eqb*(1+a))/np.maximum(pkb,1e-12))))*100
    eq=float(eqp[-1]); cagr=(eq**(1/years)-1)*100 if eq>0 else -100
    vals=r[np.abs(r)>1e-12]; sh=float(vals.mean()/vals.std(ddof=1)*np.sqrt(len(vals))) if len(vals)>1 and vals.std(ddof=1)>0 else 0
    b=np.array(b,float)
    return {'total_return_pct':(eq-1)*100,'cagr_pct':cagr,'strict_mdd_pct':mdd,'cagr_to_strict_mdd':cagr/mdd if mdd>1e-12 else 0,'trade_entries':len(events),'win_rate':float((b>0).mean()),'bar_sharpe_like':sh,'mean_trade_ret_pct':float(b.mean()*100)}

def eval_rule(m, feat, masks, years, terms, side, hold, stride):
    active=active_from_terms(feat,terms); out={}
    n=len(m)
    for sp,mask in masks.items():
        idx=np.flatnonzero(mask); start=idx[0]; end=idx[-1]+1; local=[]; nxt=0
        for p in np.arange(143,n-hold-2,stride,dtype=np.int64):
            xp=p+1+hold
            if p<nxt or not active[p] or not mask[p] or xp>=n or not mask[xp]: continue
            r,adv,real=_event_return(m,p,hold,side,cost=COST)
            local.append((r[start:end],adv[start:end],real*10000)); nxt=xp
        out[sp]=stats(local,years[sp])
    return out

def score(res):
    t,e,y=res['test2024'],res['eval2025'],res['ytd2026']
    ok=t['trade_entries']>=12 and e['trade_entries']>=8 and t['cagr_pct']>0 and e['cagr_pct']>0
    return (ok, min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd']), y['cagr_to_strict_mdd'], t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct'], t['trade_entries']+e['trade_entries'])

def main():
    m,feat,masks,years=_prep(); feat=add_extra(m,feat); train=masks['train']
    specs=[]
    # single and pairwise interpretable families. thresholds fit train only.
    fams=[
      ('oi_unwind_long',[('oi_minus_px_4h_z','<=',.2),('x_ret_72','>=',.6)],'long'),
      ('oi_build_short',[('oi_minus_px_4h_z','>=',.8),('x_ret_72','<=',.4)],'short'),
      ('funding_premium_revert_long',[('x_funding_premium_spread','<=',.1),('premium_index_zscore','<=',.2)],'long'),
      ('funding_premium_revert_short',[('x_funding_premium_spread','>=',.9),('premium_index_zscore','>=',.8)],'short'),
      ('kimchi_dxy_long',[('x_kimchi_dxy_spread','<=',.1),('dxy_momentum','<=',.3)],'long'),
      ('kimchi_dxy_short',[('x_kimchi_dxy_spread','>=',.9),('dxy_momentum','>=',.7)],'short'),
      ('vol_compress_break_long',[('x_rvol_z_288','<=',.2),('x_ret_24','>=',.7)],'long'),
      ('vol_compress_break_short',[('x_rvol_z_288','<=',.2),('x_ret_24','<=',.3)],'short'),
      ('down_absorb_long',[('x_down_vol_absorb','>=',.8),('x_ret_72','<=',.3)],'long'),
      ('up_unwind_short',[('x_up_oi_unwind','>=',.8),('x_ret_72','>=',.7)],'short'),
      ('session_us_momo_long',[('x_us','>=',.5),('x_ret_24','>=',.75),('taker_imbalance','>=',.6)],'long'),
      ('session_asia_fade_short',[('x_asia','>=',.5),('x_ret_24','>=',.8),('taker_imbalance','<=',.4)],'short'),
      ('weekend_reversal_long',[('x_weekend','>=',.5),('x_ret_72','<=',.2),('x_rvol_z_144','>=',.7)],'long'),
      ('weekend_reversal_short',[('x_weekend','>=',.5),('x_ret_72','>=',.8),('x_rvol_z_144','>=',.7)],'short'),
    ]
    for name,raw_terms,side in fams:
        terms=[]; bad=False
        for col,op,qq in raw_terms:
            if col in ['x_us','x_asia','x_weekend']:
                thr=qq
            else:
                thr=q(feat,train,col,qq)
                if thr is None: bad=True; break
            terms.append((col,op,thr))
        if bad: continue
        for hold in [24,48,72,96,144]:
            for stride in [6,12,24]:
                res=eval_rule(m,feat,masks,years,terms,side,hold,stride)
                specs.append({'name':name,'side':side,'terms':[{'feature':c,'op':op,'threshold':thr} for c,op,thr in terms],'hold':hold,'stride':stride,'stats':res,'score_tuple':score(res)})
    specs.sort(key=lambda r:r['score_tuple'], reverse=True)
    report={'protocol':'orthogonal alpha scan; thresholds fit on train<2024 only; test=2024/eval=2025/ytd2026 reported; 5bp per side; strict in-position MDD; candidate families avoid direct wave/volume sleeve reuse except generic vol regime; diagnostic not live-promoted','top':[{k:v for k,v in r.items() if k!='score_tuple'} for r in specs[:80]],'all_count':len(specs)}
    Path(OUT).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    md=['# Orthogonal alpha search (2026-07-08)','',report['protocol'],'','| rank | name | side | hold/stride | 2024 ratio/trades | 2025 ratio/trades | 2026 ratio/trades | terms |','|---:|---|---|---:|---:|---:|---:|---|']
    for i,r in enumerate(report['top'][:25],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} {t['threshold']:.4g}" for t in r['terms'])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['hold']}/{r['stride']} | {st['test2024']['cagr_to_strict_mdd']:.2f}/{st['test2024']['trade_entries']} | {st['eval2025']['cagr_to_strict_mdd']:.2f}/{st['eval2025']['trade_entries']} | {st['ytd2026']['cagr_to_strict_mdd']:.2f}/{st['ytd2026']['trade_entries']} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'top':report['top'][:10]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
