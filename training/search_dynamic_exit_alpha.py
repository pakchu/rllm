import json, math, glob, os
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from training.evaluate_portfolio_llm_selector import _prep, SPLITS
from training.wave_feature_ridge_policy import build_wave_feature_frame

OUT='results/dynamic_exit_alpha_scan_2026-07-08.json'
DOC='docs/dynamic-exit-alpha-scan-2026-07-08.md'
COST=.0005

def J(p): return json.loads(Path(p).read_text())
def gate(feat,gates):
    a=np.ones(len(feat),bool)
    for g in gates:
        x=feat[str(g['feature'])].to_numpy(float); thr=float(g.get('threshold',g.get('thr'))); op=str(g['op'])
        a &= np.isfinite(x)&((x>=thr) if op in ('>=','ge') else (x<=thr))
    return a

def z(s,n):
    mu=s.rolling(n,min_periods=max(20,n//4)).mean(); sd=s.rolling(n,min_periods=max(20,n//4)).std(ddof=0)
    return ((s-mu)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-8,8).fillna(0)

def ret(s,n): return (s/s.shift(n).replace(0,np.nan)-1).replace([np.inf,-np.inf],np.nan).clip(-10,10).fillna(0)

def add_features(m, f):
    f=pd.concat([f, build_wave_feature_frame(m, window=144).add_prefix('wr_')],axis=1)
    c=m.close.astype(float); h=m.high.astype(float); l=m.low.astype(float); v=m.volume.astype(float)
    qv=m.quote_asset_volume.astype(float) if 'quote_asset_volume' in m else v*c
    tbq=m.taker_buy_quote.astype(float) if 'taker_buy_quote' in m else qv*.5
    taker=(tbq/qv.replace(0,np.nan)*2-1).replace([np.inf,-np.inf],np.nan).fillna(0)
    signed=(2*tbq-qv).fillna(0); cvd=signed.cumsum()
    pc=c.shift(1); tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    f['dx_atr72']=(tr.rolling(72,min_periods=18).mean()/c.replace(0,np.nan)).fillna(0)
    for n in [24,72,144,288]:
        f[f'dx_ret_{n}']=ret(c,n)
        f[f'dx_taker_cvd_z_{n}']=z(ret(cvd.abs()+1,n)*np.sign(cvd.diff(n).fillna(0)),288)
    for W in [144,288,576]:
        rh=h.shift(1).rolling(W,min_periods=max(20,W//3)).max(); rl=l.shift(1).rolling(W,min_periods=max(20,W//3)).min(); rng=(rh-rl).replace(0,np.nan)
        f[f'dx_wave_pos_{W}']=((c-rl)/rng).replace([np.inf,-np.inf],np.nan).clip(-2,3).fillna(.5)
        ma=c.rolling(W,min_periods=max(20,W//3)).mean(); sd=c.rolling(W,min_periods=max(20,W//3)).std(ddof=0)
        f[f'dx_price_z_{W}']=((c-ma)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-5,5).fillna(0)
        f[f'dx_slope_atr_{W}']=((c.ewm(span=max(4,W//8),adjust=False).mean()-ma)/(c*f['dx_atr72']).replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-10,10).fillna(0)
    # alt/upbit volume ratio minimal
    total=None
    for p in glob.glob('data/binance_um_pool_5m_2023_2026/*_5m_*.csv.gz')[:20]:
        try:
            a=pd.read_csv(p,usecols=['date','quote_asset_volume']); a['date']=pd.to_datetime(a.date); a=a.sort_values('date')
            j=pd.merge_asof(m[['date']].sort_values('date'),a,on='date',direction='backward',tolerance=pd.Timedelta('7min')).sort_index()
            s=j.quote_asset_volume.fillna(0).astype(float); total=s if total is None else total+s
        except Exception: pass
    if total is None: total=pd.Series(0,index=m.index,dtype=float)
    f['dx_alt_ratio_z72']=z((total/qv.replace(0,np.nan)).fillna(0),72)
    for col in ['oi_ret_4h_z','oi_minus_px_4h_z','px_minus_oi_4h_z','funding_zscore','premium_index_zscore','taker_imbalance']:
        if col not in f: f[col]=0.0
    f['dx_cvd_bear']=(-f['dx_taker_cvd_z_72']).clip(lower=0)+z(f['dx_ret_72'],288).clip(lower=0)
    f['dx_cvd_bull']=f['dx_taker_cvd_z_72'].clip(lower=0)+(-z(f['dx_ret_72'],288)).clip(lower=0)
    f['dx_oi_unwind_long_exit']=(-f['oi_ret_4h_z']).clip(lower=0)+f['dx_cvd_bear'].clip(lower=0)
    f['dx_oi_unwind_short_exit']=(-f['oi_ret_4h_z']).clip(lower=0)+f['dx_cvd_bull'].clip(lower=0)
    return f.loc[:,~f.columns.duplicated(keep='last')].replace([np.inf,-np.inf],np.nan).fillna(0)

def q(feat,mask,col,qq):
    vals=feat.loc[mask,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return 0.0
    return float(np.quantile(vals,qq))

def build_entries(m, f, masks):
    entries=[]; dates=pd.to_datetime(m.date); n=len(m)
    def append_active(name, active, side, stride):
        for sp,sm in masks.items():
            for p in np.arange(143,n-3,int(stride),dtype=np.int64):
                if active[p] and sm[p]: entries.append({'name':name,'side':side,'split':sp,'pos':int(p)})
    nt=J('configs/live/nonpb30_taker_returnz_rangevol_htf4hrange_h72_candidate.json')['signal']
    append_active('nonpb30_taker',gate(f,nt['gates']),'long',int(nt['stride_bars_5m']))
    oi=J('configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json')
    oi_active=gate(f,oi['gates'])
    append_active('oi_raw',oi_active,'long',int(oi['stride_bars']))
    wthr=q(f,masks['train'],'dx_wave_pos_144',.2)
    append_active('oi_wave_lowpos144',oi_active&(f.dx_wave_pos_144.to_numpy(float)<=wthr),'long',int(oi['stride_bars']))
    vthr=q(f,masks['train'],'dx_alt_ratio_z72',.8)
    append_active('oi_alt_ratio72',oi_active&(f.dx_alt_ratio_z72.to_numpy(float)>=vthr),'long',int(oi['stride_bars']))
    # rex predictions
    rows=[]
    for fp in ['results/rex_dual_regime_train_2021_2023_predictions_2026-07-03.jsonl','results/rex_dual_regime_test_2024_predictions_2026-07-03.jsonl','results/rex_dual_regime_eval_2025_2026h1_predictions_2026-07-03.jsonl']:
        if Path(fp).exists(): rows += [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]
    for row in rows:
        pred=row.get('prediction') or {}
        if pred.get('gate')!='TRADE': continue
        side=str(pred.get('side','')).lower(); p=int(row['signal_pos'])
        if side not in ('long','short') or p<143 or p>=n-3: continue
        for sp,sm in masks.items():
            if sm[p]: entries.append({'name':'rex_rule','side':side,'split':sp,'pos':p})
    return entries

def sim_dynamic(m, masks, years, entries, exit_active, min_bars):
    op=m.open.to_numpy(float); hi=m.high.to_numpy(float); lo=m.low.to_numpy(float); n=len(m); out={}
    for sp,sm in masks.items():
        idx=np.flatnonzero(sm); start=idx[0]; end=idx[-1]; eq=peak=1.0; mdd=0.0; rets=[]; wins=0; next_allowed=0
        ev=sorted([e for e in entries if e['split']==sp], key=lambda x:x['pos'])
        for e in ev:
            p=int(e['pos'])
            if p<next_allowed: continue
            ep=p+1
            if ep>=end: continue
            side=e['side']; entry_eq=eq; eq*=1-COST; mdd=max(mdd,1-eq/max(peak,1e-12))
            xp=end
            for j in range(ep,end):
                oj=op[j]
                if not np.isfinite(oj) or oj<=0: continue
                adverse=(lo[j]-oj)/oj if side=='long' else (oj-hi[j])/oj
                rr=(op[j+1]-oj)/oj if side=='long' else (oj-op[j+1])/oj
                mdd=max(mdd,1-(eq*max(0,1+adverse))/max(peak,1e-12))
                eq*=max(0,1+rr); peak=max(peak,eq)
                if j-ep+1>=min_bars and exit_active[j]: xp=j+1; break
                if eq<=0: xp=j+1; break
            eq*=1-COST; peak=max(peak,eq); mdd=max(mdd,1-eq/max(peak,1e-12))
            r=eq/entry_eq-1; rets.append(r); wins += r>0; next_allowed=xp+1
            if eq<=0: break
        cagr=(eq**(1/years[sp])-1)*100 if eq>0 else -100; md=mdd*100
        arr=np.array(rets,float); sh=float(arr.mean()/arr.std(ddof=1)*math.sqrt(len(arr)/years[sp])) if len(arr)>1 and arr.std(ddof=1)>0 else 0
        out[sp]=dict(total_return_pct=(eq-1)*100,cagr_pct=cagr,strict_mdd_pct=md,cagr_to_strict_mdd=cagr/md if md>1e-12 else 0,trade_entries=len(rets),win_rate=wins/len(rets) if rets else 0,trade_sharpe_like=sh,avg_trade_pct=float(arr.mean()*100) if len(arr) else 0)
    return out

def score(r):
    t,e,y=r['test2024'],r['eval2025'],r['ytd2026']
    return (t['cagr_pct']>0 and e['cagr_pct']>0 and y['cagr_pct']>0, min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd']), y['cagr_to_strict_mdd'], t['total_return_pct']+e['total_return_pct']+.5*y['total_return_pct'])

def main():
    m,f,masks,years=_prep(); f=add_features(m,f); train=masks['train']; entries_all=build_entries(m,f,masks)
    report=[]
    # exit rules: side-specific applied to entry side subsets.
    long_exit_specs=[
      ('wave_upper_or_cvd_bear', lambda: (f.dx_wave_pos_144.to_numpy(float)>=q(f,train,'dx_wave_pos_144',.8)) | (f.dx_cvd_bear.to_numpy(float)>=q(f,train,'dx_cvd_bear',.85))),
      ('slope_down', lambda: f.dx_slope_atr_144.to_numpy(float)<=q(f,train,'dx_slope_atr_144',.2)),
      ('oi_unwind_or_funding_hot', lambda: (f.dx_oi_unwind_long_exit.to_numpy(float)>=q(f,train,'dx_oi_unwind_long_exit',.85)) | (f.funding_zscore.to_numpy(float)>=q(f,train,'funding_zscore',.85))),
      ('vwap_overheat', lambda: f.wr_vwap_dev_z.to_numpy(float)>=q(f,train,'wr_vwap_dev_z',.85)),
      ('opposite_oi_context', lambda: (f.oi_minus_px_4h_z.to_numpy(float)<=q(f,train,'oi_minus_px_4h_z',.2)) & (f.dx_ret_24.to_numpy(float)<0)),
    ]
    short_exit_specs=[
      ('wave_lower_or_cvd_bull', lambda: (f.dx_wave_pos_144.to_numpy(float)<=q(f,train,'dx_wave_pos_144',.2)) | (f.dx_cvd_bull.to_numpy(float)>=q(f,train,'dx_cvd_bull',.85))),
      ('slope_up', lambda: f.dx_slope_atr_144.to_numpy(float)>=q(f,train,'dx_slope_atr_144',.8)),
      ('oi_unwind_short_or_funding_cold', lambda: (f.dx_oi_unwind_short_exit.to_numpy(float)>=q(f,train,'dx_oi_unwind_short_exit',.85)) | (f.funding_zscore.to_numpy(float)<=q(f,train,'funding_zscore',.15))),
      ('vwap_washout', lambda: f.wr_vwap_dev_z.to_numpy(float)<=q(f,train,'wr_vwap_dev_z',.15)),
    ]
    for entry_name in sorted(set(e['name'] for e in entries_all)):
        for side in ['long','short']:
            ents=[e for e in entries_all if e['name']==entry_name and e['side']==side]
            if not ents: continue
            specs=long_exit_specs if side=='long' else short_exit_specs
            for exit_name,fn in specs:
                ex=np.asarray(fn(),bool)
                for min_bars in [1,6,12,24,48]:
                    res=sim_dynamic(m,masks,years,ents,ex,min_bars)
                    report.append({'entry':entry_name,'side':side,'exit':exit_name,'min_bars':min_bars,'stats':res,'score_tuple':score(res)})
    report.sort(key=lambda r:r['score_tuple'],reverse=True)
    out={'protocol':'Dynamic-exit scan: fixed holding period removed; entries from known alpha sleeves, exits from heterogeneous wave/CVD/OI/funding/vwap signals. Exit thresholds train<2024 only; positions force-closed at split end; non-overlap per entry sleeve; 5bp per side; strict in-position MDD.','top':[{k:v for k,v in r.items() if k!='score_tuple'} for r in report[:120]],'all_count':len(report)}
    Path(OUT).write_text(json.dumps(out,indent=2,ensure_ascii=False))
    md=['# Dynamic exit alpha scan (2026-07-08)','',out['protocol'],'','| rank | entry | side | exit | min bars | 2024 ret/CAGR/MDD/ratio/trades | 2025 ret/CAGR/MDD/ratio/trades | 2026 ret/CAGR/MDD/ratio/trades |','|---:|---|---|---|---:|---:|---:|---:|']
    for i,r in enumerate(out['top'][:40],1):
        st=r['stats']
        def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"
        md.append(f"| {i} | {r['entry']} | {r['side']} | {r['exit']} | {r['min_bars']} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'top':out['top'][:15]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
