"""Walk-forward linear alpha surface search.

Different from fixed quantile rules: each month fits a small ridge model using
only rows before that month, then trades the month from prior-only predictions.
Research diagnostic, not live-grade promotion.
"""
from __future__ import annotations

import argparse, json, math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_alpha101_derivative_alphas import add_features as add_alpha101_features
from training.search_vpin_formulaic_alpha import add_vpin_formulaic_features, stats, trade_arrays
from training.search_low_corr_feature_alpha import _add_oi_derived_features_local

SPLIT_BOUNDS={
 'train':('2020-01-01','2024-01-01'),
 'test2024':('2024-01-01','2025-01-01'),
 'eval2025':('2025-01-01','2026-01-01'),
 'ytd2026':('2026-01-01',None),
}

@dataclass(frozen=True)
class WalkForwardLinearAlphaConfig:
    input_csv: str='data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz'
    funding_csv: str='data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz'
    premium_csv: str='data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz'
    output: str='results/walkforward_linear_alpha_scan_2026-07-12.json'
    docs_output: str='docs/walkforward-linear-alpha-scan-2026-07-12.md'
    exclude_from: str='2026-06-02'
    train_start: str='2020-01-01'
    first_trade_month: str='2024-01-01'
    window_size: int=144
    horizons: str='24,72,144'
    holds: str='24,72,144'
    ridges: str='1,10,100,1000'
    quantiles: str='0.90,0.95,0.98'
    strides: str='12,24'
    min_train_rows: int=50000
    top_k: int=80

FEATURES=[
 'trend_12','trend_24','trend_96','return_zscore_48','bb_z','range_pos','taker_imbalance',
 'funding_rate','funding_zscore','premium_index_zscore','premium_index_change',
 'oi_change','oi_zscore','oi_minus_px_z_72','oi_minus_px_z_288','px_minus_oi_z_288',
 'a_ret_z_12','a_ret_z_24','a_ret_z_72','a_vwap_gap_z','a_ret_vol_corr_72','a_ret_vol_corr_288',
 'a_absret_vol_rank','a_clv_rank_288','a_intr_rank_288','a_early_session','a_late_session',
 'vp_ret_rank_72','vp_imb_z_144','vp_vpin_z_72','vx_lowtox_momo_short','btc_cvd_absorb_long','btc_oi_squeeze_short',
]

def parse_list(s, typ): return [typ(x.strip()) for x in str(s).split(',') if x.strip()]

def years(start,end): return max((end-start).total_seconds()/(365.25*24*3600),1e-9)

def load_market(cfg):
    m=pd.read_csv(cfg.input_csv,parse_dates=['date'],compression='infer')
    m['date']=pd.to_datetime(m['date'],utc=True,errors='raise').dt.tz_convert(None)
    m=m.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
    m=m[m.date < pd.Timestamp(cfg.exclude_from)].reset_index(drop=True)
    m=attach_binance_um_aux_features(m,funding_csv=cfg.funding_csv,premium_csv=cfg.premium_csv,funding_tolerance='12h',premium_tolerance='2h')
    return m

def build_features(m,cfg):
    base=build_market_feature_frame(m,window_size=cfg.window_size)
    f=pd.concat([base,build_interest_features(m,base)],axis=1)
    f=_add_oi_derived_features_local(m,f)
    f=add_vpin_formulaic_features(m,f)
    f=pd.concat([f,add_alpha101_features(m)],axis=1)
    f=f.loc[:,~f.columns.duplicated()].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    cols=[c for c in FEATURES if c in f.columns and float(np.std(f[c].to_numpy(float)))>1e-12]
    return f[cols].copy(), cols

def split_masks(dates):
    end_data=dates.max()+pd.Timedelta(minutes=5); masks={}; yrs={}
    for k,(s,e) in SPLIT_BOUNDS.items():
        st=pd.Timestamp(s); en=pd.Timestamp(e) if e else end_data; en=min(en,end_data)
        masks[k]=((dates>=st)&(dates<en)).to_numpy(bool); yrs[k]=years(st,en)
    return masks,yrs

def fwd_return(open_,h):
    entry=open_.shift(-1); exit_=open_.shift(-(1+h)); return ((exit_-entry)/entry.replace(0,np.nan)).to_numpy(float)

def fit_predict_monthly(X, y, dates, cfg, ridge):
    preds=np.full(len(dates),np.nan,float); train_start=pd.Timestamp(cfg.train_start)
    months=pd.date_range(pd.Timestamp(cfg.first_trade_month), dates.max()+pd.offsets.MonthBegin(1), freq='MS')
    summaries=[]
    for ms in months:
        me=ms+pd.offsets.MonthBegin(1)
        train=(dates>=train_start)&(dates<ms)&np.isfinite(y)
        test=(dates>=ms)&(dates<me)
        if train.sum()<cfg.min_train_rows or test.sum()==0: continue
        Xtr=X[train]; ytr=y[train]
        mu=Xtr.mean(axis=0); sd=Xtr.std(axis=0); sd[sd<1e-9]=1.0
        Xs=(Xtr-mu)/sd; ys=ytr-ytr.mean()
        reg=np.eye(Xs.shape[1])*float(ridge)
        try: w=np.linalg.solve(Xs.T@Xs+reg, Xs.T@ys)
        except np.linalg.LinAlgError: w=np.linalg.pinv(Xs.T@Xs+reg)@(Xs.T@ys)
        preds[test]=((X[test]-mu)/sd)@w + ytr.mean()
        train_pred=Xs@w + ytr.mean()
        summaries.append({'month':str(ms.date())[:7],'train_rows':int(train.sum()),'test_rows':int(test.sum()),'train_pred_std':float(np.std(train_pred))})
    return preds,summaries

def thresholds_for_monthly(preds, y, dates, cfg, q):
    hi=np.full(len(dates),np.nan); lo=np.full(len(dates),np.nan); train_start=pd.Timestamp(cfg.train_start)
    months=pd.date_range(pd.Timestamp(cfg.first_trade_month), dates.max()+pd.offsets.MonthBegin(1), freq='MS')
    for ms in months:
        me=ms+pd.offsets.MonthBegin(1)
        train=(dates>=train_start)&(dates<ms)&np.isfinite(preds)&np.isfinite(y)
        test=(dates>=ms)&(dates<me)
        vals=preds[train]
        if vals.size<cfg.min_train_rows or np.std(vals)<1e-12: continue
        hi[test]=np.quantile(vals,q); lo[test]=np.quantile(vals,1-q)
    return lo,hi

def eval_signal(m,masks,yrs,signal,side_mode,hold,stride):
    idx_mod=(np.arange(len(m))%int(stride))==0; out={}
    for split,mask in masks.items():
        local=[]; nxt=0
        for side in (['long','short'] if side_mode=='both' else [side_mode]):
            # side-specific non-overlap handled through one shared sorted stream below
            pass
        events=[]
        if side_mode in ('both','long'):
            events += [(int(i),'long') for i in np.flatnonzero((signal>0)&mask&idx_mod)]
        if side_mode in ('both','short'):
            events += [(int(i),'short') for i in np.flatnonzero((signal<0)&mask&idx_mod)]
        events.sort()
        cache={}
        for p,side in events:
            xp=p+1+int(hold)
            if p<nxt or p<300 or xp>=len(m) or not mask[xp]: continue
            key=(int(hold),side)
            if key not in cache: cache[key]=trade_arrays(m,int(hold),side)
            fac,mn,ret=cache[key]
            if not np.isfinite(fac[p]): continue
            local.append((float(fac[p]),float(mn[p]),float(ret[p]))); nxt=xp
        out[split]=stats(local,yrs[split])
    return out

def score(st):
    tr,t,e,y=st['train'],st['test2024'],st['eval2025'],st['ytd2026']
    enough=t['trade_entries']>=20 and e['trade_entries']>=20 and y['trade_entries']>=8
    pos=t['cagr_pct']>0 and e['cagr_pct']>0 and y['cagr_pct']>0
    minr=min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd'],y['cagr_to_strict_mdd'])
    return (enough and pos, minr, tr['cagr_to_strict_mdd'], t['total_return_pct']+e['total_return_pct']+0.5*y['total_return_pct'], t['trade_entries']+e['trade_entries']+y['trade_entries'])

def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"

def run(cfg):
    m=load_market(cfg); dates=pd.to_datetime(m.date); masks,yrs=split_masks(dates); f,cols=build_features(m,cfg); X=f.to_numpy(float); open_=m.open.astype(float)
    rows=[]; model_cache={}
    for h in parse_list(cfg.horizons,int):
      y=fwd_return(open_,h)
      for ridge in parse_list(cfg.ridges,float):
        preds,months=fit_predict_monthly(X,y,dates,cfg,ridge); model_cache[(h,ridge)]={'months':months}
        for q in parse_list(cfg.quantiles,float):
          lo,hi=thresholds_for_monthly(preds,y,dates,cfg,q)
          sig=np.zeros(len(m),int); sig[preds>=hi]=1; sig[preds<=lo]=-1; sig[~np.isfinite(preds)|~np.isfinite(hi)|~np.isfinite(lo)]=0
          for hold in parse_list(cfg.holds,int):
            for stride in parse_list(cfg.strides,int):
              for mode in ['both','long','short']:
                st=eval_signal(m,masks,yrs,sig,mode,hold,stride)
                row={'name':f'wflin_h{h}_r{ridge:g}_q{q}_hold{hold}_s{stride}_{mode}','horizon':h,'ridge':ridge,'quantile':q,'hold':hold,'stride':stride,'side_mode':mode,'stats':st,'score_tuple':score(st)}
                rows.append(row)
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    top=[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:cfg.top_k]]
    rep={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'input':{'rows':len(m),'start':str(m.date.iloc[0]),'end':str(m.date.iloc[-1])},'features':cols,'protocol':'Monthly expanding-window ridge. For each trade month, model and prediction thresholds are fit using rows before month start only. Ranking is diagnostic over OOS splits.','all_count':len(rows),'top':top,'leakage_guard':{'monthly_fit_uses_prior_rows_only':True,'thresholds_use_prior_predictions_only':True,'features_past_only':True}}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    write_doc(cfg,rep)
    return rep

def write_doc(cfg,rep):
    lines=['# Walk-forward linear alpha scan (2026-07-12)','','Monthly expanding-window ridge alpha surface; each month uses prior rows only. Stats format: `absolute/CAGR/strictMDD/ratio/trades`.','',f"features={len(rep['features'])}, all_count={rep['all_count']}",'','| rank | name | train | 2024 | 2025 | 2026YTD |','|---:|---|---:|---:|---:|---:|']
    for i,r in enumerate(rep['top'][:60],1):
        st=r['stats']; lines.append(f"| {i} | `{r['name']}` | {fmt(st['train'])} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    Path(cfg.docs_output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.docs_output).write_text('\n'.join(lines)+'\n')

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for k,v in WalkForwardLinearAlphaConfig().__dict__.items():
        arg='--'+k.replace('_','-')
        if isinstance(v,int): p.add_argument(arg,type=int,default=v)
        else: p.add_argument(arg,default=v)
    return p.parse_args()

def main():
    rep=run(WalkForwardLinearAlphaConfig(**vars(parse_args())))
    print(json.dumps({'output':rep['config']['output'],'docs_output':rep['config']['docs_output'],'all_count':rep['all_count'],'top':rep['top'][:12]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
