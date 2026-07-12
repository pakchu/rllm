"""Low-correlation BTC alpha search using causal alt-market breadth features."""
from __future__ import annotations
import argparse,itertools,json
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import LongComboScanConfig,_load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_component_tp_union_scan import COMPONENTS,_component_mask,_strict_long_overlay_sim
from training.nonrex_short_bear_tp_refine import _sim_short_tp

@dataclass(frozen=True)
class Config(LongComboScanConfig):
 exclude_from:str='2026-06-02';fee_rate:float=.0005;slippage_rate:float=.0001;leverage:float=.5;alt_dir:str='data/binance_um_pool_5m_2023_2026';max_abs_phi:float=.20;min_test_trades:int=20;top_n:int=250
W={'train':('2023-01-01','2024-01-01'),'test2024':('2024-01-01','2025-01-01'),'eval2025':('2025-01-01','2026-01-01'),'ytd2026':('2026-01-01','2026-06-02')}
EXTRA={'short_premium_panic':[('htf_3d_range_pos','le',-0.5114186851),('premium_index_zscore','le',-1.47209312)],'short_kimchi_unwind':[('htf_3d_return_1','le',-0.0303196833),('kimchi_premium_change','le',-0.0046123752)],'short_fx_stress':[('htf_3d_return_1','le',-0.0325294973),('usdkrw_zscore','ge',1.3870063775)]}
def mk(f,conds):
 a=np.ones(len(f),bool)
 for c,o,t in conds:
  x=f[c].to_numpy(float);a&=np.isfinite(x)&((x>=t) if o=='ge' else (x<=t))
 return a
def q(f,m,c,z):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,z))
def corr(a,b):return float(np.corrcoef(a.astype(float),b.astype(float))[0,1]) if a.std() and b.std() else 0.
def attach(m,alt_dir):
 out=m.copy();root=Path(alt_dir);syms=['ETH','SOL','BNB','XRP','ADA','DOGE']
 for s in syms:
  paths=sorted(root.glob(f'{s}USDT_5m_*.csv.gz'))
  if not paths:continue
  x=pd.read_csv(paths[0],usecols=['date','close'],parse_dates=['date']);x['date']=pd.to_datetime(x.date,utc=True).dt.tz_convert(None);x=x.rename(columns={'close':f'{s.lower()}_close'})
  out=out.merge(x,on='date',how='left')
 return out
def breadth(m,f):
 btc=np.log(m.close.astype(float));rets={}
 for c in [x for x in m.columns if x.endswith('_close')]:
  lc=np.log(pd.to_numeric(m[c],errors='coerce'))
  for n in (12,48,144,288):rets.setdefault(n,[]).append(lc-lc.shift(n))
 for n,arr in rets.items():
  a=pd.concat(arr,axis=1);b=a.median(axis=1);br=(a>0).mean(axis=1);btc_r=btc-btc.shift(n)
  f[f'alt_breadth_ret_{n}']=b;f[f'alt_positive_frac_{n}']=br;f[f'btc_alt_residual_{n}']=btc_r-b;f[f'alt_dispersion_{n}']=a.std(axis=1)
 return f
def baselines(f):
 b={n:_component_mask(f,n) for n in COMPONENTS}
 for n,c in EXTRA.items():b[n]=mk(f,c)
 for n,ms in {'long_core_union':['range_bb90','funding10_trend70','premium20_mom90'],'long_alt_union':['funding10_trend70','compress05_trend80','premium20_mom90'],'short_union':['short_premium_panic','short_kimchi_unwind']}.items():
  a=np.zeros(len(f),bool)
  for x in ms:a|=b[x]
  b[n]=a
 return b
def candidates(f,tr):
 out=[]
 def add(n,side,raw):out.append((n,side,[(c,o,q(f,tr,c,z)) for c,o,z in raw]))
 for n in (12,48,144):
  for aq,bq in itertools.product((.8,.9),(.1,.2,.3)):
   add(f'alt_lead_long_{n}','long',[(f'alt_breadth_ret_{n}','ge',aq),(f'btc_alt_residual_{n}','le',bq)])
   add(f'alt_lead_short_{n}','short',[(f'alt_breadth_ret_{n}','le',1-aq),(f'btc_alt_residual_{n}','ge',1-bq)])
   add(f'btc_underperf_revert_{n}','long',[(f'btc_alt_residual_{n}','le',1-aq),(f'alt_positive_frac_{n}','ge',aq)])
   add(f'btc_overperf_revert_{n}','short',[(f'btc_alt_residual_{n}','ge',aq),(f'alt_positive_frac_{n}','le',1-aq)])
  for dq,rq in itertools.product((.8,.9),(.1,.2,.8,.9)):
   op='le' if rq<.5 else 'ge';side='long' if rq<.5 else 'short';add(f'dispersion_reversion_{n}',side,[(f'alt_dispersion_{n}','ge',dq),(f'btc_alt_residual_{n}',op,rq)])
 return out
def run(cfg):
 m=attach(_load_market(cfg),cfg.alt_dir);base=build_market_feature_frame(m,window_size=cfg.window_size);f=breadth(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);te=_split_mask(dates,*W['test2024']);bs=baselines(f);rows=[]
 for name,side,conds in candidates(f,tr):
  a=mk(f,conds);cs={n:corr(a[te],x[te]) for n,x in bs.items()};near=max(cs,key=lambda n:abs(cs[n]));mc=abs(cs[near])
  if mc>cfg.max_abs_phi or int((a&tr).sum())<100:continue
  for hold,stride,(tp,sl) in itertools.product((48,72,144,216,288),(12,24),((None,None),(.015,.01),(.025,.015),(.04,.025))):
   pos=np.arange(143,len(m)-hold-2,stride,dtype=np.int64);sig=pos[a[pos]&te[pos]]
   if side=='short':s=_sim_short_tp(market=m,signal_positions=sig,start=W['test2024'][0],end=W['test2024'][1],hold_bars=hold,take_profit=tp,stop_loss=sl,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate)
   else:
    sim,rr=_strict_long_overlay_sim(sig,market=m,hold_bars=hold,entry_delay_bars=1,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate,take_profit=tp,stop_loss=sl,annualization_start=W['test2024'][0],annualization_end=W['test2024'][1]);s={'ret_pct':sim['total_return_pct'],'cagr_pct':sim['cagr_pct'],'mdd_pct':sim['strict_mdd_pct'],'ratio':sim['cagr_to_strict_mdd'],'trades':sim['trade_entries'],'win_rate':sim['win_rate']}
   if s['trades']>=cfg.min_test_trades:rows.append({'name':name,'side':side,'conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in conds],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'max_abs_phi_test':mc,'nearest':near,'nearest_phi':cs[near],'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio']-.5*r['max_abs_phi_test'],r['test2024']['ret_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  a=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['conditions']])
  for wn in ('train','eval2025','ytd2026'):
   st,en=W[wn];wm=_split_mask(dates,st,en);pos=np.arange(143,len(m)-r['hold_bars']-2,r['stride_bars'],dtype=np.int64);sig=pos[a[pos]&wm[pos]]
   if r['side']=='short':s=_sim_short_tp(market=m,signal_positions=sig,start=st,end=en,hold_bars=r['hold_bars'],take_profit=r['tp'],stop_loss=r['sl'],leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate)
   else:
    sim,rr=_strict_long_overlay_sim(sig,market=m,hold_bars=r['hold_bars'],entry_delay_bars=1,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate,take_profit=r['tp'],stop_loss=r['sl'],annualization_start=st,annualization_end=en);s={'ret_pct':sim['total_return_pct'],'cagr_pct':sim['cagr_pct'],'mdd_pct':sim['strict_mdd_pct'],'ratio':sim['cagr_to_strict_mdd'],'trades':sim['trade_entries'],'win_rate':sim['win_rate']}
   r[wn]=s
  enough=r['eval2025']['trades']>=8;r['passes_alpha_pool']=enough and r['test2024']['ratio']>=2.5 and r['eval2025']['ratio']>=2.5;r['passes_live_grade']=enough and r['test2024']['ratio']>=3 and r['eval2025']['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=6 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'BTC target; alt breadth predictors; train2023 thresholds; test-only rank; max phi<=.20; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--alt-dir',default=Config.alt_dir);p.add_argument('--exclude-from',default=Config.exclude_from);a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
