"""Search low-correlation BTC macro-liquidity alphas with strict exits.

Thresholds are train<2024 quantiles. Candidates are filtered by maximum absolute
phi correlation to existing alpha component masks on test2024, ranked on
test2024 only, then evaluated on sealed 2025/2026 windows.
"""
from __future__ import annotations
import argparse, json, itertools
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_correlation_report import _add_oi_derived_features, _build_component_frame
from training.long_component_tp_union_scan import _strict_long_overlay_sim
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats

@dataclass(frozen=True)
class Config(LongComboScanConfig):
    exclude_from: str="2026-06-02"; fee_rate: float=.0005; slippage_rate: float=.0001; leverage: float=1.0
    max_abs_phi: float=.15; min_test_trades: int=18; top_n: int=200

WINDOWS={"train":("2020-01-01","2024-01-01"),"test2024":("2024-01-01","2025-01-01"),"eval2025":("2025-01-01","2026-01-01"),"ytd2026":("2026-01-01","2026-06-02")}

def q(f,m,c,qq):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,qq))
def mask(f,conds):
 a=np.ones(len(f),bool)
 for c,op,t in conds:
  x=f[c].to_numpy(float);a&=np.isfinite(x)&((x>=t) if op=='ge' else (x<=t))
 return a
def phi(a,b):
 if a.std()==0 or b.std()==0:return 0.
 return float(np.corrcoef(a.astype(float),b.astype(float))[0,1])

def score(m,dates,active,cfg,hold,stride,tp,sl,w):
 start,end=WINDOWS[w];wm=_split_mask(dates,start,end); pos=np.arange(143,len(m)-hold-2,stride,dtype=np.int64);p=pos[active[pos]&wm[pos]]
 sim,rets=_strict_long_overlay_sim(p,market=m,hold_bars=hold,entry_delay_bars=1,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate,take_profit=tp,stop_loss=sl,annualization_start=start,annualization_end=end)
 ts=_trade_stats(rets); sh=float(ts['effect_size_d'])*np.sqrt(ts['n_trades'])
 return {'return_pct':sim['total_return_pct'],'cagr_pct':sim['cagr_pct'],'strict_mdd_pct':sim['strict_mdd_pct'],'ratio':sim['cagr_to_strict_mdd'],'trades':sim['trade_entries'],'win_rate':sim['win_rate'],'sharpe_like':sh,'p':ts['p_value_mean_ret_approx'],'exits':sim.get('exit_reasons',{})}

def specs(f,train):
 out=[]
 def add(name,raw): out.append((name,[(c,op,q(f,train,c,qq)) for c,op,qq in raw]))
 # Macro liquidity relief and dislocation; all combinations are predeclared.
 for mq,tq in itertools.product((.05,.1,.2),(.6,.7,.8,.9)):
  add('usdkrw_relief_momentum',[('usdkrw_momentum','le',mq),('htf_1d_return_1','ge',tq)])
  add('dxy_relief_momentum',[('dxy_momentum','le',mq),('htf_1d_return_1','ge',tq)])
  add('usdkrw_relief_weekly',[('usdkrw_momentum','le',mq),('weekly_return_1w','ge',tq)])
 for dq,kq,tq in itertools.product((.1,.2),(.1,.2,.8,.9),(.6,.7,.8)):
  kop='le' if kq<.5 else 'ge'
  add('global_local_liquidity',[('dxy_momentum','le',dq),('kimchi_premium_change',kop,kq),('htf_1d_return_1','ge',tq)])
 for uq,pq in itertools.product((.05,.1,.2),(.1,.2,.3)):
  add('fx_premium_dislocation',[('usdkrw_zscore','le',uq),('kimchi_premium_zscore','le',pq),('htf_4h_return_4','ge',.6)])
 return out

def run(cfg):
 m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=pd.concat([base,build_interest_features(m,base)],axis=1);f=_add_oi_derived_features(m,f);dates=pd.to_datetime(m.date);train=_split_mask(dates,*WINDOWS['train']);test=_split_mask(dates,*WINDOWS['test2024'])
 comps,_=_build_component_frame(f); comp_names=list(comps.columns); rows=[]
 for name,conds in specs(f,train):
  a=mask(f,conds); corrs={c:phi(a[test],comps[c].to_numpy(bool)[test]) for c in comp_names}; mc=max((abs(v) for v in corrs.values()),default=0); nearest=max(corrs,key=lambda c:abs(corrs[c])) if corrs else ''
  if mc>cfg.max_abs_phi or int((a&train).sum())<150:continue
  for hold,stride,(tp,sl) in itertools.product((72,144,216,288,432),(12,24),((None,None),(.025,.015),(.04,.025),(.06,.04))):
   s=score(m,dates,a,cfg,hold,stride,tp,sl,'test2024')
   if s['trades']>=cfg.min_test_trades:rows.append({'name':name,'conditions':[{'feature':c,'op':op,'threshold':t} for c,op,t in conds],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'max_abs_phi_test':mc,'nearest_alpha_component':nearest,'nearest_phi':corrs.get(nearest,0),'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio']-.5*r['max_abs_phi_test'],r['test2024']['return_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  a=mask(f,[(x['feature'],x['op'],x['threshold']) for x in r['conditions']])
  for w in ('train','eval2025','ytd2026'):r[w]=score(m,dates,a,cfg,r['hold_bars'],r['stride_bars'],r['tp'],r['sl'],w)
  enough = r['test2024']['trades'] >= cfg.min_test_trades and r['eval2025']['trades'] >= 8
  r['passes_alpha_pool']=enough and r['test2024']['ratio']>=2.5 and r['eval2025']['ratio']>=2.5
  r['passes_live_grade']=enough and r['test2024']['ratio']>=3 and r['eval2025']['ratio']>=3
  r['passes_2026_target']=r['ytd2026']['trades']>=6 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'train-only thresholds; max abs test2024 phi<=0.15 versus existing alpha components; test-only ranking; sealed eval2025/ytd2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]}
 Path(cfg.output).parent.mkdir(parents=True,exist_ok=True);Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default=Config.exclude_from);a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'alpha_pool':len(o['alpha_pool_qualifiers']),'live_grade':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
