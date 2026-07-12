"""Search DXY/FX/session short alphas weakly correlated to existing components."""
from __future__ import annotations
import argparse,itertools,json
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import LongComboScanConfig,_load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_component_tp_union_scan import COMPONENTS,_component_mask
from training.nonrex_short_bear_tp_refine import _sim_short_tp

@dataclass(frozen=True)
class Config(LongComboScanConfig):
 exclude_from:str='2026-06-02';fee_rate:float=.0005;slippage_rate:float=.0001;leverage:float=.5;max_abs_phi:float=.20;min_test_trades:int=20;top_n:int=250
W={'train':('2020-01-01','2024-01-01'),'test2024':('2024-01-01','2025-01-01'),'eval2025':('2025-01-01','2026-01-01'),'ytd2026':('2026-01-01','2026-06-02')}
EXTRA={'short_premium_panic':[('htf_3d_range_pos','le',-0.5114186851),('premium_index_zscore','le',-1.47209312)],'short_kimchi_unwind':[('htf_3d_return_1','le',-0.0303196833),('kimchi_premium_change','le',-0.0046123752)],'short_fx_stress':[('htf_3d_return_1','le',-0.0325294973),('usdkrw_zscore','ge',1.3870063775)]}
def q(f,m,c,x):
 v=f.loc[m,c].to_numpy(float);v=v[np.isfinite(v)];return float(np.quantile(v,x))
def mk(f,conds):
 a=np.ones(len(f),bool)
 for c,op,t in conds:
  x=f[c].to_numpy(float);a&=np.isfinite(x)&((x>=t) if op=='ge' else (x<=t))
 return a
def corr(a,b):return float(np.corrcoef(a.astype(float),b.astype(float))[0,1]) if a.std() and b.std() else 0.
def baselines(f):
 b={n:_component_mask(f,n) for n in COMPONENTS}
 for n,c in EXTRA.items():b[n]=mk(f,c)
 for n,members in {'long_core_union':['range_bb90','funding10_trend70','premium20_mom90'],'long_alt_union':['funding10_trend70','compress05_trend80','premium20_mom90'],'short_premium_kimchi_union':['short_premium_panic','short_kimchi_unwind']}.items():
  a=np.zeros(len(f),bool)
  for x in members:a|=b[x]
  b[n]=a
 return b
def candidates(f,tr):
 out=[]
 def add(n,r):out.append((n,[(c,o,q(f,tr,c,z)) for c,o,z in r]))
 for fx,px in itertools.product((.8,.9,.95),(.05,.1,.2,.3)):
  add('dxy_riskoff_weakness',[('dxy_momentum','ge',fx),('htf_1d_return_1','le',px)])
  add('usdkrw_riskoff_weakness',[('usdkrw_momentum','ge',fx),('htf_1d_return_1','le',px)])
  add('dxy_3d_weakness',[('dxy_zscore','ge',fx),('htf_3d_return_1','le',px)])
 for dx,ux,px in itertools.product((.8,.9),(.8,.9),(.1,.2,.3)):
  add('global_fx_stress',[('dxy_momentum','ge',dx),('usdkrw_momentum','ge',ux),('htf_1d_return_1','le',px)])
 for dx,kx,px in itertools.product((.8,.9),(.1,.2,.8,.9),(.1,.2)):
  op='le' if kx<.5 else 'ge';add('dxy_kimchi_dislocation',[('dxy_momentum','ge',dx),('kimchi_premium_change',op,kx),('htf_1d_return_1','le',px)])
 # Deterministic session overlay creates a structurally different event clock.
 dt=pd.to_datetime(f.index if isinstance(f.index,pd.DatetimeIndex) else pd.Series(range(len(f))))
 return out
def run(cfg):
 m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=pd.concat([base,build_interest_features(m,base)],axis=1);dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);te=_split_mask(dates,*W['test2024']);bs=baselines(f);rows=[]
 for name,conds in candidates(f,tr):
  a=mk(f,conds);cs={n:corr(a[te],x[te]) for n,x in bs.items()};nearest=max(cs,key=lambda n:abs(cs[n]));mc=abs(cs[nearest])
  if mc>cfg.max_abs_phi or int((a&tr).sum())<150:continue
  for hold,stride,(tp,sl) in itertools.product((48,72,96,144,216,288),(12,24),((.01,.008),(.015,.01),(.025,.015),(.04,.025),(.06,.04),(None,None))):
   wm=te;pos=np.arange(143,len(m)-hold-2,stride,dtype=np.int64);sig=pos[a[pos]&wm[pos]];s=_sim_short_tp(market=m,signal_positions=sig,start=W['test2024'][0],end=W['test2024'][1],hold_bars=hold,take_profit=tp,stop_loss=sl,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate)
   if s['trades']>=cfg.min_test_trades:rows.append({'name':name,'conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in conds],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'max_abs_phi_test':mc,'nearest':nearest,'nearest_phi':cs[nearest],'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio']-.5*r['max_abs_phi_test'],r['test2024']['ret_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  a=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['conditions']])
  for wn in ('train','eval2025','ytd2026'):
   st,en=W[wn];wm=_split_mask(dates,st,en);pos=np.arange(143,len(m)-r['hold_bars']-2,r['stride_bars'],dtype=np.int64);sig=pos[a[pos]&wm[pos]];r[wn]=_sim_short_tp(market=m,signal_positions=sig,start=st,end=en,hold_bars=r['hold_bars'],take_profit=r['tp'],stop_loss=r['sl'],leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate)
  enough=r['eval2025']['trades']>=8;r['passes_alpha_pool']=enough and r['test2024']['ratio']>=2.5 and r['eval2025']['ratio']>=2.5;r['passes_live_grade']=enough and r['test2024']['ratio']>=3 and r['eval2025']['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=6 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'train thresholds; test-only rank; max phi<=0.20; sealed eval/2026; 6bp/side; strict short MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default=Config.exclude_from);a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
