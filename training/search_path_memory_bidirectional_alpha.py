"""Bidirectional BTC alpha scan using path memory, entropy and variance ratio."""
from __future__ import annotations
import argparse,itertools,json
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config as BaseConfig,W,sim,mk

class Config(BaseConfig):
 pass
def q(f,m,c,z):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,z))
def entropy_sign(x):
 p=x.rolling(144,min_periods=144).mean().clip(1e-6,1-1e-6);return -(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2)
def features(m,base):
 c=m.close.astype(float);lr=np.log(c/c.shift(1)).replace([np.inf,-np.inf],np.nan);sgn=(lr>0).astype(float);f=base.copy()
 for n in (24,72,144,288):
  f[f'pm_ret_{n}']=lr.rolling(n,min_periods=n).sum();f[f'pm_eff_{n}']=f[f'pm_ret_{n}']/lr.abs().rolling(n,min_periods=n).sum().replace(0,np.nan)
 f['pm_sign_entropy']=entropy_sign(sgn)
 f['pm_sign_autocorr']=lr.rolling(144,min_periods=144).corr(lr.shift(1))
 v1=lr.rolling(144,min_periods=144).var();r12=np.log(c/c.shift(12));v12=r12.rolling(144,min_periods=144).var()/12;f['pm_variance_ratio_12']=v12/v1.replace(0,np.nan)
 up=lr.clip(lower=0).rolling(144,min_periods=144).std();dn=(-lr.clip(upper=0)).rolling(144,min_periods=144).std();f['pm_semivol_skew']=(up-dn)/(up+dn).replace(0,np.nan)
 return f.replace([np.inf,-np.inf],np.nan)
def specs(f,tr):
 out=[]
 def add(n,l,s):out.append((n,[(c,o,q(f,tr,c,z)) for c,o,z in l],[(c,o,q(f,tr,c,z)) for c,o,z in s]))
 for state,move in itertools.product((.1,.2,.8,.9),(.05,.1,.2)):
  if state>.5:
   add('autocorr_momentum',[('pm_sign_autocorr','ge',state),('pm_ret_72','ge',1-move)],[('pm_sign_autocorr','ge',state),('pm_ret_72','le',move)])
   add('variance_ratio_momentum',[('pm_variance_ratio_12','ge',state),('pm_ret_72','ge',1-move)],[('pm_variance_ratio_12','ge',state),('pm_ret_72','le',move)])
  else:
   add('autocorr_reversion',[('pm_sign_autocorr','le',state),('pm_ret_72','le',move)],[('pm_sign_autocorr','le',state),('pm_ret_72','ge',1-move)])
   add('variance_ratio_reversion',[('pm_variance_ratio_12','le',state),('pm_ret_72','le',move)],[('pm_variance_ratio_12','le',state),('pm_ret_72','ge',1-move)])
 for eq,move in itertools.product((.1,.2,.8,.9),(.1,.2)):
  if eq<.5:add('low_entropy_continuation',[('pm_sign_entropy','le',eq),('pm_eff_72','ge',1-move)],[('pm_sign_entropy','le',eq),('pm_eff_72','le',move)])
  else:add('high_entropy_reversion',[('pm_sign_entropy','ge',eq),('pm_ret_72','le',move)],[('pm_sign_entropy','ge',eq),('pm_ret_72','ge',1-move)])
 for sk,move in itertools.product((.1,.2,.8,.9),(.1,.2)):
  add('semivol_reversion',[('pm_semivol_skew','le',sk),('pm_ret_72','le',move)],[('pm_semivol_skew','ge',1-sk),('pm_ret_72','ge',1-move)])
 return out
def run(cfg):
 m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=features(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);rows=[]
 for name,lc,sc in specs(f,tr):
  la,sa=mk(f,lc),mk(f,sc)
  for hold,stride,(tp,sl) in itertools.product((24,48,72,96,144,216),(6,12,24),((.01,.008),(.015,.01),(.025,.015),(.04,.025))):
   z=sim(m,dates,la,sa,cfg,hold,stride,tp,sl,'test2024')
   if z['trades']>=cfg.min_test_trades and z['longs']>=cfg.min_each_side and z['shorts']>=cfg.min_each_side:rows.append({'name':name,'long_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in lc],'short_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in sc],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'test2024':z})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  la=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['long_conditions']]);sa=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['short_conditions']])
  for w in ('train','eval2025','ytd2026'):r[w]=sim(m,dates,la,sa,cfg,r['hold_bars'],r['stride_bars'],r['tp'],r['sl'],w)
  e=r['eval2025'];en=e['trades']>=16 and e['longs']>=4 and e['shorts']>=4;r['passes_alpha_pool']=en and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5;r['passes_live_grade']=en and r['test2024']['ratio']>=3 and e['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=8 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'path-memory bidirectional BTC; train thresholds; test-only rank; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
