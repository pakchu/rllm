"""Bidirectional BTC alpha from causal extrema-arrival and recovery-speed states."""
from __future__ import annotations
import argparse,itertools,json
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config,W,sim,mk
def q(f,m,c,z):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,z))
def age_since(flag):
 idx=np.arange(len(flag));last=np.maximum.accumulate(np.where(flag,idx,-10**9));a=idx-last;return pd.Series(np.where(last<0,np.nan,a),index=flag.index)
def features(m,base):
 f=base.copy();c=m.close.astype(float);lc=np.log(c)
 for n in (144,576,2016):
  rh=c.rolling(n,min_periods=n).max();rl=c.rolling(n,min_periods=n).min();nh=(c>=rh)&rh.notna();nl=(c<=rl)&rl.notna();ah=age_since(nh);al=age_since(nl)
  f[f'ea_pos_{n}']=(c-rl)/(rh-rl).replace(0,np.nan);f[f'ea_high_age_{n}']=ah;f[f'ea_low_age_{n}']=al;f[f'ea_high_hits_{n}']=nh.rolling(n,min_periods=n).sum();f[f'ea_low_hits_{n}']=nl.rolling(n,min_periods=n).sum()
  low_ref=rl.shift(al.fillna(0).clip(upper=n-1).astype(int)) if False else rl
  # Speed proxies use current range displacement divided by time since the last extrema event.
  f[f'ea_recovery_speed_{n}']=(f[f'ea_pos_{n}'])/(al+1);f[f'ea_fade_speed_{n}']=(1-f[f'ea_pos_{n}'])/(ah+1);f[f'ea_hit_imbalance_{n}']=f[f'ea_high_hits_{n}']-f[f'ea_low_hits_{n}']
 return f.replace([np.inf,-np.inf],np.nan)
def specs(f,tr):
 out=[]
 def add(n,l,s):out.append((n,[(c,o,q(f,tr,c,z)) for c,o,z in l],[(c,o,q(f,tr,c,z)) for c,o,z in s]))
 for n in (144,576,2016):
  for aq,pq in itertools.product((.05,.1,.2),(.1,.2,.3)):
   add(f'extrema_reversal_{n}',[(f'ea_low_age_{n}','le',aq),(f'ea_pos_{n}','le',pq),(f'ea_recovery_speed_{n}','ge',.7)],[(f'ea_high_age_{n}','le',aq),(f'ea_pos_{n}','ge',1-pq),(f'ea_fade_speed_{n}','ge',.7)])
   add(f'extrema_continuation_{n}',[(f'ea_high_age_{n}','le',aq),(f'ea_pos_{n}','ge',1-pq),(f'ea_high_hits_{n}','ge',.7)],[(f'ea_low_age_{n}','le',aq),(f'ea_pos_{n}','le',pq),(f'ea_low_hits_{n}','ge',.7)])
  for iq,pq in itertools.product((.1,.2,.8,.9),(.1,.2)):
   if iq<.5:add(f'hit_imbalance_revert_{n}',[(f'ea_hit_imbalance_{n}','le',iq),(f'ea_pos_{n}','le',pq)],[(f'ea_hit_imbalance_{n}','ge',1-iq),(f'ea_pos_{n}','ge',1-pq)])
   else:add(f'hit_imbalance_follow_{n}',[(f'ea_hit_imbalance_{n}','ge',iq),(f'ea_pos_{n}','ge',1-pq)],[(f'ea_hit_imbalance_{n}','le',1-iq),(f'ea_pos_{n}','le',pq)])
 return out
def run(cfg):
 m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=features(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);rows=[]
 for name,lc,sc in specs(f,tr):
  la,sa=mk(f,lc),mk(f,sc)
  if int((la|sa)[tr].sum())<100:continue
  for hold,stride,(tp,sl) in itertools.product((24,48,72,96,144,216),(6,12,24),((.01,.008),(.015,.01),(.025,.015),(.04,.025))):
   s=sim(m,dates,la,sa,cfg,hold,stride,tp,sl,'test2024')
   if s['trades']>=cfg.min_test_trades and s['longs']>=cfg.min_each_side and s['shorts']>=cfg.min_each_side:rows.append({'name':name,'long_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in lc],'short_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in sc],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  la=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['long_conditions']]);sa=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['short_conditions']])
  for w in ('train','eval2025','ytd2026'):r[w]=sim(m,dates,la,sa,cfg,r['hold_bars'],r['stride_bars'],r['tp'],r['sl'],w)
  e=r['eval2025'];en=e['trades']>=16 and e['longs']>=4 and e['shorts']>=4;r['passes_alpha_pool']=en and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5;r['passes_live_grade']=en and r['test2024']['ratio']>=3 and e['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=8 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'causal extrema arrival bidirectional BTC; train thresholds; test-only rank; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
