"""Leak-safe bidirectional BTC alpha from price/liquidity recovery asymmetry."""
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

def features(m,base):
 """All values at t use completed bars <=t; execution remains next-open in sim()."""
 f=base.copy();o=m.open.astype(float);h=m.high.astype(float);l=m.low.astype(float);c=m.close.astype(float)
 qv=m.quote_asset_volume.astype(float);buy=m.taker_buy_quote.astype(float);r=np.log(c).diff()
 tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
 signed=(2*buy-qv);imb=signed/qv.replace(0,np.nan)
 body=(c-o);upper=h-pd.concat([o,c],axis=1).max(axis=1);lower=pd.concat([o,c],axis=1).min(axis=1)-l
 for n in (12,24,48,72,144,288):
  path=r.abs().rolling(n,min_periods=n).sum();net=np.log(c/c.shift(n));rng=tr.rolling(n,min_periods=n).sum()
  vol=qv.rolling(n,min_periods=n).sum();flow=signed.rolling(n,min_periods=n).sum()
  f[f'lr_eff_{n}']=net.abs()/path.replace(0,np.nan)
  f[f'lr_signed_eff_{n}']=net/path.replace(0,np.nan)
  f[f'lr_displacement_{n}']=net/rng.replace(0,np.nan)
  f[f'lr_flow_{n}']=flow/vol.replace(0,np.nan)
  f[f'lr_impact_{n}']=net/(flow.abs()/vol.replace(0,np.nan)+1e-4)
  f[f'lr_range_share_{n}']=tr.rolling(n,min_periods=n).sum()/tr.rolling(288,min_periods=288).sum().replace(0,np.nan)
 f['lr_flow_recovery']=imb.rolling(12,min_periods=12).mean()-imb.rolling(72,min_periods=72).mean()
 f['lr_lower_rejection']=(lower-body.clip(upper=0).abs()).rolling(12,min_periods=12).sum()/tr.rolling(12,min_periods=12).sum().replace(0,np.nan)
 f['lr_upper_rejection']=(upper-body.clip(lower=0).abs()).rolling(12,min_periods=12).sum()/tr.rolling(12,min_periods=12).sum().replace(0,np.nan)
 return f.replace([np.inf,-np.inf],np.nan)

def specs(f,tr):
 out=[]
 def add(n,l,s):out.append((n,[(c,o,q(f,tr,c,z)) for c,o,z in l],[(c,o,q(f,tr,c,z)) for c,o,z in s]))
 for n in (24,48,72,144,288):
  for extreme,recover in itertools.product((.05,.1,.2),(.7,.8,.9)):
   add(f'liquidity_recovery_{n}',
       [(f'lr_signed_eff_{n}','le',extreme),('lr_flow_recovery','ge',recover),('lr_lower_rejection','ge',recover)],
       [(f'lr_signed_eff_{n}','ge',1-extreme),('lr_flow_recovery','le',1-recover),('lr_upper_rejection','ge',recover)])
   add(f'impact_absorption_{n}',
       [(f'lr_impact_{n}','le',extreme),(f'lr_flow_{n}','le',extreme),('lr_flow_recovery','ge',recover)],
       [(f'lr_impact_{n}','ge',1-extreme),(f'lr_flow_{n}','ge',1-extreme),('lr_flow_recovery','le',1-recover)])
  for efficient,flowq in itertools.product((.7,.8,.9),(.7,.8,.9)):
   add(f'efficient_recovery_continuation_{n}',
       [(f'lr_signed_eff_{n}','ge',efficient),(f'lr_flow_{n}','ge',flowq),('lr_flow_recovery','ge',flowq)],
       [(f'lr_signed_eff_{n}','le',1-efficient),(f'lr_flow_{n}','le',1-flowq),('lr_flow_recovery','le',1-flowq)])
 return out

def run(cfg):
 m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size)
 f=features(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);rows=[]
 for name,lc,sc in specs(f,tr):
  la,sa=mk(f,lc),mk(f,sc)
  if int((la|sa)[tr].sum())<100:continue
  for hold,stride,(tp,sl) in itertools.product((12,24,48,72,96,144,216),(6,12,24),((.01,.008),(.015,.01),(.025,.015),(.04,.025))):
   z=sim(m,dates,la,sa,cfg,hold,stride,tp,sl,'test2024')
   if z['trades']>=cfg.min_test_trades and z['longs']>=cfg.min_each_side and z['shorts']>=cfg.min_each_side:
    rows.append({'name':name,'long_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in lc],'short_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in sc],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'test2024':z})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  la=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['long_conditions']]);sa=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['short_conditions']])
  for w in ('train','eval2025','ytd2026'):r[w]=sim(m,dates,la,sa,cfg,r['hold_bars'],r['stride_bars'],r['tp'],r['sl'],w)
  e=r['eval2025'];en=e['trades']>=16 and e['longs']>=4 and e['shorts']>=4
  r['passes_alpha_pool']=en and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5
  r['passes_live_grade']=en and r['test2024']['ratio']>=3 and e['ratio']>=3
  r['passes_2026_target']=r['ytd2026']['trades']>=8 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'causal liquidity-recovery bidirectional BTC; train-fit thresholds; test-only selection; sealed eval/2026; 6bp/side; full-window CAGR; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]}
 Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)))
 print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
