"""Bidirectional BTC alpha from OI liquidation/build and taker-flow states."""
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
def zs(s,n=576):
 mu=s.rolling(n,min_periods=n).mean();sd=s.rolling(n,min_periods=n).std().replace(0,np.nan);return (s-mu)/sd
def features(m,base):
 f=base.copy();c=np.log(m.close.astype(float));oi=np.log(pd.to_numeric(m.open_interest,errors='coerce').replace(0,np.nan).ffill());qv=m.quote_asset_volume.astype(float);buy=m.taker_buy_quote.astype(float);imb=(2*buy/qv.replace(0,np.nan)-1).clip(-1,1)
 for n in (12,48,144,288):
  pr=c-c.shift(n);orr=oi-oi.shift(n);f[f'ol_px_{n}']=pr;f[f'ol_oi_{n}']=orr;f[f'ol_px_z_{n}']=zs(pr);f[f'ol_oi_z_{n}']=zs(orr);f[f'ol_div_{n}']=f[f'ol_oi_z_{n}']-f[f'ol_px_z_{n}']
 f['ol_flow_12']=imb.rolling(12,min_periods=12).mean();f['ol_flow_48']=imb.rolling(48,min_periods=48).mean();f['ol_flow_accel']=f['ol_flow_12']-f['ol_flow_48'];return f.replace([np.inf,-np.inf],np.nan)
def specs(f,tr):
 out=[]
 def add(n,l,s):out.append((n,[(c,o,q(f,tr,c,z)) for c,o,z in l],[(c,o,q(f,tr,c,z)) for c,o,z in s]))
 for n in (48,144,288):
  for oq,pq,fq in itertools.product((.05,.1,.2),(.05,.1,.2),(.7,.8,.9)):
   add(f'liquidation_exhaustion_{n}',[(f'ol_oi_z_{n}','le',oq),(f'ol_px_z_{n}','le',pq),('ol_flow_accel','ge',fq)],[(f'ol_oi_z_{n}','ge',1-oq),(f'ol_px_z_{n}','ge',1-pq),('ol_flow_accel','le',1-fq)])
  for dq,fq in itertools.product((.05,.1,.2),(.7,.8,.9)):
   add(f'oi_price_divergence_{n}',[(f'ol_div_{n}','ge',1-dq),('ol_flow_accel','ge',fq)],[(f'ol_div_{n}','le',dq),('ol_flow_accel','le',1-fq)])
   add(f'crowding_reversal_{n}',[(f'ol_div_{n}','le',dq),(f'ol_px_z_{n}','le',dq)],[(f'ol_div_{n}','ge',1-dq),(f'ol_px_z_{n}','ge',1-dq)])
 return out
def run(cfg):
 m=_load_market(cfg)
 if 'open_interest' not in m or pd.to_numeric(m.open_interest,errors='coerce').notna().sum()<1000:raise ValueError('input must contain open_interest')
 base=build_market_feature_frame(m,window_size=cfg.window_size);f=features(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);rows=[]
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
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'OI liquidation/build bidirectional BTC; train thresholds; test-only rank; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
