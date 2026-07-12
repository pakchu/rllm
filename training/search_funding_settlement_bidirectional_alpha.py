"""Causal post-settlement BTC alpha from funding/carry dislocations."""
from __future__ import annotations
import argparse,itertools,json,sys
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
if __package__ is None or __package__=="":sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market,_split_mask
from training.search_bidirectional_state_alpha import Config,W,mk,sim

def q(f,m,c,z):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,z))
def features(m,cfg):
 f=build_market_feature_frame(m,window_size=144);d=pd.to_datetime(m.date);close=m.close.astype(float);qv=m.quote_asset_volume.astype(float);buy=m.taker_buy_quote.astype(float)
 funding=pd.read_csv(cfg.funding_csv,parse_dates=['date']).sort_values('date'); fd=pd.to_datetime(funding.date).dt.floor('5min'); rates=pd.Series(funding.funding_rate.astype(float).to_numpy(),index=fd).groupby(level=0).last()
 event_rate=d.map(rates);f['fs_event']=event_rate.notna().astype(float).to_numpy();f['fs_rate']=event_rate.ffill().fillna(0).to_numpy();f['fs_prev_rate']=pd.Series(f.fs_rate).shift(96).to_numpy();f['fs_delta']=f.fs_rate-f.fs_prev_rate
 f['fs_price_ret_8h']=np.log(close/close.shift(96));f['fs_price_ret_24h']=np.log(close/close.shift(288));f['fs_flow_8h']=(2*buy/qv.replace(0,np.nan)-1).rolling(96,min_periods=96).mean();f['fs_volume_8h']=qv.rolling(96,min_periods=96).sum()/qv.rolling(384,min_periods=384).sum().mul(.25).replace(0,np.nan)
 premium=f.get('premium_index',m.get('premium_index',pd.Series(0,index=m.index))).astype(float);f['fs_premium']=premium;f['fs_basis']=premium-f.fs_rate
 oi=m.get('open_interest',pd.Series(np.nan,index=m.index)).astype(float).replace(0,np.nan);f['fs_oi_ret_8h']=np.log(oi/oi.shift(96))
 return f.replace([np.inf,-np.inf],np.nan)
def specs(f,tr):
 out=[]
 def z(ts):return [(c,o,.5) if c=='fs_event' else (c,o,q(f,tr,c,v)) for c,o,v in ts]
 for fq,rq,bq in itertools.product((.7,.8,.9),(.6,.75,.85),(.6,.75,.85)):
  out.append((f'crowded_unwind_{fq}_{rq}_{bq}',z([('fs_event','ge',.5),('fs_rate','le',1-fq),('fs_price_ret_8h','le',1-rq),('fs_basis','le',1-bq)]),z([('fs_event','ge',.5),('fs_rate','ge',fq),('fs_price_ret_8h','ge',rq),('fs_basis','ge',bq)])))
 for dq,fq,oq in itertools.product((.7,.85),(.6,.75),(.55,.7)):
  out.append((f'funding_flip_{dq}_{fq}_{oq}',z([('fs_event','ge',.5),('fs_delta','ge',dq),('fs_flow_8h','ge',fq),('fs_price_ret_24h','ge',oq)]),z([('fs_event','ge',.5),('fs_delta','le',1-dq),('fs_flow_8h','le',1-fq),('fs_price_ret_24h','le',1-oq)])))
 for bq,vq in itertools.product((.1,.2,.8,.9),(.6,.8)):
  if bq<.5: out.append((f'negative_basis_rebound_{bq}_{vq}',z([('fs_event','ge',.5),('fs_basis','le',bq),('fs_volume_8h','ge',vq)]),z([('fs_event','ge',.5),('fs_basis','ge',1-bq),('fs_volume_8h','ge',vq)])))
 return out
def run(cfg):
 m=_load_market(cfg);f=features(m,cfg);dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);rows=[]
 for name,lc,sc in specs(f,tr):
  la,sa=mk(f,lc),mk(f,sc)
  if int((la|sa)[tr].sum())<50:continue
  for hold,(tp,sl) in itertools.product((24,48,72,96,144),((.01,.008),(.015,.01),(.025,.015),(.04,.025))):
   s=sim(m,dates,la,sa,cfg,hold,1,tp,sl,'test2024')
   if s['trades']>=20 and s['longs']>=4 and s['shorts']>=4:rows.append({'name':name,'long_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in lc],'short_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in sc],'hold_bars':hold,'stride_bars':1,'tp':tp,'sl':sl,'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  la=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['long_conditions']]);sa=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['short_conditions']])
  for sp in ('train','eval2025','ytd2026'):r[sp]=sim(m,dates,la,sa,cfg,r['hold_bars'],1,r['tp'],r['sl'],sp)
  e=r['eval2025'];en=e['trades']>=12 and e['longs']>=3 and e['shorts']>=3;r['passes_alpha_pool']=bool(en and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5);r['passes_live_grade']=bool(en and r['test2024']['ratio']>=3 and e['ratio']>=3 and r['ytd2026']['trades']>=6 and r['ytd2026']['ratio']>=5)
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'post-funding settlement only; current settlement and completed prior bars; train-frozen thresholds; test2024 rank; sealed eval/2026; entry+1; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',required=True);p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=str))
if __name__=='__main__':main()
