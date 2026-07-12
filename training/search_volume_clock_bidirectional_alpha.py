"""Bidirectional BTC alpha in causal volume time rather than wall-clock time."""
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
 f=base.copy();qv=m.quote_asset_volume.astype(float).fillna(0).clip(lower=0);buy=m.taker_buy_quote.astype(float).fillna(0);signed=2*buy-qv;c=np.log(m.close.astype(float));cv=np.r_[0.,np.cumsum(qv.to_numpy(float))];cs=np.r_[0.,np.cumsum(signed.to_numpy(float))]
 # Target notional is based only on the completed preceding 24h window.
 daily=qv.rolling(288,min_periods=288).sum().shift(1).to_numpy(float);idx=np.arange(len(m))
 for frac in (.25,.5,1.0):
  target=daily*frac;level=cv[1:]-np.nan_to_num(target,nan=np.inf);j=np.searchsorted(cv,level,side='left').clip(0,len(m)-1);valid=np.isfinite(target)&(j<idx);duration=(idx-j).astype(float);ret=c.to_numpy(float)-c.to_numpy(float)[j];vol=cv[idx+1]-cv[j];sf=cs[idx+1]-cs[j]
  ret[~valid]=np.nan;duration[~valid]=np.nan;vol[~valid]=np.nan;sf[~valid]=np.nan
  tag=str(frac).replace('.','p');f[f'vc_ret_{tag}']=ret;f[f'vc_duration_{tag}']=duration;f[f'vc_imbalance_{tag}']=sf/np.where(vol==0,np.nan,vol);f[f'vc_speed_{tag}']=ret/np.where(duration==0,np.nan,duration);f[f'vc_flow_speed_{tag}']=(sf/np.where(vol==0,np.nan,vol))/np.where(duration==0,np.nan,duration)
 return f.replace([np.inf,-np.inf],np.nan)
def specs(f,tr):
 out=[]
 def add(n,l,s):out.append((n,[(c,o,q(f,tr,c,z)) for c,o,z in l],[(c,o,q(f,tr,c,z)) for c,o,z in s]))
 for tag in ('0p25','0p5','1p0'):
  for dq,rq,iq in itertools.product((.1,.2,.8,.9),(.1,.2),(.1,.2,.8,.9)):
   if dq<.5:
    add(f'fast_clock_continuation_{tag}',[(f'vc_duration_{tag}','le',dq),(f'vc_ret_{tag}','ge',1-rq),(f'vc_imbalance_{tag}','ge',max(iq,.7))],[(f'vc_duration_{tag}','le',dq),(f'vc_ret_{tag}','le',rq),(f'vc_imbalance_{tag}','le',min(iq,.3))])
    add(f'fast_clock_reversion_{tag}',[(f'vc_duration_{tag}','le',dq),(f'vc_ret_{tag}','le',rq),(f'vc_imbalance_{tag}','ge',max(iq,.7))],[(f'vc_duration_{tag}','le',dq),(f'vc_ret_{tag}','ge',1-rq),(f'vc_imbalance_{tag}','le',min(iq,.3))])
   else:
    add(f'slow_clock_breakout_{tag}',[(f'vc_duration_{tag}','ge',dq),(f'vc_speed_{tag}','ge',1-rq)],[(f'vc_duration_{tag}','ge',dq),(f'vc_speed_{tag}','le',rq)])
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
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'causal volume-clock bidirectional BTC; target notional from shifted trailing 24h; train thresholds; test-only rank; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
