"""Search standalone BTC policies that trade both long and short directions."""
from __future__ import annotations
import argparse,itertools,json,math
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import LongComboScanConfig,_load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features

@dataclass(frozen=True)
class Config(LongComboScanConfig):
 exclude_from:str='2026-06-02';fee_rate:float=.0005;slippage_rate:float=.0001;leverage:float=.5;min_test_trades:int=30;min_each_side:int=6;top_n:int=250
W={'train':('2020-01-01','2024-01-01'),'test2024':('2024-01-01','2025-01-01'),'eval2025':('2025-01-01','2026-01-01'),'ytd2026':('2026-01-01','2026-06-02')}
def q(f,m,c,z):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,z))
def mk(f,conds):
 a=np.ones(len(f),bool)
 for c,o,t in conds:
  x=f[c].to_numpy(float);a&=np.isfinite(x)&((x>=t) if o=='ge' else (x<=t))
 return a
def extra(m,f):
 c=m.close.astype(float);qv=m.quote_asset_volume.astype(float);buy=m.taker_buy_quote.astype(float);imb=(2*buy/qv.replace(0,np.nan)-1).clip(-1,1)
 for n in (12,24,48,96,144):f[f'bd_ret_{n}']=np.log(c/c.shift(n));f[f'bd_imb_{n}']=imb.rolling(n,min_periods=n).mean()
 f['bd_flow_accel']=f['bd_imb_12']-f['bd_imb_48'];return f
def specs(f,tr):
 out=[]
 def add(n,l,s):out.append((n,[(c,o,q(f,tr,c,z)) for c,o,z in l],[(c,o,q(f,tr,c,z)) for c,o,z in s]))
 for xq,pq in itertools.product((.05,.1,.2),(.05,.1,.2)):
  add('usdkrw_symmetric', [('usdkrw_momentum','le',xq),('htf_1d_return_1','ge',1-pq)], [('usdkrw_momentum','ge',1-xq),('htf_1d_return_1','le',pq)])
  add('dxy_symmetric', [('dxy_momentum','le',xq),('htf_1d_return_1','ge',1-pq)], [('dxy_momentum','ge',1-xq),('htf_1d_return_1','le',pq)])
  add('premium_flow_reversion', [('premium_index_zscore','le',xq),('bd_flow_accel','ge',1-pq)], [('premium_index_zscore','ge',1-xq),('bd_flow_accel','le',pq)])
  add('kimchi_flow_reversion', [('kimchi_premium_change','le',xq),('bd_flow_accel','ge',1-pq)], [('kimchi_premium_change','ge',1-xq),('bd_flow_accel','le',pq)])
 # Asymmetric policy: the best economically distinct long/short mechanisms.
 for lq,sq in itertools.product((.05,.1,.2),(.05,.1,.2)):
  add('funding_relief_vs_fx_stress',[('funding_rate','le',lq),('bd_flow_accel','ge',.8)],[('usdkrw_momentum','ge',1-sq),('htf_1d_return_1','le',sq)])
 return out
def sim(m,dates,long_a,short_a,cfg,hold,stride,tp,sl,w):
 st,en=W[w];wm=_split_mask(dates,st,en);op=m.open.to_numpy(float);hi=m.high.to_numpy(float);lo=m.low.to_numpy(float);cost=(cfg.fee_rate+cfg.slippage_rate)*cfg.leverage
 pos=np.arange(143,len(m)-hold-2,stride,dtype=np.int64);pos=pos[wm[pos]&(long_a[pos]|short_a[pos])];eq=peak=1.;mdd=0.;nxt=0;rets=[];sides=[]
 for p in pos:
  if p<nxt:continue
  side=1 if long_a[p] and not short_a[p] else (-1 if short_a[p] and not long_a[p] else 0)
  if not side:continue
  ep=p+1;cap=ep+hold
  if cap>=len(m) or not wm[min(cap,len(wm)-1)]:continue
  entry=op[ep];entry_eq=eq;eq*=1-cost;mdd=max(mdd,1-eq/peak);xp=cap;exit_ret=side*(op[cap]/entry-1)
  for j in range(ep,cap):
   adverse=(lo[j]/entry-1) if side>0 else (1-hi[j]/entry);mdd=max(mdd,1-max(0.,eq*(1+cfg.leverage*adverse))/peak)
   stop_hit=(lo[j]<=entry*(1-sl)) if side>0 else (hi[j]>=entry*(1+sl))
   take_hit=(hi[j]>=entry*(1+tp)) if side>0 else (lo[j]<=entry*(1-tp))
   if stop_hit:exit_ret=-sl;xp=j;break
   if take_hit:exit_ret=tp;xp=j;break
  eq*=max(0.,1+cfg.leverage*exit_ret);eq*=1-cost;mdd=max(mdd,1-eq/peak);peak=max(peak,eq);rets.append(eq/entry_eq-1);sides.append(side);nxt=xp+1
 years=(pd.Timestamp(en)-pd.Timestamp(st)).total_seconds()/(365.25*86400);ret=(eq-1)*100;cagr=(eq**(1/years)-1)*100 if eq>0 else -100;md=mdd*100;a=np.array(rets);sh=float(a.mean()/a.std(ddof=1)*np.sqrt(len(a))) if len(a)>1 and a.std(ddof=1)>0 else 0
 return {'return_pct':ret,'cagr_pct':cagr,'strict_mdd_pct':md,'ratio':cagr/md if md>1e-12 else 0,'trades':len(rets),'longs':sum(x>0 for x in sides),'shorts':sum(x<0 for x in sides),'win_rate':float((a>0).mean()) if len(a) else 0,'long_win_rate':float(np.mean([r>0 for r,s in zip(rets,sides) if s>0])) if any(s>0 for s in sides) else 0,'short_win_rate':float(np.mean([r>0 for r,s in zip(rets,sides) if s<0])) if any(s<0 for s in sides) else 0,'sharpe_like':sh}
def run(cfg):
 m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=extra(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);rows=[]
 for name,lc,sc in specs(f,tr):
  la,sa=mk(f,lc),mk(f,sc)
  for hold,stride,(tp,sl) in itertools.product((48,72,96,144,216,288),(6,12,24),((.01,.008),(.015,.01),(.025,.015),(.04,.025))):
   z=sim(m,dates,la,sa,cfg,hold,stride,tp,sl,'test2024')
   if z['trades']>=cfg.min_test_trades and z['longs']>=cfg.min_each_side and z['shorts']>=cfg.min_each_side:rows.append({'name':name,'long_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in lc],'short_conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in sc],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'test2024':z})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  la=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['long_conditions']]);sa=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['short_conditions']])
  for w in ('train','eval2025','ytd2026'):r[w]=sim(m,dates,la,sa,cfg,r['hold_bars'],r['stride_bars'],r['tp'],r['sl'],w)
  e=r['eval2025'];enough=e['trades']>=16 and e['longs']>=4 and e['shorts']>=4;r['passes_alpha_pool']=enough and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5;r['passes_live_grade']=enough and r['test2024']['ratio']>=3 and e['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=8 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'bidirectional BTC policy; train thresholds; test-only rank; sealed eval/2026; 6bp/side; strict intrabar MDD; both sides required','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default=Config.exclude_from);a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
