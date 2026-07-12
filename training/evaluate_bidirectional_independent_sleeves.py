"""Evaluate a fixed bidirectional candidate as simultaneous independent sleeves."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config,W,extra,mk

def event_path(m,p,side,hold,tp,sl,lev,cost):
 n=len(m);o=m.open.to_numpy(float);h=m.high.to_numpy(float);l=m.low.to_numpy(float);ep=p+1;cap=ep+hold
 if cap>=n:return None
 r=np.zeros(n);a=np.zeros(n);r[ep]-=cost*lev;entry=o[ep];xp=cap
 for j in range(ep,cap):
  adverse=(l[j]/o[j]-1) if side>0 else (1-h[j]/o[j]);a[j]+=lev*adverse
  stop=(l[j]<=entry*(1-sl)) if side>0 else (h[j]>=entry*(1+sl));take=(h[j]>=entry*(1+tp)) if side>0 else (l[j]<=entry*(1-tp))
  if stop:
   target=entry*(1-sl) if side>0 else entry*(1+sl);r[j]+=lev*side*(target/o[j]-1);xp=j;break
  if take:
   target=entry*(1+tp) if side>0 else entry*(1-tp);r[j]+=lev*side*(target/o[j]-1);xp=j;break
  r[j+1]+=lev*side*(o[j+1]/o[j]-1)
 r[xp]-=cost*lev
 fac=float(np.prod(np.maximum(0,1+r)));return r,a,fac-1,xp
def sleeve(m,dates,active,side,spec,cfg,w):
 st,en=W[w];wm=_split_mask(dates,st,en);pos=np.arange(143,len(m)-spec['hold_bars']-2,spec['stride_bars'],dtype=np.int64);pos=pos[active[pos]&wm[pos]];R=np.zeros(len(m));A=np.zeros(len(m));rets=[];nxt=0
 for p in pos:
  if p<nxt:continue
  x=event_path(m,int(p),side,spec['hold_bars'],spec['tp'],spec['sl'],cfg.leverage,cfg.fee_rate+cfg.slippage_rate)
  if x is None:continue
  r,a,rr,xp=x
  if not wm[min(xp,len(wm)-1)]:continue
  R+=r;A+=a;rets.append(rr);nxt=xp+1
 return R,A,rets
def stats(R,A,dates,w,long_rets,short_rets):
 st,en=W[w];wm=_split_mask(dates,st,en);idx=np.flatnonzero(wm);r=R[idx];a=A[idx];eq=np.cumprod(np.maximum(0,1+r));base=np.r_[1.,eq[:-1]];peak=np.maximum.accumulate(eq);peakb=np.maximum.accumulate(base);mdd=max(float(np.max(1-eq/np.maximum(peak,1e-12))),float(np.max(1-base*(1+a)/np.maximum(peakb,1e-12))))*100;final=float(eq[-1]);years=(pd.Timestamp(en)-pd.Timestamp(st)).total_seconds()/(365.25*86400);cagr=(final**(1/years)-1)*100 if final>0 else -100;allr=np.array(long_rets+short_rets);sh=float(allr.mean()/allr.std(ddof=1)*np.sqrt(len(allr))) if len(allr)>1 and allr.std(ddof=1)>0 else 0
 return {'return_pct':(final-1)*100,'cagr_pct':cagr,'strict_mdd_pct':mdd,'ratio':cagr/mdd if mdd>1e-12 else 0,'trades':len(allr),'longs':len(long_rets),'shorts':len(short_rets),'long_win_rate':float(np.mean(np.array(long_rets)>0)) if long_rets else 0,'short_win_rate':float(np.mean(np.array(short_rets)>0)) if short_rets else 0,'sharpe_like':sh}
def run(args):
 D=json.load(open(args.candidate_json));spec=D['alpha_pool_qualifiers'][0];cfg=Config(input_csv=args.input_csv,output='/tmp/x',funding_csv=args.funding_csv,premium_csv=args.premium_csv,exclude_from=args.exclude_from);m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=extra(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);la=mk(f,[(x['feature'],x['op'],x['threshold']) for x in spec['long_conditions']]);sa=mk(f,[(x['feature'],x['op'],x['threshold']) for x in spec['short_conditions']]);out={'candidate':spec['name'],'execution':'independent simultaneous long/short sleeves; same fixed candidate parameters','config':{'cost_per_side':cfg.fee_rate+cfg.slippage_rate,'leverage_per_sleeve':cfg.leverage,'gross_if_both':2*cfg.leverage},'stats':{}}
 for w in W:
  lr,laa,lrets=sleeve(m,dates,la,1,spec,cfg,w);sr,saa,srets=sleeve(m,dates,sa,-1,spec,cfg,w);out['stats'][w]=stats(lr+sr,laa+saa,dates,w,lrets,srets)
 Path(args.output).write_text(json.dumps(out,indent=2));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-json',required=True);p.add_argument('--input-csv',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a),indent=2))
if __name__=='__main__':main()
