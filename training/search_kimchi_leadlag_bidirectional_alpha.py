"""Bidirectional BTC alpha scan from causal Kimchi-premium lead/lag states."""
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
def z(s,n):
 mu=s.rolling(n,min_periods=n).mean();sd=s.rolling(n,min_periods=n).std().replace(0,np.nan);return (s-mu)/sd
def features(m,base):
 f=base.copy();btc=np.log(m.close.astype(float));kp=pd.to_numeric(m.kimchi_premium,errors='coerce').replace([np.inf,-np.inf],np.nan).ffill();krw=np.log(pd.to_numeric(m.usdkrw,errors='coerce').replace(0,np.nan).ffill())
 for n in (12,48,144,288):
  k=kp-kp.shift(n);b=btc-btc.shift(n);u=krw-krw.shift(n);f[f'kl_kimchi_delta_{n}']=k;f[f'kl_btc_ret_{n}']=b;f[f'kl_fx_delta_{n}']=u;f[f'kl_kimchi_btc_gap_{n}']=z(k,576)-z(b,576);f[f'kl_local_impulse_{n}']=z(k,576)-z(u,576)
 f['kl_accel_48_144']=f['kl_kimchi_delta_48']-f['kl_kimchi_delta_144']/3;return f.replace([np.inf,-np.inf],np.nan)
def specs(f,tr):
 out=[]
 def add(n,l,s):out.append((n,[(c,o,q(f,tr,c,zv)) for c,o,zv in l],[(c,o,q(f,tr,c,zv)) for c,o,zv in s]))
 for n in (48,144,288):
  for kq,bq in itertools.product((.8,.9,.95),(.2,.3,.5)):
   add(f'kimchi_lead_continuation_{n}',[(f'kl_kimchi_delta_{n}','ge',kq),(f'kl_btc_ret_{n}','le',bq)],[(f'kl_kimchi_delta_{n}','le',1-kq),(f'kl_btc_ret_{n}','ge',1-bq)])
   add(f'kimchi_gap_convergence_{n}',[(f'kl_kimchi_btc_gap_{n}','ge',kq),(f'kl_btc_ret_{n}','le',bq)],[(f'kl_kimchi_btc_gap_{n}','le',1-kq),(f'kl_btc_ret_{n}','ge',1-bq)])
  for iq,bq in itertools.product((.8,.9),(.2,.3)):
   add(f'local_impulse_lead_{n}',[(f'kl_local_impulse_{n}','ge',iq),(f'kl_btc_ret_{n}','le',bq)],[(f'kl_local_impulse_{n}','le',1-iq),(f'kl_btc_ret_{n}','ge',1-bq)])
 for aq,bq in itertools.product((.8,.9),(.2,.3)):
  add('kimchi_acceleration',[('kl_accel_48_144','ge',aq),('kl_btc_ret_48','le',bq)],[('kl_accel_48_144','le',1-aq),('kl_btc_ret_48','ge',1-bq)])
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
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'kimchi leadlag bidirectional BTC; train thresholds; test-only rank; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
