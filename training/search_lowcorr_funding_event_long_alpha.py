"""Low-correlation BTC long alpha scan around funding settlement events."""
from __future__ import annotations
import argparse,itertools,json
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import LongComboScanConfig,_load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_component_tp_union_scan import COMPONENTS,_component_mask,_strict_long_overlay_sim
from training.strict_bar_backtest import _trade_stats

@dataclass(frozen=True)
class Config(LongComboScanConfig):
 exclude_from:str='2026-06-02';fee_rate:float=.0005;slippage_rate:float=.0001;leverage:float=.5;max_abs_phi:float=.20;min_test_trades:int=16;top_n:int=250
W={'train':('2020-01-01','2024-01-01'),'test2024':('2024-01-01','2025-01-01'),'eval2025':('2025-01-01','2026-01-01'),'ytd2026':('2026-01-01','2026-06-02')}
def q(f,m,c,z):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,z))
def mk(f,conds,event):
 a=event.copy()
 for c,o,t in conds:
  x=f[c].to_numpy(float);a&=np.isfinite(x)&((x>=t) if o=='ge' else (x<=t))
 return a
def corr(a,b):return float(np.corrcoef(a.astype(float),b.astype(float))[0,1]) if a.std() and b.std() else 0.
def baselines(f):
 b={n:_component_mask(f,n) for n in COMPONENTS}
 for n,ms in {'long_core_union':['range_bb90','funding10_trend70','premium20_mom90'],'long_alt_union':['funding10_trend70','compress05_trend80','premium20_mom90'],'long_range_union':['range_bb90','mom85_pos50','compress05_trend80']}.items():
  a=np.zeros(len(f),bool)
  for x in ms:a|=b[x]
  b[n]=a
 return b
def extra(m,f):
 c=m.close.astype(float);qv=m.quote_asset_volume.astype(float);buy=m.taker_buy_quote.astype(float);imb=(2*buy/qv.replace(0,np.nan)-1).clip(-1,1)
 for n in (12,24,48,96):f[f'fe_ret_{n}']=np.log(c/c.shift(n));f[f'fe_imb_{n}']=imb.rolling(n,min_periods=n).mean()
 f['fe_flow_recovery']=f['fe_imb_12']-f['fe_imb_48'];return f
def events(dates):
 h=dates.dt.hour.to_numpy();minute=dates.dt.minute.to_numpy()
 return {'at_settlement':np.isin(h,[0,8,16])&(minute==0),'post_settlement_30m':np.isin(h,[0,8,16])&(minute<=30),'pre_settlement_30m':np.isin((h+1)%24,[0,8,16])&(minute>=30),'asia_open':(h==0)&(minute<=30)}
def candidates(f,tr):
 out=[]
 def add(n,event,raw):out.append((n,event,[(c,o,q(f,tr,c,z)) for c,o,z in raw]))
 for fq,rq in itertools.product((.05,.1,.2,.3),(.05,.1,.2,.3)):
  add('negative_funding_pullback','post_settlement_30m',[('funding_rate','le',fq),('fe_ret_48','le',rq)])
  add('negative_funding_recovery','post_settlement_30m',[('funding_rate','le',fq),('fe_flow_recovery','ge',1-rq)])
 for pq,rq in itertools.product((.05,.1,.2),(.05,.1,.2,.3)):
  add('premium_discount_recovery','post_settlement_30m',[('premium_index_zscore','le',pq),('fe_flow_recovery','ge',1-rq)])
  add('premium_discount_pullback','at_settlement',[('premium_index_zscore','le',pq),('fe_ret_24','le',rq)])
 for kq,rq in itertools.product((.1,.2,.8,.9),(.1,.2,.8,.9)):
  kop='le' if kq<.5 else 'ge';rop='le' if rq<.5 else 'ge';add('asia_kimchi_dislocation','asia_open',[('kimchi_premium_change',kop,kq),('fe_ret_48',rop,rq)])
 return out
def score(m,dates,a,cfg,hold,stride,tp,sl,w):
 st,en=W[w];wm=_split_mask(dates,st,en);pos=np.arange(143,len(m)-hold-2,stride,dtype=np.int64);sig=pos[a[pos]&wm[pos]];sim,rets=_strict_long_overlay_sim(sig,market=m,hold_bars=hold,entry_delay_bars=1,leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate,take_profit=tp,stop_loss=sl,annualization_start=st,annualization_end=en);ts=_trade_stats(rets)
 return {'ret_pct':sim['total_return_pct'],'cagr_pct':sim['cagr_pct'],'mdd_pct':sim['strict_mdd_pct'],'ratio':sim['cagr_to_strict_mdd'],'trades':sim['trade_entries'],'win_rate':sim['win_rate'],'sharpe_like':float(ts['effect_size_d'])*np.sqrt(ts['n_trades']),'p':ts['p_value_mean_ret_approx'],'exits':sim.get('exit_reasons',{})}
def run(cfg):
 m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=extra(m,pd.concat([base,build_interest_features(m,base)],axis=1));dates=pd.to_datetime(m.date);ev=events(dates);tr=_split_mask(dates,*W['train']);te=_split_mask(dates,*W['test2024']);bs=baselines(f);rows=[]
 for name,event,conds in candidates(f,tr):
  a=mk(f,conds,ev[event]);cs={n:corr(a[te],x[te]) for n,x in bs.items()};near=max(cs,key=lambda n:abs(cs[n]));mc=abs(cs[near])
  if mc>cfg.max_abs_phi or int((a&tr).sum())<80:continue
  for hold,stride,(tp,sl) in itertools.product((12,24,48,72,96,144),(1,3,6),((None,None),(.01,.008),(.015,.01),(.025,.015),(.04,.025))):
   s=score(m,dates,a,cfg,hold,stride,tp,sl,'test2024')
   if s['trades']>=cfg.min_test_trades:rows.append({'name':name,'event':event,'conditions':[{'feature':c,'op':o,'threshold':t} for c,o,t in conds],'hold_bars':hold,'stride_bars':stride,'tp':tp,'sl':sl,'max_abs_phi_test':mc,'nearest':near,'nearest_phi':cs[near],'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio']-.5*r['max_abs_phi_test'],r['test2024']['ret_pct']),reverse=True);sel=rows[:cfg.top_n]
 for r in sel:
  a=mk(f,[(x['feature'],x['op'],x['threshold']) for x in r['conditions']],ev[r['event']])
  for w in ('train','eval2025','ytd2026'):r[w]=score(m,dates,a,cfg,r['hold_bars'],r['stride_bars'],r['tp'],r['sl'],w)
  enough=r['eval2025']['trades']>=8;r['passes_alpha_pool']=enough and r['test2024']['ratio']>=2.5 and r['eval2025']['ratio']>=2.5;r['passes_live_grade']=enough and r['test2024']['ratio']>=3 and r['eval2025']['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=6 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'funding event-time long; train thresholds; test-only rank; max phi<=.20; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default=Config.exclude_from);a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
