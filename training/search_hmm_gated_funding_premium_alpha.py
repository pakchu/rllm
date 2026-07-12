"""Use a causal Gaussian HMM as a sparse regime gate for a fixed long setup."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
if __package__ is None or __package__=="":sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_new_alpha_pool import _alpha_active,_event_path
from training.search_bidirectional_state_alpha import Config,sim
from training.search_gaussian_hmm_regime_alpha import SPLITS,filtered,fit_hmm,hourly_features
from training.long_regime_combo_scan import _load_market,_split_mask

def run(cfg):
 m=_load_market(cfg);dates=pd.to_datetime(m.date);base=build_market_feature_frame(m,window_size=144);feat=pd.concat([base,build_interest_features(m,base)],axis=1).loc[:,lambda x:~x.columns.duplicated(keep='last')];setup=_alpha_active(feat,'long_minimal_funding_premium')
 h,hf=hourly_features(m);train_h=(hf.index>=SPLITS['train'][0])&(hf.index<SPLITS['train'][1]);feature_sets={'core':['ret1','trend24','vol24','volterm'],'flow':['ret1','trend24','trend72','vol24','volterm','flow24']};rows=[]
 train_mask=_split_mask(dates,*SPLITS['train'])
 for fs,cols in feature_sets.items():
  good=hf[cols].notna().all(1);fitmask=train_h&good;raw=hf.loc[fitmask,cols].to_numpy();mean=raw.mean(0);std=raw.std(0);std[std<1e-8]=1;xall=((hf.loc[good,cols].to_numpy()-mean)/std).clip(-8,8)
  for k in (3,4,5):
   model=fit_hmm(((raw-mean)/std).clip(-8,8),k);p=filtered(xall,model);hd=pd.DataFrame({'date':hf.index[good],'state':p.argmax(1),'confidence':p.max(1)});mapped=pd.merge_asof(pd.DataFrame({'date':dates,'pos':np.arange(len(m))}),hd,on='date',direction='backward',tolerance=pd.Timedelta('2h')).sort_values('pos');state=mapped.state.fillna(-1).to_numpy(int);confidence=mapped.confidence.fillna(0).to_numpy(float)
   state_returns=[[] for _ in range(k)];next_allowed=0;positions=np.arange(143,len(m)-578,12)
   for pos in positions[setup[positions]&train_mask[positions]]:
    if pos<next_allowed or state[pos]<0:continue
    ep=_event_path(m,int(pos),side='long',hold=576,cost_rate=.0006,entry_delay=1,leverage=.5)
    if ep is None:continue
    state_returns[state[pos]].append(float(ep[2]));next_allowed=int(pos)+577
   state_mean=[float(np.mean(x)) if x else 0 for x in state_returns];state_n=[len(x) for x in state_returns]
   for min_conf in (.35,.45,.55,.65,.75):
    for min_edge in (-.002,0,.002,.005,.01):
     allowed=np.array([(state_n[j]>=3 and state_mean[j]>=min_edge) for j in range(k)]);gate=(state>=0)&allowed[np.clip(state,0,k-1)]&(confidence>=min_conf);la=setup&gate;sa=np.zeros(len(m),bool)
     t=sim(m,dates,la,sa,cfg,576,12,10.0,10.0,'test2024')
     if t['trades']>=10:rows.append({'feature_set':fs,'states':k,'min_confidence':min_conf,'min_state_trade_edge':min_edge,'allowed_states':np.flatnonzero(allowed).tolist(),'state_train_mean_trade_return':state_mean,'state_train_trades':state_n,'transition_matrix':model['A'].tolist(),'test2024':t,'_la':la})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:100]
 for r in sel:
  la=r.pop('_la');sa=np.zeros(len(m),bool)
  for sp in ('train','eval2025','ytd2026'):r[sp]=sim(m,dates,la,sa,cfg,576,12,10.0,10.0,sp)
  r['passes_alpha_pool']=bool(r['test2024']['ratio']>=3 and r['eval2025']['ratio']>=3 and r['test2024']['trades']>=10 and r['eval2025']['trades']>=10)
  r['passes_live_grade']=bool(r['passes_alpha_pool'] and r['ytd2026']['ratio']>=5 and r['ytd2026']['trades']>=6)
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'base_setup':'long_minimal_funding_premium','protocol':'HMM fit pre-2024; causal filter; state trade quality fit train only; test2024 rank; sealed eval/2026; base event hold576; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=str))
if __name__=='__main__':main()
