"""Observable Markov-transition gate for the fixed funding/premium long setup."""
from __future__ import annotations
import argparse,itertools,json,sys
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
if __package__ is None or __package__=="":sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_new_alpha_pool import _alpha_active,_event_path
from training.search_bidirectional_state_alpha import Config,sim
from training.search_gaussian_hmm_regime_alpha import SPLITS,hourly_features
from training.long_regime_combo_scan import _load_market,_split_mask

def run(cfg):
 m=_load_market(cfg);dates=pd.to_datetime(m.date);base=build_market_feature_frame(m,window_size=144);feat=pd.concat([base,build_interest_features(m,base)],axis=1).loc[:,lambda x:~x.columns.duplicated(keep='last')];setup=_alpha_active(feat,'long_minimal_funding_premium');h,hf=hourly_features(m);train_h=(hf.index>=SPLITS['train'][0])&(hf.index<SPLITS['train'][1]);train_m=_split_mask(dates,*SPLITS['train']);positions=np.arange(143,len(m)-578,12);rows=[]
 for lo,hi in ((.2,.8),(.25,.75),(.33,.67),(.4,.6)):
  tq=(hf.loc[train_h,'trend24'].quantile(lo),hf.loc[train_h,'trend24'].quantile(hi));vq=hf.loc[train_h,'vol24'].quantile(.5);fq=hf.loc[train_h,'flow24'].quantile(.5)
  trend=np.where(hf.trend24<=tq[0],0,np.where(hf.trend24>=tq[1],2,1));vol=(hf.vol24>=vq).astype(int);flow=(hf.flow24>=fq).astype(int);state=trend*4+vol*2+flow;prev=pd.Series(state,index=hf.index).shift(1).fillna(-1).astype(int);key=prev*12+state
  hd=pd.DataFrame({'date':hf.index.to_numpy(),'state':state,'prev':prev.to_numpy(),'key':key.to_numpy()}).reset_index(drop=True);mapped=pd.merge_asof(pd.DataFrame({'date':dates,'pos':np.arange(len(m))}),hd,on='date',direction='backward',tolerance=pd.Timedelta('2h')).sort_values('pos');keys=mapped.key.fillna(-1).to_numpy(int);states=mapped.state.fillna(-1).to_numpy(int)
  trans=np.zeros((12,12),int)
  for a,b in zip(state[:-1],state[1:]):trans[int(a),int(b)]+=1
  prob=trans/np.maximum(trans.sum(1,keepdims=True),1);quality={};next_allowed=0
  for pos in positions[setup[positions]&train_m[positions]]:
   if pos<next_allowed or keys[pos]<0:continue
   ep=_event_path(m,int(pos),side='long',hold=576,cost_rate=.0006,entry_delay=1,leverage=.5)
   if ep is None:continue
   quality.setdefault(int(keys[pos]),[]).append(float(ep[2]));next_allowed=int(pos)+577
  for min_n,min_edge,min_prob in itertools.product((3,5,8,12),(0,.002,.005,.01),(.0,.05,.1,.2)):
   allowed=[]
   for k,x in quality.items():
    a,b=divmod(k,12)
    if len(x)>=min_n and np.mean(x)>=min_edge and prob[a,b]>=min_prob:allowed.append(k)
   if not allowed:continue
   gate=np.isin(keys,allowed);la=setup&gate;sa=np.zeros(len(m),bool);t=sim(m,dates,la,sa,cfg,576,12,10.,10.,'test2024')
   if t['trades']>=10:rows.append({'trend_quantiles':[lo,hi],'state_thresholds':{'trend_low':float(tq[0]),'trend_high':float(tq[1]),'vol_median':float(vq),'flow_median':float(fq)},'min_train_transition_trades':min_n,'min_train_trade_edge':min_edge,'min_transition_probability':min_prob,'allowed_transition_keys':allowed,'transition_quality':{str(k):{'n':len(v),'mean_trade_return':float(np.mean(v))} for k,v in quality.items() if k in allowed},'test2024':t,'_la':la})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:100]
 for r in sel:
  la=r.pop('_la');sa=np.zeros(len(m),bool)
  for sp in ('train','eval2025','ytd2026'):r[sp]=sim(m,dates,la,sa,cfg,576,12,10.,10.,sp)
  r['passes_alpha_pool']=bool(r['test2024']['ratio']>=3 and r['eval2025']['ratio']>=3 and r['test2024']['trades']>=10 and r['eval2025']['trades']>=10);r['passes_live_grade']=bool(r['passes_alpha_pool'] and r['ytd2026']['ratio']>=5 and r['ytd2026']['trades']>=6)
 baseline={sp:sim(m,dates,setup,np.zeros(len(m),bool),cfg,576,12,10.,10.,sp) for sp in SPLITS}
 stress={};leave_one={};yearly={}
 if sel:
  top=sel[0];th=top['state_thresholds'];trend=np.where(hf.trend24<=th['trend_low'],0,np.where(hf.trend24>=th['trend_high'],2,1));vol=(hf.vol24>=th['vol_median']).astype(int);flow=(hf.flow24>=th['flow_median']).astype(int);state=trend*4+vol*2+flow;key=pd.Series(state,index=hf.index).shift(1).fillna(-1).astype(int)*12+state;hd=pd.DataFrame({'date':hf.index.to_numpy(),'key':key.to_numpy()});mapped=pd.merge_asof(pd.DataFrame({'date':dates,'pos':np.arange(len(m))}),hd,on='date',direction='backward',tolerance=pd.Timedelta('2h')).sort_values('pos');keys=mapped.key.fillna(-1).to_numpy(int)
  for bps in (6,8,10,15):
   scfg=replace(cfg,fee_rate=max(0,bps/10000-cfg.slippage_rate));la=setup&np.isin(keys,top['allowed_transition_keys']);stress[str(bps)]={sp:sim(m,dates,la,np.zeros(len(m),bool),scfg,576,12,10.,10.,sp) for sp in SPLITS}
  for dropped in top['allowed_transition_keys']:
   la=setup&np.isin(keys,[x for x in top['allowed_transition_keys'] if x!=dropped]);leave_one[str(dropped)]={sp:sim(m,dates,la,np.zeros(len(m),bool),cfg,576,12,10.,10.,sp) for sp in SPLITS}
  la=setup&np.isin(keys,top['allowed_transition_keys'])
  for year in range(2020,2027):
   state_sim.W['yearly']=(f'{year}-01-01',f'{year+1}-01-01' if year<2026 else '2026-06-02');yearly[str(year)]=sim(m,dates,la,np.zeros(len(m),bool),cfg,576,12,10.,10.,'yearly')
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'base_setup':'long_minimal_funding_premium','protocol':'observable hourly state transition; bins and transition quality fit train only; test2024 rank; sealed eval/2026; hold576; 6bp/side; strict MDD','tested':len(rows),'baseline':baseline,'cost_stress_bps_per_side':stress,'leave_one_transition_out':leave_one,'yearly_top_candidate':yearly,'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=str))
if __name__=='__main__':main()
