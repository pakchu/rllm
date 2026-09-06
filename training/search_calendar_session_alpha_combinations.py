"""Search calendar/session flow alphas and prequential calendar effects."""
from __future__ import annotations
import argparse,hashlib,json
import numpy as np,pandas as pd
from training import search_meaningful_alpha_combinations as base
from training import search_regime_diverse_alpha_combinations as regime
OUT=base.ROOT/'research/calendar_session_combinations'
DESIGN={'version':1,'selection':'six half-years 2021--2023','reports':'2024,2025,2026H1 exposed diagnostics','sources':'UTC calendar known ex ante; completed BTC price, aggressive flow and realized funding','mechanisms':['prequential hour-of-week drift','session-specific flow continuation','session exhaustion reversal','funding settlement crowding','weekend trend','month-end flow'],'calendar_model':'annual trailing3y hour-of-week target means shrunk n/(n+100), mature6h/24h target','thresholds':{'flow':.02,'z':1.5,'trend':.75,'funding':5e-5,'calendar_expected_return':.0005},'rebalance_hours':[6,24],'sizing':['raw','vol20'],'portfolio':'two representatives/family, cross-family pairs and aggregates, net before costs','costs':[0.,.0006,.001],'risk':'net cap1 and five-minute MDD','no_frequency_or_fee_ratio_gate':True,'no_live_changes':True}
def register():
 d={'design':DESIGN,'code_sha256':base.sha(__file__),'base_sha256':base.sha(base.__file__),'market_sha256':base.sha(base.MARKET),'funding_sha256':base.sha(base.FUNDING)};p=OUT/'design.json'
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('Calendar design drift')
 base.write_json(p,d);return d

def calendar_expected(x,data,horizon):
 target=pd.Series(data['open']).shift(-horizon).to_numpy()/data['open']-1;slot=x.index.dayofweek*24+x.index.hour;score=np.full(len(x),np.nan);audit=[]
 for year in range(2021,2027):
  train,test=regime.annual_masks(x.index,horizon,year);train &= np.isfinite(target)
  table=pd.DataFrame({'slot':slot[train],'y':target[train]}).groupby('slot').y.agg(['mean','count']);shrink=table['mean']*table['count']/(table['count']+100)
  score[test]=pd.Series(slot[test]).map(shrink).to_numpy();audit.append({'prediction_year':year,'horizon':horizon,'train_rows':int(train.sum()),'last_train':str(x.index[train][-1])})
 return score,audit

def candidates(x,data):
 signals={};specs={};audit=[];size=np.clip(np.divide(.2,x.vol24.to_numpy()*np.sqrt(8766),out=np.ones(len(x)),where=x.vol24.to_numpy()>0),.1,1)
 def add(name,family,raw,why):
  raw=np.clip(np.nan_to_num(np.asarray(raw,float)),-1,1)
  for side in ['both','long','short']:
   direction=raw if side=='both' else np.maximum(raw,0) if side=='long' else np.minimum(raw,0)
   for sizing in ['raw','vol20']:
    for reb in [6,24]:
     n=f'{name}__{side}__reb{reb}__{sizing}';signals[n]=base.hold_signal(direction*(size if sizing=='vol20' else 1),reb,x.index);specs[n]={'family':family,'side':side,'sizing':sizing,'rebalance_hours':reb,'rationale':why,'model':family=='calendar_model'}
 flow=x.flow6.to_numpy();z=x.z24.to_numpy();mom=x.mom168.to_numpy();fund=x.funding.to_numpy();hour=x.index.hour;dow=x.index.dayofweek;trend=np.sign(mom);strong=np.abs(mom)>.75
 for h in [6,24]:
  score,notes=calendar_expected(x,data,h);audit+=notes;add(f'calendar{h}','calendar_model',np.where(np.abs(score)>.0005,np.sign(score),0),'shrunk trailing hour-of-week expected return')
 for session,(a,b) in {'asia':(0,8),'europe':(8,16),'us':(16,24)}.items():
  active=(hour>=a)&(hour<b)
  add(f'{session}_flow','session_flow',np.where(active&(np.abs(flow)>.02),np.sign(flow),0),'session-local aggressive-flow continuation')
  add(f'{session}_exhaustion','session_exhaustion',np.where(active&(np.abs(z)>1.5)&(np.sign(z)*flow<-.02),-np.sign(z),0),'session displacement contradicted by aggressive flow')
 settle=np.isin(hour,[0,8,16]);post=np.isin(hour,[1,9,17])
 add('funding_clock_fade','funding_clock',np.where(settle&(np.abs(fund)>5e-5),-np.sign(fund),0),'fade extreme crowding at known settlement clock')
 add('post_funding_flow','post_funding',np.where(post&(np.abs(flow)>.02),np.sign(flow),0),'post-settlement aggressive-flow continuation')
 add('weekend_trend','weekend',np.where((dow>=5)&strong,trend,0),'weekend continuation when established trend remains strong')
 add('weekend_reversion','weekend',np.where((dow>=5)&(~strong)&(np.abs(z)>1.5),-np.sign(z),0),'weekend mean reversion only outside trend')
 add('monthend_flow','monthend',np.where((x.index.day>=28)&(np.abs(flow)>.02),np.sign(flow),0),'month-end institutional flow persistence')
 return signals,specs,audit

def run():
 reg=json.loads((OUT/'design.json').read_text());
 if reg!=register():raise RuntimeError('Registration changed')
 m,f=base.load_sources();x=base.features(m,f);x,data,receipt=base.execution_blocks(m,f,x);signals,specs,audit=candidates(x,data);names=list(signals);p=np.column_stack(list(signals.values()));scores,st,halves=regime.rank(data,p);mask=base.window_mask(data,'2021-01-01','2024-01-01')
 reps=[];counts={};seen=set()
 for i in np.argsort(-scores,kind='stable'):
  fam=specs[names[i]]['family'];h=hashlib.sha256(p[mask,i].tobytes()).hexdigest()
  if counts.get(fam,0)>=2 or h in seen:continue
  counts[fam]=counts.get(fam,0)+1;seen.add(h);reps.append(int(i))
 for ai,i in enumerate(reps):
  for j in reps[ai+1:]:
   if specs[names[i]]['family']==specs[names[j]]['family']:continue
   for w in [.25,.5,.75]:
    n=f'mix_{i}_{j}_{w}';signals[n]=w*p[:,i]+(1-w)*p[:,j];specs[n]={'family':'portfolio','components':{names[i]:w,names[j]:1-w},'model':specs[names[i]]['model'] or specs[names[j]]['model']}
 rs=base.simulate(base.subset(data,mask),p[mask][:,reps])
 for mode in ['equal','inversevol']:
  w=np.ones(len(reps)) if mode=='equal' else 1/np.maximum(rs['returns'].std(axis=0),1e-6);w/=w.sum();n='aggregate_'+mode;signals[n]=p[:,reps]@w;specs[n]={'family':'portfolio','components':{names[i]:float(a) for i,a in zip(reps,w)},'model':True}
 names=list(signals);p=np.column_stack(list(signals.values()));scores,st,halves=regime.rank(data,p);order=np.argsort(-scores,kind='stable');final=list(map(int,order[:5]));pure=next(int(i) for i in order if not specs[names[i]]['model'])
 if pure not in final:final.append(pure)
 freeze={'selection':'six halves 2021--2023','report_reranking':False,'candidates':len(names),'top':[{'name':names[i],'spec':specs[names[i]],'score':float(scores[i]),'selection':base.stats_row(st,i),'half_sharpes':halves[:,i].tolist()} for i in final]};base.write_json(OUT/'selection_freeze.json',freeze)
 fp=np.column_stack([p[:,final],np.ones(len(x)),np.zeros(len(x))]);fn=[names[i] for i in final]+['control_long','control_cash'];reports={}
 for n,a,b in [('report2024','2024-01-01','2025-01-01'),('report2025','2025-01-01','2026-01-01'),('report2026','2026-01-01','2026-06-01'),('combined','2024-01-01','2026-06-01')]:
  use=base.window_mask(data,a,b);reports[n]={}
  for cost in DESIGN['costs']:
   rr=base.simulate(base.subset(data,use),fp[use],cost=cost,fine=True);reports[n][str(cost)]={name:base.stats_row(rr,k) for k,name in enumerate(fn)}
 result={'registration':reg,'source_receipt':receipt,'calendar_fit_audit':audit,'freeze':freeze,'reports':reports,'inventory':[{'name':names[i],'spec':specs[names[i]],'score':float(scores[i]),'selection':base.stats_row(st,i)} for i in order],'live_enabled':False};base.write_json(OUT/'report.json',result)
 base.write_json(OUT/'research_config.json',{'research_only':True,'live_enabled':False,'winner':freeze['top'][0],'net_exposure_cap':1.,'offset_long_short':True})
 print('candidates',len(names),flush=True)
 for row in freeze['top']:
  n=row['name'];print(n,json.dumps(row['spec']),flush=True)
  for period in reports:print(period,json.dumps(reports[period]['0.0006'][n]),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');q=a.parse_args();register() if q.freeze else run()
