"""Additional macro/venue-flow hypotheses, using the fixed netting simulator.

Cache macro fields are masked by availability and delayed one full hour. This
is a conservative research proxy, not a raw publication-time/source parity audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from training import search_meaningful_alpha_combinations as base
from training import search_regime_diverse_alpha_combinations as regime

OUT=base.ROOT/'research/macro_flow_combinations'
DESIGN={
 'version':1,'selection':'2021--2023 six-half-year robust rank, inherited unchanged',
 'reports':'2024,2025,2026H1; previously exposed historical diagnostics',
 'macro_source':'existing cache DXY proxy, USDKRW, kimchi premium; availability masked, completed-hour last value delayed extra1h',
 'mechanisms':['dollar headwind/tailwind trend confirmation','regional demand premium plus flow','regional speculative-premium exhaustion','currency pressure with price flow','regime-switch ensemble confirmed by cross-market direction'],
 'macro_horizons_hours':[6,24], 'rebalance_hours':[6,24], 'side_modes':['both','long','short'],
 'sizing':'20% vol target capped1x, inherited previous24hvol',
 'ML':'Ridge/HGB/ExtraTrees annual rolling3y models on baseline plus cross-market features; maturity-purged 24h targets',
 'portfolio':'best2 distinct selection paths per family, all cross-family pairs at .25/.5/.75, equal and inversevol aggregate',
 'risk':'same-symbol exposures offset before fees/risk; costs0/6/10bp plus funding, no fee-ratio/frequency gates',
 'report_finalists':'top5 plus best pure-rule fixed before report',
 'limitations':['macro cache lacks original publication timestamp receipts','macro proxy missingness/coverage shift','funding missing marks use settlement open proxy','no clean OOS or live authorization'],
}


def register():
 d={'design':DESIGN,'code_sha256':base.sha(__file__),'engine_sha256':base.sha(base.__file__),'regime_sha256':base.sha(regime.__file__), 'market_sha256':base.sha(base.MARKET),'funding_sha256':base.sha(base.FUNDING)}
 path=OUT/'design.json'
 if path.exists() and json.loads(path.read_text())!=d:raise RuntimeError('Frozen macro study drift')
 base.write_json(path,d);return d


def macro_features(raw, index):
 raw=raw.copy();raw['date']=pd.to_datetime(raw.date,utc=True).dt.tz_convert(None)
 h=raw.set_index('date').resample('1h',label='right',closed='left').last().shift(1)
 out=pd.DataFrame(index=h.index)
 for field,flag in [('dxy','dxy_available'),('usdkrw','usdkrw_available'),('kimchi_premium','kimchi_available')]:
  valid=h[flag]>.5
  values=h[field].where(valid)
  if field!='kimchi_premium':values=np.log(values.where(values>0))
  out[field+'_valid']=valid.astype(float)
  for hours in [6,24]:out[f'{field}_change{hours}']=values.diff(hours)
  out[field+'_z']=(values-values.rolling(168,min_periods=72).mean())/values.rolling(168,min_periods=72).std(ddof=0).replace(0,np.nan)
 return out.reindex(index).replace([np.inf,-np.inf],np.nan)


def candidates(x,data):
 signals,specs,notes=regime.candidates(x,data)
 sizing=np.clip(np.divide(.2,x.vol24.to_numpy()*np.sqrt(8766),out=np.ones(len(x)),where=x.vol24.to_numpy()>0),.1,1)
 def add(name,family,raw,why):
  raw=np.clip(np.nan_to_num(np.asarray(raw,float)),-1,1)*sizing
  for side in ['both','long','short']:
   direction=raw if side=='both' else np.maximum(raw,0) if side=='long' else np.minimum(raw,0)
   for reb in [6,24]:
    name2=f'{name}__{side}__reb{reb}';signals[name2]=base.hold_signal(direction,reb,x.index)
    specs[name2]={'family':family,'side':side,'rebalance_hours':reb,'sizing':'vol20','rationale':why,'model':False}
 trend=np.sign(x.mom168);strong=np.abs(x.mom168)>.75;flow=x.flow6;z=x.z24
 for hours in [6,24]:
  dollar=x[f'dxy_change{hours}'];currency=x[f'usdkrw_change{hours}'];premium=x[f'kimchi_premium_change{hours}']
  add(f'dollar_trend{hours}','dollar',np.where(strong&(trend*dollar<0),trend,0),'BTC trend aligned with falling/rising dollar headwind')
  add(f'dollar_flow{hours}','dollar_flow',np.where((np.abs(flow)>.02)&(np.sign(flow)*dollar<0),np.sign(flow),0),'aggressive BTC flow aligned with inverse-dollar direction')
  add(f'regional_flow{hours}','regional',np.where((np.abs(flow)>.02)&(np.sign(flow)*premium>0),np.sign(flow),0),'regional demand premium change agrees with futures flow')
  add(f'regional_trend{hours}','regional_trend',np.where(strong&(trend*premium>0),trend,0),'regional demand supports price trend')
  add(f'fx_confirm{hours}','currency',np.where(strong&(trend*currency<0),trend,0),'currency-risk pressure agrees with risk asset direction')
  add(f'exhaustion{hours}','regional_reversal',np.where((np.abs(z)>1.5)&(np.sign(z)*premium<0)&(np.sign(z)*flow<0),-np.sign(z),0),'price extreme contradicted by both regional demand and aggressive flow')
  add(f'cross_consensus{hours}','macro_consensus',np.where((np.sign(flow)*premium>0)&(np.sign(flow)*dollar<0)&(np.abs(flow)>.02),np.sign(flow),0),'independent currency and venue flow confirmation')
 return signals,specs,notes


def run():
 reg=json.loads((OUT/'design.json').read_text())
 if reg!=register():raise RuntimeError('Registration drift')
 m,f=base.load_sources();x=base.features(m,f);x,data,receipt=base.execution_blocks(m,f,x)
 cols=['date','dxy','usdkrw','kimchi_premium','dxy_available','usdkrw_available','kimchi_available']
 macro=macro_features(pd.read_csv(base.MARKET,usecols=cols),x.index);x=pd.concat([x,macro],axis=1)
 signals,specs,notes=candidates(x,data);names=list(signals);p=np.column_stack(list(signals.values()))
 scores,st,halves=regime.rank(data,p);select=base.window_mask(data,'2021-01-01','2024-01-01')
 seen=set();counts={};chosen=[]
 for i in np.argsort(-scores,kind='stable'):
  fam=specs[names[i]]['family'];h=hashlib.sha256(p[select,i].tobytes()).hexdigest()
  if counts.get(fam,0)>=2 or h in seen:continue
  seen.add(h);counts[fam]=counts.get(fam,0)+1;chosen.append(int(i))
 for ai,i in enumerate(chosen):
  for j in chosen[ai+1:]:
   if specs[names[i]]['family']==specs[names[j]]['family']:continue
   for w in [.25,.5,.75]:
    name=f'mix_{i}_{j}_{w}';signals[name]=w*p[:,i]+(1-w)*p[:,j]
    specs[name]={'family':'portfolio','components':{names[i]:w,names[j]:1-w},'model':specs[names[i]]['model'] or specs[names[j]]['model']}
 metrics=base.simulate(base.subset(data,select),p[select][:,chosen])
 for mode in ['equal','inversevol']:
  weights=np.ones(len(chosen)) if mode=='equal' else 1/np.maximum(metrics['returns'].std(axis=0),1e-6);weights/=weights.sum()
  name='aggregate_'+mode;signals[name]=p[:,chosen]@weights;specs[name]={'family':'portfolio','components':{names[i]:float(w) for i,w in zip(chosen,weights)},'model':True}
 names=list(signals);p=np.column_stack(list(signals.values()));scores,st,halves=regime.rank(data,p)
 order=np.argsort(-scores,kind='stable');final=list(map(int,order[:5]));pure=next(int(i) for i in order if not specs[names[i]]['model'])
 if pure not in final:final.append(pure)
 freeze={'selection':'six halves 2021--2023','report_reranking':False,'candidates':len(names),'top':[{'name':names[i],'spec':specs[names[i]],'score':float(scores[i]),'selection':base.stats_row(st,i),'half_sharpes':halves[:,i].tolist()} for i in final]}
 base.write_json(OUT/'selection_freeze.json',freeze)
 fp=np.column_stack([p[:,final],np.ones(len(x)),np.zeros(len(x))]);fn=[names[i] for i in final]+['control_long','control_cash'];reports={}
 for name,a,b in [('report2024','2024-01-01','2025-01-01'),('report2025','2025-01-01','2026-01-01'),('report2026','2026-01-01','2026-06-01'),('combined','2024-01-01','2026-06-01')]:
  mask=base.window_mask(data,a,b);reports[name]={}
  for cost in [0.,.0006,.001]:
   mm=base.simulate(base.subset(data,mask),fp[mask],cost=cost,fine=True);reports[name][str(cost)]={n:base.stats_row(mm,j) for j,n in enumerate(fn)}
 result={'registration':reg,'source_receipt':receipt,'annual_ml_fits':notes,'macro_coverage':{str(year):macro.loc[x.index.year==year,[c for c in macro if c.endswith('_valid')]].mean().to_dict() for year in range(2020,2027)},'freeze':freeze,'reports':reports,'inventory':[{'name':names[i],'spec':specs[names[i]],'score':float(scores[i]),'selection':base.stats_row(st,i)} for i in order],'live_enabled':False}
 base.write_json(OUT/'report.json',result)
 base.write_json(OUT/'research_config.json',{'live_enabled':False,'research_only':True,'winner':freeze['top'][0],'net_risk_limit':1.,'overlap_allowed':True,'long_short_offset':True,'fee_ratio_gate':False,'frequency_gate':False})
 print('candidates',len(names),flush=True)
 for row in freeze['top']:
  n=row['name'];print(n,json.dumps(row['spec']),flush=True)
  for period in reports:print(period,json.dumps(reports[period]['0.0006'][n]),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
 if args.freeze:register()
 else:run()
