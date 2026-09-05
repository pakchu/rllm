"""Search funding/premium crowding, unwind and recovery alpha combinations."""
from __future__ import annotations
import argparse, hashlib, json
import numpy as np, pandas as pd
from preprocessing.binance_aux_features import normalise_premium_index_frame
from training import search_meaningful_alpha_combinations as base
from training import search_regime_diverse_alpha_combinations as regime

OUT=base.ROOT/'research/basis_crowding_combinations';PREMIUM=base.DATA/'binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz'
DESIGN={'version':1,'selection':'2021--2023 six-half robust rank','reports':'2024,2025,2026H1 exposed diagnostics',
 'source':'completed Binance premium-index hourly close_time plus realized funding and completed BTC/flow bars',
 'mechanisms':['crowded-long unwind','crowded-short squeeze','basis-price joint reversion','negative-carry trend','premium/flow disagreement','panic continuation','discount recovery'],
 'thresholds':{'premium_z':1.25,'premium_extreme_z':1.5,'funding_extreme':.00005,'flow':.02,'trend':.75,'price_z':1.5,'panic_4d_return':-.07},
 'rebalance_hours':[6,24],'sizing':['raw','vol20'],'ml':'annual trailing3y Ridge/HGB/ExtraTrees using basis features, 24h mature target',
 'portfolio':'two representatives per family, cross-family .25/.5/.75, equal/inversevol aggregates; positions netted',
 'costs':[0.,.0006,.001],'no_frequency_or_fee_ratio_gate':True,'risk':'net exposure cap1 and five-minute conservative MDD','no_live_changes':True}

def register():
 d={'design':DESIGN,'code_sha256':base.sha(__file__),'base_sha256':base.sha(base.__file__),'regime_sha256':base.sha(regime.__file__),'premium_sha256':base.sha(PREMIUM),'market_sha256':base.sha(base.MARKET),'funding_sha256':base.sha(base.FUNDING)};p=OUT/'design.json'
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('Basis design drift')
 base.write_json(p,d);return d

def basis_features(x,market):
 raw=pd.read_csv(PREMIUM);p=normalise_premium_index_frame(raw).rename(columns={'date':'premium_date'})
 joined=pd.merge_asof(pd.DataFrame({'date':x.index}),p,left_on='date',right_on='premium_date',direction='backward',tolerance=pd.Timedelta('2h'))
 out=pd.DataFrame(index=x.index);level=pd.Series(joined.premium_index.to_numpy(float),index=x.index);out['premium_available']=joined.premium_date.notna().to_numpy(float);out['premium']=level
 for w in [6,24,168]:
  out[f'premium_change{w}']=level.diff(w);mean=level.rolling(w,min_periods=w).mean();std=level.rolling(w,min_periods=w).std(ddof=0);out[f'premium_z{w}']=(level-mean)/std.replace(0,np.nan)
 out['premium_change6_z168']=(out.premium_change6-out.premium_change6.rolling(168,min_periods=72).mean())/out.premium_change6.rolling(168,min_periods=72).std(ddof=0).replace(0,np.nan)
 funding=x.funding;out['funding_z168']=(funding-funding.rolling(168,min_periods=72).mean())/funding.rolling(168,min_periods=72).std(ddof=0).replace(0,np.nan)
 close=market.set_index('date').close.resample('1h',label='right',closed='left').last().reindex(x.index);out['price_return96']=np.log(close).diff(96)
 return out.replace([np.inf,-np.inf],np.nan)

def candidates(x,data):
 allsignals,allspecs,notes=regime.candidates(x,data)
 signals={n:v for n,v in allsignals.items() if allspecs[n]['family'].startswith('ml_')};specs={n:allspecs[n] for n in signals}
 size=np.clip(np.divide(.2,x.vol24.to_numpy()*np.sqrt(8766),out=np.ones(len(x)),where=x.vol24.to_numpy()>0),.1,1)
 def add(name,family,raw,why):
  raw=np.clip(np.nan_to_num(np.asarray(raw,float)),-1,1)
  for sizing in ['raw','vol20']:
   for reb in [6,24]:
    n=f'{name}__reb{reb}__{sizing}';signals[n]=base.hold_signal(raw*(size if sizing=='vol20' else 1),reb,x.index);specs[n]={'family':family,'rationale':why,'sizing':sizing,'rebalance_hours':reb,'model':False}
 prem=x.premium.to_numpy();pz=x.premium_z168.to_numpy();pch=x.premium_change6.to_numpy();pchz=x.premium_change6_z168.to_numpy();fund=x.funding.to_numpy();flow=x.flow6.to_numpy();mom=x.mom168.to_numpy();z=x.z24.to_numpy();ret96=x.price_return96.to_numpy()
 add('crowded_long_unwind','unwind',np.where((pz>1.25)&(fund>5e-5)&(flow<-.02),-1,0),'positive basis and funding unwind confirmed by aggressive selling')
 add('crowded_short_squeeze','squeeze',np.where((pz<-1.25)&(fund<0)&(flow>.02),1,0),'discounted futures and negative funding squeeze with buying flow')
 add('basis_price_reversion','basis_reversion',np.where((np.abs(pz)>1.5)&(np.abs(z)>1.5)&(np.sign(pz)==np.sign(z)),-np.sign(z),0),'basis and price jointly displaced from recent means')
 add('discount_trend_long','discount_trend',np.where((mom>.75)&(prem<0)&(fund<=0),1,0),'positive price trend with favorable discounted basis and carry')
 add('premium_flow_divergence','premium_flow',np.where((np.abs(pchz)>1)&(np.sign(pch)*flow<-.02),np.sign(flow),0),'premium impulse contradicted by aggressive futures flow')
 add('panic_continuation_short','panic',np.where((ret96<-.07)&(pz<-1.25),-1,0),'deep four-day selloff with futures discount remains continuation risk')
 add('discount_recovery','recovery',np.where((pz<-1.5)&(pch>0)&(flow>.02),1,0),'extreme discount starts recovering with aligned buying')
 add('premium_extreme_fade','premium_fade',np.where((np.abs(pz)>1.5)&(np.abs(mom)<.75),-np.sign(pz),0),'basis extreme fades only outside strong trend')
 return signals,specs,notes

def run():
 reg=json.loads((OUT/'design.json').read_text());
 if reg!=register():raise RuntimeError('Registration drift')
 m,f=base.load_sources();x=base.features(m,f);x,data,receipt=base.execution_blocks(m,f,x);x=pd.concat([x,basis_features(x,m)],axis=1)
 signals,specs,notes=candidates(x,data);names=list(signals);p=np.column_stack(list(signals.values()));scores,st,halves=regime.rank(data,p);mask=base.window_mask(data,'2021-01-01','2024-01-01')
 seen=set();counts={};reps=[]
 for i in np.argsort(-scores,kind='stable'):
  fam=specs[names[i]]['family'];h=hashlib.sha256(p[mask,i].tobytes()).hexdigest()
  if counts.get(fam,0)>=2 or h in seen:continue
  seen.add(h);counts[fam]=counts.get(fam,0)+1;reps.append(int(i))
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
 for name,a,b in [('report2024','2024-01-01','2025-01-01'),('report2025','2025-01-01','2026-01-01'),('report2026','2026-01-01','2026-06-01'),('combined','2024-01-01','2026-06-01')]:
  use=base.window_mask(data,a,b);reports[name]={}
  for cost in DESIGN['costs']:
   rr=base.simulate(base.subset(data,use),fp[use],cost=cost,fine=True);reports[name][str(cost)]={n:base.stats_row(rr,k) for k,n in enumerate(fn)}
 result={'registration':reg,'source_receipt':receipt,'premium_coverage':{str(y):float(x.loc[x.index.year==y,'premium_available'].mean()) for y in range(2020,2027)},'annual_ml_fits':notes,'freeze':freeze,'reports':reports,'inventory':[{'name':names[i],'spec':specs[names[i]],'score':float(scores[i]),'selection':base.stats_row(st,i)} for i in order],'live_enabled':False};base.write_json(OUT/'report.json',result)
 base.write_json(OUT/'research_config.json',{'research_only':True,'live_enabled':False,'winner':freeze['top'][0],'net_exposure_cap':1.,'offset_long_short':True,'overlap_allowed':True})
 print('candidates',len(names),flush=True)
 for row in freeze['top']:
  n=row['name'];print(n,json.dumps(row['spec']),flush=True)
  for period in reports:print(period,json.dumps(reports[period]['0.0006'][n]),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');a=parser.parse_args();register() if a.freeze else run()
