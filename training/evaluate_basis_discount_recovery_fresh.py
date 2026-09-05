"""One-shot recent DB replay of the fixed discount-recovery basis candidate."""
from __future__ import annotations
import argparse,json
import numpy as np,pandas as pd
from preprocessing.binance_aux_features import normalise_premium_index_frame
from training import evaluate_macro_flow_fixed_fresh as mf
from training import search_meaningful_alpha_combinations as base
from training import search_basis_crowding_alpha_combinations as basis
from training import build_pposm_fresh_forward_signal_inventory_v2 as pposm_db

OUT=base.ROOT/'research/basis_discount_recovery_fresh';START=mf.START;END=mf.END;SEAM=mf.SEAM
DESIGN={'version':1,'candidate':'0.25*premium_flow_divergence__reb24__vol20 + 0.75*discount_recovery__reb24__raw','historical_identity':'mix_55_61_0.25','source':'fixed cache plus authoritative DB market/premium/funding extension','fresh_window':[START,END],'costs':[0.,.0006,.001],'execution':base.DESIGN['execution'],'risk':'net cap1, overlap allowed, positions offset','decision':'one-shot report, no live promotion'}

def register():
 d={'design':DESIGN,'code_sha256':base.sha(__file__),'basis_selection_sha256':base.sha(base.ROOT/'research/basis_crowding_combinations/selection_freeze.json'),'cache_sha256':base.sha(base.MARKET)};p=OUT/'design.json'
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('Design drift')
 base.write_json(p,d);return d

def basis_from_events(x,events,market):
 p=events.sort_values('date').drop_duplicates('date',keep='last').rename(columns={'date':'premium_date'})
 joined=pd.merge_asof(pd.DataFrame({'date':x.index}),p,left_on='date',right_on='premium_date',direction='backward',tolerance=pd.Timedelta('2h'))
 out=pd.DataFrame(index=x.index);level=pd.Series(joined.premium_index.to_numpy(float),index=x.index);out['premium_available']=joined.premium_date.notna().to_numpy(float);out['premium']=level
 for w in [6,24,168]:
  out[f'premium_change{w}']=level.diff(w);mu=level.rolling(w,min_periods=w).mean();sd=level.rolling(w,min_periods=w).std(ddof=0);out[f'premium_z{w}']=(level-mu)/sd.replace(0,np.nan)
 out['premium_change6_z168']=(out.premium_change6-out.premium_change6.rolling(168,min_periods=72).mean())/out.premium_change6.rolling(168,min_periods=72).std(ddof=0).replace(0,np.nan)
 close=market.set_index('date').close.resample('1h',label='right',closed='left').last().reindex(x.index);out['price_return96']=np.log(close).diff(96)
 return out.replace([np.inf,-np.inf],np.nan)

def position(x):
 size=np.clip(np.divide(.2,x.vol24.to_numpy()*np.sqrt(8766),out=np.ones(len(x)),where=x.vol24.to_numpy()>0),.1,1)
 pz=x.premium_z168.to_numpy();pch=x.premium_change6.to_numpy();pchz=x.premium_change6_z168.to_numpy();flow=x.flow6.to_numpy()
 divergence=np.where((np.abs(pchz)>1)&(np.sign(pch)*flow<-.02),np.sign(flow),0)*size
 recovery=np.where((pz<-1.5)&(pch>0)&(flow>.02),1,0)
 return .25*base.hold_signal(divergence,24,x.index)+.75*base.hold_signal(recovery,24,x.index),{'divergence':divergence,'recovery':recovery}

def run():
 reg=json.loads((OUT/'design.json').read_text());
 if reg!=register():raise RuntimeError('Registration changed')
 db,funding,receipt=mf.load_extension();old=pd.read_csv(base.MARKET);old.date=pd.to_datetime(old.date,utc=True).dt.tz_convert(None);market=pposm_db.merge_cache_db_markets(old,db,cutoff=SEAM)
 histfund=pd.read_csv(base.FUNDING);histfund['date']=pd.to_datetime(histfund.funding_time,unit='ms',utc=True).dt.tz_convert(None);fund=pd.concat([histfund[['date','funding_rate','mark_price']][histfund.date<pd.Timestamp(SEAM).tz_localize(None)],funding[funding.date>=pd.Timestamp(SEAM).tz_localize(None)]],ignore_index=True).sort_values('date').drop_duplicates('date',keep='last')
 x=base.features(market,fund);x,data,engine_receipt=base.execution_blocks(market,fund,x)
 oldpremium=normalise_premium_index_frame(pd.read_csv(basis.PREMIUM));recent=db[['date','premium_index']].dropna();events=pd.concat([oldpremium[oldpremium.date<pd.Timestamp(SEAM).tz_localize(None)],recent[recent.date>=pd.Timestamp(SEAM).tz_localize(None)]],ignore_index=True)
 x=pd.concat([x,basis_from_events(x,events,market)],axis=1);p,components=position(x);mask=base.window_mask(data,START,END);reports={}
 for cost in DESIGN['costs']:reports[str(cost)]=base.stats_row(base.simulate(base.subset(data,mask),p[mask],cost=cost,fine=True))
 result={'registration':reg,'source_receipt':receipt,'engine_receipt':engine_receipt,'premium_event_rows':len(events),'premium_available':float(x.loc[(x.index>=pd.Timestamp(START).tz_localize(None)),'premium_available'].mean()),'component_nonzero_decisions':{k:int(np.count_nonzero(v[mask])) for k,v in components.items()},'reports':reports,'live_enabled':False}
 base.write_json(OUT/'report.json',result);print(json.dumps(reports,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
