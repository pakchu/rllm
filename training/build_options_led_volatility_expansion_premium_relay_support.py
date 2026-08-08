"""Build outcome-blind OVEPR-24 source clocks and named-family novelty evidence."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import build_options_perpetual_demand_relay_support as opdr
from training import freeze_options_perpetual_demand_relay_sources as frozen
from training import preregister_options_led_volatility_expansion_premium_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

CANDIDATE='OVEPR-24'
PREREG=Path(prereg.DEFAULT_OUTPUT)
DEFAULT_CLOCK=Path('data/options_led_volatility_expansion_premium_relay_clocks_2023_2026.csv.gz')
DEFAULT_CONTROLS=Path('data/options_led_volatility_expansion_premium_relay_controls_2023_2026')
DEFAULT_RESULT=Path('results/options_led_volatility_expansion_premium_relay_support_2026-08-08.json')
SPLITS={
 'train':(pd.Timestamp('2023-07-01',tz='UTC'),pd.Timestamp('2024-01-01',tz='UTC')),
 'test':(pd.Timestamp('2024-01-01',tz='UTC'),pd.Timestamp('2025-01-01',tz='UTC')),
 'eval':(pd.Timestamp('2025-01-01',tz='UTC'),pd.Timestamp('2026-01-01',tz='UTC')),
 'final':(pd.Timestamp('2026-01-01',tz='UTC'),pd.Timestamp('2026-07-01',tz='UTC')),
}
COMPARATORS={
 'OPDR':('data/options_perpetual_demand_relay_clocks_2023_2026.csv.gz','entry_time','exit_time'),
 'CVVH':('results/cross_venue_volatility_shape_handoff_source_support_2026-07-30/primary.csv.gz','entry_time_utc','exit_time_utc'),
 'PSR':('data/premium_snapback_recenter_clocks_2020_2026.csv.gz','entry_time','planned_exit_time'),
 'PCBR':('data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz','entry_time','exit_time'),
 'CMSR':('data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz','entry_time','exit_time'),
}
CONTROLS=('no_deribit_lead','deribit_fall_mirror','no_premium_efficiency','direction_flip','extra_latency_1h','deterministic_random_side')
CLOCK_COLS=['candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','bvol_body','dvol_body','premium_move_bp','premium_efficiency','prior_efficiency_median']

def sha(path: str|Path)->str:
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load_joint()->pd.DataFrame:
 cfg=opdr.Config(); premium=opdr.load_premium(cfg)
 ps=premium[(premium.date>=pd.Timestamp('2023-06-20',tz='UTC'))&(premium.date<pd.Timestamp('2026-07-01',tz='UTC'))]
 ph=opdr.aggregate_premium_hourly(ps)
 sc=frozen.Config(preregistration=cfg.preregistration,bvol=cfg.bvol,bvol_manifest=cfg.bvol_manifest,dvol=cfg.dvol,dvol_summary=cfg.dvol_summary,output=cfg.source_freeze)
 bv,_=frozen.load_bvol(sc); dv,_=frozen.load_dvol(sc)
 b=pd.DataFrame({'signal_time':pd.to_datetime(bv.feature_available_time_utc,utc=True),'bvol_valid':bv.feature_valid.astype(bool),'bvol_open':pd.to_numeric(bv.open),'bvol_close':pd.to_numeric(bv.close)})
 d=pd.DataFrame({'signal_time':pd.to_datetime(dv.close_time,utc=True),'dvol_open':pd.to_numeric(dv.open),'dvol_close':pd.to_numeric(dv.close)})
 j=ph.merge(b,on='signal_time',validate='one_to_one').merge(d,on='signal_time',validate='one_to_one').sort_values('signal_time').reset_index(drop=True)
 vals=['bvol_open','bvol_close','dvol_open','dvol_close']
 j['joint_valid']=j.premium_valid.astype(bool)&j.bvol_valid&np.isfinite(j[vals]).all(axis=1)&j[vals].gt(0).all(axis=1)
 j['bvol_body']=(j.bvol_close-j.bvol_open)/j.bvol_open; j['dvol_body']=(j.dvol_close-j.dvol_open)/j.dvol_open
 valid=j.joint_valid.astype(bool)
 j['prior_efficiency_median']=j.premium_efficiency.where(valid).shift(1).rolling(720,min_periods=672).median()
 return j

def build_clock(j:pd.DataFrame, control:str='primary')->pd.DataFrame:
 valid=j.joint_valid.astype(bool); b=j.bvol_body; d=j.dvol_body; eff=j.premium_efficiency; move=j.premium_move_bp; med=j.prior_efficiency_median
 if control=='no_deribit_lead': vol=b.gt(0)&d.gt(0)
 elif control=='deribit_fall_mirror': vol=b.lt(0)&d.lt(0)&d.abs().gt(b.abs())
 else: vol=b.gt(0)&d.gt(b)
 active=valid&vol&move.ne(0)
 if control!='no_premium_efficiency': active&=med.notna()&eff.ge(med)
 onset=active&~active.shift(1,fill_value=False)&valid.shift(1,fill_value=False)&j.signal_time.diff().eq(pd.Timedelta(hours=1))
 rows=[]; next_allowed=None
 for i in j.index[onset]:
  decision=pd.Timestamp(j.at[i,'signal_time']); entry=decision+pd.Timedelta(minutes=5); available=pd.Timestamp(j.at[i,'feature_available_time'])
  if available>=entry: continue
  if control=='extra_latency_1h': entry+=pd.Timedelta(hours=1)
  exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed: continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None: continue
  next_allowed=exit_; side=int(np.sign(move.at[i]))
  if control=='direction_flip':side*=-1
  elif control=='deterministic_random_side':side=1 if hashlib.sha256(f'{CANDIDATE}|{decision.isoformat()}'.encode()).digest()[0]%2==0 else -1
  rows.append([CANDIDATE,control,split,decision,available,entry,exit_,side,float(b.at[i]),float(d.at[i]),float(move.at[i]),float(eff.at[i]),float(med.at[i]) if pd.notna(med.at[i]) else None])
 out=pd.DataFrame(rows,columns=CLOCK_COLS)
 return out

def stats(x:pd.DataFrame)->dict[str,Any]:
 if x.empty:return {'events':0,'long':0,'short':0,'long_share':0.0,'short_share':0.0,'max_month_share':0.0,'month_counts':{}}
 m=x.entry_time.dt.strftime('%Y-%m').value_counts().sort_index()
 n=len(x); lo=int((x.side==1).sum()); sh=int((x.side==-1).sum())
 return {'events':n,'long':lo,'short':sh,'long_share':lo/n,'short_share':sh/n,'max_month_share':int(m.max())/n,'month_counts':{str(k):int(v) for k,v in m.items()}}
def greedy(a:list[pd.Timestamp],b:list[pd.Timestamp])->int:
 used=set();n=0
 for x in sorted(a):
  options=[(abs(x-y),y,i) for i,y in enumerate(sorted(b)) if i not in used and abs(x-y)<=pd.Timedelta(hours=6)]
  if options:
   _,_,i=min(options);used.add(i);n+=1
 return n
def bars(df:pd.DataFrame,a:str,b:str,start:pd.Timestamp,end:pd.Timestamp)->set[pd.Timestamp]:
 out=set()
 for x,y in zip(pd.to_datetime(df[a],utc=True),pd.to_datetime(df[b],utc=True)):
  if x>=start and y<=end:out.update(pd.date_range(x,y,freq='5min',inclusive='left'))
 return out
def novelty(primary:pd.DataFrame)->dict[str,Any]:
 out={}
 for name,(path,ec,xc) in COMPARATORS.items():
  d=pd.read_csv(path,compression='gzip'); pe=pd.to_datetime(primary.entry_time,utc=True); de=pd.to_datetime(d[ec],utc=True); dx=pd.to_datetime(d[xc],utc=True)
  start=max(pe.min(),de.min()); end=min(pd.to_datetime(primary.exit_time,utc=True).max(),dx.max())
  a=sorted(set(pe[(pe>=start)&(pd.to_datetime(primary.exit_time,utc=True)<=end)])); bb=sorted(set(de[(de>=start)&(dx<=end)])); inter=len(set(a)&set(bb)); uni=len(set(a)|set(bb)); mat=greedy(a,bb)
  ab=bars(primary,'entry_time','exit_time',start,end); cb=bars(d,ec,xc,start,end)
  out[name]={'candidate_rows':len(a),'comparator_rows':len(bb),'exact_entry_jaccard':inter/uni if uni else 0.0,'one_to_one_6h_max_matched_share':mat/min(len(a),len(bb)) if min(len(a),len(bb)) else 0.0,'occupied_5m_bar_jaccard':len(ab&cb)/len(ab|cb) if ab|cb else 0.0}
 return out

def run()->dict[str,Any]:
 reg=json.loads(PREREG.read_text()); prereg.validate_manifest(reg); j=load_joint(); primary=build_clock(j); controls={n:(build_clock(j,n) if n in {'no_deribit_lead','deribit_fall_mirror','no_premium_efficiency'} else build_clock(j,'primary').assign(control=n)) for n in CONTROLS}
 # rebuild transformed parent clocks correctly
 base=primary.copy()
 controls['direction_flip']=base.assign(control='direction_flip',side=-base.side)
 controls['extra_latency_1h']=base.assign(control='extra_latency_1h',entry_time=base.entry_time+pd.Timedelta(hours=1),exit_time=base.exit_time+pd.Timedelta(hours=1))
 rnd=[1 if hashlib.sha256(f'{CANDIDATE}|{t.isoformat()}'.encode()).digest()[0]%2==0 else -1 for t in base.decision_time]
 controls['deterministic_random_side']=base.assign(control='deterministic_random_side',side=rnd)
 _write_gzip_csv(primary,DEFAULT_CLOCK); DEFAULT_CONTROLS.mkdir(parents=True,exist_ok=True)
 for n,x in controls.items():_write_gzip_csv(x,DEFAULT_CONTROLS/f'{n}.csv.gz')
 support={n:stats(primary[primary.split==n]) for n in SPLITS}; gates=reg['support_gates']; checks={}
 for n,s in support.items():
  checks[f'{n}_events']=s['events']>=gates['minimum_events'][n];checks[f'{n}_side_balance']=min(s['long_share'],s['short_share'])>=.3;checks[f'{n}_month_concentration']=s['max_month_share']<=dict(train=.25,test=.2,eval=.35,final=.4)[n]
 nov=novelty(primary); nchecks={}
 for n,m in nov.items():
  nchecks[f'{n}_exact']=m['exact_entry_jaccard']<=.1;nchecks[f'{n}_near6h']=m['one_to_one_6h_max_matched_share']<=.45;nchecks[f'{n}_occupied']=m['occupied_5m_bar_jaccard']<=.3
 core={'protocol_version':'ovepr_24_source_support_v1','policy_id':CANDIDATE,'preregistration':{'path':str(PREREG),'sha256':sha(PREREG),'manifest_hash':reg['manifest_hash']},'outcomes_opened':False,'outcome_sources_opened':[],'btc_execution_rows_opened':0,'funding_rows_opened':0,'gross9_rows_opened':0,'clock':{'path':str(DEFAULT_CLOCK),'sha256':sha(DEFAULT_CLOCK),'rows':len(primary)},'controls':{n:{'path':str(DEFAULT_CONTROLS/f'{n}.csv.gz'),'sha256':sha(DEFAULT_CONTROLS/f'{n}.csv.gz'),'rows':len(x)} for n,x in controls.items()},'support':support,'support_checks':checks,'support_passed':all(checks.values()),'named_family_novelty':nov,'named_family_novelty_checks':nchecks,'named_family_novelty_passed':all(nchecks.values()),'gross9_novelty_status':'pending','advance_to_economic_outcomes':False,'failure_action':None if all(checks.values()) and all(nchecks.values()) else 'reject before outcomes'}
 result={**core,'manifest_hash':chash(core)};DEFAULT_RESULT.parent.mkdir(parents=True,exist_ok=True);DEFAULT_RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.parse_args();r=run();print(json.dumps({'output':str(DEFAULT_RESULT),'support_passed':r['support_passed'],'named_family_novelty_passed':r['named_family_novelty_passed']}))
