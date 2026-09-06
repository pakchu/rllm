"""Build source-only support clocks for CVRVPR-12."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from training import preregister_cross_venue_relative_volatility_premium_regime_crossing_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

SOURCE=Path('data/options_crowding_deleveraging_relay_sources_v4_2023_2026');CLOCK=Path('data/cross_venue_relative_volatility_premium_regime_crossing_relay_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/cross_venue_relative_volatility_premium_regime_crossing_relay_controls_2023_2026');RESULT=Path('results/cross_venue_relative_volatility_premium_regime_crossing_relay_support_2026-08-08.json')
SPLITS={'train':(pd.Timestamp('2023-07-01T00:00:00Z'),pd.Timestamp('2024-01-01T00:00:00Z')),'test':(pd.Timestamp('2024-01-01T00:00:00Z'),pd.Timestamp('2025-01-01T00:00:00Z')),'eval':(pd.Timestamp('2025-01-01T00:00:00Z'),pd.Timestamp('2026-01-01T00:00:00Z')),'final':(pd.Timestamp('2026-01-01T00:00:00Z'),pd.Timestamp('2026-08-01T00:00:00Z'))}
MIN={'train':8,'test':12,'eval':12,'final':8};CONTROLS=('absolute_dvol_level_crossing','absolute_bvol_level_crossing','one_hour_stale_ratio','no_crossing','direction_flip');ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','relative_log_level','relative_rank','bvol_close','bvol_rank','dvol_close','dvol_rank')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def causal_midrank(values:pd.Series,window:int=720,min_count:int=672)->pd.Series:
 a=values.to_numpy(float);out=np.full(len(a),np.nan)
 for i,x in enumerate(a):
  if not np.isfinite(x):continue
  prior=a[max(0,i-window):i];prior=prior[np.isfinite(prior)]
  if len(prior)>=min_count:out[i]=(np.sum(prior<x)+.5*np.sum(prior==x))/len(prior)
 return pd.Series(out,index=values.index)

def features()->pd.DataFrame:
 b=pd.read_csv(SOURCE/'bvol_hourly.csv.gz',compression='gzip');d=pd.read_csv(SOURCE/'dvol_hourly.csv.gz',compression='gzip')
 bf=pd.DataFrame({'decision_time':pd.to_datetime(b.feature_available_time_utc,utc=True,format='mixed'),'bvol_close':pd.to_numeric(b.close,errors='coerce'),'bvol_valid':b.feature_valid.astype(str).str.lower().eq('true')})
 df=pd.DataFrame({'decision_time':pd.to_datetime(d.close_time,utc=True,format='mixed'),'dvol_close':pd.to_numeric(d.close,errors='coerce')})
 j=bf.merge(df,on='decision_time',validate='one_to_one').sort_values('decision_time').reset_index(drop=True);j['base_valid']=j.bvol_valid&np.isfinite(j[['bvol_close','dvol_close']]).all(axis=1)&j[['bvol_close','dvol_close']].gt(0).all(axis=1)
 j['relative_log_level']=np.log(j.dvol_close/j.bvol_close).where(j.base_valid);j['relative_rank']=causal_midrank(j.relative_log_level);j['bvol_rank']=causal_midrank(np.log(j.bvol_close).where(j.base_valid));j['dvol_rank']=causal_midrank(np.log(j.dvol_close).where(j.base_valid));return j

def clock(j:pd.DataFrame,control:str='primary')->pd.DataFrame:
 valid=j.base_valid&j.relative_rank.notna();rank=j.relative_rank
 if control=='absolute_dvol_level_crossing':rank=j.dvol_rank;valid=j.base_valid&rank.notna();high=rank.ge(.8);low=rank.le(.2);side=np.where(low,1,-1)
 elif control=='absolute_bvol_level_crossing':rank=j.bvol_rank;valid=j.base_valid&rank.notna();high=rank.ge(.8);low=rank.le(.2);side=np.where(low,1,-1)
 else:high=rank.ge(.8);low=rank.le(.2);side=np.where(low,1,-1)
 outer=high|low
 if control=='no_crossing':active=valid&outer
 else:active=valid&outer&~outer.shift(1,fill_value=False)&valid.shift(1,fill_value=False)&j.decision_time.diff().eq(pd.Timedelta(hours=1))
 if control=='one_hour_stale_ratio':active=active.shift(1,fill_value=False)&valid&j.decision_time.diff().eq(pd.Timedelta(hours=1));side=pd.Series(side,index=j.index).shift(1).fillna(0).to_numpy()
 rows=[];next_allowed=None
 for i in j.index[active]:
  decision=pd.Timestamp(j.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  s=int(side[i]);s=-s if control=='direction_flip' else s;next_allowed=exit_;rows.append({'candidate':'CVRVPR-12','control':control,'split':split,'decision_time':decision,'feature_available_time':decision,'entry_time':entry,'exit_time':exit_,'side':s,'relative_log_level':float(j.at[i,'relative_log_level']),'relative_rank':float(j.at[i,'relative_rank']),'bvol_close':float(j.at[i,'bvol_close']),'bvol_rank':float(j.at[i,'bvol_rank']),'dvol_close':float(j.at[i,'dvol_close']),'dvol_rank':float(j.at[i,'dvol_rank'])})
 return pd.DataFrame(rows,columns=COLUMNS)

def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':lo,'shorts':sh,'minority_side_share':min(lo,sh)/len(x),'max_month_share':int(m.max())/len(x)}

def run()->dict:
 j=features();primary=clock(j);controls={n:clock(j,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f'{n}.csv.gz')
 st={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in st.items():checks[f'{n}_minimum_events']=x['events']>=MIN[n];checks[f'{n}_side_balance']=x['minority_side_share']>=.2;checks[f'{n}_month_concentration']=x['max_month_share']<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={'protocol_version':'cvrvpr_12_source_support_v1','policy_id':'CVRVPR-12','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':sha(prereg.DEFAULT_OUTPUT),'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(SOURCE/'manifest.json'),'sha256':sha(SOURCE/'manifest.json')},'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in controls.items()},'support':st,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':ECONOMIC_OUTCOMES_AUTHORIZED,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r
if __name__=='__main__':argparse.ArgumentParser().parse_args();r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
