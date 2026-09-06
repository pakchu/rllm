"""Build outcome-blind source support for RIVSCR-6."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_realized_over_implied_volatility_shock_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_cross_venue_relative_volatility_premium_regime_crossing_relay_support import causal_midrank
VOL=Path('data/options_crowding_deleveraging_relay_sources_v4_2023_2026');PRICE=Path('data/options_oi_chase_exhaustion_sources_2023_2026');CLOCK=Path('data/realized_over_implied_volatility_shock_continuation_relay_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/realized_over_implied_volatility_shock_continuation_relay_controls_2023_2026');RESULT=Path('results/realized_over_implied_volatility_shock_continuation_relay_support_2026-08-08.json')
SPLITS={'train':(pd.Timestamp('2023-07-01T00:00:00Z'),pd.Timestamp('2024-01-01T00:00:00Z')),'test':(pd.Timestamp('2024-01-01T00:00:00Z'),pd.Timestamp('2025-01-01T00:00:00Z')),'eval':(pd.Timestamp('2025-01-01T00:00:00Z'),pd.Timestamp('2026-01-01T00:00:00Z')),'final':(pd.Timestamp('2026-01-01T00:00:00Z'),pd.Timestamp('2026-08-01T00:00:00Z'))};MIN={'train':8,'test':12,'eval':12,'final':8};CONTROLS=('absolute_realized_variation','absolute_implied_level','no_final_hour_confirmation','one_block_stale_ratio','direction_flip');ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','realized_variation','implied_level','ratio_log','ratio_rank','realized_rank','implied_rank','block_return','final_hour_return')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 b=pd.read_csv(VOL/'bvol_hourly.csv.gz',compression='gzip');d=pd.read_csv(VOL/'dvol_hourly.csv.gz',compression='gzip');p=pd.read_csv(PRICE/'btc_completed_hour.csv.gz',compression='gzip')
 bf=pd.DataFrame({'decision_time':pd.to_datetime(b.feature_available_time_utc,utc=True,format='mixed'),'bvol_close':pd.to_numeric(b.close,errors='coerce'),'bvalid':b.feature_valid.astype(str).str.lower().eq('true')});df=pd.DataFrame({'decision_time':pd.to_datetime(d.close_time,utc=True,format='mixed'),'dvol_close':pd.to_numeric(d.close,errors='coerce')});pf=pd.DataFrame({'decision_time':pd.to_datetime(p.decision_time,utc=True,format='mixed'),'open':pd.to_numeric(p.open,errors='coerce'),'close':pd.to_numeric(p.close,errors='coerce'),'pvalid':p.source_valid.astype(str).str.lower().eq('true')})
 j=bf.merge(df,on='decision_time',validate='one_to_one').merge(pf,on='decision_time',validate='one_to_one').sort_values('decision_time').reset_index(drop=True);j['valid']=j.bvalid&j.pvalid&np.isfinite(j[['bvol_close','dvol_close','open','close']]).all(axis=1)&j[['bvol_close','dvol_close','open','close']].gt(0).all(axis=1);j['logret']=np.log(j.close/j.open)
 rows=[]
 for i in j.index[(j.decision_time.dt.hour%8).eq(0)]:
  if i<7:continue
  w=j.loc[i-7:i]
  if len(w)!=8 or not w.valid.all() or not w.decision_time.diff().dropna().eq(pd.Timedelta(hours=1)).all():continue
  rv=float(np.sqrt(np.square(w.logret).sum()));iv=float(np.sqrt((j.at[i,'bvol_close']/100)*(j.at[i,'dvol_close']/100)))
  if rv<=0 or iv<=0:continue
  rows.append({'decision_time':j.at[i,'decision_time'],'realized_variation':rv,'implied_level':iv,'ratio_log':float(np.log(rv/iv)),'block_return':float(j.at[i,'close']/w.iloc[0].open-1),'final_hour_return':float(j.at[i,'close']/j.at[i,'open']-1)})
 x=pd.DataFrame(rows);x['ratio_rank']=causal_midrank(x.ratio_log,270,252);x['realized_rank']=causal_midrank(np.log(x.realized_variation),270,252);x['implied_rank']=causal_midrank(np.log(x.implied_level),270,252);return x
def clock(x:pd.DataFrame,control:str='primary')->pd.DataFrame:
 same=x.block_return.ne(0)&x.final_hour_return.ne(0)&np.sign(x.block_return).eq(np.sign(x.final_hour_return));rank=x.ratio_rank
 if control=='absolute_realized_variation':active=x.realized_rank.ge(.75)&same
 elif control=='absolute_implied_level':active=x.implied_rank.le(.25)&same
 elif control=='no_final_hour_confirmation':active=rank.ge(.75)&x.block_return.ne(0)
 else:active=rank.ge(.75)&same
 side=np.sign(x.block_return).astype(int)
 if control=='one_block_stale_ratio':active=active.shift(1,fill_value=False)&same;side=side.shift(1).fillna(0).astype(int)
 rows=[];next_allowed=None
 for i in x.index[active]:
  decision=pd.Timestamp(x.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None or side.at[i]==0:continue
  s=int(side.at[i]);s=-s if control=='direction_flip' else s;next_allowed=exit_;rows.append({'candidate':'RIVSCR-6','control':control,'split':split,'decision_time':decision,'feature_available_time':decision,'entry_time':entry,'exit_time':exit_,'side':s,**{k:float(x.at[i,k]) for k in ['realized_variation','implied_level','ratio_log','ratio_rank','realized_rank','implied_rank','block_return','final_hour_return']}})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c,n):
 z=c[c.split.eq(n)]
 if z.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 lo=int(z.side.eq(1).sum());sh=int(z.side.eq(-1).sum());m=z.entry_time.dt.strftime('%Y-%m').value_counts();return {'events':len(z),'longs':lo,'shorts':sh,'minority_side_share':min(lo,sh)/len(z),'max_month_share':int(m.max())/len(z)}
def run():
 x=features();primary=clock(x);controls={n:clock(x,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,z in controls.items():_write_gzip_csv(z,CONTROL_DIR/f'{n}.csv.gz')
 st={n:stats(primary,n) for n in SPLITS};checks={}
 for n,z in st.items():checks[f'{n}_minimum_events']=z['events']>=MIN[n];checks[f'{n}_side_balance']=z['minority_side_share']>=.2;checks[f'{n}_month_concentration']=z['max_month_share']<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={'protocol_version':'rivscr_6_source_support_v1','policy_id':'RIVSCR-6','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':sha(prereg.DEFAULT_OUTPUT),'manifest_hash':reg['manifest_hash']},'source_manifests':{'volatility':{'path':str(VOL/'manifest.json'),'sha256':sha(VOL/'manifest.json')},'completed_hour_price':{'path':str(PRICE/'manifest.json'),'sha256':sha(PRICE/'manifest.json')}},'completed_preentry_feature_price_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(z)} for n,z in controls.items()},'support':st,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':ECONOMIC_OUTCOMES_AUTHORIZED,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r
if __name__=='__main__':argparse.ArgumentParser().parse_args();r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
