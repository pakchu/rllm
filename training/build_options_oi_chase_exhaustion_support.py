"""Build source-support clocks for OICER-12 without post-entry returns."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_options_oi_chase_exhaustion_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

SOURCE_DIR=Path('data/options_crowding_deleveraging_relay_sources_v4_2023_2026');PRICE_DIR=Path('data/options_oi_chase_exhaustion_sources_2023_2026');PREREG=prereg.DEFAULT_OUTPUT
CLOCK=Path('data/options_oi_chase_exhaustion_reversal_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/options_oi_chase_exhaustion_reversal_controls_2023_2026');RESULT=Path('results/options_oi_chase_exhaustion_reversal_support_2026-08-08.json')
SPLITS=base.SPLITS;MIN={'train':16,'test':24,'eval':24,'final':16};CONTROLS=('no_deribit_lead','no_oi_tail','no_return_tail','no_funding_concurrence','direction_flip')
ECONOMIC_OUTCOMES_AUTHORIZED=False
CLOCK_COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','bvol_body','dvol_body','oi_change','prior_oi_q75','hour_return','prior_abs_return_q75','funding_rate')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 b,d,o,f=base.load_sources(SOURCE_DIR);j=base.joined_features(b,d,o,f);p=pd.read_csv(PRICE_DIR/'btc_completed_hour.csv.gz',compression='gzip');p['decision_time']=pd.to_datetime(p.decision_time,utc=True,format='mixed');p['open']=pd.to_numeric(p.open,errors='coerce');p['close']=pd.to_numeric(p.close,errors='coerce');p['price_valid']=p.source_valid.astype(str).str.lower().eq('true');p['hour_return']=p.close/p.open-1;p['return_tail']=p.hour_return.abs().where(p.price_valid).shift(1).rolling(720,min_periods=672).quantile(.75);j=j.merge(p[['decision_time','price_valid','hour_return','return_tail']],on='decision_time',validate='one_to_one');j['base_valid']&=j.price_valid&np.isfinite(j[['hour_return']]).all(axis=1)&j.hour_return.ne(0);return j
def clock(j:pd.DataFrame,control:str='primary')->pd.DataFrame:
 b=j.bvol_body;d=j.dvol_body;vol=b.gt(0)&d.gt(0) if control=='no_deribit_lead' else b.gt(0)&d.gt(b);oi=j.oi_change.gt(0)
 if control!='no_oi_tail':oi&=j.oi_tail.notna()&j.oi_change.ge(j.oi_tail)
 ret=j.hour_return.ne(0)
 if control!='no_return_tail':ret&=j.return_tail.notna()&j.hour_return.abs().ge(j.return_tail)
 concurrence=j.funding_rate.ne(0)
 if control!='no_funding_concurrence':concurrence&=np.sign(j.funding_rate).eq(np.sign(j.hour_return))
 active=j.base_valid&vol&oi&ret&concurrence;onset=active&~active.shift(1,fill_value=False)&j.base_valid.shift(1,fill_value=False)&j.decision_time.diff().eq(pd.Timedelta(hours=1));rows=[];next_allowed=None
 for i in j.index[onset]:
  decision=pd.Timestamp(j.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=-int(np.sign(j.at[i,'hour_return']));side=-side if control=='direction_flip' else side;next_allowed=exit_;rows.append({'candidate':'OICER-12','control':control,'split':split,'decision_time':decision,'feature_available_time':decision,'entry_time':entry,'exit_time':exit_,'side':side,'bvol_body':float(b.at[i]),'dvol_body':float(d.at[i]),'oi_change':float(j.at[i,'oi_change']),'prior_oi_q75':float(j.at[i,'oi_tail']) if pd.notna(j.at[i,'oi_tail']) else None,'hour_return':float(j.at[i,'hour_return']),'prior_abs_return_q75':float(j.at[i,'return_tail']) if pd.notna(j.at[i,'return_tail']) else None,'funding_rate':float(j.at[i,'funding_rate'])})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.0,'max_month_share':0.0}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':lo,'shorts':sh,'minority_side_share':min(lo,sh)/len(x),'max_month_share':int(m.max())/len(x)}
def run()->dict:
 j=features();primary=clock(j);controls={n:clock(j,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f'{n}.csv.gz')
 st={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in st.items():checks[f'{n}_minimum_events']=x['events']>=MIN[n];checks[f'{n}_side_balance']=x['minority_side_share']>=.2;checks[f'{n}_month_concentration']=x['max_month_share']<=.45
 pre=json.loads(PREREG.read_text());sm=PRICE_DIR/'manifest.json';core={'protocol_version':'oicer_12_source_support_v1','policy_id':'OICER-12','preregistration':{'path':str(PREREG),'sha256':sha(PREREG),'manifest_hash':pre['manifest_hash']},'source_manifest':{'path':str(sm),'sha256':sha(sm)},'completed_preentry_feature_price_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in controls.items()},'support':st,'support_checks':checks,'support_passed':all(checks.values()),'advance_to_gross9_novelty':all(checks.values()),'advance_to_economic_outcomes':ECONOMIC_OUTCOMES_AUTHORIZED,'decision':'pass_to_novelty' if all(checks.values()) else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return r
if __name__=='__main__':argparse.ArgumentParser().parse_args();r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
