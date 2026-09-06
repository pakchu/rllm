"""Source-only OI sponsorship gate for frozen HVCAVOIS-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_cross_structure_action_vote_oi_sponsorship as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE='/home/pakchu/rllm/.env';PREREG_SHA='15699ae93b1d2cd8e44435f3376f977302c0ef63d44e8205feb20155abe38530'
BASE=Path('data/high_volatility_cross_structure_action_vote_clocks_2023_2026.csv.gz');SOURCE=Path('data/high_volatility_cross_structure_action_vote_oi_sponsorship_sources_2023_2026/open_interest.csv.gz');CLOCK=Path('data/high_volatility_cross_structure_action_vote_oi_sponsorship_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/high_volatility_cross_structure_action_vote_oi_sponsorship_controls_2023_2026');RESULT=Path('results/high_volatility_cross_structure_action_vote_oi_sponsorship_support_2026-08-16.json')
MINIMUM_EVENTS={'train':8,'test':12,'eval':12,'final':8};QUERY="SELECT ts,sum_open_interest FROM open_interest_binance WHERE symbol=:symbol AND period='5m' AND ts>=:start AND ts<:end AND extract(minute FROM ts)=0 AND extract(second FROM ts)=0 AND extract(hour FROM ts) IN (0,8,16) ORDER BY ts"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_oi():
 from sqlalchemy import text
 db=postgres_engine()
 with db.connect() as c:d=pd.read_sql_query(text(QUERY),c,params={'symbol':'BTCUSDT','start':pd.Timestamp('2023-06-30T00:00Z').to_pydatetime(),'end':pd.Timestamp('2026-08-01T00:00Z').to_pydatetime()})
 db.dispose();d.ts=pd.to_datetime(d.ts,utc=True);d.sum_open_interest=pd.to_numeric(d.sum_open_interest,errors='coerce')
 if d.ts.duplicated().any() or not np.isfinite(d.sum_open_interest).all() or not d.sum_open_interest.ge(0).all():raise RuntimeError('invalid OI source')
 SOURCE.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(d,SOURCE);return d
def apply_gate(base,oi):
 d=base.copy()
 for c in ('decision_time','feature_available_time','entry_time','exit_time'):d[c]=pd.to_datetime(d[c],utc=True)
 if 'control' in d:d=d[d.control.eq('primary')].copy()
 values=oi.set_index('ts').sum_open_interest
 d['oi_start']=d.decision_time.map(values);d['oi_end']=d.decision_time.map(values.shift(-1))
 # Exact 8h endpoints are consecutive in the filtered 00/08/16 UTC OI series.
 positive=d.oi_start.gt(0)&d.oi_end.gt(0)
 d['oi_change']=np.where(positive,np.log(d.oi_end/d.oi_start),np.nan)
 x=d[positive&d.oi_change.gt(0)].copy();x['candidate']=prereg.POLICY_ID
 return x[['candidate','control','split','eligibility_id','decision_time','feature_available_time','entry_time','exit_time','side','active_action_count','long_vote_count','short_vote_count','oi_start','oi_end','oi_change']].reset_index(drop=True)
def stats(d,k):
 x=d[d.split.eq(k)];n=len(x);l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':n,'longs':l,'shorts':s,'minority_side_share':min(l,s)/n if n else 0.,'max_month_share':float(x.entry_time.dt.strftime('%Y-%m').value_counts().max()/n) if n else 0.}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('prereg drift')
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);base=pd.read_csv(BASE);oi=load_oi();clock=apply_gate(base,oi);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(clock,CLOCK);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(base,CONTROL_DIR/'no_oi_sponsorship_gate.csv.gz')
 support={k:stats(clock,k) for k in MINIMUM_EVENTS};checks={}
 for k,v in support.items():checks.update({f'{k}_minimum_events':v['events']>=MINIMUM_EVENTS[k],f'{k}_side_balance':v['minority_side_share']>=.2,f'{k}_month_concentration':v['max_month_share']<=.45})
 passed=all(checks.values());core={'protocol_version':'hvcavois_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'source':{'query':QUERY,'path':str(SOURCE),'sha256':sha(SOURCE),'rows':len(oi)},'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(clock)},'controls':{'no_oi_sponsorship_gate':{'path':str(CONTROL_DIR/'no_oi_sponsorship_gate.csv.gz'),'sha256':sha(CONTROL_DIR/'no_oi_sponsorship_gate.csv.gz'),'rows':len(base),'promotion_authorized':False}},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'}
 r={**core,'manifest_hash':canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=='__main__':
 r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
