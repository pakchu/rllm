"""Source-only cash-side router support for HVCELVCSR-8."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_caspc_ehlers_cash_side_router as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-07-01T00:00Z');END=pd.Timestamp('2026-08-01T00:00Z');PREREG_SHA='f5ba5f811001ee26bee38579d8528e142ab1ddfbdf8cc3e0f64efcdee615667f'
BASE=Path('data/high_volatility_caspc_ehlers_active_veto_clocks_2023_2026.csv.gz');SOURCE=Path('data/high_volatility_caspc_ehlers_cash_side_router_sources_2023_2026/spot_blocks.csv.gz');CLOCK=Path('data/high_volatility_caspc_ehlers_cash_side_router_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/high_volatility_caspc_ehlers_cash_side_router_controls_2023_2026');RESULT=Path('results/high_volatility_caspc_ehlers_cash_side_router_support_2026-08-16.json');MINIMUM_EVENTS={'train':8,'test':12,'eval':12,'final':8}
QUERY="SELECT date_bin('8 hours',ts-INTERVAL '3 hours',TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '11 hours' AS decision_time,(array_agg(open ORDER BY ts))[1] AS block_open,(array_agg(close ORDER BY ts DESC))[1] AS block_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance_spot WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end GROUP BY decision_time ORDER BY decision_time"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_spot():
 from sqlalchemy import text
 db=engine()
 with db.connect() as c:d=pd.read_sql_query(text(QUERY),c,params={'symbol':'BTCUSDT','start':START,'end':END})
 db.dispose()
 for c in ('decision_time','first_ts','last_ts'):d[c]=pd.to_datetime(d[c],utc=True)
 for c in ('block_open','block_close','source_rows','distinct_rows'):d[c]=pd.to_numeric(d[c],errors='coerce')
 start=d.decision_time-pd.Timedelta(hours=8);last=d.decision_time-pd.Timedelta(minutes=1);d['source_valid']=d.source_rows.eq(480)&d.distinct_rows.eq(480)&d.first_ts.eq(start)&d.last_ts.eq(last)&d.coherent.eq(True)&np.isfinite(d[['block_open','block_close']]).all(axis=1)&d.block_open.gt(0)&d.block_close.gt(0);d['cash_return']=np.log(d.block_close/d.block_open).where(d.source_valid)
 SOURCE.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(d,SOURCE);return d
def route(base,spot):
 d=base.copy()
 for c in ('decision_time','feature_available_time','entry_time','exit_time'):d[c]=pd.to_datetime(d[c],utc=True)
 if 'control' in d:d=d[d.control.eq('primary')].copy()
 x=d.merge(spot[['decision_time','source_valid','cash_return']],on='decision_time',how='left',validate='one_to_one');x=x[x.source_valid.eq(True)&x.cash_return.notna()&x.cash_return.ne(0)].copy();x['base_side']=x.side.astype(int);x['side']=np.sign(x.cash_return).astype(int);x['candidate']=prereg.POLICY_ID
 return x[['candidate','control','split','selected_action','decision_time','feature_available_time','entry_time','exit_time','side','active_action_count','base_side','cash_return']].reset_index(drop=True)
def stats(d,k):
 x=d[d.split.eq(k)];n=len(x);l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':n,'longs':l,'shorts':s,'minority_side_share':min(l,s)/n if n else 0.,'max_month_share':float(x.entry_time.dt.strftime('%Y-%m').value_counts().max()/n) if n else 0.}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('prereg drift')
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);base=pd.read_csv(BASE);spot=load_spot();clock=route(base,spot);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(clock,CLOCK);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(base,CONTROL_DIR/'immutable_base_side.csv.gz')
 support={k:stats(clock,k) for k in MINIMUM_EVENTS};checks={}
 for k,v in support.items():checks.update({f'{k}_minimum_events':v['events']>=MINIMUM_EVENTS[k],f'{k}_side_balance':v['minority_side_share']>=.2,f'{k}_month_concentration':v['max_month_share']<=.45})
 passed=all(checks.values());core={'protocol_version':'hvcelvcsr_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'source':{'query':QUERY,'path':str(SOURCE),'sha256':sha(SOURCE),'rows':len(spot),'valid_rows':int(spot.source_valid.sum())},'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(clock)},'controls':{'immutable_base_side':{'path':str(CONTROL_DIR/'immutable_base_side.csv.gz'),'sha256':sha(CONTROL_DIR/'immutable_base_side.csv.gz'),'rows':len(base),'promotion_authorized':False}},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'}
 r={**core,'manifest_hash':canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=='__main__':
 r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
