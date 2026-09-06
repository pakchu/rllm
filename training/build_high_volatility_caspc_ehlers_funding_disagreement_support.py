"""Source-only support for frozen HVCELVFD-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_caspc_ehlers_funding_disagreement as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="0145fffe2fb89c32822cec1cf0cea427400eeb43946d4d138aed14168feb37e5"
BASE=Path("data/high_volatility_caspc_ehlers_active_veto_clocks_2023_2026.csv.gz")
SOURCE=Path("data/high_volatility_caspc_ehlers_funding_disagreement_sources_2023_2026/funding.csv.gz")
CLOCK=Path("data/high_volatility_caspc_ehlers_funding_disagreement_clocks_2023_2026.csv.gz")
CONTROL_DIR=Path("data/high_volatility_caspc_ehlers_funding_disagreement_controls_2023_2026")
RESULT=Path("results/high_volatility_caspc_ehlers_funding_disagreement_support_2026-08-16.json")
MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8}
QUERY="SELECT funding_time,funding_rate FROM funding_rates_binance WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_funding():
 from sqlalchemy import text
 db=postgres_engine()
 with db.connect() as c:d=pd.read_sql_query(text(QUERY),c,params={"symbol":"BTCUSDT","start":pd.Timestamp('2023-06-30T00:00Z').to_pydatetime(),"end":pd.Timestamp('2026-08-01T00:00Z').to_pydatetime()})
 db.dispose();d.funding_time=pd.to_datetime(d.funding_time,utc=True);d.funding_rate=pd.to_numeric(d.funding_rate,errors='coerce')
 if d.funding_time.duplicated().any() or not np.isfinite(d.funding_rate).all():raise RuntimeError("invalid funding source")
 SOURCE.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(d,SOURCE);return d
def apply_gate(base:pd.DataFrame,funding:pd.DataFrame):
 d=base.copy()
 for c in ('decision_time','feature_available_time','entry_time','exit_time'):d[c]=pd.to_datetime(d[c],utc=True)
 d=d[d.control.eq('primary')].sort_values('decision_time')
 x=pd.merge_asof(d,funding.sort_values('funding_time'),left_on='decision_time',right_on='funding_time',direction='backward',allow_exact_matches=True)
 age=x.decision_time-x.funding_time;valid=x.funding_rate.notna()&age.ge(pd.Timedelta(0))&age.lt(pd.Timedelta(hours=8))
 selected=x[valid&(x.side*x.funding_rate<0)].copy();selected['candidate']=prereg.POLICY_ID
 return selected[['candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','funding_time','funding_rate']].reset_index(drop=True)
def stats(d,split):
 x=d[d.split.eq(split)];n=len(x);l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':n,'longs':l,'shorts':s,'minority_side_share':min(l,s)/n if n else 0.,'max_month_share':float(x.entry_time.dt.strftime('%Y-%m').value_counts().max()/n) if n else 0.}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('prereg drift')
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);base=pd.read_csv(BASE);funding=load_funding();clock=apply_gate(base,funding)
 CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(clock,CLOCK);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(base,CONTROL_DIR/'no_funding_disagreement_gate.csv.gz')
 support={k:stats(clock,k) for k in MINIMUM_EVENTS};checks={}
 for k,v in support.items():checks.update({f'{k}_minimum_events':v['events']>=MINIMUM_EVENTS[k],f'{k}_side_balance':v['minority_side_share']>=.2,f'{k}_month_concentration':v['max_month_share']<=.45})
 passed=all(checks.values());core={'protocol_version':'hvcelvfd_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'source':{'query':QUERY,'path':str(SOURCE),'sha256':sha(SOURCE),'rows':len(funding)},'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(clock)},'controls':{'no_funding_disagreement_gate':{'path':str(CONTROL_DIR/'no_funding_disagreement_gate.csv.gz'),'sha256':sha(CONTROL_DIR/'no_funding_disagreement_gate.csv.gz'),'rows':len(base),'promotion_authorized':False}},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'}
 r={**core,'manifest_hash':canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=='__main__':
 r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
