"""Materialize source-only HVWRSR-8 support clocks."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_won_risk_session_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env"; PREREG_SHA="a67c8bf113952abe3ad032f16142f3325dea2498615a8935dd4ef5aab6f50c64"; START=pd.Timestamp("2023-01-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/high_volatility_won_risk_session_relay_sources_2023_2026"); SESSION=SOURCE_DIR/"won_sessions.csv.gz"; SOURCE_MANIFEST=SOURCE_DIR/"manifest.json"
PRICE=Path("data/high_volatility_palladium_asymmetric_spillover_relay_sources_2022_2026/btc_1m_ts_open_close.csv.gz"); PRICE_SHA="d2e42ca3ea6e440ce50bea01beb709a212f4e531a49c560819652ef8db735dd2"; PRICE_MANIFEST=PRICE.parent/"manifest.json"; PRICE_MANIFEST_SHA="2cfa83ff409ca7e06d18005408bae388dfda4906b1fae95c8ba9cdfea00aa3e2"
CLOCK=Path("data/high_volatility_won_risk_session_relay_clocks_2023_2026.csv.gz"); CONTROL_DIR=Path("data/high_volatility_won_risk_session_relay_controls_2023_2026"); RESULT=Path("results/high_volatility_won_risk_session_relay_support_2026-08-13.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))}; MINIMUM={"train":8,"test":12,"eval":12,"final":8}; CONTROLS=("no_variation_gate","no_shock_tail","one_session_stale_won","direction_flip","forced_long")
SYMBOLS=("USDKRW",)
COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","won_risk_direction","session_return","shock_rank","btc_realized_variation","btc_realized_variation_rank")
QUERY="""SELECT date_trunc('day',ts) AS source_day,(array_agg(open ORDER BY ts))[1] AS session_open,max(high) AS session_high,min(low) AS session_low,(array_agg(close ORDER BY ts DESC))[1] AS session_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts FROM bars_polygon WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end AND extract(isodow from ts) BETWEEN 1 AND 5 AND extract(hour from ts)>=0 AND extract(hour from ts)<8 GROUP BY 1 ORDER BY 1"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def causal_z(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);out=np.full(len(x),np.nan)
 for i,current in enumerate(x):
  prior=x[max(0,i-lookback):i];prior=prior[np.isfinite(prior)]
  if np.isfinite(current) and len(prior)>=minimum:
   std=float(np.std(prior,ddof=1));out[i]=(current-float(np.mean(prior)))/std if std>0 else np.nan
 return pd.Series(out,index=v.index)
def strict_prior_midrank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 n=pd.to_numeric(v,errors="coerce").astype(float);out=pd.Series(np.nan,index=n.index,dtype=float);history=[]
 for i,current in n.items():
  prior=history[-lookback:]
  if math.isfinite(current) and len(prior)>=minimum:
   a=np.asarray(prior);out.at[i]=(np.sum(a<current)+.5*np.sum(a==current))/len(a)
  if math.isfinite(current):history.append(current)
 return out
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def materialize_sessions()->dict:
 from sqlalchemy import text
 db=postgres_engine()
 with db.connect() as connection:raw=pd.read_sql_query(text(QUERY),connection,params={"symbol":"USDKRW","start":START.to_pydatetime(),"end":END.to_pydatetime()})
 db.dispose();raw["source_day"]=pd.to_datetime(raw.source_day,utc=True);raw["first_ts"]=pd.to_datetime(raw.first_ts,utc=True);raw["last_ts"]=pd.to_datetime(raw.last_ts,utc=True)
 for column in ("session_open","session_high","session_low","session_close"):raw[column]=pd.to_numeric(raw[column],errors="coerce")
 valid=raw.distinct_timestamps.ge(450)&raw.first_ts.le(raw.source_day+pd.Timedelta(hours=0,minutes=5))&raw.last_ts.ge(raw.source_day+pd.Timedelta(hours=7,minutes=55))&np.isfinite(raw[["session_open","session_high","session_low","session_close"]]).all(axis=1)&raw[["session_open","session_high","session_low","session_close"]].gt(0).all(axis=1)&raw.session_high.ge(raw[["session_open","session_close"]].max(axis=1))&raw.session_low.le(raw[["session_open","session_close"]].min(axis=1))
 frame=pd.DataFrame({"source_day":raw.source_day,"source_valid":valid,"session_return":np.log(raw.session_close/raw.session_open).where(valid)});frame["shock_rank"]=strict_prior_midrank(frame.session_return.abs().where(frame.source_valid));frame["won_risk_direction"]=-np.sign(frame.session_return).fillna(0).astype(int);frame["decision_time"]=frame.source_day+pd.Timedelta(hours=8)
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(frame,SESSION);core={"protocol_version":"hvwrsr_8_won_source_v1","query":QUERY,"table":"bars_polygon","symbols":["USDKRW"],"interval":"1m","window":[START.isoformat(),END.isoformat()],"outcomes_opened":False,"candidate_incidence_opened":False,"no_imputation":True,"output":{"path":str(SESSION),"sha256":sha(SESSION),"rows":len(frame),"valid_rows":int(frame.source_valid.sum())}};payload={**core,"manifest_hash":chash(core)};SOURCE_MANIFEST.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");return payload
def features()->pd.DataFrame:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVWRSR prereg drift")
 if sha(PRICE)!=PRICE_SHA or sha(PRICE_MANIFEST)!=PRICE_MANIFEST_SHA:raise RuntimeError("HVWRSR BTC source drift")
 f=pd.read_csv(SESSION,compression="gzip");f["source_day"]=pd.to_datetime(f.source_day,utc=True);f["decision_time"]=pd.to_datetime(f.decision_time,utc=True);f["source_valid"]=f.source_valid.astype(str).str.lower().eq("true")
 for column in ("won_risk_direction","session_return","shock_rank"):f[column]=pd.to_numeric(f[column],errors="coerce")
 p=pd.read_csv(PRICE,compression="gzip");p["ts"]=pd.to_datetime(p.ts,utc=True);p["open"]=pd.to_numeric(p.open,errors="coerce");p["close"]=pd.to_numeric(p.close,errors="coerce");p=p.sort_values("ts").set_index("ts");variations=[];validity=[]
 for decision in f.decision_time:
  w=p.loc[(p.index>=decision-pd.Timedelta(hours=24))&(p.index<decision)]
  valid=len(w)==1440 and w.index.is_unique and w.index.min()==decision-pd.Timedelta(hours=24) and w.index.max()==decision-pd.Timedelta(minutes=1) and w.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all() and np.isfinite(w[["open","close"]]).all(axis=None) and w[["open","close"]].gt(0).all(axis=None)
  validity.append(bool(valid));variations.append(float(np.sqrt(np.square(np.log(w.close.to_numpy(float)/w.open.to_numpy(float))).sum())) if valid else np.nan)
 f["btc_realized_variation"]=variations;f["btc_valid"]=validity;f["btc_realized_variation_rank"]=strict_prior_midrank(f.btc_realized_variation.where(f.btc_valid));f["signal_valid"]=f.source_valid&f.btc_valid&np.isfinite(f[["won_risk_direction","session_return","shock_rank","btc_realized_variation","btc_realized_variation_rank"]]).all(axis=1)&f.won_risk_direction.abs().eq(1);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 direction=f.won_risk_direction;shock=f.shock_rank
 if control=="one_session_stale_won":direction=direction.shift(1);shock=shock.shift(1)
 tail=pd.Series(True,index=f.index) if control=="no_shock_tail" else shock.ge(.75);vol=pd.Series(True,index=f.index) if control=="no_variation_gate" else f.btc_realized_variation_rank.ge(.65);active=f.signal_valid&np.isfinite(direction)&np.isfinite(shock)&direction.abs().eq(1)&tail&vol;side=direction
 if control=="direction_flip":side=-side
 if control=="forced_long":side=pd.Series(1,index=f.index)
 return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"HVWRSR-8","control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"won_risk_direction":int(f.at[i,"won_risk_direction"]),"session_return":float(f.at[i,"session_return"]),"shock_rank":float(f.at[i,"shock_rank"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_realized_variation_rank":float(f.at[i,"btc_realized_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 sm=materialize_sessions();f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"hvwrsr_8_source_support_v1","policy_id":"HVWRSR-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"won_session":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_btc":{"path":str(PRICE_MANIFEST),"sha256":sha(PRICE_MANIFEST)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};payload={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");return payload
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
