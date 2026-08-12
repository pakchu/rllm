"""Materialize source-only HVINOR-8 support clocks."""
from __future__ import annotations
import argparse, bisect, hashlib, json, math
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_india_opening_reversal_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="69941911f22565483141a87814d5f89202d4008d5a8570a8a9ad091da6eb3f78";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
ROOT=Path("data/high_volatility_india_opening_reversal_relay_sources_2023_2026");PANEL=ROOT/"india_opening_sessions.csv.gz";MANIFEST=ROOT/"manifest.json"
BTC=Path("data/high_volatility_palladium_asymmetric_spillover_relay_sources_2022_2026/btc_1m_ts_open_close.csv.gz");BTC_SHA="d2e42ca3ea6e440ce50bea01beb709a212f4e531a49c560819652ef8db735dd2";BTC_MANIFEST=BTC.parent/"manifest.json";BTC_MANIFEST_SHA="2cfa83ff409ca7e06d18005408bae388dfda4906b1fae95c8ba9cdfea00aa3e2"
CLOCK=Path("data/high_volatility_india_opening_reversal_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_india_opening_reversal_relay_controls_2023_2026");RESULT=Path("results/high_volatility_india_opening_reversal_relay_support_2026-08-13.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8}
CONTROLS=("no_variation_gate","no_reversal_tail","no_opening_dominance","one_session_stale_reversal","direction_flip","forced_long")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","pre_open_return","opening_return","reversal_rank","btc_realized_variation","variation_rank")
QUERY="""WITH tagged AS (
 SELECT ts,open,high,low,close,date_trunc('day',ts+interval '2 hours 30 minutes') AS decision_day
 FROM bars_polygon WHERE symbol='USDINR' AND interval='1m' AND ts>=:start AND ts<:end
 AND ((ts-date_trunc('day',ts))>=interval '21 hours 30 minutes' OR (ts-date_trunc('day',ts))<interval '4 hours 30 minutes')
), segmented AS (
 SELECT *,CASE WHEN ts<decision_day+interval '3 hours 30 minutes' THEN 'pre_open' ELSE 'opening' END AS segment
 FROM tagged WHERE extract(isodow from decision_day) BETWEEN 1 AND 5
)
SELECT decision_day,segment,(array_agg(open ORDER BY ts))[1] AS first_open,max(high) AS session_high,min(low) AS session_low,(array_agg(close ORDER BY ts DESC))[1] AS last_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts
FROM segmented GROUP BY decision_day,segment ORDER BY decision_day,segment"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rank(v:pd.Series,maximum:int=90,minimum:int=60)->pd.Series:
 a=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(a),np.nan);q=deque();s=[]
 for i,x in enumerate(a):
  if math.isfinite(x) and len(q)>=minimum:l=bisect.bisect_left(s,x);r=bisect.bisect_right(s,x);o[i]=(l+.5*(r-l))/len(s)
  if math.isfinite(x):
   q.append(x);bisect.insort(s,x)
   if len(q)>maximum:old=q.popleft();s.pop(bisect.bisect_left(s,old))
 return pd.Series(o,index=v.index)
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_fx()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 with db.connect() as c:raw=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 db.dispose();raw["decision_day"]=pd.to_datetime(raw.decision_day,utc=True);raw["first_ts"]=pd.to_datetime(raw.first_ts,utc=True);raw["last_ts"]=pd.to_datetime(raw.last_ts,utc=True)
 for c in ("first_open","session_high","session_low","last_close","source_rows","distinct_timestamps"):raw[c]=pd.to_numeric(raw[c],errors="coerce")
 return raw
def build_states(raw:pd.DataFrame)->pd.DataFrame:
 frames={}
 for segment in ("pre_open","opening"):
  z=raw[raw.segment.eq(segment)].copy().set_index("decision_day");z=z.add_prefix(segment+"_");frames[segment]=z
 x=frames["pre_open"].join(frames["opening"],how="outer").sort_index().reset_index()
 pre_start=x.decision_day-pd.Timedelta(hours=2,minutes=30);pre_end=x.decision_day+pd.Timedelta(hours=3,minutes=30);open_start=pre_end;open_end=x.decision_day+pd.Timedelta(hours=4,minutes=30)
 def coherent(prefix):
  cols=[f"{prefix}_first_open",f"{prefix}_session_high",f"{prefix}_session_low",f"{prefix}_last_close"]
  return np.isfinite(x[cols]).all(axis=1)&x[cols].gt(0).all(axis=1)&x[f"{prefix}_session_high"].ge(x[[f"{prefix}_first_open",f"{prefix}_last_close"]].max(axis=1))&x[f"{prefix}_session_low"].le(x[[f"{prefix}_first_open",f"{prefix}_last_close"]].min(axis=1))
 x["session_valid"]=x.pre_open_distinct_timestamps.ge(330)&x.opening_distinct_timestamps.ge(55)&x.pre_open_first_ts.le(pre_start+pd.Timedelta(minutes=5))&x.pre_open_last_ts.ge(pre_end-pd.Timedelta(minutes=5))&x.opening_first_ts.le(open_start+pd.Timedelta(minutes=5))&x.opening_last_ts.ge(open_end-pd.Timedelta(minutes=5))&coherent("pre_open")&coherent("opening")
 x["pre_open_return"]=np.log(x.pre_open_last_close/x.pre_open_first_open).where(x.session_valid);x["opening_return"]=np.log(x.opening_last_close/x.opening_first_open).where(x.session_valid);x["reversal"]=x.pre_open_return.mul(x.opening_return).lt(0);x["opening_dominance"]=x.opening_return.abs().ge(x.pre_open_return.abs());x["reversal_rank"]=rank(x.opening_return.abs().where(x.session_valid&x.reversal));x["decision_time"]=open_end
 if sha(BTC)!=BTC_SHA or sha(BTC_MANIFEST)!=BTC_MANIFEST_SHA:raise RuntimeError("HVINOR BTC source drift")
 b=pd.read_csv(BTC,compression="gzip");b["ts"]=pd.to_datetime(b.ts,utc=True);b=b.sort_values("ts").set_index("ts");variations=[];valid=[]
 for d in x.decision_time:
  w=b.loc[(b.index>=d-pd.Timedelta(hours=24))&(b.index<d)];ok=len(w)==1440 and w.index.min()==d-pd.Timedelta(hours=24) and w.index.max()==d-pd.Timedelta(minutes=1) and w.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all() and np.isfinite(w[["open","close"]]).all(axis=None) and w[["open","close"]].gt(0).all(axis=None);valid.append(ok);variations.append(float(np.sqrt(np.square(np.log(w.close.to_numpy(float)/w.open.to_numpy(float))).sum())) if ok else np.nan)
 x["btc_realized_variation"]=variations;x["btc_valid"]=valid;x["variation_rank"]=rank(x.btc_realized_variation.where(x.session_valid&x.btc_valid));return x
def conditions(x:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 used=x.shift(1) if control=="one_session_stale_reversal" else x;active=x.session_valid&x.btc_valid&used.reversal.eq(True)
 if control!="no_opening_dominance":active&=used.opening_dominance.eq(True)
 if control!="no_reversal_tail":active&=used.reversal_rank.ge(.70)
 if control!="no_variation_gate":active&=x.variation_rank.ge(.65)
 side=-np.sign(used.opening_return).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=pd.Series(1,index=x.index)
 return active,side
def clock(x:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(x,control);rows=[];reserved=None
 for i in x.index[active]:
  d=pd.Timestamp(x.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":"HVINOR-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"pre_open_return":float(x.at[i,"pre_open_return"]),"opening_return":float(x.at[i,"opening_return"]),"reversal_rank":float(x.at[i,"reversal_rank"]),"btc_realized_variation":float(x.at[i,"btc_realized_variation"]),"variation_rank":float(x.at[i,"variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c,s):
 q=c[c.split.eq(s)]
 if q.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(q.side.eq(1).sum());sh=int(q.side.eq(-1).sum());m=q.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(q),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(q),"max_month_share":int(m.max())/len(q)}
def run()->dict:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVINOR prereg drift")
 raw=load_fx();x=build_states(raw);primary=clock(x);controls={n:clock(x,n) for n in CONTROLS};ROOT.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(x,PANEL);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"hvinor_8_source_v1","query":QUERY,"table":"bars_polygon","symbol":"USDINR","window":[START.isoformat(),END.isoformat()],"raw_segment_rows":len(raw),"output":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(x)},"postentry_outcomes_opened":False,"gross9_rows_opened":False};manifest={**source_core,"manifest_hash":chash(source_core)};MANIFEST.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
 support={s:stats(primary,s) for s in SPLITS};checks={k:v for s,z in support.items() for k,v in ((f"{s}_minimum_events",z["events"]>=MINIMUM[s]),(f"{s}_side_balance",z["minority_side_share"]>=.2),(f"{s}_month_concentration",z["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvinor_8_source_support_v1","policy_id":"HVINOR-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
