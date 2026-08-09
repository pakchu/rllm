"""Build source-only ASCR-16 clocks."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_asian_session_concordance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="c8649fcceb8adbe398e15a41b3bae2dbb6ba1285387954af4bab7e981472cff1";START=pd.Timestamp("2023-06-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("full_session_return","first_half_only","second_half_only","direction_flip")
QUERY="""SELECT date_bin('4 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,(array_agg(open ORDER BY ts))[1] AS block_open,max(high) AS block_high,min(low) AS block_low,(array_agg(close ORDER BY ts DESC))[1] AS block_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end AND EXTRACT(hour FROM ts)<8 GROUP BY block_start ORDER BY block_start"""
SOURCE_DIR=Path("data/asian_session_concordance_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"daily_session_halves.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/asian_session_concordance_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/asian_session_concordance_relay_controls_2023_2026");RESULT=Path("results/asian_session_concordance_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("source_day","decision_time","feature_available_time","source_valid","first_open","first_close","second_open","second_close","first_return","second_return","full_return","concordant")
CLOCK_COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","first_return","second_return","full_return","concordant")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def build_features(raw:pd.DataFrame)->pd.DataFrame:
 req=["block_start","block_open","block_high","block_low","block_close","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if not set(req).issubset(raw.columns):raise ValueError("ASCR schema drift")
 f=raw[req].copy()
 for c in ("block_start","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,errors="coerce")
 for c in ("block_open","block_high","block_low","block_close","source_rows","distinct_rows"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values("block_start",kind="mergesort").set_index("block_start");rows=[]
 for day in pd.date_range(START.floor("D"),END,freq="D",inclusive="left"):
  expected=pd.DatetimeIndex([day,day+pd.Timedelta(hours=4)]);w=f.reindex(expected);p=w[["block_open","block_high","block_low","block_close"]]
  ok=bool(np.isfinite(w[["block_open","block_high","block_low","block_close","source_rows","distinct_rows"]]).all(axis=1).all() and p.gt(0).all(axis=1).all() and w.source_rows.eq(240).all() and w.distinct_rows.eq(240).all() and w.coherent.fillna(False).astype(bool).all() and w.first_ts.equals(pd.Series(expected,index=expected)) and w.last_ts.equals(pd.Series(expected+pd.Timedelta(minutes=239),index=expected)))
  if ok:
   fo=float(w.block_open.iloc[0]);fc=float(w.block_close.iloc[0]);so=float(w.block_open.iloc[1]);sc=float(w.block_close.iloc[1]);fr=float(np.log(fc/fo));sr=float(np.log(sc/so));full=float(np.log(sc/fo));con=bool(fr!=0 and sr!=0 and np.sign(fr)==np.sign(sr))
  else:fo=fc=so=sc=fr=sr=full=np.nan;con=False
  rows.append({"source_day":day,"decision_time":day+pd.Timedelta(hours=8),"feature_available_time":day+pd.Timedelta(hours=8),"source_valid":ok,"first_open":fo,"first_close":fc,"second_open":so,"second_close":sc,"first_return":fr,"second_return":sr,"full_return":full,"concordant":con})
 return pd.DataFrame(rows,columns=FEATURE_COLUMNS)
def signal(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 if control=="full_session_return":eligible=f.source_valid&f.full_return.ne(0);side=np.sign(f.full_return)
 elif control=="first_half_only":eligible=f.source_valid&f.first_return.ne(0);side=np.sign(f.first_return)
 elif control=="second_half_only":eligible=f.source_valid&f.second_return.ne(0);side=np.sign(f.second_return)
 else:eligible=f.source_valid&f.concordant;side=np.sign(f.second_return)*(-1 if control=="direction_flip" else 1)
 return pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int).where(eligible,0)
def build_clock(f:pd.DataFrame,control:str="primary"):
 sides=signal(f,control);rows=[];reserved=None
 for i in f.index[sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=16)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":d,"feature_available_time":f.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:f.at[i,c] for c in CLOCK_COLUMNS[9:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str):
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("ASCR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"ascr_16_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":int(raw.source_rows.sum()),"aggregate_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":canonical_hash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"ascr_16_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
