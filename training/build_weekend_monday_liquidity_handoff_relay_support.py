"""Build source-only WMLHR-16 weekly clocks."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_weekend_monday_liquidity_handoff_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="3f741f7c7750844d214c9b46fe28b123baa52e0cd3ed76e4b4755b7950e06b5c";START=pd.Timestamp("2023-06-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_liquidity_reentry","weekend_continuation","monday_fade","direction_flip")
QUERY="""SELECT ts,open,high,low,close,quote_asset_volume FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end AND (EXTRACT(ISODOW FROM ts) IN (6,7) OR (EXTRACT(ISODOW FROM ts)=1 AND EXTRACT(hour FROM ts)<8)) ORDER BY ts"""
SOURCE_DIR=Path("data/weekend_monday_liquidity_handoff_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"weekly_handoff.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/weekend_monday_liquidity_handoff_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/weekend_monday_liquidity_handoff_relay_controls_2023_2026");RESULT=Path("results/weekend_monday_liquidity_handoff_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("monday_day","decision_time","feature_available_time","source_valid","weekend_return","monday_return","weekend_quote_per_hour","monday_quote_per_hour","direction_handoff","liquidity_reentry")
CLOCK_COLUMNS=("candidate","control","split","monday_day","decision_time","feature_available_time","entry_time","exit_time","side","weekend_return","monday_return","weekend_quote_per_hour","monday_quote_per_hour","direction_handoff","liquidity_reentry")
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
 req=["ts","open","high","low","close","quote_asset_volume"]
 if not set(req).issubset(raw.columns):raise ValueError("WMLHR schema drift")
 f=raw[req].copy();f.ts=pd.to_datetime(f.ts,utc=True,errors="coerce")
 for c in req[1:]:f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values("ts",kind="mergesort");day=f.ts.dt.floor("D");weekday=day.dt.weekday;f["monday_day"]=day+pd.to_timedelta(np.where(weekday==5,2,np.where(weekday==6,1,0)),unit="D")
 rows=[]
 for monday,g in f.groupby("monday_day",sort=True):
  weekend=g[(g.ts>=monday-pd.Timedelta(days=2))&(g.ts<monday)];opening=g[(g.ts>=monday)&(g.ts<monday+pd.Timedelta(hours=8))]
  ew=pd.date_range(monday-pd.Timedelta(days=2),monday,freq="1min",inclusive="left");em=pd.date_range(monday,monday+pd.Timedelta(hours=8),freq="1min",inclusive="left")
  def valid(x,expected):
   p=x[["open","high","low","close"]];return bool(len(x)==len(expected) and not x.ts.duplicated().any() and x.ts.reset_index(drop=True).equals(pd.Series(expected,name="ts")) and np.isfinite(x[["open","high","low","close","quote_asset_volume"]]).all(axis=1).all() and p.gt(0).all(axis=1).all() and x.quote_asset_volume.ge(0).all() and x.high.ge(x[["open","close"]].max(axis=1)).all() and x.low.le(x[["open","close"]].min(axis=1)).all() and x.high.ge(x.low).all())
  ok=valid(weekend,ew)&valid(opening,em);wr=float(np.log(weekend.close.iloc[-1]/weekend.open.iloc[0])) if ok else np.nan;mr=float(np.log(opening.close.iloc[-1]/opening.open.iloc[0])) if ok else np.nan;wq=float(weekend.quote_asset_volume.sum()/48) if ok else np.nan;mq=float(opening.quote_asset_volume.sum()/8) if ok else np.nan
  hand=bool(ok and wr!=0 and mr!=0 and np.sign(wr)==-np.sign(mr));liq=bool(hand and mq>wq)
  rows.append({"monday_day":monday,"decision_time":monday+pd.Timedelta(hours=8),"feature_available_time":monday+pd.Timedelta(hours=8),"source_valid":ok,"weekend_return":wr,"monday_return":mr,"weekend_quote_per_hour":wq,"monday_quote_per_hour":mq,"direction_handoff":hand,"liquidity_reentry":liq})
 return pd.DataFrame(rows,columns=FEATURE_COLUMNS)
def signal(f:pd.DataFrame,control:str="primary")->pd.Series:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 primary=f.source_valid&f.direction_handoff&f.liquidity_reentry
 if control=="no_liquidity_reentry":eligible=f.source_valid&f.direction_handoff;stat=f.monday_return
 elif control=="weekend_continuation":eligible=primary;stat=f.weekend_return
 elif control in ("monday_fade","direction_flip"):eligible=primary;stat=-f.monday_return
 else:eligible=primary;stat=f.monday_return
 return np.sign(stat).astype("Int64").fillna(0).astype(int).where(eligible&stat.ne(0),0)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 sides=signal(f,control);rows=[];reserved=None
 for i in f.index[sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=16)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"monday_day":f.at[i,"monday_day"],"decision_time":d,"feature_available_time":f.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:f.at[i,c] for c in CLOCK_COLUMNS[9:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,float|int]:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("WMLHR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"wmlhr_16_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":canonical_hash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"wmlhr_16_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
