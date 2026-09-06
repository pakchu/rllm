"""Build source-only LEMRR-16 clocks."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_late_extreme_midpoint_rejection_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="7153527a55247b42eae6214f1505acc629eff3569df70949b3a94df475c0ba7f";START=pd.Timestamp("2023-06-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_midpoint_rejection","close_location_only","close_open_fade","direction_flip")
QUERY="""SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end AND EXTRACT(ISODOW FROM ts) IN (1,4) ORDER BY ts"""
SOURCE_DIR=Path("data/late_extreme_midpoint_rejection_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"daily_extreme_order.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/late_extreme_midpoint_rejection_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/late_extreme_midpoint_rejection_relay_controls_2023_2026");RESULT=Path("results/late_extreme_midpoint_rejection_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("source_day","decision_time","feature_available_time","source_valid","window_open","range_high","range_low","window_close","last_high_time","last_low_time","midpoint","daily_return","late_high_rejection","late_low_rejection")
CLOCK_COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","window_open","range_high","range_low","window_close","last_high_time","last_low_time","midpoint","daily_return","late_high_rejection","late_low_rejection")
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
 req=["ts","open","high","low","close"]
 if not set(req).issubset(raw.columns):raise ValueError("LEMRR schema drift")
 f=raw[req].copy();f.ts=pd.to_datetime(f.ts,utc=True,errors="coerce")
 for c in req[1:]:f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values("ts",kind="mergesort");f["source_day"]=f.ts.dt.floor("D");rows=[]
 for day,g in f.groupby("source_day",sort=True):
  expected=pd.date_range(day,day+pd.Timedelta(days=1),freq="1min",inclusive="left");p=g[["open","high","low","close"]]
  valid=bool(len(g)==1440 and not g.ts.duplicated().any() and g.ts.reset_index(drop=True).equals(pd.Series(expected,name="ts")) and np.isfinite(p).all(axis=1).all() and p.gt(0).all(axis=1).all() and g.high.ge(g[["open","close"]].max(axis=1)).all() and g.low.le(g[["open","close"]].min(axis=1)).all() and g.high.ge(g.low).all())
  if valid:
   o=float(g.open.iloc[0]);h=float(g.high.max());l=float(g.low.min());c=float(g.close.iloc[-1]);valid=h>l
  if valid:
   ht=pd.Timestamp(g.loc[g.high.eq(h),"ts"].iloc[-1]);lt=pd.Timestamp(g.loc[g.low.eq(l),"ts"].iloc[-1]);mid=.5*(h+l);valid=ht!=lt;ret=float(np.log(c/o));hi=bool(valid and ht>lt and c<mid);lo=bool(valid and lt>ht and c>mid)
  else:o=h=l=c=mid=ret=np.nan;ht=lt=pd.NaT;hi=lo=False
  rows.append({"source_day":day,"decision_time":day+pd.Timedelta(days=1),"feature_available_time":day+pd.Timedelta(days=1),"source_valid":valid,"window_open":o,"range_high":h,"range_low":l,"window_close":c,"last_high_time":ht,"last_low_time":lt,"midpoint":mid,"daily_return":ret,"late_high_rejection":hi,"late_low_rejection":lo})
 return pd.DataFrame(rows,columns=FEATURE_COLUMNS)
def signal(f:pd.DataFrame,control:str="primary")->pd.Series:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 last_side=pd.Series(np.where(f.last_high_time>f.last_low_time,-1,1),index=f.index);close_side=np.sign(f.window_close-f.midpoint);primary=f.source_valid&(f.late_high_rejection|f.late_low_rejection)
 if control=="no_midpoint_rejection":eligible=f.source_valid;side=last_side
 elif control=="close_location_only":eligible=f.source_valid;side=close_side
 elif control=="close_open_fade":eligible=primary;side=-np.sign(f.daily_return)
 elif control=="direction_flip":eligible=primary;side=-close_side
 else:eligible=primary;side=close_side
 return pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int).where(eligible&pd.Series(side,index=f.index).ne(0),0)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 sides=signal(f,control);rows=[];reserved=None
 for i in f.index[sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=16)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":d,"feature_available_time":f.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:f.at[i,c] for c in CLOCK_COLUMNS[9:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,float|int]:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("LEMRR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"lemrr_16_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":canonical_hash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"lemrr_16_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
