"""Materialize source-only HVRTR-12 support clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_range_traversal_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="c931469ea605b5b6bc036acc556e860adb8130e3fae3455ef22ceae1ec531014";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");SOURCE_DIR=Path("data/high_volatility_range_traversal_relay_sources_2023_2026");DAILY=SOURCE_DIR/"btc_daily_traversal.csv.gz";SOURCE_MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/high_volatility_range_traversal_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_range_traversal_relay_controls_2023_2026");RESULT=Path("results/high_volatility_range_traversal_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_range_gate","no_extreme_order","no_close_location","one_day_stale_geometry","direction_flip","same_clock_forced_long");COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","daily_open","daily_high","daily_low","daily_close","first_high_time","first_low_time","log_range","range_rank","close_location")
QUERY="""WITH tagged AS (SELECT ts,date_trunc('day',ts) AS source_day,open,high,low,close,max(high) OVER (PARTITION BY date_trunc('day',ts)) AS daily_high,min(low) OVER (PARTITION BY date_trunc('day',ts)) AS daily_low FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end) SELECT source_day,(array_agg(open ORDER BY ts))[1] AS daily_open,max(daily_high) AS daily_high,min(daily_low) AS daily_low,(array_agg(close ORDER BY ts DESC))[1] AS daily_close,min(ts) FILTER (WHERE high=daily_high) AS first_high_time,min(ts) FILTER (WHERE low=daily_low) AS first_low_time,count(*) AS source_rows,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM tagged GROUP BY source_day ORDER BY source_day"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(v:pd.Series,lookback:int=270,minimum:int=180)->pd.Series:
 n=pd.to_numeric(v,errors="coerce").astype(float);out=pd.Series(np.nan,index=n.index,dtype=float);h=[]
 for i,current in n.items():
  prior=h[-lookback:]
  if math.isfinite(current) and len(prior)>=minimum:
   a=np.asarray(prior);out.at[i]=(np.sum(a<current)+.5*np.sum(a==current))/len(a)
  if math.isfinite(current):h.append(current)
 return out
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def materialize()->dict:
 from sqlalchemy import text
 db=postgres_engine()
 with db.connect() as c:f=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 db.dispose();f["source_day"]=pd.to_datetime(f.source_day,utc=True);f["first_ts"]=pd.to_datetime(f.first_ts,utc=True);f["last_ts"]=pd.to_datetime(f.last_ts,utc=True);f["first_high_time"]=pd.to_datetime(f.first_high_time,utc=True);f["first_low_time"]=pd.to_datetime(f.first_low_time,utc=True)
 for c in ("daily_open","daily_high","daily_low","daily_close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f["source_valid"]=f.source_rows.eq(1440)&f.distinct_timestamps.eq(1440)&f.first_ts.eq(f.source_day)&f.last_ts.eq(f.source_day+pd.Timedelta(hours=23,minutes=59))&f.coherent.eq(True)&f.first_high_time.notna()&f.first_low_time.notna()&np.isfinite(f[["daily_open","daily_high","daily_low","daily_close"]]).all(axis=1)&f[["daily_open","daily_high","daily_low","daily_close"]].gt(0).all(axis=1)&f.daily_high.ge(f[["daily_open","daily_close"]].max(axis=1))&f.daily_low.le(f[["daily_open","daily_close"]].min(axis=1))&f.daily_high.gt(f.daily_low);f["log_range"]=np.log(f.daily_high/f.daily_low).where(f.source_valid);f["range_rank"]=strict_prior_midrank(f.log_range.where(f.source_valid));f["close_location"]=(f.daily_close-f.daily_low)/(f.daily_high-f.daily_low);f["decision_time"]=f.source_day+pd.Timedelta(days=1)
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,DAILY);core={"protocol_version":"hvrtr_12_daily_btc_source_v1","query":QUERY,"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","window":[START.isoformat(),END.isoformat()],"outcomes_opened":False,"candidate_incidence_opened":False,"no_imputation":True,"output":{"path":str(DAILY),"sha256":sha(DAILY),"rows":len(f),"valid_rows":int(f.source_valid.sum())}};p={**core,"manifest_hash":chash(core)};SOURCE_MANIFEST.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n");return p
def features()->pd.DataFrame:
 f=pd.read_csv(DAILY,compression="gzip");f["source_day"]=pd.to_datetime(f.source_day,utc=True);f["decision_time"]=pd.to_datetime(f.decision_time,utc=True);f["source_valid"]=f.source_valid.astype(str).str.lower().eq("true")
 for c in ("daily_open","daily_high","daily_low","daily_close","log_range","range_rank","close_location"):f[c]=pd.to_numeric(f[c],errors="coerce")
 for c in ("first_high_time","first_low_time"):f[c]=pd.to_datetime(f[c],utc=True)
 f["signal_valid"]=f.source_valid&np.isfinite(f[["daily_open","daily_high","daily_low","daily_close","log_range","range_rank","close_location"]]).all(axis=1)&f.log_range.gt(0);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 op,cl,loc,rank,hi,lo=f.daily_open,f.daily_close,f.close_location,f.range_rank,f.first_high_time,f.first_low_time
 if control=="one_day_stale_geometry":op,cl,loc,rank,hi,lo=op.shift(1),cl.shift(1),loc.shift(1),rank.shift(1),hi.shift(1),lo.shift(1)
 range_gate=pd.Series(True,index=f.index) if control=="no_range_gate" else rank.ge(.65);up_order=lo.lt(hi);down_order=hi.lt(lo)
 if control=="no_extreme_order":up_order=down_order=pd.Series(True,index=f.index)
 upper=up_order&loc.ge(.75)&cl.gt(op);lower=down_order&loc.le(.25)&cl.lt(op)
 if control=="no_close_location":upper=up_order&cl.gt(op);lower=down_order&cl.lt(op)
 active=f.signal_valid&np.isfinite(op)&np.isfinite(cl)&np.isfinite(loc)&np.isfinite(rank)&hi.notna()&lo.notna()&range_gate&(upper|lower);side=pd.Series(np.where(upper,1,-1),index=f.index);side=-side if control=="direction_flip" else side
 if control=="same_clock_forced_long":side=pd.Series(1,index=f.index)
 return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[]
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12);split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  rows.append({"candidate":"HVRTR-12","control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"daily_open":float(f.at[i,"daily_open"]),"daily_high":float(f.at[i,"daily_high"]),"daily_low":float(f.at[i,"daily_low"]),"daily_close":float(f.at[i,"daily_close"]),"first_high_time":f.at[i,"first_high_time"],"first_low_time":f.at[i,"first_low_time"],"log_range":float(f.at[i,"log_range"]),"range_rank":float(f.at[i,"range_rank"]),"close_location":float(f.at[i,"close_location"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVRTR preregistration drift")
 sm=materialize();f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"hvrtr_12_source_support_v1","policy_id":"HVRTR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};p={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n");return p
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
