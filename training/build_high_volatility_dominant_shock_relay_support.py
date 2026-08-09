"""Build source-only support for frozen HVDSR-12."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_dominant_shock_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="3f821d75a17044be86c1f4430e324bc209a19e9204e6553dd27e036ef0bd8466";START=pd.Timestamp("2023-01-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_dominant_share_gate","no_variation_gate","completed_day_return_side","latest_maximum_tie_break","one_day_stale_features","direction_flip","same_clock_forced_long")
QUERY="""SELECT date_trunc('hour',ts) AS hour_time,(array_agg(open ORDER BY ts))[1] AS hour_open,max(high) AS hour_high,min(low) AS hour_low,(array_agg(close ORDER BY ts DESC))[1] AS hour_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
SOURCE_DIR=Path("data/high_volatility_dominant_shock_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"daily_dominant_shock.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/high_volatility_dominant_shock_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_dominant_shock_relay_controls_2023_2026");RESULT=Path("results/high_volatility_dominant_shock_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("source_day","feature_available_time","source_valid","dominant_hour","dominant_return","latest_dominant_return","day_return","dominant_share","realized_variation","dominant_share_rank","variation_rank")
CLOCK_COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side",*FEATURE_COLUMNS[3:])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def prior_midrank(v:pd.Series,lookback:int=270,minimum:int=180)->pd.Series:
 out=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in pd.to_numeric(v,errors="coerce").items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:
   a=np.asarray(p);out.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(float(x))
 return out
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_hours()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def build_features(hours:pd.DataFrame)->pd.DataFrame:
 req={"hour_time","hour_open","hour_high","hour_low","hour_close","source_rows","distinct_rows","first_ts","last_ts","coherent"}
 if not req.issubset(hours):raise ValueError("HVDSR source schema drift")
 h=hours.copy()
 for c in ("hour_time","first_ts","last_ts"):h[c]=pd.to_datetime(h[c],utc=True,errors="coerce")
 for c in ("hour_open","hour_high","hour_low","hour_close","source_rows","distinct_rows"):h[c]=pd.to_numeric(h[c],errors="coerce")
 h=h.sort_values("hour_time",kind="mergesort").set_index("hour_time");rows=[]
 for d in pd.date_range(START,END,freq="1D",inclusive="left"):
  idx=pd.date_range(d,d+pd.Timedelta(days=1),freq="1h",inclusive="left");w=h.reindex(idx);expected=pd.Series(idx,index=idx);ok=np.isfinite(w[["hour_open","hour_high","hour_low","hour_close","source_rows","distinct_rows"]]).all(axis=1)&w[["hour_open","hour_high","hour_low","hour_close"]].gt(0).all(axis=1)&w.source_rows.eq(60)&w.distinct_rows.eq(60)&w.coherent.eq(True)&w.first_ts.eq(expected)&w.last_ts.eq(pd.Series(idx+pd.Timedelta(minutes=59),index=idx));valid=bool(len(w)==24 and ok.all());dominant_hour=pd.NaT;dominant=latest=day_return=share=variation=np.nan
  if valid:
   returns=np.log(w.hour_close.to_numpy(float)/w.hour_open.to_numpy(float));absolute=np.abs(returns);total=float(absolute.sum());maximum=float(absolute.max());first=int(np.flatnonzero(absolute==maximum)[0]);last=int(np.flatnonzero(absolute==maximum)[-1]);dominant=float(returns[first]);latest=float(returns[last]);dominant_hour=idx[first];day_return=float(np.log(w.hour_close.iloc[-1]/w.hour_open.iloc[0]));share=maximum/total if total>0 else np.nan;variation=float(np.sqrt(np.square(returns).sum()));valid=bool(dominant!=0 and day_return!=0 and variation>0 and np.isfinite([dominant,latest,day_return,share,variation]).all())
  if not valid:dominant_hour=pd.NaT;dominant=latest=day_return=share=variation=np.nan
  rows.append({"source_day":d,"feature_available_time":d+pd.Timedelta(days=1),"source_valid":valid,"dominant_hour":dominant_hour,"dominant_return":dominant,"latest_dominant_return":latest,"day_return":day_return,"dominant_share":share,"realized_variation":variation})
 out=pd.DataFrame(rows);out["dominant_share_rank"]=prior_midrank(out.dominant_share.where(out.source_valid));out["variation_rank"]=prior_midrank(out.realized_variation.where(out.source_valid));return out[list(FEATURE_COLUMNS)]
def conditions(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=f.shift(1) if control=="one_day_stale_features" else f;valid=u.source_valid.eq(True)&u.dominant_return.ne(0);dg=pd.Series(True,index=f.index) if control=="no_dominant_share_gate" else u.dominant_share_rank.ge(.65);vg=pd.Series(True,index=f.index) if control=="no_variation_gate" else u.variation_rank.ge(.65);active=valid&dg&vg;basis=u.day_return if control=="completed_day_return_side" else u.latest_dominant_return if control=="latest_maximum_tie_break" else u.dominant_return;side=np.sign(basis).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=f.index)
 return active,side,u
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides,u=conditions(f,control);rows=[]
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"source_day"])+pd.Timedelta(days=1);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  source=u.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"source_day":source.source_day,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:source[c] for c in FEATURE_COLUMNS[3:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVDSR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);hours=load_hours();features=build_features(hours);primary=clock(features);controls={n:clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"hvdsr_12_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"table":"bars_binance","window":[START.isoformat(),END.isoformat()],"physical_rows_1m":int(pd.to_numeric(hours.source_rows).sum()),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features),"valid_rows":int(features.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"hvdsr_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
