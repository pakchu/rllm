"""Build source-only RBEFR-8 clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_range_body_efficiency_fracture_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="a3611f577e227a2aba9fb6affb13fabe906d42b0da10c2995c78a54c6e9f5e72";START=pd.Timestamp("2023-04-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("raw_range_variance_tail","inverse_body_variance_tail","one_decision_stale_fracture","direction_flip")
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,(array_agg(open ORDER BY ts))[1] AS bar_open,max(high) AS bar_high,min(low) AS bar_low,(array_agg(close ORDER BY ts DESC))[1] AS bar_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY bar_time ORDER BY bar_time"""
SOURCE_DIR=Path("data/range_body_efficiency_fracture_reversal_sources_2023_2026");FEATURES=SOURCE_DIR/"two_hour_fracture.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/range_body_efficiency_fracture_reversal_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/range_body_efficiency_fracture_reversal_controls_2023_2026");RESULT=Path("results/range_body_efficiency_fracture_reversal_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","range_variance","body_variance","fracture","impulse","fracture_rank","raw_range_rank","inverse_body_rank")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","range_variance","body_variance","fracture","impulse","fracture_rank","raw_range_rank","inverse_body_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(values:pd.Series,lookback:int=1080,minimum:int=720)->pd.Series:
 out=pd.Series(np.nan,index=values.index,dtype=float);history=[]
 for i,current in pd.to_numeric(values,errors="coerce").items():
  prior=history[-lookback:]
  if math.isfinite(current) and len(prior)>=minimum:
   a=np.asarray(prior);out.at[i]=(np.sum(a<current)+.5*np.sum(a==current))/len(a)
  if math.isfinite(current):history.append(float(current))
 return out
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
 req=["bar_time","bar_open","bar_high","bar_low","bar_close","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if not set(req).issubset(raw.columns):raise ValueError("RBEFR schema drift")
 f=raw[req].copy()
 for c in ("bar_time","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,errors="coerce")
 for c in ("bar_open","bar_high","bar_low","bar_close","source_rows","distinct_rows"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values("bar_time",kind="mergesort").set_index("bar_time");rows=[]
 for d in pd.date_range(START+pd.Timedelta(hours=8),END,freq="2h",inclusive="left"):
  expected=pd.date_range(d-pd.Timedelta(hours=8),d,freq="5min",inclusive="left");w=f.reindex(expected);p=w[["bar_open","bar_high","bar_low","bar_close"]]
  ok=bool(np.isfinite(w[["bar_open","bar_high","bar_low","bar_close","source_rows","distinct_rows"]]).all(axis=1).all() and p.gt(0).all(axis=1).all() and w.source_rows.eq(5).all() and w.distinct_rows.eq(5).all() and w.coherent.fillna(False).astype(bool).all() and w.first_ts.equals(pd.Series(expected,index=expected)) and w.last_ts.equals(pd.Series(expected+pd.Timedelta(minutes=4),index=expected)))
  if ok:
   rv=float(np.square(np.log(w.bar_high/w.bar_low)).sum()/(4*np.log(2)));bv=float(np.square(np.log(w.bar_close/w.bar_open)).sum());frac=rv/bv if rv>0 and bv>0 else np.nan;imp=float(np.log(w.bar_close.iloc[-1]/w.bar_open.iloc[-24]));ok=bool(np.isfinite([rv,bv,frac,imp]).all() and frac>0 and imp!=0)
  else:rv=bv=frac=imp=np.nan
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":ok,"range_variance":rv,"body_variance":bv,"fracture":frac,"impulse":imp})
 out=pd.DataFrame(rows);out["fracture_rank"]=strict_prior_midrank(out.fracture.where(out.source_valid));out["raw_range_rank"]=strict_prior_midrank(out.range_variance.where(out.source_valid));out["inverse_body_rank"]=strict_prior_midrank((1/out.body_variance).where(out.source_valid));return out[list(FEATURE_COLUMNS)]
def crossing(rank:pd.Series)->pd.Series:return rank.ge(.85)&rank.shift(1).lt(.85)
def active_and_side(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 if control=="raw_range_variance_tail":active=f.source_valid&crossing(f.raw_range_rank);imp=f.impulse
 elif control=="inverse_body_variance_tail":active=f.source_valid&crossing(f.inverse_body_rank);imp=f.impulse
 elif control=="one_decision_stale_fracture":active=f.source_valid&crossing(f.fracture_rank.shift(1));imp=f.impulse.shift(1)
 else:active=f.source_valid&crossing(f.fracture_rank);imp=f.impulse
 side=-np.sign(imp) if control!="direction_flip" else np.sign(imp);return active&pd.Series(imp,index=f.index).ne(0),pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides=active_and_side(f,control);rows=[];reserved=None
 for i in f.index[active&sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":f.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:f.at[i,c] for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,float|int]:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("RBEFR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"rbefr_8_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":int(raw.source_rows.sum()),"aggregate_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":canonical_hash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"rbefr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
