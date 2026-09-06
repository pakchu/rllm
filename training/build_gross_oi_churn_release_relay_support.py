"""Build source-only GOICR-12 clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_gross_oi_churn_release_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="3acdda0abf96825f938e96c383649aa56c0cd146474cc9e17cfc947698747b1c";START=pd.Timestamp("2023-01-01T00:00Z");RAW_START=START-pd.Timedelta(hours=7);END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("net_oi_tail_only","gross_churn_without_cancellation","late_price_escape_only","one_decision_stale_churn","direction_flip")
BAR_QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,(array_agg(open ORDER BY ts))[1] AS bar_open,(array_agg(close ORDER BY ts DESC))[1] AS bar_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY bar_time ORDER BY bar_time"""
OI_QUERY="""SELECT ts,sum_open_interest,count(*) OVER (PARTITION BY ts) AS duplicate_count FROM open_interest_binance WHERE symbol='BTCUSDT' AND period='5m' AND ts>=:start AND ts<:end ORDER BY ts"""
SOURCE_DIR=Path("data/gross_oi_churn_release_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"three_hour_churn.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/gross_oi_churn_release_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/gross_oi_churn_release_relay_controls_2023_2026");RESULT=Path("results/gross_oi_churn_release_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","gross_churn","net_displacement","cancellation","late_escape","gross_rank","net_rank","cancellation_rank","late_escape_abs_rank")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*FEATURE_COLUMNS[3:])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def prior_midrank(v:pd.Series,lookback:int=720,minimum:int=480)->pd.Series:
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
def load_sources():
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:
   bars=pd.read_sql_query(text(BAR_QUERY),c,params={"start":RAW_START.to_pydatetime(),"end":END.to_pydatetime()});oi=pd.read_sql_query(text(OI_QUERY),c,params={"start":RAW_START.to_pydatetime(),"end":END.to_pydatetime()})
  return bars,oi
 finally:db.dispose()
def build_features(bars:pd.DataFrame,oi:pd.DataFrame)->pd.DataFrame:
 breq={"bar_time","bar_open","bar_close","source_rows","distinct_rows","first_ts","last_ts","coherent"};oreq={"ts","sum_open_interest","duplicate_count"}
 if not breq.issubset(bars) or not oreq.issubset(oi):raise ValueError("GOICR schema drift")
 b=bars.copy();o=oi.copy()
 for c in ("bar_time","first_ts","last_ts"):b[c]=pd.to_datetime(b[c],utc=True,errors="coerce")
 for c in ("bar_open","bar_close","source_rows","distinct_rows"):b[c]=pd.to_numeric(b[c],errors="coerce")
 o["ts"]=pd.to_datetime(o.ts,utc=True,errors="coerce");o["sum_open_interest"]=pd.to_numeric(o.sum_open_interest,errors="coerce");o["duplicate_count"]=pd.to_numeric(o.duplicate_count,errors="coerce")
 b=b.sort_values("bar_time",kind="mergesort").set_index("bar_time");o=o[o.duplicate_count.eq(1)].sort_values("ts",kind="mergesort").set_index("ts");rows=[]
 for d in pd.date_range(START+pd.Timedelta(hours=6),END,freq="3h",inclusive="left"):
  pidx=pd.date_range(d-pd.Timedelta(hours=1),d,freq="5min",inclusive="left");oidx=pd.date_range(d-pd.Timedelta(hours=6,minutes=5),d,freq="5min",inclusive="left");pw=b.reindex(pidx);ow=o.reindex(oidx)
  pok=np.isfinite(pw[["bar_open","bar_close","source_rows","distinct_rows"]]).all(axis=1)&pw.bar_open.gt(0)&pw.bar_close.gt(0)&pw.source_rows.eq(5)&pw.distinct_rows.eq(5)&pw.coherent.fillna(False).astype(bool)&pw.first_ts.eq(pd.Series(pidx,index=pidx))&pw.last_ts.eq(pd.Series(pidx+pd.Timedelta(minutes=4),index=pidx));ook=np.isfinite(ow[["sum_open_interest","duplicate_count"]]).all(axis=1)&ow.sum_open_interest.gt(0)&ow.duplicate_count.eq(1)
  valid=bool(pok.all()&ook.all())
  if valid:
   inc=np.diff(np.log(ow.sum_open_interest.to_numpy(float)));g=float(np.abs(inc).sum());n=float(abs(inc.sum()));c=1-n/g if g>0 else np.nan;r=float(np.log(pw.bar_close.iloc[-1]/pw.bar_open.iloc[0]));valid=bool(np.isfinite([g,n,c,r]).all() and g>0 and -1e-12<=c<=1+1e-12 and r!=0);c=float(np.clip(c,0,1)) if valid else np.nan
  else:g=n=c=r=np.nan
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":valid,"gross_churn":g,"net_displacement":n,"cancellation":c,"late_escape":r})
 out=pd.DataFrame(rows);out["gross_rank"]=prior_midrank(out.gross_churn.where(out.source_valid));out["net_rank"]=prior_midrank(out.net_displacement.where(out.source_valid));out["cancellation_rank"]=prior_midrank(out.cancellation.where(out.source_valid));out["late_escape_abs_rank"]=prior_midrank(out.late_escape.abs().where(out.source_valid));return out[list(FEATURE_COLUMNS)]
def onset(x:pd.Series)->pd.Series:return x.fillna(False)&~x.shift(1,fill_value=False)
def active_and_side(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 primary=f.source_valid&f.late_escape.ne(0)&f.gross_rank.ge(.65)&f.cancellation_rank.ge(.8)&f.late_escape_abs_rank.ge(.65);sidev=f.late_escape
 if control=="net_oi_tail_only":state=f.source_valid&f.late_escape.ne(0)&f.net_rank.ge(.65)&f.late_escape_abs_rank.ge(.65)
 elif control=="gross_churn_without_cancellation":state=f.source_valid&f.late_escape.ne(0)&f.gross_rank.ge(.65)&f.late_escape_abs_rank.ge(.65)
 elif control=="late_price_escape_only":state=f.source_valid&f.late_escape.ne(0)&f.late_escape_abs_rank.ge(.65)
 elif control=="one_decision_stale_churn":state=primary.shift(1,fill_value=False);sidev=f.late_escape.shift(1)
 else:state=primary
 active=onset(state);side=np.sign(sidev)
 if control=="direction_flip":side=-side
 return active&pd.Series(sidev,index=f.index).ne(0),pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides=active_and_side(f,control);rows=[];reserved=None
 for i in f.index[active&sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("GOICR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);bars,oi=load_sources();features=build_features(bars,oi);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"goicr_12_source_v1","query_sha256":{"bars":hashlib.sha256(BAR_QUERY.encode()).hexdigest(),"oi":hashlib.sha256(OI_QUERY.encode()).hexdigest()},"window":[RAW_START.isoformat(),END.isoformat()],"physical_rows":{"bars":int(bars.source_rows.sum()),"oi":len(oi)},"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"goicr_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
