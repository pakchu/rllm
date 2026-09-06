"""Materialize outcome-blind source support for frozen FGER-24."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from urllib.request import Request,urlopen
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from training import preregister_fear_greed_extremity_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";BUILDER=Path("training/build_fear_greed_extremity_reversal_support.py");PREREG_SHA="fe6727f99715485037c72586bff4b3995ffb225e843932b7cf4e00510cbfc8ac"
SOURCE_DIR=Path("data/fear_greed_extremity_reversal_sources_2023_2026");SENTIMENT=SOURCE_DIR/"fear_greed_daily.csv.gz";FEATURES=SOURCE_DIR/"fger_preentry_features.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/fear_greed_extremity_reversal_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/fear_greed_extremity_reversal_controls_2023_2026");RESULT=Path("results/fear_greed_extremity_reversal_support_2026-08-09.json")
SOURCE_START=pd.Timestamp("2023-03-31T00:00:00Z");SOURCE_END=pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","classification_text_only","one_day_stale_sentiment","direction_flip")
COLUMNS=("candidate","control","split","sentiment_date","decision_time","feature_available_time","entry_time","exit_time","side","fear_greed_value","value_classification","btc_realized_variation","btc_variation_rank")
QUERY="""SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=h[-lookback:]
  if np.isfinite(c) and len(q)>=minimum:
   a=np.asarray(q);o.at[i]=(np.count_nonzero(a<c)+.5*np.count_nonzero(a==c))/len(a)
  if np.isfinite(c):h.append(float(c))
 return o
def download_sentiment()->pd.DataFrame:
 req=Request(prereg.SOURCE_URL,headers={"User-Agent":"rllm-fger-source/1.0","Accept":"application/json"})
 with urlopen(req,timeout=60) as r:raw=r.read()
 payload=json.loads(raw);rows=payload.get("data")
 if not isinstance(rows,list) or not rows:raise RuntimeError("FGER sentiment data missing")
 out=[]
 for row in rows:
  if not {"value","value_classification","timestamp"}.issubset(row):raise RuntimeError("FGER sentiment schema drift")
  value=int(row["value"]);stamp=pd.to_datetime(int(row["timestamp"]),unit="s",utc=True)
  if not 0<=value<=100 or stamp!=stamp.normalize():raise RuntimeError("FGER sentiment value or timestamp invalid")
  classification=str(row["value_classification"]).strip()
  if classification not in {"Extreme Fear","Fear","Neutral","Greed","Extreme Greed"}:raise RuntimeError("FGER classification invalid")
  out.append({"sentiment_date":stamp,"fear_greed_value":value,"value_classification":classification})
 f=pd.DataFrame(out).sort_values("sentiment_date").reset_index(drop=True)
 if f.sentiment_date.duplicated().any():raise RuntimeError("FGER duplicate sentiment date")
 return f[(f.sentiment_date>=SOURCE_START-pd.Timedelta(days=1))&(f.sentiment_date<SOURCE_END)].reset_index(drop=True)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_bars()->pd.DataFrame:
 from sqlalchemy import text
 e=postgres_engine()
 try:f=pd.read_sql_query(text(QUERY),e,params={"start":SOURCE_START.to_pydatetime(),"end":SOURCE_END.to_pydatetime()})
 finally:e.dispose()
 if f.columns.tolist()!=["ts","open","close"]:raise RuntimeError("FGER BTC schema drift")
 f.ts=pd.to_datetime(f.ts,utc=True,errors="raise");f=f.sort_values("ts").reset_index(drop=True);expected=pd.date_range(SOURCE_START,SOURCE_END,freq="1min",inclusive="left")
 if len(f)!=len(expected) or not f.ts.equals(pd.Series(expected,name="ts")):raise RuntimeError("FGER BTC source is not exact 1m grid")
 for c in ("open","close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 if not np.isfinite(f[["open","close"]]).all(axis=None) or not f[["open","close"]].gt(0).all(axis=None):raise RuntimeError("FGER invalid BTC price")
 return f.set_index("ts")
def build_features(sentiment:pd.DataFrame,bars:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for x in sentiment.itertuples(index=False):
  decision=pd.Timestamp(x.sentiment_date)+pd.Timedelta(days=1)
  if decision<=SOURCE_START or decision>SOURCE_END:continue
  w=bars.loc[decision-pd.Timedelta(days=1):decision-pd.Timedelta(minutes=1)]
  variation=float(np.sqrt(np.square(np.log(w.close.to_numpy()/w.open.to_numpy())).sum())) if len(w)==1440 else np.nan
  rows.append({"sentiment_date":x.sentiment_date,"decision_time":decision,"fear_greed_value":int(x.fear_greed_value),"value_classification":x.value_classification,"btc_realized_variation":variation})
 f=pd.DataFrame(rows);f["btc_variation_rank"]=strict_prior_midrank(f.btc_realized_variation);return f
def signal(f:pd.DataFrame,control:str)->pd.Series:
 value=f.fear_greed_value;side=pd.Series(np.where(value<=25,1,np.where(value>=75,-1,0)),index=f.index,dtype=int);eligible=side.ne(0)&f.btc_variation_rank.ge(.65)
 if control=="no_volatility_gate":eligible=side.ne(0)
 elif control=="classification_text_only":side=pd.Series(np.where(f.value_classification.eq("Extreme Fear"),1,np.where(f.value_classification.eq("Extreme Greed"),-1,0)),index=f.index,dtype=int);eligible=side.ne(0)&f.btc_variation_rank.ge(.65)
 elif control=="one_day_stale_sentiment":
  stale=value.shift(1);side=pd.Series(np.where(stale<=25,1,np.where(stale>=75,-1,0)),index=f.index,dtype=int);eligible=side.ne(0)&f.btc_variation_rank.ge(.65)
 side=side.where(eligible,0);return -side if control=="direction_flip" else side
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 sides=signal(f,control);rows=[];next_allowed=None
 for i in f.index[sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_time=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  next_allowed=exit_time;rows.append({"candidate":"FGER-24","control":control,"split":split,"sentiment_date":pd.Timestamp(f.at[i,"sentiment_date"]),"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_time,"side":int(sides.at[i]),"fear_greed_value":int(f.at[i,"fear_greed_value"]),"value_classification":f.at[i,"value_classification"],"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_variation_rank":float(f.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,Any]:
 x=c[c.split.eq(s)].copy()
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 e=pd.to_datetime(x.entry_time,utc=True);lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(e.dt.strftime("%Y-%m").value_counts().max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("FGER prereg hash drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);sent=download_sentiment();bars=load_bars();features=build_features(sent,bars);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(sent,SENTIMENT);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"fger_24_sources_v1","sentiment_url":prereg.SOURCE_URL,"btc_query":QUERY,"btc_window":[SOURCE_START.isoformat(),SOURCE_END.isoformat()],"btc_rows":len(bars),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"outputs":{"sentiment":{"path":str(SENTIMENT),"sha256":sha(SENTIMENT),"rows":len(sent)},"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)}},"candidate_outcomes_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,ensure_ascii=False,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM_EVENTS[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"fger_24_source_support_v1","policy_id":"FGER-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
