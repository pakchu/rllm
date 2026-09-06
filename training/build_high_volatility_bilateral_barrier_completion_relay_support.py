"""Build outcome-blind source support for frozen HVBBC-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_bilateral_barrier_completion_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-04-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="aebc345976d7278affdd180446ff1ee75345803aeb7688b566225babaea8043d"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_bilateral_barrier_completion_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_bilateral_barrier_completion_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_bilateral_barrier_completion_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_bilateral_barrier_completion_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_bilateral_barrier_completion_relay_support_2026-08-13.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","threshold","upper_passage_index","lower_passage_index","second_passage_side","terminal_displacement","final_hour_return","terminal_confirmation","current_variation","variation_rank","eligible")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","threshold","upper_passage_index","lower_passage_index","second_passage_side","terminal_displacement","final_hour_return","terminal_confirmation","current_variation","variation_rank")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["variation_history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_decisions"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def bilateral_passages(closes:np.ndarray,reference:float,threshold:float)->tuple[int,int,int]:
 closes=np.asarray(closes,float)
 if closes.shape!=(96,) or not np.isfinite(closes).all() or np.any(closes<=0) or not math.isfinite(reference) or reference<=0 or not math.isfinite(threshold) or threshold<=0:return -1,-1,0
 upper=reference*math.exp(threshold);lower=reference*math.exp(-threshold)
 upper_hits=np.flatnonzero(closes>=upper);lower_hits=np.flatnonzero(closes<=lower)
 upper_index=int(upper_hits[0]) if len(upper_hits) else -1;lower_index=int(lower_hits[0]) if len(lower_hits) else -1
 side=1 if 0<=lower_index<upper_index else -1 if 0<=upper_index<lower_index else 0
 return upper_index,lower_index,side

def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})

def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c: frame=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally: db.dispose()
 return frame

def prepare(frame:pd.DataFrame)->pd.DataFrame:
 if frame.columns.tolist()!=["ts","open","high","low","close"]:raise RuntimeError("HVBBC source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVBBC invalid source key")
 prices=x[["open","high","low","close"]];x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
 x["minute_return"]=np.log(x.close/x.open);return x.set_index("ts").sort_index()

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("32h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("32h"),decision,freq="1min",inclusive="left");block=source.reindex(minutes);count=int(block.row_valid.eq(True).sum())
  path_valid=len(block)==1920 and bool(block.row_valid.eq(True).all())
  if path_valid:
   prior=block.iloc[:1440].close.to_numpy(float).reshape(288,5)[:,-1];current_minutes=block.iloc[1440:];current=current_minutes.close.to_numpy(float).reshape(96,5)[:,-1];reference=float(current_minutes.open.iloc[0])
   prior_returns=np.diff(np.log(prior));current_returns=np.diff(np.log(current));threshold=math.sqrt(float(np.square(prior_returns).sum())/24);variation=math.sqrt(float(np.square(current_returns).sum()));upper_index,lower_index,side=bilateral_passages(current,reference,threshold);terminal=float(np.log(current[-1]/reference));final_hour=float(np.log(current[-1]/current[-12]));confirmation=side!=0 and terminal*side>0 and final_hour*side>0;valid=threshold>0 and variation>0
  else:threshold=variation=terminal=final_hour=math.nan;upper_index=lower_index=-1;side=0;confirmation=False;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"threshold":threshold,"upper_passage_index":upper_index,"lower_passage_index":lower_index,"second_passage_side":side,"terminal_displacement":terminal,"final_hour_return":final_hour,"terminal_confirmation":confirmation,"current_variation":variation})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["variation_rank"]=prior_rank(panel.current_variation.where(valid));panel["eligible"]=valid&panel.second_passage_side.ne(0)&panel.terminal_confirmation.eq(True)&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_decision_stale_barrier":
  cols=["source_valid","threshold","upper_passage_index","lower_passage_index","second_passage_side","terminal_displacement","final_hour_return","terminal_confirmation","current_variation","variation_rank","eligible","feature_available_time"];used[cols]=panel[cols].shift(1)
 valid=used.source_valid.eq(True);variation=used.variation_rank.ge(P["variation_rank_min"]);side=pd.to_numeric(used.second_passage_side,errors="coerce").fillna(0).astype(int);state=valid&side.ne(0)&used.terminal_confirmation.eq(True)&variation
 if control=="no_variation_gate":state=valid&side.ne(0)&used.terminal_confirmation.eq(True)
 elif control=="no_terminal_confirmation":state=valid&side.ne(0)&variation
 elif control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return state&side.ne(0),side,used

def build_clock(panel:pd.DataFrame,control:str="primary")->pd.DataFrame:
 act,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[act]:
  decision=pd.Timestamp(panel.at[i,"decision_time"]);entry=decision+pd.Timedelta("5m");exit_time=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  reserved=exit_time;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":pd.Timestamp(used.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),**{c:float(used.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)

def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());months=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(months.max())/len(x)}

def csv_gz(frame):
 b=io.BytesIO();raw=frame.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(path,content):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.read_bytes()!=content:raise RuntimeError(f"refusing overwrite {path}")
 path.write_bytes(content)
def json_bytes(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()

def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVBBC prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvbbc_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvbbc_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
