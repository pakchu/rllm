"""Build outcome-blind source support for frozen HVEWCS-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_ewma_conditional_skew_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-02-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="504ce83c8f23ed94ddf20ae903c0cbab56392c147ef14363634e93caa6a97412"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_ewma_conditional_skew_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_ewma_conditional_skew_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_ewma_conditional_skew_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_ewma_conditional_skew_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_ewma_conditional_skew_relay_support_2026-08-11.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","daily_return_count","conditional_skew","skew_strength","skew_strength_rank","unweighted_skew","unweighted_skew_strength_rank","variation","variation_rank","eligible")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","conditional_skew","skew_strength","skew_strength_rank","unweighted_skew","unweighted_skew_strength_rank","variation","variation_rank","eligible")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_days"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_days"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def weighted_skew(values:np.ndarray)->float:
 x=np.asarray(values,float)
 if len(x)!=P["return_days"] or not np.isfinite(x).all():return math.nan
 age=np.arange(len(x)-1,-1,-1,dtype=float);weights=np.power(.5,age/P["weight_half_life_days"]);weights/=weights.sum();mean=float(np.sum(weights*x));variance=float(np.sum(weights*np.square(x-mean)))
 return float(np.sum(weights*np.power(x-mean,3))/variance**1.5) if variance>0 else math.nan

def unweighted_skew(values:np.ndarray)->float:
 x=np.asarray(values,float)
 if len(x)!=P["return_days"] or not np.isfinite(x).all():return math.nan
 mean=float(x.mean());variance=float(np.mean(np.square(x-mean)))
 return float(np.mean(np.power(x-mean,3))/variance**1.5) if variance>0 else math.nan

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
 if frame.columns.tolist()!=["ts","open","high","low","close"]:raise RuntimeError("HVEWCS source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVEWCS invalid source key")
 prices=x[["open","high","low","close"]]
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
 return x.set_index("ts").sort_index()

def previous_valid_onset(eligible:pd.Series,valid:pd.Series)->pd.Series:
 out=pd.Series(False,index=eligible.index);previous=None
 for i in eligible.index:
  if not bool(valid.at[i]):continue
  if bool(eligible.at[i]) and previous is not None:out.at[i]=not bool(eligible.at[previous])
  previous=i
 return out

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw).reindex(pd.date_range(START,END,freq="1min",inclusive="left"));source["block_end"]=(source.index-pd.Timedelta("2h")).floor("D")+pd.Timedelta("26h");grouped=source.groupby("block_end",sort=True);daily=pd.DataFrame({"rows":grouped.row_valid.sum(),"close":grouped.close.last()});daily["valid"]=daily.rows.eq(1440)&np.isfinite(daily.close)&daily.close.gt(0);consecutive=daily.index.to_series().diff().eq(pd.Timedelta("1d"));daily["return"]=np.log(daily.close/daily.close.shift(1)).where(daily.valid&daily.valid.shift(1,fill_value=False)&consecutive)
 returns=daily["return"];conditional=returns.rolling(P["return_days"],min_periods=P["return_days"]).apply(weighted_skew,raw=True);plain=returns.rolling(P["return_days"],min_periods=P["return_days"]).apply(unweighted_skew,raw=True);variation=np.sqrt(np.square(returns).rolling(P["variation_days"],min_periods=P["variation_days"]).sum());panel=pd.DataFrame({"decision_time":daily.index,"feature_available_time":daily.index,"daily_return_count":returns.notna().rolling(P["return_days"],min_periods=P["return_days"]).sum(),"conditional_skew":conditional,"unweighted_skew":plain,"variation":variation});panel=panel[(panel.decision_time>=START+pd.Timedelta("31d"))&(panel.decision_time<END)].reset_index(drop=True);panel["source_valid"]=panel.daily_return_count.eq(P["return_days"])&np.isfinite(panel[["conditional_skew","unweighted_skew","variation"]]).all(axis=1)&panel.conditional_skew.ne(0)&panel.unweighted_skew.ne(0)&panel.variation.gt(0);panel["skew_strength"]=panel.conditional_skew.abs();panel["skew_strength_rank"]=prior_rank(panel.skew_strength.where(panel.source_valid));panel["unweighted_skew_strength_rank"]=prior_rank(panel.unweighted_skew.abs().where(panel.source_valid));panel["variation_rank"]=prior_rank(panel.variation.where(panel.source_valid));panel["eligible"]=panel.source_valid&panel.skew_strength_rank.ge(P["skew_strength_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy();valid=used.source_valid.eq(True);strength=used.skew_strength_rank.ge(P["skew_strength_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);side=np.sign(pd.to_numeric(used.conditional_skew,errors="coerce")).fillna(0).astype(int);state=valid&side.ne(0)&strength&variation
 if control=="no_skew_strength":state=valid&side.ne(0)&variation
 elif control=="no_variation_gate":state=valid&side.ne(0)&strength
 elif control=="unweighted_skew":side=np.sign(pd.to_numeric(used.unweighted_skew,errors="coerce")).fillna(0).astype(int);state=valid&side.ne(0)&used.unweighted_skew_strength_rank.ge(P["skew_strength_rank_min"])&variation
 elif control=="one_day_stale_features":valid=valid.shift(1,fill_value=False);side=side.shift(1,fill_value=0);state=valid&side.ne(0)&strength.shift(1,fill_value=False)&variation.shift(1,fill_value=False)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=pd.Series(1,index=side.index,dtype=int)
 return state&side.ne(0),side,used

def build_clock(panel:pd.DataFrame,control:str="primary")->pd.DataFrame:
 act,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[act]:
  decision=pd.Timestamp(panel.at[i,"decision_time"]);entry=decision+pd.Timedelta("5m");exit_time=entry+pd.Timedelta("24h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  reserved=exit_time;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":pd.Timestamp(used.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),**{c:(bool(used.at[i,c]) if c=="eligible" else float(used.at[i,c])) for c in CLOCK_COLUMNS[8:]}})
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
def json_bytes(x):return (json.dumps(x,indent=2,allow_nan=False)+"\n").encode()

def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVEWCS prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvewcs_24_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvewcs_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
