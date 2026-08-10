"""Build outcome-blind source support for frozen HVDHSVPR-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_dual_half_signed_volume_persistence_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-04-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="ba0915e2efc25765aaa45a1279376d7dfa43aa93a232a4949e9f5cd6f0a1655f"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close,quote_asset_volume FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_dual_half_signed_volume_persistence_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_dual_half_signed_volume_persistence_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_dual_half_signed_volume_persistence_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_dual_half_signed_volume_persistence_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_dual_half_signed_volume_persistence_relay_support_2026-08-10.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","first_half_balance","second_half_balance","weak_half_strength","strength_rank","full_block_balance","full_balance_rank","first_half_sign_breadth","second_half_sign_breadth","equal_weight_strength","equal_weight_rank","realized_variation","variation_rank","eligible")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","first_half_balance","second_half_balance","weak_half_strength","strength_rank","full_block_balance","full_balance_rank","first_half_sign_breadth","second_half_sign_breadth","equal_weight_strength","equal_weight_rank","realized_variation","variation_rank")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_blocks"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_blocks"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def persistence_statistics(block:pd.DataFrame)->tuple[float,float,float,float,float,float,float,float]:
 returns=np.log(block.close.to_numpy(float)/block.open.to_numpy(float));signs=np.sign(returns);turnover=block.quote_asset_volume.to_numpy(float);first=slice(0,240);second=slice(240,480)
 q1=float(turnover[first].sum());q2=float(turnover[second].sum());first_balance=float(np.sum(turnover[first]*signs[first])/q1);second_balance=float(np.sum(turnover[second]*signs[second])/q2);weak=min(abs(first_balance),abs(second_balance));full=float(np.sum(turnover*signs)/turnover.sum());first_breadth=float(np.mean(signs[first]));second_breadth=float(np.mean(signs[second]));equal_weak=min(abs(first_breadth),abs(second_breadth));variation=float(np.sqrt(np.square(returns).sum()))
 return first_balance,second_balance,weak,full,first_breadth,second_breadth,equal_weak,variation

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
 if frame.columns.tolist()!=["ts","open","high","low","close","quote_asset_volume"]:raise RuntimeError("HVDHSVPR source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close","quote_asset_volume"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVDHSVPR invalid source key")
 prices=x[["open","high","low","close"]]
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&np.isfinite(x.quote_asset_volume)&x.quote_asset_volume.ge(0)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
 return x.set_index("ts").sort_index()

def previous_valid_onset(eligible:pd.Series,valid:pd.Series)->pd.Series:
 out=pd.Series(False,index=eligible.index);previous=None
 for i in eligible.index:
  if not bool(valid.at[i]):continue
  if bool(eligible.at[i]) and previous is not None:out.at[i]=not bool(eligible.at[previous])
  previous=i
 return out

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("2h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");block=source.reindex(minutes);count=int(block.row_valid.eq(True).sum())
  path_valid=len(block)==480 and bool(block.row_valid.eq(True).all())
  if path_valid:
   first_balance,second_balance,weak,full,first_breadth,second_breadth,equal_weak,var=persistence_statistics(block);valid=all(math.isfinite(v) for v in (first_balance,second_balance,weak,full,first_breadth,second_breadth,equal_weak,var)) and var>0
  else:first_balance=second_balance=weak=full=first_breadth=second_breadth=equal_weak=var=math.nan;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"first_half_balance":first_balance,"second_half_balance":second_balance,"weak_half_strength":weak,"full_block_balance":full,"first_half_sign_breadth":first_breadth,"second_half_sign_breadth":second_breadth,"equal_weight_strength":equal_weak,"realized_variation":var})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["strength_rank"]=prior_rank(panel.weak_half_strength.where(valid));panel["full_balance_rank"]=prior_rank(panel.full_block_balance.abs().where(valid));panel["equal_weight_rank"]=prior_rank(panel.equal_weight_strength.where(valid));panel["variation_rank"]=prior_rank(panel.realized_variation.where(valid));persistent=panel.first_half_balance.mul(panel.second_half_balance).gt(0);panel["eligible"]=valid&persistent&panel.strength_rank.ge(P["strength_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_participation":
  current_valid=panel.source_valid.eq(True);cols=["first_half_balance","second_half_balance","weak_half_strength","strength_rank","full_block_balance","full_balance_rank","first_half_sign_breadth","second_half_sign_breadth","equal_weight_strength","equal_weight_rank","feature_available_time"];used[cols]=panel[cols].shift(1);used["source_valid"]=current_valid&panel.source_valid.shift(1).fillna(False)
 valid=used.source_valid.eq(True);persistent=used.first_half_balance.mul(used.second_half_balance).gt(0);strong=used.strength_rank.ge(P["strength_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);state=valid&persistent&strong&variation
 if control=="no_weak_half_rank":state=valid&persistent&variation
 elif control=="no_variation_gate":state=valid&persistent&strong
 elif control=="full_block_signed_volume":state=valid&used.full_block_balance.ne(0)&used.full_balance_rank.ge(P["strength_rank_min"])&variation
 elif control=="equal_weight_half_sign_breadth":state=valid&used.first_half_sign_breadth.mul(used.second_half_sign_breadth).gt(0)&used.equal_weight_rank.ge(P["strength_rank_min"])&variation
 active=state;side=np.sign(pd.to_numeric(used.first_half_balance,errors="coerce").fillna(0)).astype(int)
 if control=="full_block_signed_volume":side=np.sign(pd.to_numeric(used.full_block_balance,errors="coerce").fillna(0)).astype(int)
 if control=="equal_weight_half_sign_breadth":side=np.sign(pd.to_numeric(used.first_half_sign_breadth,errors="coerce").fillna(0)).astype(int)
 if control=="direction_flip":side=-side
 if control=="same_clock_forced_long":side=pd.Series(1,index=side.index,dtype=int)
 return active&side.ne(0),side,used

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
def json_bytes(x):return (json.dumps(x,indent=2,allow_nan=False)+"\n").encode()

def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVDHSVPR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvdhsvpr_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvdhsvpr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
