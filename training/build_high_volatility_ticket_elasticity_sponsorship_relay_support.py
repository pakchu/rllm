"""Build outcome-blind source support for frozen HVTER-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-04-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="b5ae69a17caabfbac3eecb10e3fe9f204dc21ec8735eefa021ed8ec48aa110a5"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close,quote_asset_volume,number_of_trades FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_ticket_elasticity_sponsorship_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_ticket_elasticity_sponsorship_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_ticket_elasticity_sponsorship_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_ticket_elasticity_sponsorship_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_ticket_elasticity_sponsorship_relay_support_2026-08-10.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","ticket_elasticity","elasticity_rank","aggregate_average_ticket","average_ticket_rank","realized_variation","variation_rank","completed_return","final_two_hour_return","eligible","onset")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","ticket_elasticity","elasticity_rank","aggregate_average_ticket","average_ticket_rank","realized_variation","variation_rank","completed_return","final_two_hour_return")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_blocks"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_blocks"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def elasticity_statistics(block:pd.DataFrame)->tuple[float,float,float,float,float]:
 counts=block.number_of_trades.to_numpy(float);quote=block.quote_asset_volume.to_numpy(float)
 x=np.log1p(counts);y=np.log1p(quote);variance=float(np.var(x,ddof=0))
 if variance<=0 or float(counts.sum())<=0:return (math.nan,)*5
 elasticity=float(np.mean((x-x.mean())*(y-y.mean()))/variance)
 average_ticket=float(quote.sum()/counts.sum())
 returns=np.log(block.close.to_numpy(float)/block.open.to_numpy(float))
 variation=float(np.square(returns).sum())
 completed=float(math.log(block.close.iloc[-1]/block.open.iloc[0]))
 final_two_hour=float(math.log(block.close.iloc[-1]/block.open.iloc[-120]))
 return elasticity,average_ticket,variation,completed,final_two_hour

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
 if frame.columns.tolist()!=["ts","open","high","low","close","quote_asset_volume","number_of_trades"]:raise RuntimeError("HVTER source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close","quote_asset_volume","number_of_trades"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVTER invalid source key")
 prices=x[["open","high","low","close"]]
 integer_counts=np.isfinite(x.number_of_trades)&x.number_of_trades.ge(0)&x.number_of_trades.eq(np.floor(x.number_of_trades))
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&np.isfinite(x.quote_asset_volume)&x.quote_asset_volume.ge(0)&integer_counts&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
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
 for decision in pd.date_range(START+pd.Timedelta("6h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");block=source.reindex(minutes);count=int(block.row_valid.eq(True).sum())
  path_valid=len(block)==480 and bool(block.row_valid.eq(True).all())
  if path_valid:
   elasticity,average_ticket,var,ret,final_ret=elasticity_statistics(block);valid=all(math.isfinite(v) for v in (elasticity,average_ticket,var,ret,final_ret)) and var>0 and elasticity>P["elasticity_floor"] and ret!=0 and final_ret!=0
  else:elasticity=average_ticket=var=ret=final_ret=math.nan;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"ticket_elasticity":elasticity,"aggregate_average_ticket":average_ticket,"realized_variation":var,"completed_return":ret,"final_two_hour_return":final_ret})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["elasticity_rank"]=prior_rank(panel.ticket_elasticity.where(valid));panel["average_ticket_rank"]=prior_rank(panel.aggregate_average_ticket.where(valid));panel["variation_rank"]=prior_rank(panel.realized_variation.where(valid));agreement=np.sign(panel.completed_return).eq(np.sign(panel.final_two_hour_return));panel["eligible"]=valid&panel.elasticity_rank.ge(P["elasticity_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"])&agreement;panel["onset"]=previous_valid_onset(panel.eligible,valid);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_boundary_stale_elasticity":
  cols=["source_valid","ticket_elasticity","elasticity_rank","aggregate_average_ticket","average_ticket_rank","variation_rank","completed_return","final_two_hour_return","feature_available_time"];used[cols]=panel[cols].shift(1)
 valid=used.source_valid.eq(True);strong=used.elasticity_rank.ge(P["elasticity_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);agreement=np.sign(used.completed_return).eq(np.sign(used.final_two_hour_return));state=valid&strong&variation&agreement
 if control=="no_elasticity_tail":state=valid&variation&agreement
 elif control=="no_variation_gate":state=valid&strong&agreement
 elif control=="aggregate_average_ticket_tail":state=valid&used.average_ticket_rank.ge(P["elasticity_rank_min"])&variation&agreement
 onset=previous_valid_onset(state,valid);side=np.sign(pd.to_numeric(used.completed_return,errors="coerce").fillna(0)).astype(int)
 if control=="direction_flip":side=-side
 return onset&side.ne(0),side,used

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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVTER prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvter_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvter_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
