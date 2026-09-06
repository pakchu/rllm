"""Build outcome-blind source support for frozen HVSFL-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_spot_first_passage_leadership_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-04-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="e19fd07bebc29c3594c65dfd8c435817e00d801bc9d9bcfa3b85bd9d12b6357d"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT 'perpetual' AS venue,ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
UNION ALL SELECT 'spot' AS venue,ts,open,high,low,close FROM bars_binance_spot WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY venue,ts"""
ROOT=Path("data/high_volatility_spot_first_passage_leadership_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_spot_first_passage_leadership_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_spot_first_passage_leadership_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_spot_first_passage_leadership_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_spot_first_passage_leadership_relay_support_2026-08-13.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","spot_minute_count","perpetual_minute_count","barrier","spot_upper_index","spot_lower_index","perpetual_upper_index","perpetual_lower_index","spot_first_side","spot_lead_bars","perpetual_first_side","perpetual_lead_bars","current_variation","variation_rank","eligible")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","barrier","spot_upper_index","spot_lower_index","perpetual_upper_index","perpetual_lower_index","spot_first_side","spot_lead_bars","perpetual_first_side","perpetual_lead_bars","current_variation","variation_rank")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_decisions"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def passage_indices(closes:np.ndarray,reference:float,barrier:float)->tuple[int,int]:
 values=np.log(np.asarray(closes,float)/reference);up=np.flatnonzero(values>=barrier);down=np.flatnonzero(values<=-barrier)
 return (int(up[0]) if len(up) else -1,int(down[0]) if len(down) else -1)

def first_side(up:int,down:int)->int:return 1 if 0<=up<down or up>=0 and down<0 else -1 if 0<=down<up or down>=0 and up<0 else 0

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

def prepare(frame:pd.DataFrame)->dict[str,pd.DataFrame]:
 if frame.columns.tolist()!=["venue","ts","open","high","low","close"]:raise RuntimeError("HVSFL source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.duplicated(["venue","ts"]).any() or set(x.venue)!={"spot","perpetual"}:raise RuntimeError("HVSFL invalid source key")
 prices=x[["open","high","low","close"]];x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
 x["minute_return"]=np.log(x.close/x.open);return {v:g.drop(columns="venue").set_index("ts").sort_index() for v,g in x.groupby("venue",sort=False)}

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("34h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("32h"),decision,freq="1min",inclusive="left");spot=source["spot"].reindex(minutes);perpetual=source["perpetual"].reindex(minutes);spot_count=int(spot.row_valid.eq(True).sum());perpetual_count=int(perpetual.row_valid.eq(True).sum())
  path_valid=len(spot)==len(perpetual)==1920 and bool(spot.row_valid.eq(True).all()) and bool(perpetual.row_valid.eq(True).all())
  if path_valid:
   prior=spot.iloc[:1440].close.to_numpy(float).reshape(288,5)[:,-1];spot_now=spot.iloc[1440:];perp_now=perpetual.iloc[1440:];spot_close=spot_now.close.to_numpy(float).reshape(96,5)[:,-1];perp_close=perp_now.close.to_numpy(float).reshape(96,5)[:,-1];barrier=math.sqrt(float(np.square(np.diff(np.log(prior))).sum())/P["prior_scale_hours"]);spot_up,spot_down=passage_indices(spot_close,float(spot_now.open.iloc[0]),barrier);perp_up,perp_down=passage_indices(perp_close,float(perp_now.open.iloc[0]),barrier);spot_side=first_side(spot_up,spot_down);perp_side=first_side(perp_up,perp_down);spot_pass=spot_up if spot_side==1 else spot_down if spot_side==-1 else -1;perp_confirm=perp_up if spot_side==1 else perp_down if spot_side==-1 else -1;perp_opposite=perp_down if spot_side==1 else perp_up if spot_side==-1 else -1;spot_lead=perp_confirm-spot_pass if spot_pass>=0 and perp_confirm>=0 and (perp_opposite<0 or perp_confirm<perp_opposite) else -1;perp_pass=perp_up if perp_side==1 else perp_down if perp_side==-1 else -1;spot_confirm=spot_up if perp_side==1 else spot_down if perp_side==-1 else -1;spot_opposite=spot_down if perp_side==1 else spot_up if perp_side==-1 else -1;perp_lead=spot_confirm-perp_pass if perp_pass>=0 and spot_confirm>=0 and (spot_opposite<0 or spot_confirm<spot_opposite) else -1;variation=math.sqrt(float(np.square(np.diff(np.log(perp_close))).sum()));valid=barrier>0 and variation>0
  else:barrier=variation=math.nan;spot_up=spot_down=perp_up=perp_down=-1;spot_side=perp_side=0;spot_lead=perp_lead=-1;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"spot_minute_count":spot_count,"perpetual_minute_count":perpetual_count,"barrier":barrier,"spot_upper_index":spot_up,"spot_lower_index":spot_down,"perpetual_upper_index":perp_up,"perpetual_lower_index":perp_down,"spot_first_side":spot_side,"spot_lead_bars":spot_lead,"perpetual_first_side":perp_side,"perpetual_lead_bars":perp_lead,"current_variation":variation})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["variation_rank"]=prior_rank(panel.current_variation.where(valid));panel["eligible"]=valid&panel.spot_first_side.ne(0)&panel.spot_lead_bars.ge(P["minimum_lead_bars"])&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 valid=used.source_valid.eq(True);variation=used.variation_rank.ge(P["variation_rank_min"]);side=pd.to_numeric(used.spot_first_side,errors="coerce").fillna(0).astype(int);state=valid&side.ne(0)&used.spot_lead_bars.ge(P["minimum_lead_bars"])&variation
 if control=="no_variation_gate":state=valid&side.ne(0)&used.spot_lead_bars.ge(P["minimum_lead_bars"])
 elif control=="one_bar_minimum_lead":state=valid&side.ne(0)&used.spot_lead_bars.ge(1)&variation
 elif control=="perpetual_first_passage":side=pd.to_numeric(used.perpetual_first_side,errors="coerce").fillna(0).astype(int);state=valid&side.ne(0)&used.perpetual_lead_bars.ge(P["minimum_lead_bars"])&variation
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVSFL prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvsfl_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvsfl_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
