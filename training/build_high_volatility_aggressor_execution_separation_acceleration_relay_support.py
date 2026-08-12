"""Build outcome-blind source support for frozen HVAESAR-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_aggressor_execution_separation_acceleration_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-01-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="28cf02a7e5f6677e216c553a1951c6cd702233f3c9d6d07db3316c5835861e89"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close,volume,quote_asset_volume,taker_buy_base,taker_buy_quote FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_aggressor_execution_separation_acceleration_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_aggressor_execution_separation_acceleration_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_aggressor_execution_separation_acceleration_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_aggressor_execution_separation_acceleration_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_aggressor_execution_separation_acceleration_relay_support_2026-08-12.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","first_separation","second_separation","separation_side","separation_ratio","first_magnitude_rank","first_flow_imbalance","second_flow_imbalance","flow_side","flow_ratio","realized_variation","variation_rank","eligible","onset")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*PANEL_COLUMNS[4:15])

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(values),np.nan);history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_decisions"]:out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=series.index)
def previous_valid_onset(state:pd.Series,valid:pd.Series)->pd.Series:
 out=pd.Series(False,index=state.index);previous=None
 for i in state.index:
  if not bool(valid.at[i]):continue
  if bool(state.at[i]) and previous is not None:out.at[i]=not bool(state.at[previous])
  previous=i
 return out
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def prepare(frame:pd.DataFrame)->pd.DataFrame:
 expected=["ts","open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]
 if frame.columns.tolist()!=expected:raise RuntimeError("HVAESAR source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in expected[1:]:x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVAESAR invalid source key")
 prices=x[["open","high","low","close"]];finite=np.isfinite(x[expected[1:]]).all(axis=1)
 x["row_valid"]=finite&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)&x[["volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]].ge(0).all(axis=1)&x.taker_buy_base.le(x.volume)&x.taker_buy_quote.le(x.quote_asset_volume)
 return x.set_index("ts").sort_index()
def half_stats(x:pd.DataFrame)->tuple[float,float]:
 buy_base=float(x.taker_buy_base.sum());buy_quote=float(x.taker_buy_quote.sum());sell_base=float((x.volume-x.taker_buy_base).sum());sell_quote=float((x.quote_asset_volume-x.taker_buy_quote).sum());total_quote=float(x.quote_asset_volume.sum())
 if min(buy_base,buy_quote,sell_base,sell_quote,total_quote)<=0:return math.nan,math.nan
 separation=float(np.log((buy_quote/buy_base)/(sell_quote/sell_base)));flow=float((2*buy_quote-total_quote)/total_quote)
 return separation,flow
def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("26h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("24h"),decision,freq="1min",inclusive="left");window=source.reindex(minutes);count=int(window.row_valid.eq(True).sum());path_valid=len(window)==1440 and bool(window.row_valid.eq(True).all())
  first=second=first_flow=second_flow=side=flow_side=ratio=flow_ratio=variation=math.nan;valid=False
  if path_valid:
   block=window.iloc[-480:];first,first_flow=half_stats(block.iloc[:240]);second,second_flow=half_stats(block.iloc[240:]);returns=np.log(window.close.to_numpy(float)/window.open.to_numpy(float));variation=float(np.square(returns).sum())
   finite=np.isfinite([first,second,first_flow,second_flow,variation]).all();side=float(np.sign(first)) if finite and first!=0 and np.sign(first)==np.sign(second) else 0.;flow_side=float(np.sign(first_flow)) if finite and first_flow!=0 and np.sign(first_flow)==np.sign(second_flow) else 0.;ratio=abs(second)/abs(first) if side!=0 else math.nan;flow_ratio=abs(second_flow)/abs(first_flow) if flow_side!=0 else math.nan;valid=bool(finite and first!=0 and second!=0 and variation>0)
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"first_separation":first,"second_separation":second,"separation_side":side,"separation_ratio":ratio,"first_flow_imbalance":first_flow,"second_flow_imbalance":second_flow,"flow_side":flow_side,"flow_ratio":flow_ratio,"realized_variation":variation})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["first_magnitude_rank"]=prior_rank(panel.first_separation.abs().where(valid));panel["variation_rank"]=prior_rank(panel.realized_variation.where(valid));panel["eligible"]=valid&panel.separation_side.ne(0)&panel.separation_ratio.ge(P["minimum_acceleration_ratio"])&panel.first_magnitude_rank.ge(P["first_half_magnitude_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);panel["onset"]=previous_valid_onset(panel.eligible,valid);return panel.loc[:,PANEL_COLUMNS]
def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_features":used[list(PANEL_COLUMNS[2:15])]=panel[list(PANEL_COLUMNS[2:15])].shift(1);used["feature_available_time"]=panel.feature_available_time.shift(1)
 valid=used.source_valid.eq(True);magnitude=used.first_magnitude_rank.ge(P["first_half_magnitude_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);accel=used.separation_side.ne(0)&used.separation_ratio.ge(P["minimum_acceleration_ratio"]);state=valid&magnitude&variation&accel;side=pd.to_numeric(used.separation_side,errors="coerce").fillna(0).astype(int)
 if control=="no_acceleration_requirement":state=valid&magnitude&variation&used.separation_side.ne(0)
 elif control=="no_variation_gate":state=valid&magnitude&accel
 elif control=="flow_imbalance_acceleration_instead_of_execution_separation":state=valid&variation&used.flow_side.ne(0)&used.flow_ratio.ge(P["minimum_acceleration_ratio"]);side=pd.to_numeric(used.flow_side,errors="coerce").fillna(0).astype(int)
 onset=previous_valid_onset(state,valid)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
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
def stats(clock:pd.DataFrame,split:str)->dict[str,float|int]:
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 longs=int(x.side.eq(1).sum());shorts=int(x.side.eq(-1).sum());months=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":longs,"shorts":shorts,"minority_side_share":min(longs,shorts)/len(x),"max_month_share":int(months.max())/len(x)}
def csv_gz(frame:pd.DataFrame)->bytes:
 b=io.BytesIO();raw=frame.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(path:Path,content:bytes)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.read_bytes()!=content:raise RuntimeError(f"refusing overwrite {path}")
 path.write_bytes(content)
def json_bytes(x:Any)->bytes:return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVAESAR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvaesar_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvaesar_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
