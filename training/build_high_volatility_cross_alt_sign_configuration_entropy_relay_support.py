"""Outcome-blind source support for frozen HVCASCE-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_cross_alt_sign_configuration_entropy_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="421932e169aebd1990eb313abc754c92225e92f1fc274058c51fdeac0b14d611";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"]);SYMBOLS=("BTCUSDT",*prereg.ALTS)
QUERY="""SELECT date_bin('15 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_start,symbol,(array_agg(open ORDER BY ts))[1] AS open,(array_agg(close ORDER BY ts DESC))[1] AS close,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2 ORDER BY 1,2"""
ROOT=Path("data/high_volatility_cross_alt_sign_configuration_entropy_relay_sources_2023_2026");PANEL=ROOT/"half_hour_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_cross_alt_sign_configuration_entropy_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_cross_alt_sign_configuration_entropy_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_cross_alt_sign_configuration_entropy_relay_controls_2023_2026");RESULT=Path("results/high_volatility_cross_alt_sign_configuration_entropy_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","configuration_entropy","collapse_rank","btc_binary_entropy","final_hour_side","final_hour_breadth","btc_realized_variation","variation_rank","eligible");CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","configuration_entropy","collapse_rank","btc_binary_entropy","final_hour_breadth","btc_realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def causal(series):
 values=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(values),np.nan);history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_decisions"]:out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=series.index)
def entropy(states):
 _,counts=np.unique(np.asarray(states),return_counts=True);prob=counts/counts.sum();return float(-np.sum(prob*np.log2(prob)))
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source():
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"symbols":list(SYMBOLS),"start":START,"end":END})
 finally:db.dispose()
def prepare(raw):
 required=["bar_start","symbol","open","close","minute_squared_return","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=required:raise RuntimeError("HVCASCE source schema drift")
 x=raw.copy()
 for c in ("bar_start","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 for c in ("open","close","minute_squared_return","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x[["bar_start","symbol"]].isna().any().any() or x.duplicated(["bar_start","symbol"]).any():raise RuntimeError("HVCASCE invalid key")
 x["row_valid"]=np.isfinite(x[["open","close","minute_squared_return","source_rows","distinct_rows"]]).all(axis=1)&x.open.gt(0)&x.close.gt(0)&x.minute_squared_return.ge(0)&x.source_rows.eq(15)&x.distinct_rows.eq(15)&x.first_ts.eq(x.bar_start)&x.last_ts.eq(x.bar_start+pd.Timedelta("14m"))&x.coherent.eq(True);x["ret"]=np.log(x.close/x.open);return x.set_index(["bar_start","symbol"]).sort_index()
def window_metrics(block):
 ret=block.ret.unstack("symbol").reindex(columns=SYMBOLS);valid=block.row_valid.unstack("symbol").reindex(columns=SYMBOLS)
 if len(ret)!=96 or not bool(valid.eq(True).all().all()):return False,math.nan,math.nan,0,0,math.nan
 alt=ret.loc[:,prereg.ALTS].iloc[-24:];states=np.sign(alt.to_numpy(float)).sum(axis=1);config=entropy(states);btc_sign=np.sign(ret.BTCUSDT.iloc[-24:].to_numpy(float));btc_entropy=entropy(btc_sign);final=alt.iloc[-4:].sum(axis=0);sign=np.sign(final.to_numpy(float));positive=int(np.sum(sign>0));negative=int(np.sum(sign<0));side=1 if positive>negative else -1 if negative>positive else 0;breadth=max(positive,negative);variation=float(np.sqrt(block.xs("BTCUSDT",level="symbol").minute_squared_return.sum()));ok=math.isfinite(config) and math.isfinite(btc_entropy) and variation>0;return ok,config,btc_entropy,side,breadth,variation
def build_panel(raw):
 source=prepare(raw);rows=[]
 for d in pd.date_range(START+pd.Timedelta("24h"),END,freq="30min",inclusive="left"):
  times=pd.date_range(d-pd.Timedelta("24h"),d,freq="15min",inclusive="left");block=source.reindex(pd.MultiIndex.from_product([times,SYMBOLS],names=["bar_start","symbol"]));ok,config,btc_entropy,side,breadth,var=window_metrics(block);rows.append({"decision_time":d,"feature_available_time":d,"source_valid":ok,"configuration_entropy":config,"btc_binary_entropy":btc_entropy,"final_hour_side":side,"final_hour_breadth":breadth,"btc_realized_variation":var})
 panel=pd.DataFrame(rows);valid=panel.source_valid;panel["collapse_rank"]=causal((-panel.configuration_entropy).where(valid));panel["variation_rank"]=causal(panel.btc_realized_variation.where(valid));panel["eligible"]=valid&panel.final_hour_breadth.ge(P["minimum_directional_breadth"])&panel.final_hour_side.ne(0)&panel.collapse_rank.ge(P["collapse_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]
def onset(state,valid):
 out=pd.Series(False,index=state.index);prior=None
 for i in state.index:
  if not bool(valid.at[i]):continue
  if bool(state.at[i]) and prior is not None:out.at[i]=not bool(state.at[prior])
  prior=i
 return out
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_hour_stale_entropy":used[["configuration_entropy","collapse_rank","feature_available_time"]]=panel[["configuration_entropy","collapse_rank","feature_available_time"]].shift(2)
 strength=used.collapse_rank.ge(P["collapse_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);state=used.source_valid&used.final_hour_breadth.ge(P["minimum_directional_breadth"])&used.final_hour_side.ne(0)&strength&variation
 if control=="no_collapse_tail":state=used.source_valid&used.final_hour_breadth.ge(P["minimum_directional_breadth"])&used.final_hour_side.ne(0)&variation
 elif control=="no_variation_gate":state=used.source_valid&used.final_hour_breadth.ge(P["minimum_directional_breadth"])&used.final_hour_side.ne(0)&strength
 side=used.final_hour_side.astype(int)
 if control=="btc_binary_sign_entropy":state=state&used.btc_binary_entropy.le(used.configuration_entropy);side=side
 elif control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return onset(state,used.source_valid),side,used
def build_clock(panel,control="primary"):
 selected,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[selected]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=P["entry_delay_minutes"]);exit_=entry+pd.Timedelta(hours=P["hold_hours"])
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":used.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(used.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCASCE prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f"{n}.csv.gz",common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f"{n}.csv.gz",common.csv_gz(x))
 source_core={"protocol_version":"hvcasce_8_sources_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"table":"bars_binance","symbols":list(SYMBOLS),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":prereg.canonical_hash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcasce_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};common.immutable(RESULT,common.json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":(r:=run())["support_passed"],"support":r["support"]},indent=2))
