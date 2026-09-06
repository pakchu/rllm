"""Outcome-blind source support for frozen HVCVCSS-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_cross_venue_cash_shock_sponsorship_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-05-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="da962421c683267355689aebca4b9da12fffa861a0ac6c8cbe8162f906f0ea66";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,sum(quote_asset_volume) AS quote_turnover,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND quote_asset_volume>=0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM {table} WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
ROOT=Path("data/high_volatility_cross_venue_cash_shock_sponsorship_relay_sources_2023_2026");PANEL=ROOT/"five_minute_cash_shock_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_cross_venue_cash_shock_sponsorship_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_cross_venue_cash_shock_sponsorship_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_cross_venue_cash_shock_sponsorship_relay_controls_2023_2026");RESULT=Path("results/high_volatility_cross_venue_cash_shock_sponsorship_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","side","spot_turnover","perp_turnover","spot_q75","spot_q95","perp_q75","perp_q95","cash_only_state","perp_only_state","btc_realized_variation","variation_rank","variation_active","eligible");CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","spot_turnover","perp_turnover","spot_q95","perp_q75","btc_realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def causal(series):
 values=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(values),np.nan);history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["variation_history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_variation_history_decisions"]:out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=series.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source():
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:
   perp=pd.read_sql_query(text(QUERY.format(table="bars_binance")),c,params={"start":START,"end":END});spot=pd.read_sql_query(text(QUERY.format(table="bars_binance_spot")),c,params={"start":START,"end":END})
  return perp,spot
 finally:db.dispose()
def prepare(raw,label):
 required=["date","open","high","low","close","quote_turnover","minute_squared_return","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=required:raise RuntimeError(f"HVCVCSS {label} schema drift")
 x=raw.copy()
 for c in ("date","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 for c in ("open","high","low","close","quote_turnover","minute_squared_return","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.date.isna().any() or x.date.duplicated().any():raise RuntimeError(f"HVCVCSS {label} invalid key")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(x[["open","high","low","close","quote_turnover","minute_squared_return"]]).all(axis=1)&z.gt(0).all(axis=1)&x.quote_turnover.gt(0)&x.minute_squared_return.ge(0);x["return"]=np.log(x.close/x.open);return x.set_index("date").sort_index()
def build_panel(perp_raw,spot_raw):
 grid=pd.date_range(START,END,freq="5min",inclusive="left");perp=prepare(perp_raw,"perp").reindex(grid);spot=prepare(spot_raw,"spot").reindex(grid);joint=perp.valid.eq(True)&spot.valid.eq(True);spot_turn=spot.quote_turnover.where(spot.valid);perp_turn=perp.quote_turnover.where(perp.valid);roll=lambda x,q:x.shift(1).rolling(P["history_bars"],min_periods=P["minimum_history_bars"]).quantile(q,interpolation="linear");sq75,sq95=roll(spot_turn,.75),roll(spot_turn,.95);pq75,pq95=roll(perp_turn,.75),roll(perp_turn,.95);same=np.sign(spot["return"])==np.sign(perp["return"]);side=np.sign(spot["return"]).fillna(0).astype(int).where(same,0);cash=joint&sq95.notna()&pq75.notna()&spot_turn.ge(sq95)&perp_turn.lt(pq75)&side.ne(0);perp_only=joint&pq95.notna()&sq75.notna()&perp_turn.ge(pq95)&spot_turn.lt(sq75)&side.ne(0);variation=np.sqrt(perp.minute_squared_return.where(perp.valid).rolling(P["variation_bars"],min_periods=P["variation_bars"]).sum());vrank=causal(variation.where(joint));source_valid=joint&sq75.notna()&sq95.notna()&pq75.notna()&pq95.notna()&variation.notna()&vrank.notna();vactive=vrank.ge(P["variation_rank_min"]);eligible=source_valid&source_valid.shift(1,fill_value=False)&cash&~cash.shift(1,fill_value=False)&vactive
 return pd.DataFrame({"decision_time":grid+pd.Timedelta("5m"),"feature_available_time":grid+pd.Timedelta("5m"),"source_valid":source_valid.to_numpy(bool),"side":side.to_numpy(int),"spot_turnover":spot_turn.to_numpy(float),"perp_turnover":perp_turn.to_numpy(float),"spot_q75":sq75.to_numpy(float),"spot_q95":sq95.to_numpy(float),"perp_q75":pq75.to_numpy(float),"perp_q95":pq95.to_numpy(float),"cash_only_state":cash.to_numpy(bool),"perp_only_state":perp_only.to_numpy(bool),"btc_realized_variation":variation.to_numpy(float),"variation_rank":vrank.to_numpy(float),"variation_active":vactive.fillna(False).to_numpy(bool),"eligible":eligible.to_numpy(bool)},columns=PANEL_COLUMNS)
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy();side=u.side.astype(int);valid=u.source_valid.eq(True);state=u.cash_only_state.eq(True)
 if control=="spot_shock_without_perp_quiet":state=u.spot_turnover.ge(u.spot_q95)&side.ne(0)
 elif control=="perp_shock_spot_quiet":state=u.perp_only_state.eq(True)
 elif control=="one_bar_stale_cash_shock":u[["cash_only_state","side","feature_available_time"]]=panel[["cash_only_state","side","feature_available_time"]].shift(1);side=pd.to_numeric(u.side,errors="coerce").fillna(0).astype(int);state=u.cash_only_state.fillna(False)
 selected=valid&valid.shift(1,fill_value=False)&state&~state.shift(1,fill_value=False)&u.variation_active.eq(True)&side.ne(0)
 if control=="no_variation_gate":selected=valid&valid.shift(1,fill_value=False)&state&~state.shift(1,fill_value=False)&side.ne(0)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return selected,side,u
def build_clock(panel,control="primary"):
 selected,side,u=active(panel,control);rows=[];reserved=None
 for i in panel.index[selected]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=P["entry_delay_minutes"]);exit_=entry+pd.Timedelta(hours=P["hold_hours"])
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":u.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(u.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCVCSS prereg drift")
 perp,spot=load_source();panel=build_panel(perp,spot);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f"{n}.csv.gz",common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f"{n}.csv.gz",common.csv_gz(x))
 source_core={"protocol_version":"hvcvcss_8_sources_v1","queries":{"perpetual":QUERY.format(table="bars_binance"),"spot":QUERY.format(table="bars_binance_spot")},"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"tables":["bars_binance","bars_binance_spot"],"window":[START.isoformat(),END.isoformat()],"physical_rows":{"perpetual":len(perp),"spot":len(spot)},"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":prereg.canonical_hash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcvcss_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};common.immutable(RESULT,common.json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":(r:=run())["support_passed"],"support":r["support"]},indent=2))
