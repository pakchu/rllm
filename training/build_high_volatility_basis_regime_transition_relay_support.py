"""Outcome-blind source-support gate for frozen HVBSRT-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_basis_regime_transition_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="cd9f8ccb55a8c7e248294321e45d2357dec6ab4c364a3a99603753e33408456d";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT date_bin('1 hour',p.ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS hour_time,avg(ln(p.close/s.close)) AS hourly_basis,sum(power(ln(p.close/p.open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT p.ts) AS distinct_rows,min(p.ts) AS first_ts,max(p.ts) AS last_ts,bool_and(p.open>0 AND p.high>0 AND p.low>0 AND p.close>0 AND p.high>=greatest(p.open,p.close,p.low) AND p.low<=least(p.open,p.close,p.high) AND s.open>0 AND s.high>0 AND s.low>0 AND s.close>0 AND s.high>=greatest(s.open,s.close,s.low) AND s.low<=least(s.open,s.close,s.high)) AS coherent FROM bars_binance p JOIN bars_binance_spot s ON s.ts=p.ts AND s.symbol='BTCUSDT' AND s.interval='1m' WHERE p.symbol='BTCUSDT' AND p.interval='1m' AND p.ts>=:start AND p.ts<:end GROUP BY 1 ORDER BY 1"""
ROOT=Path("data/high_volatility_basis_regime_transition_relay_sources_2023_2026");PANEL=ROOT/"hourly_basis_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_basis_regime_transition_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_basis_regime_transition_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_basis_regime_transition_relay_controls_2023_2026");RESULT=Path("results/high_volatility_basis_regime_transition_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","hourly_basis","causal_center","basis_residual","residual_rank","raw_basis_rank","realized_variation","variation_rank","transition","eligible")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","hourly_basis","causal_center","basis_residual","residual_rank","realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def causal_stat(series:pd.Series,lookback:int,minimum:int,kind:str)->pd.Series:
 vals=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-lookback:],float)
  if math.isfinite(v) and len(prior)>=minimum:
   out[i]=float(np.median(prior)) if kind=="median" else float((np.sum(prior<v)+.5*np.sum(prior==v))/len(prior))
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=series.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source():
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START,"end":END})
 finally:db.dispose()
def prepare(raw):
 required=["hour_time","hourly_basis","minute_squared_return","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=required:raise RuntimeError("HVBSRT source schema drift")
 x=raw.copy()
 for c in ("hour_time","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 for c in ("hourly_basis","minute_squared_return","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.hour_time.isna().any() or x.hour_time.duplicated().any():raise RuntimeError("HVBSRT invalid hourly key")
 x["source_valid"]=np.isfinite(x[["hourly_basis","minute_squared_return"]]).all(axis=1)&x.hourly_basis.ne(0)&x.minute_squared_return.ge(0)&x.source_rows.eq(60)&x.distinct_rows.eq(60)&x.first_ts.eq(x.hour_time)&x.last_ts.eq(x.hour_time+pd.Timedelta("59m"))&x.coherent.eq(True)
 return x.set_index("hour_time").sort_index()
def build_panel(raw):
 x=prepare(raw).reindex(pd.date_range(START,END,freq="1h",inclusive="left"));valid=x.source_valid.eq(True);basis=x.hourly_basis.where(valid);center=causal_stat(basis,P["center_hours"],P["minimum_center_hours"],"median");residual=(basis-center).where(lambda z:z.ne(0));residual_valid=valid&np.isfinite(residual);resrank=causal_stat(residual.abs().where(residual_valid),P["history_hours"],P["minimum_history_hours"],"rank");rawrank=causal_stat(basis.abs().where(valid),P["history_hours"],P["minimum_history_hours"],"rank");variation=np.sqrt(x.minute_squared_return.where(valid).rolling(24,min_periods=24).sum());varrank=causal_stat(variation.where(residual_valid),P["history_hours"],P["minimum_history_hours"],"rank");consecutive=residual_valid&residual_valid.shift(1,fill_value=False);transition=consecutive&np.sign(residual).ne(np.sign(residual.shift(1)));eligible=transition&resrank.ge(P["residual_rank_min"])&varrank.ge(P["variation_rank_min"]);p=pd.DataFrame({"decision_time":x.index+pd.Timedelta("1h"),"feature_available_time":x.index+pd.Timedelta("1h"),"source_valid":residual_valid,"hourly_basis":basis,"causal_center":center,"basis_residual":residual,"residual_rank":resrank,"raw_basis_rank":rawrank,"realized_variation":variation,"variation_rank":varrank,"transition":transition,"eligible":eligible});return p.loc[:,PANEL_COLUMNS]
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy();res=used.basis_residual;rank=used.residual_rank;valid=used.source_valid.eq(True);transition=used.transition.eq(True)
 if control=="raw_basis_zero_cross":res=used.hourly_basis;rank=used.raw_basis_rank;transition=valid&valid.shift(1,fill_value=False)&np.sign(res).ne(np.sign(res.shift(1)))
 if control=="one_hour_stale_transition":res=used.basis_residual.shift(1);rank=used.residual_rank.shift(1);transition=used.transition.shift(1,fill_value=False)
 tail=pd.Series(True,index=used.index) if control=="no_residual_tail" else rank.ge(P["residual_rank_min"]);variation=pd.Series(True,index=used.index) if control=="no_variation_gate" else used.variation_rank.ge(P["variation_rank_min"]);selected=valid&transition&tail&variation;side=np.sign(res).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return selected&side.ne(0),side,used
def build_clock(panel,control="primary"):
 selected,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[selected]:
  decision=pd.Timestamp(panel.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=P["entry_delay_minutes"]);exit_=entry+pd.Timedelta(hours=P["hold_hours"])
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":used.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(used.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVBSRT prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f"{n}.csv.gz",common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f"{n}.csv.gz",common.csv_gz(x))
 source_core={"protocol_version":"hvbsrt_8_sources_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"tables":["bars_binance","bars_binance_spot"],"symbol":"BTCUSDT","window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvbsrt_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};common.immutable(RESULT,common.json_bytes(r));return r
if __name__=="__main__":print(json.dumps({"passed":(r:=run())["support_passed"],"support":r["support"]}))
