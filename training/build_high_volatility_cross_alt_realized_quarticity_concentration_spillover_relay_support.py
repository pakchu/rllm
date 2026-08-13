"""Build outcome-blind source support for frozen HVCARQ-6."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="9a26d30c718e6ad21d92bb4dd5498293b84664ae6a97bbc1d9be11b98571ffdc";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"]);SYMBOLS=("BTCUSDT",*prereg.ALTS);ALTS=prereg.ALTS
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,symbol,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2 ORDER BY 1,2"""
ROOT=Path("data/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_sources_2023_2026");PANEL=ROOT/"scheduled_quarticity_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_controls_2023_2026");RESULT=Path("results/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
BASE_COLS=("decision_time","feature_available_time","source_valid","quarticity_score","selected_alts","agreeing_alts","side","btc_realized_variation","variation_rank","variation_active","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","quarticity_score","selected_alts","agreeing_alts","btc_realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def midrank(series:pd.Series,lookback:int,minimum:int)->pd.Series:
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-lookback:],float)
  if math.isfinite(current) and len(prior)>=minimum:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=series.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source():
 from sqlalchemy import text
 e=postgres_engine()
 try:
  with e.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"symbols":list(SYMBOLS),"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:e.dispose()
def prepare(raw):
 expected=["date","symbol","open","high","low","close","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=expected:raise RuntimeError("HVCARQ source schema drift")
 x=raw.copy();x.date=pd.to_datetime(x.date,utc=True);x.first_ts=pd.to_datetime(x.first_ts,utc=True);x.last_ts=pd.to_datetime(x.last_ts,utc=True)
 for c in ("open","high","low","close","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="coerce")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.high.ge(z[["open","close"]].max(axis=1))&x.low.le(z[["open","close"]].min(axis=1));x["return"]=np.log(x.close/x.open);return x
def quarticity_frame(ret,valid):
 scheduled=pd.date_range(START,END,freq="8h",inclusive="left")+pd.Timedelta("2h30m");scheduled=scheduled[(scheduled>=START)&(scheduled<END)];rows=[];window_valid=valid.rolling(P["variation_bars"],min_periods=P["variation_bars"]).sum().eq(P["variation_bars"]).all(axis=1);rv=np.sqrt(ret.BTCUSDT.pow(2).where(valid.BTCUSDT).rolling(P["variation_bars"],min_periods=P["variation_bars"]).sum());alt_rv={a:ret[a].pow(2).where(valid[a]).rolling(P["path_bars"],min_periods=P["path_bars"]).sum() for a in ALTS};alt_r4={a:ret[a].pow(4).where(valid[a]).rolling(P["path_bars"],min_periods=P["path_bars"]).sum() for a in ALTS};alt_move={a:ret[a].where(valid[a]).rolling(P["path_bars"],min_periods=P["path_bars"]).sum() for a in ALTS}
 for decision in scheduled:
  end=decision-pd.Timedelta("5m")
  if end not in ret.index:continue
  row={"decision_time":decision,"feature_available_time":decision,"source_valid":bool(window_valid.at[end]),"latest_time":end,"btc_realized_variation":float(rv.at[end]) if math.isfinite(rv.at[end]) else np.nan}
  for a in ALTS:
   variance=float(alt_rv[a].at[end]) if math.isfinite(alt_rv[a].at[end]) else np.nan;fourth=float(alt_r4[a].at[end]) if math.isfinite(alt_r4[a].at[end]) else np.nan
   row[f"{a}_realized_variance"]=variance;row[f"{a}_quarticity"]=P["path_bars"]*fourth/(variance*variance) if math.isfinite(variance) and variance>0 and math.isfinite(fourth) else np.nan;row[f"{a}_displacement"]=float(alt_move[a].at[end]) if math.isfinite(alt_move[a].at[end]) else np.nan
  rows.append(row)
 return pd.DataFrame(rows)
def score_states(f,control="primary"):
 u=f.copy()
 for a in ALTS:
  u[f"{a}_quarticity_rank"]=midrank(u[f"{a}_quarticity"].where(u.source_valid),P["quarticity_prior_decisions"],P["quarticity_minimum_decisions"])
 u["variation_rank"]=midrank(u.btc_realized_variation.where(u.source_valid),P["variation_prior_decisions"],P["variation_minimum_decisions"]);scores=[];selected=[];agree=[]
 for i,row in u.iterrows():
  chosen=[a for a in ALTS if math.isfinite(row[f"{a}_quarticity"]) and math.isfinite(row[f"{a}_displacement"]) and row[f"{a}_displacement"]!=0 and (control=="no_quarticity_tail" or row[f"{a}_quarticity_rank"]>=P["quarticity_rank_min"])]
  signs={a:int(np.sign(row[f"{a}_displacement"])) for a in chosen};weight_name="realized_variance" if control=="variance_weighted_score" else "quarticity";weights={a:row[f"{a}_{weight_name}"] for a in chosen};den=sum(weights.values());score=sum(weights[a]*signs[a] for a in chosen)/den if den>0 else np.nan;sgn=int(np.sign(score)) if math.isfinite(score) else 0;scores.append(score);selected.append(len(chosen));agree.append(sum(v==sgn for v in signs.values()) if sgn else 0)
 u["quarticity_score"]=scores;u["selected_alts"]=selected;u["agreeing_alts"]=agree;minimum=3 if control=="three_alt_consensus" else P["minimum_selected_alts"];u["side"]=np.sign(u.quarticity_score).fillna(0).astype(int);u["variation_active"]=u.variation_rank.ge(P["variation_rank_min"]);u["eligible"]=u.source_valid&u.selected_alts.ge(minimum)&u.quarticity_score.abs().ge(P["score_absolute_min"])&u.agreeing_alts.ge(P["minimum_agreeing_alts"])&u.variation_active
 if control=="no_variation_gate":u["eligible"]=u.source_valid&u.selected_alts.ge(minimum)&u.quarticity_score.abs().ge(P["score_absolute_min"])&u.agreeing_alts.ge(P["minimum_agreeing_alts"])
 if control=="one_decision_stale_consensus":
  for c in ("quarticity_score","selected_alts","agreeing_alts","side"):u[c]=u[c].shift(1)
  u["eligible"]=u.source_valid&u.selected_alts.ge(minimum)&u.quarticity_score.abs().ge(P["score_absolute_min"])&u.agreeing_alts.ge(P["minimum_agreeing_alts"])&u.variation_active
 if control=="direction_flip":u.side=-u.side
 if control=="forced_long":u.side=1
 return u
def build_panel(raw,control="primary"):
 x=prepare(raw);grid=pd.date_range(START,END,freq="5min",inclusive="left");ret=x.pivot(index="date",columns="symbol",values="return").reindex(grid,columns=SYMBOLS);valid=x.pivot(index="date",columns="symbol",values="valid").reindex(grid,columns=SYMBOLS).eq(True);return score_states(quarticity_frame(ret,valid),control)
def clock(panel,control="primary"):
 rows=[]
 for i in panel.index[panel.eligible&panel.side.ne(0)]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("6h");split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":pd.Timestamp(panel.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_,"side":int(panel.at[i,"side"]),**{c:float(panel.at[i,c]) for c in CLOCK_COLS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLS)
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def gz(x):
 b=io.BytesIO();raw=x.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode();
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(p,b):p.parent.mkdir(parents=True,exist_ok=True);(p.exists() and p.read_bytes()!=b) and (_ for _ in ()).throw(RuntimeError(f"refusing overwrite {p}"));p.write_bytes(b)
def jb(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCARQ prereg drift")
 raw=load_source();primary_panel=build_panel(raw);primary=clock(primary_panel);controls={}
 for n in CONTROLS:controls[n]=clock(build_panel(raw,n),n)
 immutable(PANEL,gz(primary_panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvcarq_6_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(primary_panel),"valid_rows":int(primary_panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcarq_6_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
