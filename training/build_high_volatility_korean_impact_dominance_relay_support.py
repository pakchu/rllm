"""Source-only support gate for frozen HVKID-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_korean_impact_dominance_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="f7c369f980a8c788a585a17dc5409b46005188b4a38e0743920c33ae748ac5c3";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"]);VENUES=("upbit","binance")
QUERY="""WITH source AS (
 SELECT ts,'upbit'::text AS venue,open,high,low,close,volume FROM bars_upbit WHERE symbol='KRW-BTC' AND interval='1m' AND ts>=:start AND ts<:end
 UNION ALL
 SELECT ts,'binance'::text AS venue,open,high,low,close,volume FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end)
SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,venue,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,sum(volume) AS base_volume,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND volume>=0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM source GROUP BY 1,2 ORDER BY 1,2"""
ROOT=Path("data/high_volatility_korean_impact_dominance_relay_sources_2023_2026");PANEL=ROOT/"scheduled_impact_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_korean_impact_dominance_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_korean_impact_dominance_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_korean_impact_dominance_relay_controls_2023_2026");RESULT=Path("results/high_volatility_korean_impact_dominance_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLS=("decision_time","feature_available_time","source_valid","upbit_impact","binance_impact","upbit_impact_z","binance_impact_z","korean_impact_dominance","dominance_rank","raw_impact_difference","raw_difference_rank","binance_realized_variation","variation_rank","upbit_block_return","side","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","korean_impact_dominance","dominance_rank","binance_realized_variation","variation_rank","upbit_block_return")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(series:pd.Series)->pd.Series:
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-P["prior_decisions"]:],float)
  if math.isfinite(current) and len(prior)>=P["minimum_prior_decisions"]:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=series.index)
def prior_zscore(series:pd.Series)->pd.Series:
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-P["prior_decisions"]:],float)
  if math.isfinite(current) and len(prior)>=P["minimum_prior_decisions"]:
   std=float(np.std(prior,ddof=1))
   if std>0 and math.isfinite(std):out[i]=(current-float(np.mean(prior)))/std
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
  with e.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:e.dispose()
def prepare(raw):
 expected=["date","venue","open","high","low","close","base_volume","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=expected:raise RuntimeError("HVKID source schema drift")
 x=raw.copy()
 for c in ("date","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 x.venue=x.venue.astype(str)
 for c in expected[2:9]:x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.date.isna().any() or x.duplicated(["date","venue"]).any() or not x.venue.isin(VENUES).all():raise RuntimeError("HVKID invalid source key")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.high.ge(z[["open","close"]].max(axis=1))&x.low.le(z[["open","close"]].min(axis=1))&np.isfinite(x.base_volume)&x.base_volume.ge(0);x["return"]=np.log(x.close/x.open);return x.sort_values(["date","venue"],kind="mergesort")
def block_metrics(source:pd.DataFrame,decision:pd.Timestamp)->dict[str,Any]:
 idx=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="5min",inclusive="left");b=source[source.date.isin(idx)];valid=len(b)==192 and b.valid.eq(True).all() and all(b[b.venue.eq(v)].date.reset_index(drop=True).equals(pd.Series(idx)) for v in VENUES)
 if not valid:return {"source_valid":False,"upbit_impact":np.nan,"binance_impact":np.nan,"raw_impact_difference":np.nan,"binance_realized_variation":np.nan,"upbit_block_return":np.nan,"side":0}
 values={}
 for v in VENUES:
  q=b[b.venue.eq(v)].sort_values("date");returns=q["return"].to_numpy(float);volume=float(q.base_volume.sum());numerator=float(np.abs(returns).sum());values[v]=(returns,volume,numerator)
 valid=all(volume>0 and numerator>0 for _,volume,numerator in values.values())
 if not valid:return {"source_valid":False,"upbit_impact":np.nan,"binance_impact":np.nan,"raw_impact_difference":np.nan,"binance_realized_variation":np.nan,"upbit_block_return":np.nan,"side":0}
 upbit=float(np.log(values["upbit"][2]/values["upbit"][1]));binance=float(np.log(values["binance"][2]/values["binance"][1]));rv=float(np.sqrt(np.square(values["binance"][0]).sum()));u=b[b.venue.eq("upbit")].sort_values("date");block=float(np.log(u.close.iloc[-1]/u.open.iloc[0]));valid=np.isfinite([upbit,binance,rv,block]).all() and rv>0 and block!=0
 return {"source_valid":bool(valid),"upbit_impact":upbit if valid else np.nan,"binance_impact":binance if valid else np.nan,"raw_impact_difference":upbit-binance if valid else np.nan,"binance_realized_variation":rv if valid else np.nan,"upbit_block_return":block if valid else np.nan,"side":int(np.sign(block)) if valid else 0}
def build_panel(raw):
 x=prepare(raw);first=START.normalize()+pd.Timedelta("4h");rows=[{"decision_time":d,"feature_available_time":d,**block_metrics(x,d)} for d in pd.date_range(first,END,freq="8h",inclusive="left")];f=pd.DataFrame(rows);valid=f.source_valid.eq(True);f["upbit_impact_z"]=prior_zscore(f.upbit_impact.where(valid));f["binance_impact_z"]=prior_zscore(f.binance_impact.where(valid));f["korean_impact_dominance"]=f.upbit_impact_z-f.binance_impact_z;f["dominance_rank"]=prior_rank(f.korean_impact_dominance.abs());f["raw_difference_rank"]=prior_rank(f.raw_impact_difference.abs().where(valid));f["variation_rank"]=prior_rank(f.binance_realized_variation.where(valid));f["eligible"]=valid&f.korean_impact_dominance.gt(0)&f.dominance_rank.ge(P["dominance_rank_min"])&f.variation_rank.ge(P["variation_rank_min"])&f.side.ne(0);return f.loc[:,PANEL_COLS]
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy()
 if control=="one_decision_stale_dominance":
  for c in ("korean_impact_dominance","dominance_rank"):u[c]=u[c].shift(1)
 if control=="raw_impact_difference":dominance=u.raw_impact_difference.gt(0)&u.raw_difference_rank.ge(P["dominance_rank_min"])
 elif control=="no_dominance_tail":dominance=u.korean_impact_dominance.gt(0)
 else:dominance=u.korean_impact_dominance.gt(0)&u.dominance_rank.ge(P["dominance_rank_min"])
 variation=pd.Series(True,index=u.index) if control=="no_variation_gate" else u.variation_rank.ge(P["variation_rank_min"]);eligible=u.source_valid.eq(True)&dominance&variation&u.side.ne(0);decisions=pd.to_datetime(panel.decision_time,utc=True);onset=eligible&decisions.shift(1).add(pd.Timedelta("8h")).eq(decisions)&panel.source_valid.shift(1,fill_value=False).eq(True)&~eligible.shift(1,fill_value=False);side=u.side.fillna(0).astype(int)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=side.where(side.eq(0),1)
 return onset,side
def clock(panel,control="primary"):
 onset,side=active(panel,control);rows=[];reserved=None
 for i in panel.index[onset]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:reserved=exit_;rows.append({"candidate":"HVKID-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(panel.at[i,c]) for c in CLOCK_COLS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLS)
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def gz(x):
 b=io.BytesIO();raw=x.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def jb(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def immutable(p,b):p.parent.mkdir(parents=True,exist_ok=True);(p.exists() and p.read_bytes()!=b) and (_ for _ in ()).throw(RuntimeError(f"refusing overwrite {p}"));p.write_bytes(b)
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVKID prereg drift")
 raw=load_source();panel=build_panel(raw);primary=clock(panel);controls={n:clock(panel,n) for n in CONTROLS};immutable(PANEL,gz(panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvkid_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvkid_8_source_support_v1","policy_id":"HVKID-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":
 r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
