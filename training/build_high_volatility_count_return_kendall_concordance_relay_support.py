"""Source-only support gate for frozen HVCRKC-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_count_return_kendall_concordance_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="44e43bbb1bfbbb26f1983ce9557867fb7cac1819ebf9833a5e4d7ab299041a4f";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,sum(number_of_trades) AS execution_count,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND number_of_trades>=0 AND number_of_trades=floor(number_of_trades) AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
ROOT=Path("data/high_volatility_count_return_kendall_concordance_relay_sources_2023_2026");PANEL=ROOT/"scheduled_concordance_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_count_return_kendall_concordance_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_count_return_kendall_concordance_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_count_return_kendall_concordance_relay_controls_2023_2026");RESULT=Path("results/high_volatility_count_return_kendall_concordance_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLS=("decision_time","feature_available_time","source_valid","kendall_tau_b","pearson_correlation","absolute_tau","strength_rank","realized_variation","variation_rank","completed_displacement","side","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","kendall_tau_b","absolute_tau","strength_rank","realized_variation","variation_rank","completed_displacement")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rank(series:pd.Series)->pd.Series:
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-P["prior_decisions"]:],float)
  if math.isfinite(current) and len(prior)>=P["minimum_prior_decisions"]:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=series.index)
def kendall_tau_b(x:np.ndarray,y:np.ndarray)->float:
 x=np.asarray(x,float);y=np.asarray(y,float)
 if len(x)!=96 or len(y)!=96 or not np.isfinite(x).all() or not np.isfinite(y).all():return math.nan
 i,j=np.triu_indices(len(x),1);dx=np.sign(x[i]-x[j]);dy=np.sign(y[i]-y[j]);product=dx*dy;c=int(np.sum(product>0));d=int(np.sum(product<0));tx=int(np.sum((dx==0)&(dy!=0)));ty=int(np.sum((dy==0)&(dx!=0)));den=math.sqrt((c+d+tx)*(c+d+ty))
 return (c-d)/den if den>0 else math.nan
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
 expected=["date","open","high","low","close","execution_count","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=expected:raise RuntimeError("HVCRKC source schema drift")
 x=raw.copy()
 for c in ("date","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 for c in expected[1:8]:x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.date.isna().any() or x.date.duplicated().any():raise RuntimeError("HVCRKC invalid source key")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.high.ge(z[["open","close"]].max(axis=1))&x.low.le(z[["open","close"]].min(axis=1))&np.isfinite(x.execution_count)&x.execution_count.ge(0)&x.execution_count.eq(np.floor(x.execution_count));x["return"]=np.log(x.close/x.open);return x.set_index("date").sort_index()
def metrics(x:pd.DataFrame,decision:pd.Timestamp)->dict[str,Any]:
 idx=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="5min",inclusive="left");b=x.reindex(idx);valid=len(b)==96 and b.valid.eq(True).all()
 if not valid:return {"source_valid":False,"kendall_tau_b":np.nan,"pearson_correlation":np.nan,"absolute_tau":np.nan,"realized_variation":np.nan,"completed_displacement":np.nan,"side":0}
 returns=b["return"].to_numpy(float);activity=np.log1p(b.execution_count.to_numpy(float));tau=kendall_tau_b(activity,returns);pearson=float(np.corrcoef(activity,returns)[0,1]) if np.var(activity)>0 and np.var(returns)>0 else np.nan;rv=float(np.sqrt(np.square(returns).sum()));displacement=float(returns.sum());valid=np.isfinite([tau,pearson,rv,displacement]).all() and tau!=0 and rv>0;side=int(np.sign(tau)) if valid and displacement!=0 and np.sign(tau)==np.sign(displacement) else 0
 return {"source_valid":bool(valid),"kendall_tau_b":tau if valid else np.nan,"pearson_correlation":pearson if valid else np.nan,"absolute_tau":abs(tau) if valid else np.nan,"realized_variation":rv if valid else np.nan,"completed_displacement":displacement if valid else np.nan,"side":side}
def build_panel(raw):
 x=prepare(raw);first=START.normalize()+pd.Timedelta("2h");rows=[{"decision_time":d,"feature_available_time":d,**metrics(x,d)} for d in pd.date_range(first,END,freq="8h",inclusive="left")];f=pd.DataFrame(rows);valid=f.source_valid.eq(True);f["strength_rank"]=rank(f.absolute_tau.where(valid));f["variation_rank"]=rank(f.realized_variation.where(valid));f["eligible"]=valid&f.strength_rank.ge(P["strength_rank_min"])&f.variation_rank.ge(P["variation_rank_min"])&f.side.ne(0);return f.loc[:,PANEL_COLS]
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy()
 if control=="one_decision_stale_tau":
  for c in ("kendall_tau_b","absolute_tau","strength_rank","side"):u[c]=u[c].shift(1)
 strength=pd.Series(True,index=u.index) if control=="no_strength_tail" else u.strength_rank.ge(P["strength_rank_min"])
 if control=="pearson_instead_of_kendall":strength=rank(u.pearson_correlation.abs().where(u.source_valid)).ge(P["strength_rank_min"]);u["side"]=np.sign(u.pearson_correlation).fillna(0).astype(int);confirmation=np.sign(u.pearson_correlation).eq(np.sign(u.completed_displacement))
 else:confirmation=u.side.ne(0)
 variation=pd.Series(True,index=u.index) if control=="no_variation_gate" else u.variation_rank.ge(P["variation_rank_min"]);eligible=u.source_valid.eq(True)&strength&variation&confirmation;decisions=pd.to_datetime(panel.decision_time,utc=True);onset=eligible&decisions.shift(1).add(pd.Timedelta("8h")).eq(decisions)&panel.source_valid.shift(1,fill_value=False).eq(True)&~eligible.shift(1,fill_value=False);side=u.side.fillna(0).astype(int)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=side.where(side.eq(0),1)
 return onset,side
def clock(panel,control="primary"):
 onset,side=active(panel,control);rows=[];reserved=None
 for i in panel.index[onset]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:reserved=exit_;rows.append({"candidate":"HVCRKC-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(panel.at[i,c]) for c in CLOCK_COLS[8:]}})
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCRKC prereg drift")
 raw=load_source();panel=build_panel(raw);primary=clock(panel);controls={n:clock(panel,n) for n in CONTROLS};immutable(PANEL,gz(panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvcrkc_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcrkc_8_source_support_v1","policy_id":"HVCRKC-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
