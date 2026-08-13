"""Source-only support gate for frozen HVRVSC-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_return_volume_spectral_coherence_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="1f90da4731be308d9859c87be56f5158693768a7fdc0808d6cb1d1fe5c52725b";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,sum(quote_asset_volume) AS quote_turnover,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND quote_asset_volume>=0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
ROOT=Path("data/high_volatility_return_volume_spectral_coherence_relay_sources_2023_2026");PANEL=ROOT/"scheduled_spectral_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_return_volume_spectral_coherence_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_return_volume_spectral_coherence_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_return_volume_spectral_coherence_relay_controls_2023_2026");RESULT=Path("results/high_volatility_return_volume_spectral_coherence_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLS=("decision_time","feature_available_time","source_valid","spectral_coherence","all_frequency_coherence","coherence_rank","all_frequency_rank","realized_variation","variation_rank","block_return","late_return","side","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","spectral_coherence","coherence_rank","realized_variation","variation_rank","block_return","late_return")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(series:pd.Series)->pd.Series:
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-P["prior_decisions"]:],float)
  if math.isfinite(current) and len(prior)>=P["minimum_prior_decisions"]:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=series.index)
def spectral_coherence(turnover:np.ndarray,returns:np.ndarray,bins:np.ndarray)->float:
 v=np.asarray(turnover,float);r=np.asarray(returns,float);k=np.asarray(bins,int)
 if len(v)!=96 or len(r)!=96 or not np.isfinite(v).all() or not np.isfinite(r).all() or (v<0).any() or v.sum()<=0:return math.nan
 vf=np.fft.rfft(np.log1p(v)-np.log1p(v).mean());rf=np.fft.rfft(r-r.mean())
 if len(k)==0 or k.min()<1 or k.max()>=len(vf):return math.nan
 ve=float(np.square(np.abs(vf[k])).sum());re=float(np.square(np.abs(rf[k])).sum())
 if ve<=0 or re<=0:return math.nan
 value=float(abs(np.sum(vf[k]*np.conj(rf[k])))/math.sqrt(ve*re));return min(1.,value) if math.isfinite(value) else math.nan
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
 expected=["date","open","high","low","close","quote_turnover","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=expected:raise RuntimeError("HVRVSC source schema drift")
 x=raw.copy()
 for c in ("date","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 for c in expected[1:8]:x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.date.isna().any() or x.date.duplicated().any():raise RuntimeError("HVRVSC invalid source key")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.high.ge(z[["open","close"]].max(axis=1))&x.low.le(z[["open","close"]].min(axis=1))&np.isfinite(x.quote_turnover)&x.quote_turnover.ge(0);x["return"]=np.log(x.close/x.open);return x.set_index("date").sort_index()
def metrics(x:pd.DataFrame,decision:pd.Timestamp)->dict[str,Any]:
 idx=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="5min",inclusive="left");b=x.reindex(idx);valid=len(b)==96 and b.valid.eq(True).all()
 if not valid:return {"source_valid":False,"spectral_coherence":np.nan,"all_frequency_coherence":np.nan,"realized_variation":np.nan,"block_return":np.nan,"late_return":np.nan,"side":0}
 returns=b["return"].to_numpy(float);turnover=b.quote_turnover.to_numpy(float);coh=spectral_coherence(turnover,returns,np.asarray(P["frequency_bins"]));all_coh=spectral_coherence(turnover,returns,np.arange(1,49));rv=float(np.sqrt(np.square(returns).sum()));block=float(returns.sum());late=float(returns[-24:].sum());valid=np.isfinite([coh,all_coh,rv,block,late]).all() and rv>0;side=int(np.sign(block)) if valid and block!=0 and late!=0 and np.sign(block)==np.sign(late) else 0
 return {"source_valid":bool(valid),"spectral_coherence":coh if valid else np.nan,"all_frequency_coherence":all_coh if valid else np.nan,"realized_variation":rv if valid else np.nan,"block_return":block if valid else np.nan,"late_return":late if valid else np.nan,"side":side}
def build_panel(raw):
 x=prepare(raw);first=START.normalize()+pd.Timedelta("3h");rows=[{"decision_time":d,"feature_available_time":d,**metrics(x,d)} for d in pd.date_range(first,END,freq="8h",inclusive="left")];f=pd.DataFrame(rows);valid=f.source_valid.eq(True);f["coherence_rank"]=prior_rank(f.spectral_coherence.where(valid));f["all_frequency_rank"]=prior_rank(f.all_frequency_coherence.where(valid));f["variation_rank"]=prior_rank(f.realized_variation.where(valid));f["eligible"]=valid&f.coherence_rank.ge(P["coherence_rank_min"])&f.variation_rank.ge(P["variation_rank_min"])&f.side.ne(0);return f.loc[:,PANEL_COLS]
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy()
 if control=="one_decision_stale_spectrum":
  for c in ("spectral_coherence","coherence_rank"):u[c]=u[c].shift(1)
 strength=pd.Series(True,index=u.index) if control=="no_coherence_tail" else (u.all_frequency_rank.ge(P["coherence_rank_min"]) if control=="all_positive_frequency_coherence" else u.coherence_rank.ge(P["coherence_rank_min"]))
 variation=pd.Series(True,index=u.index) if control=="no_variation_gate" else u.variation_rank.ge(P["variation_rank_min"]);eligible=u.source_valid.eq(True)&strength&variation&u.side.ne(0);decisions=pd.to_datetime(panel.decision_time,utc=True);onset=eligible&decisions.shift(1).add(pd.Timedelta("8h")).eq(decisions)&panel.source_valid.shift(1,fill_value=False).eq(True)&~eligible.shift(1,fill_value=False);side=u.side.fillna(0).astype(int)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=side.where(side.eq(0),1)
 return onset,side
def clock(panel,control="primary"):
 onset,side=active(panel,control);rows=[];reserved=None
 for i in panel.index[onset]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:reserved=exit_;rows.append({"candidate":"HVRVSC-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(panel.at[i,c]) for c in CLOCK_COLS[8:]}})
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVRVSC prereg drift")
 raw=load_source();panel=build_panel(raw);primary=clock(panel);controls={n:clock(panel,n) for n in CONTROLS};immutable(PANEL,gz(panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvrvsc_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvrvsc_8_source_support_v1","policy_id":"HVRVSC-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":
 r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
