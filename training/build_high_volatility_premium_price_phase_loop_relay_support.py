"""Source-only support gate for frozen HVPPL-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_premium_price_phase_loop_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="9af958f3414f44cc540f6d520cb0b3de0ebae268fabb5d354b67f66d3b0619e8";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"]);VENUES=("btc","premium")
QUERY="""SELECT ts,'btc'::text AS venue,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end UNION ALL SELECT ts,'premium'::text AS venue,open,high,low,close FROM bars_binance_premium WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts,venue"""
ROOT=Path("data/high_volatility_premium_price_phase_loop_relay_sources_2023_2026");PANEL=ROOT/"scheduled_phase_loop_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_premium_price_phase_loop_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_premium_price_phase_loop_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_premium_price_phase_loop_relay_controls_2023_2026");RESULT=Path("results/high_volatility_premium_price_phase_loop_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLS=("decision_time","feature_available_time","source_valid","phase_loop_area","area_magnitude_rank","premium_displacement","btc_displacement","btc_realized_variation","variation_rank","side","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","phase_loop_area","area_magnitude_rank","premium_displacement","btc_displacement","btc_realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(series):
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-P["prior_decisions"]:],float)
  if math.isfinite(current) and len(prior)>=P["minimum_prior_decisions"]:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=series.index)
def phase_loop(btc_close:np.ndarray,premium_close:np.ndarray)->tuple[float,float,float,float]:
 b=np.asarray(btc_close,float);p=np.asarray(premium_close,float)
 if len(b)!=480 or len(p)!=480 or not np.isfinite(b).all() or not np.isfinite(p).all() or (b<=0).any():return (math.nan,)*4
 r=np.diff(np.log(b));dp=np.diff(p);re=float(np.sqrt(np.square(r).sum()));pe=float(np.sqrt(np.square(dp).sum()))
 if re<=0 or pe<=0:return (math.nan,)*4
 x=np.r_[0.,np.cumsum(r)/re];y=np.r_[0.,np.cumsum(dp)/pe];area=float(.5*np.sum(x[:-1]*y[1:]-x[1:]*y[:-1]));return area,float(dp.sum()),float(r.sum()),re
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
 expected=["ts","venue","open","high","low","close"]
 if raw.columns.tolist()!=expected:raise RuntimeError("HVPPL source schema drift")
 x=raw.copy();x.ts=pd.to_datetime(x.ts,utc=True,errors="coerce");x.venue=x.venue.astype(str)
 for c in expected[2:]:x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.duplicated(["ts","venue"]).any() or not x.venue.isin(VENUES).all():raise RuntimeError("HVPPL invalid source key")
 z=x[["open","high","low","close"]];finite=np.isfinite(z).all(axis=1);coherent=x.high.ge(z[["open","close","low"]].max(axis=1))&x.low.le(z[["open","close","high"]].min(axis=1))&x.high.ge(x.low);x["valid"]=finite&coherent&(~x.venue.eq("btc")|z.gt(0).all(axis=1));return x.sort_values(["ts","venue"],kind="mergesort").set_index(["ts","venue"])
def metrics(source,decision):
 minutes=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");expected=pd.MultiIndex.from_product([minutes,VENUES],names=["ts","venue"]);b=source.reindex(expected);valid=len(b)==960 and b.valid.eq(True).all()
 if not valid:return {"source_valid":False,"phase_loop_area":np.nan,"premium_displacement":np.nan,"btc_displacement":np.nan,"btc_realized_variation":np.nan,"side":0}
 area,premium,btc,rv=phase_loop(b.xs("btc",level="venue").close.to_numpy(float),b.xs("premium",level="venue").close.to_numpy(float));valid=np.isfinite([area,premium,btc,rv]).all() and area!=0 and premium!=0 and btc!=0 and rv>0;side=int(np.sign(btc)) if valid and np.sign(premium)==np.sign(btc) else 0
 return {"source_valid":bool(valid),"phase_loop_area":area if valid else np.nan,"premium_displacement":premium if valid else np.nan,"btc_displacement":btc if valid else np.nan,"btc_realized_variation":rv if valid else np.nan,"side":side}
def build_panel(raw):
 x=prepare(raw);first=START.normalize()+pd.Timedelta("3h30m");rows=[{"decision_time":d,"feature_available_time":d,**metrics(x,d)} for d in pd.date_range(first,END,freq="8h",inclusive="left")];f=pd.DataFrame(rows);valid=f.source_valid.eq(True);f["area_magnitude_rank"]=prior_rank(f.phase_loop_area.abs().where(valid));f["variation_rank"]=prior_rank(f.btc_realized_variation.where(valid));f["eligible"]=valid&f.phase_loop_area.lt(0)&f.area_magnitude_rank.ge(P["area_magnitude_rank_min"])&f.variation_rank.ge(P["variation_rank_min"])&f.side.ne(0);return f.loc[:,PANEL_COLS]
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy()
 if control=="one_decision_stale_loop":
  for c in ("phase_loop_area","area_magnitude_rank","premium_displacement","btc_displacement","side"):u[c]=u[c].shift(1)
 orientation=u.phase_loop_area.ne(0) if control=="either_loop_orientation" else u.phase_loop_area.lt(0);strength=pd.Series(True,index=u.index) if control=="no_area_tail" else u.area_magnitude_rank.ge(P["area_magnitude_rank_min"]);variation=pd.Series(True,index=u.index) if control=="no_variation_gate" else u.variation_rank.ge(P["variation_rank_min"]);eligible=u.source_valid.eq(True)&orientation&strength&variation&u.side.ne(0);d=pd.to_datetime(panel.decision_time,utc=True);onset=eligible&d.shift(1).add(pd.Timedelta("8h")).eq(d)&panel.source_valid.shift(1,fill_value=False).eq(True)&~eligible.shift(1,fill_value=False);side=u.side.fillna(0).astype(int)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=side.where(side.eq(0),1)
 return onset,side,u
def clock(panel,control="primary"):
 onset,side,u=active(panel,control);rows=[];reserved=None
 for i in panel.index[onset]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:reserved=exit_;rows.append({"candidate":"HVPPL-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(u.at[i,c]) for c in CLOCK_COLS[8:]}})
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVPPL prereg drift")
 raw=load_source();panel=build_panel(raw);primary=clock(panel);controls={n:clock(panel,n) for n in CONTROLS};immutable(PANEL,gz(panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvppl_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvppl_8_source_support_v1","policy_id":"HVPPL-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":
 r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
