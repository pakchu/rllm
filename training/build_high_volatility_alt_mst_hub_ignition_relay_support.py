"""Source-only support gate for frozen HVAMST-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_alt_mst_hub_ignition_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="a41fdf5ce071ed241172a2d2b1247d952d8992a55c7dcf64d3ff2b12127e1654";REG=prereg.build();P=REG["policy"];ALTS=prereg.ALTS;SYMBOLS=("BTCUSDT",*ALTS);SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,symbol,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2 ORDER BY 1,2"""
ROOT=Path("data/high_volatility_alt_mst_hub_ignition_relay_sources_2023_2026");PANEL=ROOT/"scheduled_mst_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_alt_mst_hub_ignition_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_alt_mst_hub_ignition_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_alt_mst_hub_ignition_relay_controls_2023_2026");RESULT=Path("results/high_volatility_alt_mst_hub_ignition_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLS=("decision_time","feature_available_time","source_valid","hub_symbol","hub_degree","degree_centralization","centralization_rank","hub_final_hour_return","equal_weight_final_hour_return","btc_realized_variation","variation_rank","side","equal_weight_side","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","hub_symbol","hub_degree","degree_centralization","centralization_rank","btc_realized_variation","variation_rank","hub_final_hour_return")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(series):
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-P["prior_decisions"]:],float)
  if math.isfinite(current) and len(prior)>=P["minimum_prior_decisions"]:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=series.index)
def mst_topology(correlation:np.ndarray)->tuple[str|None,int,float]:
 c=np.asarray(correlation,float);n=len(ALTS)
 if c.shape!=(n,n) or not np.isfinite(c).all() or not np.allclose(c,c.T,atol=1e-12) or not np.allclose(np.diag(c),1.,atol=1e-12):return None,0,math.nan
 edges=[]
 for i in range(n):
  for j in range(i+1,n):
   value=2*(1-c[i,j])
   if value<0 and value>=-1e-12:value=0.
   if value<0:return None,0,math.nan
   edges.append((math.sqrt(value),i,j))
 edges.sort();parent=list(range(n));degree=[0]*n
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 chosen=0
 for _,i,j in edges:
  a,b=find(i),find(j)
  if a==b:continue
  if a>b:a,b=b,a
  parent[b]=a;degree[i]+=1;degree[j]+=1;chosen+=1
  if chosen==n-1:break
 if chosen!=n-1:return None,0,math.nan
 maximum=max(degree);hubs=[i for i,d in enumerate(degree) if d==maximum]
 if len(hubs)!=1:return None,maximum,sum(maximum-d for d in degree)/20.
 return ALTS[hubs[0]],maximum,sum(maximum-d for d in degree)/20.
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
 if raw.columns.tolist()!=expected:raise RuntimeError("HVAMST source schema drift")
 x=raw.copy()
 for c in ("date","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 for c in expected[2:8]:x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.date.isna().any() or x.duplicated(["date","symbol"]).any() or not x.symbol.isin(SYMBOLS).all():raise RuntimeError("HVAMST invalid source key")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.high.ge(z[["open","close"]].max(axis=1))&x.low.le(z[["open","close"]].min(axis=1));x["return"]=np.log(x.close/x.open);return x
def metrics(source,decision):
 idx=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="5min",inclusive="left");b=source[source.date.isin(idx)];valid=len(b)==96*len(SYMBOLS) and b.valid.eq(True).all() and all(b[b.symbol.eq(s)].date.reset_index(drop=True).equals(pd.Series(idx)) for s in SYMBOLS)
 bad={"source_valid":False,"hub_symbol":"","hub_degree":0,"degree_centralization":np.nan,"hub_final_hour_return":np.nan,"equal_weight_final_hour_return":np.nan,"btc_realized_variation":np.nan,"side":0,"equal_weight_side":0}
 if not valid:return bad
 panel=b.pivot(index="date",columns="symbol",values="return").reindex(idx,columns=SYMBOLS);alt=panel.loc[:,ALTS];variances=alt.var(ddof=1)
 if not np.isfinite(alt).all().all() or not variances.gt(0).all():return bad
 hub,degree,central=mst_topology(alt.corr().to_numpy(float))
 if hub is None or not math.isfinite(central):return bad
 hub_return=float(alt[hub].iloc[-12:].sum());equal=float(alt.iloc[-12:].sum(axis=0).mean());rv=float(np.sqrt(np.square(panel.BTCUSDT.to_numpy(float)).sum()));valid=np.isfinite([hub_return,equal,rv]).all() and hub_return!=0 and equal!=0 and rv>0
 return {"source_valid":bool(valid),"hub_symbol":hub if valid else "","hub_degree":degree if valid else 0,"degree_centralization":central if valid else np.nan,"hub_final_hour_return":hub_return if valid else np.nan,"equal_weight_final_hour_return":equal if valid else np.nan,"btc_realized_variation":rv if valid else np.nan,"side":int(np.sign(hub_return)) if valid else 0,"equal_weight_side":int(np.sign(equal)) if valid else 0}
def build_panel(raw):
 x=prepare(raw);first=START.normalize()+pd.Timedelta("5h");rows=[{"decision_time":d,"feature_available_time":d,**metrics(x,d)} for d in pd.date_range(first,END,freq="8h",inclusive="left")];f=pd.DataFrame(rows);valid=f.source_valid.eq(True);f["centralization_rank"]=prior_rank(f.degree_centralization.where(valid));f["variation_rank"]=prior_rank(f.btc_realized_variation.where(valid));f["eligible"]=valid&f.centralization_rank.ge(P["centralization_rank_min"])&f.variation_rank.ge(P["variation_rank_min"])&f.side.ne(0);return f.loc[:,PANEL_COLS]
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy()
 if control=="one_decision_stale_tree":
  for c in ("hub_symbol","hub_degree","degree_centralization","centralization_rank","hub_final_hour_return","side"):u[c]=u[c].shift(1)
 strength=pd.Series(True,index=u.index) if control=="no_centralization_tail" else u.centralization_rank.ge(P["centralization_rank_min"]);variation=pd.Series(True,index=u.index) if control=="no_variation_gate" else u.variation_rank.ge(P["variation_rank_min"]);side=(u.equal_weight_side if control=="equal_weight_alt_direction" else u.side).fillna(0).astype(int);eligible=u.source_valid.eq(True)&strength&variation&side.ne(0);d=pd.to_datetime(panel.decision_time,utc=True);onset=eligible&d.shift(1).add(pd.Timedelta("8h")).eq(d)&panel.source_valid.shift(1,fill_value=False).eq(True)&~eligible.shift(1,fill_value=False)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=side.where(side.eq(0),1)
 return onset,side,u
def clock(panel,control="primary"):
 onset,side,u=active(panel,control);rows=[];reserved=None
 for i in panel.index[onset]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:reserved=exit_;rows.append({"candidate":"HVAMST-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"hub_symbol":str(u.at[i,"hub_symbol"]),"hub_degree":int(u.at[i,"hub_degree"]),"degree_centralization":float(u.at[i,"degree_centralization"]),"centralization_rank":float(u.at[i,"centralization_rank"]),"btc_realized_variation":float(u.at[i,"btc_realized_variation"]),"variation_rank":float(u.at[i,"variation_rank"]),"hub_final_hour_return":float(u.at[i,"hub_final_hour_return"])})
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVAMST prereg drift")
 raw=load_source();panel=build_panel(raw);primary=clock(panel);controls={n:clock(panel,n) for n in CONTROLS};immutable(PANEL,gz(panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvamst_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvamst_8_source_support_v1","policy_id":"HVAMST-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":
 r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
