"""Source-only support gate for frozen HVAMF-8."""
from __future__ import annotations
import gzip,hashlib,io,itertools,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_alt_modularity_fragmentation_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="578613ce3da6717f0a2ba5fd799bfa829fa5e1f17e575a9b47fda9ed2f4a66d4";REG=prereg.build();P=REG["policy"];ALTS=prereg.ALTS;SYMBOLS=("BTCUSDT",*ALTS);SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,symbol,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2 ORDER BY 1,2"""
ROOT=Path("data/high_volatility_alt_modularity_fragmentation_relay_sources_2023_2026");PANEL=ROOT/"scheduled_modularity_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_alt_modularity_fragmentation_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_alt_modularity_fragmentation_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_alt_modularity_fragmentation_relay_controls_2023_2026");RESULT=Path("results/high_volatility_alt_modularity_fragmentation_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLS=("decision_time","feature_available_time","source_valid","community_a","community_b","modularity","modularity_rank","community_a_final_hour_return","community_b_final_hour_return","dominant_community_return","all_alt_final_hour_return","btc_realized_variation","variation_rank","side","all_alt_side","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","community_a","community_b","modularity","modularity_rank","btc_realized_variation","variation_rank","dominant_community_return")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(series):
 a=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-P["prior_decisions"]:],float)
  if math.isfinite(current) and len(prior)>=P["minimum_prior_decisions"]:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=series.index)
def modularity_partition(correlation:np.ndarray)->tuple[tuple[int,...]|None,tuple[int,...]|None,float]:
 c=np.asarray(correlation,float);n=len(ALTS)
 if c.shape!=(n,n) or not np.isfinite(c).all() or not np.allclose(c,c.T,atol=1e-12) or not np.allclose(np.diag(c),1.,atol=1e-12):return None,None,math.nan
 adjacency=np.maximum(c,0.);np.fill_diagonal(adjacency,0.)
 total=float(np.triu(adjacency,1).sum())
 if not math.isfinite(total) or total<=0:return None,None,math.nan
 degree=adjacency.sum(axis=1);candidates=[]
 for size in range(P["minimum_community_size"],n-P["minimum_community_size"]+1):
  for rest in itertools.combinations(range(1,n),size-1):
   a=(0,*rest);b=tuple(i for i in range(n) if i not in a)
   if len(b)<P["minimum_community_size"]:continue
   q=0.
   for group in (a,b):
    internal=float(sum(adjacency[i,j] for pos,i in enumerate(group) for j in group[pos+1:]))
    weighted_degree=float(degree[list(group)].sum())
    q+=internal/total-(weighted_degree/(2*total))**2
   candidates.append((float(q),a,b))
 if not candidates:return None,None,math.nan
 maximum=max(q for q,_,_ in candidates);winners=[x for x in candidates if x[0]==maximum]
 if len(winners)!=1:return None,None,maximum
 return winners[0][1],winners[0][2],maximum
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
 if raw.columns.tolist()!=expected:raise RuntimeError("HVAMF source schema drift")
 x=raw.copy()
 for c in ("date","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 for c in expected[2:8]:x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.date.isna().any() or x.duplicated(["date","symbol"]).any() or not x.symbol.isin(SYMBOLS).all():raise RuntimeError("HVAMF invalid source key")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.high.ge(z[["open","close"]].max(axis=1))&x.low.le(z[["open","close"]].min(axis=1));x["return"]=np.log(x.close/x.open);return x
def metrics(source,decision):
 idx=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="5min",inclusive="left");b=source[source.date.isin(idx)];valid=len(b)==96*len(SYMBOLS) and b.valid.eq(True).all() and all(b[b.symbol.eq(s)].date.reset_index(drop=True).equals(pd.Series(idx)) for s in SYMBOLS)
 bad={"source_valid":False,"community_a":"","community_b":"","modularity":np.nan,"community_a_final_hour_return":np.nan,"community_b_final_hour_return":np.nan,"dominant_community_return":np.nan,"all_alt_final_hour_return":np.nan,"btc_realized_variation":np.nan,"side":0,"all_alt_side":0}
 if not valid:return bad
 panel=b.pivot(index="date",columns="symbol",values="return").reindex(idx,columns=SYMBOLS);alt=panel.loc[:,ALTS];variances=alt.var(ddof=1)
 if not np.isfinite(alt).all().all() or not variances.gt(0).all():return bad
 a,b,modularity=modularity_partition(alt.corr().to_numpy(float))
 if a is None or b is None or not math.isfinite(modularity):return bad
 final_hour=alt.iloc[-12:].sum(axis=0);a_return=float(final_hour.iloc[list(a)].median());b_return=float(final_hour.iloc[list(b)].median());all_alt=float(final_hour.median());rv=float(np.sqrt(np.square(panel.BTCUSDT.to_numpy(float)).sum()))
 dominant=a_return if abs(a_return)>abs(b_return) else b_return if abs(b_return)>abs(a_return) else math.nan
 valid=np.isfinite([a_return,b_return,dominant,all_alt,rv]).all() and a_return!=0 and b_return!=0 and dominant!=0 and all_alt!=0 and rv>0
 return {"source_valid":bool(valid),"community_a":"|".join(ALTS[i] for i in a) if valid else "","community_b":"|".join(ALTS[i] for i in b) if valid else "","modularity":modularity if valid else np.nan,"community_a_final_hour_return":a_return if valid else np.nan,"community_b_final_hour_return":b_return if valid else np.nan,"dominant_community_return":dominant if valid else np.nan,"all_alt_final_hour_return":all_alt if valid else np.nan,"btc_realized_variation":rv if valid else np.nan,"side":int(np.sign(dominant)) if valid else 0,"all_alt_side":int(np.sign(all_alt)) if valid else 0}
def build_panel(raw):
 x=prepare(raw);first=START.normalize()+pd.Timedelta("4h");rows=[{"decision_time":d,"feature_available_time":d,**metrics(x,d)} for d in pd.date_range(first,END,freq="8h",inclusive="left")];f=pd.DataFrame(rows);valid=f.source_valid.eq(True);f["modularity_rank"]=prior_rank(f.modularity.where(valid));f["variation_rank"]=prior_rank(f.btc_realized_variation.where(valid));f["eligible"]=valid&f.modularity_rank.ge(P["modularity_rank_min"])&f.variation_rank.ge(P["variation_rank_min"])&f.side.ne(0);return f.loc[:,PANEL_COLS]
def active(panel,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy()
 if control=="one_decision_stale_partition":
  for c in ("community_a","community_b","modularity","modularity_rank","community_a_final_hour_return","community_b_final_hour_return","dominant_community_return","side"):u[c]=u[c].shift(1)
 strength=pd.Series(True,index=u.index) if control=="no_modularity_tail" else u.modularity_rank.ge(P["modularity_rank_min"]);variation=pd.Series(True,index=u.index) if control=="no_variation_gate" else u.variation_rank.ge(P["variation_rank_min"]);side=(u.all_alt_side if control=="all_alt_final_hour_median" else u.side).fillna(0).astype(int);eligible=u.source_valid.eq(True)&strength&variation&side.ne(0);d=pd.to_datetime(panel.decision_time,utc=True);onset=eligible&d.shift(1).add(pd.Timedelta("8h")).eq(d)&panel.source_valid.shift(1,fill_value=False).eq(True)&~eligible.shift(1,fill_value=False)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=side.where(side.eq(0),1)
 return onset,side,u
def clock(panel,control="primary"):
 onset,side,u=active(panel,control);rows=[];reserved=None
 for i in panel.index[onset]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:reserved=exit_;rows.append({"candidate":"HVAMF-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"community_a":str(u.at[i,"community_a"]),"community_b":str(u.at[i,"community_b"]),"modularity":float(u.at[i,"modularity"]),"modularity_rank":float(u.at[i,"modularity_rank"]),"btc_realized_variation":float(u.at[i,"btc_realized_variation"]),"variation_rank":float(u.at[i,"variation_rank"]),"dominant_community_return":float(u.at[i,"dominant_community_return"])})
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVAMF prereg drift")
 raw=load_source();panel=build_panel(raw);primary=clock(panel);controls={n:clock(panel,n) for n in CONTROLS};immutable(PANEL,gz(panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvamf_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvamf_8_source_support_v1","policy_id":"HVAMF-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":
 r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
