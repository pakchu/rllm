"""Build outcome-blind source support for frozen HVMFCR-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_multifractal_coherence_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-01-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="807de7fc4dc2cb5762794d562e4178c6277c9903e5610797aa03b32a3a1a5476"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_multifractal_coherence_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_multifractal_coherence_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_multifractal_coherence_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_multifractal_coherence_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_multifractal_coherence_relay_support_2026-08-10.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","h1","h4","scaling_gap","gap_rank","h1_rank","realized_variation","variation_rank","completed_return","eligible","onset")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","h1","h4","scaling_gap","gap_rank","h1_rank","realized_variation","variation_rank","completed_return")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["prior_blocks"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_prior_blocks"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def multifractal_statistics(returns:np.ndarray)->tuple[float,float,float,float]:
 values=np.asarray(returns,float)
 if values.shape!=(480,) or not np.isfinite(values).all():return (math.nan,)*4
 profile=np.cumsum(values-values.mean());f1=[];f4=[]
 for scale in P["scales"]:
  segments=profile.reshape(-1,scale);t=np.arange(scale,dtype=float);center=t-t.mean();den=float(np.square(center).sum())
  centered=segments-segments.mean(axis=1,keepdims=True);slopes=(centered@center)/den;residual=centered-slopes[:,None]*center;variances=np.square(residual).mean(axis=1)
  if not np.isfinite(variances).all() or np.any(variances<=0):return (math.nan,)*4
  f1.append(float(np.sqrt(variances).mean()));f4.append(float(np.square(variances).mean()**.25))
 log_scales=np.log(np.asarray(P["scales"],float));h1=float(np.polyfit(log_scales,np.log(f1),1)[0]);h4=float(np.polyfit(log_scales,np.log(f4),1)[0]);gap=abs(h1-h4);variation=float(np.square(values).sum())
 return h1,h4,gap,variation

def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})

def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c: frame=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally: db.dispose()
 return frame

def prepare(frame:pd.DataFrame)->pd.DataFrame:
 if frame.columns.tolist()!=["ts","open","high","low","close"]:raise RuntimeError("HVMFCR source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVMFCR invalid source key")
 prices=x[["open","high","low","close"]]
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
 x["minute_return"]=np.log(x.close/x.open);return x.set_index("ts").sort_index()

def previous_valid_onset(eligible:pd.Series,valid:pd.Series)->pd.Series:
 out=pd.Series(False,index=eligible.index);previous=None
 for i in eligible.index:
  if not bool(valid.at[i]):continue
  if bool(eligible.at[i]) and previous is not None:out.at[i]=not bool(eligible.at[previous])
  previous=i
 return out

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("12h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");block=source.reindex(minutes);count=int(block.row_valid.eq(True).sum())
  path_valid=len(block)==480 and bool(block.row_valid.eq(True).all())
  if path_valid:
   h1,h4,gap,var=multifractal_statistics(block.minute_return.to_numpy(float));ret=float(math.log(block.close.iloc[-1]/block.open.iloc[0]));valid=all(math.isfinite(v) for v in (h1,h4,gap,var,ret)) and var>0 and ret!=0
  else:h1=h4=gap=var=ret=math.nan;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"h1":h1,"h4":h4,"scaling_gap":gap,"realized_variation":var,"completed_return":ret})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["gap_rank"]=prior_rank(panel.scaling_gap.where(valid));panel["h1_rank"]=prior_rank(panel.h1.where(valid));panel["variation_rank"]=prior_rank(panel.realized_variation.where(valid));panel["eligible"]=valid&panel.gap_rank.le(P["gap_rank_max"])&panel.variation_rank.ge(P["variation_rank_min"]);panel["onset"]=previous_valid_onset(panel.eligible,valid);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_geometry":
  cols=["source_valid","gap_rank","h1_rank","variation_rank","completed_return","feature_available_time"];used[cols]=panel[cols].shift(1)
 valid=used.source_valid.eq(True);coherent=used.gap_rank.le(P["gap_rank_max"]);variation=used.variation_rank.ge(P["variation_rank_min"]);state=valid&coherent&variation
 if control=="no_coherence_tail_gate":state=valid&variation
 elif control=="no_variation_gate":state=valid&coherent
 elif control=="q1_persistence":state=valid&used.h1_rank.ge(P["q1_persistence_rank_min"])&variation
 onset=previous_valid_onset(state,valid);side=np.sign(pd.to_numeric(used.completed_return,errors="coerce").fillna(0)).astype(int)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return onset&side.ne(0),side,used

def build_clock(panel:pd.DataFrame,control:str="primary")->pd.DataFrame:
 act,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[act]:
  decision=pd.Timestamp(panel.at[i,"decision_time"]);entry=decision+pd.Timedelta("5m");exit_time=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  reserved=exit_time;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":pd.Timestamp(used.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),**{c:float(used.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)

def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());months=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(months.max())/len(x)}

def csv_gz(frame):
 b=io.BytesIO();raw=frame.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(path,content):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.read_bytes()!=content:raise RuntimeError(f"refusing overwrite {path}")
 path.write_bytes(content)
def json_bytes(x):return (json.dumps(x,indent=2,allow_nan=False)+"\n").encode()

def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVMFCR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvmfcr_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvmfcr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
