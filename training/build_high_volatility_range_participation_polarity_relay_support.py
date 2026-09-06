"""Build outcome-blind source support for frozen HVRPPR-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_range_participation_polarity_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-01-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="5a6191b6a660849b4a33ef23094db55fe690b843262a7ee638709fa49d47459f"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close,quote_asset_volume FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_range_participation_polarity_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_range_participation_polarity_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_range_participation_polarity_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_range_participation_polarity_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_range_participation_polarity_relay_support_2026-08-10.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","range_participation_polarity","polarity_magnitude","polarity_magnitude_rank","unweighted_range_polarity","realized_variation","variation_rank","eligible","onset")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","range_participation_polarity","polarity_magnitude","polarity_magnitude_rank","unweighted_range_polarity","realized_variation","variation_rank")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["prior_blocks"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_prior_blocks"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def polarity_statistics(block:pd.DataFrame)->tuple[float,float,float,float]:
 if len(block)!=480:return (math.nan,)*4
 groups=np.arange(480)//5;five=block.groupby(groups).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),turnover=("quote_asset_volume","sum"))
 direction=np.sign(np.log(five.close/five.open).to_numpy(float));energy=np.square(np.log(five.high/five.low).to_numpy(float))/(4*math.log(2));weights=energy*five.turnover.to_numpy(float)
 total=float(weights.sum());energy_total=float(energy.sum())
 if not np.isfinite(weights).all() or total<=0 or energy_total<=0:return (math.nan,)*4
 polarity=float(np.dot(weights,direction)/total);unweighted=float(np.dot(energy,direction)/energy_total);variation=float(np.square(block.minute_return.to_numpy(float)).sum())
 return polarity,abs(polarity),unweighted,variation

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
 if frame.columns.tolist()!=["ts","open","high","low","close","quote_asset_volume"]:raise RuntimeError("HVRPPR source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close","quote_asset_volume"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVRPPR invalid source key")
 prices=x[["open","high","low","close"]]
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)&np.isfinite(x.quote_asset_volume)&x.quote_asset_volume.ge(0)
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
   polarity,magnitude,unweighted,var=polarity_statistics(block);valid=all(math.isfinite(v) for v in (polarity,magnitude,unweighted,var)) and polarity!=0 and unweighted!=0 and var>0
  else:polarity=magnitude=unweighted=var=math.nan;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"range_participation_polarity":polarity,"polarity_magnitude":magnitude,"unweighted_range_polarity":unweighted,"realized_variation":var})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["polarity_magnitude_rank"]=prior_rank(panel.polarity_magnitude.where(valid));panel["variation_rank"]=prior_rank(panel.realized_variation.where(valid));panel["eligible"]=valid&panel.polarity_magnitude_rank.ge(P["polarity_magnitude_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);panel["onset"]=previous_valid_onset(panel.eligible,valid);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_geometry":
  cols=["source_valid","polarity_magnitude_rank","variation_rank","range_participation_polarity","unweighted_range_polarity","feature_available_time"];used[cols]=panel[cols].shift(1)
 valid=used.source_valid.eq(True);polarity=used.polarity_magnitude_rank.ge(P["polarity_magnitude_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);state=valid&polarity&variation
 if control=="no_polarity_tail_gate":state=valid&variation
 elif control=="no_variation_gate":state=valid&polarity
 onset=previous_valid_onset(state,valid);side=np.sign(pd.to_numeric(used.range_participation_polarity,errors="coerce").fillna(0)).astype(int)
 if control=="unweighted_range_polarity":side=np.sign(pd.to_numeric(used.unweighted_range_polarity,errors="coerce").fillna(0)).astype(int)
 elif control=="direction_flip":side=-side
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVRPPR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvrppr_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvrppr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
