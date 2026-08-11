"""Build outcome-blind source support for frozen HVCARTC-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_cross_alt_return_turnover_concordance_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-01-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="0a2d2efd2ed8642d5befc9a48923d7354dc1aeb6f60e01f640bb865e18269618"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
SYMBOLS=("BTCUSDT","ADAUSDT","BNBUSDT","DOGEUSDT","ETHUSDT","SOLUSDT","XRPUSDT");ALTS=SYMBOLS[1:]
QUERY="""SELECT ts,symbol,open,high,low,close,quote_asset_volume FROM bars_binance WHERE symbol IN ('BTCUSDT','ADAUSDT','BNBUSDT','DOGEUSDT','ETHUSDT','SOLUSDT','XRPUSDT') AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts,symbol"""
ROOT=Path("data/high_volatility_cross_alt_return_turnover_concordance_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_cross_alt_return_turnover_concordance_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_cross_alt_return_turnover_concordance_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_cross_alt_return_turnover_concordance_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_cross_alt_return_turnover_concordance_relay_support_2026-08-11.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","concordance","raw_level_covariance","realized_variation","variation_rank","eligible")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","concordance","raw_level_covariance","realized_variation","variation_rank")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["variation_history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_variation_history_decisions"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def population_covariance(left,right)->float:return float(np.mean((left-np.mean(left))*(right-np.mean(right))))
def prior_median(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(values),np.nan);history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["turnover_history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_turnover_history_decisions"]:out[i]=float(np.median(prior))
  if math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=series.index)

def block_statistics(block:pd.DataFrame)->tuple[list[float],list[float],float]:
 alt_returns=[];turnovers=[]
 for symbol in ALTS:
  x=block.xs(symbol,level="symbol");alt_returns.append(float(np.log(float(x.close.iloc[-1])/float(x.open.iloc[0]))));turnovers.append(float(x.quote_asset_volume.sum()))
 btc=block.xs("BTCUSDT",level="symbol");returns=np.log(btc.close.to_numpy(float)/btc.open.to_numpy(float));variation=float(np.square(returns).sum())
 return alt_returns,turnovers,variation

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
 if frame.columns.tolist()!=["ts","symbol","open","high","low","close","quote_asset_volume"]:raise RuntimeError("HVCARTC source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 x["symbol"]=x.symbol.astype(str)
 for c in ("open","high","low","close","quote_asset_volume"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.duplicated(["ts","symbol"]).any() or not x.symbol.isin(SYMBOLS).all():raise RuntimeError("HVCARTC invalid source key")
 prices=x[["open","high","low","close"]]
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)&np.isfinite(x.quote_asset_volume)&x.quote_asset_volume.ge(0)
 return x.set_index(["ts","symbol"]).sort_index()

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("8h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");expected=pd.MultiIndex.from_product([minutes,SYMBOLS],names=["ts","symbol"]);block=source.reindex(expected);count=int(block.row_valid.eq(True).sum())
  path_valid=len(block)==480*len(SYMBOLS) and bool(block.row_valid.eq(True).all())
  if path_valid:
   returns,turnovers,var=block_statistics(block);valid=all(math.isfinite(v) for v in (*returns,*turnovers,var)) and all(v>0 for v in turnovers) and var>0
  else:returns=[math.nan]*6;turnovers=[math.nan]*6;var=math.nan;valid=False
  row={"decision_time":decision,"feature_available_time":decision,"path_valid":valid,"minute_count":count,"block_variation":var};row.update({f"return_{s}":v for s,v in zip(ALTS,returns)});row.update({f"turnover_{s}":v for s,v in zip(ALTS,turnovers)});rows.append(row)
 panel=pd.DataFrame(rows)
 for s in ALTS:panel[f"baseline_{s}"]=prior_median(panel[f"turnover_{s}"].where(panel.path_valid))
 concordance=[];raw=[]
 for _,r in panel.iterrows():
  returns=np.asarray([r[f"return_{s}"] for s in ALTS],float);turnovers=np.asarray([r[f"turnover_{s}"] for s in ALTS],float);baselines=np.asarray([r[f"baseline_{s}"] for s in ALTS],float)
  if np.isfinite(returns).all() and np.isfinite(turnovers).all() and np.isfinite(baselines).all() and (baselines>0).all():concordance.append(population_covariance(returns,np.log(turnovers/baselines)));raw.append(population_covariance(returns,np.log(turnovers)))
  else:concordance.append(math.nan);raw.append(math.nan)
 panel["concordance"]=concordance;panel["raw_level_covariance"]=raw;panel["realized_variation"]=np.sqrt(panel.block_variation.where(panel.path_valid).rolling(3,min_periods=3).sum());panel["source_valid"]=panel.path_valid&np.isfinite(panel[["concordance","raw_level_covariance","realized_variation"]]).all(axis=1)&panel.concordance.ne(0)&panel.realized_variation.gt(0);panel["variation_rank"]=prior_rank(panel.realized_variation.where(panel.source_valid));panel["eligible"]=panel.source_valid&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_concordance":used=panel.shift(1)
 valid=used.source_valid.eq(True);variation=used.variation_rank.ge(P["variation_rank_min"]);state=valid&variation
 if control=="no_variation_gate":state=valid
 value=used.raw_level_covariance if control=="raw_turnover_level_covariance" else used.concordance;side=np.sign(pd.to_numeric(value,errors="coerce").fillna(0)).astype(int)
 if control=="direction_flip":side=-side
 return state&side.ne(0),side,used

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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCARTC prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvcartc_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvcartc_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
