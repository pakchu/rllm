"""Build outcome-blind source support for frozen HVCASFR-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_cross_alt_synchronized_flip_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-01-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="dde630cabdc943a225bada19fd5e2f6000698ae50514d919450014b9c351a5bb"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
SYMBOLS=("BTCUSDT","ADAUSDT","BNBUSDT","DOGEUSDT","ETHUSDT","SOLUSDT","XRPUSDT");ALTS=SYMBOLS[1:]
QUERY="""SELECT ts,symbol,open,high,low,close FROM bars_binance WHERE symbol IN ('BTCUSDT','ADAUSDT','BNBUSDT','DOGEUSDT','ETHUSDT','SOLUSDT','XRPUSDT') AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts,symbol"""
ROOT=Path("data/high_volatility_cross_alt_synchronized_flip_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_cross_alt_synchronized_flip_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_cross_alt_synchronized_flip_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_cross_alt_synchronized_flip_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_cross_alt_synchronized_flip_relay_support_2026-08-12.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","first_positive_count","second_positive_count","first_majority","second_majority","symbol_flip_count","majority_flip","realized_variation","variation_rank","eligible")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*PANEL_COLUMNS[4:12])

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_decisions"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def flip_statistics(alt_block:pd.DataFrame,btc_block:pd.DataFrame)->tuple[int,int,int,int,int,bool,float]:
 first_signs=[];second_signs=[]
 for symbol in ALTS:
  x=alt_block.xs(symbol,level="symbol");first=x.iloc[:P["half_minutes"]];second=x.iloc[P["half_minutes"]:]
  first_return=float(np.log(float(first.close.iloc[-1])/float(first.open.iloc[0])));second_return=float(np.log(float(second.close.iloc[-1])/float(second.open.iloc[0])))
  if not math.isfinite(first_return) or not math.isfinite(second_return) or first_return==0 or second_return==0:return (0,0,0,0,0,False,math.nan)
  first_signs.append(int(np.sign(first_return)));second_signs.append(int(np.sign(second_return)))
 first_positive=sum(value>0 for value in first_signs);second_positive=sum(value>0 for value in second_signs)
 first_sum=sum(first_signs);second_sum=sum(second_signs);first_majority=int(np.sign(first_sum));second_majority=int(np.sign(second_sum));flip_count=sum(a==-b for a,b in zip(first_signs,second_signs));majority_flip=first_majority!=0 and second_majority==-first_majority
 btc_returns=np.log(btc_block.close.to_numpy(float)/btc_block.open.to_numpy(float));variation=float(np.square(btc_returns).sum())
 return first_positive,second_positive,first_majority,second_majority,flip_count,majority_flip,variation

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
 if frame.columns.tolist()!=["ts","symbol","open","high","low","close"]:raise RuntimeError("HVCASFR source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 x["symbol"]=x.symbol.astype(str)
 for c in ("open","high","low","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.duplicated(["ts","symbol"]).any() or not x.symbol.isin(SYMBOLS).all():raise RuntimeError("HVCASFR invalid source key")
 prices=x[["open","high","low","close"]]
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
 return x.set_index(["ts","symbol"]).sort_index()

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("25h"),END,freq="8h",inclusive="left"):
  alt_minutes=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");btc_minutes=pd.date_range(decision-pd.Timedelta("24h"),decision,freq="1min",inclusive="left")
  alt_expected=pd.MultiIndex.from_product([alt_minutes,ALTS],names=["ts","symbol"]);btc_expected=pd.MultiIndex.from_product([btc_minutes,("BTCUSDT",)],names=["ts","symbol"])
  alt_block=source.reindex(alt_expected);btc_block=source.reindex(btc_expected);count=int(alt_block.row_valid.eq(True).sum()+btc_block.row_valid.eq(True).sum())
  path_valid=len(alt_block)==480*len(ALTS) and len(btc_block)==1440 and bool(alt_block.row_valid.eq(True).all()) and bool(btc_block.row_valid.eq(True).all())
  if path_valid:
   first_positive,second_positive,first_majority,second_majority,flip_count,majority_flip,var=flip_statistics(alt_block,btc_block);valid=math.isfinite(var) and var>0 and first_majority!=0 and second_majority!=0
  else:first_positive=second_positive=first_majority=second_majority=flip_count=0;majority_flip=False;var=math.nan;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"first_positive_count":first_positive,"second_positive_count":second_positive,"first_majority":first_majority,"second_majority":second_majority,"symbol_flip_count":flip_count,"majority_flip":majority_flip,"realized_variation":var})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["variation_rank"]=prior_rank(panel.realized_variation.where(valid));panel["eligible"]=valid&panel.majority_flip.eq(True)&panel.symbol_flip_count.ge(P["minimum_symbol_flips"])&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_features":
  cols=["source_valid","first_positive_count","second_positive_count","first_majority","second_majority","symbol_flip_count","majority_flip","variation_rank","feature_available_time"];used[cols]=panel[cols].shift(1)
 valid=used.source_valid.eq(True);variation=used.variation_rank.ge(P["variation_rank_min"]);majority=used.majority_flip.eq(True);enough=used.symbol_flip_count.ge(P["minimum_symbol_flips"]);state=valid&variation&majority&enough;side=pd.to_numeric(used.second_majority,errors="coerce").fillna(0).astype(int)
 if control=="no_variation_gate":state=valid&majority&enough
 elif control=="three_symbol_flips":state=valid&variation&majority&used.symbol_flip_count.ge(3)
 elif control=="majority_flip_only":state=valid&variation&majority
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
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
def json_bytes(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()

def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCASFR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvcasfr_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvcasfr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
