"""Build outcome-blind source support for frozen HVCABRR-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_cross_alt_barrier_rejection_reversal as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-05-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="3da9c8a42036ba3dadc9553ed350018ab71f560dcbc411ce056901433bc2e0ac";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"]);SYMBOLS=("BTCUSDT",*prereg.ALTS);ALTS=prereg.ALTS
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,symbol,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol IN ('BTCUSDT','ADAUSDT','BNBUSDT','DOGEUSDT','ETHUSDT','SOLUSDT','XRPUSDT') AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,symbol ORDER BY 1,symbol"""
ROOT=Path("data/high_volatility_cross_alt_barrier_rejection_reversal_sources_2023_2026");PANEL=ROOT/"five_minute_barrier_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_cross_alt_barrier_rejection_reversal_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_cross_alt_barrier_rejection_reversal_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_cross_alt_barrier_rejection_reversal_controls_2023_2026");RESULT=Path("results/high_volatility_cross_alt_barrier_rejection_reversal_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLS=("decision_time","feature_available_time","source_valid","rejection_side","upper_rejection_count","lower_rejection_count","close_outside_side","close_outside_up_count","close_outside_down_count","btc_realized_variation","variation_rank","variation_active","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","upper_rejection_count","lower_rejection_count","btc_realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(series:pd.Series,window:int,minimum:int)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(values),np.nan);history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-window:],float)
  if math.isfinite(value) and len(prior)>=minimum:out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=series.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 e=postgres_engine()
 try:
  with e.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:e.dispose()
def prepare(raw:pd.DataFrame)->pd.DataFrame:
 expected=["date","symbol","open","high","low","close","minute_squared_return","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=expected:raise RuntimeError("HVCABRR source schema drift")
 x=raw.copy();x["date"]=pd.to_datetime(x.date,utc=True,errors="coerce");x["symbol"]=x.symbol.astype(str)
 for c in ("open","high","low","close","minute_squared_return","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="coerce")
 for c in ("first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="coerce")
 if x.date.isna().any() or x.duplicated(["date","symbol"]).any() or not x.symbol.isin(SYMBOLS).all():raise RuntimeError("HVCABRR invalid source key")
 z=x[["open","high","low","close"]];x["row_valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(x[["open","high","low","close","minute_squared_return"]]).all(axis=1)&z.gt(0).all(axis=1)&x.minute_squared_return.ge(0);return x
def rejection_side(up:pd.DataFrame,down:pd.DataFrame,breadth:int):
 pos=up.sum(axis=1);neg=down.sum(axis=1);side=pd.Series(0,index=up.index,dtype=int);side.loc[pos.ge(breadth)&neg.eq(0)]=1;side.loc[neg.ge(breadth)&pos.eq(0)]=-1;return side,pos.astype(int),neg.astype(int)
def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 x=prepare(raw);grid=pd.date_range(START,END,freq="5min",inclusive="left");close=x.pivot(index="date",columns="symbol",values="close").reindex(index=grid,columns=SYMBOLS);high=x.pivot(index="date",columns="symbol",values="high").reindex(index=grid,columns=SYMBOLS);low=x.pivot(index="date",columns="symbol",values="low").reindex(index=grid,columns=SYMBOLS);sq=x.pivot(index="date",columns="symbol",values="minute_squared_return").reindex(index=grid,columns=SYMBOLS);valid=x.pivot(index="date",columns="symbol",values="row_valid").reindex(index=grid,columns=SYMBOLS).eq(True);alts=close.loc[:,list(ALTS)];prior_high=alts.shift(1).rolling(P["barrier_bars"],min_periods=P["barrier_bars"]).max();prior_low=alts.shift(1).rolling(P["barrier_bars"],min_periods=P["barrier_bars"]).min();history_valid=valid.loc[:,list(ALTS)].shift(1).rolling(P["barrier_bars"],min_periods=P["barrier_bars"]).sum().eq(P["barrier_bars"]);current_valid=valid.loc[:,list(ALTS)];up=high.loc[:,list(ALTS)].gt(prior_high)&alts.le(prior_high)&history_valid&current_valid;down=low.loc[:,list(ALTS)].lt(prior_low)&alts.ge(prior_low)&history_valid&current_valid;side,pos,neg=rejection_side(up,down,P["minimum_rejection_breadth"]);co_up=alts.gt(prior_high)&history_valid&current_valid;co_down=alts.lt(prior_low)&history_valid&current_valid;co_side,co_pos,co_neg=rejection_side(co_up,co_down,P["minimum_rejection_breadth"]);variation=np.sqrt(sq.BTCUSDT.where(valid.BTCUSDT).rolling(P["barrier_bars"],min_periods=P["barrier_bars"]).sum());vrank=strict_prior_midrank(variation,P["variation_history_decisions"],P["minimum_variation_history_decisions"]);source_valid=valid.all(axis=1)&history_valid.all(axis=1)&variation.notna()&vrank.notna();vactive=vrank.ge(P["variation_rank_min"]);broad=side.ne(0);prior_valid=source_valid.shift(1,fill_value=False);prior_broad=broad.shift(1,fill_value=False);eligible=source_valid&prior_valid&broad&~prior_broad&vactive
 return pd.DataFrame({"decision_time":grid+pd.Timedelta("5m"),"feature_available_time":grid+pd.Timedelta("5m"),"source_valid":source_valid.to_numpy(bool),"rejection_side":side.to_numpy(int),"upper_rejection_count":pos.to_numpy(int),"lower_rejection_count":neg.to_numpy(int),"close_outside_side":co_side.to_numpy(int),"close_outside_up_count":co_pos.to_numpy(int),"close_outside_down_count":co_neg.to_numpy(int),"btc_realized_variation":variation.to_numpy(float),"variation_rank":vrank.to_numpy(float),"variation_active":vactive.fillna(False).to_numpy(bool),"eligible":eligible.to_numpy(bool)},columns=PANEL_COLS)
def active(panel:pd.DataFrame,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy();sweep=pd.to_numeric(u.rejection_side,errors="coerce").fillna(0).astype(int);side=-sweep;broad=sweep.ne(0);prior_valid=u.source_valid.shift(1,fill_value=False);state=u.source_valid.eq(True)&prior_valid&broad&~broad.shift(1,fill_value=False)&u.variation_active.eq(True)
 if control=="no_variation_gate":state=u.source_valid.eq(True)&prior_valid&broad&~broad.shift(1,fill_value=False)
 elif control=="three_of_six_rejection":
  pos=u.upper_rejection_count;neg=u.lower_rejection_count;sweep=pd.Series(0,index=u.index,dtype=int);sweep.loc[pos.ge(3)&neg.eq(0)]=1;sweep.loc[neg.ge(3)&pos.eq(0)]=-1;side=-sweep;broad=sweep.ne(0);state=u.source_valid.eq(True)&prior_valid&broad&~broad.shift(1,fill_value=False)&u.variation_active.eq(True)
 elif control=="close_outside_barrier":
  side=pd.to_numeric(u.close_outside_side,errors="coerce").fillna(0).astype(int);broad=side.ne(0);state=u.source_valid.eq(True)&prior_valid&broad&~broad.shift(1,fill_value=False)&u.variation_active.eq(True)
 elif control=="one_bar_stale_rejection":
  cols=["rejection_side","upper_rejection_count","lower_rejection_count","feature_available_time"];u[cols]=panel[cols].shift(1);sweep=pd.to_numeric(u.rejection_side,errors="coerce").fillna(0).astype(int);side=-sweep;broad=sweep.ne(0);state=u.source_valid.eq(True)&prior_valid&broad&~broad.shift(1,fill_value=False)&u.variation_active.eq(True)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=pd.Series(1,index=u.index,dtype=int)
 return state&side.ne(0),side,u
def build_clock(panel,control="primary"):
 act,side,u=active(panel,control);rows=[];reserved=None
 for i in panel.index[act]:
  decision=pd.Timestamp(panel.at[i,"decision_time"]);entry=decision+pd.Timedelta("5m");exit_time=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  reserved=exit_time;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":pd.Timestamp(u.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),**{c:float(u.at[i,c]) for c in CLOCK_COLS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLS)
def stats(c,split):
 x=c[c.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def csv_gz(x):
 b=io.BytesIO();raw=x.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(p:Path,b:bytes):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=b:raise RuntimeError(f"refusing overwrite {p}")
 p.write_bytes(b)
def jb(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCABRR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvcabrr_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={key:ok for n,x in support.items() for key,ok in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcabrr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
