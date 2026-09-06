"""Build outcome-blind source support for frozen HVPSAR-24."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_parabolic_sar_reversal_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2020-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="5ce0a2137c1afbd11d5739b1595aa0ccaa79adc45ec01e6ff45637de02f38cc8";REG=prereg.build();P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_parabolic_sar_reversal_relay_sources_2023_2026");PANEL=ROOT/"four_hour_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_parabolic_sar_reversal_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_parabolic_sar_reversal_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_parabolic_sar_reversal_relay_controls_2023_2026");RESULT=Path("results/high_volatility_parabolic_sar_reversal_relay_support_2026-08-11.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("source_start","feature_available_time","source_valid","bar_open","bar_high","bar_low","bar_close","sar","trend","extreme","acceleration","reversal","unclamped_sar","unclamped_trend","unclamped_reversal","variation","variation_rank","eligible")
CLOCK_COLUMNS=("candidate","control","split","source_start","feature_available_time","entry_time","exit_time","side","bar_close","sar","trend","extreme","acceleration","reversal","unclamped_sar","unclamped_trend","unclamped_reversal","variation","variation_rank","eligible")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float);out=np.full(len(values),np.nan);history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["variation_history_decisions"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_variation_history_decisions"]:out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=series.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def prepare(frame:pd.DataFrame)->pd.DataFrame:
 if frame.columns.tolist()!=["ts","open","high","low","close"]:raise RuntimeError("HVPSAR source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVPSAR invalid source key")
 prices=x[["open","high","low","close"]];x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low);x["minute_sq_return"]=np.square(np.log(x.close/x.open)).where(x.row_valid);return x.set_index("ts").sort_index()
def parabolic_sar(high:pd.Series,low:pd.Series,close:pd.Series,valid:pd.Series,clamp:bool=True)->pd.DataFrame:
 n=len(high);sar=np.full(n,np.nan);trend=np.full(n,np.nan);extreme=np.full(n,np.nan);acceleration=np.full(n,np.nan);reversal=np.zeros(n,dtype=bool);active=False
 for i in range(n):
  if not bool(valid.iloc[i]):active=False;continue
  if not active:
   if i==0 or not bool(valid.iloc[i-1]):continue
   is_long=float(close.iloc[i])>=float(close.iloc[i-1]);trend[i]=1 if is_long else -1;sar[i]=min(float(low.iloc[i-1]),float(low.iloc[i])) if is_long else max(float(high.iloc[i-1]),float(high.iloc[i]));extreme[i]=max(float(high.iloc[i-1]),float(high.iloc[i])) if is_long else min(float(low.iloc[i-1]),float(low.iloc[i]));acceleration[i]=P["acceleration_start"];active=True;continue
  prior_trend=int(trend[i-1]);projected=float(sar[i-1])+float(acceleration[i-1])*(float(extreme[i-1])-float(sar[i-1]))
  if clamp and prior_trend==1:projected=min(projected,float(low.iloc[i-1]),float(low.iloc[i-2]) if i>=2 and bool(valid.iloc[i-2]) else float(low.iloc[i-1]))
  if clamp and prior_trend==-1:projected=max(projected,float(high.iloc[i-1]),float(high.iloc[i-2]) if i>=2 and bool(valid.iloc[i-2]) else float(high.iloc[i-1]))
  if prior_trend==1 and float(low.iloc[i])<projected:trend[i]=-1;sar[i]=float(extreme[i-1]);extreme[i]=float(low.iloc[i]);acceleration[i]=P["acceleration_start"];reversal[i]=True
  elif prior_trend==-1 and float(high.iloc[i])>projected:trend[i]=1;sar[i]=float(extreme[i-1]);extreme[i]=float(high.iloc[i]);acceleration[i]=P["acceleration_start"];reversal[i]=True
  else:
   trend[i]=prior_trend;sar[i]=projected;new_extreme=(prior_trend==1 and float(high.iloc[i])>float(extreme[i-1])) or (prior_trend==-1 and float(low.iloc[i])<float(extreme[i-1]));extreme[i]=float(high.iloc[i]) if prior_trend==1 and new_extreme else float(low.iloc[i]) if prior_trend==-1 and new_extreme else float(extreme[i-1]);acceleration[i]=min(P["acceleration_max"],float(acceleration[i-1])+P["acceleration_step"]) if new_extreme else float(acceleration[i-1])
 return pd.DataFrame({"sar":sar,"trend":trend,"extreme":extreme,"acceleration":acceleration,"reversal":reversal},index=high.index)

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw).reindex(pd.date_range(START,END,freq="1min",inclusive="left"));g=source.groupby(source.index.floor("4h"),sort=True);bars=pd.DataFrame({"rows":g.row_valid.sum(),"bar_open":g.open.first(),"bar_high":g.high.max(),"bar_low":g.low.min(),"bar_close":g.close.last(),"variation_component":g.minute_sq_return.sum(min_count=240)});bars["valid_bar"]=bars.rows.eq(240)&np.isfinite(bars[["bar_open","bar_high","bar_low","bar_close"]]).all(axis=1)&bars[["bar_open","bar_high","bar_low","bar_close"]].gt(0).all(axis=1);primary=parabolic_sar(bars.bar_high,bars.bar_low,bars.bar_close,bars.valid_bar,True);control=parabolic_sar(bars.bar_high,bars.bar_low,bars.bar_close,bars.valid_bar,False).rename(columns={"sar":"unclamped_sar","trend":"unclamped_trend","reversal":"unclamped_reversal","extreme":"unused_extreme","acceleration":"unused_acceleration"});bars=pd.concat([bars,primary,control[["unclamped_sar","unclamped_trend","unclamped_reversal"]]],axis=1);bars["variation"]=np.sqrt(bars.variation_component.rolling(P["variation_hours"]//4,min_periods=P["variation_hours"]//4).sum());bars["source_valid"]=bars.valid_bar&np.isfinite(bars[["sar","trend","extreme","acceleration","unclamped_sar","unclamped_trend","variation"]]).all(axis=1)&bars.variation.gt(0);panel=bars.reset_index(names="source_start");panel["feature_available_time"]=panel.source_start+pd.Timedelta("4h");panel["variation_rank"]=prior_rank(panel.variation.where(panel.source_valid));panel["eligible"]=panel.source_valid&panel.reversal.eq(True)&panel.variation_rank.ge(P["variation_rank_min"]);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy();valid=used.source_valid.eq(True);reversal=used.unclamped_reversal.eq(True) if control=="no_two_bar_clamp" else used.reversal.eq(True);trend=pd.to_numeric(used.unclamped_trend if control=="no_two_bar_clamp" else used.trend,errors="coerce").fillna(0).astype(int);variation=used.variation_rank.ge(P["variation_rank_min"]);state=valid&reversal&variation;side=trend
 if control=="no_variation_gate":state=valid&reversal
 elif control=="one_bar_stale_reversal":state=state.shift(1,fill_value=False);side=side.shift(1,fill_value=0)
 elif control=="direction_flip":side=-side
 return state&side.ne(0),side,used

def build_clock(panel:pd.DataFrame,control:str="primary")->pd.DataFrame:
 activity,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[activity]:
  decision=pd.Timestamp(panel.at[i,"feature_available_time"]);entry=decision+pd.Timedelta(minutes=P["entry_delay_minutes"]);exit_time=entry+pd.Timedelta(hours=P["hold_hours"])
  if reserved is not None and entry<reserved:continue
  split=next((name for name,(start,end) in SPLITS.items() if entry>=start and exit_time<=end),None)
  if split is None:continue
  reserved=exit_time;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"source_start":pd.Timestamp(used.at[i,"source_start"]),"feature_available_time":decision,"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),**{c:bool(used.at[i,c]) if c=="eligible" else float(used.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 longs=int(x.side.eq(1).sum());shorts=int(x.side.eq(-1).sum());months=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":longs,"shorts":shorts,"minority_side_share":min(longs,shorts)/len(x),"max_month_share":int(months.max())/len(x)}
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVPSAR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvpsar_24_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":canonical_hash(source_core)};immutable(MANIFEST,json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvpsar_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":canonical_hash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
