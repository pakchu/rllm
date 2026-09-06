"""Materialize outcome-blind source support for frozen IRBR-12."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from training import preregister_indian_rupee_risk_barometer_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";BUILDER=Path("training/build_indian_rupee_risk_barometer_relay_support.py");PREREG_SHA="ed832bab6c1122499993030fcece97caf94111b8c6bf5472ade534021a3200c0";SOURCE_DIR=Path("data/indian_rupee_risk_barometer_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"irbr_preentry_features.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/indian_rupee_risk_barometer_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/indian_rupee_risk_barometer_relay_controls_2023_2026");RESULT=Path("results/indian_rupee_risk_barometer_relay_support_2026-08-10.json")
BTC_START=pd.Timestamp("2022-12-29T00:00:00Z");FX_START=pd.Timestamp("2022-01-01T00:00:00Z");SOURCE_END=pd.Timestamp("2026-08-01T00:00:00Z");NY=ZoneInfo("America/New_York")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_fx_tail","no_volatility_gate","one_session_stale_fx","direction_flip","forced_long");COLUMNS=("candidate","control","split","session_date","decision_time","feature_available_time","entry_time","exit_time","side","usdinr_return","absolute_fx_return_rank","btc_realized_variation","btc_variation_rank")
BTC_QUERY="SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts";FX_QUERY="SELECT ts,open,high,low,close FROM bars_polygon WHERE symbol='USDINR' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rank(v:pd.Series,lookback=252,minimum=126)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=h[-lookback:]
  if np.isfinite(c) and len(q)>=minimum:
   a=np.asarray(q);o.at[i]=(np.count_nonzero(a<c)+.5*np.count_nonzero(a==c))/len(a)
  if np.isfinite(c):h.append(float(c))
 return o
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_sources()->tuple[pd.DataFrame,pd.DataFrame]:
 from sqlalchemy import text
 e=engine()
 try:
  btc=pd.read_sql_query(text(BTC_QUERY),e,params={"start":BTC_START.to_pydatetime(),"end":SOURCE_END.to_pydatetime()})
  fx=pd.read_sql_query(text(FX_QUERY),e,params={"start":FX_START.to_pydatetime(),"end":SOURCE_END.to_pydatetime()})
 finally:e.dispose()
 for label,f,required in (("BTC",btc,["ts","open","close"]),("FX",fx,["ts","open","high","low","close"])):
  if f.columns.tolist()!=required:raise RuntimeError(f"IRBR {label} schema drift")
  f.ts=pd.to_datetime(f.ts,utc=True,errors="raise");f.sort_values("ts",inplace=True);f.reset_index(drop=True,inplace=True)
  if f.ts.duplicated().any():raise RuntimeError(f"IRBR {label} duplicate timestamp")
  for c in required[1:]:f[c]=pd.to_numeric(f[c],errors="coerce")
  if not np.isfinite(f[required[1:]]).all(axis=None) or not f[required[1:]].gt(0).all(axis=None):raise RuntimeError(f"IRBR {label} invalid price")
 expected=pd.date_range(BTC_START,SOURCE_END,freq="1min",inclusive="left")
 if len(btc)!=len(expected) or not btc.ts.equals(pd.Series(expected,name="ts")):raise RuntimeError("IRBR BTC source not exact 1m grid")
 return btc.set_index("ts"),fx.set_index("ts")

def build_features(btc:pd.DataFrame,fx:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for day in pd.date_range("2022-01-03","2026-07-31",freq="B"):
  start=pd.Timestamp(day,tz="UTC")+pd.Timedelta(hours=3,minutes=45);decision=pd.Timestamp(day,tz="UTC")+pd.Timedelta(hours=10)
  session=fx[(fx.index>=start)&(fx.index<decision)].copy()
  valid=(len(session)>=270 and not session.index.duplicated().any() and session.index.min()<=start+pd.Timedelta(minutes=15) and session.index.max()>=decision-pd.Timedelta(minutes=15) and session.high.ge(session[["open","close"]].max(axis=1)).all() and session.low.le(session[["open","close"]].min(axis=1)).all() and session.high.ge(session.low).all())
  if not valid:continue
  bw=btc.loc[decision-pd.Timedelta(hours=24):decision-pd.Timedelta(minutes=1)]
  if len(bw)!=1440:continue
  fxret=float(np.log(session.close.iloc[-1]/session.open.iloc[0]));variation=float(np.sqrt(np.square(np.log(bw.close.to_numpy()/bw.open.to_numpy())).sum()))
  rows.append({"session_date":day,"decision_time":decision,"usdinr_return":fxret,"btc_realized_variation":variation,"fx_source_minutes":len(session)})
 f=pd.DataFrame(rows);f["absolute_fx_return_rank"]=rank(f.usdinr_return.abs());f["btc_variation_rank"]=rank(f.btc_realized_variation);return f

def signal(f:pd.DataFrame,control:str)->pd.Series:
 ret=f.usdinr_return;side=-np.sign(ret).astype("Int64").fillna(0).astype(int);tail=f.absolute_fx_return_rank.ge(.70);vol=f.btc_variation_rank.ge(.65);eligible=ret.ne(0)&tail&vol
 if control=="no_fx_tail":eligible=ret.ne(0)&vol
 elif control=="no_volatility_gate":eligible=ret.ne(0)&tail
 elif control=="one_session_stale_fx":
  ret=ret.shift(1);tail=f.absolute_fx_return_rank.shift(1).ge(.70);side=-np.sign(ret).astype("Int64").fillna(0).astype(int);eligible=ret.ne(0)&tail&vol
 side=side.where(eligible,0)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.abs()
 return side

def build_clock(f:pd.DataFrame,control="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 sides=signal(f,control);rows=[];next_allowed=None
 for i in f.index[sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_time=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  next_allowed=exit_time;rows.append({"candidate":"IRBR-12","control":control,"split":split,"session_date":pd.Timestamp(f.at[i,"session_date"]),"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_time,"side":int(sides.at[i]),"usdinr_return":float(f.at[i,"usdinr_return"]),"absolute_fx_return_rank":float(f.at[i,"absolute_fx_return_rank"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_variation_rank":float(f.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c,s):
 x=c[c.split.eq(s)].copy()
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 e=pd.to_datetime(x.entry_time,utc=True);lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(e.dt.strftime("%Y-%m").value_counts().max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("IRBR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);btc,fx=load_sources();features=build_features(btc,fx);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"irbr_12_sources_v1","queries":{"btc":BTC_QUERY,"fx":FX_QUERY},"windows":{"btc":[BTC_START.isoformat(),SOURCE_END.isoformat()],"fx":[FX_START.isoformat(),SOURCE_END.isoformat()]},"rows":{"btc":len(btc),"fx":len(fx),"features":len(features)},"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"feature_output":{"path":str(FEATURES),"sha256":sha(FEATURES)},"candidate_outcomes_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,ensure_ascii=False,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM_EVENTS[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"irbr_12_source_support_v1","policy_id":"IRBR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
