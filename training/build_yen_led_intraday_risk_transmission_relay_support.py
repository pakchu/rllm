"""Materialize outcome-blind source support for frozen YLIRTR-12."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from training import preregister_yen_led_intraday_risk_transmission_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";BUILDER=Path("training/build_yen_led_intraday_risk_transmission_relay_support.py");PREREG_SHA="eabbd35292e07ea611a9027c5d9c9fd5067f1aab07a897b3732a652f44d449c6";SOURCE_DIR=Path("data/yen_led_intraday_risk_transmission_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"ylirtr_preentry_features.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/yen_led_intraday_risk_transmission_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/yen_led_intraday_risk_transmission_relay_controls_2023_2026");RESULT=Path("results/yen_led_intraday_risk_transmission_relay_support_2026-08-10.json")
BTC_START=pd.Timestamp("2022-12-29T00:00:00Z");FX_START=pd.Timestamp("2023-01-01T00:00:00Z");SOURCE_END=pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_lead_coupling_gate","no_volatility_gate","raw_positive_lead_correlation","one_session_stale_coupling","direction_flip");COLUMNS=("candidate","control","split","session_date","decision_time","feature_available_time","entry_time","exit_time","side","usdjpy_return","lead_correlation","lead_correlation_rank","btc_realized_variation","btc_variation_rank")
BTC_QUERY="SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts";FX_QUERY="SELECT ts,open,high,low,close FROM bars_polygon WHERE symbol='USDJPY' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
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
  btc=pd.read_sql_query(text(BTC_QUERY),e,params={"start":BTC_START.to_pydatetime(),"end":SOURCE_END.to_pydatetime()});fx=pd.read_sql_query(text(FX_QUERY),e,params={"start":FX_START.to_pydatetime(),"end":SOURCE_END.to_pydatetime()})
 finally:e.dispose()
 for label,f in (("BTC",btc),("FX",fx)):
  if f.columns.tolist()!=["ts","open","high","low","close"]:raise RuntimeError(f"YLIRTR {label} schema drift")
  f.ts=pd.to_datetime(f.ts,utc=True,errors="raise");f.sort_values("ts",inplace=True);f.reset_index(drop=True,inplace=True)
  if f.ts.duplicated().any():raise RuntimeError(f"YLIRTR {label} duplicate timestamp")
  for c in ("open","high","low","close"):f[c]=pd.to_numeric(f[c],errors="coerce")
  if not np.isfinite(f[["open","high","low","close"]]).all(axis=None) or not f[["open","high","low","close"]].gt(0).all(axis=None):raise RuntimeError(f"YLIRTR {label} invalid price")
  if not (f.high.ge(f[["open","close"]].max(axis=1)).all() and f.low.le(f[["open","close"]].min(axis=1)).all() and f.high.ge(f.low).all()):raise RuntimeError(f"YLIRTR {label} incoherent OHLC")
 expected=pd.date_range(BTC_START,SOURCE_END,freq="1min",inclusive="left")
 if len(btc)!=len(expected) or not btc.ts.equals(pd.Series(expected,name="ts")):raise RuntimeError("YLIRTR BTC source not exact 1m grid")
 return btc.set_index("ts"),fx.set_index("ts")
def build_features(btc:pd.DataFrame,fx:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for day in pd.date_range("2023-01-02","2026-07-31",freq="B"):
  start=pd.Timestamp(day).tz_localize("UTC")+pd.Timedelta(hours=13);decision=pd.Timestamp(day).tz_localize("UTC")+pd.Timedelta(hours=21)
  session=fx.loc[start:decision-pd.Timedelta(minutes=1)]
  if len(session)<450 or session.index[0]>start+pd.Timedelta(minutes=5) or session.index[-1]<decision-pd.Timedelta(minutes=5):continue
  fx_minute=np.log(session.close/session.open)
  btc_next=np.log(btc.close/btc.open).reindex(session.index+pd.Timedelta(minutes=1))
  valid=np.isfinite(fx_minute.to_numpy())&np.isfinite(btc_next.to_numpy())
  if valid.sum()<449:continue
  left=fx_minute.to_numpy()[valid][:-1];right=btc_next.to_numpy()[valid][:-1]
  if len(left)<448 or np.std(left)<=0 or np.std(right)<=0:continue
  lead=float(np.corrcoef(left,right)[0,1])
  bw=btc.loc[decision-pd.Timedelta(hours=24):decision-pd.Timedelta(minutes=1)]
  if len(bw)!=1440:continue
  fxret=float(np.log(session.close.iloc[-1]/session.open.iloc[0]));variation=float(np.sqrt(np.square(np.log(bw.close.to_numpy()/bw.open.to_numpy())).sum()));rows.append({"session_date":day,"decision_time":decision,"usdjpy_return":fxret,"lead_correlation":lead,"btc_realized_variation":variation})
 f=pd.DataFrame(rows);f["lead_correlation_rank"]=rank(f.lead_correlation);f["btc_variation_rank"]=rank(f.btc_realized_variation);return f
def signal(f:pd.DataFrame,control:str)->pd.Series:
 ret=f.usdjpy_return;correlation=f.lead_correlation;coupling_rank=f.lead_correlation_rank;side=np.sign(ret).astype("Int64").fillna(0).astype(int)
 if control=="one_session_stale_coupling":correlation=correlation.shift(1);coupling_rank=coupling_rank.shift(1)
 coupling=pd.Series(True,index=f.index) if control=="no_lead_coupling_gate" else correlation.gt(0) if control=="raw_positive_lead_correlation" else coupling_rank.ge(.70)
 volatility=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.btc_variation_rank.ge(.65)
 eligible=ret.ne(0)&np.isfinite(correlation)&coupling&volatility
 side=side.where(eligible,0);return -side if control=="direction_flip" else side
def build_clock(f:pd.DataFrame,control="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 sides=signal(f,control);rows=[];next_allowed=None
 for i in f.index[sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_time=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  next_allowed=exit_time;rows.append({"candidate":"YLIRTR-12","control":control,"split":split,"session_date":pd.Timestamp(f.at[i,"session_date"]),"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_time,"side":int(sides.at[i]),"usdjpy_return":float(f.at[i,"usdjpy_return"]),"lead_correlation":float(f.at[i,"lead_correlation"]),"lead_correlation_rank":float(f.at[i,"lead_correlation_rank"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_variation_rank":float(f.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c,s):
 x=c[c.split.eq(s)].copy()
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 e=pd.to_datetime(x.entry_time,utc=True);lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(e.dt.strftime("%Y-%m").value_counts().max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("YLIRTR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);btc,fx=load_sources();features=build_features(btc,fx);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"ylirtr_12_sources_v1","queries":{"btc":BTC_QUERY,"fx":FX_QUERY},"windows":{"btc":[BTC_START.isoformat(),SOURCE_END.isoformat()],"fx":[FX_START.isoformat(),SOURCE_END.isoformat()]},"rows":{"btc":len(btc),"fx":len(fx),"features":len(features)},"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"feature_output":{"path":str(FEATURES),"sha256":sha(FEATURES)},"candidate_outcomes_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,ensure_ascii=False,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM_EVENTS[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"ylirtr_12_source_support_v1","policy_id":"YLIRTR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
