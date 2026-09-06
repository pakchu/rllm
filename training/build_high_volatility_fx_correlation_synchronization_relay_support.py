"""Materialize source-only HVFXCSR-12 support clocks."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
if __package__ in (None, ""):
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from training import preregister_high_volatility_fx_correlation_synchronization_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-01-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="802764a22fac7331570843626679dde3ba01b37b27c3002d407af9e976fa1311"
SOURCE_DIR=Path("data/high_volatility_fx_correlation_synchronization_relay_sources_2023_2026"); SESSION=SOURCE_DIR/"fx_synchronization_sessions.csv.gz"; SOURCE_MANIFEST=SOURCE_DIR/"manifest.json"
PRICE=Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz"); PRICE_SHA="f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"; PRICE_MANIFEST=PRICE.parent/"manifest.json"; PRICE_MANIFEST_SHA="3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
CLOCK=Path("data/high_volatility_fx_correlation_synchronization_relay_clocks_2023_2026.csv.gz"); CONTROL_DIR=Path("data/high_volatility_fx_correlation_synchronization_relay_controls_2023_2026"); RESULT=Path("results/high_volatility_fx_correlation_synchronization_relay_support_2026-08-10.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))}; MINIMUM={"train":8,"test":12,"eval":12,"final":8}; CONTROLS=("no_volatility_gate","no_synchronization_gate","one_session_stale_features","direction_flip","forced_long")
SYMBOLS=("EURUSD","GBPUSD","USDAUD","USDCAD","USDCHF","USDJPY"); DOLLAR_MULTIPLIER={"EURUSD":-1.,"GBPUSD":-1.,"USDAUD":1.,"USDCAD":1.,"USDCHF":1.,"USDJPY":1.}
COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","fx_synchronization","synchronization_rank","dollar_direction_factor","btc_realized_variation","btc_realized_variation_rank")
QUERY="""SELECT ts,symbol,open,high,low,close FROM bars_polygon WHERE symbol IN ('EURUSD','GBPUSD','USDAUD','USDCAD','USDCHF','USDJPY') AND interval='1m' AND ts>=:start AND ts<:end AND extract(isodow from ts) BETWEEN 1 AND 5 AND extract(hour from ts)>=13 AND extract(hour from ts)<21 ORDER BY ts,symbol"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 n=pd.to_numeric(v,errors="coerce").astype(float);out=pd.Series(np.nan,index=n.index,dtype=float);history=[]
 for i,current in n.items():
  prior=history[-lookback:]
  if math.isfinite(current) and len(prior)>=minimum:
   a=np.asarray(prior);out.at[i]=(np.sum(a<current)+.5*np.sum(a==current))/len(a)
  if math.isfinite(current):history.append(current)
 return out
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def materialize_sessions()->dict:
 from sqlalchemy import text
 db=postgres_engine()
 with db.connect() as connection:
  raw=pd.read_sql_query(text(QUERY),connection,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 db.dispose();raw["ts"]=pd.to_datetime(raw.ts,utc=True);raw["source_day"]=raw.ts.dt.floor("D")
 for column in ("open","high","low","close"):raw[column]=pd.to_numeric(raw[column],errors="coerce")
 if raw.duplicated(["ts","symbol"]).any():raise RuntimeError("duplicate HVFXCSR source timestamps")
 rows=[]
 for day,day_frame in raw.groupby("source_day",sort=True):
  returns={};session_returns={};all_valid=True
  for symbol in SYMBOLS:
   x=day_frame[day_frame.symbol.eq(symbol)].sort_values("ts")
   valid=len(x)>=450 and x.ts.nunique()>=450 and x.ts.min()<=day+pd.Timedelta(hours=13,minutes=5) and x.ts.max()>=day+pd.Timedelta(hours=20,minutes=55) and np.isfinite(x[["open","high","low","close"]]).all().all() and x[["open","high","low","close"]].gt(0).all().all() and x.high.ge(x[["open","close"]].max(axis=1)).all() and x.low.le(x[["open","close"]].min(axis=1)).all()
   all_valid=all_valid and valid
   if valid:
    close=x.set_index("ts").close.astype(float);returns[symbol]=(DOLLAR_MULTIPLIER[symbol]*np.log(close/close.shift(1))).dropna();session_returns[symbol]=float(DOLLAR_MULTIPLIER[symbol]*np.log(close.iloc[-1]/close.iloc[0]))
  correlations=[]
  if all_valid:
   for a_i,a in enumerate(SYMBOLS):
    for b in SYMBOLS[a_i+1:]:
     pair=pd.concat([returns[a],returns[b]],axis=1,join="inner").dropna()
     if len(pair)<420 or pair.iloc[:,0].std(ddof=1)<=0 or pair.iloc[:,1].std(ddof=1)<=0:all_valid=False;break
     correlations.append(float(pair.iloc[:,0].corr(pair.iloc[:,1])))
    if not all_valid:break
  synchronization=float(np.median(correlations)) if all_valid and len(correlations)==15 else math.nan
  direction=float(np.median(list(session_returns.values()))) if all_valid else math.nan
  rows.append({"source_day":day,"source_valid":bool(all_valid and math.isfinite(synchronization) and math.isfinite(direction) and direction!=0),"fx_synchronization":synchronization,"dollar_direction_factor":direction,"minimum_pairwise_returns":420})
 frame=pd.DataFrame(rows).sort_values("source_day").reset_index(drop=True);frame["synchronization_rank"]=strict_prior_midrank(frame.fx_synchronization.where(frame.source_valid));frame["decision_time"]=frame.source_day+pd.Timedelta(hours=21)
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(frame,SESSION);core={"protocol_version":"hvfxcsr_12_fx_synchronization_source_v1","query":QUERY,"table":"bars_polygon","symbols":list(SYMBOLS),"canonical_dollar_multipliers":DOLLAR_MULTIPLIER,"interval":"1m","window":[START.isoformat(),END.isoformat()],"outcomes_opened":False,"candidate_incidence_opened":False,"no_imputation":True,"output":{"path":str(SESSION),"sha256":sha(SESSION),"rows":len(frame),"valid_rows":int(frame.source_valid.sum())}};payload={**core,"manifest_hash":chash(core)};SOURCE_MANIFEST.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");return payload
def features()->pd.DataFrame:
 if sha(PRICE)!=PRICE_SHA or sha(PRICE_MANIFEST)!=PRICE_MANIFEST_SHA:raise RuntimeError("HVFXCSR BTC source drift")
 f=pd.read_csv(SESSION,compression="gzip");f["source_day"]=pd.to_datetime(f.source_day,utc=True);f["decision_time"]=pd.to_datetime(f.decision_time,utc=True);f["source_valid"]=f.source_valid.astype(str).str.lower().eq("true")
 for column in ("fx_synchronization","synchronization_rank","dollar_direction_factor"):f[column]=pd.to_numeric(f[column],errors="coerce")
 p=pd.read_csv(PRICE,compression="gzip");p["decision_time"]=pd.to_datetime(p.decision_time,utc=True,format="mixed");p["open"]=pd.to_numeric(p.open,errors="coerce");p["close"]=pd.to_numeric(p.close,errors="coerce");p["valid"]=p.source_valid.astype(str).str.lower().eq("true")&np.isfinite(p[["open","close"]]).all(axis=1)&p[["open","close"]].gt(0).all(axis=1);p=p.sort_values("decision_time").reset_index(drop=True);p["hour_return"]=np.log(p.close/p.open);consecutive=p.decision_time.diff().eq(pd.Timedelta(hours=1));p["btc_realized_variation"]=np.sqrt(p.hour_return.pow(2).rolling(24,min_periods=24).sum());p["btc_valid"]=p.valid.rolling(24,min_periods=24).sum().eq(24)&consecutive.rolling(23,min_periods=23).sum().eq(23)&np.isfinite(p.btc_realized_variation)
 d=p[["decision_time","btc_realized_variation","btc_valid"]];f=f.merge(d,on="decision_time",how="left",validate="one_to_one");f["btc_realized_variation_rank"]=strict_prior_midrank(f.btc_realized_variation.where(f.btc_valid));f["signal_valid"]=f.source_valid&f.btc_valid&np.isfinite(f[["fx_synchronization","synchronization_rank","dollar_direction_factor","btc_realized_variation","btc_realized_variation_rank"]]).all(axis=1)&f.dollar_direction_factor.ne(0);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 factor=f.dollar_direction_factor;rank=f.synchronization_rank;vol_rank=f.btc_realized_variation_rank
 if control=="one_session_stale_features":factor=factor.shift(1);rank=rank.shift(1);vol_rank=vol_rank.shift(1)
 sync=pd.Series(True,index=f.index) if control=="no_synchronization_gate" else rank.ge(.75);vol=pd.Series(True,index=f.index) if control=="no_volatility_gate" else vol_rank.ge(.65);active=f.signal_valid&np.isfinite(factor)&factor.ne(0)&sync&vol;side=-np.sign(factor)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=pd.Series(1.,index=f.index)
 return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"HVFXCSR-12","control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"fx_synchronization":float(f.at[i,"fx_synchronization"]),"synchronization_rank":float(f.at[i,"synchronization_rank"]),"dollar_direction_factor":float(f.at[i,"dollar_direction_factor"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_realized_variation_rank":float(f.at[i,"btc_realized_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVFXCSR preregistration hash drift")
 sm=materialize_sessions();f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"hvfxcsr_12_source_support_v1","policy_id":"HVFXCSR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"fx_synchronization":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_btc":{"path":str(PRICE_MANIFEST),"sha256":sha(PRICE_MANIFEST)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};payload={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");return payload
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
