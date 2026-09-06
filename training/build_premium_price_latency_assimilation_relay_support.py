"""Build source-only PPLAR-8 clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_premium_price_latency_assimilation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="1dc4df19b2621fb54e346f3754c79c2e496adb08fbfbb2b06634d41920306e99";START=pd.Timestamp("2023-04-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("premium_persistence_without_underresponse","late_btc_return_only","contemporaneous_agreement","one_hour_stale_premium","direction_flip")
QUERY="""SELECT date_bin('1 hour',p.ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS hour_start,(array_agg(p.open ORDER BY p.ts))[1] AS premium_open,(array_agg(p.close ORDER BY p.ts DESC))[1] AS premium_close,(array_agg(b.open ORDER BY b.ts))[1] AS btc_open,(array_agg(b.close ORDER BY b.ts DESC))[1] AS btc_close,count(*) AS source_rows,count(DISTINCT p.ts) AS premium_distinct_rows,count(DISTINCT b.ts) AS btc_distinct_rows,min(p.ts) AS first_ts,max(p.ts) AS last_ts,bool_and(p.open IS NOT NULL AND p.high IS NOT NULL AND p.low IS NOT NULL AND p.close IS NOT NULL AND p.high>=greatest(p.open,p.close) AND p.low<=least(p.open,p.close) AND p.high>=p.low AND b.open>0 AND b.high>0 AND b.low>0 AND b.close>0 AND b.high>=greatest(b.open,b.close) AND b.low<=least(b.open,b.close) AND b.high>=b.low) AS coherent FROM bars_binance_premium p JOIN bars_binance b ON b.symbol='BTCUSDT' AND b.interval='1m' AND b.ts=p.ts WHERE p.symbol='BTCUSDT' AND p.interval='1m' AND p.ts>=:start AND p.ts<:end GROUP BY hour_start ORDER BY hour_start"""
SOURCE_DIR=Path("data/premium_price_latency_assimilation_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"two_hour_assimilation.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/premium_price_latency_assimilation_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/premium_price_latency_assimilation_relay_controls_2023_2026");RESULT=Path("results/premium_price_latency_assimilation_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","stale_source_valid","early_premium_change","late_premium_change","early_btc_return","late_btc_return","stale_early_premium_change","stale_late_premium_change","early_premium_rank","early_btc_rank","late_btc_abs_median","late_btc_rank","late_premium_rank","stale_early_premium_rank")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*FEATURE_COLUMNS[4:])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def prior_midrank(v:pd.Series,lookback:int=540,minimum:int=360)->pd.Series:
 out=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in pd.to_numeric(v,errors="coerce").items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:
   a=np.asarray(p);out.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(float(x))
 return out
def prior_median(v:pd.Series,lookback:int=540,minimum:int=360)->pd.Series:
 out=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in pd.to_numeric(v,errors="coerce").items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:out.at[i]=float(np.median(p))
  if math.isfinite(x):h.append(float(x))
 return out
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def build_features(raw:pd.DataFrame)->pd.DataFrame:
 req=["hour_start","premium_open","premium_close","btc_open","btc_close","source_rows","premium_distinct_rows","btc_distinct_rows","first_ts","last_ts","coherent"]
 if not set(req).issubset(raw):raise ValueError("PPLAR schema drift")
 f=raw[req].copy()
 for c in ("hour_start","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,errors="coerce")
 for c in ("premium_open","premium_close","btc_open","btc_close","source_rows","premium_distinct_rows","btc_distinct_rows"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values("hour_start",kind="mergesort").set_index("hour_start");rows=[]
 for d in pd.date_range(START+pd.Timedelta(hours=4),END,freq="2h",inclusive="left"):
  expected=pd.DatetimeIndex([d-pd.Timedelta(hours=3),d-pd.Timedelta(hours=2),d-pd.Timedelta(hours=1)]);w=f.reindex(expected)
  valid=w.source_rows.eq(60)&w.premium_distinct_rows.eq(60)&w.btc_distinct_rows.eq(60)&w.coherent.fillna(False).astype(bool)&np.isfinite(w[["premium_open","premium_close","btc_open","btc_close"]]).all(axis=1)&w.btc_open.gt(0)&w.btc_close.gt(0)&w.first_ts.eq(pd.Series(expected,index=expected))&w.last_ts.eq(pd.Series(expected+pd.Timedelta(minutes=59),index=expected))
  current=bool(valid.iloc[1:].all());stale=bool(valid.all())
  if current:
   pe=float(w.premium_close.iloc[1]-w.premium_open.iloc[1]);pl=float(w.premium_close.iloc[2]-w.premium_open.iloc[2]);re=float(np.log(w.btc_close.iloc[1]/w.btc_open.iloc[1]));rl=float(np.log(w.btc_close.iloc[2]/w.btc_open.iloc[2]))
  else:pe=pl=re=rl=np.nan
  if stale:spe=float(w.premium_close.iloc[0]-w.premium_open.iloc[0]);spl=float(w.premium_close.iloc[1]-w.premium_open.iloc[1])
  else:spe=spl=np.nan
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":current,"stale_source_valid":stale,"early_premium_change":pe,"late_premium_change":pl,"early_btc_return":re,"late_btc_return":rl,"stale_early_premium_change":spe,"stale_late_premium_change":spl})
 out=pd.DataFrame(rows);out["early_premium_rank"]=prior_midrank(out.early_premium_change.abs().where(out.source_valid));out["early_btc_rank"]=prior_midrank(out.early_btc_return.abs().where(out.source_valid));out["late_btc_abs_median"]=prior_median(out.late_btc_return.abs().where(out.source_valid));out["late_btc_rank"]=prior_midrank(out.late_btc_return.abs().where(out.source_valid));out["late_premium_rank"]=prior_midrank(out.late_premium_change.abs().where(out.source_valid));out["stale_early_premium_rank"]=prior_midrank(out.stale_early_premium_change.abs().where(out.stale_source_valid));return out[list(FEATURE_COLUMNS)]
def onset(x:pd.Series)->pd.Series:return x.fillna(False)&~x.shift(1,fill_value=False)
def state_primary(f:pd.DataFrame,underresponse:bool=True,stale:bool=False):
 pe=f.stale_early_premium_change if stale else f.early_premium_change;pl=f.stale_late_premium_change if stale else f.late_premium_change;rank=f.stale_early_premium_rank if stale else f.early_premium_rank;valid=f.stale_source_valid if stale else f.source_valid
 state=valid&pe.ne(0)&rank.ge(.8)&pe.mul(pl).gt(0)&f.late_btc_return.ne(0)&np.sign(f.late_btc_return).eq(np.sign(pe))&f.late_btc_return.abs().le(.5*f.late_btc_abs_median)
 if underresponse:state&=f.early_btc_rank.le(.4)
 return state,pe
def active_and_side(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 if control=="premium_persistence_without_underresponse":state,sidev=state_primary(f,underresponse=False)
 elif control=="late_btc_return_only":state=f.source_valid&f.late_btc_return.ne(0)&f.late_btc_rank.ge(.8);sidev=f.late_btc_return
 elif control=="contemporaneous_agreement":state=f.source_valid&f.late_premium_change.ne(0)&f.late_btc_return.ne(0)&f.late_premium_rank.ge(.8)&np.sign(f.late_btc_return).eq(np.sign(f.late_premium_change));sidev=f.late_premium_change
 elif control=="one_hour_stale_premium":state,sidev=state_primary(f,stale=True)
 else:state,sidev=state_primary(f)
 side=np.sign(sidev)
 if control=="direction_flip":side=-side
 return onset(state)&pd.Series(sidev,index=f.index).ne(0),pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides=active_and_side(f,control);rows=[];reserved=None
 for i in f.index[active&sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":f.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:f.at[i,c] for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,float|int]:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("PPLAR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"pplar_8_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":int(raw.source_rows.sum()),"aggregate_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"pplar_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
