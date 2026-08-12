"""Materialize source-only HVFSE-8 support clocks."""
from __future__ import annotations
import argparse, bisect, hashlib, json, math
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_funding_streak_exhaustion_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV=Path("/home/pakchu/rllm/.env");PREREG_SHA="f412a4a592c59721807f50309022436f146c5be627dc8f65160acd3f62ba830c"
START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
ROOT=Path("data/high_volatility_funding_streak_exhaustion_reversal_sources_2023_2026");PANEL=ROOT/"funding_streak_states.csv.gz";MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_funding_streak_exhaustion_reversal_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_funding_streak_exhaustion_reversal_controls_2023_2026");RESULT=Path("results/high_volatility_funding_streak_exhaustion_reversal_support_2026-08-13.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8}
CONTROLS=("no_magnitude_tail","no_variation_gate","single_settlement_level","one_settlement_stale_streak","direction_flip","forced_long")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","funding_rate","streak_sign","streak_length","cumulative_funding","cumulative_magnitude","magnitude_rank","btc_realized_variation","variation_rank")
FUNDING_QUERY="SELECT funding_time AS decision_time,funding_rate FROM funding_rates_binance WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"
BAR_QUERY="SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rank(v:pd.Series,maximum:int=270,minimum:int=180)->pd.Series:
 a=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(a),np.nan);q=deque();s=[]
 for i,x in enumerate(a):
  if math.isfinite(x) and len(q)>=minimum:l=bisect.bisect_left(s,x);r=bisect.bisect_right(s,x);o[i]=(l+.5*(r-l))/len(s)
  if math.isfinite(x):
   q.append(x);bisect.insort(s,x)
   if len(q)>maximum:old=q.popleft();s.pop(bisect.bisect_left(s,old))
 return pd.Series(o,index=v.index)
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV);return create_engine(postgres_url_from_env(ENV),connect_args={"connect_timeout":10})
def load_sources()->tuple[pd.DataFrame,pd.DataFrame]:
 from sqlalchemy import text
 db=engine()
 with db.connect() as c:
  f=pd.read_sql_query(text(FUNDING_QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
  b=pd.read_sql_query(text(BAR_QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 db.dispose();f["decision_time"]=pd.to_datetime(f.decision_time,utc=True);f["funding_rate"]=pd.to_numeric(f.funding_rate,errors="coerce");b["ts"]=pd.to_datetime(b.ts,utc=True)
 for c in ("open","close"):b[c]=pd.to_numeric(b[c],errors="coerce")
 if f.decision_time.duplicated().any() or b.ts.duplicated().any():raise RuntimeError("HVFSE duplicate source key")
 return f,b
def score(f:pd.DataFrame,b:pd.DataFrame)->pd.DataFrame:
 x=f.copy();sign=np.sign(x.funding_rate).astype(int);length=[];cumulative=[];run_sign=0;run_len=0;run_sum=0.
 for i in x.index:
  current=int(sign.at[i]);adjacent=i>0 and x.at[i,"decision_time"]-x.at[i-1,"decision_time"]==pd.Timedelta(hours=8)
  if current!=0 and adjacent and current==run_sign:run_len+=1;run_sum+=float(x.at[i,"funding_rate"])
  elif current!=0:run_sign=current;run_len=1;run_sum=float(x.at[i,"funding_rate"])
  else:run_sign=0;run_len=0;run_sum=0.
  length.append(run_len);cumulative.append(run_sum)
 x["streak_sign"]=sign;x["streak_length"]=length;x["cumulative_funding"]=cumulative;x["cumulative_magnitude"]=np.abs(x.cumulative_funding)
 source_streak=x.streak_length.ge(3)&x.streak_sign.ne(0)&np.isfinite(x.cumulative_magnitude)
 x["magnitude_rank"]=rank(x.cumulative_magnitude.where(source_streak))
 bars=b.sort_values("ts").set_index("ts");variations=[];valids=[]
 for d in x.decision_time:
  w=bars.loc[(bars.index>=d-pd.Timedelta(hours=24))&(bars.index<d)];valid=len(w)==1440 and w.index.min()==d-pd.Timedelta(hours=24) and w.index.max()==d-pd.Timedelta(minutes=1) and w.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all() and np.isfinite(w[["open","close"]]).all(axis=None) and w[["open","close"]].gt(0).all(axis=None)
  valids.append(valid);variations.append(float(np.sqrt(np.square(np.log(w.close.to_numpy(float)/w.open.to_numpy(float))).sum())) if valid else np.nan)
 x["btc_realized_variation"]=variations;x["variation_rank"]=rank(x.btc_realized_variation.where(source_streak&pd.Series(valids,index=x.index)));x["source_valid"]=source_streak&pd.Series(valids,index=x.index)&np.isfinite(x[["magnitude_rank","btc_realized_variation","variation_rank"]]).all(axis=1);return x
def conditions(x:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 used=x.shift(1) if control=="one_settlement_stale_streak" else x;mag=used.magnitude_rank;streak=used.streak_length.ge(3)&used.streak_sign.ne(0)
 if control=="single_settlement_level":streak=used.funding_rate.ne(0);mag=rank(x.funding_rate.abs()).reindex(x.index)
 active=used.source_valid.eq(True)&streak
 if control!="no_magnitude_tail":active&=mag.ge(.75)
 if control!="no_variation_gate":active&=x.variation_rank.ge(.65)
 eligible=active;adjacent=x.decision_time.diff().eq(pd.Timedelta(hours=8));onset=eligible&~eligible.shift(1,fill_value=False)&adjacent&x.source_valid.shift(1,fill_value=False);side=-np.sign(used.streak_sign).fillna(0).astype(int)
 if control=="single_settlement_level":side=-np.sign(used.funding_rate).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=pd.Series(1,index=x.index)
 return onset,side
def clock(x:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(x,control);rows=[];reserved=None
 for i in x.index[active]:
  d=pd.Timestamp(x.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":"HVFSE-8","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"funding_rate":float(x.at[i,"funding_rate"]),"streak_sign":int(x.at[i,"streak_sign"]),"streak_length":int(x.at[i,"streak_length"]),"cumulative_funding":float(x.at[i,"cumulative_funding"]),"cumulative_magnitude":float(x.at[i,"cumulative_magnitude"]),"magnitude_rank":float(x.at[i,"magnitude_rank"]),"btc_realized_variation":float(x.at[i,"btc_realized_variation"]),"variation_rank":float(x.at[i,"variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict:
 q=c[c.split.eq(s)]
 if q.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(q.side.eq(1).sum());sh=int(q.side.eq(-1).sum());m=pd.to_datetime(q.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(q),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(q),"max_month_share":int(m.max())/len(q)}
def run()->dict:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVFSE preregistration drift")
 f,b=load_sources();x=score(f,b);primary=clock(x);controls={n:clock(x,n) for n in CONTROLS};ROOT.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(x,PANEL);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"hvfse_8_source_v1","funding_query":FUNDING_QUERY,"bar_query":BAR_QUERY,"window":[START.isoformat(),END.isoformat()],"funding_rows":len(f),"bar_rows":len(b),"output":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(x)},"postentry_outcomes_opened":False,"gross9_rows_opened":False};manifest={**source_core,"manifest_hash":chash(source_core)};MANIFEST.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
 support={s:stats(primary,s) for s in SPLITS};checks={k:v for s,z in support.items() for k,v in ((f"{s}_minimum_events",z["events"]>=MINIMUM[s]),(f"{s}_side_balance",z["minority_side_share"]>=.2),(f"{s}_month_concentration",z["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvfse_8_source_support_v1","policy_id":"HVFSE-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
