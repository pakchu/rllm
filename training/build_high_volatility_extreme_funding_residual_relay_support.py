"""Materialize source-only HVEFR-8 clocks from actual funding settlements."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_extreme_funding_residual_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market

ENV=Path("/home/pakchu/rllm/.env");PREREG_SHA="d0ade3697a477e2b261dbc51ef35d098b132034527da08e01b44799bf9f5ea5c"
HELPER=Path("training/build_scheduled_trend_concordance_relay_support.py");HELPER_SHA="8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f"
START=pd.Timestamp("2020-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/high_volatility_extreme_funding_residual_relay_sources_2020_2026");PANEL=SOURCE_DIR/"funding_residual_states.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json"
CLOCK=Path("data/high_volatility_extreme_funding_residual_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_extreme_funding_residual_relay_controls_2023_2026");RESULT=Path("results/high_volatility_extreme_funding_residual_relay_support_2026-08-10.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8}
CONTROLS=("no_variation_gate","no_residual_tail","raw_funding_sign_reversal","one_settlement_stale_residual","direction_flip","same_clock_forced_long")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","funding_rate","funding_median","funding_residual","residual_rank","btc_variation","variation_rank")
QUERY="SELECT funding_time AS decision_time,funding_rate FROM funding_rates_binance WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v:pd.Series,maximum:int=270,minimum:int=180)->pd.Series:
 v=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in v.items():
  prior=np.asarray(h[-maximum:],dtype=float)
  if np.isfinite(x) and len(prior)>=minimum:o.at[i]=((prior<x).sum()+.5*(prior==x).sum())/len(prior)
  if np.isfinite(x):h.append(float(x))
 return o
def prior_median(v:pd.Series,maximum:int=90,minimum:int=60)->pd.Series:
 v=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in v.items():
  prior=h[-maximum:]
  if np.isfinite(x) and len(prior)>=minimum:o.at[i]=float(np.median(prior))
  if np.isfinite(x):h.append(float(x))
 return o
def funding_source()->pd.DataFrame:
 from sqlalchemy import text
 from preprocessing.live_db_features import sqlalchemy_engine_from_env
 db=sqlalchemy_engine_from_env(ENV)
 try:
  with db.connect() as c:f=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
 f["decision_time"]=pd.to_datetime(f.decision_time,utc=True);f["funding_rate"]=pd.to_numeric(f.funding_rate,errors="coerce")
 if f.decision_time.duplicated().any() or not f.decision_time.is_monotonic_increasing or not np.isfinite(f.funding_rate).all():raise RuntimeError("HVEFR funding source drift")
 return f
def score(f:pd.DataFrame,market:pd.DataFrame)->pd.DataFrame:
 x=f.copy();x["funding_median"]=prior_median(x.funding_rate);x["funding_residual"]=x.funding_rate-x.funding_median;x["residual_rank"]=rank(x.funding_residual.abs())
 m=market.sort_values("date").set_index("date");close=pd.to_numeric(m.close,errors="coerce");valid=np.isfinite(close)&close.gt(0);cont=m.index.to_series().diff().eq(pd.Timedelta(minutes=5));sq=np.log(close/close.shift()).pow(2);var=np.sqrt(sq.rolling(288,min_periods=288).sum());complete=valid.rolling(289,min_periods=289).sum().eq(289)&cont.rolling(288,min_periods=288).sum().eq(288)
 x["btc_variation"]=var.where(complete).reindex(x.decision_time-pd.Timedelta(minutes=5)).to_numpy();x["variation_rank"]=rank(x.btc_variation);x["source_valid"]=np.isfinite(x[["funding_rate","funding_median","funding_residual","residual_rank","btc_variation","variation_rank"]]).all(axis=1)&x.funding_residual.ne(0);return x
def clock(x:pd.DataFrame,control:str="primary")->pd.DataFrame:
 used=x.shift(1) if control=="one_settlement_stale_residual" else x;res=used.funding_residual;rr=used.residual_rank;active=used.source_valid.eq(True)&np.isfinite(res)&res.ne(0)&np.isfinite(rr)&np.isfinite(x.variation_rank)
 if control!="no_residual_tail":active&=rr.ge(.75)
 if control!="no_variation_gate":active&=x.variation_rank.ge(.65)
 side=-np.sign(used.funding_rate if control=="raw_funding_sign_reversal" else res).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=x.index)
 rows=[];reserved=None
 for i in x.index[active]:
  d=pd.Timestamp(x.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"funding_rate":float(used.at[i,"funding_rate"]),"funding_median":float(used.at[i,"funding_median"]),"funding_residual":float(res.at[i]),"residual_rank":float(rr.at[i]),"btc_variation":float(x.at[i,"btc_variation"]),"variation_rank":float(x.at[i,"variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,Any]:
 q=c[c.split.eq(s)]
 if q.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(q.side.eq(1).sum());h=int(q.side.eq(-1).sum());months=pd.to_datetime(q.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(q),"longs":l,"shorts":h,"minority_side_share":min(l,h)/len(q),"max_month_share":int(months.max())/len(q)}
def run()->dict[str,Any]:
 for p,e in ((prereg.DEFAULT_OUTPUT,PREREG_SHA),(prereg.MARKET,prereg.MARKET_SHA),(HELPER,HELPER_SHA)):
  if sha(p)!=e:raise RuntimeError(f"HVEFR binding drift {p}")
 market,market_source=load_market();f=funding_source();x=score(f,market);primary=clock(x);controls={n:clock(x,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(x,PANEL);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"hvefr_8_source_v1","query":QUERY,"table":"funding_rates_binance","symbol":"BTCUSDT","rows":len(f),"output":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(x)},"candidate_incidence_opened":False,"postentry_outcomes_opened":False};manifest={**source_core,"manifest_hash":prereg.canonical_hash(source_core)};MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n")
 support={s:stats(primary,s) for s in SPLITS};checks={k:v for s,z in support.items() for k,v in ((f"{s}_minimum_events",z["events"]>=MINIMUM[s]),(f"{s}_side_balance",z["minority_side_share"]>=.2),(f"{s}_month_concentration",z["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvefr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"market_source":market_source,"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":
 argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
