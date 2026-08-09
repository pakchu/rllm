"""Build source-only HVORFR-6 support clocks."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from training import preregister_high_volatility_opening_range_failure_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market

PREREG_SHA="fc118094df11dc7f4d0d18139ab7bef9a8aa5b41fa0507f25d374916911cea2e";END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/high_volatility_opening_range_failure_relay_sources_2020_2026");FEATURES=SOURCE_DIR/"opening_range_failure_features.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json"
CLOCK=Path("data/high_volatility_opening_range_failure_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_opening_range_failure_relay_controls_2023_2026");RESULT=Path("results/high_volatility_opening_range_failure_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8}
CONTROLS=("no_variation_gate","inside_close_only","no_midpoint_confirmation","one_block_stale_geometry","direction_flip","same_clock_forced_long")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","break_side","opening_high","opening_low","opening_midpoint","middle_close","final_close","final_return","realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def strict_prior_midrank(v:pd.Series,lookback:int=252,minimum:int=180)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);out=np.full(len(x),np.nan);history=[]
 for i,current in enumerate(x):
  prior=history[-lookback:]
  if np.isfinite(current) and len(prior)>=minimum:
   a=np.asarray(prior);out[i]=(np.count_nonzero(a<current)+.5*np.count_nonzero(a==current))/len(a)
  if np.isfinite(current):history.append(float(current))
 return pd.Series(out,index=v.index)
def build_features(market:pd.DataFrame)->pd.DataFrame:
 bars=market.copy();bars["date"]=pd.to_datetime(bars.date,utc=True);bars=bars.sort_values("date").set_index("date");rows=[]
 first=bars.index.min().ceil("D")+pd.Timedelta(hours=6)
 decisions=[]
 for day in pd.date_range(first.floor("D"),END,freq="1D",inclusive="left"):
  decisions.extend([day+pd.Timedelta(hours=6),day+pd.Timedelta(hours=18)])
 for decision in decisions:
  expected=pd.date_range(decision-pd.Timedelta(hours=6),decision,freq="5min",inclusive="left");window=bars.reindex(expected)
  prices=window[["open","high","low","close"]].apply(pd.to_numeric,errors="coerce")
  valid=bool(len(window)==72 and np.isfinite(prices).all(axis=1).all() and prices.gt(0).all(axis=1).all() and prices.high.ge(prices[["open","close"]].max(axis=1)).all() and prices.low.le(prices[["open","close"]].min(axis=1)).all())
  if not valid:continue
  opening,middle,final=prices.iloc[:24],prices.iloc[24:48],prices.iloc[48:]
  oh,ol=float(opening.high.max()),float(opening.low.min());mid=(oh+ol)/2;mc=float(middle.close.iloc[-1]);fc=float(final.close.iloc[-1]);fr=float(np.log(fc/final.open.iloc[0]));up=bool(middle.high.max()>oh);down=bool(middle.low.min()<ol);exclusive=up!=down;break_side=1 if up and not down else -1 if down and not up else 0;inside=bool(ol<mc<oh);opposite=bool(fr!=0 and np.sign(fr)==-break_side);midpoint=bool(fc<mid if break_side==1 else fc>mid if break_side==-1 else False);variation=float(np.sqrt(np.square(np.log(prices.close/prices.open)).sum()))
  rows.append({"decision_time":decision,"source_valid":True,"opening_high":oh,"opening_low":ol,"opening_midpoint":mid,"middle_close":mc,"final_close":fc,"final_return":fr,"exclusive_break":exclusive,"break_side":break_side,"inside_close":inside,"opposite_final_return":opposite,"midpoint_confirmation":midpoint,"realized_variation":variation})
 frame=pd.DataFrame(rows);frame["variation_rank"]=strict_prior_midrank(frame.realized_variation.where(frame.source_valid));frame["primary_state"]=frame.source_valid&frame.exclusive_break&frame.inside_close&frame.opposite_final_return&frame.midpoint_confirmation;return frame
def conditions(f:pd.DataFrame,control:str="primary")->tuple[pd.Series,pd.Series]:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=f.shift(1) if control=="one_block_stale_geometry" else f
 if control=="inside_close_only":state=used.source_valid.fillna(False)&used.exclusive_break.fillna(False)&used.inside_close.fillna(False)
 elif control=="no_midpoint_confirmation":state=used.source_valid.fillna(False)&used.exclusive_break.fillna(False)&used.inside_close.fillna(False)&used.opposite_final_return.fillna(False)
 else:state=used.primary_state.fillna(False)
 active=state if control=="no_variation_gate" else state&f.variation_rank.ge(.65);side=-used.break_side.fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=f.index)
 return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"HVORFR-6","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),"break_side":int(f.at[i,"break_side"]),"opening_high":float(f.at[i,"opening_high"]),"opening_low":float(f.at[i,"opening_low"]),"opening_midpoint":float(f.at[i,"opening_midpoint"]),"middle_close":float(f.at[i,"middle_close"]),"final_close":float(f.at[i,"final_close"]),"final_return":float(f.at[i,"final_return"]),"realized_variation":float(f.at[i,"realized_variation"]),"variation_rank":float(f.at[i,"variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA or sha(prereg.MARKET)!=prereg.MARKET_SHA:raise RuntimeError("HVORFR predecessor drift")
 market,source=load_market();f=build_features(market);primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());sc={"protocol_version":"hvorfr_6_source_v1","policy_id":"HVORFR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"market":source,"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(f)},"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"hvorfr_6_source_support_v1","policy_id":"HVORFR-6","preregistration":sc["preregistration"],"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
