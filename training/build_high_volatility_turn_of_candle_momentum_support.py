"""Build source-only HVTOCM-30M clocks before Gross9 or economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_turn_of_candle_momentum as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market
PREREG_SHA="1ef020ed2d3220400e8a4488e8670dd932dc75f8dd20867ce18abff4fcddd265";HELPER=Path("training/build_scheduled_trend_concordance_relay_support.py");HELPER_SHA="8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f";END=pd.Timestamp("2026-08-01T00:00:00Z")
STATE=Path("data/high_volatility_turn_of_candle_momentum_sources_2023_2026/daily_states.csv.gz");CLOCK=Path("data/high_volatility_turn_of_candle_momentum_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_turn_of_candle_momentum_controls_2023_2026");RESULT=Path("results/high_volatility_turn_of_candle_momentum_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_variation_gate","opening_half_hour_fade","second_half_hour_momentum","one_day_stale_opening_return","same_clock_forced_long");COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","opening_return","second_half_hour_return","pre_day_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _valid(w:pd.DataFrame)->bool:
 v=w[["open","high","low","close"]];return bool(np.isfinite(v).all(axis=1).all() and v.gt(0).all(axis=1).all() and w.high.ge(w[["open","close"]].max(axis=1)).all() and w.low.le(w[["open","close"]].min(axis=1)).all() and w.high.ge(w.low).all())
def score_states(market:pd.DataFrame)->pd.DataFrame:
 f=market.copy();f["date"]=pd.to_datetime(f.date,utc=True);f=f.sort_values("date").set_index("date");days=pd.date_range(f.index.min().ceil("D"),f.index.max().floor("D"),freq="D");rows=[];prior=[]
 for d in days:
  vi=pd.date_range(d-pd.Timedelta(hours=24,minutes=5),d-pd.Timedelta(minutes=5),freq="5min");oi=pd.date_range(d,d+pd.Timedelta(hours=1),freq="5min",inclusive="left");vw,opening=f.reindex(vi),f.reindex(oi)
  if len(vw)!=289 or len(opening)!=12 or not _valid(vw) or not _valid(opening):continue
  variation=float(np.sqrt(np.square(np.diff(np.log(vw.close.to_numpy(float)))).sum()));r1=float(np.log(float(opening.close.iloc[5])/float(opening.open.iloc[0])));r2=float(np.log(float(opening.close.iloc[11])/float(opening.open.iloc[6])));h=np.asarray(prior[-90:],float);rank=float(((h<variation).sum()+.5*(h==variation).sum())/len(h)) if len(h)>=60 else np.nan
  rows.append({"day":d,"decision_time":d+pd.Timedelta(minutes=30),"opening_return":r1,"second_half_hour_return":r2,"pre_day_variation":variation,"variation_rank":rank});prior.append(variation)
 return pd.DataFrame(rows)
def build_clock(s:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 rows=[]
 for i,r in s.iterrows():
  signal=float(r.opening_return);decision=pd.Timestamp(r.decision_time)
  if control=="second_half_hour_momentum":signal=float(r.second_half_hour_return);decision=pd.Timestamp(r.day)+pd.Timedelta(hours=1)
  elif control=="one_day_stale_opening_return":
   if i==0:continue
   signal=float(s.iloc[i-1].opening_return)
  ranked=bool(np.isfinite(r.variation_rank) and r.variation_rank>=.65)
  if not np.isfinite(signal) or signal==0 or (not ranked and control!="no_variation_gate"):continue
  side=int(np.sign(signal));side=-side if control=="opening_half_hour_fade" else side;side=1 if control=="same_clock_forced_long" else side
  entry=pd.Timestamp(r.day)+pd.Timedelta(hours=23,minutes=30);exit_=pd.Timestamp(r.day)+pd.Timedelta(days=1);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":side,"opening_return":signal,"second_half_hour_return":float(r.second_half_hour_return),"pre_day_variation":float(r.pre_day_variation),"variation_rank":float(r.variation_rank)})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 bindings={prereg.DEFAULT_OUTPUT:PREREG_SHA,HELPER:HELPER_SHA,prereg.MARKET:prereg.MARKET_SHA}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError(f"HVTOCM binding drift: {p}")
 market,source=load_market();states=score_states(market);primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS};STATE.parent.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATE);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvtocm_30m_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"bindings":{str(p):h for p,h in bindings.items()},"source":source,"information_embargo_audit":{"eligibility_inputs_after_00_30_opened":False,"scheduled_entry_time":"23:30 UTC","postentry_return_pnl_opened":False},"source_state":{"path":str(STATE),"sha256":sha(STATE),"rows":len(states)},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
