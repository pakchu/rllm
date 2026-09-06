"""Build causal source-only HVCRMR-12 clocks before Gross9 or economic metrics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_causal_response_memory_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_high_volatility_dominant_volume_bar_relay_support import load_combined_market,sha256,END

PREREG_SHA="0313017b1c3aebd690a13fb2c490159782537ede1f1971b01a9b6e611912351b"
HELPER=Path("training/build_high_volatility_dominant_volume_bar_relay_support.py");HELPER_SHA="773be1497be1727d1f2d8916c4288f8a417a62511786cd98d00cf1b836c0e55a"
SNAPSHOT=Path("data/high_volatility_causal_response_memory_relay_sources_2023_2026/causal_states.csv.gz");CLOCK=Path("data/high_volatility_causal_response_memory_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_causal_response_memory_relay_controls_2023_2026");RESULT=Path("results/high_volatility_causal_response_memory_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("fixed_momentum","fixed_reversal","sixteen_response_memory","one_opportunity_stale_memory","direction_flip")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","shock_return","range_vol","memory_count","memory_mean_response")
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def _valid(w:pd.DataFrame)->bool:
 p=w[["open","high","low","close"]];return bool((np.isfinite(p).all(axis=1)&p.gt(0).all(axis=1)&w.high.ge(w[["open","close"]].max(axis=1))&w.low.le(w[["open","close"]].min(axis=1))&w.high.ge(w.low)).all())
def score_snapshot(market:pd.DataFrame)->tuple[pd.DataFrame,dict[str,float]]:
 f=market.copy();f["date"]=pd.to_datetime(f.date,utc=True)
 for c in ("open","high","low","close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 dates=pd.DatetimeIndex(f.date);pos=np.flatnonzero((dates.minute.to_numpy()==0)&np.isin(dates.hour.to_numpy(),[0,12]));pos=pos[(pos>=144)&(pos+145<len(f))];rows=[]
 for i in pos:
  w=f.iloc[i-144:i];ok=len(w)==144 and dates[i]-dates[i-144]==pd.Timedelta(hours=12) and _valid(w);entry=i+1;exit_=i+145
  if not ok or dates[exit_]-dates[entry]!=pd.Timedelta(hours=12):continue
  shock=float(np.log(float(w.close.iloc[-1])/float(w.open.iloc[0])));rv=float(float(w.high.max())/float(w.low.min())-1);response=float(np.sign(shock)*np.log(float(f.open.iloc[exit_])/float(f.open.iloc[entry]))) if shock!=0 else np.nan
  rows.append({"decision_time":dates[i],"entry_time":dates[entry],"response_available_time":dates[exit_],"shock_return":shock,"range_vol":rv,"signed_response":response})
 s=pd.DataFrame(rows);cal=s[s.decision_time.ge(pd.Timestamp("2023-01-01T00:00:00Z"))&s.decision_time.lt(pd.Timestamp("2023-07-01T00:00:00Z"))].replace([np.inf,-np.inf],np.nan).dropna()
 if len(cal)<350:raise RuntimeError("HVCRMR calibration floor failed")
 th={"absolute_shock_return_q60":float(cal.shock_return.abs().quantile(.60)),"range_vol_q65":float(cal.range_vol.quantile(.65))};s["eligible_opportunity"]=s.shock_return.abs().ge(th["absolute_shock_return_q60"])&s.range_vol.ge(th["range_vol_q65"]);return s,th
def causal_state(scores:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 out=scores.copy();out["memory_count"]=0;out["memory_mean_response"]=np.nan;out["active"]=False;out["side"]=0
 eligible=out.index[out.eligible_opportunity].tolist()
 for i in eligible:
  decision=pd.Timestamp(out.at[i,"decision_time"]);prior=[j for j in eligible if j<i and pd.Timestamp(out.at[j,"response_available_time"])<=decision and np.isfinite(out.at[j,"signed_response"])]
  if control=="fixed_momentum":mean=1.;used=[]
  elif control=="fixed_reversal":mean=-1.;used=[]
  else:
   if control=="one_opportunity_stale_memory" and prior:prior=prior[:-1]
   width=16 if control=="sixteen_response_memory" else 32;used=prior[-width:]
   if len(used)<16:continue
   mean=float(out.loc[used,"signed_response"].mean())
  if not np.isfinite(mean) or mean==0:continue
  side=int(np.sign(out.at[i,"shock_return"])*np.sign(mean));side=-side if control=="direction_flip" else side
  out.at[i,"memory_count"]=len(used);out.at[i,"memory_mean_response"]=mean;out.at[i,"active"]=True;out.at[i,"side"]=side
 return out
def clock(scores:pd.DataFrame,control:str="primary")->pd.DataFrame:
 state=causal_state(scores,control);rows=[]
 for _,r in state[state.active&state.decision_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))].iterrows():
  entry=pd.Timestamp(r.entry_time);exit_=entry+pd.Timedelta(hours=12);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  rows.append({"candidate":"HVCRMR-12","control":control,"split":split,"decision_time":r.decision_time,"feature_available_time":r.decision_time,"entry_time":entry,"exit_time":exit_,"side":int(r.side),"shock_return":float(r.shock_return),"range_vol":float(r.range_vol),"memory_count":int(r.memory_count),"memory_mean_response":float(r.memory_mean_response)})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA or sha256(HELPER)!=HELPER_SHA:raise RuntimeError("HVCRMR binding drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());market,source=load_combined_market();scores,thresholds=score_snapshot(market);primary=clock(scores);controls={n:clock(scores,n) for n in CONTROLS};SNAPSHOT.parent.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(causal_state(scores),SNAPSHOT);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());core={"protocol_version":"hvcrmr_12_source_support_v1","policy_id":"HVCRMR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source":source,"helper_binding":{"path":str(HELPER),"sha256":HELPER_SHA},"feature_contract":{"calibration_window":["2023-01-01T00:00:00Z","2023-07-01T00:00:00Z"],**thresholds,"causal_prior_response_values_consumed":True,"stage_economic_metrics_opened":False},"source_snapshot":{"path":str(SNAPSHOT),"sha256":sha256(SNAPSHOT),"rows":len(scores)},"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha256(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha256(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_metrics":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
