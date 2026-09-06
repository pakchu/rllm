"""Build source-support clocks for JVCTR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_trend_pullback_relay_support as source
from training import preregister_joint_volatility_cooling_trend_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
CLOCK=Path("data/joint_volatility_cooling_trend_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/joint_volatility_cooling_trend_relay_controls_2023_2026");RESULT=Path("results/joint_volatility_cooling_trend_relay_support_2026-08-08.json");SPLITS=source.SPLITS;MIN=source.MIN;CONTROLS=("no_joint_cooling","no_trend_tail","no_same_direction","no_deceleration_cap","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","trend_start_time","continuation_start_time","decision_time","feature_available_time","entry_time","exit_time","side","bvol_body","dvol_body","trend_return_3h","prior_abs_trend_q60","continuation_return_1h","deceleration_ratio")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 cooling=pd.Series(True,index=f.index) if control=="no_joint_cooling" else f.bvol_body.lt(0)&f.dvol_body.lt(0);trend=f.trend_return_3h.ne(0)&f.prior_abs_trend_q60.notna();trend&=True if control=="no_trend_tail" else f.trend_return_3h.abs().ge(f.prior_abs_trend_q60);current=f.pullback_return_1h.ne(0);current&=True if control=="no_same_direction" else np.sign(f.pullback_return_1h).eq(np.sign(f.trend_return_3h));ratio=f.pullback_return_1h.abs()/f.trend_return_3h.abs();decelerate=pd.Series(True,index=f.index) if control=="no_deceleration_cap" else ratio.le(.5);active=f.four_valid&cooling&trend&current&decelerate;on=active&~active.shift(1,fill_value=False);rows=[];next_allowed=None
 for i in f.index[on]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=int(np.sign(f.at[i,"trend_return_3h"]));side=-side if control=="direction_flip" else side;next_allowed=exit_;rows.append({"candidate":"JVCTR-6","control":control,"split":split,"trend_start_time":decision-pd.Timedelta(hours=4),"continuation_start_time":decision-pd.Timedelta(hours=1),"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":side,"bvol_body":float(f.at[i,"bvol_body"]),"dvol_body":float(f.at[i,"dvol_body"]),"trend_return_3h":float(f.at[i,"trend_return_3h"]),"prior_abs_trend_q60":float(f.at[i,"prior_abs_trend_q60"]),"continuation_return_1h":float(f.at[i,"pullback_return_1h"]),"deceleration_ratio":float(ratio.at[i])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 f=source.features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MIN[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());vm=source.intrahour.NONPRICE_DIR/"manifest.json";pm=source.intrahour.PRICE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"jvctr_6_source_support_v1","policy_id":"JVCTR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"volatility":{"path":str(vm),"sha256":sha(vm)},"completed_price":{"path":str(pm),"sha256":sha(pm)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
