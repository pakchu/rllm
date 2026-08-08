"""Build source-support clocks for HVTPR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_led_intrahour_absorption_support as intrahour
from training import preregister_high_volatility_trend_pullback_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
CLOCK=Path("data/high_volatility_trend_pullback_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_trend_pullback_relay_controls_2023_2026");RESULT=Path("results/high_volatility_trend_pullback_relay_support_2026-08-08.json");SPLITS=base.SPLITS;MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_high_volatility","no_trend_tail","no_opposite_pullback","no_shallow_cap","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","trend_start_time","pullback_start_time","decision_time","feature_available_time","entry_time","exit_time","side","bvol_close","prior_bvol_q60","dvol_close","prior_dvol_q60","trend_return_3h","prior_abs_trend_q60","pullback_return_1h","pullback_ratio")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 v=intrahour.features().drop(columns=["price_valid"],errors="ignore");p=pd.read_csv(intrahour.PRICE_DIR/"btc_intrahour_path.csv.gz",compression="gzip");p.decision_time=pd.to_datetime(p.decision_time,utc=True,format="mixed");p.hour_open=pd.to_numeric(p.hour_open,errors="coerce");p.hour_close=pd.to_numeric(p.hour_close,errors="coerce");p["path_valid"]=p.source_valid.astype(str).str.lower().eq("true");j=v.merge(p[["decision_time","hour_open","hour_close","path_valid"]],on="decision_time",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
 vol_valid=j.bvol_valid&np.isfinite(j[["bvol_close","dvol_close"]]).all(axis=1)&j[["bvol_close","dvol_close"]].gt(0).all(axis=1);price_valid=j.path_valid&np.isfinite(j[["hour_open","hour_close"]]).all(axis=1)&j[["hour_open","hour_close"]].gt(0).all(axis=1)
 for name in ("bvol","dvol"):j[f"prior_{name}_q60"]=j[f"{name}_close"].where(vol_valid).shift(1).rolling(720,min_periods=672).quantile(.60)
 j["trend_return_3h"]=j.hour_open/j.hour_open.shift(3)-1;j["pullback_return_1h"]=j.hour_close/j.hour_open-1;j["trend_endpoint_valid"]=price_valid&price_valid.shift(3,fill_value=False)&j.decision_time.diff(3).eq(pd.Timedelta(hours=3));j["prior_abs_trend_q60"]=j.trend_return_3h.abs().where(j.trend_endpoint_valid).shift(1).rolling(720,min_periods=672).quantile(.60);j["four_valid"]=(vol_valid&price_valid).rolling(4,min_periods=4).sum().eq(4)&j.decision_time.diff().eq(pd.Timedelta(hours=1)).rolling(3,min_periods=3).sum().eq(3);return j
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 vol=pd.Series(True,index=f.index) if control=="no_high_volatility" else f.bvol_close.ge(f.prior_bvol_q60)&f.dvol_close.ge(f.prior_dvol_q60);trend=f.trend_return_3h.ne(0)&f.prior_abs_trend_q60.notna();trend&=True if control=="no_trend_tail" else f.trend_return_3h.abs().ge(f.prior_abs_trend_q60);pull=f.pullback_return_1h.ne(0);pull&=True if control=="no_opposite_pullback" else np.sign(f.pullback_return_1h).eq(-np.sign(f.trend_return_3h));ratio=f.pullback_return_1h.abs()/f.trend_return_3h.abs();shallow=pd.Series(True,index=f.index) if control=="no_shallow_cap" else ratio.le(.5);active=f.four_valid&vol&trend&pull&shallow;on=active&~active.shift(1,fill_value=False);rows=[];next_allowed=None
 for i in f.index[on]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=int(np.sign(f.at[i,"trend_return_3h"]));side=-side if control=="direction_flip" else side;next_allowed=exit_;rows.append({"candidate":"HVTPR-6","control":control,"split":split,"trend_start_time":decision-pd.Timedelta(hours=4),"pullback_start_time":decision-pd.Timedelta(hours=1),"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":side,"bvol_close":float(f.at[i,"bvol_close"]),"prior_bvol_q60":float(f.at[i,"prior_bvol_q60"]),"dvol_close":float(f.at[i,"dvol_close"]),"prior_dvol_q60":float(f.at[i,"prior_dvol_q60"]),"trend_return_3h":float(f.at[i,"trend_return_3h"]),"prior_abs_trend_q60":float(f.at[i,"prior_abs_trend_q60"]),"pullback_return_1h":float(f.at[i,"pullback_return_1h"]),"pullback_ratio":float(ratio.at[i])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MIN[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());vm=intrahour.NONPRICE_DIR/"manifest.json";pm=intrahour.PRICE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"hvtpr_6_source_support_v1","policy_id":"HVTPR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"volatility":{"path":str(vm),"sha256":sha(vm)},"completed_price":{"path":str(pm),"sha256":sha(pm)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
