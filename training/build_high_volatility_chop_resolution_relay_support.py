"""Build source-support clocks for HVCRR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_led_intrahour_absorption_support as intrahour
from training import preregister_high_volatility_chop_resolution_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
CLOCK=Path("data/high_volatility_chop_resolution_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_chop_resolution_relay_controls_2023_2026");RESULT=Path("results/high_volatility_chop_resolution_relay_support_2026-08-08.json");SPLITS=base.SPLITS;MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_high_volatility","no_low_efficiency","no_resolution_tail","no_range_escape","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","block_start_time","decision_time","feature_available_time","entry_time","exit_time","side","bvol_close","prior_bvol_q60","dvol_close","prior_dvol_q60","chop_range","chop_return","chop_efficiency","prior_chop_efficiency_q40","resolution_return","prior_abs_resolution_q60","chop_min","chop_max","final_close")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 v=intrahour.features().drop(columns=["price_valid"],errors="ignore");p=pd.read_csv(intrahour.PRICE_DIR/"btc_intrahour_path.csv.gz",compression="gzip");p.decision_time=pd.to_datetime(p.decision_time,utc=True,format="mixed")
 for c in ("hour_open","hour_close"):p[c]=pd.to_numeric(p[c],errors="coerce")
 p["path_valid"]=p.source_valid.astype(str).str.lower().eq("true");j=v.merge(p[["decision_time","hour_open","hour_close","path_valid"]],on="decision_time",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
 j["prior_bvol_q60"]=j.bvol_close.where(j.bvol_valid&j.bvol_close.gt(0)).shift(1).rolling(720,min_periods=672).quantile(.60);j["prior_dvol_q60"]=j.dvol_close.where(j.bvol_valid&j.dvol_close.gt(0)).shift(1).rolling(720,min_periods=672).quantile(.60)
 consecutive=j.decision_time.diff().eq(pd.Timedelta(hours=1));j["eight_valid"]=(j.base_valid&j.path_valid).rolling(8,min_periods=8).sum().eq(8)&consecutive.rolling(7,min_periods=7).sum().eq(7)
 levels=pd.concat([j.hour_open.shift(7).rename("open")]+[j.hour_close.shift(k).rename(str(k)) for k in range(2,8)],axis=1);initial=j.hour_open.shift(7);sixth=j.hour_close.shift(2);j["chop_min"]=levels.min(axis=1);j["chop_max"]=levels.max(axis=1);j["chop_range"]=(j.chop_max-j.chop_min)/initial;j["chop_return"]=sixth/initial-1;j["chop_efficiency"]=j.chop_return.abs()/j.chop_range.where(j.chop_range.gt(0));j["resolution_return"]=j.hour_close/j.hour_open.shift(1)-1;j["final_close"]=j.hour_close
 b=j[j.decision_time.dt.hour.mod(8).eq(0)].copy().reset_index(drop=True);b["prior_chop_efficiency_q40"]=b.chop_efficiency.where(b.eight_valid).shift(1).rolling(270,min_periods=252).quantile(.40);b["prior_abs_resolution_q60"]=b.resolution_return.abs().where(b.eight_valid).shift(1).rolling(270,min_periods=252).quantile(.60);vals=["bvol_close","dvol_close","chop_range","chop_efficiency","resolution_return","chop_min","chop_max","final_close"];b["block_valid"]=b.eight_valid&np.isfinite(b[vals]).all(axis=1)&b.chop_range.gt(0);return b
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 vol=pd.Series(True,index=f.index) if control=="no_high_volatility" else f.bvol_close.ge(f.prior_bvol_q60)&f.dvol_close.ge(f.prior_dvol_q60)
 chop=f.prior_chop_efficiency_q40.notna();chop&=True if control=="no_low_efficiency" else f.chop_efficiency.le(f.prior_chop_efficiency_q40)
 resolution=f.resolution_return.ne(0)&f.prior_abs_resolution_q60.notna();resolution&=True if control=="no_resolution_tail" else f.resolution_return.abs().ge(f.prior_abs_resolution_q60)
 above=f.final_close.gt(f.chop_max);below=f.final_close.lt(f.chop_min);escape=above|below
 if control=="no_range_escape":escape=f.resolution_return.ne(0);side=np.sign(f.resolution_return)
 else:side=pd.Series(np.where(above,1,np.where(below,-1,0)),index=f.index)
 active=f.block_valid&vol&chop&resolution&escape;rows=[]
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6);split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  s=int(side.at[i]);s=-s if control=="direction_flip" else s;rows.append({"candidate":"HVCRR-6","control":control,"split":split,"block_start_time":decision-pd.Timedelta(hours=8),"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":s,"bvol_close":float(f.at[i,"bvol_close"]),"prior_bvol_q60":float(f.at[i,"prior_bvol_q60"]),"dvol_close":float(f.at[i,"dvol_close"]),"prior_dvol_q60":float(f.at[i,"prior_dvol_q60"]),"chop_range":float(f.at[i,"chop_range"]),"chop_return":float(f.at[i,"chop_return"]),"chop_efficiency":float(f.at[i,"chop_efficiency"]),"prior_chop_efficiency_q40":float(f.at[i,"prior_chop_efficiency_q40"]),"resolution_return":float(f.at[i,"resolution_return"]),"prior_abs_resolution_q60":float(f.at[i,"prior_abs_resolution_q60"]),"chop_min":float(f.at[i,"chop_min"]),"chop_max":float(f.at[i,"chop_max"]),"final_close":float(f.at[i,"final_close"])})
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
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());vm=intrahour.NONPRICE_DIR/"manifest.json";pm=intrahour.PRICE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"hvcrr_6_source_support_v1","policy_id":"HVCRR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"volatility":{"path":str(vm),"sha256":sha(vm)},"completed_price":{"path":str(pm),"sha256":sha(pm)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
