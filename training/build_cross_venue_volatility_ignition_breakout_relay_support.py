"""Build source-support clocks for CVVIB-6 without post-entry outcomes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_led_intrahour_absorption_support as intrahour
from training import preregister_cross_venue_volatility_ignition_breakout_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

CLOCK=Path("data/cross_venue_volatility_ignition_breakout_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR=Path("data/cross_venue_volatility_ignition_breakout_relay_controls_2023_2026")
RESULT=Path("results/cross_venue_volatility_ignition_breakout_relay_support_2026-08-08.json")
SPLITS=base.SPLITS; MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8}
CONTROLS=("no_joint_expansion","no_late_ignition","no_compression","no_breakout","direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","block_start_time","decision_time","feature_available_time","entry_time","exit_time","side","bvol_block_return","dvol_block_return","bvol_fourth_body","dvol_fourth_body","first_three_hour_return","prior_abs_first_three_q50","fourth_hour_return","prior_abs_fourth_q60")

def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(payload:Any)->str:return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def features()->pd.DataFrame:
    volatility=intrahour.features().copy()
    price=pd.read_csv(intrahour.PRICE_DIR/"btc_intrahour_path.csv.gz",compression="gzip")
    price["decision_time"]=pd.to_datetime(price["decision_time"],utc=True,format="mixed")
    for column in ("hour_open","hour_close"):price[column]=pd.to_numeric(price[column],errors="coerce")
    price["block_price_valid"]=price["source_valid"].astype(str).str.lower().eq("true")
    volatility=volatility.drop(columns=["price_valid"],errors="ignore")
    joined=volatility.merge(price[["decision_time","hour_open","hour_close","block_price_valid"]],on="decision_time",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
    consecutive=joined["decision_time"].diff().eq(pd.Timedelta(hours=1))
    joined["four_hour_valid"]=(joined["base_valid"]&joined["block_price_valid"]).rolling(4,min_periods=4).sum().eq(4)&consecutive.rolling(3,min_periods=3).sum().eq(3)
    joined["bvol_block_return"]=joined["bvol_close"]/joined["bvol_open"].shift(3)-1.0
    joined["dvol_block_return"]=joined["dvol_close"]/joined["dvol_open"].shift(3)-1.0
    joined["first_three_hour_return"]=joined["hour_open"]/joined["hour_open"].shift(3)-1.0
    joined["fourth_hour_return"]=joined["hour_close"]/joined["hour_open"]-1.0
    blocks=joined[joined["decision_time"].dt.hour.mod(4).eq(0)].copy().reset_index(drop=True)
    blocks["prior_abs_first_three_q50"]=blocks["first_three_hour_return"].abs().where(blocks["four_hour_valid"]).shift(1).rolling(180,min_periods=168).quantile(.50)
    blocks["prior_abs_fourth_q60"]=blocks["fourth_hour_return"].abs().where(blocks["four_hour_valid"]).shift(1).rolling(180,min_periods=168).quantile(.60)
    values=["bvol_block_return","dvol_block_return","bvol_body","dvol_body","first_three_hour_return","fourth_hour_return"]
    blocks["block_valid"]=blocks["four_hour_valid"]&np.isfinite(blocks[values]).all(axis=1)
    return blocks

def build_clock(frame:pd.DataFrame,control:str="primary")->pd.DataFrame:
    expansion=pd.Series(True,index=frame.index) if control=="no_joint_expansion" else frame.bvol_block_return.gt(0)&frame.dvol_block_return.gt(0)
    ignition=pd.Series(True,index=frame.index) if control=="no_late_ignition" else frame.bvol_body.gt(0)&frame.dvol_body.gt(0)
    compression=frame.prior_abs_first_three_q50.notna()
    if control!="no_compression":compression&=frame.first_three_hour_return.abs().le(frame.prior_abs_first_three_q50)
    breakout=frame.fourth_hour_return.ne(0)&frame.prior_abs_fourth_q60.notna()
    if control!="no_breakout":breakout&=frame.fourth_hour_return.abs().ge(frame.prior_abs_fourth_q60)
    active=frame.block_valid&expansion&ignition&compression&breakout;rows=[];next_allowed=None
    for index in frame.index[active]:
        decision=pd.Timestamp(frame.at[index,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_time=entry+pd.Timedelta(hours=6)
        if next_allowed is not None and entry<next_allowed:continue
        split=next((name for name,(start,end) in SPLITS.items() if entry>=start and exit_time<=end),None)
        if split is None:continue
        side=int(np.sign(frame.at[index,"fourth_hour_return"]));side=-side if control=="direction_flip" else side;next_allowed=exit_time
        rows.append({"candidate":"CVVIB-6","control":control,"split":split,"block_start_time":decision-pd.Timedelta(hours=4),"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_time,"side":side,"bvol_block_return":float(frame.at[index,"bvol_block_return"]),"dvol_block_return":float(frame.at[index,"dvol_block_return"]),"bvol_fourth_body":float(frame.at[index,"bvol_body"]),"dvol_fourth_body":float(frame.at[index,"dvol_body"]),"first_three_hour_return":float(frame.at[index,"first_three_hour_return"]),"prior_abs_first_three_q50":float(frame.at[index,"prior_abs_first_three_q50"]),"fourth_hour_return":float(frame.at[index,"fourth_hour_return"]),"prior_abs_fourth_q60":float(frame.at[index,"prior_abs_fourth_q60"])})
    return pd.DataFrame(rows,columns=COLUMNS)

def split_stats(clock:pd.DataFrame,split:str)->dict[str,Any]:
    subset=clock[clock.split.eq(split)]
    if subset.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.0,"max_month_share":0.0}
    longs=int(subset.side.eq(1).sum());shorts=int(subset.side.eq(-1).sum());months=subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events":len(subset),"longs":longs,"shorts":shorts,"minority_side_share":min(longs,shorts)/len(subset),"max_month_share":int(months.max())/len(subset)}

def run()->dict[str,Any]:
    frame=features();primary=build_clock(frame);controls={name:build_clock(frame,name) for name in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
    for name,control in controls.items():_write_gzip_csv(control,CONTROL_DIR/f"{name}.csv.gz")
    support={name:split_stats(primary,name) for name in SPLITS};checks={}
    for name,stats in support.items():checks[f"{name}_minimum_events"]=stats["events"]>=MINIMUM_EVENTS[name];checks[f"{name}_side_balance"]=stats["minority_side_share"]>=.20;checks[f"{name}_month_concentration"]=stats["max_month_share"]<=.45
    registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());vol_manifest=intrahour.NONPRICE_DIR/"manifest.json";price_manifest=intrahour.PRICE_DIR/"manifest.json";passed=all(checks.values())
    core={"protocol_version":"cvvib_6_source_support_v1","policy_id":"CVVIB-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha256(prereg.DEFAULT_OUTPUT),"manifest_hash":registration["manifest_hash"]},"source_manifests":{"volatility":{"path":str(vol_manifest),"sha256":sha256(vol_manifest)},"completed_price":{"path":str(price_manifest),"sha256":sha256(price_manifest)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha256(CLOCK),"rows":len(primary)},"controls":{name:{"path":str(CONTROL_DIR/f"{name}.csv.gz"),"sha256":sha256(CONTROL_DIR/f"{name}.csv.gz"),"rows":len(control)} for name,control in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"}
    result={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return result

if __name__=="__main__":
    argparse.ArgumentParser().parse_args();report=run();print(json.dumps({"passed":report["support_passed"],"support":report["support"]},indent=2))
