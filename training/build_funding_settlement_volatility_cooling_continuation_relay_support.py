"""Build source-support clocks for FSVCCR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_funding_settlement_volatility_unwind_relay_support as source
from training import preregister_funding_settlement_volatility_cooling_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
CLOCK=Path("data/funding_settlement_volatility_cooling_continuation_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/funding_settlement_volatility_cooling_continuation_relay_controls_2023_2026");RESULT=Path("results/funding_settlement_volatility_cooling_continuation_relay_support_2026-08-08.json");SPLITS=source.SPLITS;MIN=source.MINIMUM_EVENTS;CONTROLS=("no_funding_extreme","no_trend_tail","no_joint_cooling","no_same_direction","no_deceleration_cap","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","settlement_time","decision_time","feature_available_time","entry_time","exit_time","side","funding_rate","prior_abs_funding_q60","pre_settlement_return_8h","prior_abs_pre_return_q60","post_settlement_return_1h","continuation_ratio","bvol_body","dvol_body")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 funding=f.funding_rate.ne(0)&f.prior_abs_funding_q60.notna();funding&=True if control=="no_funding_extreme" else f.funding_rate.abs().ge(f.prior_abs_funding_q60);trend=f.pre_settlement_return_8h.ne(0)&f.prior_abs_pre_return_q60.notna();trend&=True if control=="no_trend_tail" else f.pre_settlement_return_8h.abs().ge(f.prior_abs_pre_return_q60);post=f.post_settlement_return_1h.ne(0);post&=True if control=="no_same_direction" else np.sign(f.post_settlement_return_1h).eq(np.sign(f.pre_settlement_return_8h));ratio=f.post_settlement_return_1h.abs()/f.pre_settlement_return_8h.abs();decelerate=pd.Series(True,index=f.index) if control=="no_deceleration_cap" else ratio.le(.5);cool=pd.Series(True,index=f.index) if control=="no_joint_cooling" else f.bvol_body.lt(0)&f.dvol_body.lt(0);values=["funding_rate","pre_settlement_return_8h","post_settlement_return_1h","bvol_body","dvol_body"];valid=np.isfinite(f[values]).all(axis=1)&f.price_valid&f.pre_valid_8h;active=valid&funding&trend&post&decelerate&cool;rows=[];next_allowed=None
 for i in f.index[active]:
  settlement=pd.Timestamp(f.at[i,"settlement_time"]);decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=int(np.sign(f.at[i,"pre_settlement_return_8h"]));side=-side if control=="direction_flip" else side;next_allowed=exit_;rows.append({"candidate":"FSVCCR-6","control":control,"split":split,"settlement_time":settlement,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":side,"funding_rate":float(f.at[i,"funding_rate"]),"prior_abs_funding_q60":float(f.at[i,"prior_abs_funding_q60"]),"pre_settlement_return_8h":float(f.at[i,"pre_settlement_return_8h"]),"prior_abs_pre_return_q60":float(f.at[i,"prior_abs_pre_return_q60"]),"post_settlement_return_1h":float(f.at[i,"post_settlement_return_1h"]),"continuation_ratio":float(ratio.at[i]),"bvol_body":float(f.at[i,"bvol_body"]),"dvol_body":float(f.at[i,"dvol_body"])})
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
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());vm=source.intrahour.NONPRICE_DIR/"manifest.json";pm=source.intrahour.PRICE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"fsvccr_6_source_support_v1","policy_id":"FSVCCR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"volatility":{"path":str(vm),"sha256":sha(vm)},"completed_price":{"path":str(pm),"sha256":sha(pm)},"train_funding":{"path":str(source.engine.TRAIN_FUNDING),"sha256":sha(source.engine.TRAIN_FUNDING)},"later_funding":{"table":"funding_rates_binance","symbol":"BTCUSDT"}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
