"""Build outcome-blind OOS support for frozen CLMSRR-6."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_confirmation_ladder_multistate_ridge_relay as prereg
from training import train_confirmation_ladder_multistate_ridge_relay as train
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="7e09b0452fddc9310e57bfb9aa8f611da84d29d6a76556495e4c70a0255bd1f0";MODEL=Path("results/confirmation_ladder_multistate_ridge_relay_model_freeze_2026-08-12.json");MODEL_SHA="ecb2d241d202c50f4396eb087c9dfa822306e0a471bee4c42e3cd26fb60fabd4"
ROOT=Path("data/confirmation_ladder_multistate_ridge_relay_sources_2023_2026");FEATURES=ROOT/"scored_ladders.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/confirmation_ladder_multistate_ridge_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/confirmation_ladder_multistate_ridge_relay_controls_2023_2026");RESULT=Path("results/confirmation_ladder_multistate_ridge_relay_support_2026-08-12.json")
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in prereg.build()["stages"].items()};CONTROLS=tuple(prereg.build()["diagnostic_controls"]["names"]);GATES=prereg.build()["source_support_gates"]
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def score(features:pd.DataFrame,model:dict[str,Any])->pd.DataFrame:
 out=features.copy();matrix=out.loc[:,prereg.FEATURES].to_numpy(float);mean=np.asarray(model["standardization"]["mean"],float);scale=np.asarray(model["standardization"]["scale"],float);weights=np.asarray(model["ridge"]["weights"],float);out["prediction"]=float(model["ridge"]["intercept"])+((matrix-mean)/scale)@weights;threshold=model["frozen_thresholds"];out["eligible"]=out.source_valid&out.variation.ge(threshold["variation_q65"])&out.prediction.abs().ge(threshold["absolute_prediction_q75"])&out.prediction.ne(0);return out
def active_and_side(features:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 prediction=features.prediction.copy();variation=features.variation.copy();valid=features.source_valid.copy();available=features.feature_available_time.copy();threshold=json.loads(MODEL.read_text())["frozen_thresholds"]
 if control=="one_anchor_stale_features":prediction=prediction.shift(1);variation=variation.shift(1);valid=valid.shift(1,fill_value=False)
 variation_gate=variation.ge(threshold["variation_q65"]);strength=prediction.abs().ge(threshold["absolute_prediction_q75"]);state=valid&variation_gate&strength&prediction.ne(0)
 if control=="no_variation_gate":state=valid&strength&prediction.ne(0)
 elif control=="no_prediction_strength_gate":state=valid&variation_gate&prediction.ne(0)
 side=np.sign(prediction).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return state&side.ne(0),side,available
def build_clock(features:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side,available=active_and_side(features,control);rows=[];reserved=None
 for i in features.index[active]:
  entry=pd.Timestamp(available.at[i])+pd.Timedelta("5m");exit_time=entry+pd.Timedelta("6h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  reserved=exit_time;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"anchor_height":int(features.at[i,"anchor_height"]),"feature_available_time":available.at[i],"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),"prediction":float(features.at[i,"prediction"]),"variation":float(features.at[i,"variation"])})
 return pd.DataFrame(rows,columns=["candidate","control","split","anchor_height","feature_available_time","entry_time","exit_time","side","prediction","variation"])
def stats(clock:pd.DataFrame,split:str)->dict[str,float|int]:
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 longs=int(x.side.eq(1).sum());shorts=int(x.side.eq(-1).sum());months=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":longs,"shorts":shorts,"minority_side_share":min(longs,shorts)/len(x),"max_month_share":int(months.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA or sha(MODEL)!=MODEL_SHA or sha(train.PANEL)!=train.PANEL_SHA or sha(train.DURATION)!=train.DURATION_SHA:raise RuntimeError("CLMSRR frozen input drift")
 model=json.loads(MODEL.read_text());model_core={k:v for k,v in model.items() if k!="manifest_hash"}
 if model["manifest_hash"]!=chash(model_core) or model["oos_source_incidence_opened"] is not False:raise RuntimeError("CLMSRR model freeze drift")
 features=score(train.feature_frame(pd.read_csv(train.PANEL,compression="gzip"),pd.read_csv(train.DURATION,compression="gzip")),model);features=features[features.feature_available_time.ge(pd.Timestamp("2023-07-01T00:00:00Z"))].reset_index(drop=True);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};ROOT.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"clmsrr_6_source_v1","inputs":{"panel":{"path":str(train.PANEL),"sha256":train.PANEL_SHA},"duration":{"path":str(train.DURATION),"sha256":train.DURATION_SHA},"model":{"path":str(MODEL),"sha256":MODEL_SHA}},"scored_features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"preentry_sources_opened":True,"oos_incidence_opened":True,"oos_execution_price_return_pnl_opened":False,"gross9_rows_opened":False};manifest={**source_core,"manifest_hash":chash(source_core)};MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"clmsrr_6_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"model":{"path":str(MODEL),"sha256":MODEL_SHA,"manifest_hash":model["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
