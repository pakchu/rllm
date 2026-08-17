"""Deterministic source support for HVSMRSR-8."""
from __future__ import annotations
import json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_spot_median_return_shift_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="a920ebc4bd9be6126f1889b263314d55d8c3ef7535fc337e404a3bc1aae12b6a";FEATURES=Path("data/high_volatility_spot_median_return_shift_relay_features_2023_2026.csv.gz");CLOCK=Path("data/high_volatility_spot_median_return_shift_relay_clocks_2023_2026.csv.gz");RESULT=Path("results/high_volatility_spot_median_return_shift_relay_support_2026-08-18.json");POLICY=prereg.build()["policy"];STAGES={k:tuple(pd.Timestamp(v) for v in values) for k,values in prereg.build()["stages"].items()};GATES=prereg.build()["source_support_gates"]
def causal_rank(values:pd.Series,valid:pd.Series)->pd.Series:
 out=np.full(len(values),np.nan);history=[]
 for i,(value,is_valid) in enumerate(zip(values.to_numpy(float),valid.to_numpy(bool))):
  prior=np.asarray(history[-POLICY["history_decisions"]:],float)
  if is_valid and math.isfinite(value) and len(prior)>=POLICY["minimum_history_decisions"]:out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if is_valid and math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=values.index)
def prepare(perp:pd.DataFrame,spot:pd.DataFrame)->pd.DataFrame:
 p=perp[["decision_time","feature_available_time","source_valid","median_shift","variation_rank"]].copy();s=spot[["decision_time","source_valid","spot_shift"]].copy().rename(columns={"source_valid":"spot_source_valid"})
 for x in (p,s):x["decision_time"]=pd.to_datetime(x["decision_time"],utc=True,errors="raise")
 p["feature_available_time"]=pd.to_datetime(p["feature_available_time"],utc=True,errors="raise");x=p.merge(s,on="decision_time",how="inner",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
 for c in ("median_shift","spot_shift","variation_rank"):x[c]=pd.to_numeric(x[c],errors="coerce")
 x["joint_valid"]=(x["source_valid"].eq(True)&x["spot_source_valid"].eq(True)&np.isfinite(x[["median_shift","spot_shift","variation_rank"]]).all(axis=1));x["spot_shift_signal"]=x["spot_shift"];rank_valid=x["joint_valid"]&x["spot_shift_signal"].ne(0);x["shift_rank"]=causal_rank(x["spot_shift_signal"].abs(),rank_valid)
 level=(rank_valid&x["shift_rank"].ge(POLICY["shift_rank_min"])&x["variation_rank"].ge(POLICY["variation_rank_min"]));consecutive=x["decision_time"].sub(x["decision_time"].shift(1)).eq(pd.Timedelta("8h"));x["eligible"]=level&consecutive&~level.shift(1,fill_value=False)&x["joint_valid"].shift(1,fill_value=False);return x
def stage_for(entry:pd.Timestamp,exit_:pd.Timestamp)->str|None:return next((k for k,(a,b) in STAGES.items() if a<=entry and exit_<=b),None)
def build_clock(features:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for r in features[features["eligible"]].itertuples(index=False):
  d=pd.Timestamp(r.decision_time);e=d+pd.Timedelta("5m");z=e+pd.Timedelta("8h");split=stage_for(e,z)
  if split is None:continue
  side=int(np.sign(r.spot_shift_signal));rows.append({"candidate":prereg.POLICY_ID,"control":"primary","split":split,"decision_time":d,"feature_available_time":r.feature_available_time,"entry_time":e,"exit_time":z,"side":side,"spot_shift_signal":float(r.spot_shift_signal),"shift_rank":float(r.shift_rank),"spot_shift":float(r.spot_shift),"variation_rank":float(r.variation_rank)})
 return pd.DataFrame(rows,columns=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","spot_shift_signal","shift_rank","spot_shift","variation_rank"))
def stats(clock:pd.DataFrame,k:str)->dict[str,Any]:
 x=clock[clock["split"].eq(k)];n=len(x);l=int(x["side"].eq(1).sum());s=int(x["side"].eq(-1).sum());m=float(x["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.;return {"events":n,"longs":l,"shorts":s,"minority_side_share":min(l,s)/n if n else 0.,"max_month_share":m}
def run(features_path:Path=FEATURES,clock_path:Path=CLOCK,result_path:Path=RESULT)->dict[str,Any]:
 if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVSMRSR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);features=prepare(pd.read_csv(prereg.PERP),pd.read_csv(prereg.SPOT));_write_gzip_csv(features,features_path);clock=build_clock(features);_write_gzip_csv(clock,clock_path);support={k:stats(clock,k) for k in STAGES};checks={}
 for k,v in support.items():checks[f"{k}_minimum_events"]=v["events"]>=GATES["minimum_events"][k];checks[f"{k}_side_balance"]=v["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{k}_month_concentration"]=v["max_month_share"]<=GATES["max_month_share"]
 passed=all(checks.values());core={"protocol_version":"hvsmrsr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_opened":False,"gross9_rows_opened":False,"features":{"path":str(features_path),"sha256":prereg.sha256(features_path),"rows":len(features),"joint_valid_rows":int(features["joint_valid"].sum())},"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
