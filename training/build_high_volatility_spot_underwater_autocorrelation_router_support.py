"""Source-only long-memory autocorrelation routing for HVSAUDAR-8."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_spot_underwater_autocorrelation_router as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="a8a9a64cc9eda46c02fc4f0f511d641ccd754c740c9cb41d18194e4780842feb";BASE=Path(prereg.BASE["clock"]["path"]);STATES=Path("data/high_volatility_spot_underwater_autocorrelation_router_states_2023_2026.csv.gz");CLOCK=Path("data/high_volatility_spot_underwater_autocorrelation_router_clocks_2023_2026.csv.gz");RESULT=Path("results/high_volatility_spot_underwater_autocorrelation_router_support_2026-08-18.json");POLICY=prereg.build()["policy"];GATES=prereg.build()["source_support_gates"]
def state_scores(states:pd.DataFrame)->pd.DataFrame:
 x=states[["decision_time","source_valid","block_return"]].copy();x["decision_time"]=pd.to_datetime(x["decision_time"],utc=True,errors="raise");x["block_return"]=pd.to_numeric(x["block_return"],errors="coerce");scores=[];counts=[]
 for i,row in x.iterrows():
  prior=x.iloc[:i];prior=prior[prior["source_valid"].eq(True)&np.isfinite(prior["block_return"])].tail(POLICY["history_blocks"]);counts.append(len(prior))
  if len(prior)<POLICY["minimum_history_blocks"]:scores.append(np.nan);continue
  values=prior["block_return"].to_numpy(float);left,right=values[:-1],values[1:]
  scores.append(float(np.corrcoef(left,right)[0,1]) if np.std(left,ddof=1)>0 and np.std(right,ddof=1)>0 else np.nan)
 x["history_count"]=counts;x["autocorrelation"]=scores;return x
def route(base:pd.DataFrame,scores:pd.DataFrame)->pd.DataFrame:
 b=base.copy()
 for c in ("decision_time","feature_available_time","entry_time","exit_time"):b[c]=pd.to_datetime(b[c],utc=True,errors="raise")
 x=b.merge(scores[["decision_time","history_count","autocorrelation"]],on="decision_time",how="left",validate="one_to_one");keep=np.isfinite(x["autocorrelation"])&x["autocorrelation"].ne(0)&x["history_count"].ge(POLICY["minimum_history_blocks"]);x=x[keep].copy();x["base_side"]=x["side"].astype(int);x["side"]=x["base_side"]*np.sign(x["autocorrelation"]).astype(int);x["candidate"]=prereg.POLICY_ID;return x[["candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","base_side","history_count","autocorrelation"]].sort_values("decision_time",kind="stable").reset_index(drop=True)
def stats(clock:pd.DataFrame,k:str)->dict[str,Any]:
 x=clock[clock["split"].eq(k)];n=len(x);l=int(x["side"].eq(1).sum());s=int(x["side"].eq(-1).sum());m=float(x["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.;return {"events":n,"longs":l,"shorts":s,"minority_side_share":min(l,s)/n if n else 0.,"max_month_share":m}
def run(states_path:Path=STATES,clock_path:Path=CLOCK,result_path:Path=RESULT)->dict[str,Any]:
 if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVSAUDAR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);scores=state_scores(pd.read_csv(prereg.STATE));_write_gzip_csv(scores,states_path);clock=route(pd.read_csv(BASE),scores);_write_gzip_csv(clock,clock_path);support={k:stats(clock,k) for k in prereg.build()["stages"]};checks={}
 for k,v in support.items():checks[f"{k}_minimum_events"]=v["events"]>=GATES["minimum_events"][k];checks[f"{k}_side_balance"]=v["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{k}_month_concentration"]=v["max_month_share"]<=GATES["max_month_share"]
 passed=all(checks.values());core={"protocol_version":"hvsaudar_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"completed_preentry_sources_opened":True,"current_or_future_state_returns_used":False,"postentry_return_pnl_execution_price_opened":False,"funding_opened":False,"gross9_rows_opened":False,"states":{"path":str(states_path),"sha256":prereg.sha256(states_path),"rows":len(scores)},"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"causality_checks":{"current_block_excluded":True,"stage_resets":False},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
