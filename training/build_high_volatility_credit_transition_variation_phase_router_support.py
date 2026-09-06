"""Deterministic source support for HVCQVAR-24."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_credit_transition_variation_phase_router as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="f11c7a2a5b02f8e1b2cdddaccf98d66c7943d29e3e271a36119096d50dfc8559";CLOCK=Path("data/high_volatility_credit_transition_variation_phase_router_clocks_2023_2026.csv.gz");RESULT=Path("results/high_volatility_credit_transition_variation_phase_router_support_2026-08-18.json");STAGES={k:tuple(pd.Timestamp(v) for v in values) for k,values in prereg.build()["stages"].items()};GATES=prereg.build()["source_support_gates"]
def build_clock(states:pd.DataFrame)->pd.DataFrame:
 x=states.copy();x["decision_time"]=pd.to_datetime(x["decision_time"],utc=True,errors="raise")
 for c in ("hyg_return","lqd_return","relative_credit_return","btc_variation","btc_variation_rank"):x[c]=pd.to_numeric(x[c],errors="coerce")
 previous=x["relative_credit_return"].shift(1);previous_variation=x["btc_variation"].shift(1);previous_valid=x["source_valid"].shift(1,fill_value=False);x["variation_change"]=x["btc_variation"]-previous_variation
 active=(x["source_valid"].eq(True)&previous_valid&np.isfinite(x[["relative_credit_return","btc_variation","btc_variation_rank","variation_change"]]).all(axis=1)&np.isfinite(previous)&x["relative_credit_return"].ne(0)&previous.ne(0)&x["variation_change"].ne(0)&(np.sign(x["relative_credit_return"])==-np.sign(previous))&x["btc_variation_rank"].ge(prereg.build()["policy"]["variation_rank_min"]));rows=[];reserved=None
 for r in x[active].itertuples(index=False):
  d=pd.Timestamp(r.decision_time);e=d+pd.Timedelta("5m");z=e+pd.Timedelta("24h")
  if reserved is not None and e<reserved:continue
  split=next((k for k,(a,b) in STAGES.items() if a<=e and z<=b),None)
  if split is None:continue
  reserved=z;rows.append({"candidate":prereg.POLICY_ID,"control":"primary","split":split,"session_date":r.session_date,"decision_time":d,"feature_available_time":d,"entry_time":e,"exit_time":z,"side":int(np.sign(r.relative_credit_return)*np.sign(r.variation_change)),"hyg_return":float(r.hyg_return),"lqd_return":float(r.lqd_return),"relative_credit_return":float(r.relative_credit_return),"btc_variation":float(r.btc_variation),"btc_variation_rank":float(r.btc_variation_rank),"variation_change":float(r.variation_change)})
 return pd.DataFrame(rows,columns=("candidate","control","split","session_date","decision_time","feature_available_time","entry_time","exit_time","side","hyg_return","lqd_return","relative_credit_return","btc_variation","btc_variation_rank","variation_change"))
def stats(clock:pd.DataFrame,k:str)->dict[str,Any]:
 x=clock[clock["split"].eq(k)];n=len(x);l=int(x["side"].eq(1).sum());s=int(x["side"].eq(-1).sum());m=float(x["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.;return {"events":n,"longs":l,"shorts":s,"minority_side_share":min(l,s)/n if n else 0.,"max_month_share":m}
def run(clock_path:Path=CLOCK,result_path:Path=RESULT)->dict[str,Any]:
 if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCQVAR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);states=pd.read_csv(prereg.SOURCE);clock=build_clock(states);_write_gzip_csv(clock,clock_path);support={k:stats(clock,k) for k in STAGES};checks={}
 for k,v in support.items():checks[f"{k}_minimum_events"]=v["events"]>=GATES["minimum_events"][k];checks[f"{k}_side_balance"]=v["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{k}_month_concentration"]=v["max_month_share"]<=GATES["max_month_share"]
 passed=all(checks.values());core={"protocol_version":"hvcqvar_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source":{"path":str(prereg.SOURCE),"sha256":prereg.SOURCE_SHA,"rows":len(states)},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_opened":False,"gross9_rows_opened":False,"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
