"""Deterministic source support for HVECTR-24."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_energy_commodity_transition_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="e822bc5895b39fcdef97997178bbf3409ea94fd412b25e00555ad4e66802f4f5";CLOCK=Path("data/high_volatility_energy_commodity_transition_relay_clocks_2023_2026.csv.gz");RESULT=Path("results/high_volatility_energy_commodity_transition_relay_support_2026-08-18.json");STAGES={k:tuple(pd.Timestamp(v) for v in values) for k,values in prereg.build()["stages"].items()};GATES=prereg.build()["source_support_gates"]
def build_clock(states:pd.DataFrame)->pd.DataFrame:
 x=states.copy();x["cash_close_time"]=pd.to_datetime(x["cash_close_time"],utc=True,errors="raise")
 for c in ("uso_intraday_return","bno_intraday_return","ung_intraday_return","spillover_score","btc_realized_variation","btc_variation_rank"):x[c]=pd.to_numeric(x[c],errors="coerce")
 previous=x["spillover_score"].shift(1);valid=np.isfinite(x[["spillover_score","btc_realized_variation","btc_variation_rank"]]).all(axis=1);previous_valid=valid.shift(1,fill_value=False)
 active=(valid&previous_valid&np.isfinite(previous)&x["spillover_score"].ne(0)&previous.ne(0)&(np.sign(x["spillover_score"])==-np.sign(previous))&x["btc_variation_rank"].ge(prereg.build()["policy"]["variation_rank_min"]));rows=[];reserved=None
 for r in x[active].itertuples(index=False):
  close=pd.Timestamp(r.cash_close_time);d=close+pd.Timedelta("5m");e=close+pd.Timedelta("10m");z=e+pd.Timedelta("24h")
  if reserved is not None and e<reserved:continue
  split=next((k for k,(a,b) in STAGES.items() if a<=e and z<=b),None)
  if split is None:continue
  reserved=z;rows.append({"candidate":prereg.POLICY_ID,"control":"primary","split":split,"session_date":r.session_date,"decision_time":d,"feature_available_time":d,"entry_time":e,"exit_time":z,"side":int(np.sign(r.spillover_score)),"uso_return":float(r.uso_intraday_return),"bno_return":float(r.bno_intraday_return),"ung_return":float(r.ung_intraday_return),"spillover_score":float(r.spillover_score),"btc_variation":float(r.btc_realized_variation),"btc_variation_rank":float(r.btc_variation_rank)})
 return pd.DataFrame(rows,columns=("candidate","control","split","session_date","decision_time","feature_available_time","entry_time","exit_time","side","uso_return","bno_return","ung_return","spillover_score","btc_variation","btc_variation_rank"))
def stats(clock:pd.DataFrame,k:str)->dict[str,Any]:
 x=clock[clock["split"].eq(k)];n=len(x);l=int(x["side"].eq(1).sum());s=int(x["side"].eq(-1).sum());m=float(x["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.;return {"events":n,"longs":l,"shorts":s,"minority_side_share":min(l,s)/n if n else 0.,"max_month_share":m}
def run(clock_path:Path=CLOCK,result_path:Path=RESULT)->dict[str,Any]:
 if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVECTR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);states=pd.read_csv(prereg.SOURCE);clock=build_clock(states);_write_gzip_csv(clock,clock_path);support={k:stats(clock,k) for k in STAGES};checks={}
 for k,v in support.items():checks[f"{k}_minimum_events"]=v["events"]>=GATES["minimum_events"][k];checks[f"{k}_side_balance"]=v["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{k}_month_concentration"]=v["max_month_share"]<=GATES["max_month_share"]
 passed=all(checks.values());core={"protocol_version":"hvectr_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source":{"path":str(prereg.SOURCE),"sha256":prereg.SOURCE_SHA,"rows":len(states)},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_opened":False,"gross9_rows_opened":False,"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
