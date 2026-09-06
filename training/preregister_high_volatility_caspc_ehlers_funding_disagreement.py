"""Outcome-blind exploratory preregistration for HVCELVFD-8."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_cross_structure_action_vote as contract
POLICY_ID="HVCELVFD-8"
DEFAULT_OUTPUT=Path("results/high_volatility_caspc_ehlers_funding_disagreement_preregistration_2026-08-16.json")
BASE={"preregistration":{"path":"results/high_volatility_caspc_ehlers_active_veto_preregistration_2026-08-16.json","sha256":"7dc8f6228fa32eba7d5aded586d389855a47fbf12917800c43537e5e18c6244f"},"support":{"path":"results/high_volatility_caspc_ehlers_active_veto_support_2026-08-16.json","sha256":"9b7d2230588465a73dd9646387d3d7000b3f26d98daeef4e25cf51f4118c6375"},"gross9":{"path":"results/high_volatility_caspc_ehlers_active_veto_gross9_novelty_2026-08-16.json","sha256":"b7d94871688fce546a6a187594cf9665856193d7129cba4c2ca6e69fa9aa3b17"},"clock":{"path":"data/high_volatility_caspc_ehlers_active_veto_clocks_2023_2026.csv.gz","sha256":"1f8512a74906921169b115ec917c80c76b714c5a2f5fda25698feb6b4b026294"}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build():
 c=contract.build();core={"protocol_version":"high_volatility_caspc_ehlers_funding_disagreement_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-16","exploratory_discovery":True,"fresh_confirmatory_evidence":False,"source_incidence_opened":False,"outcomes_opened":False,"gross9_rows_opened":False,"singleton":True,"candidate_family":[POLICY_ID],"candidate_family_size":1,"base_artifacts":BASE,
 "construction":{"base":"immutable HVCELV-8 primary clock and side","funding_source":"latest BTCUSDT funding_rates_binance settlement with funding_time<=decision and funding_time>decision-8h","gate":"strict side*funding_rate<0; zero or missing is ineligible","entry":"immutable base D+5m","hold":"8 elapsed hours","additional_or_tuned_magnitude_thresholds":"none","alternatives":"none"},
 "mechanism":{"claim":"Cross-alt serial persistence that survives an active Ehlers opposite veto is less crowded when its side disagrees with the latest settled perpetual funding sign, favoring continuation during volatile regimes.","why_low_gross9_overlap_is_plausible":"a sparse three-way cross-alt, oscillator-veto and settled-funding disagreement conjunction is absent from Gross9"},
 "clock":{"decision":"immutable base 03:00/11:00/19:00 UTC","entry":"D+5m","hold":"8 elapsed hours","funding":"latest already settled event is a source signal; held-interval settlements remain sealed until economics"},
 "stages":c["stages"],"source_support_gates":c["source_support_gates"],"gross9_novelty_gates":c["gross9_novelty_gates"],"economic_gates":c["economic_gates"],
 "source_plan":{"table":"funding_rates_binance","symbol":"BTCUSDT","columns":["funding_time","funding_rate"],"read_after_preregistration":True,"postentry_prices_returns_pnl":"sealed"},
 "research_boundary":{"HVCELV_train_pass_test_positive_but_terminal_known":True,"exact_funding_disagreement_incidence_or_outcomes_known":False,"source_incidence_opened":False,"postentry_return_or_pnl_opened":False,"classification":"exploratory discovery; not fresh confirmatory evidence","repair_of_prior_candidate":False,"selection_basis":"causal funding-crowding disagreement added to immutable cross-alt oscillator-veto base"},
 "stopping_rule":"source support, Gross9, train/test/eval/final; stop first failure; no threshold, sign, subset, base, clock, hold, funding-window, substitution, or control repair."}
 return {**core,"manifest_hash":canonical_hash(core)}
def validate(v):
 if v["manifest_hash"]!=canonical_hash({k:x for k,x in v.items() if k!="manifest_hash"}):raise RuntimeError("manifest drift")
 for a in BASE.values():
  if sha(a["path"])!=a["sha256"]:raise RuntimeError("base drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n");print(a.output)
