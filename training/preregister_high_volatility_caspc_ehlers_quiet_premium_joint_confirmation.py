"""Exploratory outcome-blind preregistration for HVCELVPQA-8."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_cross_structure_action_vote as contract

POLICY_ID="HVCELVPQA-8"
DEFAULT_OUTPUT=Path("results/high_volatility_caspc_ehlers_quiet_premium_joint_confirmation_preregistration_2026-08-16.json")
COMPONENTS={
 "HVCELV-8":{
  "preregistration":{"path":"results/high_volatility_caspc_ehlers_active_veto_preregistration_2026-08-16.json","sha256":"7dc8f6228fa32eba7d5aded586d389855a47fbf12917800c43537e5e18c6244f"},
  "support":{"path":"results/high_volatility_caspc_ehlers_active_veto_support_2026-08-16.json","sha256":"9b7d2230588465a73dd9646387d3d7000b3f26d98daeef4e25cf51f4118c6375"},
  "gross9":{"path":"results/high_volatility_caspc_ehlers_active_veto_gross9_novelty_2026-08-16.json","sha256":"b7d94871688fce546a6a187594cf9665856193d7129cba4c2ca6e69fa9aa3b17"},
  "clock":{"path":"data/high_volatility_caspc_ehlers_active_veto_clocks_2023_2026.csv.gz","sha256":"1f8512a74906921169b115ec917c80c76b714c5a2f5fda25698feb6b4b026294"}},
 "HVCASPCPQA-8":{
  "preregistration":{"path":"results/high_volatility_caspc_quiet_premium_activity_preregistration_2026-08-16.json","sha256":"d3fc1e0d152abd2c87a568efa6c2236389bf787de2271c847713ab99c2783e01"},
  "support":{"path":"results/high_volatility_caspc_quiet_premium_activity_support_2026-08-16.json","sha256":"f0498c805b3bf93a9fe63fc374f03b7afe0d803382104bc2aa7c6b0fc92c67a2"},
  "gross9":{"path":"results/high_volatility_caspc_quiet_premium_activity_gross9_novelty_2026-08-16.json","sha256":"b8cf184ed63dc8b9b8934413ec3641a7992d2d161802711f784548b7e1fa0fe1"},
  "clock":{"path":"data/high_volatility_caspc_quiet_premium_activity_clocks_2023_2026.csv.gz","sha256":"684ddfc240679d99ed5d5f30f095a359940c9ed098e5c6b0cf7f15d6fdf6bcce"}}}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_hash(v:Any): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build():
 c=contract.build()
 core={"protocol_version":"high_volatility_caspc_ehlers_quiet_premium_joint_confirmation_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-16","exploratory_discovery":True,"fresh_confirmatory_evidence":False,"combined_incidence_opened":False,"combined_outcomes_opened":False,"gross9_rows_opened":False,"singleton":True,"candidate_family":[POLICY_ID],"candidate_family_size":1,"component_ids":list(COMPONENTS),"component_artifacts":COMPONENTS,
 "construction":{"operator":"exact-entry joint sponsorship intersection","decision_join":"emit only when frozen HVCELV-8 and HVCASPCOIS-8 clocks have identical decision_time, entry_time, exit_time and side","timestamp_tolerance":"none","direction":"immutable common side","entry":"immutable common entry","hold":"8 elapsed hours","additional_or_tuned_thresholds":"none","alternatives":"none"},
 "mechanism":{"claim":"Cross-alt serial persistence must survive an active opposite Ehlers veto while relative premium activity remains in its frozen quiet rank state; exact agreement isolates continuation without crowded derivative repricing.","why_low_gross9_overlap_is_plausible":"the sparse exact intersection of oscillator-veto and quiet-premium cross-alt persistence is absent from Gross9 primitives"},
 "clock":{"decision":"exact common 03:00/11:00/19:00 UTC component decision","entry":"D+5m","hold":"8 elapsed hours","funding":"sealed until novelty pass"},
 "stages":c["stages"],"source_support_gates":c["source_support_gates"],"gross9_novelty_gates":c["gross9_novelty_gates"],"economic_gates":c["economic_gates"],
 "research_boundary":{"all_component_prior_outcomes_known":True,"HVCELV_train_pass_test_positive_but_terminal_known":True,"HVCASPCPQA_train_stress_and_half_failure_known":True,"combined_incidence_opened":False,"combined_postentry_returns_or_pnl_opened":False,"classification":"exploratory discovery; not fresh confirmatory evidence","repair_of_prior_candidate":False,"selection_basis":"independent oscillator veto and quiet-premium confirmation on one frozen cross-alt persistence primary"},
 "stopping_rule":"source support, Gross9, then train/test/eval/final; stop first failure; no substitution, threshold, side, clock, intersection, component, or control repair."}
 return {**core,"manifest_hash":canonical_hash(core)}
def validate(v):
 if v["manifest_hash"]!=canonical_hash({k:x for k,x in v.items() if k!="manifest_hash"}): raise RuntimeError("manifest drift")
 for artifacts in COMPONENTS.values():
  for a in artifacts.values():
   if sha(a["path"])!=a["sha256"]: raise RuntimeError(f"component drift: {a['path']}")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n");print(a.output)
