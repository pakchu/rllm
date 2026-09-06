"""Outcome-blind preregistration for HVTRIX-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVTRIX-24"
DEFAULT_OUTPUT=Path("results/high_volatility_trix_zero_cross_relay_preregistration_2026-08-11.json")

def canonical_hash(value:Any)->str:
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 contract=copy.deepcopy(template.build());contract.pop("manifest_hash")
 contract.update(
  protocol_version="high_volatility_trix_zero_cross_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
  mechanism={"claim":"During elevated realized variation, a completed four-hour canonical TRIX zero crossing identifies a reversal in the rate of change of a triple-smoothed trend; follow the new sign for twenty-four hours.","side":"long when TRIX crosses strictly above zero; short when TRIX crosses strictly below zero","why_distinct":"TRIX is the one-period percentage change of a third consecutive EMA(15), suppressing shorter cycles before detecting trend acceleration. It is not a raw return, price/average cross, prior candidate repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"sparse four-hour triple-smoothed rate zero crossings are absent from Gross9 primitives"},
  external_basis={"origin":"Jack Hutson canonical TRIX convention","fixed_definition":"one-period percentage rate of change of the third consecutive 15-period EMA of close; strict zero-line crossover","selection_use":"published 15-period triple-EMA definition and zero-line direction only; no incidence or outcomes"},
  features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)","ema_chain":"three consecutive causal EMA(15) transforms, reset across source gaps and each emitted only after its full warmup","trix":"100*(EMA3/current prior EMA3-1), finite with strict positive prior EMA3","event":"current completed TRIX strictly crosses above zero from <=0 or below zero from >=0","variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
  clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":0.5,"funding":"not a signal input; exact settlements only after novelty passes"},
  policy={"ema_periods":15,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":0.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
  diagnostic_controls={"names":["no_variation_gate","single_ema_rate_cross","one_bar_stale_cross","direction_flip"],"cannot_be_promoted":True},
  source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
  research_boundary={"canonical_trix_definition_read":True,"repository_trix_candidate_found":False,"prior_candidate_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical TRIX(15) zero crossing under the requested high-variation regime"},
  stopping_rule="Terminal first failure; no EMA period, rate definition, variation, side, hold, clock, subset, control, or other repair.")
 return {**contract,"manifest_hash":canonical_hash(contract)}

def validate(contract:dict[str,Any])->None:
 core={k:v for k,v in contract.items() if k!="manifest_hash"}
 if contract.get("manifest_hash")!=canonical_hash(core) or contract!=build():raise RuntimeError("HVTRIX preregistration drift")

if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();value=build();validate(value);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");print(args.output)
