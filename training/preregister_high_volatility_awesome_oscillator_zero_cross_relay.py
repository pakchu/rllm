"""Outcome-blind preregistration for HVAO-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVAO-24";DEFAULT_OUTPUT=Path("results/high_volatility_awesome_oscillator_zero_cross_relay_preregistration_2026-08-11.json")
def canonical_hash(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_awesome_oscillator_zero_cross_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"During elevated realized variation, a completed four-hour canonical Awesome Oscillator zero crossing identifies a shift between short and long median-price momentum; follow the new sign for twenty-four hours.","side":"long when AO crosses strictly above zero; short when AO crosses strictly below zero","why_distinct":"AO subtracts a 34-period simple average of bar median price from its 5-period simple average. It uses neither EMA recursion nor close-only crossover, prior candidate repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"sparse four-hour median-price dual-horizon zero crossings are absent from Gross9 primitives"},
 external_basis={"origin":"Bill Williams canonical Awesome Oscillator convention","fixed_definition":"5-period SMA minus 34-period SMA of median price (high+low)/2; strict zero-line crossover","selection_use":"published 5/34 median-price formula and zero-line direction only; no incidence or outcomes"},
 features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)","median_price":"(completed bar high+completed bar low)/2","awesome_oscillator":"causal SMA(5) median price minus causal SMA(34) median price, both complete and reset across source gaps","event":"current completed AO strictly crosses above zero from <=0 or below zero from >=0","variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
 clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":0.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"fast_periods":5,"slow_periods":34,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":0.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
 diagnostic_controls={"names":["no_variation_gate","close_price_sma_cross","one_bar_stale_cross","direction_flip"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"canonical_awesome_oscillator_definition_read":True,"repository_awesome_oscillator_candidate_found":False,"prior_candidate_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical Awesome Oscillator 5/34 zero crossing under the requested high-variation regime"},
 stopping_rule="Terminal first failure; no SMA period, price basis, variation, side, hold, clock, subset, control, or other repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x:dict[str,Any])->None:
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVAO preregistration drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();x=build();validate(x);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(a.output)
