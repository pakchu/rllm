"""Outcome-blind preregistration for HVCMO-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVCMO-24";DEFAULT_OUTPUT=Path("results/high_volatility_chande_momentum_oscillator_cross_relay_preregistration_2026-08-11.json")
def canonical_hash(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_chande_momentum_oscillator_cross_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"During elevated realized variation, a completed four-hour canonical Chande Momentum Oscillator zero crossing identifies a reversal in the balance of raw upward versus downward close movement; follow the new sign for twenty-four hours.","side":"long when CMO crosses strictly above zero; short when CMO crosses strictly below zero","why_distinct":"CMO uses unsmoothed rolling sums of all positive and negative completed close changes over 14 periods. It is not RSI Wilder smoothing, a regression endpoint, moving-average crossover, prior candidate repair, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"sparse four-hour raw gain-loss balance zero crossings are absent from Gross9 primitives"},
 external_basis={"origin":"Tushar Chande canonical Momentum Oscillator convention","fixed_definition":"CMO=100*(sum positive close changes-sum absolute negative close changes)/(their sum) over 14 completed changes; strict zero-line crossover","selection_use":"published unsmoothed formula, common default 14, and zero-line direction only; no incidence or outcomes"},
 features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)","change":"current completed close minus immediately prior completed close","gains_losses":"positive change in gain bucket and absolute negative change in loss bucket","cmo":"100*(rolling 14 gain sum-loss sum)/(gain sum+loss sum), strict positive denominator and complete gap-free window","event":"current completed CMO strictly crosses above zero from <=0 or below zero from >=0","variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
 clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":0.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"momentum_periods":14,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":0.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
 diagnostic_controls={"names":["no_variation_gate","raw_close_direction","one_bar_stale_cross","direction_flip"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"canonical_cmo_definition_read":True,"repository_cmo_candidate_found":False,"prior_rsi_and_cfo_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical Chande Momentum Oscillator(14) zero crossing under the requested high-variation regime"},
 stopping_rule="Terminal first failure; no momentum period, sum formula, variation, side, hold, clock, subset, control, or other repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x:dict[str,Any])->None:
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVCMO preregistration drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();x=build();validate(x);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(a.output)
