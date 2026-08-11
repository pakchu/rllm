"""Outcome-blind preregistration for HVTDS-S9-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVTDS-S9-24"
DEFAULT_OUTPUT=Path("results/high_volatility_tom_demark_setup_completion_relay_preregistration_2026-08-11.json")
def canonical_hash(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(
  protocol_version="high_volatility_tom_demark_setup_completion_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
  mechanism={
   "claim":"During elevated realized variation, completion of a canonical nine-bar Tom DeMark Sequential Buy or Sell Setup on completed four-hour BTC auctions identifies exhaustion of a persistent four-bar-relative close trend; trade in the setup's published reversal direction for twenty-four hours.",
   "side":"long only when a Buy Setup reaches step 9; short only when a Sell Setup reaches step 9",
   "why_distinct":"Tom DeMark Setup requires a prerequisite price flip followed by nine uninterrupted closes each strictly below or above the close four bars earlier. It is not a one-bar return sign, moving-average or oscillator crossover, channel, range normalization, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
   "why_suited_to_volatile_regimes":"the mechanism waits for a long directional sequence before fading potential exhaustion, and completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
   "why_low_gross9_overlap_is_plausible":"sparse four-hour completions of a price-flip-qualified nine-step exhaustion sequence are absent from Gross9 primitives"},
  external_basis={
   "origin":"Tom DeMark Sequential; QuantConnect LEAN canonical implementation",
   "definition_source":"https://github.com/QuantConnect/Lean/blob/master/Indicators/TomDemarkSequential.cs",
   "documentation_source":"https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/tom-demark-sequential",
   "fixed_definition":"a Buy Setup starts on a bearish price flip (previous close strictly above its four-bar-prior close, current close strictly below its four-bar-prior close) and completes after nine consecutive qualifying current-close-below-close-four-bars-ago bars; Sell Setup is the strict mirror; emit only at setup step 9",
   "scope":"freeze the official Setup completion only; countdown, perfect-setup qualifier, TDST levels, and all later phases are neither inputs nor selection filters",
   "selection_use":"official price-flip prerequisite, fixed four-bar comparison, fixed nine-step Setup count, and published exhaustion direction only; no incidence or outcomes"},
  features={
   "decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
   "four_bar_relation":"strict comparison of current completed close with the completed close four decisions earlier",
   "price_flip":"Buy Setup starts only on above-to-below four-bar relation; Sell Setup starts only on below-to-above relation",
   "setup_state":"after a flip, increment only while every current relation remains strict in the setup direction; reset immediately on any nonqualifying relation",
   "event":"emit +1 exactly when Buy Setup reaches step 9 and -1 exactly when Sell Setup reaches step 9; no event on steps 1 through 8 or countdown",
   "variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive",
   "variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65",
   "recursive_reset":"all rolling relation and setup state resets after any invalid completed four-hour bar; a new event requires an uninterrupted valid sequence","no_imputation":True},
  clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
  policy={"comparison_lag_bars":4,"setup_bars":9,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
  diagnostic_controls={"names":["no_variation_gate","raw_four_bar_reversal","one_bar_stale_completion","direction_flip"],"cannot_be_promoted":True},
  source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
  research_boundary={"canonical_tom_demark_sequential_definition_read":True,"repository_tom_demark_setup_candidate_found":False,"prior_trend_exhaustion_candidate_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"official Tom DeMark nine-bar Setup completion under the requested high-variation regime"},
  stopping_rule="Terminal first failure; no comparison lag, flip, setup count, qualifier, variation, side, hold, clock, subset, control, countdown, or other repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x:dict[str,Any])->None:
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVTDS-S9-24 preregistration drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();x=build();validate(x);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
