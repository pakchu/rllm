"""Outcome-blind preregistration for HVPLO-8."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVPLO-8";DEFAULT_OUTPUT=Path("results/high_volatility_price_level_occupation_relay_preregistration_2026-08-10.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_price_level_occupation_relay_v1",policy_id=POLICY_ID,
 mechanism={"claim":"A volatile completed BTC block that spends an unusually large fraction of its minutes strictly on one side of its initial auction price reflects persistent directional acceptance rather than a terminal jump. Follow the occupied side for eight hours.","side":"strict sign of minutes above minus minutes below the first minute open","why_distinct":"Return-sign persistence counts incremental directions; median breadth summarizes bar returns; close location uses terminal position inside a range; path efficiency uses displacement divided by travel; median-shift compares two return halves. HVPLO measures time occupation of a fixed initial price level across the whole block and uses no volume, flow, funding, OI, cross-asset, endpoint gate, prior event, fitted outcome, or control.","why_suited_to_volatile_regimes":"completed realized variation must be in its causal upper 35% while absolute signed level occupation enters its upper quartile","why_low_gross9_overlap_is_plausible":"offset three-daily fixed-level occupation onsets are absent from Gross9 primitives"},
 features={"decision_grid":"exact 03:00/11:00/19:00 UTC boundaries","block":"480 exact coherent bars_binance BTCUSDT one-minute rows [D-8h,D)","minute_return":"log(close/open), finite","reference_level":"open of the first completed minute in the block, finite strict positive","above_count":"number of minute closes strictly above reference_level","below_count":"number of minute closes strictly below reference_level","level_occupation":"(above_count-below_count)/(above_count+below_count), finite strict nonzero denominator and contrast","occupation_rank":"strict-prior midrank of abs(level_occupation) over at most 270 earlier source-valid blocks, minimum 180, current excluded; rank>=0.75","realized_variation":"sqrt(sum squared minute returns), finite strict positive","variation_rank":"strict-prior 270/180 midrank, current excluded; rank>=0.65","eligible_state":"occupation and variation gates pass with level_occupation strict nonzero","onset":"eligible now and immediately previous exact source-valid block ineligible; missing prior cannot trigger","no_imputation":True},
 clock={"decision":"completed eight-hour boundary","entry":"exact BTCUSDT D+5m open","hold":"8 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"history_blocks":270,"minimum_history_blocks":180,"occupation_rank_min":.75,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_occupation_tail","no_variation_gate","median_level_displacement","one_boundary_stale_occupation","direction_flip","same_clock_forced_long"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"prior_sign_persistence_median_breadth_close_location_efficiency_and_median_shift_outcomes_known":True,"repository_fixed_initial_level_occupation_candidate_found":False,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"fixed-level path occupation during elevated variation"},
 stopping_rule="Terminal first failure; no reference level, occupation formula, rank, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVPLO preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
