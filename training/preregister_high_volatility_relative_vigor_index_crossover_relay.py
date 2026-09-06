"""Outcome-blind preregistration for HVRVI-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVRVI-24"
DEFAULT_OUTPUT=Path("results/high_volatility_relative_vigor_index_crossover_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_relative_vigor_index_crossover_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"During elevated realized variation, a canonical ten-period Relative Vigor Index crossing its four-period signal line on completed four-hour BTC auctions identifies a change in the dominance of close-versus-open candle-body conviction relative to traded range; follow the new relation for twenty-four hours.","side":"long when RVI crosses strictly above its signal line; short on the strict reverse crossover","why_distinct":"HVRVI compares symmetrically smoothed close-open bodies with equally smoothed high-low ranges, then crosses a second symmetric signal line. It is neither a close-location stochastic, return oscillator, volume/flow measure, OI, funding, premium, nor Gross9 primitive.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%, restricting body-versus-range conviction changes to July-like volatile conditions","why_low_gross9_overlap_is_plausible":"sparse four-hour RVI/signal crossover clocks are absent from Gross9 primitives"},
 external_basis={"origin":"John Ehlers, canonical Relative Vigor Index convention","fixed_definition":"four-bar symmetric weights 1,2,2,1 divided by 6 applied separately to close-open and high-low; ten-period sums of smoothed numerator and denominator; RVI signal uses the same four-bar symmetric weights; strict line crossover","selection_use":"published formula, periods, and crossover direction only; no incidence or outcomes"},
 features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)","body_smooth":"((C-O)+2*(C1-O1)+2*(C2-O2)+(C3-O3))/6 on completed bars","range_smooth":"((H-L)+2*(H1-L1)+2*(H2-L2)+(H3-L3))/6","relative_vigor_index":"rolling ten-period sum(body_smooth)/rolling ten-period sum(range_smooth), denominator finite and strictly positive","signal":"four-period symmetric 1,2,2,1 smoothing of RVI divided by 6","cross":"strict current RVI/signal crossover from the immediately prior valid state","variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
 clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"rvi_periods":10,"symmetric_weights":[1,2,2,1],"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_variation_gate","raw_body_range_cross","one_bar_stale_crossover","direction_flip"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"canonical_relative_vigor_index_definition_read":True,"repository_relative_vigor_index_candidate_found":False,"prior_candle_oscillator_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical Relative Vigor Index signal crossover under the requested high-variation regime"},
 stopping_rule="Terminal first failure; no RVI formula, period, weighting, crossover, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVRVI preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
