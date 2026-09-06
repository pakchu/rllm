"""Outcome-blind preregistration for HVVZO-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVVZO-24"
DEFAULT_OUTPUT=Path("results/high_volatility_volume_zone_oscillator_zero_cross_relay_preregistration_2026-08-11.json")

def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_volume_zone_oscillator_zero_cross_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"During elevated realized variation, a canonical fourteen-period Volume Zone Oscillator crossing zero on completed four-hour BTC auctions identifies a change in the direction of volume participation; follow the new participation direction for twenty-four hours.","side":"long when VZO crosses strictly above zero from at or below zero; short when VZO crosses strictly below zero from at or above zero","why_distinct":"HVVZO normalizes exponentially smoothed direction-signed base volume by exponentially smoothed total base volume. It is a bounded directional volume-participation ratio rather than a price-only band/channel/oscillator, taker-flow split, OI, funding, premium, or Gross9 state primitive.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%, focusing the volume-participation reversal clock on July-like volatile conditions","why_low_gross9_overlap_is_plausible":"sparse four-hour normalized signed-volume zero-cross clocks are absent from Gross9 primitives"},
 external_basis={"origin":"Walid Khalil and David Steckler, Volume Zone Oscillator","fixed_definition":"signed volume is positive when close exceeds prior close, negative when close is below prior close, and zero on a tie; VZO is 100 times the 14-period EMA of signed volume divided by the 14-period EMA of total volume, adjust=false; zero-line direction","selection_use":"published formula, smoothing period, and zero-line direction only; no incidence or outcomes"},
 features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T), including summed base volume and quote_asset_volume","signed_volume":"current four-hour base volume times sign(current completed close-prior completed close) on consecutive valid bars; zero on an exact close tie","volume_zone_oscillator":"100*EMA14(signed volume)/EMA14(base volume), alpha=2/15, adjust=false, minimum 14 consecutive valid transitions; denominator must be finite and strictly positive","cross":"current VZO strictly crosses zero from immediately prior valid VZO","variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
 clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"ema_periods":14,"zero_level":0.,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_variation_gate","quote_volume_vzo","one_bar_stale_cross","direction_flip"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close","volume","quote_asset_volume"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"canonical_vzo_definition_read":True,"repository_vzo_candidate_found":False,"prior_volume_indicator_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical Volume Zone Oscillator zero cross under the requested high-variation regime"},
 stopping_rule="Terminal first failure; no volume field, signed-volume formula, EMA period, zero line, crossing, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}

def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVVZO preregistration drift")

if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
