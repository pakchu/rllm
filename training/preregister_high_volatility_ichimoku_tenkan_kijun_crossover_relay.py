"""Outcome-blind preregistration for HVICHI-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVICHI-24";DEFAULT_OUTPUT=Path("results/high_volatility_ichimoku_tenkan_kijun_crossover_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_ichimoku_tenkan_kijun_crossover_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"During elevated realized variation, a canonical Ichimoku Tenkan/Kijun strict crossover on completed four-hour BTC auctions identifies a reversal in short-versus-medium equilibrium range location; follow the new relation for twenty-four hours.","side":"long when Tenkan crosses strictly above Kijun; short on the strict reverse crossover","why_distinct":"Tenkan and Kijun are 9- and 26-period high-low equilibrium midpoints. HVITK trades only their strict crossover and does not use visible cloud spans, close cloud breakout, prior candidate controls, return thresholds, volume, flow, OI, funding, fitted outcomes, reused events, or promoted controls.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"sparse short-versus-medium range-equilibrium reversals are absent from Gross9 primitives"},
 external_basis={"origin":"Goichi Hosoda canonical Ichimoku Kinko Hyo convention","fixed_definition":"Tenkan is 9-period highest-high/lowest-low midpoint; Kijun is the corresponding 26-period midpoint; strict Tenkan/Kijun crossover direction","selection_use":"published 9/26 periods and crossover direction only; no incidence or outcomes"},
 features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)","tenkan":"midpoint of highest high and lowest low over latest 9 uninterrupted completed bars","kijun":"same midpoint over latest 26 uninterrupted completed bars","event":"current finite nonzero Tenkan-Kijun sign differs from immediately prior finite nonzero sign","variation":"sqrt(sum squared completed one-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
 clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"tenkan_periods":9,"kijun_periods":26,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_variation_gate","persistent_equilibrium_state","one_bar_stale_cross","direction_flip"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"canonical_ichimoku_tenkan_kijun_definition_read":True,"repository_ichimoku_tenkan_kijun_candidate_found":False,"prior_donchian_aroon_moving_average_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical Ichimoku Tenkan/Kijun equilibrium crossover under the requested high-variation regime"},
 stopping_rule="Terminal first failure; no Tenkan or Kijun period, midpoint formula, crossing, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVICHI preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
