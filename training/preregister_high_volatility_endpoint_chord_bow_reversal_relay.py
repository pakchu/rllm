"""Outcome-blind preregistration for HVECBR-8."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVECBR-8";DEFAULT_OUTPUT=Path("results/high_volatility_endpoint_chord_bow_reversal_relay_preregistration_2026-08-10.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_endpoint_chord_bow_reversal_relay_v1",policy_id=POLICY_ID,
 mechanism={"claim":"A volatile completed BTC path that remains systematically above or below the straight chord joining its completed endpoints represents an early overshoot followed by incomplete adjustment. Trade opposite the signed bow for eight hours.","side":"negative strict sign of normalized integrated endpoint-chord residual","why_distinct":"Trend fit uses unconstrained OLS slope and R-squared; signed path hysteresis integrates price against realized variation; path efficiency uses endpoint displacement divided by travel; median-shift compares return halves. HVECBR integrates deviation from the deterministic straight chord fixed by the same completed path endpoints and uses no endpoint-direction gate, median, half split, volume, flow, funding, OI, cross-asset, prior event, fitted outcome, or control.","why_suited_to_volatile_regimes":"completed chord-path variation must rank in its causal upper 35% while absolute normalized bow enters its upper 30%","why_low_gross9_overlap_is_plausible":"offset three-daily endpoint-chord bow onsets are absent from Gross9 primitives"},
 features={"decision_grid":"exact 05:00/13:00/21:00 UTC boundaries","block":"480 exact coherent bars_binance BTCUSDT one-minute rows [D-8h,D)","path_points":"p0=log(first open), p(k+1)=log(close_k) for k=0..479","endpoint_chord":"L_i=p0+i*(p480-p0)/480 for interior i=1..479","chord_residual":"e_i=p_i-L_i","realized_variation":"sqrt(sum((p_i-p_(i-1))^2) for i=1..480), finite strict positive","normalized_integrated_bow":"mean(e_i for i=1..479)/realized_variation, finite strict nonzero","bow_rank":"strict-prior midrank of abs(normalized_integrated_bow) over at most 270 earlier source-valid blocks, minimum 180, current excluded; rank>=0.70","variation_rank":"strict-prior 270/180 midrank, current excluded; rank>=0.65","eligible_state":"bow and variation gates pass","onset":"eligible now and immediately previous exact source-valid block ineligible; missing prior cannot trigger","no_imputation":True},
 clock={"decision":"completed eight-hour boundary","entry":"exact BTCUSDT D+5m open","hold":"8 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"history_blocks":270,"minimum_history_blocks":180,"bow_rank_min":.70,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_bow_tail","no_variation_gate","one_block_stale_geometry","direction_flip","forced_long"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"prior_trend_fit_hysteresis_efficiency_and_median_shift_outcomes_known":True,"repository_endpoint_chord_bow_candidate_found":False,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"integrated deviation from the completed endpoint chord"},
 stopping_rule="Terminal first failure; no chord, normalization, rank, onset, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVECBR preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
