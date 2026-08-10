"""Outcome-blind preregistration for HVWRBC-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVWRBC-24";DEFAULT_OUTPUT=Path("results/high_volatility_wide_range_body_continuation_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_wide_range_body_continuation_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"A completed eight-hour BTC auction whose high-low range exceeds each of the three immediately preceding eight-hour auction ranges and whose directional body occupies at least half of that range is a wide-range initiative auction rather than two-way noise. Follow its body direction for twenty-four hours when realized variation is elevated.","side":"strict sign of the completed wide-range auction body","why_distinct":"Adjacent range migration compares two one-hour interval sets; slow Donchian requires a close outside a thirty-block price envelope; return acceleration compares directional returns. HVWRBC compares only current versus three prior range widths and requires a large contemporaneous body, without a channel breakout, close-location gate, volume, flow, funding, OI, cross-asset, fitted outcome, prior event, or control.","why_suited_to_volatile_regimes":"the completed eight-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"three daily wide-range body clocks with twenty-four-hour reservation are absent from Gross9 primitives"},
 features={"decision_grid":"exact 02:00/10:00/18:00 UTC boundaries","blocks":"current and three immediately preceding exact coherent 480-row BTCUSDT one-minute blocks","range":"block maximum high minus minimum low, finite strict positive","wide_range":"current range strictly exceeds all three prior ranges","body":"current last close minus first open, strict nonzero","body_efficiency":"absolute body divided by current range, requiring >=0.50","realized_variation":"sqrt(sum squared current-block one-minute log(close/open) returns), finite strict positive","variation_rank":"strict-prior midrank over at most 270 earlier source-valid current blocks, minimum 180, current excluded; rank>=0.65","eligible_state":"wide range, body efficiency, and variation gate pass","no_imputation":True},
 clock={"decision":"completed current block boundary","entry":"exact BTCUSDT D+5m open","hold":"24 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"prior_range_blocks":3,"history_blocks":270,"minimum_history_blocks":180,"body_efficiency_min":.5,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_wide_range_requirement","no_variation_gate","body_direction_only","one_block_stale_prior_ranges","direction_flip","forced_long"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-03-31T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"prior_adjacent_range_migration_slow_donchian_return_acceleration_and_body_outcomes_known":True,"repository_wide_range_body_continuation_candidate_found":False,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"classic wide-range directional auction using range-width expansion and body efficiency"},
 stopping_rule="Terminal first failure; no prior-range count, range comparison, body efficiency, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVWRBC preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
