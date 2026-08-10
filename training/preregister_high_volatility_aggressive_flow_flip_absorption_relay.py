"""Outcome-blind preregistration for HVFFA-8."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVFFA-8";DEFAULT_OUTPUT=Path("results/high_volatility_aggressive_flow_flip_absorption_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_aggressive_flow_flip_absorption_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"Within a volatile completed eight-hour BTC block, if both four-hour price returns persist in one direction while aggressive quote flow changes from supporting that direction in the first half to opposing it in the second half, price has absorbed a newly adverse flow regime. Follow the resilient price direction for eight hours.","side":"common strict sign of the two completed four-hour price returns","why_distinct":"Dual-half flow absorption requires persistent same-sign aggressive flow opposing the full-block return; passive absorption uses only a final two-hour contradiction; flow confirmation requires agreement. HVFFA requires an ordered supporting-to-opposing taker-flow sign transition while both price halves retain one direction, with no magnitude threshold, prior event, fitted outcome, or promoted control.","why_suited_to_volatile_regimes":"the completed eight-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"offset three-daily ordered flow-flip absorption clocks are absent from Gross9 primitives"},
 features={"decision_grid":"exact 02:00/10:00/18:00 UTC boundaries","block":"480 exact coherent BTCUSDT bars_binance one-minute rows [D-8h,D)","half_returns":"log(last close/first open) separately in fixed 240-minute halves, each strict nonzero and sharing one sign","half_taker_imbalance":"(2*sum(taker_buy_quote)-sum(quote_asset_volume))/sum(quote_asset_volume) in each half, each denominator positive and imbalance strict nonzero","ordered_flow_flip":"first-half imbalance sign equals common price sign and second-half imbalance sign is opposite","realized_variation":"sqrt(sum squared one-minute log(close/open) returns), finite strict positive","variation_rank":"strict-prior midrank over at most 270 earlier source-valid blocks, minimum 180, current excluded; rank>=0.65","eligible_state":"price persistence, ordered flow flip, and variation gate pass","no_imputation":True},
 clock={"decision":"completed eight-hour boundary","entry":"exact BTCUSDT D+5m open","hold":"8 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"history_blocks":270,"minimum_history_blocks":180,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_flow_flip_requirement","no_variation_gate","flow_transition_only","one_block_stale_flow","direction_flip","forced_long"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close","quote_asset_volume","taker_buy_quote"],"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"prior_dual_half_passive_and_confirmation_flow_outcomes_known":True,"repository_aggressive_flow_flip_absorption_candidate_found":False,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"ordered supporting-to-opposing aggressive-flow transition under persistent price direction"},
 stopping_rule="Terminal first failure; no half partition, flow definition, sign ordering, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVFFA preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
