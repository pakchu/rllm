"""Outcome-blind preregistration for HVFAAR-8."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVFAAR-8"
DEFAULT_OUTPUT=Path("results/high_volatility_flat_auction_absorption_relay_preregistration_2026-08-10.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 core=copy.deepcopy(template.build());core.pop("manifest_hash")
 core.update(
  protocol_version="high_volatility_flat_auction_absorption_relay_v1",policy_id=POLICY_ID,
  mechanism={"claim":"High quote turnover on exact zero-body minutes represents inventory absorption without completed directional price concession. When that turnover share enters its upper tail during elevated BTC variation and the full block agrees with the final two hours, follow the accepted direction for eight hours.","side":"strict sign of the completed final-two-hour return, equal to the full-block return sign","why_distinct":"HVFAAR conditions on exact price-stasis minutes and measures their share of block quote turnover. Small-ticket, count-memory, directional-ticket, VWAP, wick, range-turnover, and passive-flow candidates use different objects; none measures turnover completed with zero candle body.","why_suited_to_volatile_regimes":"elevated path variation combined with high turnover at unchanged one-minute opens and closes isolates absorption inside an otherwise volatile auction","why_low_gross9_overlap_is_plausible":"offset three-daily stasis-turnover onsets are absent from Gross9 primitives"},
  features={"decision_grid":"exact 03:00/11:00/19:00 UTC boundaries","block":"480 exact coherent bars_binance BTCUSDT one-minute rows [D-8h,D)","flat_minute":"exact close==open on the stored source values","flat_support":"at least five flat minutes and strict-positive total block quote turnover","flat_turnover_share":"sum quote_asset_volume on flat minutes divided by total block quote_asset_volume; finite in [0,1] and strict positive","flat_share_rank":"strict-prior midrank over at most 270 earlier source-valid blocks, minimum 180, current excluded; rank>=0.75","realized_variation":"sum squared one-minute log(close/open) returns, finite strict positive","variation_rank":"strict-prior 270/180 midrank, current excluded; rank>=0.65","block_return":"log(last close/first open), finite strict nonzero","final_two_hour_return":"log(last close/first open) over final 120 completed minutes, finite strict nonzero and same sign as block_return","eligible_state":"flat-share and variation gates pass with directional agreement","onset":"eligible now and immediately previous exact source-valid block ineligible; missing prior cannot trigger","no_imputation":True},
  clock={"decision":"completed eight-hour boundary","entry":"exact BTCUSDT D+5m open","hold":"8 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":0.5,"funding":"not a signal input; exact settlements only after novelty passes"},
  policy={"history_blocks":270,"minimum_history_blocks":180,"minimum_flat_minutes":5,"flat_share_rank_min":0.75,"variation_rank_min":0.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
  diagnostic_controls={"names":["no_flat_share_tail","no_variation_gate","flat_minute_count_share","one_boundary_stale_geometry","direction_flip"],"cannot_be_promoted":True},
  source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close","quote_asset_volume"],"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
  research_boundary={"prior_stasis_turnover_and_related_candidate_outcomes_known":True,"repository_flat_auction_turnover_share_candidate_found":False,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"turnover-bearing exact price stasis as absorption"},
  stopping_rule="Terminal first failure; no flat definition, support, share, rank, side, hold, clock, subset, or control repair.")
 return {**core,"manifest_hash":canonical_hash(core)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVFAAR preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=a.parse_args();x=build();validate(x);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(args.output)
