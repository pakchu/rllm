"""Outcome-blind preregistration for HVCDBR-8."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVCDBR-8";DEFAULT_OUTPUT=Path("results/high_volatility_closing_drive_breakout_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_closing_drive_breakout_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"When a completed seven-hour BTC auction establishes a range and the final one-hour auction opens strictly inside that range but closes beyond exactly one boundary with an agreeing body, late initiative flow has converted balance into price discovery. Follow the closing-drive breakout for eight hours during elevated realized variation.","side":"long for an upper closing-drive breakout and short for a lower breakout","why_distinct":"Break-retest requires a four-hour range, separate two-hour break, and two-hour retest; inside-resolution requires a separate contained phase; chop-resolution uses path efficiency. HVCDBR uses one seven-hour balance range and a terminal one-hour inside-open/outside-close drive, with no retest, volume, flow, funding, OI, cross-asset, fitted outcome, prior event, or control.","why_suited_to_volatile_regimes":"the full completed eight-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"three daily terminal closing-drive clocks are absent from Gross9 primitives"},
 features={"decision_grid":"exact 02:00/10:00/18:00 UTC boundaries","block":"480 exact coherent BTCUSDT one-minute rows [D-8h,D)","balance_range":"first 420 rows; high=max high and low=min low","closing_drive":"final 60 rows","inside_open":"closing-drive first open strictly between balance low and high","long_breakout":"inside open, final close>balance high, and final close>drive open","short_breakout":"inside open, final close<balance low, and final close<drive open","ambiguity":"exactly one breakout side must hold","realized_variation":"sqrt(sum squared one-minute log(close/open) returns across [D-8h,D)), finite strict positive","variation_rank":"strict-prior midrank over at most 270 earlier source-valid blocks, minimum 180, current excluded; rank>=0.65","eligible_state":"accepted closing drive and variation gate pass","no_imputation":True},
 clock={"decision":"completed closing-drive boundary","entry":"exact BTCUSDT D+5m open","hold":"8 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"history_blocks":270,"minimum_history_blocks":180,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_inside_open","no_variation_gate","drive_body_only","one_block_stale_balance_range","direction_flip","forced_long"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"prior_break_retest_inside_resolution_chop_and_donchian_outcomes_known":True,"repository_closing_drive_breakout_candidate_found":False,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"seven-hour balance followed by a one-hour inside-open/outside-close drive"},
 stopping_rule="Terminal first failure; no balance range, drive, inside-open, body, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVCDBR preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
