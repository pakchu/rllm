"""Outcome-blind preregistration for HVPSY-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVPSY-24";DEFAULT_OUTPUT=Path("results/high_volatility_psychological_line_reentry_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_psychological_line_reentry_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"During elevated realized variation, a canonical twelve-period Psychological Line returning inside its 25/75 envelope on completed four-hour BTC auctions identifies one-sided sign breadth beginning to normalize; follow the inward re-entry direction for twenty-four hours.","side":"long when PSY crosses strictly above 25 after being at or below 25; short when PSY crosses strictly below 75 after being at or above 75","why_distinct":"HVPSY counts only the fraction of strict up-closes in a fixed window. It ignores return magnitude, high-low range, volume, flow, OI, funding, premium, fitted outcomes, and Gross9 state primitives.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%, focusing sign-breadth normalization on July-like volatile auctions","why_low_gross9_overlap_is_plausible":"sparse four-hour binary close-sign breadth re-entry clocks are absent from Gross9 primitives"},
 external_basis={"origin":"canonical Psychological Line (PSY) convention","fixed_definition":"100 times the number of strict up-closes among 12 completed close-to-close transitions divided by 12; ties are zero; 25/75 extreme envelope; inward re-entry","selection_use":"published period, thresholds, and re-entry interpretation only; no incidence or outcomes"},
 features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)","up_indicator":"1 when current completed close is strictly above prior completed close, else 0 including exact ties, on consecutive valid bars","psychological_line":"100*sum of latest 12 up indicators/12 with all 12 transitions valid","reentry":"strict inward crossing above 25 or below 75 from immediately prior valid PSY","variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
 clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"psy_periods":12,"lower_level":25.,"upper_level":75.,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_variation_gate","outward_break","one_bar_stale_reentry","direction_flip"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"canonical_psychological_line_definition_read":True,"repository_psychological_line_candidate_found":False,"prior_sign_breadth_and_oscillator_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical Psychological Line inward re-entry under the requested high-variation regime"},
 stopping_rule="Terminal first failure; no PSY period, tie rule, threshold, re-entry, variation, side, hold, clock, subset, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVPSY preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
