"""Outcome-blind preregistration for HVIMI-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template
POLICY_ID="HVIMI-24";DEFAULT_OUTPUT=Path("results/high_volatility_intraday_momentum_index_reentry_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_intraday_momentum_index_reentry_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
 mechanism={"claim":"During elevated realized variation, canonical fourteen-period Intraday Momentum Index returning inside its 30/70 envelope on completed four-hour BTC auctions marks one-sided candle-body pressure beginning to normalize; follow the inward direction for twenty-four hours.","side":"long when IMI crosses strictly above 30 after being at or below 30; short when it crosses strictly below 70 after being at or above 70","why_distinct":"IMI sums positive and negative completed open-to-close candle bodies over fourteen bars and forms their bounded ratio. It is not close-to-close RSI, price range location, DeMarker high/low expansion, prior control, OI, flow, volume, funding, fitted outcome, reused event set, or promoted control.","why_suited_to_volatile_regimes":"completed trailing twenty-four-hour realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"sparse four-hour candle-body pressure re-entry clocks are absent from Gross9 primitives"},
 external_basis={"origin":"Tushar Chande and Stanley Kroll Intraday Momentum Index convention","fixed_definition":"up body=max(close-open,0), down body=max(open-close,0), 14-period sums, IMI=100*up_sum/(up_sum+down_sum), conventional 30/70 envelope re-entry","selection_use":"published period, formula, levels, and inward direction only; no incidence or outcomes"},
 features={"decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)","up_down_body":"max(completed close-open,0) and max(open-completed close,0)","imi":"100*rolling 14 up-body sum/(up-body plus down-body sums), invalid at zero denominator","reentry":"current IMI strictly crosses inward through 30 or 70 from immediately prior valid IMI","variation":"sqrt(sum squared completed one-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
 clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
 policy={"imi_periods":14,"lower_level":30.0,"upper_level":70.0,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 diagnostic_controls={"names":["no_variation_gate","outward_crossing","one_bar_stale_reentry","direction_flip"],"cannot_be_promoted":True},
 source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
 research_boundary={"canonical_intraday_momentum_index_definition_read":True,"repository_intraday_momentum_index_candidate_found":False,"prior_body_oscillator_rsi_demarker_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical candle-body pressure ratio re-entry under the requested high-variation regime"},
 stopping_rule="Terminal first failure; no IMI period, formula, level, crossing, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVIMI preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
