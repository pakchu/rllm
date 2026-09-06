"""Outcome-blind preregistration for HVIPRAPR-8."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Mapping
from training import preregister_high_volatility_cross_structure_action_vote as hvcav

POLICY_ID="HVIPRAPR-8"
DEFAULT_OUTPUT=Path("results/high_volatility_intraday_premium_activity_polarity_router_preregistration_2026-08-16.json")
def canonical_hash(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 c=hvcav.build();core={
  "protocol_version":"high_volatility_intraday_premium_activity_polarity_router_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-16","singleton":True,
  "exploratory_discovery":True,"fresh_confirmatory_evidence":False,"source_incidence_opened":False,"outcomes_opened":False,"gross9_rows_opened":False,
  "candidate_family":[POLICY_ID],"candidate_family_size":1,
  "mechanism":{"claim":"Within each completed Binance eight-hour derivatives cycle, premium-index path activity relative to BTC realized variation separates spot-led continuation from derivatives-heavy overreaction. Follow the completed BTC direction in the lower relative-activity tail and fade it in the upper tail for the next eight hours.","side":"BTC cycle direction in the lower relative-activity tail and its negative in the upper tail","why_distinct":"HVIPRAPR is an intraday eight-hour cross-market efficiency state. It is not the rejected daily HVPRAPR clock, uses no daily session aggregation, funding, OI, spot table, alt, fitted outcome, prior event rows, or promoted control.","why_suited_to_volatile_regimes":"completed eight-hour BTC realized variation must rank at least 0.65 causally","why_low_gross9_overlap_is_plausible":"tail-qualified intraday premium-efficiency states at exact derivatives-cycle boundaries are absent from Gross9 primitives"},
  "features":{"decision_grid":"exact 00:00, 08:00, and 16:00 UTC boundaries D","premium_path":"480 exact unique bars_binance_premium BTCUSDT one-minute closes [D-8h,D), finite signed values","premium_total_variation":"sum(abs(close[i]-close[i-1])) over 479 within-block pairs, strict positive","btc_path":"480 exact coherent bars_binance BTCUSDT one-minute OHLC rows over the same block","btc_return":"log(final close/first open), strict nonzero","btc_realized_variation":"sqrt(sum squared one-minute log(close/open) returns), strict positive","relative_premium_activity":"premium_total_variation/btc_realized_variation, finite strict positive","causal_ranks":"strict-prior midranks of relative activity and BTC variation over at most 270 earlier source-valid cycles, minimum 180; current excluded","relative_activity_tails":"lower rank<=0.25 or upper rank>=0.75; middle ineligible","routing":"lower tail follows btc_return; upper tail fades btc_return","eligible":"source valid, either relative-activity tail, BTC variation rank>=0.65, and strict nonzero BTC return","side":"sign(btc_return) in lower tail and -sign(btc_return) in upper tail","no_imputation":True},
  "clock":{"decision":"completed exact derivatives-cycle boundary D","entry":"exact BTCUSDT perpetual D+5m open","hold":"8 elapsed hours","reservation":"global half-open; exit first on equal entry","split_crossing_action":"skip","gross_exposure":0.5,"funding":"not a signal input; exact held settlements only after source and Gross9 pass"},
  "policy":{"block_minutes":480,"history_cycles":270,"minimum_history_cycles":180,"lower_relative_activity_rank_max":0.25,"upper_relative_activity_rank_min":0.75,"btc_variation_rank_min":0.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.0010},
  "stages":c["stages"],"source_support_gates":c["source_support_gates"],"gross9_novelty_gates":c["gross9_novelty_gates"],"economic_gates":c["economic_gates"],
  "source_plan":{"premium":"Postgres bars_binance_premium BTCUSDT exact 1m closes","btc":"Postgres bars_binance BTCUSDT exact coherent 1m OHLC","window":["2023-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True,"execution_prices":"sealed until source support and Gross9 pass"},
  "research_boundary":{"prior_daily_HVPRAPR_gross9_failure_known":True,"prior_daily_HVPRQC_source_failure_known":True,"same_mechanism_as_daily_candidates":False,"exact_intraday_incidence_or_outcomes_known":False,"prior_outcomes_used_to_set_rank_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"classification":"exploratory discovery; not fresh confirmatory evidence"},
  "stopping_rule":"terminal first failure; no source, block, relative-activity tails, variation rank, side, clock, hold, subset, threshold, comparator, or control repair"}
 return {**core,"manifest_hash":canonical_hash(core)}
def validate(v:Mapping[str,Any])->None:
 if v!=build():raise RuntimeError("HVIPRAPR-8 preregistration drift")
 c=hvcav.build()
 for k in ("stages","source_support_gates","gross9_novelty_gates","economic_gates"):
  if v[k]!=c[k]:raise RuntimeError(f"HVIPRAPR-8 {k} drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
