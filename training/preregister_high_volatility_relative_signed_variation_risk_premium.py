"""Outcome-blind preregistration for HVRSVRP-12."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVRSVRP-12"
DEFAULT_OUTPUT=Path("results/high_volatility_relative_signed_variation_risk_premium_preregistration_2026-08-11.json")
def canonical_hash(value:Any)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(
  protocol_version="high_volatility_relative_signed_variation_risk_premium_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
  mechanism={"claim":"Cryptocurrency realized signed variation is negatively related to future excess returns. On a completed high-variation BTC day, fade an extreme upper relative signed-variation state and buy an extreme lower state for twelve hours.","side":"SHORT for causal upper-tail relative signed variation and LONG for causal lower-tail relative signed variation","why_distinct":"The prior bipower candidate isolated unsigned jump magnitude and faded the dominant return; realized skew used a standardized third moment. HVRSVRP directly decomposes quadratic variation into positive and negative return semivariances, normalizes their difference by total realized variation, uses a fixed daily clock, and maps the published negative risk-premium relation without reusing an event set or control.","why_suited_to_volatile_regimes":"the same completed-day total realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"one daily two-sided semivariance-tail decision is absent from Gross9 primitives"},
  external_basis={"paper":"https://cirforum.org/cirf2022/forum_files/papers/CIRF-306.pdf","paper_sha256":"1d3bed80a512dc79d248560de414d88d1ab7294d8e3987c848e8231957a18ac3","published_doi":"10.1016/j.irfa.2023.102712","support":"the cryptocurrency study constructs positive and negative realized semivariances, reports economically material signed-jump variation, and finds a negative relation between realized signed jump and future excess returns","selection_use":"fixed relative signed-variation construction and negative direction only; daily rank tails, volatility gate and twelve-hour application are untested; no incidence or outcomes"},
  features={"decision_grid":"every exact 00:00 UTC boundary after the completed prior UTC day","source":"1,440 exact coherent BTCUSDT perpetual one-minute rows over [D-24h,D)","five_minute_returns":"288 exact nonoverlapping groups; log of each group fifth-minute close divided by first-minute open","positive_semivariance":"sum squared strictly positive five-minute returns","negative_semivariance":"sum squared strictly negative five-minute returns","realized_variation":"positive plus negative semivariance, finite strict positive","relative_signed_variation":"(positive_semivariance-negative_semivariance)/realized_variation, finite","signed_variation_rank":"strict-prior midrank over at most 90 earlier source-valid daily decisions, minimum 60, current excluded","two_sided_tail":"rank>=0.80 is upper and rank<=0.20 is lower","variation_rank":"strict-prior 90/60 midrank of realized variation, current excluded; rank>=0.65","eligible":"source valid, signed variation in either frozen tail, and variation rank passes","no_imputation":True},
  clock={"feature_available":"exact daily boundary","entry":"exact BTCUSDT perpetual D+5m open","side":"-1 upper tail, +1 lower tail","hold":"12 elapsed hours","reservation":"daily clocks do not overlap; global half-open, exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
  policy={"five_minute_bars":288,"history_days":90,"minimum_history_days":60,"lower_signed_variation_rank_max":.20,"upper_signed_variation_rank_min":.80,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":12,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
  diagnostic_controls={"names":["no_signed_variation_tail","no_variation_gate","raw_day_return_reversal","one_day_stale_state","direction_flip","forced_long"],"cannot_be_promoted":True},
  source_plan={"perpetual":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"]},"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True,"execution_price":"sealed until source support and Gross9 novelty pass"},
  research_boundary={"crypto_signed_jump_paper_read":True,"repository_exact_relative_signed_variation_candidate_found":False,"prior_jump_skew_semivariance_outcomes_known":True,"prior_event_sets_reused":False,"prior_outcomes_used_to_set_formula_ranks_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_bipower_jump_or_realized_skew":False,"promoted_prior_control":False,"selection_basis":"published cryptocurrency signed-jump risk premium mapped once to a sparse volatility-gated BTC policy"},
  stopping_rule="Terminal first failure; no return sampling, semivariance formula, normalization, rank tails, volatility rank, side, hold, clock, subset, control, or other repair.")
 return {**c,"manifest_hash":canonical_hash(c)}

def validate(v:dict[str,Any])->None:
 core={k:x for k,x in v.items() if k!="manifest_hash"}
 if v.get("manifest_hash")!=canonical_hash(core) or v!=build():raise RuntimeError("HVRSVRP-12 preregistration drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
