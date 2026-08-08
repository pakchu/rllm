"""Outcome-blind preregistration for FARR-6."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path("results/funding_acceleration_reversal_relay_preregistration_2026-08-09.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 core={
  "protocol_version":"funding_acceleration_reversal_relay_v1","policy_id":"FARR-6","as_of_date":"2026-08-09","outcomes_opened":False,"source_incidence_opened":False,"singleton":True,
  "mechanism":{"claim":"The change in BTCUSDT funding from the completed 00:00 UTC settlement to the completed 08:00 UTC settlement measures newly accelerating leverage crowding. During high realized BTC variation, a large funding increase identifies fresh long crowding and maps short BTC, while a large decrease identifies fresh short crowding and maps long BTC for a six-hour unwind.","side":"negative sign of funding_rate_08 minus funding_rate_00","why_distinct":"BTC-specific intraday funding acceleration is not a funding-rate level, polarity, settlement-price gap, OI purge, or funding-harvest payoff. FPMR used 28-day cross-alt funding changes for an alt-pair score; FARR uses one BTC eight-hour change at one daily clock. No prior control or terminal event set is promoted.","why_suited_to_volatile_regimes":"prior-24h BTC realized-variation rank must be at least 0.65","why_low_gross9_overlap_is_plausible":"one daily 08:05 UTC funding-acceleration clock is absent from Gross9 primitives"},
  "features":{"funding_pair":"the unique BTCUSDT funding_rates_binance rows whose funding_time floors to exact 00:00 and 08:00 UTC on the same calendar date; finite funding rates required","funding_change":"funding_rate at 08:00 minus funding_rate at 00:00; strict zero is ineligible","absolute_change_rank":"strict-prior midrank of abs(funding_change) against at most 252 prior valid daily pairs; minimum 126; current excluded; rank>=0.75","btc_variation":"sqrt(sum squared log(close/open)) over exact BTCUSDT 1m bars in [prior 08:00,current 08:00)","btc_variation_rank":"strict-prior midrank against at most 252 prior valid daily variations; minimum 126; current excluded; rank>=0.65","availability":"after the actual 08:00 funding record and completed 07:59 BTC bar; entry strictly later","missing_duplicate_or_late":"ineligible or source failure; no imputation"},
  "clock":{"decision":"each calendar day after completed 08:00 UTC funding record","entry":"exact 08:05 UTC BTCUSDT 5m open, strictly after feature availability","hold":"6 elapsed hours","reservation":"global half-open; exit first on equal open","split_crossing_action":"skip","gross_exposure":.5,"funding_accounting":"all exact settlements are applied to later PnL","no_imputation":True},
  "policy":{"history_observations":252,"minimum_history_observations":126,"absolute_change_rank_min":.75,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":6,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
  "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
  "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.2,"max_month_share":.45},
  "novelty_gates":{"exact_entry_jaccard_max":.1,"candidate_near_6h_share_max":.35,"occupied_5m_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
  "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.1,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"future_can_rank_repair_or_reselect":False,"accounting":"fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
  "source_plan":{"funding":{"table":"funding_rates_binance","symbol":"BTCUSDT","columns":["funding_time","funding_rate"],"mark_price_forbidden_as_signal":True,"read_only":True},"btc":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","close"],"read_only":True},"execution_price":"sealed until source support and Gross9 novelty pass"},
  "diagnostic_controls":{"names":["no_volatility_gate","no_change_tail","funding_level","one_day_stale_change","direction_flip"],"diagnostic_controls_cannot_be_promoted":True},
  "research_boundary":{"database_metadata_only_opened_before_preregistration":True,"funding_or_market_values_used_to_select_rule":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"independent BTC intraday funding-crowding acceleration plus user-required high volatility"},
  "stopping_rule":"terminal first-failure sequence: source support, Gross9 novelty, strict economics; no settlement clock, threshold, side, hold, timing, volatility, or subset repair"
 };return {**core,"manifest_hash":canonical_hash(core)}

def validate(r:dict[str,Any])->None:
 if r.get("manifest_hash")!=canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"}) or r.get("outcomes_opened") is not False or r.get("source_incidence_opened") is not False:raise RuntimeError("FARR preregistration drift")

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();r=build();validate(r);a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
