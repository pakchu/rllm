"""Outcome-blind preregistration for OSPLR-24."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path("results/ofac_sanctions_pressure_lifecycle_relay_preregistration_2026-08-09.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 core={
  "protocol_version":"ofac_sanctions_pressure_lifecycle_relay_v1","policy_id":"OSPLR-24","as_of_date":"2026-08-09","outcomes_opened":False,"source_incidence_opened":False,"gross9_rows_opened":False,"singleton":True,
  "mechanism":{"claim":"Official OFAC sanctions designations and restrictive list updates tighten global dollar-payment and risk-transfer rails, while explicit removals and general-license publications relax those constraints. During already high BTC variation, tightening maps short and relief maps long for twenty-four elapsed hours.","side":"SHORT for unambiguous pressure; LONG for unambiguous relief","why_distinct":"OSPLR uses archived irregular federal sanctions-action dates, not scheduled macro values, market direction, crypto flow, Gross9 clocks, prior terminal events, or diagnostic controls.","why_suited_to_volatile_regimes":"payment-rail and geopolitical shocks should transmit most strongly when BTC liquidity is already unstable; prior-24h variation rank must be >=0.65"},
  "authority":{"recent_actions":"https://ofac.treasury.gov/recent-actions","sdn_change_archive":"https://ofac.treasury.gov/specially-designated-nationals-list-sdn-list/archive-of-changes-to-the-sdn-list","historical_rule":"use dated Recent Actions and archived change records, never reconstruct history from the mutable active SDN list","availability":"22:00 UTC on the displayed action date because no authoritative intraday time is frozen"},
  "taxonomy":{"normalization":"HTML-unescape, strip tags, Unicode NFKC, lowercase, collapse whitespace, whole-token/exact-phrase matching","pressure_terms":["designation","designations","designating","sanctions","sanctions evasion","sdn list update","specially designated nationals list update"],"relief_terms":["removal","removals","removed from","general license","general licenses"],"classification":"title plus official action summary: pressure XOR relief; both/neither ineligible; same date/side deduplicates; opposite eligible sides invalidate the date"},
  "volatility_gate":{"source":"BTCUSDT bars_binance 1m strictly before availability","variation":"sum squared 1m close-to-close log returns over exact prior 24h","rank":"strict-prior midrank at all valid OFAC action days, max 252, minimum 126, current excluded, >=0.65","no_imputation":True},
  "clock":{"decision":"22:00 UTC on eligible OFAC action date","entry":"decision+5m BTCUSDT open","hold":"24 elapsed hours","reservation":"global half-open; exit first on equal open","split_crossing_action":"skip","gross_exposure":.5,"funding":"exact only after novelty pass"},
  "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
  "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.2,"max_month_share":.45},
  "novelty_gates":{"exact_entry_jaccard_max":.1,"candidate_near_6h_share_max":.35,"occupied_5m_bar_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
  "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.1,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"accounting":"fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
  "diagnostic_controls":{"names":["no_volatility_gate","title_only_taxonomy","one_event_stale_side","direction_flip"],"diagnostic_controls_cannot_be_promoted":True},
  "research_boundary":{"official_documentation_opened":True,"ofac_action_values_or_incidence_opened":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"new archived irregular sanctions lifecycle source"},
  "stopping_rule":"terminal first failure: source integrity/support, Gross9 novelty, sequential economics; no archive, term, ambiguity, time, volatility, side, hold, or subset repair"
 };return {**core,"manifest_hash":canonical_hash(core)}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();r=build();a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print(a.output)
