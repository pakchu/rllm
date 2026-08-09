"""Outcome-blind preregistration for HVTCTR-24."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVTCTR-24"
DEFAULT_OUTPUT = Path("results/high_volatility_treasury_curve_twist_relay_preregistration_2026-08-09.json")
SOURCE = Path("data/treasury_parallel_yield_shock_relay_sources_2023_2026/treasury_2y_10y_observations.csv.gz")
SOURCE_SHA = "85260826f66345c33c5e590df1c9c0a5853ddf46f21cb4377307ba3ca6f58ccd"
SOURCE_MANIFEST = Path("data/treasury_parallel_yield_shock_relay_sources_2023_2026/manifest.json")
SOURCE_MANIFEST_SHA = "fb6db916c42724cc3b74e8aef262d422c651537fd9d835f379b233ce2c80b9fd"
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"

def canonical_hash(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = {
        "protocol_version":"high_volatility_treasury_curve_twist_relay_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-09","singleton":True,"outcomes_opened":False,"source_incidence_opened":False,"gross9_rows_opened":False,
        "mechanism":{"claim":"An opposite-direction move in official two-year and ten-year Treasury par yields is a curve twist. A falling front end with a rising long end is easing/growth steepening and maps long BTC; a rising front end with a falling long end is front-end tightening and maps short BTC during elevated BTC variation.","side":"strict sign(delta10y-delta2y) when the two changes have opposite strict signs","why_distinct":"TPYSR requires same-sign parallel shocks and never opened economics; HVTCTR uses the disjoint opposite-sign twist state, no TPYSR control, and an extra conservative midday embargo.","why_suited_to_volatile_regimes":"completed seven-day BTC variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"a sparse next-day 12:05 UTC sovereign-curve twist clock is absent from Gross9"},
        "features":{"source":"official US Treasury Daily Treasury Par Yield Curve Rates XML snapshot already hash-bound without outcomes","transition":"consecutive valid Treasury observations one to five calendar days apart","twist":"delta2 and delta10 strict nonzero with opposite signs","direction":"sign(delta10-delta2)","availability":"source day D plus one calendar day at 12:00 UTC, twelve hours later than the source manifest's conservative floor","btc_variation":"sqrt(sum squared exact completed 5m BTC log returns over seven elapsed days ending at decision)","variation_rank":"strict-prior midrank over all daily 12:00 UTC market states, at most 270 and minimum 180, current excluded; rank>=0.65","no_imputation":True},
        "clock":{"decision":"D+1 12:00 UTC","entry":"decision+5m exact BTCUSDT open","hold":"24 elapsed hours","reservation":"source-time global half-open nonoverlap; exit first on equal open","split_crossing_action":"skip","gross_exposure":.5,"funding":"not signal input; exact settlements only after novelty"},
        "policy":{"variation_history_days":270,"minimum_history_days":180,"variation_rank_min":.65,"availability_delay_hours":36,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
        "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
        "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.2,"max_month_share":.45},
        "novelty_gates":{"exact_entry_jaccard_max":.1,"candidate_near_6h_share_max":.35,"occupied_5m_bar_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
        "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.1,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"accounting":"fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit":{"prerequisite":"unchanged candidate passes all stages","rv20_q90_entry_filter":False,"minimum_q90_trades":8,"candidate_q90_absolute_return_positive":True,"identical_clock_forced_long_residual_positive":True},
        "diagnostic_controls":{"definitions":{"no_variation_gate":"all valid curve twists","two_year_only":"negative sign of 2y change","ten_year_only":"positive sign of 10y change","one_observation_stale_twist":"prior twist with current variation","direction_flip":"negative primary side","same_clock_forced_long":"side +1 on primary clock"},"cannot_be_promoted":True},
        "source_plan":{"treasury":{"path":str(SOURCE),"sha256":SOURCE_SHA,"manifest":str(SOURCE_MANIFEST),"manifest_sha256":SOURCE_MANIFEST_SHA},"historical_market":{"path":str(MARKET),"sha256":MARKET_SHA},"live_extension":"read-only Postgres BTCUSDT 1m through 2026-08-01","execution_prices":"sealed until source support and novelty pass"},
        "research_boundary":{"prior_tpysr_source_support_known":True,"prior_tpysr_or_control_outcomes_known":False,"prior_event_sets_or_controls_reused":False,"exact_hvtctr_incidence_or_outcomes_known":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"disjoint official sovereign curve-twist mechanism"},
        "stopping_rule":"terminal first failure; no source, twist, side, variation, delay, clock, hold, subset, threshold, comparator, or control repair"
    }
    return {**core,"manifest_hash":canonical_hash(core)}

def validate(x: dict[str, Any]) -> None:
    if x["manifest_hash"] != canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"}): raise RuntimeError("HVTCTR prereg drift")
    for p,h in ((SOURCE,SOURCE_SHA),(SOURCE_MANIFEST,SOURCE_MANIFEST_SHA),(MARKET,MARKET_SHA)):
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise RuntimeError(f"HVTCTR source drift: {p}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();r=build();validate(r);a.output.write_text(json.dumps(r,indent=2)+"\n");print(a.output)
