"""Outcome-blind preregistration for HVSBCR-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_energy_technology_spillover_relay as template

DEFAULT_OUTPUT = Path("results/high_volatility_stock_bitcoin_coskewness_relay_preregistration_2026-08-11.json")
SPY_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build()); core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_stock_bitcoin_coskewness_relay_v1", policy_id="HVSBCR-24", as_of_date="2026-08-11",
        mechanism={
            "claim": "The completed rolling coskewness of Bitcoin with squared S&P 500 returns measures asymmetric cross-market dependence. Published evidence maps higher coskewness to lower following-day Bitcoin excess returns; in elevated causal BTC variation, trade opposite a large standardized coskewness deviation for twenty-four hours.",
            "side": "negative sign of the strictly-prior standardized 63-session BTC-with-SPY coskewness deviation; zero is ineligible",
            "external_support": {
                "paper": "Chen, Liu, and Zhang (2024), Coskewness and the short-term predictability for Bitcoin return",
                "doi": "10.1016/j.techfore.2023.123196",
                "paper_fixed_facts": [
                    "daily coskewness between Bitcoin and S&P 500 returns forecasts following-day Bitcoin excess returns",
                    "the in-sample coskewness regression coefficient is significantly negative",
                    "rolling and recursive out-of-sample forecasts are significant",
                    "the predictor remains informative alongside volatility, momentum, and illiquidity and produces economic benefits",
                ],
                "implementation_choices_not_claimed_as_replication": [
                    "SPY raw closes as a liquid investable S&P 500 proxy", "a fixed 63-common-session normalized cross third moment",
                    "a 252-session prior-only z-score with 126 minimum observations and absolute threshold 0.75",
                    "a causal BTC prior-24-hour variation-rank gate of 0.65", "24-hour BTC hold starting ten minutes after actual NYSE close",
                ],
            },
            "why_distinct": "Repository scans found no coskewness, co-skewness, systematic-skew, or BTC-SPY cross-third-moment candidate. Existing realized and conditional skew candidates use BTC-only third moments; this candidate uses the cross moment BTC return times squared SPY return and no prior event or control.",
            "why_suited_to_volatile_regimes": "Asymmetric stock-Bitcoin dependence is most relevant when joint tails matter; the frozen policy independently requires prior-24-hour BTC variation rank >=0.65.",
            "why_low_gross9_overlap_is_plausible": "A completed NYSE-close cross-asset third-moment state is absent from Gross9 primitives.",
        },
        features={
            "daily_returns": "for each actual SPY session close, raw SPY close/current prior common-session close and BTC exact close-to-close log return over the same two NYSE closes",
            "coskewness": "over the latest 63 completed common-session pairs, mean((BTC_return-mean_BTC)*(SPY_return-mean_SPY)^2)/(population_sd_BTC*population_variance_SPY); both scales strict positive",
            "coskewness_z": "causal z-score versus at most 252 strictly prior finite coskewness values; minimum 126; population scale; current excluded",
            "event": "absolute coskewness_z >=0.75; side is negative sign(coskewness_z)",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars ending at but excluding actual NYSE close",
            "btc_variation_rank": "strict-prior midrank versus at most 252 prior valid common-session variations; minimum 126; current excluded; rank >=0.65",
            "missing": "any missing/nonfinite/nonpositive source or incomplete BTC grid is ineligible; no imputation",
        },
        clock={"source_session":"actual NYSE regular-session close, including early closes","feature_available":"five minutes after actual NYSE close","entry":"exact BTCUSDT five-minute open ten minutes after actual NYSE close","hold":"24 elapsed hours","reservation":"global half-open; exit first on equal open","split_crossing_action":"skip","gross_exposure":0.5,"funding_oi_premium":"not signal inputs; exact funding only after novelty passes","no_imputation":True},
        policy={"coskewness_sessions":63,"coskewness_z_prior_sessions":252,"coskewness_z_prior_minimum":126,"coskewness_abs_z_min":0.75,"variation_prior_sessions":252,"variation_prior_minimum":126,"variation_midrank_min":0.65,"feature_delay_minutes":5,"entry_delay_minutes_after_feature":5,"hold_hours":24,"gross_exposure":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
        source_plan={"spy":{"url":SPY_YAHOO_URL,"fields":["timestamp","open","high","low","close","volume"],"raw_unadjusted_only":True,"download_after_preregistration":True},"btc_1m":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","close"],"read_only":True},"execution_price":"sealed until source support and Gross9 novelty pass"},
        diagnostic_controls={"names":["no_btc_volatility_gate","raw_coskewness_level","btc_skew_only","one_session_stale_coskewness","direction_flip"],"diagnostic_controls_cannot_be_promoted":True},
        research_boundary={"source_schema_and_transport_checked":False,"source_values_used_to_select_rule":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"repository_stock_bitcoin_coskewness_candidate_found":False,"prior_btc_only_skew_candidates_known":True,"prior_event_sets_reused":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"primary published negative stock-Bitcoin coskewness relation plus exact repository absence"},
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no proxy, return alignment, window, normalization, threshold, side, hold, clock, subset, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {k: v for k, v in value.items() if k != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build(): raise RuntimeError("HVSBCR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False: raise RuntimeError("HVSBCR boundary drift")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    result=build(); validate(result); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(args.output)
