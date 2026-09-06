"""Outcome-blind preregistration for HVETSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_energy_technology_spillover_relay_preregistration_2026-08-11.json"
)
XLE_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/XLE"
XLK_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/XLK"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_energy_technology_spillover_relay_v1",
        "policy_id": "HVETSR-12",
        "as_of_date": "2026-08-11",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Same-session returns shared by US energy and technology equities carry cross-market "
                "information that Bitcoin incorporates with delay. Trade the equal-weight factor sign for "
                "twelve hours after the completed New York cash session only when BTC is already in a high "
                "realized-variation regime."
            ),
            "side": "sign of the frozen equal-weight XLE/XLK same-session intraday return factor; zero is ineligible",
            "external_support": {
                "paper": (
                    "Symitsi and Chalvatzis (2018), Return, volatility and shock spillovers of Bitcoin "
                    "with energy and technology companies"
                ),
                "doi": "10.1016/j.econlet.2018.06.012",
                "paper_fixed_facts": [
                    "the paper studies return, volatility, and shock spillovers between Bitcoin and energy and technology companies",
                    "it reports return spillovers from energy and technology stock indices to Bitcoin",
                    "it reports a short-term volatility spillover from technology companies to Bitcoin",
                ],
                "implementation_choices_not_claimed_as_replication": [
                    "XLE and XLK as liquid investable sector proxies",
                    "equal-weight same-session raw open-to-close log-return factor",
                    "absolute causal z-score threshold of 1.0",
                    "causal BTC realized-variation rank gate of 0.65",
                    "12-hour BTC execution hold beginning ten minutes after the actual NYSE close",
                ],
            },
            "why_distinct": (
                "Repository collision scans found no XLE/XLK energy-technology spillover preregistration. "
                "Existing US-equity relays use rotation, risk parity, defensive disagreement, volatility-index "
                "spreads, or BTC-equity correlation/residual mechanisms rather than a published joint sector "
                "return-spillover channel."
            ),
            "why_suited_to_volatile_regimes": (
                "The published mechanism includes volatility and shock transmission; the frozen policy further "
                "requires the causal prior-24-hour BTC realized-variation rank to be at least 0.65."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "A completed New York cash-session energy/technology factor and its sector spillover clock are "
                "absent from Gross9 primitives."
            ),
        },
        "features": {
            "xle_intraday_return": "log(raw unadjusted XLE close / open) on the completed source session",
            "xlk_intraday_return": "log(raw unadjusted XLK close / open) on the same completed source session",
            "factor": "0.5 * (xle_intraday_return + xlk_intraday_return)",
            "factor_z": (
                "causal z-score against exactly the previous at most 252 finite common-session factors; "
                "minimum 126; current excluded; population standard deviation; zero variance is ineligible"
            ),
            "factor_event": "absolute factor_z >= 1.0; side is sign(factor), not sign(factor_z)",
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars ending at but excluding "
                "the actual NYSE close"
            ),
            "btc_variation_rank": (
                "strict-prior midrank against at most 252 previous valid common-session BTC variations; "
                "minimum 126; current excluded; rank >= 0.65"
            ),
            "common_rows": (
                "exact common XLE/XLK NYSE session dates with finite positive raw opens and closes and a complete "
                "BTC 1m variation window; missing rows are ineligible and never imputed"
            ),
        },
        "clock": {
            "source_session": "actual NYSE regular-session close for each common XLE/XLK session, including early closes",
            "feature_available": "five minutes after the completed actual NYSE close",
            "entry": "exact BTCUSDT five-minute open ten minutes after the actual NYSE close",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        "policy": {
            "factor_weights": {"XLE": 0.5, "XLK": 0.5},
            "factor_z_prior_sessions": 252,
            "factor_z_prior_minimum": 126,
            "factor_abs_z_min": 1.0,
            "variation_prior_sessions": 252,
            "variation_prior_minimum": 126,
            "variation_midrank_min": 0.65,
            "feature_delay_minutes": 5,
            "entry_delay_minutes_after_feature": 5,
            "hold_hours": 12,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held "
                "5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "equities": {
                "urls": {"XLE": XLE_YAHOO_URL, "XLK": XLK_YAHOO_URL},
                "fields": ["timestamp", "open", "high", "low", "close", "volume"],
                "raw_unadjusted_only": True,
                "download_after_preregistration": True,
            },
            "btc_1m": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": ["no_btc_volatility_gate", "xle_only", "xlk_only", "one_session_stale_factor", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "post_stage_volatility_audit": {
            "rv20_q90_entry_filter": False,
            "opened_only_after_all_train_test_eval_final_economic_gates_pass": True,
            "cannot_repair_or_promote": True,
        },
        "research_boundary": {
            "source_schema_and_transport_checked": False,
            "source_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_xle_xlk_spillover_candidate_found": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "primary published sector-to-Bitcoin spillover evidence plus exact repository absence",
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90 audit; no predictor, lag, window, threshold, side, hold, clock, subset, factor weighting, "
            "or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVETSR preregistration hash drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVETSR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
