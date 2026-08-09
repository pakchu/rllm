"""Outcome-blind preregistration for HVLIR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_liquidity_impact_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_liquidity_impact_relay_v1",
        "policy_id": "HVLIR-8",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "During an already volatile BTC regime, an hourly directional move that consumes "
                "unusually little quote turnover per unit of absolute return identifies a depleted "
                "opposing book rather than broad two-sided participation. The liquidity-vacuum "
                "impulse should continue for eight elapsed hours."
            ),
            "side": "strict sign of the completed one-hour open-to-close log return",
            "why_distinct": (
                "HVLIR uses a causal price-impact ratio: absolute hourly return divided by hourly "
                "quote turnover normalized by its own trailing median. Existing high-volatility "
                "candidates use raw return, range, variance, wick, VWAP, ticket, aggressive-flow, "
                "or volume-location geometry; no retired event set or diagnostic control is reused."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed prior-24-hour realized variation must be in its causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "entries are sparse endogenous impact-rank crossings on hourly boundaries, not a "
                "fixed funding, macro, FX, or Gross9 structural clock"
            ),
        },
        "features": {
            "decision": "each exact UTC hour after 60 completed one-minute bars [D-1h,D)",
            "hour_return": "log(close[D-1m]/open[D-1h]); strict nonzero",
            "hour_quote_turnover": "sum quote_asset_volume over the exact 60-row hour; positive",
            "turnover_baseline": (
                "median hourly quote turnover over the 168 complete hours strictly before D-1h; "
                "minimum 120; current hour excluded"
            ),
            "normalized_impact": (
                "abs(hour_return)/(hour_quote_turnover/turnover_baseline); finite and positive"
            ),
            "impact_rank": (
                "strict-prior midrank of normalized impact among at most 2160 prior valid hours; "
                "minimum 1440; current excluded"
            ),
            "prior_24h_variation": (
                "sum squared one-minute close-to-close log returns over [D-24h,D); exact rows"
            ),
            "variation_rank": (
                "strict-prior midrank among at most 2160 prior valid hourly observations; minimum "
                "1440; current excluded"
            ),
            "eligibility": (
                "impact_rank>=0.80 and variation_rank>=0.65 and immediately previous valid hourly "
                "impact_rank<0.80; no rank, volume, or direction imputation"
            ),
        },
        "clock": {
            "entry": "exact D+5m BTCUSDT open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; chronological events; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "materialize_after_preregistration": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "raw_absolute_return_rank",
                "one_hour_stale_impact",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_high_volatility_candidate_outcomes_known": True,
            "exact_normalized_impact_candidate_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent liquidity-vacuum price-impact mechanism",
        },
        "stopping_rule": (
            "terminal first failure: source support, Gross9 novelty, then sequential economics; "
            "no threshold, rank, baseline, side, hold, clock, or subset repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
