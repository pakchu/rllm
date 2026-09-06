"""Outcome-blind preregistration for QHOIR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/quarter_hour_opening_imbalance_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "quarter_hour_opening_imbalance_relay_v1",
        "policy_id": "QHOIR-8",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "At a UTC quarter-hour boundary during an already high-volatility BTC regime, "
                "the just-completed opening minute's normalized aggressive order imbalance "
                "reveals periodic algorithmic demand whose direction should persist for eight hours."
            ),
            "side": "strict nonzero sign of the quarter-hour opening-minute order imbalance",
            "external_support": {
                "paper": "Kim and Hansen (2026), The Quarter-Hour Effect",
                "arxiv": "2607.09426v2",
                "supported_facts": [
                    "quarter-hour openings concentrate periodic order-flow dependence",
                    "opening imbalance direction positively predicts four-to-twelve-hour returns",
                    "sign-only imbalance and exclusion of funding-settlement boundaries preserve the result",
                ],
                "implementation_is_not_a_replication": True,
                "untested_adaptation": (
                    "the paper's medium-horizon regression uses the first 10 seconds; QHOIR uses "
                    "the full first minute available in the PostgreSQL candle source"
                ),
            },
            "why_distinct": (
                "HVAFC-6 and HVPAR-6 aggregated two hours of flow at fixed eight-hour decisions "
                "and required price-flow confirmation or contradiction. QHOIR reads only the first "
                "minute after each 15-minute boundary, has no price-direction or magnitude gate, and "
                "tests a clock-phase mechanism. HVAFC train failure did not set this rule."
            ),
            "why_suited_to_volatile_regimes": (
                "strictly prior 24-hour BTC realized variation must rank in the causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "quarter-hour phase entries at minutes 05, 20, 35, or 50 are absent as a Gross9 primitive"
            ),
        },
        "clock": {
            "boundary": "every UTC T with minute in {00,15,30,45} and second 00",
            "opening_bar": "exact source-complete BTCUSDT one-minute bar [T,T+1m)",
            "order_imbalance": "(2*taker_buy_base-volume)/volume in [-1,1], volume strictly positive",
            "feature_available": "T+1m after the opening bar completes",
            "entry": "exact BTCUSDT T+5m open, the first five-minute open after availability",
            "side": "sign(order_imbalance); zero cannot signal",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        "volatility_gate": {
            "variation": (
                "sqrt(sum of squared log(close/open)) over 1,440 exact source-valid one-minute "
                "bars in [T-24h,T); the opening bar [T,T+1m) is excluded"
            ),
            "rank": (
                "strict-prior midrank against at most 8,640 prior valid quarter-hour variations "
                "(90 elapsed days), minimum 5,760 (60 days); current excluded; rank>=0.65"
            ),
        },
        "policy": {
            "quarter_hour_minutes": [0, 15, 30, 45],
            "variation_minutes": 1440,
            "variation_prior_observations": 8640,
            "variation_prior_min_observations": 5760,
            "variation_midrank_min": 0.65,
            "imbalance_abs_min": 0.0,
            "feature_delay_minutes": 1,
            "entry_delay_minutes_from_boundary": 5,
            "hold_hours": 8,
            "leverage": 0.5,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "btc_1m": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close", "volume", "taker_buy_base"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:01:00Z"],
                "materialize_after_preregistration": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "shifted_phase_plus_2m",
                "five_minute_phase_only",
                "no_volatility_gate",
                "one_quarter_stale_imbalance",
                "direction_flip",
                "exclude_funding_boundaries",
            ],
            "phase_controls_are_falsification_only": True,
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "paper_read_before_candidate": True,
            "prior_flow_candidate_incidence_and_outcomes_known": True,
            "prior_candidate_outcomes_used_to_set_qhoir": False,
            "qhoir_candidate_incidence_opened": False,
            "qhoir_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published quarter-hour mechanism plus user-required high-volatility regime",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no phase, threshold, side, hold, volatility, or subset repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
