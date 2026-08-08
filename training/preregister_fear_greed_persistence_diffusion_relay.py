"""Outcome-blind preregistration for FGPDR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/fear_greed_persistence_diffusion_relay_preregistration_2026-08-09.json"
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "fear_greed_persistence_diffusion_relay_v1",
        "policy_id": "FGPDR-24",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Three exact consecutive same-sign daily changes in the lagged-published "
                "Crypto Fear & Greed index identify a persistent behavioral repricing rather "
                "than a one-day sentiment print. When the cumulative change is unusually large "
                "and completed BTC variation is high, the behavioral direction should diffuse "
                "into BTC for one further day."
            ),
            "side": "strict common sign of the three consecutive sentiment changes",
            "why_distinct": (
                "FGER used static index-level extremes and contrarian sides. FGPLR required a "
                "one-day sentiment-versus-price sign disagreement and followed price. FGPDR "
                "uses neither index level nor price direction: its economic object is monotone "
                "three-day behavioral persistence, and it follows that persistence."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed pre-entry BTC daily realized variation must rank in its causal "
                "upper 35%."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "a sparse daily behavioral-persistence clock is absent from Gross9 primitives."
            ),
        },
        "features": {
            "sentiment_source": (
                "hash-bound Alternative.me Crypto Fear & Greed daily history already "
                "materialized outcome-blind"
            ),
            "availability": (
                "index row labeled UTC date D is not usable until D+1 00:00 UTC; D through "
                "D-3 must be exact consecutive dates"
            ),
            "daily_changes": "delta_j=value[D-j]-value[D-j-1] for j in {0,1,2}",
            "persistence": (
                "all three daily changes are strict nonzero and have one identical sign"
            ),
            "cumulative_change": "value[D]-value[D-3], necessarily strict nonzero",
            "persistence_magnitude_rank": (
                "strict-prior midrank of abs(cumulative_change) over at most 180 prior valid "
                "persistent episodes, minimum 90, current excluded; rank>=0.60"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared exact 24 completed hourly BTC log returns for date D)"
            ),
            "volatility_rank": (
                "strict-prior midrank over at most 180 valid daily decisions, minimum 90, "
                "current excluded; rank>=0.65"
            ),
            "btc_direction_forbidden": True,
            "no_imputation": True,
        },
        "clock": {
            "decision": (
                "exact D+1 00:00 UTC after the conservative index availability rule and BTC "
                "day D are complete"
            ),
            "entry": "exact BTCUSDT D+1 00:05 UTC open",
            "side": "strict common sign of the three sentiment changes",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding_oi_premium": (
                "not signal inputs; exact funding only after novelty passes"
            ),
        },
        "policy": {
            "persistence_days": 3,
            "sentiment_prior_episodes": 180,
            "sentiment_prior_min_episodes": 90,
            "persistence_magnitude_rank_min": 0.60,
            "variation_prior_days": 180,
            "variation_prior_min_days": 90,
            "realized_variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
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
            "occupied_5m_bar_jaccard_max": 0.25,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "sentiment": {
                "path": (
                    "data/fear_greed_extremity_reversal_sources_2023_2026/"
                    "fear_greed_daily.csv.gz"
                ),
                "sha256": (
                    "a50769db6ca15b9cbb538b4f03fd71956a42a3ca418a7628d8ba0c63d0b8f1dd"
                ),
            },
            "completed_btc": (
                "hash-bound completed-hour BTC source through 2026-08-01"
            ),
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "no_persistence_magnitude_tail",
                "two_day_persistence",
                "one_day_stale_persistence",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_sentiment_family_outcomes_known": True,
            "known_terminal_candidates": ["FGER-24", "FGPLR-24"],
            "prior_sentiment_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_fgpdr_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent monotone behavioral-persistence diffusion mechanism"
            ),
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no persistence length, rank, side, hold, clock, volatility, or subset repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(args.output)
