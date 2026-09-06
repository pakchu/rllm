"""Preregister the outcome-sealed HVRBFR-2 singleton."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_range_boundary_fade_relay_preregistration_2026-08-08.json"
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_range_boundary_fade_relay_v1",
        "policy_id": "HVRBFR-2",
        "as_of_date": "2026-08-08",
        "exact_candidate_outcomes_opened": False,
        "exact_candidate_source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "At a fixed four-hour boundary, a close near an edge of the completed "
                "twelve-hour range during an unusually high realized-range regime is a "
                "short-horizon liquidity overshoot. Fade the edge for two elapsed hours."
            ),
            "long": "completed close location <=0.10",
            "short": "completed close location >=0.90",
            "why_suited_to_volatile_regimes": (
                "the completed twelve-hour high-low log range must rank in the causal "
                "upper 20% of prior twelve-hour observations"
            ),
            "why_distinct": (
                "HVRBFR uses fixed four-hour range-boundary snapshots and a symmetric "
                "two-hour fade. DRCMR used one daily decisive-close momentum clock; DTFAR "
                "used a different daily tail geometry. No prior control is promoted."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "six sparse UTC boundary clocks plus trailing-range edge geometry are not "
                "a Gross9 sleeve clock"
            ),
        },
        "features": {
            "source": "BTCUSDT bars_binance interval=5m",
            "twelve_hour_bar": (
                "144 exact, distinct, finite, positive and coherent completed 5m bars in "
                "[decision-12h, decision); no imputation"
            ),
            "log_range": "log(max(high)/min(low))",
            "close_location": "(last_close-min(low))/(max(high)-min(low))",
            "range_rank": (
                "strict-prior midrank among at most 180 valid fixed four-hour observations, "
                "minimum 120, current excluded"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact 00:00,04:00,08:00,12:00,16:00,20:00 UTC",
            "entry": "exact decision+5m BTCUSDT open",
            "hold": "2 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding_oi_premium_implied_vol": (
                "not signal inputs; exact funding is used only after novelty passes"
            ),
        },
        "policy": {
            "history_observations": 180,
            "minimum_history_observations": 120,
            "range_rank_min": 0.80,
            "lower_close_location_max": 0.10,
            "upper_close_location_min": 0.90,
            "entry_delay_minutes": 5,
            "hold_hours": 2,
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
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per "
                "notional side, every held 5m favorable then adverse, global HWM, "
                "full-calendar CAGR"
            ),
        },
        "source_plan": {
            "btc_5m": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "5m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-06-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "materialize_after_preregistration": True,
            },
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "no_range_edge_gate",
                "one_boundary_stale_geometry",
                "direction_follow",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "related_price_structure_family_outcomes_known": True,
            "exact_symmetric_candidate_event_set_previously_evaluated": False,
            "exact_candidate_incidence_opened": False,
            "exact_candidate_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_terminal_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent symmetric high-volatility boundary-overshoot mechanism"
            ),
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict "
            "economics; no threshold, side, hold, clock, range, or subset repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(args.output)
