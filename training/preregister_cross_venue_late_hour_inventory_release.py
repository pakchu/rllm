"""Outcome-blind preregistration for CVLIR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/cross_venue_late_hour_inventory_release_preregistration_2026-08-08.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "cross_venue_late_hour_inventory_release_v1",
        "policy_id": "CVLIR-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "mechanism": {
            "claim": (
                "opposite BVOL and DVOL repricing with a quiet first half, extreme absolute "
                "OI repricing, and a late-hour BTC breakout identifies delayed inventory "
                "release whose second-half direction should persist for six hours"
            ),
            "side": "sign of the completed-hour second-half BTC return",
            "why_distinct": (
                "CVLIR is a quiet-to-breakout phase transition conditioned on absolute OI "
                "repricing; it neither requires CVDAR absorption nor uses CVDR's moderate "
                "full-hour displacement"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "the clock requires cross-venue option-vol disagreement and a within-hour "
                "quiet-to-tail transition absent from the frozen REX, kimchi, Rank7, and "
                "Markov sleeve clocks"
            ),
        },
        "clock": {
            "decision": "T after a completed UTC hour",
            "volatility": "normalized BVOL and DVOL bodies have strictly opposite nonzero signs",
            "price_source": "60 exact BTCUSDT 1m bars in [T-1h,T), split at exact minute 30",
            "quiet_first_half": (
                "absolute first-half return <= strictly-prior 720h q40 with 672 observations"
            ),
            "late_breakout": (
                "nonzero absolute second-half return >= strictly-prior 720h q90 with 672 observations"
            ),
            "oi": (
                "raw-time backward-asof observations at T and T-60m, each age<=5m; absolute "
                "one-hour change >= strictly-prior 720h q75 with 672 observations"
            ),
            "funding": "not a signal input; opened only for later exact PnL accounting",
            "trigger": "false-to-true onset, prior hour source-valid and consecutive",
            "entry": "exact BTCUSDT T+5m open, all features available by T",
            "side": "sign of second-half return",
            "hold": "6 elapsed hours",
            "reservation": "global half-open, exit first on equal open",
            "no_imputation": True,
        },
        "policy": {
            "prior_hours": 720,
            "prior_min_hours": 672,
            "first_half_absolute_return_quantile": 0.40,
            "second_half_absolute_return_quantile": 0.90,
            "absolute_oi_change_quantile": 0.75,
            "oi_asof_max_age_minutes": 5,
            "entry_delay_minutes": 5,
            "hold_hours": 6,
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
            "candidate_near_6h_share_max": 0.45,
            "occupied_5m_jaccard_max": 0.30,
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
            "vol_oi_funding": "reuse hash-bound OCDR-12C nonprice snapshot",
            "intrahour_price": "reuse hash-bound OLIAH completed-hour intrahour snapshot",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_candidate_incidence_known": True,
            "prior_candidate_outcomes_used_for_cvlir": False,
            "cvlir_candidate_incidence_opened": False,
            "cvlir_post_entry_return_or_pnl_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
