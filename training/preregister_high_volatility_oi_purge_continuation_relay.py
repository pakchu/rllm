"""Outcome-blind preregistration for HVOPCR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_oi_purge_continuation_relay_preregistration_2026-08-08.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_oi_purge_continuation_relay_v1",
        "policy_id": "HVOPCR-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "mechanism": {
            "claim": (
                "When BVOL and DVOL are already elevated, a large completed-hour BTC move "
                "accompanied by an unusually large contraction in perpetual open interest marks "
                "forced inventory removal rather than new-position chasing; the liquidation-driven "
                "price direction should continue for six hours."
            ),
            "side": "sign of the completed-hour BTC return",
            "why_distinct": (
                "HVOPCR requires falling OI, high cross-venue implied-volatility levels, and price "
                "continuation. OCDR and OICER required rising OI and traded reversal; CVLIR used "
                "absolute OI repricing, opposite volatility bodies, and a late-half breakout."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "dual implied-volatility levels plus causal perpetual-OI purge and completed-hour "
                "price shock are absent from Gross9"
            ),
        },
        "clock": {
            "decision": "T after one completed UTC hour [T-1h,T)",
            "volatile_regime": (
                "BVOL close and DVOL close at T are each >= own strictly-prior 720h q60, "
                "requiring 672 valid observations"
            ),
            "oi": (
                "raw-time backward-asof positive BTCUSDT OI observations at T and T-60m, each "
                "age<=5m; change=current/prior-1 must be strictly negative"
            ),
            "oi_purge": (
                "absolute one-hour OI change >= strictly-prior 720h q75 with 672 observations"
            ),
            "price_shock": (
                "nonzero absolute completed-hour BTC return >= strictly-prior 720h q60 with "
                "672 observations"
            ),
            "trigger": "false-to-true onset on an exact consecutive valid hourly grid",
            "entry": "exact BTCUSDT T+5m open after every signal feature is available",
            "side": "sign of completed-hour BTC return",
            "hold": "6 elapsed hours",
            "reservation": "global half-open, exit first on equal open",
            "funding": "not a signal input; exact settlements only for later PnL",
            "no_imputation": True,
        },
        "policy": {
            "prior_hours": 720,
            "prior_min_hours": 672,
            "volatility_level_quantile": 0.60,
            "absolute_oi_change_quantile": 0.75,
            "absolute_price_return_quantile": 0.60,
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
            "volatility_oi_funding": "reuse hash-bound OCDR-12C source snapshot through 2026-08-01",
            "completed_price_hour": "reuse hash-bound OICER completed-hour snapshot through 2026-08-01",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "no_oi_purge",
                "no_price_shock",
                "one_hour_stale_regime",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "related_rising_oi_reversal_candidates_known": True,
            "related_candidate_outcomes_used_to_define_hvopcr": False,
            "hvopcr_candidate_incidence_opened": False,
            "hvopcr_post_entry_return_or_pnl_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent falling-OI cascade mechanism targeted to high-volatility markets",
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
