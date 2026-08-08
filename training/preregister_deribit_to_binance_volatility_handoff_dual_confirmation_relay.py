"""Outcome-blind preregistration for DBVHDR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/deribit_to_binance_volatility_handoff_dual_confirmation_relay_preregistration_2026-08-08.json")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "deribit_to_binance_volatility_handoff_dual_confirmation_relay_v1",
        "policy_id": "DBVHDR-6",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "mechanism": {
            "claim": "A completed hour in which Deribit DVOL expands while Binance BVOL contracts, followed by a completed hour in which both expand, marks an options-led volatility discovery handoff to the broader venue. If BTC moves in the same nonzero direction in both hours, the twice-confirmed direction should relay for six hours.",
            "side": "common sign of the two completed-hour BTC returns",
            "why_distinct": "DBVHDR uses an ordered two-hour venue transition from Deribit-only expansion to joint expansion plus two-hour directional BTC confirmation. It is not contemporaneous BVOL/DVOL disagreement, a fixed four-hour ignition block, a high-volatility level gate, or intrahour absorption.",
            "why_low_gross9_overlap_is_plausible": "the ordered cross-venue implied-volatility handoff and its two completed-hour confirmation geometry are absent from Gross9",
            "why_suited_to_volatile_regimes": "the clock activates on newly broadening implied-volatility expansion rather than on static calm-market levels",
        },
        "clock": {
            "decision": "T after two consecutive completed source-valid UTC hours [T-2h,T-1h) and [T-1h,T)",
            "leader_hour": "prior-hour normalized DVOL body > 0 and normalized BVOL body < 0",
            "handoff_hour": "current-hour normalized DVOL body > 0 and normalized BVOL body > 0",
            "price_confirmation": "BTC completed-hour returns in the leader and handoff hours have the same nonzero sign",
            "trigger": "false-to-true onset with all required source hours consecutive",
            "entry": "exact BTCUSDT T+5m open, after all features are available",
            "side": "common sign of the two completed-hour BTC returns",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; chronological signals; exit first on equal open",
            "oi": "not a signal input",
            "funding": "not a signal input; opened only for later exact PnL accounting",
            "no_imputation": True,
        },
        "policy": {
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
            "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "diagnostic_controls": {
            "no_ordered_handoff": "current-hour joint expansion plus same-direction two-hour BTC confirmation",
            "leader_hour_only": "prior-hour DVOL expansion/BVOL contraction plus prior-hour BTC direction",
            "handoff_hour_only": "current-hour joint expansion plus current-hour BTC direction",
            "one_hour_stale_handoff": "the frozen handoff state lagged by one completed hour",
            "direction_flip": "same entries with opposite side",
        },
        "source_plan": {
            "volatility": "hash-bound official Binance BTCBVOLUSDT and Deribit BTC DVOL completed-hour candles, including the sealed 2026-07 extension",
            "completed_price_features": "hash-bound preentry BTC intrahour path snapshot; only completed-hour returns are opened for support",
            "execution_price": "sealed until source-support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_candidate_incidence_known": True,
            "prior_candidate_outcomes_used_for_dbvhdr": False,
            "dbvhdr_candidate_incidence_opened": False,
            "dbvhdr_post_entry_return_or_pnl_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "diagnostic_controls_cannot_be_promoted": True,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
