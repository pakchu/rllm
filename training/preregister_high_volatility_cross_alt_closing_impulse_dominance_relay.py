"""Outcome-blind preregistration for HVCACID-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVCACID-8"
SLUG = "high_volatility_cross_alt_closing_impulse_dominance_relay"
ALTS = ("ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "When at least four liquid alt perpetuals concentrate a large share of their completed five-minute auction path in the same final one-minute impulse, the synchronized late information shock should propagate into BTC during elevated variation. Follow the common closing impulse direction in BTC for eight hours.",
            "side": "same as the common final-minute impulse direction",
            "why_distinct": "This is neither a barrier event nor a five-minute return, volume, flow, funding, OI, premium, or wick consensus. It requires the final native one-minute return to dominate each alt's complete five-minute absolute one-minute path and to be unusually large relative to that alt's strictly prior final-minute impulses. It uses no BTC return direction, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "causal trailing BTC twenty-four-hour realized variation must occupy its upper 35 percent",
            "why_low_gross9_overlap_is_plausible": "irregular synchronized final-native-minute path concentration is absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "every exact UTC five-minute boundary D",
            "five_minute_bar": "five exact coherent native one-minute bars [D-5m,D) per symbol",
            "final_minute_return": "log(last native one-minute close/open) in [D-1m,D)",
            "absolute_path": "sum of absolute native one-minute log(close/open) returns over [D-5m,D)",
            "impulse_share": "absolute final-minute return divided by absolute_path, finite strict positive denominator",
            "impulse_dominant": "impulse_share at least 0.90 and final-minute absolute-return strict-prior midrank at least 0.85",
            "impulse_rank_history": "per alt, at most 8640 strictly prior source-valid five-minute decisions, minimum 6048, current excluded",
            "broad_impulse": "at least four of six alts have dominant final-minute impulses in one common direction; any simultaneous dominant opposite impulse invalidates the event",
            "btc_realized_variation": "sqrt(sum squared exact BTC one-minute log(close/open) returns over [D-24h,D)), finite strict positive",
            "variation_history": "at most 8640 strictly prior source-valid five-minute decisions, minimum 6048, current excluded",
            "variation_rank": "strict-prior midrank; rank>=0.65",
            "onset": "broad impulse true now and false at immediately prior exact source-valid five-minute decision; missing prior cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "D immediately after all seven completed five-minute bars are available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "same as common dominant final-minute impulse",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements only after novelty",
        },
        "policy": {
            "impulse_share_min": 0.90,
            "impulse_rank_history_decisions": 8640,
            "minimum_impulse_rank_history_decisions": 6048,
            "impulse_rank_min": 0.85,
            "minimum_impulse_breadth": 4,
            "variation_bars": 288,
            "variation_history_decisions": 8640,
            "minimum_variation_history_decisions": 6048,
            "variation_rank_min": 0.65,
            "onset_required": True,
            "entry_delay_minutes": 5,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "three_of_six_impulse", "whole_bar_direction", "one_bar_stale_impulse", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"table": "bars_binance", "symbols": ["BTCUSDT", *ALTS], "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close"], "window": ["2023-05-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration_commit": True, "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_cross_alt_outcomes_known": True, "repository_exact_cross_alt_final_native_minute_path_dominance_found": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent synchronized late-information impulse propagation"},
        "stopping_rule": "terminal first failure; no universe, native-minute definition, path share, impulse rank, breadth, variation, onset, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVCACID prereg drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registration = build()
    validate(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(registration, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != payload:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(payload)
    print(args.output)
