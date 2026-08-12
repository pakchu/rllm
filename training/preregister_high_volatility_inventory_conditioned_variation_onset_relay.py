"""Outcome-blind preregistration for HVICVO-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVICVO-12"
DEFAULT_OUTPUT = Path("results/high_volatility_inventory_conditioned_variation_onset_relay_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_inventory_conditioned_variation_onset_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "public_source_basis": {
            "realized_variation": "https://www.nber.org/papers/w8160",
            "claim_used": "sums of high-frequency squared returns measure completed realized variation without supplying future direction.",
            "perpetual_mechanics": "https://arxiv.org/abs/2212.06888",
            "claim_used_2": "open interest measures outstanding leveraged inventory and is used only through observations available by decision time.",
            "implementation_is_unpublished_adaptation": True,
        },
        "mechanism": {
            "claim": "When completed eight-hour BTC realized variation crosses from below into its causal upper regime, the volatility onset persists directionally only when leveraged inventory is being added. Follow the completed BTC direction if OI expands and fade it if OI contracts, for twelve elapsed hours.",
            "side": "product of strict completed BTC block-return sign and strict completed block OI-change sign",
            "why_distinct": "HVTVIR ranks the late share of a daily variance budget and requires early/late direction agreement. HVOIP-family candidates rank within-block OI-price coupling or endpoint states without a realized-variation threshold crossing. HVICVO trades only the below-to-above variation onset and uses OI sign solely as a direction transform. It uses no DVOL, premium, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "the completed eight-hour variation rank must cross from below to at least 0.65",
            "why_low_gross9_overlap_is_plausible": "three offset daily variation-onset opportunities with OI-conditioned direction are absent from Gross9 definitions",
        },
        "features": {
            "decision_grid": "exact 02:00, 10:00, and 18:00 UTC boundary D",
            "price_block": "96 exact coherent BTCUSDT five-minute bars aggregated from 480 unique one-minute rows in [D-8h,D)",
            "block_return": "log(last completed close/first completed open), finite strict nonzero",
            "realized_variation": "sqrt(sum of 96 squared five-minute close-to-close log returns), finite strict positive",
            "oi_path": "97 exact positive BTCUSDT period=5m observations from D-8h through D; timestamp<=D; no imputation",
            "oi_change": "log(last/first OI), finite strict nonzero",
            "variation_rank": "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current excluded",
            "onset": "current variation rank>=0.65 and immediately prior exact source-valid decision variation rank<0.65; missing prior cannot trigger",
            "availability": "D after every completed bar and OI observation is available",
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact offset eight-hour boundary D",
            "entry": "exact BTCUSDT D+5m open",
            "side": "follow completed BTC return when OI expands; fade it when OI contracts",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after novelty passes",
            "rv20": "not a signal input; q90 audit only after unchanged all-stage economics pass",
        },
        "policy": {"block_hours": 8, "price_bars": 96, "oi_points": 97, "history_decisions": 270, "minimum_history_decisions": 180, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_onset_crossing", "ignore_inventory_sign_follow_return", "one_block_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "read_only": True}, "oi": {"table": "open_interest_binance", "symbol": "BTCUSDT", "period": "5m", "read_only": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {"prior_variation_and_OI_family_outcomes_known": True, "exact_candidate_incidence_or_outcomes_known": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent realized-variation threshold onset with OI-sign direction transformation"},
        "stopping_rule": "Terminal first failure: source, novelty, train, test, eval, final; no threshold, side, hold, clock, subset, source, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVICVO preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
