"""Outcome-blind preregistration for HVOILVT-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVOILVT-12"
DEFAULT_OUTPUT = Path("results/high_volatility_oi_sponsored_late_variation_takeover_relay_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_oi_sponsored_late_variation_takeover_relay_v1",
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
            "claim": "When an already high-variation completed eight-hour BTC auction shifts unusually large quadratic variation into its final two hours, reverses the prior six-hour direction, and expands open interest, new leveraged inventory is sponsoring a late directional takeover. Follow the completed final-two-hour direction for twelve elapsed hours.",
            "side": "strict sign of completed final-two-hour BTC return",
            "why_distinct": "HVTVIR requires early/late direction agreement on a once-daily 24-hour clock. HVOILVI uses OI sign as a direction transform without requiring a directional takeover. HVOILVT requires strict early/late disagreement plus current OI expansion and follows only the late direction. No prior event set or diagnostic control is reused.",
            "why_suited_to_volatile_regimes": "overall variation rank>=0.65 and late-two-hour variance-share rank>=0.75",
            "why_low_gross9_overlap_is_plausible": "offset OI-sponsored late-direction takeover events are absent from Gross9 definitions",
        },
        "features": {
            "decision_grid": "exact 02:00, 10:00, and 18:00 UTC boundary D",
            "price_block": "96 exact coherent BTCUSDT five-minute bars aggregated from 480 unique one-minute rows in [D-8h,D)",
            "early_return": "log(close of bar 72/open of bar 1) over the completed first six hours, finite strict nonzero",
            "late_return": "log(last completed close/open of bar 73) over the completed final two hours, finite strict nonzero",
            "directional_takeover": "strict signs of early_return and late_return are opposite",
            "realized_variation": "sqrt(sum of 96 squared five-minute intrabar log(close/open) returns), finite strict positive",
            "late_variance_share": "sum of final 24 squared returns divided by sum of all 96 squared returns",
            "oi_path": "97 exact positive BTCUSDT period=5m observations from D-8h through D; timestamp<=D; no imputation",
            "oi_change": "log(last/first OI), finite and strictly positive",
            "causal_ranks": "separate strict-prior midranks of realized variation and late variance share over at most 270 earlier source-valid decisions, minimum 180, current excluded",
            "eligibility": "variation rank>=0.65, late-variance-share rank>=0.75, directional takeover, and OI expansion",
            "ignition": "eligible now and immediately prior exact source-valid decision ineligible; missing prior cannot trigger",
            "availability": "D after every completed bar and OI observation is available",
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact offset eight-hour boundary D",
            "entry": "exact BTCUSDT D+5m open",
            "side": "follow completed final-two-hour BTC return",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after novelty passes",
            "rv20": "not a signal input; q90 audit only after unchanged all-stage economics pass",
        },
        "policy": {"block_hours": 8, "price_bars": 96, "late_bars": 24, "oi_points": 97, "history_decisions": 270, "minimum_history_decisions": 180, "variation_rank_min": 0.65, "late_variance_share_rank_min": 0.75, "entry_delay_minutes": 5, "hold_hours": 12, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_overall_variation_gate", "no_late_share_gate", "no_fresh_ignition", "no_directional_takeover", "no_oi_expansion", "one_block_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "read_only": True}, "oi": {"table": "open_interest_binance", "symbol": "BTCUSDT", "period": "5m", "read_only": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {"prior_temporal_variance_and_OI_family_outcomes_known": True, "exact_candidate_incidence_or_outcomes_known": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent late-direction takeover sponsored by expanding leveraged inventory"},
        "stopping_rule": "Terminal first failure: source, novelty, train, test, eval, final; no threshold, side, hold, clock, subset, source, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVOILVT preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
