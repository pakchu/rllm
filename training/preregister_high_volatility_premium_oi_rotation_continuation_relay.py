"""Outcome-blind preregistration for HVPORC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVPORC-8"
DEFAULT_OUTPUT = Path("results/high_volatility_premium_oi_rotation_continuation_relay_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_premium_oi_rotation_continuation_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "public_source_basis": {
            "deribit_dvol": "https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/",
            "claim_used": "DVOL supplies a forward-looking volatility regime and no directional input.",
            "perpetual_mechanics": "https://arxiv.org/abs/2212.06888",
            "claim_used_2": "Perpetual premium links the contract to spot while open interest measures outstanding leveraged inventory.",
            "implementation_is_unpublished_adaptation": True,
        },
        "mechanism": {
            "claim": "During elevated implied volatility, an eight-hour premium-index displacement whose strict sign reverses the preceding block while current open interest expands marks a newly sponsored derivatives-pressure rotation. Follow the completed new premium polarity for one eight-hour inventory cycle.",
            "side": "strict sign of the current completed eight-hour premium-index close-minus-open displacement",
            "why_distinct": "PPCSR follows a one-minute premium zero-cross after sixty persistent minutes and has no OI or magnitude regime. HVOIRR fades premium after an OI contraction-to-refill transition. HVPORC requires a block-to-block premium-displacement sign rotation plus contemporaneous OI expansion and follows, rather than fades, the new polarity. No prior event set, control, or clock is reused.",
            "why_suited_to_volatile_regimes": "DVOL and absolute current premium displacement must each occupy their causal upper 40%.",
            "why_low_gross9_overlap_is_plausible": "The derivatives-only two-block polarity rotation and OI sponsorship predicate is absent from Gross9 definitions.",
        },
        "features": {
            "decision_grid": "exact 00:00, 08:00, and 16:00 UTC boundary D",
            "prior_premium_block": "480 exact unique BTCUSDT premium-index 1m rows in [D-16h,D-8h); finite strict-nonzero displacement",
            "current_premium_block": "480 exact unique BTCUSDT premium-index 1m rows in [D-8h,D); finite strict-nonzero displacement",
            "premium_rotation": "strict signs of prior and current completed displacements are opposite",
            "current_oi_block": "96 exact unique BTCUSDT 5m OI rows in [D-8h,D); finite positive endpoints and current log change>0",
            "dvol_regime": "exact completed DVOL hour ending D with positive close",
            "causal_ranks": "strict-prior midranks over at most 270 source-valid boundaries, minimum 180; current excluded",
            "eligibility": "premium sign rotation, current OI expansion, abs(current premium displacement) rank>=0.60, and DVOL close rank>=0.60",
            "availability": "D after every constituent source row is complete",
            "no_imputation": True,
            "grid": False,
        },
        "clock": {
            "decision": "exact eight-hour boundary D",
            "entry": "exact BTCUSDT D+5m open",
            "side": "follow current completed premium displacement",
            "hold": "8 elapsed hours",
            "reservation": "fixed nonoverlapping boundaries; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after novelty passes",
            "rv20": "not a signal input; q90 audit only after unchanged all-stage economics pass",
        },
        "policy": {"block_hours": 8, "premium_rows_per_block": 480, "oi_rows": 96, "history_boundaries": 270, "minimum_history_boundaries": 180, "premium_displacement_rank_min": 0.60, "dvol_level_rank_min": 0.60, "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_dvol_gate", "no_premium_tail", "no_oi_expansion", "one_block_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"premium": {"table": "bars_binance_premium", "symbol": "BTCUSDT", "interval": "1m", "read_only": True}, "oi": {"table": "open_interest_binance", "symbol": "BTCUSDT", "period": "5m", "read_only": True}, "dvol": {"path": "data/options_crowding_deleveraging_relay_sources_v4_2023_2026/dvol_hourly.csv.gz", "read_only": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {"prior_premium_and_OI_family_outcomes_known": True, "exact_candidate_incidence_or_outcomes_known": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent block-polarity rotation with contemporaneous leverage sponsorship"},
        "stopping_rule": "Terminal first failure: source, novelty, train, test, eval, final; no threshold, side, hold, clock, subset, source, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVPORC preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
