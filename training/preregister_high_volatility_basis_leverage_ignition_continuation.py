"""Outcome-blind preregistration for HVBLIC-6."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVBLIC-6"
DEFAULT_OUTPUT = Path("results/high_volatility_basis_leverage_ignition_continuation_preregistration_2026-08-11.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_basis_leverage_ignition_continuation_v1", "policy_id": POLICY_ID, "as_of_date": "2026-08-11", "singleton": True,
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False,
        "public_source_basis": {
            "dvol": "https://support.deribit.com/hc/en-us/articles/31424954825373-DVOL-Futures",
            "perpetual_mechanics": "https://arxiv.org/html/2212.06888v5",
            "open_interest": "https://arxiv.org/abs/2310.14973",
            "claims_used": ["DVOL is a forward-looking implied-volatility state variable.", "The preceding premium/basis path drives periodic perpetual funding and convergence pressure.", "Open interest measures outstanding leveraged inventory and is used only as fresh-leverage confirmation."],
            "implementation_is_unpublished_adaptation": True,
        },
        "mechanism": {
            "claim": "When DVOL is elevated and rises during a completed hour, a large same-hour Binance premium-index displacement accompanied by increasing BTC perpetual open interest identifies basis pressure sponsored by fresh leveraged inventory. Follow the premium direction for six elapsed hours.",
            "side": "strict sign of the completed one-hour premium close-minus-open displacement",
            "why_distinct": "OVEPR requires both BVOL and DVOL positive bodies, Deribit body leadership, and premium path efficiency but no OI. OICER uses BTC price chase plus funding confirmation and fades it. OCDR uses extreme funding for direction and deleveraging reversal. HVBLIC uses no BTC price, BVOL, funding, prior event set, or prior control; DVOL supplies regime/ignition, premium supplies direction, and positive OI change supplies fresh-leverage confirmation.",
            "why_suited_to_volatile_regimes": "DVOL close is in its causal upper 40% and its completed-hour body is positive.",
            "why_low_gross9_overlap_is_plausible": "derivatives-only hourly false-to-true onsets are absent from Gross9 clocks.",
        },
        "features": {
            "decision_grid": "each exact UTC hour D",
            "premium_hour": "60 exact unique BTCUSDT premium-index one-minute rows in [D-1h,D); displacement=last close-first open; strict nonzero",
            "oi_hour": "12 exact unique BTCUSDT period=5m OI rows in [D-1h,D); log(last/first)>0",
            "dvol_hour": "exact completed Deribit DVOL candle ending D with close>open and positive finite OHLC",
            "causal_ranks": "strict-prior midranks over at most 720 source-valid hours, minimum 672; current excluded",
            "eligibility": "DVOL-close rank>=0.60, absolute-premium-displacement rank>=0.60, DVOL body>0, and OI change>0",
            "onset": "eligible now and immediately prior exact source-valid hour ineligible; a missing predecessor cannot trigger",
            "availability": "D after all constituent source rows complete", "no_imputation": True, "grid": False,
        },
        "clock": {"decision": "completed UTC hour D", "entry": "exact BTCUSDT D+5m open", "side": "premium displacement sign", "hold": "6 elapsed hours", "reservation": "global half-open, earliest onset wins, exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding": "not a signal input; exact held settlements only after novelty", "rv20": "q90 audit only after all economics pass"},
        "policy": {"hour_minutes": 60, "oi_points": 12, "history_hours": 720, "minimum_history_hours": 672, "dvol_level_rank_min": .60, "premium_displacement_rank_min": .60, "entry_delay_minutes": 5, "hold_hours": 6, "leverage": .5, "base_cost_per_notional_side": .0006, "stress_cost_per_notional_side": .001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": .20, "max_month_share": .45},
        "novelty_gates": {"exact_entry_jaccard_max": .10, "candidate_near_6h_share_max": .35, "occupied_5m_bar_jaccard_max": .25, "absolute_signed_exposure_pearson_max": .35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3., "strict_mdd_max_pct": 15., "mean_gross_underlying_min_bp": 20., "weekly_signflip_one_sided_p_max": .10, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_dvol_level", "no_dvol_rise", "no_premium_tail", "no_oi_increase", "one_hour_stale_features", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"premium": {"table": "bars_binance_premium", "symbol": "BTCUSDT", "interval": "1m", "read_only": True}, "oi": {"table": "open_interest_binance", "symbol": "BTCUSDT", "period": "5m", "read_only": True}, "dvol": {"path": "data/options_crowding_deleveraging_relay_sources_v4_2023_2026/dvol_hourly.csv.gz", "read_only": True}, "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_derivatives_family_outcomes_known": True, "exact_candidate_incidence_or_outcomes_known": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "public-source basis-led continuation with fresh-leverage confirmation"},
        "stopping_rule": "Terminal first failure; no threshold, side, hold, clock, subset, source, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {k: v for k, v in value.items() if k != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core): raise RuntimeError("HVBLIC preregistration hash mismatch")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(); value=build(); validate(value); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(args.output)
