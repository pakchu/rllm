"""Outcome-blind preregistration for HVIGDR-6."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVIGDR-6"
DEFAULT_OUTPUT = Path("results/high_volatility_interbar_gap_dominance_relay_preregistration_2026-08-10.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_interbar_gap_dominance_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-10",
        "outcomes_opened": False, "source_incidence_opened": False,
        "gross9_rows_opened": False, "singleton": True,
        "mechanism": {
            "claim": (
                "In a volatile completed BTC path, concentration of close-to-next-open gap energy "
                "in one one-minute transition identifies discontinuous price discovery rather than "
                "diffuse candle movement. When the dominant gap and completed six-hour displacement "
                "agree, follow the discontinuity direction for six hours."
            ),
            "side": "common strict sign of the dominant inter-bar gap and six-hour BTC return",
            "why_distinct": (
                "Bipower jump candidates infer jumps from return variation; dominant-shock and dominant-"
                "volume candidates select large candle bodies or participation bars; range and wick "
                "candidates use intrabar extrema. HVIGDR uses only the discontinuity between each "
                "completed minute close and the next minute open, with no prior event or control reused."
            ),
            "why_suited_to_volatile_regimes": "prior-24-hour BTC realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "sparse endogenous inter-bar discontinuity tails are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "every exact UTC hour D",
            "block": "360 exact unique coherent positive BTCUSDT one-minute OHLC rows [D-6h,D); no imputation",
            "interbar_gap": "g_i=log(open_i/close_(i-1)) for block rows i=1..359",
            "gap_energy": "sum(g_i^2), strict positive",
            "dominant_gap": "earliest gap attaining max(abs(g_i)); strict nonzero",
            "gap_dominance": "dominant_gap^2/gap_energy",
            "block_return": "log(last close/first open), strict nonzero",
            "direction_alignment": "dominant_gap and block_return have one strict sign",
            "btc_realized_variation": "sqrt(sum squared one-minute log(close/open) returns over [D-24h,D))",
            "causal_ranks": "strict-prior midranks over at most 2160 source-valid hours, minimum 1440, current excluded",
            "eligibility": "gap-dominance rank>=0.80, absolute block-return rank>=0.60, variation rank>=0.65, and direction alignment",
        },
        "clock": {
            "decision": "exact completed hourly boundary",
            "entry": "exact BTCUSDT D+5m open", "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on an equal open",
            "split_crossing_action": "skip", "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding cash only after novelty passes",
            "oi_premium_rv20": "not signal inputs; RV20 q90 only after all economic stages pass",
        },
        "policy": {
            "block_minutes": 360, "gap_count": 359,
            "history_hours": 2160, "minimum_history_hours": 1440,
            "gap_dominance_rank_min": 0.80, "block_return_rank_min": 0.60,
            "variation_rank_min": 0.65, "entry_delay_minutes": 5,
            "hold_hours": 6, "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.20, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.10, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {
            "absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10, "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True,
            "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False,
            "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, and final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_gap_dominance_gate", "no_block_return_tail", "no_volatility_gate", "latest_dominant_gap_tie_break", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "source_plan": {
            "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_only_after_preregistration": True},
            "funding_rate_values": "sealed during source support", "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_jump_range_and_dominant_bar_family_outcomes_known": True,
            "prior_event_sets_or_controls_reused": False,
            "exact_interbar_gap_candidate_found_in_repository": False,
            "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "independent continuous-auction inter-bar discontinuity mechanism",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no gap definition, tie break, rank, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVIGDR preregistration drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); print(args.output)
