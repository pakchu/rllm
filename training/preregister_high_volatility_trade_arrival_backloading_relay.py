"""Outcome-blind preregistration for HVTAB-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVTAB-6"
DEFAULT_OUTPUT = Path("results/high_volatility_trade_arrival_backloading_relay_preregistration_2026-08-10.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_trade_arrival_backloading_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "When executions concentrate in the final hour of a completed high-variation four-hour BTC block and that final hour extends the block move, participation is arriving behind directional price discovery rather than merely accompanying an early shock. Follow the common direction for six elapsed hours.",
            "side": "common strict sign of completed four-hour and final-hour returns",
            "why_distinct": "HVTAB uses temporal concentration of raw execution counts inside fixed four-hour BTC blocks. DLVCR used final-six-hour quote-volume share in a daily block; TECC used early-versus-late average ticket size; CATACF used cross-asset five-minute arrival HHI. HVTAB uses neither quote volume, ticket size, cross-sectional breadth, funding, OI, external assets, nor any prior event set or control.",
            "why_suited_to_volatile_regimes": "completed four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "six fixed UTC temporal execution-arrival clocks conditioned on a causal high-variation state are absent from Gross9 primitives",
        },
        "features": {
            "decision_grid": "exact 00:00/04:00/08:00/12:00/16:00/20:00 UTC boundaries",
            "block": "240 exact coherent BTCUSDT bars_binance interval=1m rows in [decision-4h,decision)",
            "returns": "log(last close/first open) for the full four hours and final 60 minutes; both strict nonzero and same sign",
            "trade_counts": "finite nonnegative integer number_of_trades each minute; positive full-block total",
            "late_arrival_share": "sum number_of_trades in final 60 minutes divided by sum over all 240 minutes",
            "late_share_rank": "strict-prior midrank over at most 540 valid fixed blocks, minimum 360, current excluded; rank>=0.75",
            "realized_variation": "sqrt(sum squared one-minute close-to-close log returns in the completed block), strict positive",
            "variation_rank": "strict-prior midrank over at most 540 valid fixed blocks, minimum 360, current excluded; rank>=0.65",
            "eligibility": "late-share and variation ranks pass and full/final-hour returns share one strict sign",
            "onset": "eligible now and immediately preceding exact four-hour opportunity ineligible; missing prior opportunity cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact completed four-hour boundary",
            "entry": "exact BTCUSDT perpetual decision+5m open",
            "side": "common strict completed-return sign",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "block_minutes": 240,
            "late_minutes": 60,
            "history_blocks": 540,
            "minimum_history_blocks": 360,
            "late_share_rank_min": 0.75,
            "variation_rank_min": 0.65,
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
            "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": ["no_variation_gate", "no_arrival_tail", "one_block_stale_features", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "number_of_trades"], "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_temporal_volume_ticket_and_arrival_family_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_hvtab_direction_rank_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "raw execution-arrival timing inside fixed volatile BTC blocks",
        },
        "stopping_rule": "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no feature, rank, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
