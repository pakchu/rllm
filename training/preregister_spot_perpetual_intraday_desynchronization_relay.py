"""Outcome-blind preregistration for SPIDR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "SPIDR-8"
DEFAULT_OUTPUT = Path(
    "results/spot_perpetual_intraday_desynchronization_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "spot_perpetual_intraday_desynchronization_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A completed high-variation eight-hour BTC block whose aligned Binance spot and "
                "perpetual one-minute returns are unusually desynchronized represents fragmented "
                "price transmission. When both venues nevertheless finish in the same direction, "
                "follow the cash-confirmed displacement for eight hours as arbitrage reconciles the paths."
            ),
            "side": "common strict sign of completed spot and perpetual eight-hour returns",
            "why_distinct": (
                "SPIDR uses one Pearson correlation across 480 aligned spot/perpetual minute returns. "
                "SPVTA compares early/late venue variation; HVCBR trades a final-two-hour basis change; "
                "SLVCR compares endpoint transmission under implied-volatility expansion; participation "
                "and flow candidates use volume or taker direction. SPIDR uses no volume, flow, premium "
                "index, funding, OI, implied volatility, prior events, fitted outcomes, or controls."
            ),
            "why_suited_to_volatile_regimes": (
                "the causal perpetual-variation tail selects volatile blocks while a desynchronization "
                "tail isolates venue-fragmented volatility rather than ordinary common movement"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "three fixed UTC spot-perpetual desynchronization onsets with eight-hour reservation are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries",
            "block": "480 exact coherent aligned BTCUSDT 1m rows [D-8h,D) from bars_binance and bars_binance_spot",
            "minute_return": "log(close/open) independently for each completed minute and venue",
            "return_correlation": "Pearson correlation across all 480 aligned minute returns; finite positive variance required",
            "desynchronization": "one minus return correlation",
            "desynchronization_rank": "strict-prior midrank over at most 270 jointly valid blocks, minimum 180, current excluded; rank>=0.75",
            "perpetual_variation": "sqrt(sum squared perpetual minute returns), strict positive",
            "variation_rank": "strict-prior midrank over at most 270 jointly valid blocks, minimum 180, current excluded; rank>=0.65",
            "spot_return": "log(last spot close/first spot open) over eight hours",
            "perpetual_return": "log(last perpetual close/first perpetual open) over eight hours",
            "direction_confirmation": "spot and perpetual block returns have one strict nonzero sign",
            "onset": "eligible now and immediately prior exact jointly valid block ineligible; missing prior opportunity cannot trigger",
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed eight-hour boundary",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "common completed-return sign",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "variation_rank_min": 0.65,
            "desynchronization_rank_min": 0.75,
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
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_desynchronization_gate", "no_variation_gate", "raw_correlation_below_half",
                "one_block_stale_desynchronization", "direction_flip", "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "perpetual": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True},
            "spot": {"table": "bars_binance_spot", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_spot_perpetual_transmission_family_outcomes_known": True,
            "repository_spot_perpetual_return_correlation_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_spidr_formula_ranks_direction_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed aligned spot-perpetual return desynchronization mechanism",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no correlation definition, rank, side, hold, clock, subset, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
