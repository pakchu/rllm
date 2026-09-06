"""Outcome-blind preregistration for HVFSCS-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVFSCS-6"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_funding_settlement_cash_sponsorship_relay_"
    "preregistration_2026-08-13.json"
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
        "protocol_version": "high_volatility_funding_settlement_cash_sponsorship_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "An extreme realized BTC perpetual funding transfer aligned with a volatile "
                "pre-settlement price move identifies leveraged directional demand. If the "
                "next completed hour is independently sponsored by same-direction Binance "
                "spot aggressive quote flow and spot price, cash demand has taken over from "
                "leverage rather than merely inheriting a crowded derivative position. Relay "
                "the cash-confirmed direction for six hours."
            ),
            "side": "common strict sign of funding, pre-settlement return, post-settlement spot return, and spot aggressive quote flow",
            "why_distinct": (
                "FSVUR requires post-settlement price reversal and joint BVOL/DVOL contraction. "
                "PFCR is a factor-neutral cross-alt funding-spread release. CSPR is an "
                "unscheduled spot/perpetual rejection topology. HVFSCS instead chains an "
                "actual BTC funding settlement to a later, independent spot cash-flow "
                "confirmation and uses no options volatility, OI, premium, or cross-alt leg."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed pre-settlement eight-hour BTC variation must rank in its "
                "causal upper 35 percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "a settlement-plus-delayed-cash-confirmation clock is absent from Gross9"
            ),
        },
        "features": {
            "event": "actual Binance BTCUSDT USD-M funding settlement S and its exact settled rate",
            "funding_rank": (
                "strict-prior midrank of abs(settled rate) over at most 270 earlier "
                "settlements, minimum 252, current excluded; rank>=0.60"
            ),
            "pre_settlement_path": (
                "96 exact coherent BTCUSDT perpetual five-minute bars in [S-8h,S); "
                "pre_return=log(last close/first open), strict nonzero"
            ),
            "pre_settlement_variation": (
                "sqrt(sum squared exact close-to-close five-minute log returns), including "
                "the first bar return from its open to close; strict positive"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 earlier source-valid settlement "
                "paths, minimum 252, current excluded; rank>=0.65"
            ),
            "leverage_alignment": "strict sign(settled rate)=strict sign(pre_return)",
            "post_settlement_cash_hour": (
                "60 exact coherent Binance spot BTCUSDT one-minute rows [S,S+1h), with "
                "finite nonnegative quote_asset_volume and taker_buy_quote, positive total "
                "quote volume, and no imputation"
            ),
            "spot_return": "log(final one-minute close/first one-minute open), strict nonzero",
            "spot_aggressive_quote_flow": (
                "2*sum(taker_buy_quote)-sum(quote_asset_volume), strict nonzero"
            ),
            "cash_confirmation": (
                "spot return and spot aggressive quote flow both have the leverage-alignment sign"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "S+1h after the funding settlement and full spot confirmation hour are complete",
            "feature_available_time": "maximum of settlement timestamp and final spot source-row availability, conservatively S+1h",
            "entry": "exact BTCUSDT perpetual S+1h+5m open",
            "hold": "6 elapsed hours, ending before the next regular eight-hour settlement",
            "side": "common confirmed sign",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": (
                "the trigger settlement occurs before entry and is not earned; exact later "
                "settlements are included only after novelty"
            ),
        },
        "policy": {
            "prior_settlements": 270,
            "minimum_prior_settlements": 252,
            "absolute_funding_rank_min": 0.60,
            "variation_rank_min": 0.65,
            "pre_settlement_hours": 8,
            "cash_confirmation_hours": 1,
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
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes all stages",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "definitions": {
                "no_funding_tail": "aligned settlement and cash confirmation without funding-rank gate",
                "no_variation_gate": "funding tail and cash confirmation without variation rank",
                "funding_and_price_only": "funding/pre-return alignment without post-settlement cash confirmation",
                "one_settlement_stale_funding": "prior settled rate and rank with current paths and cash confirmation",
                "direction_flip": "negative primary side",
                "same_clock_forced_long": "side +1 on primary clock",
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "funding": {"table": "funding_rates_binance", "symbol": "BTCUSDT", "exact_settlements": True},
            "perpetual": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "5m"},
            "spot": {
                "table": "bars_binance_spot",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"],
            },
            "window": ["2022-12-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration_commit": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_funding_settlement_and_cash_flow_family_outcomes_known": True,
            "repository_settlement_to_delayed_spot_cash_confirmation_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent realized-funding-to-cash-sponsorship handoff mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no funding source, path, variation, alignment, cash "
            "confirmation, side, clock, hold, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVFSCS preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
