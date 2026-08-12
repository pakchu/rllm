"""Outcome-blind preregistration for HVCBFC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCBFC-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cash_basis_feedback_continuation_relay_preregistration_2026-08-13.json"
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
        "protocol_version": "high_volatility_cash_basis_feedback_continuation_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "During elevated BTC variation, an unusually strong positive lagged correlation "
                "from completed spot returns to the next minute's perpetual-minus-spot basis "
                "change identifies cash price discovery being reinforced by delayed leveraged "
                "demand. Follow the completed cash direction for eight hours at the fresh "
                "feedback-tail onset."
            ),
            "side": "strict sign of the completed eight-hour spot return",
            "why_distinct": (
                "HVCBR ranks one final-two-hour endpoint basis change and trades convergence; "
                "premium-to-price leadership asks whether premium changes precede price; SPVTA "
                "compares venue variation relocation; SPIDR uses contemporaneous venue-return "
                "correlation. HVCBFC instead measures spot return[t] against basis_change[t+1] "
                "over the full aligned minute path and uses no endpoint basis shock, premium-index "
                "candle, venue-variance share, prior event, or control."
            ),
            "volatile_market_target": (
                "completed perpetual realized variation and positive cash-to-basis feedback "
                "must enter independent causal upper tails"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "lagged cross-venue cash-to-basis feedback onsets are absent from Gross9"
            ),
        },
        "features": {
            "decision_grid": "exact 00:00/08:00/16:00 UTC boundaries D",
            "block": (
                "480 exact timestamp-aligned coherent BTCUSDT bars_binance_spot and bars_binance "
                "one-minute rows [D-8h,D)"
            ),
            "spot_minute_return": "log(spot close/spot open) for each completed minute",
            "basis_change": (
                "log(perpetual close/spot close)-log(perpetual open/spot open) for each minute"
            ),
            "cash_basis_feedback": (
                "Pearson correlation of spot_minute_return[t] with basis_change[t+1] across 479 "
                "ordered pairs; both vectors require finite strict-positive population variance"
            ),
            "feedback_gate": "cash_basis_feedback strict positive",
            "feedback_rank": (
                "strict-prior midrank of cash_basis_feedback over at most 270 valid blocks, "
                "minimum 180, current excluded; rank>=0.80"
            ),
            "completed_spot_return": "log(last spot close/first spot open), finite strict nonzero",
            "realized_variation": (
                "sum squared log(perpetual close/perpetual open) minute returns, finite strict positive"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded; "
                "rank>=0.65"
            ),
            "eligible_state": "positive feedback tail and variation gates pass",
            "onset": (
                "eligible now and immediately previous exact source-valid decision block ineligible"
            ),
            "no_imputation": True,
        },
        "clock": {
            "entry": "D+5m BTCUSDT perpetual open",
            "side": "sign completed_spot_return",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty",
        },
        "policy": {
            "prior_blocks": 270,
            "minimum_prior_blocks": 180,
            "feedback_rank_min": 0.80,
            "variation_rank_min": 0.65,
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
            "occupied_5m_bar_jaccard_max": 0.25,
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
                "fixed quantity, exact funding, 6bp/10bp per notional side, held 5m favorable "
                "then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged all-stage pass",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_feedback_tail",
                "no_variation_gate",
                "contemporaneous_feedback",
                "one_block_stale_geometry",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "perpetual": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "spot": {
                "table": "bars_binance_spot",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source and novelty pass",
        },
        "research_boundary": {
            "prior_spot_perpetual_basis_variation_and_leadlag_outcomes_known": True,
            "repository_spot_return_to_next_basis_change_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent lagged cash-price to leveraged-basis feedback",
        },
        "stopping_rule": (
            "terminal first failure; no feedback formula, rank, onset, side, clock, hold, subset, "
            "threshold, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVCBFC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
