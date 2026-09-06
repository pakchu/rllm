"""Outcome-blind preregistration for FMGRR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/funding_mark_gap_reconciliation_relay_preregistration_2026-08-09.json")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "funding_mark_gap_reconciliation_relay_v1",
        "policy_id": "FMGRR-6",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "The Binance funding settlement mark is a broad fair-price estimate distinct "
                "from the last traded perpetual price. At the fixed 08:00 UTC settlement, an "
                "unusually large mark-minus-last gap during high BTC variation should reconcile "
                "by the traded perpetual moving toward the completed settlement mark."
            ),
            "side": "sign of log(settlement mark price / 07:59 UTC completed-minute close)",
            "why_distinct": (
                "Prior funding candidates used the funding rate, settlement absorption, age, "
                "or funding-price direction. FMGRR uses neither funding-rate sign nor magnitude; "
                "it uses the contemporaneously published settlement mark versus the already "
                "completed last-trade close. It is not a prior control or candidate repair."
            ),
            "why_suited_to_volatile_regimes": (
                "the prior-24h BTC realized-variation rank must be at least 0.65"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one daily 08:05 UTC mark-reconciliation clock with a separate settlement source "
                "is absent from Gross9 primitives"
            ),
        },
        "features": {
            "settlement": (
                "the sole BTCUSDT funding_rates_binance row with funding_time in [08:00,08:01) "
                "UTC for a calendar date; finite positive mark_price required"
            ),
            "last_trade_close": (
                "the exact bars_binance BTCUSDT 1m close at 07:59 UTC, completed before settlement"
            ),
            "mark_gap": "log(mark_price / last_trade_close); strict zero is ineligible",
            "absolute_gap_rank": (
                "strict-prior midrank of abs(mark_gap) against at most 252 prior valid daily "
                "08:00 settlements; minimum 126; current excluded; rank>=0.75"
            ),
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over the exact 1440 BTCUSDT 1m bars in "
                "[prior 08:00 UTC, current 08:00 UTC)"
            ),
            "btc_variation_rank": (
                "strict-prior midrank against at most 252 prior valid daily variations; minimum "
                "126; current excluded; rank>=0.65"
            ),
            "availability": (
                "actual funding_time after 08:00 plus the completed 07:59 bar; entry is not before "
                "the next exact 5m boundary"
            ),
            "missing_duplicate_nonpositive_or_late": "ineligible or source failure; no imputation",
        },
        "clock": {
            "decision": "each calendar day after the completed 08:00 UTC funding record",
            "entry": "exact 08:05 UTC BTCUSDT 5m open, strictly after feature availability",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_accounting": (
                "funding rate is forbidden as a signal input but exact settlements are applied "
                "to later PnL"
            ),
            "no_imputation": True,
        },
        "policy": {
            "history_observations": 252,
            "minimum_history_observations": 126,
            "absolute_gap_rank_min": 0.75,
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
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
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
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "settlement": {
                "table": "funding_rates_binance",
                "symbol": "BTCUSDT",
                "columns": ["funding_time", "mark_price"],
                "funding_rate_signal_forbidden": True,
                "read_only": True,
            },
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": ["no_volatility_gate", "no_gap_tail", "one_day_stale_gap", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "database_metadata_only_opened_before_preregistration": True,
            "settlement_or_market_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent fair-mark versus last-trade reconciliation channel plus user-required "
                "high volatility"
            ),
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no settlement clock, threshold, side, hold, timing, volatility, or subset repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(registration: dict[str, Any]) -> None:
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if (
        registration.get("manifest_hash") != canonical_hash(core)
        or registration.get("outcomes_opened") is not False
        or registration.get("source_incidence_opened") is not False
    ):
        raise RuntimeError("FMGRR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
