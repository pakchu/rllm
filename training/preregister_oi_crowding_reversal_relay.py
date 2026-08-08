"""Outcome-blind preregistration for OICRR-18."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/oi_crowding_reversal_relay_preregistration_2026-08-09.json")
SOURCE_BINDINGS = {
    "training/search_oi_liquidation_bidirectional_alpha.py": "cb86883f47eefa3a01a94390dffee167b217f32f04b760afa6e0fe4877a1bbab",
    "training/backtest_all_alpha_month.py": "3de3bc013cfd880d1f14740eb9a51f0c3506949dc9a716a36c44f15123226fe6",
    "preprocessing/live_db_features.py": "a4b903913a51e1322e8946cd8dad8fea08b2361655520d20312a65f4219d5099",
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "oi_crowding_reversal_relay_v1",
        "policy_id": "OICRR-18",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Extreme same-direction price displacement and OI-versus-price divergence "
                "identify crowded builds and liquidation overshoots. Joint negative tails "
                "map long after liquidation exhaustion; joint positive tails map short "
                "against crowded upside extension."
            ),
            "why_distinct": (
                "OICRR uses 24-hour price/OI displacement standardized over a trailing "
                "48-hour causal window and mirrored same-sign tails. OIDCR used opposite-"
                "sign four-hour contradiction with range/RSI gates; HVOPCR used an OI purge "
                "continuation law. No terminal candidate threshold or control is modified."
            ),
            "why_suited_to_volatile_regimes": (
                "both states require simultaneous extreme standardized price movement and "
                "extreme OI-price crowding divergence"
            ),
        },
        "feature_contract": {
            "bar": "completed BTCUSDT perpetual 5m bar",
            "price_log_change": "log(close_t)-log(close_t-288)",
            "oi_log_change": (
                "log(last causally observed positive open_interest_t)-"
                "log(last causally observed positive open_interest_t-288)"
            ),
            "zscore": (
                "rolling 576-bar sample mean/std including the completed current value; "
                "minimum 576, zero std invalid"
            ),
            "price_z": "zscore(price_log_change)",
            "oi_z": "zscore(oi_log_change)",
            "divergence": "oi_z-price_z",
            "no_future_fill": True,
            "no_imputation_except_causal_oi_forward_fill": True,
        },
        "frozen_states": {
            "long_liquidation_exhaustion": {
                "gates": [
                    {"feature": "ol_div_288", "op": "<=", "threshold": -1.3784369641584517},
                    {"feature": "ol_px_z_288", "op": "<=", "threshold": -1.188625763484384},
                ],
                "side": 1,
            },
            "short_crowded_extension": {
                "gates": [
                    {"feature": "ol_div_288", "op": ">=", "threshold": 1.4037721996297907},
                    {"feature": "ol_px_z_288", "op": ">=", "threshold": 1.1835598702396652},
                ],
                "side": -1,
            },
            "decision_grid": "UTC minute divisible by 30 on completed 5m bars",
            "conflict": "impossible under mirrored tails; skip defensively",
            "availability": "decide after completed bar and enter next 5m open",
            "global_reservation": "half-open; ignore new states while active",
            "no_threshold_side_hold_stride_or_subset_tuning": True,
        },
        "clock": {
            "entry": "next 5m open",
            "hold": "18 elapsed hours",
            "path_dependent_exit": False,
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per "
                "notional side, every held 5m favorable then adverse, global HWM, "
                "full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": ["price_tail_only", "divergence_tail_only", "one_bar_stale_features", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "database_contract": {
            "env_file": "/home/pakchu/rllm/.env",
            "tables": ["bars_binance", "open_interest_binance"],
            "symbol": "BTCUSDT",
            "interval": "1m aggregated to complete 5m bars",
            "columns": ["ts", "open", "high", "low", "close", "open_interest"],
            "read_only": True,
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "prior_rule_family_outcomes_known": True,
            "prior_outcome_warning": (
                "Thresholds and an 18-hour rule with 4% take-profit and 2.5% stop-loss "
                "were selected after inspection of 2024 and later diagnostics. A pass "
                "cannot be represented as fresh-data discovery."
            ),
            "fixed_clock_candidate_outcomes_known": False,
            "candidate_incidence_opened": False,
            "post_entry_outcomes_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "terminal first failure; no repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("OICRR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"OICRR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
