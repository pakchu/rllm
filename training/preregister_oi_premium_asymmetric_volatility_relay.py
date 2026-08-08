"""Outcome-blind preregistration for OIPAR-ASYM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/oi_premium_asymmetric_volatility_relay_preregistration_2026-08-08.json")
LONG_CONFIG = Path("configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json")
SHORT_CONFIG = Path("configs/live/short_premium_panic_candidate.json")
SOURCE_BINDINGS = {
    str(LONG_CONFIG): "6533650bb6800308762dc02f310dbfe7dbd59c8a217d55305f6c5388eb2a480b",
    str(SHORT_CONFIG): "c9a6ef798c4834cd9eaf8cc6e522117a0e4a12e39c88a376b3da7bc1b0e39119",
    "training/backtest_all_alpha_month.py": "3de3bc013cfd880d1f14740eb9a51f0c3506949dc9a716a36c44f15123226fe6",
    "preprocessing/live_db_features.py": "a4b903913a51e1322e8946cd8dad8fea08b2361655520d20312a65f4219d5099",
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    long_cfg = json.loads(LONG_CONFIG.read_text())
    short_cfg = json.loads(SHORT_CONFIG.read_text())
    core = {
        "protocol_version": "oi_premium_asymmetric_volatility_relay_v1",
        "policy_id": "OIPAR-ASYM",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "Volatile pullbacks supported by persistent open interest are inventory-backed absorption and map long, while deep daily selloffs accompanied by extreme negative perp premium are deleveraging continuation and map short.",
            "why_distinct": "A single asymmetric inventory-versus-premium stress state machine; it is neither a Gross9 sleeve nor a direction/control repair of a terminal 2026-08-08 candidate.",
            "why_july_like": "The long state was active and profitable in the July 2026 volatility replay, while the short state is reserved for premium-panic selloffs.",
        },
        "frozen_states": {
            "long_inventory_absorption": {"source": str(LONG_CONFIG), "gates": long_cfg["signal"]["gates"], "stride_bars": long_cfg["signal"]["stride_bars_5m"], "hold_bars": long_cfg["signal"]["hold_bars_5m"], "side": 1},
            "short_premium_panic": {"source": str(SHORT_CONFIG), "gates": short_cfg["gates"], "stride_bars": short_cfg["stride_bars"], "hold_bars": short_cfg["hold_bars"], "side": -1},
            "conflict": "skip when both states are true on the same completed bar",
            "availability": "all features from completed 5m bars or backward-asof source observations; enter next 5m open",
            "global_reservation": "half-open; ignore new states while a trade is active",
            "no_threshold_side_hold_stride_tuning": True,
        },
        "clock": {"entry": "next completed 5m bar open", "long_hold": "8 elapsed hours", "short_hold": "12 elapsed hours", "split_crossing_action": "skip", "gross_exposure": 0.5},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "diagnostic_controls": {"names": ["long_state_only", "short_state_only", "no_long_range_vol_gate", "no_short_premium_z_gate", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "database_contract": {"env_file": "/home/pakchu/rllm/.env", "tables": ["bars_binance", "open_interest_binance", "funding_rates_binance", "premium_index_binance"], "symbol": "BTCUSDT", "read_only": True},
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {"component_outcomes_previously_known": True, "combined_candidate_outcomes_known": False, "candidate_incidence_opened": False, "post_entry_outcomes_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False},
        "stopping_rule": "terminal first failure; no repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    if payload["manifest_hash"] != canonical_hash({k: v for k, v in payload.items() if k != "manifest_hash"}):
        raise RuntimeError("OIPAR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"OIPAR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
