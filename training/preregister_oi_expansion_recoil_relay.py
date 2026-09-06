"""Outcome-blind preregistration for OIER-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "OIER-8"
DEFAULT_OUTPUT = Path(
    "results/oi_expansion_recoil_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "oi_expansion_recoil_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "When four-hour open interest expands unusually during a large volatile BTC move and RSI confirms the same price extreme, newly added directional inventory is vulnerable to recoil; trade opposite the price shock for eight elapsed hours.",
            "side": "negative strict sign of completed four-hour BTC return z-score",
            "why_distinct": "OIER is one symmetric two-sided OI-expansion recoil rule. It does not combine a known long clock with an unrelated short family, does not promote an OIPAR control, and uses neither premium panic nor future-selected direction. Exact symmetric incidence and outcomes are unknown.",
            "volatile_market_target": "range volatility and absolute four-hour price shock are mandatory fixed gates; causal RV20 q90 remains only a later audit",
            "why_low_gross9_overlap_is_plausible": "30-minute stride, rare joint OI/range/price/RSI state, and eight-hour reservation create a sparse inventory clock absent from Gross9",
        },
        "features": {
            "schedule": "completed five-minute bars on the frozen 30-minute research-offset stride",
            "oi_expansion": "oi_ret_4h_z >= 0.8954018630586817 with OI backward-asof and delayed to completed-bar availability",
            "price_shock": "abs(px_ret_4h_z) >= 0.7389570664259131 and px_ret_4h_z strict nonzero",
            "range_volatility": "range_vol >= 0.04008415457867338",
            "rsi_confirmation": "side*rsi_norm <= -0.04507656773717145, where side=-sign(px_ret_4h_z)",
            "eligible_state": "all four gates true; no rank refit, grid, or directional threshold asymmetry",
            "availability": "all market features use completed five-minute bars; OI is backward-asof and delayed one complete bar",
            "side": "-sign(px_ret_4h_z)",
            "source_valid": "finite coherent BTC bars and finite delayed OI-derived features; missing OI fails closed",
        },
        "rv20_stress_slice": {
            "rv20": "sqrt(365*mean exact daily returns^2 over t-20 through t-1)",
            "threshold": "numpy linear q90 over 756 strictly prior available RV20 observations",
            "entry_filter": False,
            "future_use": "only after all sequential full-calendar stages pass",
        },
        "clock": {
            "entry": "next exact five-minute BTCUSDT open after the completed decision bar",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty",
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
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "persistent_long_vol_comparator": "same accepted clock and 0.5 gross, side forced long",
            "full_calendar_decomposition": "candidate minus comparator net return",
            "rv20_q90_decomposition": "same decomposition on causal RV20 q90 decisions",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "candidate_specific_q90_residual_positive": True,
            "comparator_cannot_satisfy_candidate_claim": True,
        },
        "diagnostic_controls": {
            "definitions": {
                "no_range_vol_gate": "symmetric primary without the range-volatility gate",
                "no_rsi_gate": "symmetric primary without RSI confirmation",
                "one_stride_stale_features": "primary state and side shifted one 30-minute stride",
                "direction_flip": "negative primary side",
                "same_clock_forced_long": "same primary clock with side forced long",
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "database": {
                "env_file": "/home/pakchu/rllm/.env",
                "tables": ["bars_binance", "open_interest_binance"],
                "symbol": "BTCUSDT",
                "read_only": True,
                "feature_builder": "preprocessing/live_db_features.py via training/backtest_all_alpha_month.py",
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_long_only_oi_pullback_outcomes_known": True,
            "prior_oipar_asymmetric_source_failure_known": True,
            "numeric_cutoffs_inherited_without_change": True,
            "exact_symmetric_oier_incidence_known": False,
            "exact_oier_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "terminal first failure; no block predicate, threshold, confirmation, embargo, "
            "side, clock, hold, RV20, subset, comparator, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("OIER preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
