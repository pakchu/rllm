"""Outcome-blind preregistration for HVSCRD-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVSCRD-8"
SLUG = "high_volatility_safehaven_relative_carry_dislocation_relay"
FX = ("USDJPY", "USDCHF")
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "A large completed London-session divergence between JPY and CHF safe-haven strength isolates yen-funded carry repricing from a broad dollar or generic safe-haven move. In elevated BTC variation, BTC follows the relative carry-risk direction for eight hours.",
            "side": "negative strict sign of standardized JPY safe-haven strength minus standardized CHF safe-haven strength",
            "why_distinct": "Dollar breadth and FX synchronization retain a common-dollar factor. The safe-haven/cyclical barbell requires broad agreement across four currencies. HVSCRD instead differences two safe havens, removes their common component, and selects a tail in the relative JPY-versus-CHF dislocation. It uses no BTC return direction, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "causal trailing BTC 24-hour realized variation must occupy its upper 35 percent",
            "why_low_gross9_overlap_is_plausible": "one fixed weekday external relative-safe-haven dislocation clock at 13:05 UTC is absent from Gross9 primitives",
        },
        "features": {
            "fx_session": "exact observed bars_polygon one-minute rows during weekday [07:00,13:00) UTC",
            "session_valid": "each pair has at least 330 distinct minutes, first no later than 07:05, last no earlier than 12:55, finite positive coherent OHLC; no imputation",
            "safehaven_returns": {"USDJPY": "negative session log return", "USDCHF": "negative session log return"},
            "pair_zscores": "each safe-haven return standardized against at most 90 strictly prior jointly valid sessions, minimum 60, current excluded; prior sample std strict positive",
            "relative_dislocation": "USDJPY safe-haven z-score minus USDCHF safe-haven z-score; finite strict nonzero",
            "dislocation_rank": "strict-prior midrank of absolute relative_dislocation over at most 90 prior jointly valid sessions, minimum 60, current excluded; rank>=0.75",
            "btc_realized_variation": "sqrt(sum squared exact BTC one-minute log(close/open) returns over [D-24h,D))",
            "variation_rank": "strict-prior midrank over at most 90 prior source-valid weekday decisions, minimum 60, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact weekday D 13:00 UTC after FX session and BTC variation are complete",
            "entry": "exact BTCUSDT perpetual D 13:05 UTC open",
            "side": "negative relative_dislocation sign",
            "hold": "8 elapsed hours",
            "reservation": "fixed weekday decisions; global chronological half-open, exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements only after novelty",
        },
        "policy": {
            "session_start_utc": "07:00",
            "session_end_utc": "13:00",
            "minimum_session_minutes": 330,
            "prior_sessions": 90,
            "minimum_prior_sessions": 60,
            "dislocation_rank_min": 0.75,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "no_dislocation_tail", "raw_return_difference", "one_session_stale_dislocation", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"fx": {"table": "bars_polygon", "symbols": list(FX), "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]}, "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "completed_variation_only": True}, "read_after_preregistration_commit": True, "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_fx_family_outcomes_known": True, "repository_exact_relative_safehaven_dislocation_event_found": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent common-factor-cancelled JPY carry-dislocation mechanism"},
        "stopping_rule": "terminal first failure; no pair universe, session, validity, standardization, dislocation rank, variation, side, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVSCRD prereg drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != body:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(body)
    print(args.output)
