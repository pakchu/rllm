"""Outcome-blind preregistration for HVSVL-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVSVL-8"
SLUG = "high_volatility_safehaven_variation_leadership_relay"
FX = ("USDJPY", "USDCHF")
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1", "policy_id": POLICY_ID, "slug": SLUG, "as_of_date": "2026-08-13", "singleton": True,
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False,
        "mechanism": {
            "claim": "When completed London-session realized variation concentrates unusually in one of JPY or CHF, the dominant safe haven identifies the active regional deleveraging channel. During elevated BTC variation, follow the risk direction implied by that dominant currency for eight hours.",
            "side": "negative sign of the dominant pair's canonical safe-haven session return; safe-haven strengthening is BTC short and weakening is BTC long",
            "why_distinct": "HVSCRD selects a standardized return difference. HVFXBR requires directional agreement across safe havens and cyclicals. HVSVL instead selects a tail in the JPY-versus-CHF realized-variation concentration and uses only the ex-ante dominant pair's return sign. It has no return-spread threshold, common-dollar breadth, BTC direction, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "causal trailing BTC 24-hour realized variation must occupy its upper 35 percent",
            "why_low_gross9_overlap_is_plausible": "fixed weekday external-FX variation-leadership clocks at 13:05 UTC are absent from Gross9 primitives",
        },
        "features": {
            "fx_session": "exact observed bars_polygon one-minute rows during weekday [07:00,13:00) UTC",
            "session_valid": "each pair has at least 330 distinct minutes, first no later than 07:05, last no earlier than 12:55, finite positive coherent OHLC; no imputation",
            "pair_realized_variation": "sqrt(sum of squared one-minute log(close/open) returns) within the completed session; finite strict positive",
            "leadership_score": "absolute log(USDJPY realized variation / USDCHF realized variation)",
            "leadership_rank": "strict-prior midrank of leadership_score over at most 90 prior jointly valid sessions, minimum 60, current excluded; rank>=0.65",
            "dominant_pair": "USDJPY iff its realized variation is strictly greater than USDCHF; otherwise USDCHF iff strictly greater; ties invalid",
            "dominant_safehaven_return": "negative completed session log return of the dominant USD pair; finite strict nonzero",
            "btc_realized_variation": "sqrt(sum squared exact BTC one-minute log(close/open) returns over [D-24h,D))",
            "variation_rank": "strict-prior midrank over at most 90 prior source-valid weekday decisions, minimum 60, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {"decision": "exact weekday D 13:00 UTC", "entry": "exact BTCUSDT perpetual D 13:05 UTC open", "side": "negative dominant_safehaven_return sign", "hold": "8 elapsed hours", "reservation": "global chronological half-open; exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding": "not signal input; exact settlements only after novelty"},
        "policy": {"session_start_utc": "07:00", "session_end_utc": "13:00", "minimum_session_minutes": 330, "prior_sessions": 90, "minimum_prior_sessions": 60, "leadership_rank_min": 0.65, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "no_leadership_tail", "one_session_stale_leader", "subordinate_pair_direction", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"fx": {"table": "bars_polygon", "symbols": list(FX), "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]}, "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "completed_variation_only": True}, "read_after_preregistration_commit": True, "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_fx_family_outcomes_known": True, "repository_exact_safehaven_variation_leadership_event_found": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent cross-safehaven realized-variation migration mechanism"},
        "stopping_rule": "terminal first failure; no pair universe, session, validity, variation formula, rank, side, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVSVL prereg drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    payload = build(); validate(payload); args.output.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != body: raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(body); print(args.output)
