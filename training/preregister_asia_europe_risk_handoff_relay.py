"""Outcome-blind preregistration for AERHR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/asia_europe_risk_handoff_relay_preregistration_2026-08-10.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "asia_europe_risk_handoff_relay_v1",
        "policy_id": "AERHR-12",
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When the completed first four hours of European trading reverse and fully "
                "dominate the preceding eight-hour Asia BTC displacement, price discovery has "
                "transferred between regional liquidity regimes rather than merely retraced. In "
                "a high-variation twelve-hour path, follow the European handoff direction."
            ),
            "side": "strict sign of the completed 08:00-12:00 UTC BTC return",
            "why_distinct": (
                "ASCR followed concordant halves within Asia and is terminal; AERHR requires a "
                "cross-session sign reversal whose shorter European opening move exceeds the "
                "entire Asia move. It is not a threshold, side, hold, clock, subset, or control "
                "repair of ASCR, a US cash-session acceptance rule, or a block-confirmation relay."
            ),
            "volatile_market_target": (
                "the completed twelve-hour one-minute variation must rank in its causal upper "
                "35%; RV20 q90 remains only a post-stage audit"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one conditional 12:05 UTC regional-handoff clock is absent from Gross9 primitives"
            ),
        },
        "features": {
            "asia_window": "exact 480 BTCUSDT one-minute bars [00:00,08:00) UTC",
            "europe_open_window": "exact 240 BTCUSDT one-minute bars [08:00,12:00) UTC",
            "asia_return": "log(07:59 close/00:00 open), strict nonzero",
            "europe_return": "log(11:59 close/08:00 open), strict nonzero",
            "handoff": "returns have opposite signs and abs(europe_return)>=abs(asia_return)",
            "path_variation": (
                "sqrt(sum squared one-minute log(close/open)) over exact [00:00,12:00) UTC"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 prior valid daily paths, minimum 180, "
                "current excluded; rank>=0.65"
            ),
            "source_valid": (
                "exact unique minute grids, coherent finite positive OHLC, no imputation"
            ),
        },
        "clock": {
            "decision": "daily 12:00 UTC after both regional windows complete",
            "entry": "exact BTCUSDT 12:05 UTC open",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium_rv20": (
                "not signal inputs; exact funding only after novelty; RV20 q90 only after all "
                "economic stages pass"
            ),
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
            "absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True,
            "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "market": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": ["no_variation_gate", "any_reversal", "asia_continuation", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "database_metadata_only_opened_before_preregistration": True,
            "market_values_used_to_select_rule": False, "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False,
            "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent regional-liquidity handoff mechanism plus volatile-market focus",
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no window, dominance, variation, side, hold, clock, subset, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(registration: dict[str, Any]) -> None:
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if (
        registration.get("manifest_hash") != canonical_hash(core)
        or registration.get("outcomes_opened") is not False
        or registration.get("source_incidence_opened") is not False
        or registration.get("gross9_rows_opened") is not False
    ):
        raise RuntimeError("AERHR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
