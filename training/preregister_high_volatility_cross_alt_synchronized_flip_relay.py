"""Outcome-blind preregistration for HVCASFR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVCASFR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_alt_synchronized_flip_relay_preregistration_2026-08-12.json"
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
        "protocol_version": "high_volatility_cross_alt_synchronized_flip_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-12",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When at least four of six liquid alt perpetuals reverse their completed "
                "first-half direction in the second half, and the cross-sectional majority "
                "itself flips, the crypto market has undergone a broad directional reset "
                "rather than an idiosyncratic retracement. During elevated BTC variation, "
                "follow the new second-half alt majority in BTC for eight elapsed hours."
            ),
            "side": "strict sign of the six-alt second-half return majority",
            "why_distinct": (
                "HVCARP compares continuous cross-alt leadership ranks across halves and "
                "requires persistence; HVCACFR compares changing BTC-alt return correlation; "
                "HVMRSR compares BTC minute-return medians. HVCASFR instead requires a discrete "
                "market-wide reversal: opposite first- and second-half cross-sectional majorities "
                "plus at least four symbol-level sign flips. It uses no BTC return direction, "
                "return magnitude threshold, volume, flow, funding, OI, premium, fitted outcome, "
                "prior event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "BTC completed trailing twenty-four-hour realized variation must rank in its "
                "causal upper 35%, targeting July-like unstable directional resets."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "offset three-daily synchronized six-alt half-block sign-flip events are absent "
                "from Gross9 primitives"
            ),
        },
        "features": {
            "alts": ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            "block": "exact 480 one-minute rows [D-8h,D) for every alt, split into two exact 240-minute halves",
            "half_return": "log(last minute close / first minute open) in each half, finite and strict nonzero",
            "half_majority": "strict sign of the sum of six half-return signs; a 3-versus-3 tie is invalid",
            "majority_flip": "first-half and second-half majority signs must be exact opposites",
            "symbol_flip_count": "count of alts whose first- and second-half return signs are exact opposites; at least four",
            "btc_variation": "sum squared log(close/open) over exact 1440 BTCUSDT one-minute rows [D-24h,D)",
            "variation_rank": "strict-prior midrank against at most 270 prior source-valid decisions; minimum 180; current excluded; rank at least 0.65",
            "trigger": "the synchronized majority-and-symbol flip state at each fixed decision; no onset filter",
            "invalid": "missing, duplicate, nonpositive, inconsistent OHLC, zero half return, tied half majority, or incomplete exact grid; no imputation",
        },
        "clock": {
            "decision": "exact 01:00, 09:00, and 17:00 UTC after the completed eight-hour block",
            "entry": "exact BTCUSDT open five elapsed minutes after decision",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "block_minutes": 480,
            "half_minutes": 240,
            "history_decisions": 270,
            "minimum_history_decisions": 180,
            "minimum_symbol_flips": 4,
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
            "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_variation_gate",
                "three_symbol_flips",
                "majority_flip_only",
                "one_block_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "table": "bars_binance",
            "interval": "1m",
            "symbols": ["BTCUSDT", "ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            "columns": ["ts", "symbol", "open", "high", "low", "close"],
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_only": True,
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "repository_collision_search_opened": True,
            "exact_cross_alt_synchronized_half_sign_flip_candidate_found": False,
            "prior_cross_alt_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_controls_promoted": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "selection_basis": "independent discrete market-wide directional-reset mechanism",
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final "
            "strict economics, then RV20 q90 audit; no alt set, block, half, return, majority, "
            "flip count, history, variation, side, hold, clock, subset, threshold, source, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    payload = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != payload:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(json.dumps({"policy_id": POLICY_ID, "output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
