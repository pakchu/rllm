"""Preregister FPMR-1 before opening any post-entry return."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/funding_pressure_momentum_rotation_preregistration_2026-07-18.json"
)
DEFAULT_DOCS = Path(
    "docs/funding-pressure-momentum-rotation-preregistration-2026-07-18.md"
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


def protocol() -> dict[str, Any]:
    """Return the immutable FPMR-1 singleton protocol."""

    return {
        "protocol_version": "fpmr_v1_2026-07-18",
        "name": "FPMR-1 — Funding-Pressure Momentum Rotation",
        "claim": (
            "A cross-alt residual trend is more persistent when its relative funding "
            "pressure is easing, while a weak residual trend with worsening funding "
            "pressure continues to underperform over the following week."
        ),
        "evidence_boundary": {
            "dcrm_and_other_cross_alt_outcomes_seen": True,
            "exact_fpmr_score_or_post_entry_return_opened": False,
            "support_selection_uses_only_clock_density_and_concentration": True,
            "selection_2023_outcomes_opened": False,
            "test_2024_outcomes_opened": False,
            "eval_2025_outcomes_opened": False,
            "holdout_2026_outcomes_opened": False,
            "historical_pass_can_promote_live": False,
            "minimum_forward_shadow_days": 90,
        },
        "novelty_boundary": {
            "versus_dcrm": (
                "combines cross-sectional residual level, one-week residual rotation, "
                "and the change in a 28-day realized-funding burden; DCRM used only a "
                "30-day residual level plus dispersion sizing"
            ),
            "versus_afch": (
                "funding is a crowding-change predictor, not the harvested payoff; the "
                "pair is selected by price/funding interaction and held seven days"
            ),
            "versus_current_portfolio": (
                "one beta-neutral alt pair, no BTC leg, REX, OI, Kimchi, FX, DXY, "
                "premium-index gate, tree, Markov state, LLM, or manual regime"
            ),
        },
        "universe": {
            "venue": "Binance USD-M perpetual futures",
            "symbols": [
                "ETHUSDT",
                "SOLUSDT",
                "BNBUSDT",
                "XRPUSDT",
                "ADAUSDT",
                "DOGEUSDT",
            ],
            "position": "one long alt and one short alt; no BTC leg",
            "maximum_active_pairs": 1,
            "maximum_gross": 1.0,
        },
        "source_contract": {
            "market": "completed Binance USD-M 5m OHLC",
            "funding": "exact reported funding rate and settlement timestamp",
            "required_complete_symbols": 6,
            "support_prefix": "2023-01-01 through 2024-12-31",
            "no_forward_fill": True,
            "no_post_decision_row_in_signal": True,
        },
        "feature_formula": {
            "weekly_boundary": "Monday 00:00 UTC",
            "price_cutoff": "Sunday 23:55 UTC completed 5m close",
            "funding_cutoff": "Monday 00:00 UTC settlement, observed by 00:05",
            "momentum": "log(close[t-5m] / close[t-30d-5m])",
            "leave_one_out_factor": (
                "per-symbol median completed hourly return of the other five alts"
            ),
            "beta": {
                "lookback_completed_hours": 720,
                "minimum_completed_hours": 336,
                "shift_completed_hours": 1,
                "clip": [0.25, 2.5],
            },
            "residual_30d": "30-day return minus clipped beta times 30-day factor return",
            "residual_level_z": "same-boundary six-symbol population z-score",
            "residual_rotation": "current residual_level_z minus its value seven days earlier",
            "funding_current_28d": "sum for t-28d < settlement_time <= t",
            "funding_prior_28d": "sum for t-35d < settlement_time <= t-7d",
            "funding_pressure_change_z": (
                "same-boundary population z-score of current_28d minus prior_28d"
            ),
            "score": (
                "residual_level_z + residual_rotation - funding_pressure_change_z"
            ),
            "long_symbol": "lexically tie-broken maximum score",
            "short_symbol": "lexically tie-broken minimum score",
            "weights": (
                "positive gross-one weights satisfying long_beta*w_long = "
                "short_beta*w_short"
            ),
        },
        "clock": {
            "decision_time": "Monday 00:05 UTC after the 00:00 funding settlement",
            "entry_time": "Monday 00:10 UTC open",
            "exit_time": "next Monday 00:10 UTC open",
            "hold_days": 7,
            "nonoverlap": True,
            "same_boundary_order": "close old pair before opening the new pair",
        },
        "execution": {
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "cost_applied": "entry and exit on both legs by absolute notional",
            "funding_interval": "entry_time < funding_time <= exit_time",
            "strict_mdd": (
                "global/pre-entry HWM; entry, exit, and hypothetical liquidation costs; "
                "exact funding; favorable-before-adverse two-leg OHLC"
            ),
            "cagr": "full declared calendar including warm-up and idle cash",
            "tp_sl": None,
        },
        "support_gate": {
            "events_2023_2024_at_least": 90,
            "events_each_year_at_least": 45,
            "events_each_half_at_least": 20,
            "unique_ordered_pairs_at_least": 15,
            "maximum_ordered_pair_share_at_most": 0.15,
            "maximum_symbol_side_share_at_most": 0.40,
            "all_six_symbols_long_and_short_required": True,
            "outcome_columns_forbidden": True,
        },
        "selection_2023": {
            "singleton_no_parameter_ranking": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 15.0,
            "trades_at_least": 40,
            "each_half_absolute_return_positive": True,
            "each_half_trades_at_least": 18,
            "ten_bp_stress_absolute_return_positive": True,
            "entry_and_exit_delay_plus_5m_absolute_return_positive": True,
            "direction_flip_cagr_lower": True,
            "weekly_cluster_signflip_p_at_most": 0.10,
        },
        "sequential_oos": {
            "2024_opened_only_after_2023_pass": True,
            "2024_absolute_return_positive": True,
            "2024_cagr_to_strict_mdd_at_least": 3.0,
            "2024_strict_mdd_at_most_pct": 15.0,
            "2024_each_half_absolute_return_positive": True,
            "2025_opened_only_after_2024_pass": True,
            "2026_opened_only_after_2025_pass": True,
            "no_sign_weight_lookback_hold_or_score_repair": True,
        },
        "frozen_controls_not_repair_candidates": [
            "exact long-short direction flip",
            "entry and exit delayed five minutes",
            "ten basis points per notional side",
            "price-only residual level plus rotation",
            "funding-change-only cross-sectional rank",
            "static residual-level rank",
        ],
        "orthogonality_after_standalone_pass": {
            "compare_against": "all promoted/live/shadow sleeves on one marked ledger",
            "absolute_daily_pnl_pearson_at_most": 0.30,
            "absolute_weekly_pnl_pearson_at_most": 0.40,
            "synchronized_portfolio_marginal_improvement_required": True,
        },
        "stop_rule": (
            "Reject before outcomes if support fails. Otherwise freeze the strict "
            "evaluator, open 2023 exactly once, and retire on any gate failure without "
            "opening 2024 or changing the score, sign, hold, weights, or execution."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    protocol_hash = payload["protocol_hash"]
    return f"""# FPMR-1 preregistration — 2026-07-18

## Hypothesis

FPMR-1 is a weekly market-neutral alt pair. It buys the alt whose 30-day
beta-residual trend is both strong and improving while its relative 28-day
funding pressure is easing. It shorts the opposite profile. The score is a
fixed equal-coefficient sum of three cross-sectional weak signals; no outcome
threshold is fitted.

## Causal clock

- Price inputs end at Sunday 23:55 UTC.
- The Monday 00:00 funding settlement is allowed only after it is observed.
- Decision is Monday 00:05, entry is the 00:10 open, and exit is the following
  Monday 00:10 open.
- The two legs use gross-one factor-beta-neutral weights.

## Evidence boundary

Earlier cross-alt results are known, but this exact level + rotation - funding
change score has not been evaluated. Support may inspect only timestamps,
predictor values, pair identity, and concentration. The singleton strict
evaluator must be committed and hash-frozen before 2023 post-entry outcomes
are opened. A failed 2023 gate keeps 2024+ sealed.

## Qualification

2023 must have positive absolute return, full-calendar CAGR/strict-MDD at
least 3, strict MDD at most 15%, at least 40 trades, positive halves, positive
10 bp/side stress, and weekly-cluster p <= 0.10. Controls are falsification
only and cannot repair the primary policy.

Protocol hash: `{protocol_hash}`
"""


def run(
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    frozen = protocol()
    payload = {
        "protocol": frozen,
        "protocol_hash": canonical_hash(frozen),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--docs-output", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
