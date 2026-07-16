"""Freeze LORC v1 before any calendar-2025 strategy outcome is opened."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/leave_one_out_residual_continuation_v1_preregistration_2026-07-17.json"
DEFAULT_DOCS = "docs/leave-one-out-residual-continuation-v1-preregistration-2026-07-17.md"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def protocol() -> dict[str, Any]:
    symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
    return {
        "protocol_version": "lorc_v1_2026-07-17",
        "name": "LORC — Leave-One-Out Residual Continuation",
        "claim": (
            "An extreme cross-sectional alt-perpetual residual that is not confirmed by "
            "aggressor flow represents informed or inventory-driven repricing that continues "
            "for the next 12 hours; buy the residual winner and short the residual loser with "
            "ex-ante factor-beta-neutral weights."
        ),
        "evidence_boundary": {
            "research_2023_2024_opened": True,
            "research_2023_2024_contaminated_for_confirmation": True,
            "holdout_2025_post_entry_returns_opened": False,
            "final_2026_post_entry_returns_opened": False,
            "single_policy_frozen_before_2025": True,
            "repository_wide_human_pristine_claim": False,
        },
        "hypothesis_origin": {
            "parent_family": "LORE v1 mean reversion",
            "parent_decision": "rejected_before_2025_holdout",
            "parent_selection_result_hash": "67d3defc3f373d698012bfdeb3dbdf76539491a3c08ee4d5a0ef208794bc3e4b",
            "origin_control": "preregistered exact direction flip on the frozen L03 clock",
            "no_additional_2023_2024_threshold_or_pair_search": True,
            "research_stats": {
                "2023": {
                    "absolute_return_pct": 38.291954,
                    "cagr_pct": 38.322661,
                    "strict_mdd_pct": 22.933691,
                    "cagr_to_strict_mdd": 1.671022,
                    "trades": 107,
                },
                "2024": {
                    "absolute_return_pct": 110.659103,
                    "cagr_pct": 110.337657,
                    "strict_mdd_pct": 11.773264,
                    "cagr_to_strict_mdd": 9.371882,
                    "trades": 117,
                },
                "combined_2023_2024": {
                    "absolute_return_pct": 191.324663,
                    "cagr_pct": 70.619968,
                    "strict_mdd_pct": 22.933691,
                    "cagr_to_strict_mdd": 3.079311,
                    "trades": 224,
                    "weekly_cluster_signflip_raw_p": 0.002099895,
                },
                "ten_bp_cost_stress": {
                    "absolute_return_pct": 143.589407,
                    "cagr_pct": 56.025989,
                    "strict_mdd_pct": 24.662704,
                    "cagr_to_strict_mdd": 2.271710,
                },
            },
        },
        "universe": {
            "venue": "Binance USD-M perpetual futures",
            "symbols": symbols,
            "trading_target": "two-leg alt-perpetual pair; no BTC position",
            "benchmark_only": "BTCUSDT perpetual 5m bars for daily beta/correlation",
            "structural_orthogonality": (
                "cross-sectional market-neutral alt residual continuation; no BTC REX, BTC OI, "
                "funding/premium gate, Kimchi/FX, Markov state, tree model, or LLM prediction"
            ),
        },
        "source_contract": {
            "physical_2025_prefix": ["2024-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "warmup_before_holdout": "2024 only; no post-2025 row may be written",
            "holdout": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "base_interval": "5m bar-open timestamps",
            "required_columns": [
                "date", "open", "high", "low", "close", "quote_asset_volume",
                "taker_buy_quote", "tic",
            ],
            "completed_hour": (
                "exactly 12 unique 5m bars in [hour_open, hour_open+1h); feature available "
                "only at the right boundary"
            ),
            "all_six_symbols_required": True,
            "no_fill_or_nearest_join": True,
            "funding": "exact symbol funding timestamps; no synthetic or forward-filled rate",
        },
        "feature_formula": {
            "hourly_return": "log(hour_close_t/hour_close_t-1)",
            "leave_one_out_factor": "cross-sectional median hourly return of the other five symbols",
            "beta": {
                "formula": "rolling cov(asset_return, loo_factor) / rolling var(loo_factor)",
                "lookback_hours": 720,
                "minimum_hours": 336,
                "shift_hours": 1,
                "clip": [0.25, 2.5],
            },
            "residual_horizon_hours": 12,
            "residual": "log(close_t/close_t-12h) - beta_t * sum(loo_factor over completed 12h)",
            "residual_z": {"lookback_hours": 2160, "minimum_hours": 1080, "history_shift_hours": 1},
            "flow": "2*sum(taker_buy_quote over 12h)/sum(quote_asset_volume over 12h)-1",
            "flow_z": {"lookback_hours": 2160, "minimum_hours": 1080, "history_shift_hours": 1},
            "winner": "argmax residual_z with lexical symbol tie-break",
            "loser": "argmin residual_z with lexical symbol tie-break",
        },
        "frozen_policy": {
            "policy_id": "LORC01",
            "residual_horizon_hours": 12,
            "hold_hours": 12,
            "winner_residual_z_at_least": 1.5,
            "loser_residual_z_at_most": -1.5,
            "winner_price_minus_flow_z_at_least": 1.0,
            "loser_flow_minus_price_z_at_least": 1.0,
            "long_leg": "winner",
            "short_leg": "loser",
            "weight_formula": {
                "long_abs": "loser_beta/(winner_beta+loser_beta)",
                "short_abs": "winner_beta/(winner_beta+loser_beta)",
                "minimum_leg_weight": 0.25,
                "gross": 1.0,
                "target_factor_beta": 0.0,
            },
            "reservation": "start flat at 2025-01-01; first eligible 2025 signal reserves through exit",
        },
        "execution": {
            "signal_time": "right edge of completed hourly bar",
            "entry": "5m open at signal_time + 5m",
            "exit": "5m open at entry_time + 12h",
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "funding_interval": "entry_time < funding_time <= exit_time",
            "funding_cash": "-signed_leg_notional * funding_rate at causal mark",
            "strict_mdd": (
                "global/pre-entry HWM; entry cost first; each held bar marks simultaneous "
                "long-high/short-low favorable before simultaneous long-low/short-high adverse; "
                "funding debits before adverse, credits cannot raise intratrade peak; hypothetical "
                "liquidation cost at every mark; scheduled exit cost"
            ),
            "cagr": "full declared wall-clock including idle time",
            "tp_sl": None,
        },
        "support_gate_2025": {
            "events_at_least": 60,
            "h1_events_at_least": 25,
            "h2_events_at_least": 25,
            "unique_ordered_pairs_at_least": 10,
            "maximum_ordered_pair_share": 0.15,
            "symbols_seen_each_side_at_least": 5,
            "monthly_source_quarantine_at_most": 0.01,
            "outcome_columns_forbidden": True,
        },
        "holdout_2025": {
            "single_confirmatory_hypothesis": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 15.0,
            "h1_absolute_return_positive": True,
            "h2_absolute_return_positive": True,
            "trades_at_least": 60,
            "ten_bp_cost_stress_absolute_return_positive": True,
            "weekly_cluster_signflip_p_at_most": 0.10,
            "entry_delay_plus_one_5m_bar_absolute_return_positive": True,
        },
        "orthogonality_2025": {
            "daily_mark_to_market_returns": "UTC calendar-day close-to-close strategy equity",
            "absolute_pearson_to_btc_daily_return_at_most": 0.30,
            "absolute_ols_beta_to_btc_daily_return_at_most": 0.15,
            "nonzero_strategy_days_at_least": 60,
            "comparison_to_current_portfolio_after_strategy_pass": True,
            "marginal_portfolio_improvement_required_for_promotion": True,
        },
        "final_2026": {
            "opened_only_after_all_2025_strategy_gates_pass": True,
            "available_window": ["2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z"],
            "absolute_return_positive": True,
            "annualized_cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 15.0,
            "trades_at_least": 25,
            "ten_bp_cost_stress_absolute_return_positive": True,
        },
        "diagnostics_not_repair_candidates": [
            "mean-reversion direction on the same frozen clock",
            "equal 50/50 legs",
            "entry delayed one hour",
            "seven-day clock shift",
            "monthly ordered-pair permutation",
        ],
        "stop_rule": (
            "Reject before 2026 if the exact single LORC01 policy fails any strategy gate in "
            "calendar 2025. No threshold, direction, hold, pair whitelist, leverage, or regime "
            "repair is allowed after 2025 is opened. Orthogonality is reported independently "
            "and is required for portfolio promotion."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    return f"""# LORC v1 preregistration — 2026-07-17

## Evidence boundary

LORC is an explicitly **derived** family. The 2023–2024 LORE direction-flip
diagnostic is research/selection data and is not claimed as confirmation.
Calendar 2025 post-entry returns remain unopened and are the first single-policy
confirmatory test. Calendar 2026 remains sealed unless every 2025 strategy gate
passes.

## Orthogonal mechanism

LORC holds no BTC position. It removes a leave-one-out alt factor, then buys the
12-hour residual winner and shorts the residual loser only when taker flow fails
to confirm both tails. Cross-leg weights neutralize the causal rolling factor
beta at gross 1.0. The mechanism and data source are distinct from BTC REX/OI,
funding/premium, Kimchi/FX, Markov, tree, and LLM alphas.

## Frozen single policy

- six Binance USD-M alts: ETH, SOL, BNB, XRP, ADA, DOGE;
- residual/flow horizon: 12 hours;
- residual z tails: winner >= 1.5 and loser <= -1.5;
- price/flow disagreement on both tails >= 1.0 z;
- long winner, short loser, factor-beta-neutral gross 1.0;
- entry: completed-hour signal + 5m open; fixed 12h exit;
- 6 bp/notional/side base cost, 10 bp stress, exact funding;
- full-calendar CAGR and global favorable-before-adverse strict MDD.

No second 2025 policy is available, so no hold/threshold/sign ranking can occur.

## Research evidence, not confirmation

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | +38.292% | +38.323% | 22.934% | 1.671 | 107 |
| 2024 | +110.659% | +110.338% | 11.773% | 9.372 | 117 |
| 2023–2024 | +191.325% | +70.620% | 22.934% | 3.079 | 224 |
| 2023–2024, 10 bp | +143.589% | +56.026% | 24.663% | 2.272 | 224 |

The 20,000-draw weekly-cluster sign-flip p-value was 0.00210. These numbers
only justify spending untouched 2025; they do not count as OOS evidence.

## 2025 pass contract

Calendar 2025 must have positive absolute return, CAGR/strict-MDD >= 3.0,
strict MDD <= 15%, positive H1 and H2, >= 60 trades, positive 10-bp stress,
weekly-cluster sign-flip p <= 0.10, and positive +5m entry-delay performance.
Daily mark-to-market correlation to BTC must be <= 0.30 in absolute value and
absolute BTC beta <= 0.15 for portfolio promotion.

Protocol hash: `{payload['protocol_hash']}`
"""


def run(output: str = DEFAULT_OUTPUT, docs_output: str = DEFAULT_DOCS) -> dict[str, Any]:
    frozen = protocol()
    payload = {
        "protocol": frozen,
        "protocol_hash": canonical_hash(frozen),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).write_text(markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
