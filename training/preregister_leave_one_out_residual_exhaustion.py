"""Freeze the LORE v1 hypothesis before any post-entry outcome is opened."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/leave_one_out_residual_exhaustion_v1_preregistration_2026-07-17.json"
DEFAULT_DOCS = "docs/leave-one-out-residual-exhaustion-v1-preregistration-2026-07-17.md"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def protocol() -> dict[str, Any]:
    symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
    policies = [
        {"policy_id": "L01", "residual_horizon_hours": 6, "hold_hours": 12},
        {"policy_id": "L02", "residual_horizon_hours": 6, "hold_hours": 24},
        {"policy_id": "L03", "residual_horizon_hours": 12, "hold_hours": 12},
        {"policy_id": "L04", "residual_horizon_hours": 12, "hold_hours": 24},
    ]
    return {
        "protocol_version": "lore_v1_2026-07-17",
        "name": "LORE — Leave-One-Out Residual Exhaustion",
        "claim": (
            "A cross-sectional alt-perpetual residual extreme that is not confirmed by "
            "aggressor flow is transient; buy the residual loser and short the residual "
            "winner with ex-ante factor-beta-neutral weights."
        ),
        "evidence_boundary": {
            "support_only_before_selector": True,
            "post_entry_returns_opened": False,
            "selection_source_end_exclusive": "2025-01-01T00:00:00Z",
            "holdout_2025_opened": False,
            "final_2026_opened": False,
            "repository_wide_human_pristine_claim": False,
            "exact_family_algorithmically_unopened_claim": True,
        },
        "universe": {
            "venue": "Binance USD-M perpetual futures",
            "symbols": symbols,
            "trading_target": "two-leg alt-perpetual pair; no BTC position",
            "current_portfolio_exclusions": [
                "BTC REX", "BTC OI", "BTC funding/premium", "Kimchi/FX", "BTC Markov",
                "BTC ExtraTrees", "Coinbase lead", "CVTT", "TAAR",
            ],
        },
        "source_contract": {
            "base_interval": "5m bar-open timestamps",
            "required_columns": [
                "date", "open", "high", "low", "close", "quote_asset_volume",
                "taker_buy_quote", "tic",
            ],
            "completed_hour": (
                "exactly 12 unique 5m bars in [hour_open, hour_open+1h); hourly feature "
                "becomes available at the right boundary"
            ),
            "no_fill_or_nearest_join": True,
            "all_six_symbols_required": True,
            "funding": "exact symbol funding timestamps; no synthetic or forward-filled rate",
            "physical_selection_prefix_required": True,
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
            "residual": (
                "log(close_t/close_t-h) - beta_t * sum(loo_factor over the completed h hours)"
            ),
            "residual_z": {
                "lookback_hours": 2160,
                "minimum_hours": 1080,
                "history_shift_hours": 1,
            },
            "flow": "2*sum(taker_buy_quote over h)/sum(quote_asset_volume over h)-1",
            "flow_z": {
                "lookback_hours": 2160,
                "minimum_hours": 1080,
                "history_shift_hours": 1,
            },
            "winner": "argmax residual_z with lexical symbol tie-break",
            "loser": "argmin residual_z with lexical symbol tie-break",
        },
        "signal": {
            "winner_residual_z_at_least": 1.5,
            "loser_residual_z_at_most": -1.5,
            "winner_price_minus_flow_z_at_least": 1.0,
            "loser_flow_minus_price_z_at_least": 1.0,
            "long_leg": "loser",
            "short_leg": "winner",
            "weight_formula": {
                "long_abs": "winner_beta/(winner_beta+loser_beta)",
                "short_abs": "loser_beta/(winner_beta+loser_beta)",
                "minimum_leg_weight": 0.25,
                "gross": 1.0,
                "target_factor_beta": 0.0,
            },
            "ties": "lexicographically first symbol after sorting columns",
            "reservation": "policy-local first eligible signal reserves through scheduled exit",
        },
        "policies": policies,
        "execution": {
            "signal_time": "right edge of completed hourly bar",
            "entry": "5m open at signal_time + 5m",
            "exit": "5m open at entry_time + hold_hours",
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "funding_interval": "entry_time < funding_time <= exit_time",
            "funding_cash": "-signed_leg_notional * funding_rate at each settlement",
            "strict_mdd": (
                "global/pre-entry HWM; entry cost first; each held bar marks simultaneous "
                "long-high/short-low favorable point before simultaneous long-low/short-high "
                "adverse point; funding debits before adverse, credits cannot raise intratrade "
                "peak; hypothetical liquidation cost at every mark; scheduled exit costs"
            ),
            "cagr": "full declared wall-clock including idle time",
            "tp_sl": None,
        },
        "support_gate": {
            "combined_events_at_least": 150,
            "each_year_events_at_least": 60,
            "unique_ordered_pairs_at_least": 10,
            "maximum_ordered_pair_share": 0.15,
            "symbols_seen_as_long_at_least": 5,
            "symbols_seen_as_short_at_least": 5,
            "monthly_source_quarantine_at_most": 0.01,
            "outcome_columns_forbidden": True,
        },
        "selection": {
            "fit": ["2023-01-01", "2024-01-01"],
            "test": ["2024-01-01", "2025-01-01"],
            "combined": ["2023-01-01", "2025-01-01"],
            "half_years": ["2023H1", "2023H2", "2024H1", "2024H2"],
            "ranking": (
                "passing policies only; maximize minimum annual CAGR/strict-MDD, then "
                "combined ratio, then lower MDD, then policy_id"
            ),
            "gates": {
                "each_year_absolute_return_positive": True,
                "each_year_cagr_to_strict_mdd_at_least": 1.5,
                "positive_half_years_at_least": 3,
                "combined_cagr_to_strict_mdd_at_least": 3.0,
                "combined_strict_mdd_at_most_pct": 12.0,
                "combined_trades_at_least": 150,
                "each_year_trades_at_least": 60,
                "ten_bp_cost_stress_positive": True,
                "bonferroni_weekly_signflip_p_at_most": 0.10,
            },
            "multiple_testing_hypotheses": len(policies),
        },
        "holdout_2025": {
            "opened_only_after_policy_commit": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 10.0,
            "h1_h2_nonnegative": True,
            "trades_at_least": 60,
            "ten_bp_cost_stress_positive": True,
            "entry_delay_plus_one_5m_bar_positive": True,
        },
        "final_2026": {
            "opened_only_after_2025_pass": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 10.0,
            "trades_at_least": 25,
            "ten_bp_cost_stress_positive": True,
        },
        "diagnostic_controls_not_selection_candidates": [
            "exact direction flip on reserved clock",
            "remove flow-disconfirmation but retain residual thresholds",
            "raw-return winner/loser without leave-one-out residualization",
            "equal 50/50 legs without beta neutrality",
            "entry delayed one additional hour",
            "candidate clock shifted by seven days",
            "ordered-pair labels permuted within calendar month",
        ],
        "portfolio_orthogonality_after_holdout": {
            "absolute_daily_pnl_pearson_at_most": 0.30,
            "absolute_daily_btc_return_beta_at_most": 0.10,
            "nonzero_pnl_days_at_least": 60,
            "marginal_portfolio_improvement_required": True,
            "comparison": "current live anchor and full frozen rank-1 shadow portfolio",
        },
        "stop_rule": (
            "Reject before 2025 if no policy passes selection. Reject before 2026 if the "
            "single frozen policy fails 2025. No post-outcome sign flip, threshold repair, "
            "hold replacement, pair whitelist, or regime gate is allowed under LORE v1."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    p = payload["protocol"]
    rows = "\n".join(
        f"| {x['policy_id']} | {x['residual_horizon_hours']}h | {x['hold_hours']}h |"
        for x in p["policies"]
    )
    return f"""# LORE v1 preregistration — 2026-07-17

## Evidence boundary

**Support-only. No LORE post-entry return has been opened.** The physical
selection prefix ends before 2025. Calendar 2025 is opened only after one
2023–2024 policy is committed; 2026 is opened only after that policy passes
2025. Repository-wide human-pristine status is not claimed.

## Orthogonal mechanism

LORE trades six Binance USD-M alt perpetuals, not BTC. It removes a causal
leave-one-out crypto factor from each asset, identifies the residual winner and
loser, and acts only when aggressor flow fails to confirm both residual tails.
It buys the loser and shorts the winner with ex-ante factor-beta-neutral,
gross-one weights. It uses no BTC REX, OI, funding/premium gate, Kimchi/FX,
Markov state, or LLM prediction.

## Frozen policies

| Policy | Residual horizon | Hold |
|---|---:|---:|
{rows}

The signal requires winner residual z >= 1.5, loser residual z <= -1.5,
winner residual-minus-flow z >= 1.0, and loser flow-minus-residual z >= 1.0.
Betas use a shifted 720-hour rolling estimate; residual and flow z-scores use
only the prior shifted 2,160 hours. Every completed hour requires all 12 five-
minute bars for all six symbols, with no fill or nearest join.

## Execution and strict risk

- signal: right edge of a completed hour;
- entry: signal + 5 minutes open;
- exit: entry + fixed 12h or 24h;
- base cost: 6 bp/notional/side; stress: 10 bp;
- exact per-symbol funding for `entry < funding_time <= exit`;
- full-calendar CAGR;
- strict MDD uses global/pre-entry HWM, entry and hypothetical liquidation
  costs, funding debit/credit ordering, and simultaneous conservative
  long-high/short-low favorable then long-low/short-high adverse marks.

## Selection and holdout

2023 fit and 2024 test select at most one of four policies. A policy needs
positive return and ratio >= 1.5 in each year, at least three positive halves,
combined CAGR/strict-MDD >= 3, strict MDD <= 12%, >= 150 combined trades,
10-bp cost survival, and Bonferroni weekly sign-flip p <= 0.10. The frozen
winner then needs 2025 ratio >= 3 and strict MDD <= 10% before 2026 is opened.

## Anti-repair rule

If no policy passes, LORE v1 ends without opening 2025. Direction flip,
threshold changes, pair whitelists, alternate holds, and regime gates are
diagnostics only and cannot rescue the family.

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
