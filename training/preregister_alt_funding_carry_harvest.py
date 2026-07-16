"""Freeze AFCH v1 before exact multi-sleeve carry PnL is opened."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/alt_funding_carry_harvest_v1_preregistration_2026-07-17.json"
DEFAULT_DOCS = "docs/alt-funding-carry-harvest-v1-preregistration-2026-07-17.md"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def protocol() -> dict[str, Any]:
    symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
    return {
        "protocol_version": "afch_v1_2026-07-17",
        "name": "AFCH — Alt Funding Carry Harvest",
        "claim": (
            "Cross-sectional perpetual funding dispersion is persistent enough that a "
            "factor-beta-neutral long of the lowest-funding alt and short of the highest-"
            "funding alt can harvest exact funding cashflow over four weeks after costs."
        ),
        "evidence_boundary": {
            "source_rows_broadly_research_seen": True,
            "adjacent_funding_score_price_pnl_seen": True,
            "exact_afch_cashflow_portfolio_pnl_opened": False,
            "fit_2023_opened": False,
            "test_2024_opened": False,
            "eval_2025_opened": False,
            "final_2026_opened": False,
            "repository_wide_human_pristine_claim": False,
            "exact_family_algorithmically_unopened_claim": True,
            "forward_shadow_required_for_promotion": True,
        },
        "novelty_boundary": {
            "different_from_lore_lorc": "funding cashflow is the payoff object; no residual/flow price direction",
            "different_from_alt_derivatives_crowding": (
                "trades six-alt long/short carry directly; does not use alt funding as a BTC directional state"
            ),
            "different_from_multiasset_funding_contra": (
                "exact funding settlements, 28-day overlapping sleeves, causal beta neutrality, "
                "strict held-path OHLC and carry attribution replace close-to-close score PnL"
            ),
            "different_from_btc_delta_neutral_carry": "cross-sectional six-alt perpetual pair; no spot leg or BTC",
        },
        "universe": {
            "venue": "Binance USD-M perpetual futures",
            "symbols": symbols,
            "position": "one long alt and one short alt per sleeve; no BTC position",
            "maximum_concurrent_sleeves": 4,
            "maximum_portfolio_gross": 1.0,
            "structural_orthogonality": (
                "market-neutral cross-alt carry; no BTC REX, OI, premium gate, Kimchi/FX, "
                "Markov, tree, LLM, spot-perp lead, or price-tail direction"
            ),
        },
        "source_contract": {
            "market": "exact 5m Binance USD-M OHLC; no fill or nearest join",
            "funding": "exact per-symbol realized settlement rows and millisecond timestamps",
            "physical_sources": {
                "2023": "frozen LORE 2023-2024 prefix, filtered before 2024",
                "2024_2025": "frozen LORC 2024-2025 prefix",
            },
            "causal_availability": (
                "a funding row is usable only when event_time <= Monday 00:05 UTC; hourly beta "
                "uses the completed hour ending Monday 00:00 UTC"
            ),
            "no_2026_source_before_2023_2025_pass": True,
        },
        "signal_clock": {
            "frequency": "every Monday",
            "signal_time": "00:05 UTC",
            "entry_time": "00:10 UTC 5m open",
            "exit_time": "entry + 28 calendar days at the 5m open",
            "overlap": "each qualifying weekly vintage is an independent 0.25-gross sleeve",
            "maximum_active_vintages": 4,
            "same_timestamp_order": "settle eligible funding, exit matured sleeve, then enter new sleeve",
        },
        "feature_formula": {
            "trailing_funding_window_days": 28,
            "trailing_funding_sum": "sum exact rates where signal-28d < event_time <= signal, per symbol",
            "high_funding_symbol": "argmax trailing funding sum; lexical tie-break",
            "low_funding_symbol": "argmin trailing funding sum; lexical tie-break",
            "hourly_return": "log(hour_close_t/hour_close_t-1)",
            "leave_one_out_factor": "median completed hourly return of the other five symbols",
            "beta": {
                "formula": "rolling cov(asset_return, loo_factor) / rolling var(loo_factor)",
                "lookback_hours": 720,
                "minimum_hours": 336,
                "history_shift_hours": 1,
                "clip": [0.25, 2.5],
            },
            "normalized_weights": {
                "long_low_funding": "high_beta/(high_beta+low_beta)",
                "short_high_funding_abs": "low_beta/(high_beta+low_beta)",
                "minimum_leg_weight": 0.25,
                "target_factor_beta": 0.0,
            },
            "projected_28d_carry": (
                "short_weight*high_trailing_funding_sum - long_weight*low_trailing_funding_sum"
            ),
            "minimum_projected_28d_carry": 0.0018,
            "hurdle_rationale": "18 bp equals 1.5 times the frozen 12 bp gross-one round-trip cost",
        },
        "frozen_policy": {
            "policy_id": "AFCH01",
            "long_leg": "lowest trailing-28d funding symbol",
            "short_leg": "highest trailing-28d funding symbol",
            "sleeve_gross": 0.25,
            "hold_days": 28,
            "entry_hurdle_projected_28d_carry": 0.0018,
            "no_price_direction_or_regime_gate": True,
        },
        "execution": {
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "funding_interval": "entry_time < funding_time <= exit_time",
            "funding_cash": "-signed_contract_quantity * causal_mark * realized_funding_rate",
            "sizing": "0.25 gross times current marked portfolio equity at each sleeve entry",
            "strict_mdd": (
                "global/pre-entry HWM on the aggregate portfolio; net symbol quantities; "
                "entry/exit costs; active-sleeve funding debits before adverse marks; active "
                "positive funding credits cannot raise HWM until sleeve exit; every held 5m bar "
                "marks favorable highs/lows before adverse lows/highs; hypothetical liquidation cost"
            ),
            "cagr": "full declared wall-clock including idle and warm-up periods",
            "tp_sl": None,
        },
        "support_gate": {
            "combined_2023_2025_events_at_least": 110,
            "each_year_events_at_least": 35,
            "each_half_year_events_at_least": 12,
            "unique_ordered_pairs_at_least": 8,
            "maximum_ordered_pair_share": 0.25,
            "long_symbols_at_least": 3,
            "short_symbols_at_least": 5,
            "minimum_leg_weight_enforced": True,
            "monthly_source_quarantine_at_most": 0.01,
            "outcome_columns_forbidden": True,
        },
        "selection_2023_2024": {
            "fit": ["2023-01-01", "2024-01-01"],
            "test": ["2024-01-01", "2025-01-01"],
            "single_policy_no_ranking": True,
            "each_year_absolute_return_positive": True,
            "each_year_cagr_to_strict_mdd_at_least": 1.5,
            "positive_half_years_at_least": 3,
            "combined_cagr_to_strict_mdd_at_least": 3.0,
            "combined_strict_mdd_at_most_pct": 15.0,
            "ten_bp_cost_stress_absolute_return_positive": True,
            "weekly_cluster_signflip_p_at_most": 0.10,
            "realized_funding_cash_positive": True,
            "realized_funding_cash_at_least_transaction_cost": True,
        },
        "eval_2025": {
            "opened_only_after_2023_2024_pass": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 15.0,
            "h1_h2_absolute_return_positive": True,
            "sleeves_at_least": 35,
            "ten_bp_cost_stress_absolute_return_positive": True,
            "weekly_cluster_signflip_p_at_most": 0.10,
            "realized_funding_cash_at_least_transaction_cost": True,
        },
        "orthogonality_after_2025": {
            "daily_mark_to_market_absolute_btc_beta_at_most": 0.10,
            "daily_mark_to_market_absolute_btc_correlation_at_most": 0.30,
            "daily_pnl_absolute_correlation_to_live_anchor_at_most": 0.30,
            "marginal_portfolio_improvement_required": True,
        },
        "final_2026_and_forward": {
            "2026_opened_only_after_2025_pass": True,
            "historical_2026_is_research_seen_elsewhere": True,
            "minimum_forward_shadow_days_for_promotion": 90,
            "multi_symbol_funding_and_execution_ledger_required": True,
            "promotion_never_based_on_historical_2026_alone": True,
        },
        "diagnostics_not_repair_candidates": [
            "exact carry direction flip",
            "equal 50/50 leg weights",
            "funding cash removed from accounting",
            "weekly clock shifted by three days",
            "ordered pairs permuted within quarter",
        ],
        "stop_rule": (
            "Reject before 2025 if AFCH01 fails the frozen 2023-2024 gate. Reject before 2026 "
            "if it fails 2025. No funding lookback, carry hurdle, hold, sleeve gross, direction, "
            "pair whitelist, beta window, or regime repair is allowed after outcomes are opened."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    return f"""# AFCH v1 preregistration — 2026-07-17

## Evidence boundary

The six-alt source history and adjacent funding-score experiments have been
seen elsewhere, so historical results cannot by themselves promote AFCH. The
**exact payoff object**—four overlapping, factor-beta-neutral sleeves with
exact realized funding cashflow and portfolio-level strict MDD—has not been
opened. AFCH must pass sequential 2023/2024/2025 research gates and then at
least 90 forward-shadow days before promotion.

## Orthogonal economic object

Every Monday at 00:05 UTC, AFCH sums each alt's exact realized funding over the
prior 28 days. It goes long the lowest-funding symbol and short the highest-
funding symbol, with causal rolling beta-neutral weights. It has no BTC
position and uses no REX, OI, premium, Kimchi/FX, Markov, tree, LLM, or price-
direction gate. Profit is required to come from realized cross-sectional
funding transfer, not merely favorable price drift.

## Frozen policy AFCH01

- universe: ETH, SOL, BNB, XRP, ADA, DOGE USD-M perpetuals;
- trailing exact funding window: 28 days;
- beta: shifted 720-hour leave-one-out factor estimate, clipped `[0.25, 2.5]`;
- trade only if beta-weighted projected 28-day carry is at least `18 bp`;
- one new sleeve per qualifying Monday, gross `0.25`, hold exactly 28 days;
- at most four concurrent sleeves, portfolio gross at most `1.0`;
- signal `00:05`, next 5m-open entry `00:10`, scheduled exit after 28 days;
- base cost `6 bp/notional/side`, stress `10 bp`, exact funding settlements;
- aggregate favorable-before-adverse strict MDD and full-calendar CAGR.

The 18 bp hurdle is fixed by economics: 1.5 times the gross-one 12 bp
round-trip cost. It is not selected from post-entry returns.

## Qualification

Support must provide at least 110 sleeves across 2023–2025, 35 per year, and
12 per half-year. The single policy then needs positive 2023 and 2024 returns,
ratio >= 1.5 in each, combined CAGR/strict-MDD >= 3, strict MDD <= 15%, positive
10 bp stress, weekly cluster p <= 0.10, and realized funding cash at least equal
to all transaction costs. Only then may 2025 be opened under ratio >= 3 and the
same carry-attribution requirement.

## Anti-repair and production boundary

No sign, hold, funding lookback, hurdle, beta, pair, sleeve, or regime repair is
allowed after outcomes open. Even a historical pass remains research-only
until a multi-symbol live ledger and at least 90 forward-shadow days confirm it.

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
