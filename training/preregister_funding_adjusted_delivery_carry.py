"""Freeze the FADC-21 market-neutral delivery-carry hypothesis before PnL."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT = "results/funding_adjusted_delivery_carry_preregistration_2026-07-17.json"
DOCS_OUTPUT = "docs/funding-adjusted-delivery-carry-preregistration-2026-07-17.md"

QUARTERLY_PANEL = (
    "data/binance_cross_collateral_quarterly_curve_2021_2023/"
    "BTCUSDT_BTCUSD_CURRENT_QUARTER_5m_2021_2023.csv.gz"
)
PERPETUAL_1M = "data/binance_perp_btc_1m_2020_2023.csv.gz"
FUNDING_MARKS = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"

FROZEN_SOURCE_SHA256 = {
    QUARTERLY_PANEL: "54addc04b997cfb077197cd845f2aa286a219bdae4a29b49c2086667007046f7",
    PERPETUAL_1M: "0b55bb0c3b845a90da738e746c769b19c1de4ac230ca8f1fccb6c361c4a9a41f",
    FUNDING_MARKS: "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol() -> dict[str, Any]:
    return {
        "candidate_id": "FADC-21",
        "name": "funding-adjusted USD-M delivery carry",
        "hypothesis": (
            "The BTCUSDT current-quarter/perpetual carry gap contains a persistent, "
            "same-collateral relative-value premium. A delta-matched long/short pair "
            "can earn delivery convergence plus perpetual funding without taking an "
            "explicit BTC direction."
        ),
        "novelty_boundary": {
            "not_ccbs": (
                "CCBS-12 traded USD-M versus inverse COIN-M same-maturity dislocations "
                "for twelve-hour snapback. FADC-21 trades two linear USD-M instruments, "
                "uses the delivery-versus-perpetual term structure, and holds a carry "
                "state rather than a cross-collateral z-score."
            ),
            "not_spot_perp_basis_compression": (
                "The rejected spot/perpetual compression family traded a short-horizon "
                "basis z-score. FADC-21 compares delivery-implied annualized basis with "
                "strictly prior settled funding and has a deterministic delivery anchor."
            ),
            "portfolio_relation": (
                "No REX, price-direction, OI, kimchi/FX, Markov, tree, LLM, or manual "
                "regime gate is permitted. Orthogonality is judged on executed daily PnL, "
                "not merely on feature correlation."
            ),
        },
        "source_contract": {
            "physical_outcome_boundary": "all inputs end strictly before 2024-01-01",
            "quarterly_leg": {
                "instrument": "Binance USD-M BTCUSDT CURRENT_QUARTER continuous contract",
                "columns_allowed_for_signal": [
                    "open_time",
                    "available_time",
                    "um_close",
                    "um_ohlc_valid",
                    "source_complete",
                    "delivery_time",
                    "contract_segment",
                ],
                "execution_columns_open_only_after_evaluator_freeze": [
                    "um_open",
                    "um_high",
                    "um_low",
                ],
            },
            "perpetual_leg": {
                "instrument": "Binance USD-M BTCUSDT perpetual",
                "aggregation": (
                    "resample immutable one-minute OHLC to left-labeled five-minute OHLC; "
                    "require exactly five rows and never fill a missing minute"
                ),
            },
            "funding": {
                "instrument": "Binance USD-M BTCUSDT perpetual funding history",
                "settlement_mark": (
                    "uniform frozen official eight-hour mark-price-kline open proxy "
                    "mapped to the exact funding timestamp"
                ),
            },
            "source_sha256": FROZEN_SOURCE_SHA256,
            "historical_point_in_time_limit": (
                "market/funding rows are immutable official or DB-backfilled history, not "
                "a retrieval-timestamped PIT archive; a passing backtest still requires "
                "forward source-parity shadowing"
            ),
        },
        "signal": {
            "decision_clock": (
                "after each settled funding timestamp t, require the complete quarterly "
                "and perpetual five-minute candle [t,t+5m); signal is available at t+5m"
            ),
            "entry_clock": "enter both legs at the t+10m open",
            "funding_forecast": (
                "arithmetic mean of the 21 funding rates with timestamps <= t, including "
                "the just-settled rate; require all 21; annualize as mean*3*365"
            ),
            "annualized_basis": (
                "log(quarterly_close/perpetual_close)*365/days_to_delivery measured at "
                "signal availability"
            ),
            "carry_gap": "annualized_basis - annualized_funding_forecast",
            "edge_to_scheduled_exit": (
                "abs(carry_gap)*days_between_signal_and_(delivery-24h)/365"
            ),
            "entry_requirements": {
                "minimum_days_to_delivery": 14.0,
                "maximum_days_to_delivery": 80.0,
                "minimum_expected_edge_fraction": 0.003,
                "rationale": (
                    "30 bp expected carry exceeds the frozen 20 bp account-level round "
                    "trip at the 10 bp/notional-side stress cost"
                ),
            },
            "direction": {
                "carry_gap_positive": "long perpetual, short current-quarter",
                "carry_gap_negative": "short perpetual, long current-quarter",
            },
            "exit": {
                "minimum_hold_hours": 24,
                "normalization_edge_fraction": 0.0005,
                "normalization_rule": (
                    "after minimum hold, exit at the next t+10m open when expected edge "
                    "is <=5 bp or carry-gap sign differs from entry"
                ),
                "mandatory_exit": "delivery_time-24h open, strictly before delivery",
                "cooldown_hours": 24,
            },
            "overlap": "one FADC position at a time; no pyramiding or overlapping segment",
        },
        "ledger": {
            "initial_gross": 1.0,
            "quantity": (
                "freeze identical BTC quantity q=pre_entry_equity/"
                "(quarterly_entry+perpetual_entry), so combined entry gross is exactly 1x"
            ),
            "entry_cost_does_not_resize_quantity": True,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
            "funding_interval": "entry_time <= funding_time < exit_time",
            "funding_cash": (
                "-perpetual_side*q*funding_rate*frozen_settlement_mark_price"
            ),
            "strict_mdd": (
                "global/pre-entry HWM; on every held five-minute bar combine independent "
                "favorable leg extrema before independent adverse extrema, apply funding "
                "in timestamp order, and include hypothetical two-leg liquidation costs"
            ),
            "cagr": "full wall-clock evaluation period including warm-up and idle time",
            "reporting": "always report absolute return with CAGR, strict MDD, ratio, trades",
            "execution_limit": (
                "historical continuous-contract OHLC is a research proxy; live promotion "
                "requires current symbol resolution, tick/step rounding, two-leg atomicity, "
                "margin/liquidation modeling, and forward slippage evidence"
            ),
        },
        "support_gates_before_pnl": {
            "selection_period": "2021-02-03 through 2022-12-31",
            "minimum_entries_total": 24,
            "minimum_entries_each_year": 10,
            "minimum_entries_each_half": 5,
            "minimum_each_direction_share": 0.25,
            "maximum_entry_month_share": 0.15,
            "2023_diagnostic_only": {
                "minimum_entries": 8,
                "minimum_entries_each_half": 3,
                "minimum_each_direction_share": 0.25,
                "maximum_entry_month_share": 0.25,
            },
            "forbidden": (
                "support code may load current/past closes, settled funding and clocks but "
                "must not load future entry/exit prices, held OHLC paths, returns or PnL"
            ),
        },
        "sequential_outcome_gates": {
            "stage_1_2021_2022": {
                "combined_absolute_return_positive": True,
                "each_calendar_year_absolute_return_positive": True,
                "at_least_three_of_four_halves_positive": True,
                "combined_cagr_to_strict_mdd_minimum": 2.0,
                "strict_mdd_maximum_pct": 15.0,
                "stress_cost_absolute_return_positive": True,
                "both_carry_gap_directions_positive": True,
                "weekly_cluster_signflip_pvalue_maximum": 0.10,
            },
            "stage_2_2023_holdout": {
                "opened_only_if_stage_1_passes": True,
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_minimum": 3.0,
                "strict_mdd_maximum_pct": 15.0,
                "minimum_trades": 8,
                "h1_and_h2_absolute_return_positive": True,
                "stress_cost_absolute_return_positive": True,
                "both_carry_gap_directions_positive": True,
                "weekly_cluster_signflip_pvalue_maximum": 0.10,
            },
            "controls": [
                "zero funding cashflow",
                "carry direction flip",
                "one funding-event decision delay",
                "basis-only sign using zero funding forecast",
                "constant long-perpetual/short-quarterly while eligible",
            ],
            "no_repair": (
                "After any outcome opens, do not change lookback, edge threshold, DTE, "
                "direction, exit, cooldown, costs, sizing, side filter or control mapping."
            ),
            "future_boundary": (
                "2024 is the first source-and-outcome-unopened OOS year and stays sealed "
                "unless both pre-2024 stages and later orthogonality gates pass."
            ),
        },
        "orthogonality_gate_after_economic_pass": {
            "daily_pnl_pearson_abs_maximum": 0.30,
            "daily_pnl_spearman_abs_maximum": 0.30,
            "minimum_nonzero_pnl_days": 20,
            "fixed_incremental_weight": 0.25,
            "require_synchronized_portfolio_ratio_improvement": True,
            "btc_beta_diagnostic": "report zero-filled daily PnL beta to BTC daily close return",
        },
    }


def render_doc(body: dict[str, Any], protocol_hash: str) -> str:
    support = body["support_gates_before_pnl"]
    return f"""# FADC-21 preregistration — 2026-07-17

## Hypothesis

FADC-21 is a same-collateral, first-order delta-matched relative-value sleeve:
it trades Binance USD-M BTCUSDT perpetual against the USD-M current-quarter
future. It does not use BTC direction, REX, OI, kimchi/FX, Markov, tree, LLM or
a manual regime. The expected edge is delivery basis minus a strictly prior
21-settlement funding forecast.

This is not the rejected CCBS twelve-hour cross-collateral snapback and not the
rejected spot/perpetual z-score compression. Both legs are linear USD-M; the
economic anchor is delivery convergence plus actual perpetual funding.

## Frozen causal policy

- At funding time `t`, wait for both complete `[t,t+5m)` candles; decide at
  `t+5m` and execute both legs at the `t+10m` open.
- `funding_ann = mean(last 21 settled rates including t) * 3 * 365`.
- `basis_ann = log(quarter_close/perp_close) * 365 / DTE`.
- `gap = basis_ann - funding_ann`.
- Enter only for DTE 14–80 days and expected edge through `delivery-24h` of at
  least **30 bp**. Positive gap means long perp/short quarter; negative means
  short perp/long quarter.
- After 24 hours, exit when expected edge is at most 5 bp or the gap changes
  sign. Always exit at least 24 hours before delivery and wait 24 hours before
  re-entry.

The 30 bp gate is economic, not outcome-fit: at gross 1x the two-leg round trip
cost is 12 bp at base costs and 20 bp under the frozen 10 bp/notional-side
stress.

## Strict ledger

Both legs receive the same frozen BTC quantity, so entry gross is exactly 1x.
Funding uses the frozen settlement-mark proxy for
`entry_time <= funding_time < exit_time`. Strict MDD uses the global/pre-entry
HWM, favorable-before-adverse independent leg extrema, funding in timestamp
order, and hypothetical two-leg liquidation costs. CAGR covers the full
wall-clock period; absolute return is mandatory in every result table.

## Outcome boundary and gates

The preregistration reads no price outcome or PnL. All three bound inputs end
before 2024. Outcome-blind support must first find at least
{support['minimum_entries_total']} pre-2023 entries with year/half/direction and
month-concentration floors. Only then may 2021–2022 PnL open. A failure rejects
the candidate without opening 2023. A stage-1 pass may open 2023 exactly once;
2024 remains sealed until the 2023 and executed-PnL orthogonality gates pass.

Passing history is still not live-ready: continuous-contract symbol resolution,
tick/step rounding, atomic two-leg execution, margin/liquidation accounting and
forward source/slippage parity remain hard blockers.

Protocol hash: `{protocol_hash}`
"""


def run(
    *,
    output: str = OUTPUT,
    docs_output: str = DOCS_OUTPUT,
) -> dict[str, Any]:
    for path, expected in FROZEN_SOURCE_SHA256.items():
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"frozen source hash drifted: {path}: {actual}")
    body = protocol()
    protocol_hash = canonical_hash(body)
    artifact = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "protocol": body,
        "protocol_hash": protocol_hash,
        "outcome_columns_loaded": [],
        "pnl_opened": False,
        "oos_2024_plus_opened": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).write_text(render_doc(body, protocol_hash))
    return artifact


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
