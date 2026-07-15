#!/usr/bin/env python3
"""Audit one frozen rank-7/Fresh-Kimchi fixed-subaccount portfolio.

This is not a weight search.  It evaluates exactly one 75/25 capital split and
keeps each sleeve in its own subaccount.  Five-minute OHLC envelopes are
combined at the same underlying BTC price point before drawdown is measured,
so opposite positions can offset instead of being assigned impossible
simultaneous adverse prices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_fresh_kimchi_orthogonal_alpha import (
    CANDIDATE_SPEC,
    DEFAULT_FUNDING,
    DEFAULT_INPUT,
    DEFAULT_PREMIUM,
    Config as AuditConfig,
    build_candidate_context,
    build_rank7_context,
    candidate_schedule,
    compact_stats,
    rank7_schedule,
)
from training.audit_weak_feature_responsibility_stability import (
    _action_spec as rank7_action_spec,
)
from training.evaluate_expanding_extratrees_top10_oos import FULL_CUTOFF
from training.search_inventory_purge_reclaim_alpha import Config as ExecutionConfig
from training.search_inventory_purge_reclaim_alpha import Trade


DEFAULT_OUTPUT = "results/rank7_fresh_kimchi_fixed_portfolio_2026-07-16.json"
DEFAULT_DOCS = "docs/rank7-fresh-kimchi-fixed-portfolio-2026-07-16.md"

WEIGHTS = {"frozen_annual_rank7": 0.75, "fresh_kimchi_fx": 0.25}
PORTFOLIO_SPEC: dict[str, Any] = {
    "name": "rank7_75_fresh_kimchi_25_fixed_subaccounts",
    "weights": WEIGHTS,
    "weight_grid_cells": 1,
    "rebalancing": "none",
    "selection": "design-fixed diagnostic allocation; no future ranking",
    "mdd_clock": "shared 5-minute BTC OHLC envelope",
    "same_bar_order": "portfolio upper envelope before lower envelope",
    "peak_carry": "one peak across the full calendar split, including cash and pre-entry history",
}
EXPECTED_PORTFOLIO_SPEC_HASH = "57c25f64b14d63cc700914bd8de9b4091bd6da7a020fc124eaf328fa92e9607c"
WINDOWS: dict[str, tuple[str, str]] = {
    "selection_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026h1": ("2026-01-01", FULL_CUTOFF),
    "future_2025_2026h1": ("2025-01-01", FULL_CUTOFF),
    "oos_2024_2026h1": ("2024-01-01", FULL_CUTOFF),
}


@dataclass(frozen=True)
class Config:
    input_csv: str = DEFAULT_INPUT
    funding_csv: str = DEFAULT_FUNDING
    premium_csv: str = DEFAULT_PREMIUM
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    exclude_from: str = FULL_CUTOFF


@dataclass(frozen=True)
class BarPath:
    """Normalized subaccount values at synchronized points of each market bar."""

    dates: pd.DatetimeIndex
    open_value: np.ndarray
    market_low_value: np.ndarray
    market_high_value: np.ndarray
    close_value: np.ndarray
    active: np.ndarray
    final_equity: float


def portfolio_spec_hash() -> str:
    payload = json.dumps(PORTFOLIO_SPEC, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _price_factor(price: float, entry_price: float, *, side: int, leverage: float) -> float:
    factor = 1.0 + leverage * side * (price / entry_price - 1.0)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("non-positive leveraged mark factor")
    return float(factor)


def _funding_factor_through(
    funding_times: np.ndarray,
    funding_rates: np.ndarray,
    *,
    entry_ns: int,
    mark_ns: int,
    side: int,
    leverage: float,
) -> float:
    left = int(np.searchsorted(funding_times, entry_ns, side="left"))
    right = int(np.searchsorted(funding_times, mark_ns, side="right"))
    factors = 1.0 - leverage * side * funding_rates[left:right]
    if not np.isfinite(factors).all() or (factors <= 0.0).any():
        raise ValueError("invalid realized funding mark factor")
    return float(np.prod(factors, dtype=float)) if len(factors) else 1.0


def subaccount_bar_path(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    trades: Iterable[Trade],
    cfg: ExecutionConfig,
    *,
    start: str,
    end: str,
    hold_bars: Callable[[Trade], int],
) -> BarPath:
    """Reconstruct an exact-final-equity, same-price OHLC subaccount path.

    Barrier exits use the execution engine's stop-first result.  The triggering
    adverse price is capped at the realized stop and includes the exit fee;
    favorable extremes remain as a conservative peak envelope.  Cap exits
    happen at the cap-bar open, matching ``ExecutionEngine.trade_at``.
    """

    dates = pd.DatetimeIndex(pd.to_datetime(market["date"]))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("market dates must be unique and increasing")
    start_pos = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    end_pos = int(dates.searchsorted(pd.Timestamp(end), side="left"))
    if end_pos <= start_pos:
        raise ValueError("empty portfolio window")
    window_dates = dates[start_pos:end_pos]
    size = len(window_dates)
    open_value = np.full(size, np.nan, dtype=float)
    low_value = np.full(size, np.nan, dtype=float)
    high_value = np.full(size, np.nan, dtype=float)
    close_value = np.full(size, np.nan, dtype=float)
    active = np.zeros(size, dtype=bool)

    opens = pd.to_numeric(market["open"], errors="raise").to_numpy(float)
    highs = pd.to_numeric(market["high"], errors="raise").to_numpy(float)
    lows = pd.to_numeric(market["low"], errors="raise").to_numpy(float)
    closes = pd.to_numeric(market["close"], errors="raise").to_numpy(float)
    funding_times = pd.to_datetime(funding["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    funding_rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    leverage = float(cfg.leverage)
    cost_factor = 1.0 - leverage * float(cfg.fee_rate + cfg.slippage_rate)
    if cost_factor <= 0.0:
        raise ValueError("non-positive entry/exit cost factor")

    ordered = sorted(trades, key=lambda trade: (trade.entry_position, trade.exit_position))
    cursor = 0
    realized_equity = 1.0
    previous_exit = start_pos - 1
    for trade in ordered:
        entry = int(trade.entry_position)
        exit_position = int(trade.exit_position)
        if entry < start_pos or exit_position >= end_pos:
            raise ValueError("all trades must be contained inside the requested split")
        if entry <= previous_exit:
            raise ValueError("a sleeve may not hold overlapping trades")
        local_entry = entry - start_pos
        local_exit = exit_position - start_pos
        open_value[cursor:local_entry] = realized_equity
        low_value[cursor:local_entry] = realized_equity
        high_value[cursor:local_entry] = realized_equity
        close_value[cursor:local_entry] = realized_equity

        trade_start_equity = realized_equity
        entry_price = float(opens[entry])
        hold = int(hold_bars(trade))
        if hold <= 0 or exit_position > entry + hold:
            raise ValueError("invalid hold resolver for trade")
        cap_exit = exit_position == entry + hold
        entry_ns = int(dates[entry].value)
        realized_after_exit = (
            trade_start_equity
            * cost_factor
            * float(trade.price_factor)
            * float(trade.funding_factor)
            * cost_factor
        )

        for position in range(entry, exit_position + 1):
            local = position - start_pos
            funding_factor = _funding_factor_through(
                funding_times,
                funding_rates,
                entry_ns=entry_ns,
                mark_ns=int(dates[position].value),
                side=int(trade.side),
                leverage=leverage,
            )
            marked_base = trade_start_equity * cost_factor * funding_factor
            open_mark = marked_base * _price_factor(
                float(opens[position]), entry_price, side=int(trade.side), leverage=leverage
            )
            open_value[local] = open_mark
            active[local] = not (cap_exit and position == exit_position)

            if cap_exit and position == exit_position:
                low_value[local] = open_mark
                high_value[local] = open_mark
                close_value[local] = realized_after_exit
                continue

            low_mark = marked_base * _price_factor(
                float(lows[position]), entry_price, side=int(trade.side), leverage=leverage
            )
            high_mark = marked_base * _price_factor(
                float(highs[position]), entry_price, side=int(trade.side), leverage=leverage
            )
            close_mark = marked_base * _price_factor(
                float(closes[position]), entry_price, side=int(trade.side), leverage=leverage
            )
            if position == exit_position:
                if float(trade.gross_return) < 0.0:  # stop exit
                    if int(trade.side) > 0:
                        low_mark = realized_after_exit
                    else:
                        high_mark = realized_after_exit
                else:  # take exit; preserve the raw favorable extreme as a strict peak
                    if int(trade.side) > 0:
                        high_mark = max(high_mark, realized_after_exit)
                    else:
                        low_mark = max(low_mark, realized_after_exit)
                close_mark = realized_after_exit
            low_value[local] = low_mark
            high_value[local] = high_mark
            close_value[local] = close_mark

        observed_funding = _funding_factor_through(
            funding_times,
            funding_rates,
            entry_ns=entry_ns,
            mark_ns=int(dates[exit_position].value),
            side=int(trade.side),
            leverage=leverage,
        )
        if not np.isclose(observed_funding, trade.funding_factor, rtol=1e-12, atol=1e-15):
            raise RuntimeError("bar path funding differs from execution trade")
        realized_equity = realized_after_exit
        previous_exit = exit_position
        cursor = local_exit + 1

    open_value[cursor:] = realized_equity
    low_value[cursor:] = realized_equity
    high_value[cursor:] = realized_equity
    close_value[cursor:] = realized_equity
    for name, values in {
        "open": open_value,
        "low": low_value,
        "high": high_value,
        "close": close_value,
    }.items():
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise RuntimeError(f"invalid {name} subaccount path")
    if not np.isclose(float(close_value[-1]), realized_equity, rtol=1e-12, atol=1e-15):
        raise RuntimeError("subaccount path does not preserve exact final equity")
    return BarPath(
        dates=window_dates,
        open_value=open_value,
        market_low_value=low_value,
        market_high_value=high_value,
        close_value=close_value,
        active=active,
        final_equity=float(realized_equity),
    )


def synchronized_portfolio_stats(
    paths: dict[str, BarPath],
    weights: dict[str, float],
    *,
    start: str,
    end: str,
    trade_counts: dict[str, int],
) -> dict[str, Any]:
    if set(paths) != set(weights) or set(paths) != set(trade_counts):
        raise ValueError("paths, weights, and trade counts must have identical sleeves")
    if any(weight < 0.0 for weight in weights.values()) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("fixed-subaccount weights must be non-negative and sum to one")
    first = next(iter(paths.values()))
    for path in paths.values():
        if not path.dates.equals(first.dates):
            raise ValueError("subaccount paths must share one market grid")

    def combine(field: str) -> np.ndarray:
        return sum(float(weights[name]) * getattr(path, field) for name, path in paths.items())

    open_value = combine("open_value")
    low_value = combine("market_low_value")
    high_value = combine("market_high_value")
    close_value = combine("close_value")
    peak = 1.0
    strict_mdd = 0.0
    for row in zip(open_value, low_value, high_value, close_value, strict=True):
        upper = float(max(row))
        lower = float(min(row))
        peak = max(peak, upper)
        strict_mdd = max(strict_mdd, 1.0 - lower / peak)
        peak = max(peak, float(row[-1]))

    final_equity = float(close_value[-1])
    expected_final = float(sum(weights[name] * path.final_equity for name, path in paths.items()))
    if not np.isclose(final_equity, expected_final, rtol=1e-12, atol=1e-15):
        raise RuntimeError("portfolio final equity differs from fixed subaccounts")
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (final_equity - 1.0) * 100.0
    cagr = (final_equity ** (1.0 / years) - 1.0) * 100.0
    mdd = strict_mdd * 100.0
    active_matrix = np.column_stack([path.active for path in paths.values()])
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "synchronized_strict_mdd_pct": float(mdd),
        "cagr_to_synchronized_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "final_equity": final_equity,
        "trades": int(sum(trade_counts.values())),
        "trades_by_sleeve": dict(trade_counts),
        "bars_any_sleeve_active": int(active_matrix.any(axis=1).sum()),
        "bars_multiple_sleeves_active": int((active_matrix.sum(axis=1) > 1).sum()),
    }


def _render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen rank-7 + Fresh Kimchi fixed portfolio audit",
        "",
        "Exactly one allocation is evaluated: 75% frozen annual rank-7 and 25% "
        "Fresh Kimchi/FX, held in fixed subaccounts with no rebalancing or weight grid.",
        "",
        "The strict portfolio path combines both sleeves at the same five-minute BTC "
        "open/low/high/close point, includes entry/exit costs and realized funding, "
        "and applies the execution engine's stop-first barrier result. This avoids the "
        "impossible assumption that a simultaneous long and short both receive their "
        "own adverse BTC price.",
        "",
        "| Window | Portfolio abs return | CAGR | Sync strict MDD | CAGR/MDD | "
        "Rank7 CAGR/MDD | Delta ratio | Trades (R7/Kimchi) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in WINDOWS:
        row = payload["windows"][name]
        portfolio = row["portfolio"]
        rank7 = row["synchronized_components"]["frozen_annual_rank7"]
        lines.append(
            f"| {name} | {portfolio['absolute_return_pct']:.4f}% | "
            f"{portfolio['cagr_pct']:.4f}% | {portfolio['synchronized_strict_mdd_pct']:.4f}% | "
            f"{portfolio['cagr_to_synchronized_strict_mdd']:.4f} | "
            f"{rank7['cagr_to_synchronized_strict_mdd']:.4f} | "
            f"{row['marginal_vs_rank7']['ratio_delta']:+.4f} | "
            f"{row['canonical_components']['frozen_annual_rank7']['trades']}/"
            f"{row['canonical_components']['fresh_kimchi_fx']['trades']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["verdict"],
            "",
            "This is a diagnostic marginal-value audit, not pristine OOS evidence: the "
            "Fresh Kimchi gate used 2024 for selection and 2025/2026 had already been "
            "viewed. No threshold or weight was changed after replay.",
            "",
            "Canonical component strict MDD from the original execution evaluator and "
            "the synchronized portfolio MDD are both retained in the JSON. Promotion "
            "still requires an untouched forward window.",
            "",
        ]
    )
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    if cfg.exclude_from != FULL_CUTOFF:
        raise ValueError(f"exclude_from must equal frozen cutoff {FULL_CUTOFF}")
    audit_cfg = AuditConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output="/tmp/no_write_fixed_portfolio_source.json",
        docs_output="",
        exclude_from=cfg.exclude_from,
    )
    candidate = build_candidate_context(audit_cfg)
    rank7 = build_rank7_context(audit_cfg)
    market = candidate["market"]
    rank7_market = rank7["base"]["context"]["market"]
    if not np.array_equal(pd.to_datetime(market["date"]), pd.to_datetime(rank7_market["date"])):
        raise RuntimeError("candidate and rank-7 market grids differ")
    for column in ("open", "high", "low", "close"):
        if not np.array_equal(market[column].to_numpy(float), rank7_market[column].to_numpy(float)):
            raise RuntimeError(f"candidate and rank-7 {column} differ")

    source_audit = json.loads(Path("results/fresh_kimchi_orthogonal_alpha_audit_2026-07-16.json").read_text())
    windows: dict[str, Any] = {}
    for name, (start, end) in WINDOWS.items():
        candidate_trades = candidate_schedule(candidate, start=start, end=end)
        rank7_trades = rank7_schedule(rank7, start=start, end=end)
        candidate_canonical = compact_stats(
            candidate_trades, start=start, end=end, cfg=candidate["execution_cfg"]
        )
        rank7_canonical = compact_stats(
            rank7_trades, start=start, end=end, cfg=rank7["base"]["execution_cfg"]
        )
        for actual, expected, label in (
            (candidate_canonical, source_audit["candidate_stats"][name], "candidate"),
            (rank7_canonical, source_audit["rank7_stats"][name], "rank7"),
        ):
            if actual["schedule_hash"] != expected["schedule_hash"]:
                raise RuntimeError(f"{name} {label} schedule drifted")

        candidate_path = subaccount_bar_path(
            market,
            candidate["funding"],
            candidate_trades,
            candidate["execution_cfg"],
            start=start,
            end=end,
            hold_bars=lambda _trade: int(CANDIDATE_SPEC["hold_bars"]),
        )
        funding_leg = np.asarray(rank7["base"]["context"]["funding_leg"], dtype=bool)
        rank7_path = subaccount_bar_path(
            rank7_market,
            rank7["base"]["context"]["funding"],
            rank7_trades,
            rank7["base"]["execution_cfg"],
            start=start,
            end=end,
            hold_bars=lambda trade: int(rank7_action_spec(bool(funding_leg[trade.signal_position]))[0]),
        )
        synchronized_components = {
            "frozen_annual_rank7": synchronized_portfolio_stats(
                {"frozen_annual_rank7": rank7_path},
                {"frozen_annual_rank7": 1.0},
                start=start,
                end=end,
                trade_counts={"frozen_annual_rank7": len(rank7_trades)},
            ),
            "fresh_kimchi_fx": synchronized_portfolio_stats(
                {"fresh_kimchi_fx": candidate_path},
                {"fresh_kimchi_fx": 1.0},
                start=start,
                end=end,
                trade_counts={"fresh_kimchi_fx": len(candidate_trades)},
            ),
        }
        portfolio = synchronized_portfolio_stats(
            {"frozen_annual_rank7": rank7_path, "fresh_kimchi_fx": candidate_path},
            WEIGHTS,
            start=start,
            end=end,
            trade_counts={
                "frozen_annual_rank7": len(rank7_trades),
                "fresh_kimchi_fx": len(candidate_trades),
            },
        )
        for component, canonical in (
            (synchronized_components["frozen_annual_rank7"], rank7_canonical),
            (synchronized_components["fresh_kimchi_fx"], candidate_canonical),
        ):
            if not np.isclose(
                component["absolute_return_pct"], canonical["absolute_return_pct"], rtol=1e-11, atol=1e-10
            ):
                raise RuntimeError(f"{name} synchronized component final return drifted")
        baseline = synchronized_components["frozen_annual_rank7"]
        windows[name] = {
            "canonical_components": {
                "frozen_annual_rank7": rank7_canonical,
                "fresh_kimchi_fx": candidate_canonical,
            },
            "synchronized_components": synchronized_components,
            "portfolio": portfolio,
            "marginal_vs_rank7": {
                "absolute_return_delta_pct": portfolio["absolute_return_pct"]
                - baseline["absolute_return_pct"],
                "cagr_delta_pct": portfolio["cagr_pct"] - baseline["cagr_pct"],
                "strict_mdd_delta_pct": portfolio["synchronized_strict_mdd_pct"]
                - baseline["synchronized_strict_mdd_pct"],
                "ratio_delta": portfolio["cagr_to_synchronized_strict_mdd"]
                - baseline["cagr_to_synchronized_strict_mdd"],
            },
        }

    future = windows["future_2025_2026h1"]
    delta = future["marginal_vs_rank7"]
    improves_risk_efficiency = (
        delta["ratio_delta"] > 0.0 and delta["strict_mdd_delta_pct"] < 0.0
    )
    verdict = (
        "The fixed 25% Fresh Kimchi sleeve improves synchronized risk efficiency in "
        "2025-2026H1: strict MDD falls and CAGR/MDD rises, while raw CAGR is slightly "
        "diluted. Treat it as a risk-budget diversification shadow candidate, not a "
        "return-maximizing replacement for rank-7. Do not enable live until a new "
        "untouched period confirms the marginal gain."
        if improves_risk_efficiency
        else "The fixed 25% Fresh Kimchi sleeve does not improve synchronized strict MDD "
        "and CAGR/MDD versus rank-7 alone. Do not allocate live capital; retain only as "
        "a frozen forward-shadow candidate."
    )
    spec_hash = portfolio_spec_hash()
    if spec_hash != EXPECTED_PORTFOLIO_SPEC_HASH:
        raise RuntimeError(
            f"fixed portfolio specification drifted: {spec_hash} != {EXPECTED_PORTFOLIO_SPEC_HASH}"
        )
    payload = {
        "schema_version": 1,
        "mode": "frozen_rank7_fresh_kimchi_fixed_subaccount_portfolio",
        "config": asdict(cfg),
        "portfolio_spec": PORTFOLIO_SPEC,
        "portfolio_spec_hash": spec_hash,
        "weights": WEIGHTS,
        "weight_grid_cells": 1,
        "weight_selection": "preregistered diagnostic 75/25; no future reranking",
        "future_is_pristine_discovery_oos": False,
        "future_did_not_rerank": True,
        "mdd_method": (
            "synchronized 5-minute fixed-subaccount BTC open/low/high/close envelope; "
            "same underlying price for both sleeves; realized funding and entry/exit costs; "
            "stop-first barrier fills; upper-before-lower same-bar conservative ordering"
        ),
        "windows": windows,
        "verdict": verdict,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if cfg.docs_output:
        docs = Path(cfg.docs_output)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(_render_docs(payload))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--funding-csv", default=DEFAULT_FUNDING)
    parser.add_argument("--premium-csv", default=DEFAULT_PREMIUM)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    parser.add_argument("--exclude-from", default=FULL_CUTOFF)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                name: {
                    "portfolio": row["portfolio"],
                    "marginal_vs_rank7": row["marginal_vs_rank7"],
                }
                for name, row in payload["windows"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
