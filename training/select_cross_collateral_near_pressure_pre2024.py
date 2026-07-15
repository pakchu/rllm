#!/usr/bin/env python3
"""Select and freeze a cross-collateral near-book pressure alpha before 2024.

The feature family uses only outcome-blind Binance Vision USD-M/COIN-M
``bookDepth`` panels.  H1 2023 owns every rolling calibration and quantile;
H2 2023 selects one candidate.  No source row at or after 2024 is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.preregister_cross_collateral_liquidity_void_refill import lagged_robust_zscore
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)


SELECTION_END = "2024-01-01"
DEFAULT_SHELLS = (
    "data/binance_cross_collateral_book_shells_btc_2023/"
    "BTC_cross_collateral_book_shells_5m_2023.csv.gz"
)
DEFAULT_CREDIBILITY = (
    "data/binance_cross_collateral_book_credibility_btc_2023/"
    "BTC_cross_collateral_book_credibility_5m_2023.csv.gz"
)
DEFAULT_MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
DEFAULT_FUNDING = (
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_OUTPUT = "results/cross_collateral_near_pressure_pre2024_selection_2026-07-16.json"
DEFAULT_MANIFEST = "results/cross_collateral_near_pressure_pre2024_manifest_2026-07-16.json"
DEFAULT_DOCS = "docs/cross-collateral-near-pressure-pre2024-selection-2026-07-16.md"

WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2023h1": ("2023-01-01", "2023-07-01"),
    "selection_2023h2": ("2023-07-01", SELECTION_END),
    "q1": ("2023-01-01", "2023-04-01"),
    "q2": ("2023-04-01", "2023-07-01"),
    "q3": ("2023-07-01", "2023-10-01"),
    "q4": ("2023-10-01", SELECTION_END),
    "full_2023": ("2023-01-01", SELECTION_END),
}

WEIGHT_SETS = {
    "radial": (1.0, 0.7, -0.1, -0.2, -0.3),
    "near": (1.0, 0.5, 0.0, 0.0, 0.0),
    "decay": (1.0, 0.8, 0.6, 0.4, 0.2),
}
BASE_QUANTILES = (0.95, 0.975, 0.99)
BASE_HOLDS = (12, 36, 72, 144, 288)
REFINEMENT_QUANTILES = (0.985, 0.99, 0.995)
REFINEMENT_HOLDS = (216, 288, 360, 432, 576)
EXPECTED_GRID_CELLS = 104
EXPECTED_SELECTED = {
    "feature": "near_plain",
    "quantile": 0.985,
    "threshold": 4.434387570833191,
    "hold_bars": 288,
}
EXPECTED_STATS = {
    "fit_2023h1": {
        "absolute_return_pct": 40.84070068783197,
        "cagr_pct": 99.5845355087271,
        "strict_mdd_pct": 6.317967347813025,
        "cagr_to_strict_mdd": 15.76211620390828,
        "trades": 107,
    },
    "selection_2023h2": {
        "absolute_return_pct": 14.31176455956269,
        "cagr_pct": 30.410827498585036,
        "strict_mdd_pct": 9.281453963374265,
        "cagr_to_strict_mdd": 3.276515470376713,
        "trades": 131,
    },
    "full_2023": {
        "absolute_return_pct": 60.997490174312844,
        "cagr_pct": 61.050012436714574,
        "strict_mdd_pct": 9.281453963374254,
        "cagr_to_strict_mdd": 6.577634568638206,
        "trades": 238,
    },
}


@dataclass(frozen=True)
class Config:
    shells_csv: str = DEFAULT_SHELLS
    credibility_csv: str = DEFAULT_CREDIBILITY
    market_manifest: str = DEFAULT_MARKET_MANIFEST
    funding_csv: str = DEFAULT_FUNDING
    output: str = DEFAULT_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST
    docs_output: str = DEFAULT_DOCS
    robust_window_bars: int = 8_640
    robust_min_periods: int = 2_016
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001


def resolve_existing(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / path
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_complete(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    parsed = values.astype("string").str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError("source_complete contains an unknown value")
    return parsed.astype(bool)


def _validate_2023_grid(frame: pd.DataFrame, label: str) -> None:
    dates = pd.to_datetime(frame["date"])
    if len(dates) != 105_120:
        raise ValueError(f"{label} does not contain every calendar-2023 5m row")
    if dates.iloc[0] != pd.Timestamp("2023-01-01") or dates.iloc[-1] != pd.Timestamp("2023-12-31 23:55"):
        raise ValueError(f"{label} calendar bounds differ")
    if not dates.diff().dropna().eq(pd.Timedelta("5min")).all():
        raise ValueError(f"{label} is not a complete 5m grid")
    if dates.max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError(f"{label} opened a post-2023 row")


def load_sources(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    shell_path = resolve_existing(cfg.shells_csv)
    credibility_path = resolve_existing(cfg.credibility_csv)
    shells = pd.read_csv(shell_path, compression="infer", parse_dates=["date"])
    credibility = pd.read_csv(credibility_path, compression="infer", parse_dates=["date"])
    for frame, label in ((shells, "shells"), (credibility, "credibility")):
        _validate_2023_grid(frame, label)
        frame["source_complete"] = _parse_complete(frame["source_complete"])
    if not np.array_equal(shells["date"].to_numpy(), credibility["date"].to_numpy()):
        raise RuntimeError("shell and credibility grids differ")

    manifest_path = resolve_existing(cfg.market_manifest)
    market_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    market_path = resolve_existing(market_manifest["combined_output"])
    if sha256(market_path) != market_manifest["combined_sha256"]:
        raise RuntimeError("frozen execution-market hash mismatch")
    market = pd.read_csv(market_path, compression="infer", parse_dates=["date"])
    market = market[(market["date"] >= "2023-01-01") & (market["date"] < SELECTION_END)].reset_index(drop=True)
    _validate_2023_grid(market, "market")
    if not np.array_equal(shells["date"].to_numpy(), market["date"].to_numpy()):
        raise RuntimeError("book and execution-market grids differ")

    funding_path = resolve_existing(cfg.funding_csv)
    funding = pd.read_csv(funding_path, compression="infer")[["date", "funding_rate"]]
    funding["date"] = pd.to_datetime(funding["date"], utc=True, errors="raise", format="mixed").dt.tz_convert(None)
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    funding = funding[(funding["date"] >= "2023-01-01") & (funding["date"] < SELECTION_END)]
    funding = funding.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return shells, credibility, market, funding, {
        "shells_sha256": sha256(shell_path),
        "credibility_sha256": sha256(credibility_path),
        "market_manifest_sha256": sha256(manifest_path),
        "market_sha256": sha256(market_path),
        "funding_prefix_hash": hashlib.sha256(
            funding.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode()
        ).hexdigest(),
    }


def raw_pressure(
    shells: pd.DataFrame,
    credibility: pd.DataFrame,
    *,
    venue: str,
    weights: tuple[float, ...],
    credibility_weighted: bool,
) -> pd.Series:
    if venue not in ("um", "cm"):
        raise ValueError("venue must be um or cm")
    side_values: list[pd.Series] = []
    for side in ("m", "p"):
        value = pd.Series(0.0, index=shells.index)
        for shell, weight in enumerate(weights, 1):
            net = shells[f"{venue}_shell_flow_net_{side}{shell}"].astype(float)
            if credibility_weighted:
                efficiency = shells[f"{venue}_shell_flow_efficiency_{side}{shell}"].astype(float)
                flicker = credibility[f"{venue}_log_step_{side}{shell}"].astype(float)
                net = net * efficiency / (1.0 + 20.0 * flicker)
            value = value + float(weight) * net
        side_values.append(value)
    complete = shells["source_complete"].astype(bool) & credibility["source_complete"].astype(bool)
    return (side_values[0] - side_values[1]).where(complete)


def build_scores(shells: pd.DataFrame, credibility: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    scores: dict[str, pd.Series] = {}
    for family, weights in WEIGHT_SETS.items():
        for weighted in (False, True):
            venue_scores: list[pd.Series] = []
            for venue in ("um", "cm"):
                pressure = raw_pressure(
                    shells,
                    credibility,
                    venue=venue,
                    weights=weights,
                    credibility_weighted=weighted,
                )
                venue_scores.append(
                    lagged_robust_zscore(
                        pressure,
                        window=cfg.robust_window_bars,
                        minimum=cfg.robust_min_periods,
                    )
                )
            score = (venue_scores[0] + venue_scores[1]) / np.sqrt(2.0)
            score = score.where(shells["source_complete"].astype(bool) & score.notna())
            scores[f"{family}_{'cred' if weighted else 'plain'}"] = score
    return scores


def event_mask(score: pd.Series, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    strong = score.abs().ge(float(threshold)) & score.notna()
    side = np.sign(score).fillna(0.0).astype(np.int8)
    onset = strong & (~strong.shift(1, fill_value=False) | side.ne(side.shift(1, fill_value=0)))
    return onset.to_numpy(bool), side.to_numpy(np.int8)


def schedule(
    market: pd.DataFrame,
    engine: ExecutionEngine,
    score: pd.Series,
    *,
    threshold: float,
    hold_bars: int,
    start: str,
    end: str,
) -> list[Trade]:
    onset, side = event_mask(score, threshold)
    dates = pd.to_datetime(market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    for signal in np.flatnonzero(onset & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        trade = engine.trade_at(signal, int(side[signal]), int(hold_bars), 1_000_000, 1_000_000)
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def candidate_grid() -> list[dict[str, Any]]:
    cells: dict[tuple[str, float, int], dict[str, Any]] = {}
    for feature in (f"{family}_{suffix}" for family in WEIGHT_SETS for suffix in ("plain", "cred")):
        for quantile in BASE_QUANTILES:
            for hold in BASE_HOLDS:
                cells[(feature, quantile, hold)] = {
                    "feature": feature,
                    "quantile": quantile,
                    "hold_bars": hold,
                    "stage": "base",
                }
    for quantile in REFINEMENT_QUANTILES:
        for hold in REFINEMENT_HOLDS:
            cells[("near_plain", quantile, hold)] = {
                "feature": "near_plain",
                "quantile": quantile,
                "hold_bars": hold,
                "stage": "refinement",
            }
    if len(cells) != EXPECTED_GRID_CELLS:
        raise RuntimeError("candidate-grid multiplicity drifted")
    return list(cells.values())


def slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "schedule_hash"}


def evaluate_cell(
    cell: dict[str, Any],
    scores: dict[str, pd.Series],
    market: pd.DataFrame,
    engine: ExecutionEngine,
    execution_cfg: ExecutionConfig,
) -> dict[str, Any]:
    score = scores[cell["feature"]]
    fit = (market["date"] >= "2023-01-01") & (market["date"] < "2023-07-01")
    reference = score.loc[fit].dropna().abs()
    threshold = float(reference.quantile(float(cell["quantile"])))
    result = {**cell, "threshold": threshold, "stats": {}, "schedule_hashes": {}}
    for name, (start, end) in WINDOWS.items():
        trades = schedule(
            market,
            engine,
            score,
            threshold=threshold,
            hold_bars=int(cell["hold_bars"]),
            start=start,
            end=end,
        )
        result["stats"][name] = equity_stats(trades, start=start, end=end, cfg=execution_cfg)
        result["schedule_hashes"][name] = _schedule_hash(trades)
    half = [result["stats"][name] for name in ("fit_2023h1", "selection_2023h2")]
    result["support_pass"] = all(
        row["trades"] >= 40 and row["longs"] >= 15 and row["shorts"] >= 15 for row in half
    )
    result["half_gate_pass"] = result["support_pass"] and all(
        row["cagr_to_strict_mdd"] >= 3.0 for row in half
    )
    result["worst_half_ratio"] = min(row["cagr_to_strict_mdd"] for row in half)
    result["minimum_quarter_ratio"] = min(
        result["stats"][name]["cagr_to_strict_mdd"] for name in ("q1", "q2", "q3", "q4")
    )
    return result


def assert_selected(selected: dict[str, Any]) -> None:
    for key, expected in EXPECTED_SELECTED.items():
        actual = selected[key]
        if isinstance(expected, float):
            if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-15):
                raise RuntimeError(f"selected {key} drifted")
        elif actual != expected:
            raise RuntimeError(f"selected {key} drifted")
    for window, expected in EXPECTED_STATS.items():
        actual = selected["stats"][window]
        for key, value in expected.items():
            if isinstance(value, int):
                if actual[key] != value:
                    raise RuntimeError(f"selected {window} {key} drifted")
            elif not np.isclose(float(actual[key]), value, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"selected {window} {key} drifted")


def render_docs(payload: dict[str, Any]) -> str:
    selected = payload["selected"]
    lines = [
        "# Cross-collateral near-pressure pre-2024 selection",
        "",
        "The candidate uses no funding, premium, REX, price, or return input.  It measures the "
        "signed near-book (1%-2%) net depth flow in USD-M and COIN-M, standardizes each venue "
        "against a strictly lagged robust 30-day baseline, and combines their pressure.",
        "",
        "## Frozen policy",
        "",
        f"- Feature: `{selected['feature']}`",
        f"- H1-only absolute-score quantile: `{selected['quantile']}` = `{selected['threshold']:.15f}`",
        f"- Entry: threshold-onset side at next 5-minute open; fixed hold `{selected['hold_bars']}` bars",
        "- 0.5x, 6 bp/notional/side, realized funding, no TP/SL, non-overlap, split-contained exits",
        f"- Multiplicity disclosed: `{payload['grid_cells']}` unique cells; 2024+ remained unopened",
        "",
        "## Results",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in WINDOWS:
        stats = selected["stats"][name]
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Both H1 fit and H2 selection clear CAGR/strict-MDD 3, and every 2023 quarter is positive.",
            "- Q3 is the weak block (ratio about 1.35), so this is not live-grade evidence by itself.",
            "- Selection used 104 cells and one year of book data; overfit risk remains high until 2024+ OOS.",
            "- The policy is now immutable. Future data may reject it but cannot change its formula, threshold, or hold.",
            "",
        ]
    )
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    shells, credibility, market, funding, source = load_sources(cfg)
    scores = build_scores(shells, credibility, cfg)
    execution_cfg = ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv=str(resolve_existing(cfg.funding_csv)),
        output="/tmp/no_write_ccnp.json",
        manifest_output="/tmp/no_write_ccnp_manifest.json",
        exclude_from=SELECTION_END,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    engine = ExecutionEngine(market, funding, execution_cfg)
    rows = [evaluate_cell(cell, scores, market, engine, execution_cfg) for cell in candidate_grid()]
    rows.sort(
        key=lambda row: (
            row["half_gate_pass"],
            row["worst_half_ratio"],
            row["stats"]["full_2023"]["cagr_to_strict_mdd"],
            row["stats"]["full_2023"]["trades"],
        ),
        reverse=True,
    )
    selected = rows[0]
    assert_selected(selected)
    payload = {
        "schema_version": 1,
        "mode": "cross_collateral_near_pressure_pre2024_selection",
        "selection_cutoff_exclusive": SELECTION_END,
        "post_2023_rows_opened": False,
        "config": asdict(cfg),
        "source": source,
        "grid_cells": len(rows),
        "selection_rule": "half gate, worst-half ratio, full-2023 ratio, trade count",
        "selected": selected,
        "top10": rows[:10],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_payload = {
        "schema_version": 1,
        "selection_cutoff_exclusive": SELECTION_END,
        "future_windows": {
            "test_2024": ["2024-01-01", "2025-01-01"],
            "eval_2025": ["2025-01-01", "2026-01-01"],
            "holdout_2026h1": ["2026-01-01", "2026-06-02"],
        },
        "future_outcomes_opened": False,
        "selection_result": str(output),
        "selection_result_sha256": sha256(output),
        "selected_spec": {key: selected[key] for key in ("feature", "quantile", "threshold", "hold_bars")},
        "selected_schedule_hashes": selected["schedule_hashes"],
        "source": source,
        "grid_cells": len(rows),
    }
    manifest_payload["manifest_hash"] = json_hash(manifest_payload)
    manifest = Path(cfg.manifest_output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cfg.docs_output:
        docs = Path(cfg.docs_output)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(render_docs(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shells-csv", default=DEFAULT_SHELLS)
    parser.add_argument("--credibility-csv", default=DEFAULT_CREDIBILITY)
    parser.add_argument("--market-manifest", default=DEFAULT_MARKET_MANIFEST)
    parser.add_argument("--funding-csv", default=DEFAULT_FUNDING)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(json.dumps({"grid_cells": payload["grid_cells"], "selected": payload["selected"]}, indent=2))


if __name__ == "__main__":
    main()
