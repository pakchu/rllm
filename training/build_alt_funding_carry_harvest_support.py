"""Build the outcome-blind AFCH01 2023-2025 weekly sleeve clock."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.compose_alt_funding_carry_sources import (
    END,
    EXPECTED_PROTOCOL_HASH,
    HANDOFF,
    LORC_DIR,
    LORE_DIR,
    START,
    SYMBOLS,
)
from training.export_leave_one_out_residual_exhaustion_sources import deterministic_csv_gz, sha256_file
from training.preregister_alt_funding_carry_harvest import canonical_hash, protocol


SOURCE_COMPOSITION = "results/alt_funding_carry_harvest_v1_source_composition_2026-07-17.json"
EXPECTED_SOURCE_COMPOSITION_HASH = "18f5455ea248b6ab5e451d734c0d6849ed25a4e67aa19970e19617f95f67fb0d"
DEFAULT_CLOCK = "data/alt_funding_carry_harvest_v1_support_clock_2023_2025.csv.gz"
DEFAULT_MANIFEST = "results/alt_funding_carry_harvest_v1_support_manifest_2026-07-17.json"
DEFAULT_DOCS = "docs/alt-funding-carry-harvest-v1-support-2026-07-17.md"
POLICY_ID = "AFCH01"
FUNDING_LOOKBACK = pd.Timedelta(days=28)
HOLD = pd.Timedelta(days=28)
SLEEVE_GROSS = 0.25
MIN_PROJECTED_CARRY = 0.0018
MIN_LEG_WEIGHT = 0.25
FORBIDDEN_OUTPUT_TOKENS = {
    "future", "return", "pnl", "profit", "high", "low", "open", "close",
    "cagr", "mdd", "target", "label",
}
CLOCK_COLUMNS = (
    "policy_id", "signal_time", "feature_available_time", "entry_time", "exit_time",
    "hold_days", "sleeve_gross", "long_symbol", "short_symbol",
    "long_weight_norm", "short_weight_norm", "long_beta", "short_beta",
    "long_trailing_funding_sum", "short_trailing_funding_sum", "projected_28d_carry",
)


def _manifest(path: str, expected: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != expected:
        raise RuntimeError(f"manifest hash changed: {path}")
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != expected:
        raise RuntimeError(f"manifest body changed: {path}")
    return payload


def _composite_market(symbol: str) -> pd.DataFrame:
    columns = ["date", "close", "quote_asset_volume", "tic"]
    old = pd.read_csv(LORE_DIR / f"{symbol}_5m_2023_2024.csv.gz", usecols=columns, parse_dates=["date"])
    recent = pd.read_csv(LORC_DIR / f"{symbol}_5m_2024_2025.csv.gz", usecols=columns, parse_dates=["date"])
    frame = pd.concat([
        old.loc[(old["date"] >= START) & (old["date"] < HANDOFF)],
        recent.loc[(recent["date"] >= HANDOFF) & (recent["date"] < END)],
    ], ignore_index=True).sort_values("date")
    if frame["date"].duplicated().any() or not frame["tic"].astype(str).eq(symbol).all():
        raise RuntimeError(f"{symbol} AFCH composite market identity failure")
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise RuntimeError(f"{symbol} AFCH composite market grid failure")
    return frame


def _composite_funding(symbol: str) -> pd.Series:
    columns = ["funding_time", "funding_rate", "symbol"]
    old = pd.read_csv(LORE_DIR / f"{symbol}_funding_2023_2024.csv.gz", usecols=columns)
    recent = pd.read_csv(LORC_DIR / f"{symbol}_funding_2024_2025.csv.gz", usecols=columns)
    for frame in (old, recent):
        if not frame["symbol"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} AFCH funding identity failure")
        frame["event_time"] = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="raise"), unit="ms")
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    frame = pd.concat([
        old.loc[(old["event_time"] >= START) & (old["event_time"] < HANDOFF)],
        recent.loc[(recent["event_time"] >= HANDOFF) & (recent["event_time"] < END)],
    ], ignore_index=True).sort_values("event_time")
    if frame["event_time"].duplicated().any() or not frame["event_time"].is_monotonic_increasing:
        raise RuntimeError(f"{symbol} AFCH funding composition failure")
    return pd.Series(frame["funding_rate"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame["event_time"]), name=symbol)


def load_feature_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    close: dict[str, pd.Series] = {}
    clean: dict[str, pd.Series] = {}
    funding: dict[str, pd.Series] = {}
    for symbol in sorted(SYMBOLS):
        market = _composite_market(symbol).set_index("date")
        hourly = pd.DataFrame({
            "close": market["close"].astype(float).resample("1h", closed="left", label="right").last(),
            "bar_count": market["close"].resample("1h", closed="left", label="right").count(),
            "positive_quote": market["quote_asset_volume"].gt(0).resample("1h", closed="left", label="right").sum(),
        })
        close[symbol] = hourly["close"]
        clean[symbol] = (hourly["bar_count"].eq(12) & hourly["positive_quote"].eq(12) & hourly["close"].gt(0)).rename(symbol)
        funding[symbol] = _composite_funding(symbol)
    close_frame = pd.DataFrame(close)
    quality = pd.DataFrame(clean, index=close_frame.index).astype(bool)
    expected_hours = pd.date_range(START + pd.Timedelta(hours=1), END, freq="1h")
    if not close_frame.index.equals(expected_hours) or not quality.index.equals(expected_hours):
        raise RuntimeError("AFCH hourly grid escaped 2023-2025")
    return close_frame, quality, funding


def causal_beta(close: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(close).diff()
    clean_returns = quality & quality.shift(1, fill_value=False)
    returns = returns.where(clean_returns)
    factors = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for symbol in returns.columns:
        factors[symbol] = returns.drop(columns=symbol).median(axis=1)
    beta = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for symbol in returns.columns:
        covariance = returns[symbol].rolling(720, min_periods=336).cov(factors[symbol])
        variance = factors[symbol].rolling(720, min_periods=336).var()
        beta[symbol] = (covariance / variance.replace(0.0, np.nan)).shift(1).clip(0.25, 2.5)
    return beta


def weekly_anchors() -> pd.DatetimeIndex:
    first = (START + FUNDING_LOOKBACK).normalize()
    while first.weekday() != 0:
        first += pd.Timedelta(days=1)
    first += pd.Timedelta(minutes=5)
    last = END - pd.Timedelta(days=3, minutes=-5)
    return pd.date_range(first, last, freq="7D")


def build_clock(beta: pd.DataFrame, quality: pd.DataFrame, funding: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    symbols = list(sorted(SYMBOLS))
    for signal in weekly_anchors():
        feature_hour = signal.floor("h")
        if feature_hour not in beta.index or not bool(quality.loc[feature_hour, symbols].all()):
            continue
        current_beta = beta.loc[feature_hour, symbols]
        if current_beta.isna().any() or not np.isfinite(current_beta.to_numpy(dtype=float)).all():
            continue
        sums = {
            symbol: float(series.loc[(series.index > signal - FUNDING_LOOKBACK) & (series.index <= signal)].sum())
            for symbol, series in funding.items()
        }
        high_symbol = max(symbols, key=lambda symbol: sums[symbol])
        low_symbol = min(symbols, key=lambda symbol: sums[symbol])
        high_beta, low_beta = float(current_beta[high_symbol]), float(current_beta[low_symbol])
        denominator = high_beta + low_beta
        long_weight = high_beta / denominator
        short_weight = low_beta / denominator
        projected = short_weight * sums[high_symbol] - long_weight * sums[low_symbol]
        if min(long_weight, short_weight) < MIN_LEG_WEIGHT or projected < MIN_PROJECTED_CARRY:
            continue
        entry = signal + pd.Timedelta(minutes=5)
        exit_time = entry + HOLD
        if exit_time >= END:
            continue
        rows.append({
            "policy_id": POLICY_ID,
            "signal_time": signal,
            "feature_available_time": signal,
            "entry_time": entry,
            "exit_time": exit_time,
            "hold_days": 28,
            "sleeve_gross": SLEEVE_GROSS,
            "long_symbol": low_symbol,
            "short_symbol": high_symbol,
            "long_weight_norm": long_weight,
            "short_weight_norm": short_weight,
            "long_beta": low_beta,
            "short_beta": high_beta,
            "long_trailing_funding_sum": sums[low_symbol],
            "short_trailing_funding_sum": sums[high_symbol],
            "projected_28d_carry": projected,
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def max_active_sleeves(clock: pd.DataFrame) -> int:
    points: list[tuple[pd.Timestamp, int]] = []
    for row in clock.itertuples(index=False):
        points.append((pd.Timestamp(row.entry_time), 1))
        points.append((pd.Timestamp(row.exit_time), -1))
    # Exits are processed before entries at an equal timestamp.
    active = maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if clock.empty:
        raise RuntimeError("empty AFCH support clock")
    for column in clock.columns:
        if set(column.lower().split("_")) & FORBIDDEN_OUTPUT_TOKENS:
            raise RuntimeError(f"outcome-like AFCH clock column forbidden: {column}")
    for column in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    if not clock["policy_id"].eq(POLICY_ID).all():
        raise RuntimeError("AFCH policy drift")
    if not (clock["feature_available_time"] == clock["signal_time"]).all():
        raise RuntimeError("AFCH feature availability drift")
    if not (clock["entry_time"] == clock["signal_time"] + pd.Timedelta(minutes=5)).all():
        raise RuntimeError("AFCH entry delay drift")
    if not (clock["exit_time"] == clock["entry_time"] + HOLD).all():
        raise RuntimeError("AFCH hold drift")
    if not ((clock["signal_time"] >= START) & (clock["exit_time"] < END)).all():
        raise RuntimeError("AFCH clock escaped 2023-2025")
    if not clock["signal_time"].dt.weekday.eq(0).all() or not (
        clock["signal_time"].dt.hour.eq(0) & clock["signal_time"].dt.minute.eq(5)
    ).all():
        raise RuntimeError("AFCH weekly signal clock drift")
    if not np.allclose(clock["long_weight_norm"] + clock["short_weight_norm"], 1.0, atol=1e-12):
        raise RuntimeError("AFCH normalized gross drift")
    exposure = clock["long_weight_norm"] * clock["long_beta"] - clock["short_weight_norm"] * clock["short_beta"]
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("AFCH factor beta drift")
    if (clock[["long_weight_norm", "short_weight_norm"]].min(axis=1) < MIN_LEG_WEIGHT - 1e-12).any():
        raise RuntimeError("AFCH minimum leg weight drift")
    if (clock["projected_28d_carry"] < MIN_PROJECTED_CARRY - 1e-12).any():
        raise RuntimeError("AFCH projected carry hurdle drift")
    if max_active_sleeves(clock) > 4:
        raise RuntimeError("AFCH active sleeve cap drift")


def support_stats(clock: pd.DataFrame, max_monthly_quarantine: float) -> dict[str, Any]:
    pair_counts = Counter(clock["short_symbol"] + ">" + clock["long_symbol"])
    n = len(clock)
    year_counts = clock.groupby(clock["signal_time"].dt.year).size().to_dict()
    half_counts = {
        f"{year}H{half}": int(((clock["signal_time"].dt.year == year) & (((clock["signal_time"].dt.month - 1) // 6 + 1) == half)).sum())
        for year in (2023, 2024, 2025) for half in (1, 2)
    }
    stats: dict[str, Any] = {
        "events": n,
        "year_counts": {str(year): int(count) for year, count in year_counts.items()},
        "half_year_counts": half_counts,
        "unique_ordered_pairs": len(pair_counts),
        "maximum_ordered_pair_share": max(pair_counts.values()) / n,
        "ordered_pair_counts": dict(sorted(pair_counts.items())),
        "long_symbols": sorted(clock["long_symbol"].unique()),
        "short_symbols": sorted(clock["short_symbol"].unique()),
        "maximum_active_sleeves": max_active_sleeves(clock),
        "minimum_projected_28d_carry_bp": float(clock["projected_28d_carry"].min() * 10_000.0),
        "median_projected_28d_carry_bp": float(clock["projected_28d_carry"].median() * 10_000.0),
        "maximum_monthly_source_quarantine": max_monthly_quarantine,
    }
    gates = {
        "combined_events_at_least_110": n >= 110,
        "each_year_events_at_least_35": all(year_counts.get(year, 0) >= 35 for year in (2023, 2024, 2025)),
        "each_half_year_events_at_least_12": all(count >= 12 for count in half_counts.values()),
        "unique_ordered_pairs_at_least_8": len(pair_counts) >= 8,
        "maximum_ordered_pair_share_at_most_0_25": stats["maximum_ordered_pair_share"] <= 0.25,
        "long_symbols_at_least_3": len(stats["long_symbols"]) >= 3,
        "short_symbols_at_least_5": len(stats["short_symbols"]) >= 5,
        "maximum_active_sleeves_at_most_4": stats["maximum_active_sleeves"] <= 4,
        "monthly_source_quarantine_at_most_0_01": max_monthly_quarantine <= 0.01,
    }
    stats["gates"] = gates
    stats["passes_support"] = all(gates.values())
    return stats


def _markdown(result: dict[str, Any]) -> str:
    stats = result["support"]
    pairs = "\n".join(f"| {pair} | {count} |" for pair, count in stats["ordered_pair_counts"].items())
    return f"""# AFCH v1 outcome-blind support — 2026-07-17

> Exact funding features and sleeve clocks only. No post-entry price return, PnL, CAGR, or MDD was calculated.

## Support result

- events: `{stats['events']}`; years `{stats['year_counts']}`
- halves: `{stats['half_year_counts']}`
- unique ordered high>low pairs: `{stats['unique_ordered_pairs']}`
- maximum pair share: `{stats['maximum_ordered_pair_share']:.4%}`
- long symbols: `{stats['long_symbols']}`
- short symbols: `{stats['short_symbols']}`
- maximum active sleeves: `{stats['maximum_active_sleeves']}`
- projected 28d carry: minimum `{stats['minimum_projected_28d_carry_bp']:.3f} bp`, median `{stats['median_projected_28d_carry_bp']:.3f} bp`
- maximum monthly source quarantine: `{stats['maximum_monthly_source_quarantine']:.4%}`
- decision: **{'PASS' if stats['passes_support'] else 'REJECT'}**
- clock SHA-256: `{result['clock_sha256']}`

| Short high-funding > long low-funding | Sleeves |
|---|---:|
{pairs}

Each row is fixed before outcomes, enters at Monday 00:10 UTC, holds 28 days,
uses a 0.25-gross sleeve, and has projected carry of at least 18 bp at normalized
gross one. At most four vintages overlap.
"""


def run(
    clock_path: str = DEFAULT_CLOCK,
    manifest_path: str = DEFAULT_MANIFEST,
    docs_path: str = DEFAULT_DOCS,
) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("AFCH preregistration drifted")
    source = _manifest(SOURCE_COMPOSITION, EXPECTED_SOURCE_COMPOSITION_HASH)
    close, quality, funding = load_feature_sources()
    beta = causal_beta(close, quality)
    clock = build_clock(beta, quality, funding)
    assert_clock_contract(clock)
    source_quality = quality.all(axis=1).loc[(quality.index >= START + pd.Timedelta(hours=1)) & (quality.index < END)]
    monthly_quarantine = (~source_quality).groupby(source_quality.index.to_period("M")).mean()
    stats = support_stats(clock, float(monthly_quarantine.max()))
    if not stats["passes_support"]:
        raise RuntimeError(f"AFCH support failed: {stats['gates']}")
    deterministic_csv_gz(clock, Path(clock_path))
    reread = pd.read_csv(clock_path)
    assert_clock_contract(reread)
    result: dict[str, Any] = {
        "protocol_version": "afch_v1_support_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_composition_hash": EXPECTED_SOURCE_COMPOSITION_HASH,
        "post_entry_returns_calculated": False,
        "fit_2023_opened": False,
        "test_2024_opened": False,
        "eval_2025_opened": False,
        "final_2026_opened": False,
        "source_hours": int(len(source_quality)),
        "quarantined_hours": int((~source_quality).sum()),
        "clock_path": clock_path,
        "clock_sha256": sha256_file(Path(clock_path)),
        "support": stats,
        "source_manifest": source["manifest_hash"],
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_path).write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", default=DEFAULT_CLOCK)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    args = parser.parse_args()
    print(json.dumps(run(args.clock, args.manifest, args.docs), indent=2))


if __name__ == "__main__":
    main()
