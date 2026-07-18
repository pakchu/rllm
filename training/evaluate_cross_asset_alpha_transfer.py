"""Evaluate frozen daily alpha translations on QQQ, KODEX200, and GLD.

The policy specification is frozen by
``results/cross_asset_alpha_transfer_preregistration_2026-07-19.json``.  The
evaluator downloads adjusted daily OHLCV, fits only per-instrument train
quantiles, and reports every policy without test/eval reranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import t as student_t

from training import preregister_cross_asset_alpha_transfer as prereg


OUTPUT = "results/cross_asset_alpha_transfer_evaluation_2026-07-19.json"
DOCS_OUTPUT = "docs/cross-asset-alpha-transfer-evaluation-2026-07-19.md"
CACHE_DIR = "data/cache_cross_asset_alpha_transfer"
PERIOD1 = 946684800  # 2000-01-01T00:00:00Z
PERIOD2 = 1784419200  # 2026-07-19T00:00:00Z, exclusive
INSTRUMENTS = ("QQQ", "069500.KS", "GLD")
POLICIES = (
    "rex_pullback_reclaim_session",
    "rex_multiscale_extreme_fade_session",
    "persistent_barrier_mass_density_fade_session",
)
SPLITS = {
    "train": ("2007-01-29", "2017-01-01"),
    "test": ("2017-01-01", "2022-01-01"),
    "eval": ("2022-01-01", "2026-07-19"),
}


@dataclass(frozen=True)
class Trade:
    signal_index: int
    entry_index: int
    exit_index: int
    side: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def yahoo_url(symbol: str) -> str:
    encoded = urllib.parse.quote(symbol, safe="")
    query = urllib.parse.urlencode(
        {
            "period1": PERIOD1,
            "period2": PERIOD2,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?{query}"


def download_payload(symbol: str, cache_dir: str = CACHE_DIR, *, refresh: bool = False) -> tuple[bytes, str]:
    cache = Path(cache_dir) / f"{urllib.parse.quote(symbol, safe='')}.json"
    if cache.exists() and not refresh:
        raw = cache.read_bytes()
        return raw, "cache"
    request = urllib.request.Request(yahoo_url(symbol), headers={"User-Agent": "Mozilla/5.0 rllm-research"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - frozen public endpoint
        raw = response.read()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(raw)
    return raw, "download"


def parse_yahoo_payload(raw: bytes, symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(raw)
    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {chart['error']}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise RuntimeError(f"Yahoo chart result count for {symbol}: {len(results)}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_rows = ((result.get("indicators") or {}).get("quote") or [])
    adj_rows = ((result.get("indicators") or {}).get("adjclose") or [])
    if len(quote_rows) != 1 or not timestamps:
        raise RuntimeError(f"Yahoo chart payload is incomplete for {symbol}")
    quote = quote_rows[0]
    adjclose = adj_rows[0].get("adjclose") if len(adj_rows) == 1 else quote.get("close")
    columns = {name: quote.get(name) for name in ("open", "high", "low", "close", "volume")}
    if adjclose is None or any(values is None or len(values) != len(timestamps) for values in columns.values()):
        raise RuntimeError(f"Yahoo chart vector length mismatch for {symbol}")
    if len(adjclose) != len(timestamps):
        raise RuntimeError(f"Yahoo adjusted-close length mismatch for {symbol}")

    timezone = str((result.get("meta") or {}).get("exchangeTimezoneName") or "UTC")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone).tz_localize(None).normalize()
    frame = pd.DataFrame({"date": dates, **columns, "adjclose": adjclose})
    for name in ("open", "high", "low", "close", "volume", "adjclose"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    factor = frame["adjclose"] / frame["close"].replace(0.0, np.nan)
    for name in ("open", "high", "low", "close"):
        frame[name] = frame[name] * factor
    frame = frame.drop(columns=["adjclose"])
    valid = np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=1)
    valid &= (frame[["open", "high", "low", "close"]] > 0.0).all(axis=1)
    valid &= frame["volume"] >= 0.0
    frame["source_valid"] = np.asarray(valid, dtype=bool)

    # Yahoo's KODEX history contains a short valid prefix followed by a 549-row
    # null quote block (2007-02-08 through 2009-04-16).  Treat a long null block
    # as an unusable listing prefix, retain the first valid suffix, and preserve
    # all later null rows as explicit gaps.  No price or return value influences
    # this date/availability-only recovery rule.
    invalid = ~frame["source_valid"].to_numpy(bool)
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_invalid in enumerate(invalid):
        if is_invalid and run_start is None:
            run_start = index
        if run_start is not None and (not is_invalid or index == len(invalid) - 1):
            stop = index if not is_invalid else index + 1
            runs.append((run_start, stop))
            run_start = None
    longest = max(runs, key=lambda row: row[1] - row[0], default=(0, 0))
    discarded_prefix_rows = 0
    discarded_prefix_through: str | None = None
    if longest[1] - longest[0] >= 20:
        suffix = frame.index[(frame.index >= longest[1]) & frame["source_valid"]]
        if not len(suffix):
            raise RuntimeError(f"Yahoo source has no valid suffix after long null block for {symbol}")
        start_index = int(suffix[0])
        discarded_prefix_rows = start_index
        discarded_prefix_through = frame["date"].iloc[start_index - 1].strftime("%Y-%m-%d")
        frame = frame.iloc[start_index:].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty or not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise RuntimeError(f"normalized Yahoo frame order/uniqueness failed for {symbol}")
    clean = frame.loc[frame["source_valid"]]
    high_floor = clean[["open", "close"]].max(axis=1)
    low_ceiling = clean[["open", "close"]].min(axis=1)
    if not ((clean["high"] >= high_floor) & (clean["low"] <= low_ceiling)).all():
        raise RuntimeError(f"normalized Yahoo OHLC geometry failed for {symbol}")
    gaps = frame["date"].diff().dropna()
    if len(gaps) and gaps.max() > pd.Timedelta(days=14):
        raise RuntimeError(f"unexpected daily source gap for {symbol}: {gaps.max()}")
    invalid_dates = frame.loc[~frame["source_valid"], "date"].dt.strftime("%Y-%m-%d").tolist()
    meta = {
        "symbol": symbol,
        "source_url": yahoo_url(symbol),
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "exchange_timezone": timezone,
        "rows": len(frame),
        "valid_rows": int(frame["source_valid"].sum()),
        "invalid_rows_quarantined": int((~frame["source_valid"]).sum()),
        "invalid_dates_sha256": _sha256(json.dumps(invalid_dates, separators=(",", ":")).encode()),
        "discarded_unusable_prefix_rows": discarded_prefix_rows,
        "discarded_unusable_prefix_through": discarded_prefix_through,
        "start": frame["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "max_calendar_gap_days": int(gaps.max().days) if len(gaps) else 0,
    }
    return frame, meta


def load_market(symbol: str, cache_dir: str = CACHE_DIR, *, refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, mode = download_payload(symbol, cache_dir, refresh=refresh)
    frame, meta = parse_yahoo_payload(raw, symbol)
    meta["load_mode"] = mode
    return frame, meta


def _range_location(frame: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = frame["high"].rolling(window, min_periods=window).max()
    low = frame["low"].rolling(window, min_periods=window).min()
    close = frame["close"]
    span = (high - low).replace(0.0, np.nan)
    location = ((close - low) / span) * 2.0 - 1.0
    max_gap = high / close - 1.0
    min_gap = close / low - 1.0
    return location.to_numpy(float), max_gap.to_numpy(float), min_gap.to_numpy(float)


def rex_signal_bank(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    windows = (2, 7, 26, 111)
    locations: list[np.ndarray] = []
    max_gaps: list[np.ndarray] = []
    min_gaps: list[np.ndarray] = []
    for window in windows:
        location, max_gap, min_gap = _range_location(frame, window)
        locations.append(location)
        max_gaps.append(max_gap)
        min_gaps.append(min_gap)
    location_stack = np.vstack(locations)
    rex_loc = np.mean(location_stack, axis=0)
    rex_short_loc = np.mean(location_stack[:2], axis=0)
    rex_long_loc = np.mean(location_stack[2:], axis=0)
    rex_max_gap = np.mean(np.vstack(max_gaps), axis=0)
    rex_min_gap = np.mean(np.vstack(min_gaps), axis=0)
    close = frame["close"].astype(float)
    higher_trend = sum(close.pct_change(period, fill_method=None).to_numpy(float) for period in (4, 12, 20))
    local_trend = close.pct_change(1, fill_method=None).to_numpy(float)
    volume = frame["volume"].astype(float)
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    volume_zscore = ((volume - volume_mean) / volume_std).to_numpy(float)

    htf_side = np.nan_to_num(np.sign(higher_trend), nan=0.0).astype(np.int8)
    pullback_alignment = -np.sign(rex_loc) * htf_side
    pullback = np.maximum(0.0, pullback_alignment) * (
        np.abs(higher_trend) + 0.25 * np.abs(rex_max_gap - rex_min_gap)
    )
    local_reclaim = np.maximum(0.0, np.sign(local_trend) * htf_side)
    vol_confirm = np.maximum(0.0, volume_zscore)
    reclaim = pullback * (0.5 + local_reclaim) * (1.0 + 0.25 * vol_confirm)

    extreme = np.maximum(0.0, np.abs(rex_loc) - 0.55) * (1.0 + np.abs(rex_short_loc - rex_long_loc))
    extreme_side = -np.nan_to_num(np.sign(rex_loc), nan=0.0).astype(np.int8)
    source_valid = frame.get("source_valid", pd.Series(True, index=frame.index)).astype(bool)
    full_context = source_valid.rolling(111, min_periods=111).sum().to_numpy(float) == 111.0
    reclaim = np.where(full_context, reclaim, np.nan)
    extreme = np.where(full_context, extreme, np.nan)
    htf_side = np.where(full_context, htf_side, 0).astype(np.int8)
    extreme_side = np.where(full_context, extreme_side, 0).astype(np.int8)
    return {
        "rex_pullback_reclaim_session": (reclaim, htf_side),
        "rex_multiscale_extreme_fade_session": (extreme, extreme_side),
    }


def barrier_signal(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    horizon = 26
    log_price = np.log(frame["close"].to_numpy(float))
    daily_return = pd.Series(log_price).diff()
    scale = daily_return.shift(1).rolling(26, min_periods=13).std(ddof=0).to_numpy(float)
    score = np.full(len(frame), np.nan, dtype=float)
    traversal_side = np.zeros(len(frame), dtype=np.int8)
    source_valid = frame.get("source_valid", pd.Series(True, index=frame.index)).to_numpy(bool)
    for position in range(horizon + 1, len(frame)):
        freeze = position - 1
        if not source_valid[freeze - horizon : position + 1].all():
            continue
        if not np.isfinite(scale[freeze]) or scale[freeze] <= 0.0:
            continue
        window = log_price[freeze - horizon : freeze]
        if len(window) != horizon or not np.isfinite(window).all():
            continue
        start = log_price[freeze]
        end = log_price[position]
        minimum_prominence = 0.5 * scale[freeze]
        if end > start:
            extrema, properties = find_peaks(window, prominence=minimum_prominence)
            levels = window[extrema]
            crossed = (levels > start) & (levels <= end)
            direction = 1
        elif end < start:
            extrema, properties = find_peaks(-window, prominence=minimum_prominence)
            levels = window[extrema]
            crossed = (levels < start) & (levels >= end)
            direction = -1
        else:
            continue
        if not crossed.any():
            continue
        normalized = np.asarray(properties["prominences"], dtype=float)[crossed] / scale[freeze]
        movement = abs(end - start) / scale[freeze]
        score[position] = float(normalized.sum()) / max(movement, 1e-12)
        traversal_side[position] = direction
    return score, -traversal_side


def fit_threshold(strength: np.ndarray, dates: pd.Series, quantile: float) -> tuple[float, int]:
    start, end = SPLITS["train"]
    mask = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    values = np.asarray(strength, dtype=float)
    reference = values[np.asarray(mask) & np.isfinite(values) & (values > 0.0)]
    if len(reference) < 40:
        raise RuntimeError(f"insufficient train support for threshold: {len(reference)}")
    return float(np.quantile(reference, quantile)), int(len(reference))


def signal_clock(strength: np.ndarray, side: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(strength, dtype=float)
    directions = np.asarray(side, dtype=np.int8)
    active = np.isfinite(values) & (values > float(threshold)) & (directions != 0)
    return active, directions


def build_trades(
    active: np.ndarray,
    side: np.ndarray,
    dates: pd.Series,
    *,
    split: str,
    hold_sessions: int,
    source_valid: np.ndarray | None = None,
) -> list[Trade]:
    start, end = SPLITS[split]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    trades: list[Trade] = []
    next_free = -1
    clean = np.ones(len(dates), dtype=bool) if source_valid is None else np.asarray(source_valid, dtype=bool)
    if len(clean) != len(dates):
        raise ValueError("source_valid length mismatch")
    for signal_index in np.flatnonzero(active):
        if signal_index < next_free or not (start_ts <= dates.iloc[signal_index] < end_ts):
            continue
        entry_index = int(signal_index) + 1
        exit_index = entry_index + int(hold_sessions)
        if exit_index >= len(dates) or dates.iloc[exit_index] >= end_ts:
            continue
        if not clean[signal_index : exit_index + 1].all():
            continue
        trades.append(
            Trade(
                signal_index=int(signal_index),
                entry_index=entry_index,
                exit_index=exit_index,
                side=int(side[signal_index]),
            )
        )
        next_free = exit_index
    return trades


def _years(start: str, end: str) -> float:
    return max((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.2425 * 86400.0), 1e-9)


def simulate(
    frame: pd.DataFrame,
    trades: list[Trade],
    *,
    split: str,
    cost_bps_per_side: float,
) -> dict[str, Any]:
    cost = float(cost_bps_per_side) / 10_000.0
    equity = peak = 1.0
    strict_mdd = 0.0
    returns: list[float] = []
    records: list[dict[str, Any]] = []
    for trade in trades:
        start_equity = equity
        equity_after_entry = start_equity * (1.0 - cost)
        strict_mdd = max(strict_mdd, 1.0 - equity_after_entry / peak)
        entry = float(frame["open"].iloc[trade.entry_index])
        for position in range(trade.entry_index, trade.exit_index):
            high = float(frame["high"].iloc[position])
            low = float(frame["low"].iloc[position])
            if trade.side > 0:
                favorable = equity_after_entry * max(0.0, high / entry)
                adverse = equity_after_entry * max(0.0, low / entry)
            else:
                favorable = equity_after_entry * max(0.0, 2.0 - low / entry)
                adverse = equity_after_entry * max(0.0, 2.0 - high / entry)
            peak = max(peak, favorable)
            strict_mdd = max(strict_mdd, 1.0 - adverse / max(peak, 1e-15))
        exit_price = float(frame["open"].iloc[trade.exit_index])
        gross_factor = max(0.0, 1.0 + trade.side * (exit_price / entry - 1.0))
        equity = equity_after_entry * gross_factor * (1.0 - cost)
        strict_mdd = max(strict_mdd, 1.0 - equity / max(peak, 1e-15))
        peak = max(peak, equity)
        net_return = equity / start_equity - 1.0
        returns.append(net_return)
        records.append(
            {
                "signal_date": frame["date"].iloc[trade.signal_index].strftime("%Y-%m-%d"),
                "entry_date": frame["date"].iloc[trade.entry_index].strftime("%Y-%m-%d"),
                "exit_date": frame["date"].iloc[trade.exit_index].strftime("%Y-%m-%d"),
                "side": "LONG" if trade.side > 0 else "SHORT",
                "net_return": net_return,
            }
        )
        if equity <= 0.0:
            break
    start, end = SPLITS[split]
    years = _years(start, end)
    absolute = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd_pct = strict_mdd * 100.0
    array = np.asarray(returns, dtype=float)
    if len(array) >= 2 and np.std(array, ddof=1) > 0.0:
        statistic = float(np.mean(array) / (np.std(array, ddof=1) / math.sqrt(len(array))))
        p_value = float(student_t.sf(statistic, df=len(array) - 1))
    else:
        statistic, p_value = 0.0, 1.0
    split_years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + (pd.Timestamp(end).month > 1)))
    yearly_returns: dict[str, float] = {}
    for year in split_years:
        factors = [1.0 + row["net_return"] for row in records if pd.Timestamp(row["exit_date"]).year == year]
        yearly_returns[str(year)] = (float(np.prod(factors)) - 1.0) * 100.0 if factors else 0.0
    positive_year_share = (
        sum(value > 0.0 for value in yearly_returns.values()) / len(yearly_returns) if yearly_returns else 0.0
    )
    return {
        "absolute_return_pct": absolute,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd_pct,
        "cagr_to_strict_mdd": cagr / mdd_pct if mdd_pct > 1e-12 else 0.0,
        "trades": len(records),
        "longs": sum(row["side"] == "LONG" for row in records),
        "shorts": sum(row["side"] == "SHORT" for row in records),
        "win_rate": float(np.mean(array > 0.0)) if len(array) else 0.0,
        "mean_trade_return_pct": float(np.mean(array) * 100.0) if len(array) else 0.0,
        "one_sided_t_pvalue": p_value,
        "t_statistic": statistic,
        "yearly_return_pct": yearly_returns,
        "positive_calendar_year_share": positive_year_share,
        "trade_clock_hash": hashlib.sha256(
            json.dumps([(row["signal_date"], row["side"]) for row in records], separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _policy_inputs(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, float, int]]:
    bank = rex_signal_bank(frame)
    barrier_strength, barrier_side = barrier_signal(frame)
    return {
        "rex_pullback_reclaim_session": (*bank["rex_pullback_reclaim_session"], 0.75, 2),
        "rex_multiscale_extreme_fade_session": (*bank["rex_multiscale_extreme_fade_session"], 0.75, 2),
        "persistent_barrier_mass_density_fade_session": (barrier_strength, barrier_side, 0.975, 4),
    }


def evaluate_instrument(frame: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"])
    source_valid = frame.get("source_valid", pd.Series(True, index=frame.index)).to_numpy(bool)
    output: dict[str, Any] = {}
    for policy, (strength, side, quantile, hold) in _policy_inputs(frame).items():
        threshold, train_positive_support = fit_threshold(strength, dates, quantile)
        active, directions = signal_clock(strength, side, threshold)
        raw_counts = {
            split: int(
                np.sum(
                    active
                    & np.asarray(
                        (dates >= pd.Timestamp(bounds[0])) & (dates < pd.Timestamp(bounds[1])), dtype=bool
                    )
                )
            )
            for split, bounds in SPLITS.items()
        }
        windows: dict[str, Any] = {}
        for split in SPLITS:
            trades = build_trades(
                active,
                directions,
                dates,
                split=split,
                hold_sessions=hold,
                source_valid=source_valid,
            )
            flipped = [Trade(t.signal_index, t.entry_index, t.exit_index, -t.side) for t in trades]
            windows[split] = {
                "base_5bp": simulate(frame, trades, split=split, cost_bps_per_side=5.0),
                "stress_10bp": simulate(frame, trades, split=split, cost_bps_per_side=10.0),
                "direction_flip_5bp": simulate(frame, flipped, split=split, cost_bps_per_side=5.0),
            }
        output[policy] = {
            "threshold": threshold,
            "threshold_quantile": quantile,
            "threshold_fit_positive_events": train_positive_support,
            "hold_sessions": hold,
            "raw_signal_counts": raw_counts,
            "windows": windows,
        }
    return output


def transfer_gate(policy_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_instrument: dict[str, dict[str, Any]] = {}
    for symbol in INSTRUMENTS:
        row = policy_results[symbol]
        eval_base = row["windows"]["eval"]["base_5bp"]
        eval_stress = row["windows"]["eval"]["stress_10bp"]
        flip = row["windows"]["eval"]["direction_flip_5bp"]
        checks = {
            "positive_absolute_return": eval_base["absolute_return_pct"] > 0.0,
            "ratio_at_least_3": eval_base["cagr_to_strict_mdd"] >= 3.0,
            "strict_mdd_at_most_15": eval_base["strict_mdd_pct"] <= 15.0,
            "at_least_20_trades": eval_base["trades"] >= 20,
            "stress_positive": eval_stress["absolute_return_pct"] > 0.0,
            "positive_year_share_at_least_60pct": eval_base["positive_calendar_year_share"] >= 0.60,
            "direction_flip_weaker": flip["cagr_to_strict_mdd"] < eval_base["cagr_to_strict_mdd"],
        }
        per_instrument[symbol] = {"checks": checks, "passed": all(checks.values())}
    return {
        "per_instrument": per_instrument,
        "all_three_passed": all(row["passed"] for row in per_instrument.values()),
    }


def render_docs(report: dict[str, Any]) -> str:
    lines = [
        "# QQQ / KODEX200 / GLD alpha-transfer evaluation — 2026-07-19",
        "",
        f"Preregistration: `{report['preregistration_manifest_hash']}`",
        "",
        "The original gross-8 sleeves remain nonportable. These are fixed daily OHLCV translations, not exact ports.",
        "",
        "Metric cells: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades / positive-year share`.",
        "",
        "## Eval 2022–2026-07-18",
        "",
        "| Policy | Asset | Base 5 bp/side | Stress 10 bp/side | Direction flip ratio | Gate |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for policy in POLICIES:
        gate = report["transfer_decision"][policy]
        for symbol in INSTRUMENTS:
            row = report["instruments"][symbol]["policies"][policy]["windows"]["eval"]
            base, stress, flip = row["base_5bp"], row["stress_10bp"], row["direction_flip_5bp"]
            fmt = lambda m: (
                f"{m['absolute_return_pct']:.2f}% / {m['cagr_pct']:.2f}% / {m['strict_mdd_pct']:.2f}% / "
                f"{m['cagr_to_strict_mdd']:.2f} / {m['trades']} / {m['positive_calendar_year_share']:.0%}"
            )
            passed = gate["per_instrument"][symbol]["passed"]
            lines.append(
                f"| `{policy}` | {symbol} | {fmt(base)} | {fmt(stress)} | "
                f"{flip['cagr_to_strict_mdd']:.2f} | {'PASS' if passed else 'FAIL'} |"
            )
    lines += ["", "## Frozen decision", ""]
    for policy in POLICIES:
        decision = report["transfer_decision"][policy]
        lines.append(f"- `{policy}`: **{'TRANSFER PASS' if decision['all_three_passed'] else 'REJECT'}**")
    lines += [
        "",
        "No test/eval row selected or repaired another policy. Raw provider payloads remain local; source URL, SHA256, row range, and timezone are recorded in the JSON result.",
    ]
    return "\n".join(lines) + "\n"


def run(
    *,
    output: str = OUTPUT,
    docs_output: str = DOCS_OUTPUT,
    cache_dir: str = CACHE_DIR,
    refresh: bool = False,
) -> dict[str, Any]:
    frozen = prereg.manifest()
    frozen_path = Path(prereg.OUTPUT)
    if not frozen_path.is_file() or json.loads(frozen_path.read_text()) != frozen:
        raise RuntimeError("cross-asset preregistration artifact mismatch")
    instruments: dict[str, Any] = {}
    policy_matrix: dict[str, dict[str, Any]] = {policy: {} for policy in POLICIES}
    for symbol in INSTRUMENTS:
        frame, source = load_market(symbol, cache_dir, refresh=refresh)
        policies = evaluate_instrument(frame)
        instruments[symbol] = {"source": source, "policies": policies}
        for policy in POLICIES:
            policy_matrix[policy][symbol] = policies[policy]
    decision = {policy: transfer_gate(policy_matrix[policy]) for policy in POLICIES}
    report = {
        "schema_version": 1,
        "research_id": "CAT-XA-1",
        "preregistration_manifest_hash": frozen["manifest_hash"],
        "protocol": {
            "exact_gross8_port": False,
            "thresholds_fit_train_only": True,
            "test_eval_reranking": False,
            "entry": "next session open",
            "nonoverlap": True,
            "base_cost_bps_per_side": 5.0,
            "stress_cost_bps_per_side": 10.0,
            "cagr_uses_full_split_calendar": True,
            "strict_mdd_uses_held_daily_high_low": True,
        },
        "splits": SPLITS,
        "instruments": instruments,
        "transfer_decision": decision,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    report["result_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).write_text(render_docs(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--docs-output", default=DOCS_OUTPUT)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(output=args.output, docs_output=args.docs_output, cache_dir=args.cache_dir, refresh=args.refresh)
    compact = {
        policy: {
            "all_three_passed": report["transfer_decision"][policy]["all_three_passed"],
            "eval": {
                symbol: report["instruments"][symbol]["policies"][policy]["windows"]["eval"]["base_5bp"]
                for symbol in INSTRUMENTS
            },
        }
        for policy in POLICIES
    }
    print(json.dumps({"output": args.output, "docs": args.docs_output, "summary": compact}, indent=2))


if __name__ == "__main__":
    main()
