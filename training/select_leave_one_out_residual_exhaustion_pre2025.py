"""Open only 2023-2024 outcomes for the frozen LORE v1 clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_leave_one_out_residual_exhaustion_support import (
    DEFAULT_CLOCKS,
    DEFAULT_SOURCE_DIR,
    EXPECTED_SOURCE_MANIFEST_HASH,
    SOURCE_MANIFEST,
    _shifted_z,
    assert_clock_contract,
    build_feature_panels,
    candidate_frame,
    load_hourly_panel,
    reserve_clock,
)
from training.export_leave_one_out_residual_exhaustion_sources import (
    END,
    EXPECTED_PROTOCOL_HASH,
    START,
    SYMBOLS,
    sha256_file,
)
from training.preregister_leave_one_out_residual_exhaustion import canonical_hash, protocol


SUPPORT_MANIFEST = "results/leave_one_out_residual_exhaustion_v1_support_manifest_2026-07-17.json"
EXPECTED_SUPPORT_MANIFEST_HASH = "1dc91c0775825a6bcbc76ba8956639e020bdcf5a59d6188fd3d06235f8ce177e"
EXPECTED_CLOCK_HASH = "76c0d78c7c703dc16145a5ff86a32700afe77c8ecce46b0d5042afc3ead5135c"
DEFAULT_OUTPUT = "results/leave_one_out_residual_exhaustion_v1_selection_2026-07-17.json"
DEFAULT_DOCS = "docs/leave-one-out-residual-exhaustion-v1-selection-2026-07-17.md"
DEFAULT_FROZEN_POLICY = "results/leave_one_out_residual_exhaustion_v1_frozen_policy_2026-07-17.json"
SELECTOR_PATH = "training/select_leave_one_out_residual_exhaustion_pre2025.py"
TEST_PATH = "tests/test_select_leave_one_out_residual_exhaustion_pre2025.py"
BASE_COST_BP = 6.0
STRESS_COST_BP = 10.0
SIGNFLIP_SAMPLES = 5_000
SIGNFLIP_SEED = 170_717


@dataclass(frozen=True)
class MarketBundle:
    dates: pd.DatetimeIndex
    market: dict[str, dict[str, np.ndarray]]
    funding: dict[str, pd.DataFrame]
    source_hashes: dict[str, dict[str, str]]


def _file_hash(path: str | Path) -> str:
    return sha256_file(Path(path))


def _git_attestation() -> dict[str, str]:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("tracked/untracked repository state must be clean before opening LORE outcomes")
    for path in (SELECTOR_PATH, TEST_PATH):
        subprocess.check_call(["git", "ls-files", "--error-unmatch", path], stdout=subprocess.DEVNULL)
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selector_sha256": _file_hash(SELECTOR_PATH),
        "test_sha256": _file_hash(TEST_PATH),
    }


def _load_json_with_body_hash(path: str, expected: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != expected:
        raise RuntimeError(f"manifest hash mismatch: {path}")
    body = {k: v for k, v in payload.items() if k not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != expected:
        raise RuntimeError(f"manifest body mismatch: {path}")
    return payload


def load_bundle(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest_path: str = SOURCE_MANIFEST,
) -> MarketBundle:
    source = _load_json_with_body_hash(source_manifest_path, EXPECTED_SOURCE_MANIFEST_HASH)
    records = {str(row["symbol"]): row for row in source["records"]}
    market: dict[str, dict[str, np.ndarray]] = {}
    funding: dict[str, pd.DataFrame] = {}
    dates: pd.DatetimeIndex | None = None
    source_hashes: dict[str, dict[str, str]] = {}
    for symbol in sorted(SYMBOLS):
        market_path = Path(source_dir) / f"{symbol}_5m_2023_2024.csv.gz"
        funding_path = Path(source_dir) / f"{symbol}_funding_2023_2024.csv.gz"
        market_hash = _file_hash(market_path)
        funding_hash = _file_hash(funding_path)
        if market_hash != records[symbol]["output_market_sha256"]:
            raise RuntimeError(f"{symbol} market source changed")
        if funding_hash != records[symbol]["output_funding_sha256"]:
            raise RuntimeError(f"{symbol} funding source changed")
        frame = pd.read_csv(
            market_path,
            usecols=["date", "open", "high", "low", "close", "tic"],
            parse_dates=["date"],
        ).sort_values("date")
        if frame["date"].duplicated().any() or not frame["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} market identity/grid failure")
        current_dates = pd.DatetimeIndex(frame["date"])
        if dates is None:
            dates = current_dates
        elif not dates.equals(current_dates):
            raise RuntimeError("LORE symbol market grids differ")
        arrays = {col: pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float) for col in ("open", "high", "low", "close")}
        if not np.isfinite(np.column_stack(list(arrays.values()))).all():
            raise RuntimeError(f"{symbol} non-finite market source")
        market[symbol] = arrays
        fund = pd.read_csv(funding_path)
        fund["event_time"] = pd.to_datetime(pd.to_numeric(fund["funding_time"], errors="raise"), unit="ms")
        fund["funding_rate"] = pd.to_numeric(fund["funding_rate"], errors="raise")
        if fund["event_time"].duplicated().any() or not fund["event_time"].is_monotonic_increasing:
            raise RuntimeError(f"{symbol} funding order failure")
        funding[symbol] = fund[["event_time", "funding_rate"]].copy()
        source_hashes[symbol] = {"market": market_hash, "funding": funding_hash}
    assert dates is not None
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not dates.equals(expected):
        raise RuntimeError("LORE execution grid is not exact 2023-2024 prefix")
    return MarketBundle(dates=dates, market=market, funding=funding, source_hashes=source_hashes)


def load_clock(path: str = DEFAULT_CLOCKS) -> pd.DataFrame:
    if _file_hash(path) != EXPECTED_CLOCK_HASH:
        raise RuntimeError("LORE frozen clock hash changed")
    clock = pd.read_csv(path)
    assert_clock_contract(clock)
    for col in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        clock[col] = pd.to_datetime(clock[col], errors="raise")
    return clock


def _funding_events_by_bar(bundle: MarketBundle) -> dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]]:
    out: dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]] = {}
    dates_ns = bundle.dates.to_numpy(dtype="datetime64[ns]")
    completed_ns = (bundle.dates + pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]")
    for symbol, frame in bundle.funding.items():
        mapped: dict[int, list[tuple[pd.Timestamp, float, float]]] = {}
        for row in frame.itertuples(index=False):
            event = pd.Timestamp(row.event_time)
            event64 = event.to_datetime64()
            bar_index = int(np.searchsorted(dates_ns, event64, side="right") - 1)
            mark_index = int(np.searchsorted(completed_ns, event64, side="right") - 1)
            if bar_index < 0 or mark_index < 0 or bar_index >= len(bundle.dates):
                continue
            mark = float(bundle.market[symbol]["close"][mark_index])
            mapped.setdefault(bar_index, []).append((event, float(row.funding_rate), mark))
        out[symbol] = mapped
    return out


def _normal_p_value_one_sided_greater(t_value: float) -> float:
    return float(1.0 - 0.5 * (1.0 + math.erf(t_value / math.sqrt(2.0))))


def weekly_cluster_signflip(
    trade_rows: list[dict[str, Any]],
    *,
    seed: int,
    samples: int = SIGNFLIP_SAMPLES,
) -> dict[str, Any]:
    if not trade_rows:
        return {"samples": samples, "weekly_clusters": 0, "observed_net_log_return": 0.0, "raw_p_value": 1.0}
    frame = pd.DataFrame(trade_rows)
    frame["week"] = pd.to_datetime(frame["signal_time"]).dt.to_period("W-SUN").astype(str)
    clusters = frame.groupby("week")["net_log_return"].sum().to_numpy(dtype=float)
    observed = float(clusters.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    batch = 500
    for left in range(0, samples, batch):
        right = min(left + batch, samples)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(right - left, len(clusters)))
        draws[left:right] = signs @ clusters
    p = float((1 + np.count_nonzero(draws >= observed - 1e-15)) / (samples + 1))
    return {
        "samples": samples,
        "weekly_clusters": int(len(clusters)),
        "observed_net_log_return": observed,
        "raw_p_value": p,
        "random_positive_fraction": float(np.mean(draws > 0.0)),
        "random_q95_net_log_return": float(np.quantile(draws, 0.95)),
    }


def _event_indices(bundle: MarketBundle, entry: pd.Timestamp, exit_time: pd.Timestamp) -> tuple[int, int]:
    entry_index = int(bundle.dates.searchsorted(entry))
    exit_index = int(bundle.dates.searchsorted(exit_time))
    if entry_index >= len(bundle.dates) or bundle.dates[entry_index] != entry:
        raise RuntimeError(f"missing exact LORE entry open: {entry}")
    if exit_index >= len(bundle.dates) or bundle.dates[exit_index] != exit_time:
        raise RuntimeError(f"missing exact LORE exit open: {exit_time}")
    if exit_index <= entry_index:
        raise RuntimeError("non-positive LORE hold")
    return entry_index, exit_index


def simulate(
    bundle: MarketBundle,
    clock: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_bp: float = BASE_COST_BP,
) -> dict[str, Any]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not START <= start_ts < end_ts <= END:
        raise ValueError("LORE selection simulation escaped 2023-2024")
    if cost_bp < 0:
        raise ValueError("negative execution cost")
    rate = cost_bp / 10_000.0
    selected = clock.loc[(clock["entry_time"] >= start_ts) & (clock["exit_time"] < end_ts)].sort_values("entry_time")
    funding_by_bar = _funding_events_by_bar(bundle)
    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    close_mdd = 0.0
    close_peak = 1.0
    total_funding = 0.0
    total_cost = 0.0
    trade_rows: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in selected.itertuples(index=False):
        entry_time, exit_time = pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry_time < previous_exit:
            raise RuntimeError("LORE execution clock overlaps")
        previous_exit = exit_time
        entry_index, exit_index = _event_indices(bundle, entry_time, exit_time)
        long_symbol, short_symbol = str(row.long_symbol), str(row.short_symbol)
        long_weight, short_weight = float(row.long_weight), float(row.short_weight_abs)
        if not math.isclose(long_weight + short_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("LORE event gross drift")
        start_equity = equity
        long_entry = float(bundle.market[long_symbol]["open"][entry_index])
        short_entry = float(bundle.market[short_symbol]["open"][entry_index])
        long_qty = long_weight * start_equity / long_entry
        short_qty = short_weight * start_equity / short_entry
        entry_cost = rate * (long_qty * long_entry + short_qty * short_entry)
        total_cost += entry_cost
        cumulative_funding = 0.0
        cumulative_funding_debits_for_peak = 0.0
        equity_after_entry = start_equity - entry_cost
        strict_mdd = max(strict_mdd, 1.0 - equity_after_entry / peak)

        def settle_funding(bar_index: int) -> None:
            nonlocal cumulative_funding, cumulative_funding_debits_for_peak, total_funding
            for symbol, signed_qty in ((long_symbol, long_qty), (short_symbol, -short_qty)):
                for event_time, funding_rate, mark in funding_by_bar[symbol].get(bar_index, []):
                    if entry_time < event_time <= exit_time:
                        cash = -signed_qty * mark * funding_rate
                        cumulative_funding += cash
                        cumulative_funding_debits_for_peak += min(cash, 0.0)
                        total_funding += cash

        for bar_index in range(entry_index, exit_index):
            settle_funding(bar_index)
            long_high = float(bundle.market[long_symbol]["high"][bar_index])
            long_low = float(bundle.market[long_symbol]["low"][bar_index])
            short_high = float(bundle.market[short_symbol]["high"][bar_index])
            short_low = float(bundle.market[short_symbol]["low"][bar_index])
            favorable_pnl = long_qty * (long_high - long_entry) + short_qty * (short_entry - short_low)
            adverse_pnl = long_qty * (long_low - long_entry) + short_qty * (short_entry - short_high)
            favorable_liquidation = rate * (long_qty * long_high + short_qty * short_low)
            adverse_liquidation = rate * (long_qty * long_low + short_qty * short_high)
            favorable_for_peak = (
                start_equity
                - entry_cost
                + cumulative_funding_debits_for_peak
                + favorable_pnl
                - favorable_liquidation
            )
            peak = max(peak, favorable_for_peak)
            adverse_equity = start_equity - entry_cost + cumulative_funding + adverse_pnl - adverse_liquidation
            strict_mdd = max(strict_mdd, 1.0 - adverse_equity / peak)
            long_close = float(bundle.market[long_symbol]["close"][bar_index])
            short_close = float(bundle.market[short_symbol]["close"][bar_index])
            close_pnl = long_qty * (long_close - long_entry) + short_qty * (short_entry - short_close)
            close_liquidation = rate * (long_qty * long_close + short_qty * short_close)
            close_equity = start_equity - entry_cost + cumulative_funding + close_pnl - close_liquidation
            close_peak = max(close_peak, close_equity)
            close_mdd = max(close_mdd, 1.0 - close_equity / close_peak)
        # A settlement exactly at the scheduled exit belongs to the held interval,
        # but maps to the exit bar whose OHLC must not be consumed after exit.
        settle_funding(exit_index)
        long_exit = float(bundle.market[long_symbol]["open"][exit_index])
        short_exit = float(bundle.market[short_symbol]["open"][exit_index])
        exit_cost = rate * (long_qty * long_exit + short_qty * short_exit)
        total_cost += exit_cost
        pnl = long_qty * (long_exit - long_entry) + short_qty * (short_entry - short_exit)
        equity = start_equity - entry_cost + cumulative_funding + pnl - exit_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        close_peak = max(close_peak, equity)
        close_mdd = max(close_mdd, 1.0 - equity / close_peak)
        peak = max(peak, equity)
        trade_return = equity / start_equity - 1.0
        trade_rows.append({
            "signal_time": str(row.signal_time),
            "entry_time": str(entry_time),
            "exit_time": str(exit_time),
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "net_return": float(trade_return),
            "net_log_return": float(math.log(max(equity, 1e-15) / max(start_equity, 1e-15))),
            "funding_cash": float(cumulative_funding),
        })
        if equity <= 0.0:
            strict_mdd = 1.0
            break
    years = (end_ts - start_ts).total_seconds() / (365.25 * 86_400.0)
    absolute = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else -100.0
    strict_pct = min(max(strict_mdd * 100.0, 0.0), 100.0)
    trade_returns = np.asarray([r["net_return"] for r in trade_rows], dtype=float)
    return {
        "absolute_return_pct": float(absolute),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_pct),
        "close_mdd_pct": float(min(max(close_mdd * 100.0, 0.0), 100.0)),
        "cagr_to_strict_mdd": float(cagr / strict_pct) if strict_pct > 1e-12 else 0.0,
        "trades": len(trade_rows),
        "mean_net_bps": float(trade_returns.mean() * 10_000.0) if len(trade_returns) else 0.0,
        "win_rate": float(np.mean(trade_returns > 0.0)) if len(trade_returns) else 0.0,
        "funding_cash_pct_initial": float(total_funding * 100.0),
        "transaction_cost_pct_initial": float(total_cost * 100.0),
        "calendar_start": str(start_ts.date()),
        "calendar_end_exclusive": str(end_ts.date()),
        "trade_rows": trade_rows,
    }


def _transform_same_clock(clock: pd.DataFrame, kind: str, seed: int = SIGNFLIP_SEED) -> pd.DataFrame:
    out = clock.copy()
    if kind == "direction_flip":
        out[["long_symbol", "short_symbol"]] = out[["short_symbol", "long_symbol"]].to_numpy()
        out[["long_weight", "short_weight_abs"]] = out[["short_weight_abs", "long_weight"]].to_numpy()
        out[["long_beta", "short_beta"]] = out[["short_beta", "long_beta"]].to_numpy()
    elif kind == "equal_weight":
        out["long_weight"] = 0.5
        out["short_weight_abs"] = 0.5
    elif kind == "delay_one_hour":
        for col in ("entry_time", "exit_time"):
            out[col] = pd.to_datetime(out[col]) + pd.Timedelta(hours=1)
    elif kind == "shift_seven_days":
        for col in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
            out[col] = pd.to_datetime(out[col]) + pd.Timedelta(days=7)
    elif kind == "monthly_pair_permutation":
        rng = np.random.default_rng(seed)
        out["signal_time"] = pd.to_datetime(out["signal_time"])
        bundle_cols = ["long_symbol", "short_symbol", "long_weight", "short_weight_abs", "long_beta", "short_beta"]
        for _, idx in out.groupby(out["signal_time"].dt.to_period("M")).groups.items():
            positions = np.asarray(list(idx), dtype=int)
            shuffled = positions.copy()
            rng.shuffle(shuffled)
            out.loc[positions, bundle_cols] = out.loc[shuffled, bundle_cols].to_numpy()
    else:
        raise ValueError(kind)
    return out


def _score_candidate_frame(features: dict[str, Any], horizon: int, score_z: pd.DataFrame, require_flow: bool) -> pd.DataFrame:
    base = candidate_frame(features, horizon)
    symbols: list[str] = features["symbols"]
    scores = score_z.to_numpy(dtype=float)
    flow = features["flow_z"].to_numpy(dtype=float)
    beta = features["beta"].to_numpy(dtype=float)
    rows = np.arange(len(score_z))
    safe = np.where(np.isfinite(scores), scores, 0.0)
    winner_idx, loser_idx = np.argmax(safe, axis=1), np.argmin(safe, axis=1)
    winner_score, loser_score = scores[rows, winner_idx], scores[rows, loser_idx]
    winner_flow, loser_flow = flow[rows, winner_idx], flow[rows, loser_idx]
    winner_beta, loser_beta = beta[rows, winner_idx], beta[rows, loser_idx]
    denom = winner_beta + loser_beta
    long_weight, short_weight = winner_beta / denom, loser_beta / denom
    eligible = (
        features["source_clean"].to_numpy(dtype=bool)
        & score_z.notna().all(axis=1).to_numpy(dtype=bool)
        & features["flow_z"].notna().all(axis=1).to_numpy(dtype=bool)
        & features["beta"].notna().all(axis=1).to_numpy(dtype=bool)
        & (winner_score >= 1.5)
        & (loser_score <= -1.5)
        & (np.minimum(long_weight, short_weight) >= 0.25)
    )
    if require_flow:
        eligible &= (winner_score - winner_flow >= 1.0) & (loser_flow - loser_score >= 1.0)
    base["long_symbol"] = np.asarray(symbols, dtype=object)[loser_idx]
    base["short_symbol"] = np.asarray(symbols, dtype=object)[winner_idx]
    base["long_weight"] = long_weight
    base["short_weight_abs"] = short_weight
    base["long_beta"] = loser_beta
    base["short_beta"] = winner_beta
    base["loser_residual_z"] = loser_score
    base["winner_residual_z"] = winner_score
    base["loser_flow_z"] = loser_flow
    base["winner_flow_z"] = winner_flow
    base["exhaustion_score"] = np.minimum(winner_score - winner_flow, loser_flow - loser_score)
    base["eligible"] = eligible
    return base


def build_different_clock_controls() -> dict[str, dict[str, pd.DataFrame]]:
    panels, quality = load_hourly_panel(DEFAULT_SOURCE_DIR, SOURCE_MANIFEST)
    controls: dict[str, dict[str, pd.DataFrame]] = {"no_flow": {}, "raw_return": {}}
    for policy in protocol()["policies"]:
        horizon = int(policy["residual_horizon_hours"])
        hold = int(policy["hold_hours"])
        pid = str(policy["policy_id"])
        features = build_feature_panels(panels, quality, horizon)
        no_flow = _score_candidate_frame(features, horizon, features["residual_z"], require_flow=False)
        raw_score = _shifted_z(np.log(features["close"] / features["close"].shift(horizon)))
        raw = _score_candidate_frame(features, horizon, raw_score, require_flow=True)
        controls["no_flow"][pid] = reserve_clock(no_flow, pid, hold)
        controls["raw_return"][pid] = reserve_clock(raw, pid, hold)
    return controls


WINDOWS = {
    "fit_2023": ("2023-01-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "combined_2023_2024": ("2023-01-01", "2025-01-01"),
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
    "2024_h1": ("2024-01-01", "2024-07-01"),
    "2024_h2": ("2024-07-01", "2025-01-01"),
}


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if k != "trade_rows"}


def _policy_result(
    bundle: MarketBundle,
    clock: pd.DataFrame,
    policy: dict[str, Any],
    different_controls: dict[str, dict[str, pd.DataFrame]],
    rank_index: int,
) -> dict[str, Any]:
    pid = str(policy["policy_id"])
    primary_clock = clock.loc[clock["policy_id"] == pid].copy()
    stats = {name: simulate(bundle, primary_clock, start=a, end=b) for name, (a, b) in WINDOWS.items()}
    stress = simulate(bundle, primary_clock, start="2023-01-01", end="2025-01-01", cost_bp=STRESS_COST_BP)
    signflip = weekly_cluster_signflip(
        stats["combined_2023_2024"]["trade_rows"],
        seed=SIGNFLIP_SEED + rank_index,
    )
    signflip["bonferroni_hypotheses"] = 4
    signflip["bonferroni_p_value"] = min(1.0, signflip["raw_p_value"] * 4)
    same_controls = {}
    for kind in ("direction_flip", "equal_weight", "delay_one_hour", "shift_seven_days", "monthly_pair_permutation"):
        transformed = _transform_same_clock(primary_clock, kind, seed=SIGNFLIP_SEED + rank_index)
        same_controls[kind] = _slim(simulate(bundle, transformed, start="2023-01-01", end="2025-01-01"))
    same_controls["remove_flow_disconfirmation"] = _slim(simulate(
        bundle, different_controls["no_flow"][pid], start="2023-01-01", end="2025-01-01"
    ))
    same_controls["raw_return_without_residualization"] = _slim(simulate(
        bundle, different_controls["raw_return"][pid], start="2023-01-01", end="2025-01-01"
    ))
    annual = [stats["fit_2023"], stats["test_2024"]]
    halves = [stats[k] for k in ("2023_h1", "2023_h2", "2024_h1", "2024_h2")]
    combined = stats["combined_2023_2024"]
    gates = {
        "each_year_absolute_return_positive": all(x["absolute_return_pct"] > 0 for x in annual),
        "each_year_cagr_to_strict_mdd_at_least_1_5": all(x["cagr_to_strict_mdd"] >= 1.5 for x in annual),
        "positive_half_years_at_least_3_of_4": sum(x["absolute_return_pct"] > 0 for x in halves) >= 3,
        "combined_cagr_to_strict_mdd_at_least_3": combined["cagr_to_strict_mdd"] >= 3.0,
        "combined_strict_mdd_at_most_12": combined["strict_mdd_pct"] <= 12.0,
        "combined_trades_at_least_150": combined["trades"] >= 150,
        "each_year_trades_at_least_60": all(x["trades"] >= 60 for x in annual),
        "ten_bp_cost_stress_positive": stress["absolute_return_pct"] > 0.0,
        "bonferroni_weekly_signflip_p_at_most_0_10": signflip["bonferroni_p_value"] <= 0.10,
    }
    return {
        "policy": policy,
        "stats": {k: _slim(v) for k, v in stats.items()},
        "ten_bp_notional_side_cost_stress": _slim(stress),
        "weekly_cluster_signflip": signflip,
        "diagnostic_controls": same_controls,
        "selection_gates": gates,
        "passes_selection": all(gates.values()),
    }


def _rank_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    s = trial["stats"]
    worst_annual = min(s["fit_2023"]["cagr_to_strict_mdd"], s["test_2024"]["cagr_to_strict_mdd"])
    combined = s["combined_2023_2024"]
    return (-worst_annual, -combined["cagr_to_strict_mdd"], combined["strict_mdd_pct"], trial["policy"]["policy_id"])


def _markdown(result: dict[str, Any]) -> str:
    rows = []
    ranked = sorted(result["trials"], key=lambda t: (
        not t["passes_selection"], *_rank_key(t)
    ))
    for rank, trial in enumerate(ranked, 1):
        s = trial["stats"]["combined_2023_2024"]
        p = trial["policy"]
        rows.append(
            f"| {rank} | {p['policy_id']} | {p['residual_horizon_hours']}h/{p['hold_hours']}h | "
            f"{s['absolute_return_pct']:+.3f}% | {s['cagr_pct']:+.3f}% | {s['strict_mdd_pct']:.3f}% | "
            f"{s['cagr_to_strict_mdd']:.3f} | {s['trades']} | {'PASS' if trial['passes_selection'] else 'REJECT'} |"
        )
    return "\n".join([
        "# LORE v1 2023–2024 selection — 2026-07-17",
        "",
        "> Only 2023–2024 outcomes were opened. Calendar 2025 and 2026 remain sealed for LORE v1.",
        "",
        "| Rank | Policy | Residual/hold | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Decision |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## Decision",
        "",
        f"- status: **{result['decision']}**",
        f"- selected policy: `{result['selected_policy_id']}`",
        f"- passing policies: `{result['passing_policies']}`",
        "- CAGR includes the full calendar, including idle periods.",
        "- strict MDD includes global/pre-entry HWM, favorable-before-adverse two-leg OHLC, entry and hypothetical liquidation costs, exact funding event ordering, and scheduled exit cost.",
        "- Diagnostics cannot rescue a rejected policy; no sign/threshold/hold/pair repair is allowed.",
        "",
    ])


def run(
    output: str = DEFAULT_OUTPUT,
    docs_output: str = DEFAULT_DOCS,
    frozen_policy_output: str = DEFAULT_FROZEN_POLICY,
) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("LORE preregistration drifted")
    support = _load_json_with_body_hash(SUPPORT_MANIFEST, EXPECTED_SUPPORT_MANIFEST_HASH)
    if support.get("clock_sha256") != EXPECTED_CLOCK_HASH or not support.get("all_policies_pass_support"):
        raise RuntimeError("LORE support freeze is not approved")
    attestation = _git_attestation()
    bundle = load_bundle()
    clock = load_clock()
    different_controls = build_different_clock_controls()
    trials = [
        _policy_result(bundle, clock, policy, different_controls, i)
        for i, policy in enumerate(protocol()["policies"])
    ]
    passing = sorted((t for t in trials if t["passes_selection"]), key=_rank_key)
    selected = passing[0] if passing else None
    result: dict[str, Any] = {
        "protocol_version": "lore_v1_pre2025_selector_2026-07-17",
        "outcomes_opened": True,
        "opened_window": ["2023-01-01", "2025-01-01"],
        "holdout_2025_opened": False,
        "final_2026_opened": False,
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "support_manifest_hash": EXPECTED_SUPPORT_MANIFEST_HASH,
        "clock_hash": EXPECTED_CLOCK_HASH,
        "execution_contract": protocol()["execution"],
        "multiple_testing": {"policies": 4, "weekly_signflip_samples": SIGNFLIP_SAMPLES, "bonferroni": 4},
        "trials": trials,
        "passing_policies": [t["policy"]["policy_id"] for t in passing],
        "selected_policy_id": selected["policy"]["policy_id"] if selected else None,
        "decision": "pre2025_policy_frozen" if selected else "rejected_before_2025_holdout",
        "pre_outcome_selector_attestation": attestation,
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).write_text(_markdown(result))
    if selected:
        frozen = {
            "protocol_version": "lore_v1_frozen_policy_2026-07-17",
            "selection_result_hash": result["manifest_hash"],
            "policy": selected["policy"],
            "selection_stats": selected["stats"],
            "holdout_2025_opened": False,
            "final_2026_opened": False,
        }
        frozen["manifest_hash"] = canonical_hash(frozen)
        Path(frozen_policy_output).write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    parser.add_argument("--frozen-policy-output", default=DEFAULT_FROZEN_POLICY)
    args = parser.parse_args()
    result = run(args.output, args.docs_output, args.frozen_policy_output)
    summary = []
    for trial in sorted(result["trials"], key=lambda x: x["stats"]["combined_2023_2024"]["cagr_to_strict_mdd"], reverse=True):
        s = trial["stats"]["combined_2023_2024"]
        summary.append({
            "policy_id": trial["policy"]["policy_id"],
            "absolute_return_pct": s["absolute_return_pct"],
            "cagr_pct": s["cagr_pct"],
            "strict_mdd_pct": s["strict_mdd_pct"],
            "cagr_to_strict_mdd": s["cagr_to_strict_mdd"],
            "trades": s["trades"],
            "pass": trial["passes_selection"],
        })
    print(json.dumps({"decision": result["decision"], "selected_policy_id": result["selected_policy_id"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
