"""Select one Coinbase venue-leadership proxy using only frozen 2020-2022 outcomes."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.build_coinbase_spot_leadership_support import schedule_nonoverlap
from training.export_wikimedia_attention_source import sha256_file
from training.preregister_coinbase_spot_leadership_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    Policy,
    canonical_hash,
    policy_grid,
    validate_manifest as validate_preregistration,
)


DEFAULT_SOURCE_MANIFEST = "results/coinbase_spot_leadership_source_manifest_2026-07-16.json"
DEFAULT_SUPPORT_MANIFEST = "results/coinbase_spot_leadership_support_manifest_2026-07-16.json"
DEFAULT_OUTPUT = "results/coinbase_spot_leadership_selection_2026-07-16.json"
DEFAULT_POLICY_OUTPUT = "results/coinbase_spot_leadership_frozen_policy_2026-07-16.json"
DEFAULT_DOCS = "docs/coinbase-spot-leadership-selection-2026-07-16.md"
SELECTOR_TEST = "tests/test_select_coinbase_spot_leadership_pre2023.py"
SELECTION_END = "2023-01-01"

WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2020": ("2020-01-01", "2021-01-01"),
    "fit_2021": ("2021-01-01", "2022-01-01"),
    "selection_2022": ("2022-01-01", SELECTION_END),
    "combined_2020_2022": ("2020-01-01", SELECTION_END),
    "2020_h1": ("2020-01-01", "2020-07-01"),
    "2020_h2": ("2020-07-01", "2021-01-01"),
    "2021_h1": ("2021-01-01", "2021-07-01"),
    "2021_h2": ("2021-07-01", "2022-01-01"),
    "2022_h1": ("2022-01-01", "2022-07-01"),
    "2022_h2": ("2022-07-01", SELECTION_END),
}
YEAR_WINDOWS = ("fit_2020", "fit_2021", "selection_2022")
HALF_WINDOWS = ("2020_h1", "2020_h2", "2021_h1", "2021_h2", "2022_h1", "2022_h2")


@dataclass(frozen=True)
class Config:
    preregistration: str = DEFAULT_PREREGISTRATION
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    support_manifest: str = DEFAULT_SUPPORT_MANIFEST
    output: str = DEFAULT_OUTPUT
    policy_output: str = DEFAULT_POLICY_OUTPUT
    docs_output: str = DEFAULT_DOCS
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0008
    signflip_samples: int = 5_000
    random_control_samples: int = 5_000
    random_seed: int = 20260716


@dataclass(frozen=True)
class Trade:
    signal_position: int
    entry_position: int
    exit_position: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: int
    price_factor: float
    funding_factor: float
    funding_debit_factor: float
    funding_credit_factor: float
    favorable_price_factor: float
    adverse_price_factor: float


def validate_hashed_manifest(path: str | Path, timestamp_key: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", timestamp_key}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError(f"manifest hash mismatch: {path}")
    return payload


def git_anchor(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    try:
        pathspec = str(candidate.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        pathspec = str(candidate)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"pre-outcome selector artifact is not clean: {pathspec}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"pre-outcome selector artifact is not committed: {pathspec}")
    return {"path": pathspec, "commit": commit, "sha256": sha256_file(candidate)}


def require_clean_tracked_tree() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("selection refuses to open outcomes with a dirty Git tree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("selection cannot resolve a committed HEAD")
    return head


def _verified_output(manifest: dict[str, Any], name: str) -> Path:
    metadata = manifest["outputs"][name]
    path = Path(metadata["path"])
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"frozen {name} output is missing or hash-mismatched")
    return path


def load_sources(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prereg = json.loads(Path(cfg.preregistration).read_text())
    validate_preregistration(prereg)
    source = validate_hashed_manifest(cfg.source_manifest, "retrieved_at")
    support = validate_hashed_manifest(cfg.support_manifest, "created_at")
    if source.get("end_exclusive") != SELECTION_END or source.get("future_data_requested") is not False:
        raise RuntimeError("source manifest did not preserve the 2023 holdout")
    if source.get("audit_amendment", {}).get("before_forward_trade_outcomes") is not True:
        raise RuntimeError("source manifest lacks pre-outcome provenance attestation")
    if support.get("forward_trade_outcomes_opened") is not False:
        raise RuntimeError("support manifest already opened forward outcomes")
    if support.get("selection_may_open_forward_returns") is not True:
        raise RuntimeError("support gates do not permit selection")
    if source["preregistration_manifest_hash"] != prereg["manifest_hash"]:
        raise RuntimeError("source/preregistration hash mismatch")
    if support["hashes"]["preregistration_manifest_hash"] != prereg["manifest_hash"]:
        raise RuntimeError("support/preregistration hash mismatch")
    if support["hashes"]["source_manifest_hash"] != source["manifest_hash"]:
        raise RuntimeError("support/source manifest hash mismatch")
    if sorted(support["passing_policy_ids"]) != [policy.policy_id for policy in policy_grid()]:
        raise RuntimeError("not all preregistered policies passed support")

    market = pd.read_csv(
        _verified_output(source, "binance"),
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    funding = pd.read_csv(
        _verified_output(source, "funding"),
        usecols=["date", "funding_rate"],
    )
    funding["date"] = pd.to_datetime(funding["date"], format="mixed", errors="raise")
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    features = pd.read_csv(
        _verified_output(support, "features"), parse_dates=["date"]
    )
    clocks = pd.read_csv(
        _verified_output(support, "clocks"), parse_dates=["signal_date"]
    )
    if not market["date"].equals(features["date"]):
        raise RuntimeError("market and frozen feature grids differ")
    if market["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection evaluator refuses to open 2023")
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="raise")
        if not np.isfinite(market[column]).all() or (market[column] <= 0).any():
            raise ValueError(f"invalid execution market column: {column}")
    if funding["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("funding source crossed 2023")
    if clocks["signal_date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("policy clocks crossed 2023")
    rows = clocks["signal_row"].to_numpy(np.int64)
    if not clocks["signal_date"].reset_index(drop=True).equals(
        features.loc[rows, "date"].reset_index(drop=True)
    ):
        raise RuntimeError("clock row/date mapping differs from frozen features")
    if features.loc[rows, "source_quarantined"].ne(0).any():
        raise RuntimeError("frozen clocks contain quarantined signals")
    for policy in policy_grid():
        regenerated = schedule_nonoverlap(
            transformed_policy_mask(features, policy, "primary"), policy.hold_bars
        )
        frozen = clock_indices(clocks, policy.policy_id)
        if not np.array_equal(regenerated, frozen):
            raise RuntimeError(f"frozen clock differs from evaluator rule: {policy.policy_id}")
    source_record = {
        "preregistration": {
            "path": cfg.preregistration,
            "file_sha256": sha256_file(cfg.preregistration),
            "manifest_hash": prereg["manifest_hash"],
        },
        "source_manifest": {
            "path": cfg.source_manifest,
            "file_sha256": sha256_file(cfg.source_manifest),
            "manifest_hash": source["manifest_hash"],
        },
        "support_manifest": {
            "path": cfg.support_manifest,
            "file_sha256": sha256_file(cfg.support_manifest),
            "manifest_hash": support["manifest_hash"],
        },
        "rows": {"market": len(market), "funding": len(funding), "clocks": len(clocks)},
        "maximum_market_date": str(market["date"].max()),
    }
    return market, funding, features, clocks, source_record


class ExecutionEngine:
    def __init__(self, market: pd.DataFrame, funding: pd.DataFrame, cfg: Config) -> None:
        self.cfg = cfg
        self.dates = market["date"].reset_index(drop=True)
        self.open = market["open"].to_numpy(float)
        self.high = market["high"].to_numpy(float)
        self.low = market["low"].to_numpy(float)
        self.funding_times = funding["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.funding_rates = funding["funding_rate"].to_numpy(float)
        self._cache: dict[tuple[int, int, int, int], Trade | None] = {}

    def trade_at(self, signal: int, side: int, hold_bars: int, delay_bars: int = 0) -> Trade | None:
        key = (int(signal), int(side), int(hold_bars), int(delay_bars))
        if key in self._cache:
            return self._cache[key]
        if side not in (-1, 1) or hold_bars < 1 or delay_bars < 0:
            raise ValueError("invalid trade direction, hold, or delay")
        entry = int(signal) + 1 + int(delay_bars)
        exit_position = entry + int(hold_bars)
        if entry < 0 or exit_position >= len(self.open):
            self._cache[key] = None
            return None
        entry_price = float(self.open[entry])
        exit_price = float(self.open[exit_position])
        gross = side * (exit_price / entry_price - 1.0)
        leverage = float(self.cfg.leverage)
        price_factor = 1.0 + leverage * gross
        # The position is still marked at the exit open before exit cost is paid.
        held_high = max(float(np.max(self.high[entry:exit_position])), exit_price)
        held_low = min(float(np.min(self.low[entry:exit_position])), exit_price)
        if side > 0:
            favorable = held_high
            adverse = held_low
        else:
            favorable = held_low
            adverse = held_high
        favorable_factor = 1.0 + leverage * side * (favorable / entry_price - 1.0)
        adverse_factor = 1.0 + leverage * side * (adverse / entry_price - 1.0)
        entry_ns = int(self.dates.iloc[entry].value)
        exit_ns = int(self.dates.iloc[exit_position].value)
        left = int(np.searchsorted(self.funding_times, entry_ns, side="right"))
        right = int(np.searchsorted(self.funding_times, exit_ns, side="right"))
        factors = 1.0 - leverage * side * self.funding_rates[left:right]
        if (
            min(price_factor, favorable_factor, adverse_factor) <= 0.0
            or not np.isfinite(factors).all()
            or (factors <= 0.0).any()
        ):
            raise ValueError("invalid leveraged price or funding factor")
        trade = Trade(
            signal_position=int(signal),
            entry_position=entry,
            exit_position=exit_position,
            entry_date=self.dates.iloc[entry],
            exit_date=self.dates.iloc[exit_position],
            side=int(side),
            price_factor=float(price_factor),
            funding_factor=float(np.prod(factors)) if len(factors) else 1.0,
            funding_debit_factor=float(np.prod(np.minimum(factors, 1.0))) if len(factors) else 1.0,
            funding_credit_factor=float(np.prod(np.maximum(factors, 1.0))) if len(factors) else 1.0,
            favorable_price_factor=float(favorable_factor),
            adverse_price_factor=float(adverse_factor),
        )
        self._cache[key] = trade
        return trade


def clock_indices(clocks: pd.DataFrame, policy_id: str) -> np.ndarray:
    return clocks.loc[clocks["policy_id"].eq(policy_id), "signal_row"].to_numpy(np.int64)


def trades_from_indices(
    engine: ExecutionEngine,
    indices: Iterable[int],
    *,
    side: int,
    hold_bars: int,
    delay_bars: int = 0,
) -> list[Trade]:
    trades: list[Trade] = []
    for signal in indices:
        trade = engine.trade_at(int(signal), side, hold_bars, delay_bars)
        if trade is not None:
            trades.append(trade)
    return trades


def window_trades(trades: Iterable[Trade], start: str, end: str) -> list[Trade]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return [trade for trade in trades if trade.entry_date >= start_ts and trade.exit_date < end_ts]


def strict_equity_stats(
    trades: Iterable[Trade],
    *,
    start: str,
    end: str,
    leverage: float,
    cost_notional_per_side: float,
) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda trade: (trade.entry_position, trade.exit_position))
    cost_factor = 1.0 - leverage * cost_notional_per_side
    if cost_factor <= 0.0:
        raise ValueError("cost factor must remain positive")
    equity = peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_returns: list[float] = []
    previous_exit = -1
    for trade in ordered:
        if trade.entry_position < previous_exit:
            raise RuntimeError("strict stats received overlapping positions")
        previous_exit = trade.exit_position
        entry_equity = equity
        favorable_equity = (
            equity
            * cost_factor
            * trade.favorable_price_factor
            * trade.funding_credit_factor
        )
        intratrade_peak = max(peak, favorable_equity)
        liquidation_equity = (
            equity
            * cost_factor
            * trade.adverse_price_factor
            * trade.funding_debit_factor
            * cost_factor
        )
        strict_mdd = max(strict_mdd, 1.0 - liquidation_equity / intratrade_peak)
        peak = intratrade_peak
        gross_factor = trade.price_factor * trade.funding_factor
        equity *= cost_factor * gross_factor * cost_factor
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        net_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(gross_factor - 1.0)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    net = np.asarray(net_returns)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": len(ordered),
        "longs": sum(trade.side > 0 for trade in ordered),
        "shorts": sum(trade.side < 0 for trade in ordered),
        "mean_net_bps": float(net.mean() * 10_000) if len(net) else 0.0,
        "mean_gross_bps": float(np.mean(gross_returns) * 10_000) if gross_returns else 0.0,
        "win_rate": float((net > 0).mean()) if len(net) else 0.0,
        "calendar_start": start,
        "calendar_end_exclusive": end,
    }


def stats_windows(trades: list[Trade], cfg: Config, cost: float) -> dict[str, dict[str, Any]]:
    return {
        name: strict_equity_stats(
            window_trades(trades, start, end),
            start=start,
            end=end,
            leverage=cfg.leverage,
            cost_notional_per_side=cost,
        )
        for name, (start, end) in WINDOWS.items()
    }


def transformed_policy_mask(features: pd.DataFrame, policy: Policy, mode: str) -> np.ndarray | None:
    tokens = {name: features[name].to_numpy(float) for name in ("ZR", "ZP", "ZV", "ZCB", "ZBN")}
    if mode == "venue_role_swap":
        tokens = {
            "ZR": -tokens["ZR"],
            "ZP": -tokens["ZP"],
            "ZV": -tokens["ZV"],
            "ZCB": tokens["ZBN"],
            "ZBN": tokens["ZCB"],
        }
    side = float(policy.side)
    conditions: list[tuple[str, np.ndarray]]
    if policy.family == "relative_return_lead":
        conditions = [
            ("return", side * tokens["ZR"] >= 2.0),
            ("return", side * tokens["ZCB"] >= 1.0),
            ("return", side * tokens["ZBN"] < 1.5),
        ]
    elif policy.family == "premium_shock":
        conditions = [
            ("premium", side * tokens["ZP"] >= 2.0),
            ("return", side * tokens["ZCB"] >= 0.5),
        ]
    elif policy.family == "activity_confirmed_relative":
        conditions = [
            ("activity", tokens["ZV"] >= 2.0),
            ("return", side * tokens["ZCB"] >= 1.5),
            ("return", side * tokens["ZR"] >= 1.0),
        ]
    elif policy.family == "activity_premium_confluence":
        conditions = [
            ("activity", tokens["ZV"] >= 1.5),
            ("premium", side * tokens["ZP"] >= 1.5),
        ]
    elif policy.family == "return_premium_confluence":
        conditions = [
            ("return", side * tokens["ZR"] >= 1.5),
            ("premium", side * tokens["ZP"] >= 1.5),
        ]
    else:
        raise ValueError(policy.family)
    if mode in {"return_only", "premium_only", "activity_only"}:
        category = mode.removesuffix("_only")
        conditions = [condition for condition in conditions if condition[0] == category]
    elif mode == "no_premium":
        if not any(category == "premium" for category, _ in conditions):
            return None
        conditions = [condition for condition in conditions if condition[0] != "premium"]
    elif mode not in {"primary", "venue_role_swap"}:
        raise ValueError(mode)
    if not conditions:
        return None
    mask = features["source_quarantined"].eq(0).to_numpy(bool)
    for _, condition in conditions:
        mask &= condition
    return mask


def generated_control_trades(
    engine: ExecutionEngine,
    features: pd.DataFrame,
    policy: Policy,
    mode: str,
) -> list[Trade] | None:
    mask = transformed_policy_mask(features, policy, mode)
    if mask is None:
        return None
    indices = schedule_nonoverlap(mask, policy.hold_bars)
    return trades_from_indices(
        engine, indices, side=policy.side, hold_bars=policy.hold_bars
    )


def weekly_cluster_signflip(
    trades: list[Trade], cfg: Config, policy_number: int
) -> dict[str, Any]:
    if not trades:
        return {"samples": 0, "raw_p_value": 1.0, "bonferroni_p_value": 1.0}
    cost_log = 2.0 * np.log(1.0 - cfg.leverage * cfg.base_cost_notional_per_side)
    frame = pd.DataFrame(
        {
            "week": [trade.entry_date.to_period("W-SUN") for trade in trades],
            "pre_cost_log": [np.log(trade.price_factor * trade.funding_factor) for trade in trades],
        }
    )
    weekly = frame.groupby("week", sort=True)["pre_cost_log"].sum().to_numpy(float)
    observed = float(weekly.sum() + len(trades) * cost_log)
    rng = np.random.default_rng(cfg.random_seed + policy_number * 1009)
    simulated = np.empty(cfg.signflip_samples)
    for index in range(cfg.signflip_samples):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(weekly))
        simulated[index] = float(np.dot(signs, weekly) + len(trades) * cost_log)
    raw_p = float((1 + np.sum(simulated >= observed)) / (1 + len(simulated)))
    return {
        "samples": int(len(simulated)),
        "weekly_clusters": int(len(weekly)),
        "observed_net_log_return": observed,
        "raw_p_value": raw_p,
        "bonferroni_hypotheses": len(policy_grid()),
        "bonferroni_p_value": min(1.0, raw_p * len(policy_grid())),
    }


def precompute_signal_log_factors(
    engine: ExecutionEngine, policy: Policy, cfg: Config
) -> np.ndarray:
    output = np.full(len(engine.dates), np.nan)
    cost_log = 2.0 * np.log(1.0 - cfg.leverage * cfg.base_cost_notional_per_side)
    count = len(output) - policy.hold_bars - 1
    signal = np.arange(count, dtype=np.int64)
    entry = signal + 1
    exit_position = entry + policy.hold_bars
    gross = policy.side * (engine.open[exit_position] / engine.open[entry] - 1.0)
    price_factor = 1.0 + cfg.leverage * gross
    entry_ns = engine.dates.iloc[entry].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    exit_ns = engine.dates.iloc[exit_position].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    left = np.searchsorted(engine.funding_times, entry_ns, side="right")
    right = np.searchsorted(engine.funding_times, exit_ns, side="right")
    if ((right - left) > 1).any():
        raise RuntimeError("short random-control hold unexpectedly crossed multiple funding events")
    funding_factor = np.ones(count)
    crosses = right > left
    funding_factor[crosses] = (
        1.0 - cfg.leverage * policy.side * engine.funding_rates[left[crosses]]
    )
    combined = price_factor * funding_factor
    if not np.isfinite(combined).all() or (combined <= 0.0).any():
        raise ValueError("invalid vectorized random-control return factor")
    output[:count] = np.log(combined) + cost_log
    return output


def time_of_week_block_random(
    engine: ExecutionEngine,
    trades: list[Trade],
    policy: Policy,
    cfg: Config,
    policy_number: int,
) -> dict[str, Any]:
    if not trades:
        return {"samples": 0, "p_value_net_log_return": 1.0}
    log_factors = precompute_signal_log_factors(engine, policy, cfg)
    signals = np.asarray([trade.signal_position for trade in trades], dtype=np.int64)
    observed = float(np.sum(log_factors[signals]))
    rng = np.random.default_rng(cfg.random_seed + policy_number * 2027)
    year_payloads: list[dict[str, Any]] = []
    for year in (2020, 2021, 2022):
        start = int(np.searchsorted(engine.dates.to_numpy(), np.datetime64(f"{year}-01-01")))
        end = int(np.searchsorted(engine.dates.to_numpy(), np.datetime64(f"{year + 1}-01-01")))
        year_signals = signals[(signals >= start) & (signals < end)]
        relative = year_signals - start
        full_blocks = (end - start) // 2016
        in_full = relative < full_blocks * 2016
        block = relative[in_full] // 2016
        offset = relative[in_full] % 2016
        tail = year_signals[~in_full]
        first = np.full(full_blocks, 2016, dtype=np.int64)
        last = np.full(full_blocks, -2016, dtype=np.int64)
        for value in range(full_blocks):
            local = offset[block == value]
            if len(local):
                first[value], last[value] = int(local.min()), int(local.max())
        year_payloads.append(
            {
                "start": start,
                "full_blocks": full_blocks,
                "block": block,
                "offset": offset,
                "tail": tail,
                "first": first,
                "last": last,
            }
        )
    samples: list[float] = []
    attempts = 0
    maximum_attempts = cfg.random_control_samples * 20
    while len(samples) < cfg.random_control_samples and attempts < maximum_attempts:
        attempts += 1
        total = 0.0
        valid = True
        year_boundaries: list[tuple[int, int]] = []
        for payload in year_payloads:
            count = payload["full_blocks"]
            source_to_target = rng.permutation(count)
            target_to_source = np.argsort(source_to_target)
            first = payload["first"][target_to_source]
            last = payload["last"][target_to_source]
            nonempty = np.flatnonzero(last >= 0)
            if len(nonempty) > 1:
                gaps = (
                    (nonempty[1:] - nonempty[:-1]) * 2016
                    + first[nonempty[1:]]
                    - last[nonempty[:-1]]
                )
                if (gaps < policy.hold_bars).any():
                    valid = False
                    break
            mapped = (
                payload["start"]
                + source_to_target[payload["block"]] * 2016
                + payload["offset"]
            )
            values = log_factors[mapped]
            tail_values = log_factors[payload["tail"]]
            if not np.isfinite(values).all() or not np.isfinite(tail_values).all():
                valid = False
                break
            all_mapped = np.concatenate([mapped, payload["tail"]])
            if len(all_mapped):
                first_signal = int(all_mapped.min())
                last_signal = int(all_mapped.max())
                if len(payload["tail"]) and len(mapped):
                    if int(payload["tail"].min()) - int(mapped.max()) < policy.hold_bars:
                        valid = False
                        break
                year_boundaries.append((first_signal, last_signal))
            total += float(values.sum() + tail_values.sum())
        if valid and len(year_boundaries) > 1:
            boundary_gaps = [
                current[0] - previous[1]
                for previous, current in zip(year_boundaries, year_boundaries[1:])
            ]
            valid = min(boundary_gaps) >= policy.hold_bars
        if valid:
            samples.append(total)
    values = np.asarray(samples)
    return {
        "samples": int(len(values)),
        "attempts": attempts,
        "same_trade_count": len(trades),
        "preserved": "calendar-year counts and exact 7d-block time offsets",
        "observed_net_log_return": observed,
        "p_value_net_log_return": (
            float((1 + np.sum(values >= observed)) / (1 + len(values)))
            if len(values)
            else 1.0
        ),
        "random_positive_fraction": float((values > 0).mean()) if len(values) else None,
        "random_q95_net_log_return": float(np.quantile(values, 0.95)) if len(values) else None,
    }


def selection_gates(
    stats: dict[str, dict[str, Any]],
    stress: dict[str, Any],
    significance: dict[str, Any],
) -> dict[str, bool]:
    years = [stats[name] for name in YEAR_WINDOWS]
    halves = [stats[name] for name in HALF_WINDOWS]
    combined = stats["combined_2020_2022"]
    return {
        "every_calendar_year_absolute_return_positive": all(
            row["absolute_return_pct"] > 0.0 for row in years
        ),
        "positive_half_years_at_least_5_of_6": sum(
            row["absolute_return_pct"] > 0.0 for row in halves
        )
        >= 5,
        "strict_mdd_each_year_at_most_10": all(
            row["strict_mdd_pct"] <= 10.0 for row in years
        ),
        "combined_cagr_to_strict_mdd_at_least_2": combined[
            "cagr_to_strict_mdd"
        ]
        >= 2.0,
        "combined_trades_at_least_120": combined["trades"] >= 120,
        "each_calendar_year_trades_at_least_25": all(
            row["trades"] >= 25 for row in years
        ),
        "eight_bp_notional_side_stress_positive": stress["absolute_return_pct"] > 0.0,
        "bonferroni_weekly_signflip_p_at_most_0_10": significance[
            "bonferroni_p_value"
        ]
        <= 0.10,
    }


def evaluate_policy(
    engine: ExecutionEngine,
    features: pd.DataFrame,
    clocks: pd.DataFrame,
    policy: Policy,
    cfg: Config,
    policy_number: int,
) -> dict[str, Any]:
    indices = clock_indices(clocks, policy.policy_id)
    primary = trades_from_indices(
        engine, indices, side=policy.side, hold_bars=policy.hold_bars
    )
    stats = stats_windows(primary, cfg, cfg.base_cost_notional_per_side)
    combined = window_trades(primary, *WINDOWS["combined_2020_2022"])
    stress = strict_equity_stats(
        combined,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        leverage=cfg.leverage,
        cost_notional_per_side=cfg.stress_cost_notional_per_side,
    )
    direction_flip = trades_from_indices(
        engine, indices, side=-policy.side, hold_bars=policy.hold_bars
    )
    delayed_1 = trades_from_indices(
        engine, indices, side=policy.side, hold_bars=policy.hold_bars, delay_bars=1
    )
    delayed_12 = trades_from_indices(
        engine, indices, side=policy.side, hold_bars=policy.hold_bars, delay_bars=12
    )
    controls: dict[str, Any] = {}
    for name, trades in {
        "direction_flip": direction_flip,
        "delay_1_bar": delayed_1,
        "delay_12_bars": delayed_12,
        "venue_role_swap": generated_control_trades(engine, features, policy, "venue_role_swap"),
        "return_only": generated_control_trades(engine, features, policy, "return_only"),
        "premium_only": generated_control_trades(engine, features, policy, "premium_only"),
        "activity_only": generated_control_trades(engine, features, policy, "activity_only"),
        "no_premium": generated_control_trades(engine, features, policy, "no_premium"),
    }.items():
        controls[name] = (
            None
            if trades is None
            else strict_equity_stats(
                window_trades(trades, *WINDOWS["combined_2020_2022"]),
                start=WINDOWS["combined_2020_2022"][0],
                end=WINDOWS["combined_2020_2022"][1],
                leverage=cfg.leverage,
                cost_notional_per_side=cfg.base_cost_notional_per_side,
            )
        )
    significance = weekly_cluster_signflip(combined, cfg, policy_number)
    random_control = time_of_week_block_random(
        engine, combined, policy, cfg, policy_number
    )
    gates = selection_gates(stats, stress, significance)
    return {
        "policy": asdict(policy),
        "frozen_clock_events": int(len(indices)),
        "stats": stats,
        "eight_bp_notional_side_cost_stress": stress,
        "controls": controls,
        "weekly_cluster_signflip": significance,
        "time_of_week_block_random": random_control,
        "selection_gates": gates,
        "passes_selection": all(gates.values()),
    }


def rank_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    stats = trial["stats"]
    minimum_year_ratio = min(stats[name]["cagr_to_strict_mdd"] for name in YEAR_WINDOWS)
    return (
        -minimum_year_ratio,
        -stats["combined_2020_2022"]["cagr_to_strict_mdd"],
        trial["policy"]["policy_id"],
    )


def write_once(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# Coinbase–Binance Venue-Leadership Selection",
        "",
        "> 2020–2022만 열었으며 2023과 2024+는 봉인 상태다.",
        "",
        "| Rank | Policy | Family | Side | Hold | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래 | 판정 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, trial in enumerate(payload["trials"], start=1):
        policy = trial["policy"]
        stats = trial["stats"]["combined_2020_2022"]
        lines.append(
            f"| {rank} | {policy['policy_id']} | {policy['family']} | {policy['side']} | "
            f"{policy['hold_bars']} | {stats['absolute_return_pct']:.3f}% | "
            f"{stats['cagr_pct']:.3f}% | {stats['strict_mdd_pct']:.3f}% | "
            f"{stats['cagr_to_strict_mdd']:.3f} | {stats['trades']} | "
            f"{'PASS' if trial['passes_selection'] else 'REJECT'} |"
        )
    lines.extend(
        [
            "",
            "## 판정",
            "",
            f"- 상태: **{payload['decision']}**",
            f"- 통과 정책 수: {payload['passing_policies']}",
            f"- 선택 정책: `{payload.get('selected_policy_id')}`",
            "- 절대수익과 CAGR은 거래하지 않은 기간까지 포함한 전체 2020–2022 달력으로 계산했다.",
            "- strict MDD는 global/pre-entry HWM, 보유 중 favorable-before-adverse OHLC, "
            "진입·가상청산 비용 및 funding debit/credit의 최악 순서를 포함한다.",
            "- 2023은 선택 정책 manifest가 별도 커밋되기 전에는 열지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_config(cfg: Config) -> None:
    if cfg.leverage != 0.5:
        raise RuntimeError("selection leverage differs from preregistration")
    if cfg.base_cost_notional_per_side != 0.0006 or cfg.stress_cost_notional_per_side != 0.0008:
        raise RuntimeError("selection cost convention differs from preregistration")
    if cfg.signflip_samples != 5_000 or cfg.random_control_samples != 5_000:
        raise RuntimeError("selection control sample count differs from preregistration")
    for path in (cfg.output, cfg.policy_output, cfg.docs_output):
        if Path(path).exists():
            raise RuntimeError(f"selection output is append-only and already exists: {path}")


def run(cfg: Config) -> dict[str, Any]:
    validate_config(cfg)
    clean_head = require_clean_tracked_tree()
    selector_attestation = {
        "clean_head": clean_head,
        "selector": git_anchor(__file__),
        "selector_test": git_anchor(SELECTOR_TEST),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    market, funding, features, clocks, source_record = load_sources(cfg)
    engine = ExecutionEngine(market, funding, cfg)
    trials = [
        evaluate_policy(engine, features, clocks, policy, cfg, number)
        for number, policy in enumerate(policy_grid(), start=1)
    ]
    trials.sort(key=rank_key)
    passing = [trial for trial in trials if trial["passes_selection"]]
    selected = passing[0] if passing else None
    core: dict[str, Any] = {
        "protocol_version": "coinbase_spot_leadership_selection_v1",
        "outcomes_opened": True,
        "opened_window": ["2020-01-01", SELECTION_END],
        "holdout_2023_opened": False,
        "future_2024_plus_opened": False,
        "source_record": source_record,
        "execution_contract": {
            "entry": "next Binance 5m open",
            "fixed_hold_bars": True,
            "leverage": cfg.leverage,
            "base_cost_notional_per_side": cfg.base_cost_notional_per_side,
            "base_cost_account_per_side": cfg.leverage * cfg.base_cost_notional_per_side,
            "stress_cost_notional_per_side": cfg.stress_cost_notional_per_side,
            "funding": "exact source milliseconds; entry_time < funding_time <= exit_time",
            "strict_mdd": (
                "global/pre-entry HWM plus favorable-before-adverse held OHLC, "
                "funding credits at favorable HWM, funding debits at adverse path, "
                "and entry/hypothetical-liquidation costs"
            ),
            "cagr_clock": "full calendar including idle periods",
        },
        "multiple_testing": {
            "policies": len(policy_grid()),
            "weekly_cluster_signflip_samples_each": cfg.signflip_samples,
            "adjustment": "Bonferroni",
            "familywise_p_max": 0.10,
        },
        "control_contract": {
            "venue_role_swap_tokens": "ZR=-ZR,ZP=-ZP,ZV=-ZV,ZCB=ZBN,ZBN=ZCB",
            "time_of_week_random": (
                "permute complete 7d event blocks within each calendar year; preserve "
                "event count, side, within-block offsets, and nonoverlap"
            ),
            "ablation_controls_are_diagnostics_not_selection_gates": True,
        },
        "trials": trials,
        "passing_policies": len(passing),
        "selected_policy_id": selected["policy"]["policy_id"] if selected else None,
        "decision": "selected_pre_2023" if selected else "rejected_before_2023_holdout",
        "pre_outcome_selector_attestation": selector_attestation,
    }
    payload = {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_once(cfg.output, payload)
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text(render_docs(payload))
    if selected is not None:
        result_sha = sha256_file(cfg.output)
        policy_core = {
            "protocol_version": "coinbase_spot_leadership_frozen_policy_v1",
            "policy": selected["policy"],
            "selection_result_path": cfg.output,
            "selection_result_sha256": result_sha,
            "selection_result_manifest_hash": payload["manifest_hash"],
            "selection_rank": 1,
            "selection_gates": selected["selection_gates"],
            "holdout_2023_opened": False,
            "future_2024_plus_opened": False,
        }
        write_once(
            cfg.policy_output,
            {
                **policy_core,
                "manifest_hash": canonical_hash(policy_core),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    summary = []
    for rank, trial in enumerate(payload["trials"], start=1):
        stats = trial["stats"]["combined_2020_2022"]
        summary.append(
            {
                "rank": rank,
                "policy_id": trial["policy"]["policy_id"],
                "absolute_return_pct": stats["absolute_return_pct"],
                "cagr_pct": stats["cagr_pct"],
                "strict_mdd_pct": stats["strict_mdd_pct"],
                "cagr_to_strict_mdd": stats["cagr_to_strict_mdd"],
                "trades": stats["trades"],
                "pass": trial["passes_selection"],
            }
        )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "selected_policy_id": payload["selected_policy_id"],
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
