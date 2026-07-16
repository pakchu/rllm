"""Select one CVTT v2 policy using only frozen 2020-2022 outcomes."""
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

from training.build_cross_venue_temporal_torsion_v2_support import (
    build_clocks,
    episode_start,
    schedule_nonoverlap,
)
from training.export_wikimedia_attention_source import sha256_file
from training.preregister_cross_venue_temporal_torsion_alpha_v2 import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    Policy,
    canonical_hash,
    policy_grid,
    validate_manifest as validate_preregistration,
)


DEFAULT_SOURCE_MANIFEST = (
    "results/cross_venue_temporal_torsion_v2_source_manifest_2026-07-16.json"
)
DEFAULT_SUPPORT_MANIFEST = (
    "results/cross_venue_temporal_torsion_v2_support_manifest_2026-07-16.json"
)
DEFAULT_OUTPUT = "results/cross_venue_temporal_torsion_v2_selection_2026-07-16.json"
DEFAULT_POLICY_OUTPUT = (
    "results/cross_venue_temporal_torsion_v2_frozen_policy_2026-07-16.json"
)
DEFAULT_DOCS = "docs/cross-venue-temporal-torsion-v2-selection-2026-07-16.md"
SELECTOR_TEST = "tests/test_select_cross_venue_temporal_torsion_v2_pre2023.py"
SELECTION_END = "2023-01-01"
ENTRY_DELAY_BARS = 2
WEEK_BARS = 7 * 24 * 12

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


def validate_hashed_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError(f"manifest hash mismatch: {path}")
    return payload


def git_anchor(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    pathspec = str(candidate.resolve().relative_to(Path.cwd().resolve()))
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
        raise RuntimeError("CVTT v2 selection refuses a dirty Git tree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("CVTT v2 selection cannot resolve committed HEAD")
    return head


def verified_output(manifest: dict[str, Any], name: str) -> Path:
    metadata = manifest["outputs"][name]
    path = Path(metadata["path"])
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"frozen CVTT v2 {name} output is missing or mismatched")
    return path


def load_sources(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prereg = json.loads(Path(cfg.preregistration).read_text())
    validate_preregistration(prereg)
    source = validate_hashed_manifest(cfg.source_manifest)
    support = validate_hashed_manifest(cfg.support_manifest)
    if source.get("end_exclusive") != SELECTION_END:
        raise RuntimeError("CVTT v2 source did not preserve the 2023 holdout")
    if source.get("future_data_requested") is not False:
        raise RuntimeError("CVTT v2 source requested future data")
    if support.get("forward_trade_outcomes_opened") is not False:
        raise RuntimeError("CVTT v2 support already opened forward outcomes")
    if support.get("selection_may_open_forward_returns") is not True:
        raise RuntimeError("CVTT v2 support gates do not permit selection")
    if source.get("preregistration", {}).get("manifest_hash") != prereg["manifest_hash"]:
        raise RuntimeError("CVTT v2 source/preregistration hash mismatch")
    if support["hashes"]["preregistration_manifest_hash"] != prereg["manifest_hash"]:
        raise RuntimeError("CVTT v2 support/preregistration hash mismatch")
    if support["hashes"]["source_manifest_hash"] != source["manifest_hash"]:
        raise RuntimeError("CVTT v2 support/source hash mismatch")
    if sorted(support["passing_policy_ids"]) != [p.policy_id for p in policy_grid()]:
        raise RuntimeError("not all CVTT v2 policies passed support")

    market = pd.read_csv(
        verified_output(source, "market"),
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    funding = pd.read_csv(verified_output(source, "funding"))
    funding["date"] = pd.to_datetime(funding["date"], format="mixed", errors="raise")
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    features = pd.read_csv(
        verified_output(support, "features"),
        parse_dates=[
            "date",
            "feature_available_time_utc",
            "strategy_entry_earliest_time_utc",
        ],
    )
    clocks = pd.read_csv(
        verified_output(support, "clocks"),
        parse_dates=["signal_date", "entry_date"],
    )
    if not market["date"].equals(features["date"]):
        raise RuntimeError("CVTT v2 market and support grids differ")
    if market["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("CVTT v2 selector refuses 2023 market rows")
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="raise")
        if not np.isfinite(market[column]).all() or market[column].le(0).any():
            raise ValueError(f"invalid CVTT v2 market column: {column}")
    if funding["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("CVTT v2 funding crossed 2023")
    if clocks["signal_date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("CVTT v2 policy clocks crossed 2023")
    rows = clocks["signal_row"].to_numpy(np.int64)
    if not clocks["signal_date"].reset_index(drop=True).equals(
        features.loc[rows, "date"].reset_index(drop=True)
    ):
        raise RuntimeError("CVTT v2 clock row/date mapping is invalid")
    if not clocks["entry_date"].reset_index(drop=True).equals(
        features.loc[rows, "strategy_entry_earliest_time_utc"].reset_index(drop=True)
    ):
        raise RuntimeError("CVTT v2 clock entry timestamp is not t+10m")
    if features.loc[rows, "source_quarantined"].ne(0).any():
        raise RuntimeError("CVTT v2 clocks contain quarantined signals")
    regenerated = build_clocks(features)
    columns = ["policy_id", "route", "side", "hold_bars", "signal_date", "signal_row", "entry_date"]
    if not regenerated[columns].reset_index(drop=True).equals(
        clocks[columns].reset_index(drop=True)
    ):
        raise RuntimeError("CVTT v2 clocks differ from frozen causal rules")

    record = {
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
    return market, funding, features, clocks, record


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

    def trade_at(
        self, signal: int, side: int, hold_bars: int, extra_delay_bars: int = 0
    ) -> Trade | None:
        key = (int(signal), int(side), int(hold_bars), int(extra_delay_bars))
        if key in self._cache:
            return self._cache[key]
        if side not in (-1, 1) or hold_bars < 1 or extra_delay_bars < 0:
            raise ValueError("invalid CVTT v2 trade direction, hold, or delay")
        entry = int(signal) + ENTRY_DELAY_BARS + int(extra_delay_bars)
        exit_position = entry + int(hold_bars)
        if entry < 0 or exit_position >= len(self.open):
            self._cache[key] = None
            return None
        entry_price = float(self.open[entry])
        exit_price = float(self.open[exit_position])
        leverage = float(self.cfg.leverage)
        price_factor = 1.0 + leverage * side * (exit_price / entry_price - 1.0)
        held_high = max(float(np.max(self.high[entry:exit_position])), exit_price)
        held_low = min(float(np.min(self.low[entry:exit_position])), exit_price)
        favorable = held_high if side > 0 else held_low
        adverse = held_low if side > 0 else held_high
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
            raise ValueError("invalid CVTT v2 leveraged price/funding factor")
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


def trades_from_arrays(
    engine: ExecutionEngine,
    signals: Iterable[int],
    sides: Iterable[int],
    *,
    hold_bars: int,
    extra_delay_bars: int = 0,
) -> list[Trade]:
    trades: list[Trade] = []
    for signal, side in zip(signals, sides):
        trade = engine.trade_at(int(signal), int(side), hold_bars, extra_delay_bars)
        if trade is not None:
            trades.append(trade)
    return trades


def policy_clock(clocks: pd.DataFrame, policy_id: str) -> tuple[np.ndarray, np.ndarray]:
    selected = clocks.loc[clocks["policy_id"].eq(policy_id)]
    return (
        selected["signal_row"].to_numpy(np.int64),
        selected["side"].to_numpy(np.int8),
    )


def window_trades(trades: Iterable[Trade], start: str, end: str) -> list[Trade]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return [t for t in trades if t.entry_date >= start_ts and t.exit_date < end_ts]


def strict_equity_stats(
    trades: Iterable[Trade],
    *,
    start: str,
    end: str,
    leverage: float,
    cost_notional_per_side: float,
) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda t: (t.entry_position, t.exit_position))
    cost_factor = 1.0 - leverage * cost_notional_per_side
    if cost_factor <= 0.0:
        raise ValueError("CVTT v2 cost factor must remain positive")
    equity = peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_returns: list[float] = []
    previous_exit = -1
    for trade in ordered:
        if trade.entry_position < previous_exit:
            raise RuntimeError("CVTT v2 strict stats received overlapping positions")
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
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else -100.0
    mdd = strict_mdd * 100.0
    net = np.asarray(net_returns)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": len(ordered),
        "longs": sum(t.side > 0 for t in ordered),
        "shorts": sum(t.side < 0 for t in ordered),
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


def generated_control_clock(
    features: pd.DataFrame, policy: Policy, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    clean = features["source_quarantined"].eq(0).to_numpy(bool)
    if policy.route == "spot_preload_um_echo":
        side = features["spot_source_side"].to_numpy(np.int8)
        confirmed = features["spot_direction_confirmed"].eq(1).to_numpy(bool)
        source_preload = features["spot_flow_to_return_delay"].gt(0).to_numpy(bool)
        destination_echo = features["um_flow_to_return_delay"].lt(0).to_numpy(bool)
    elif policy.route == "um_preload_spot_echo":
        side = features["um_source_side"].to_numpy(np.int8)
        confirmed = features["um_direction_confirmed"].eq(1).to_numpy(bool)
        source_preload = features["um_flow_to_return_delay"].gt(0).to_numpy(bool)
        destination_echo = features["spot_flow_to_return_delay"].lt(0).to_numpy(bool)
    else:
        raise ValueError(policy.route)
    if mode == "aggregate_flow_without_crossed_clock":
        mask = clean & confirmed
    elif mode == "same_venue_preload_only":
        mask = clean & confirmed & source_preload
    elif mode == "same_venue_echo_only":
        mask = clean & confirmed & destination_echo
    else:
        raise ValueError(mode)
    starts = episode_start(pd.Series(mask)).to_numpy(bool)
    indices = schedule_nonoverlap(starts, policy.hold_bars)
    return indices, side[indices]


def weekly_cluster_signflip(
    trades: list[Trade], cfg: Config, policy_number: int
) -> dict[str, Any]:
    if not trades:
        return {"samples": 0, "raw_p_value": 1.0, "bonferroni_p_value": 1.0}
    cost_log = 2.0 * np.log(1.0 - cfg.leverage * cfg.base_cost_notional_per_side)
    frame = pd.DataFrame(
        {
            "week": [t.entry_date.to_period("W-SUN") for t in trades],
            "pre_cost_log": [np.log(t.price_factor * t.funding_factor) for t in trades],
        }
    )
    weekly = frame.groupby("week", sort=True)["pre_cost_log"].sum().to_numpy(float)
    observed = float(weekly.sum() + len(trades) * cost_log)
    rng = np.random.default_rng(cfg.random_seed + policy_number * 1009)
    simulated = np.empty(cfg.signflip_samples)
    for index in range(cfg.signflip_samples):
        simulated[index] = float(
            np.dot(rng.choice(np.asarray([-1.0, 1.0]), size=len(weekly)), weekly)
            + len(trades) * cost_log
        )
    raw_p = float((1 + np.sum(simulated >= observed)) / (1 + len(simulated)))
    return {
        "samples": int(len(simulated)),
        "weekly_clusters": int(len(weekly)),
        "observed_net_log_return": observed,
        "raw_p_value": raw_p,
        "bonferroni_hypotheses": len(policy_grid()),
        "bonferroni_p_value": min(1.0, raw_p * len(policy_grid())),
    }


def vectorized_log_factors(
    engine: ExecutionEngine,
    signals: np.ndarray,
    sides: np.ndarray,
    hold_bars: int,
    cfg: Config,
) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.int64)
    sides = np.asarray(sides, dtype=np.int8)
    entry = signals + ENTRY_DELAY_BARS
    exit_position = entry + hold_bars
    if (
        len(signals) != len(sides)
        or np.any(~np.isin(sides, (-1, 1)))
        or np.any(entry < 0)
        or np.any(exit_position >= len(engine.open))
    ):
        raise ValueError("invalid CVTT v2 vectorized trade clock")
    gross = sides * (engine.open[exit_position] / engine.open[entry] - 1.0)
    price_factor = 1.0 + cfg.leverage * gross
    entry_ns = engine.dates.iloc[entry].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    exit_ns = engine.dates.iloc[exit_position].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    left = np.searchsorted(engine.funding_times, entry_ns, side="right")
    right = np.searchsorted(engine.funding_times, exit_ns, side="right")
    if ((right - left) > 1).any():
        raise RuntimeError("CVTT v2 short hold crossed multiple funding events")
    funding_factor = np.ones(len(signals))
    crosses = right > left
    funding_factor[crosses] = 1.0 - (
        cfg.leverage * sides[crosses] * engine.funding_rates[left[crosses]]
    )
    combined = price_factor * funding_factor
    if not np.isfinite(combined).all() or (combined <= 0).any():
        raise ValueError("invalid CVTT v2 vectorized return factor")
    cost_log = 2.0 * np.log(1.0 - cfg.leverage * cfg.base_cost_notional_per_side)
    return np.log(combined) + cost_log


def time_of_week_block_random(
    engine: ExecutionEngine,
    signals: np.ndarray,
    sides: np.ndarray,
    policy: Policy,
    cfg: Config,
    policy_number: int,
) -> dict[str, Any]:
    if not len(signals):
        return {"samples": 0, "p_value_net_log_return": 1.0}
    observed = float(vectorized_log_factors(engine, signals, sides, policy.hold_bars, cfg).sum())
    date_values = engine.dates.to_numpy(dtype="datetime64[ns]")
    payloads: list[dict[str, Any]] = []
    for year in (2020, 2021, 2022):
        start = int(np.searchsorted(date_values, np.datetime64(f"{year}-01-01")))
        end = int(np.searchsorted(date_values, np.datetime64(f"{year + 1}-01-01")))
        selected = (signals >= start) & (signals < end)
        year_signals, year_sides = signals[selected], sides[selected]
        relative = year_signals - start
        full_blocks = (end - start) // WEEK_BARS
        in_full = relative < full_blocks * WEEK_BARS
        payloads.append(
            {
                "start": start,
                "full_blocks": full_blocks,
                "block": relative[in_full] // WEEK_BARS,
                "offset": relative[in_full] % WEEK_BARS,
                "sides": year_sides[in_full],
                "tail_signals": year_signals[~in_full],
                "tail_sides": year_sides[~in_full],
            }
        )
    rng = np.random.default_rng(cfg.random_seed + policy_number * 2027)
    samples: list[float] = []
    attempts = 0
    maximum_attempts = cfg.random_control_samples * 30
    while len(samples) < cfg.random_control_samples and attempts < maximum_attempts:
        attempts += 1
        mapped_signals: list[np.ndarray] = []
        mapped_sides: list[np.ndarray] = []
        for item in payloads:
            permutation = rng.permutation(item["full_blocks"])
            mapped_signals.extend(
                [
                    item["start"] + permutation[item["block"]] * WEEK_BARS + item["offset"],
                    item["tail_signals"],
                ]
            )
            mapped_sides.extend([item["sides"], item["tail_sides"]])
        candidate_signals = np.concatenate(mapped_signals)
        candidate_sides = np.concatenate(mapped_sides)
        order = np.argsort(candidate_signals, kind="stable")
        candidate_signals, candidate_sides = candidate_signals[order], candidate_sides[order]
        if len(candidate_signals) > 1 and np.diff(candidate_signals).min() < policy.hold_bars:
            continue
        if candidate_signals.max() > len(engine.open) - policy.hold_bars - ENTRY_DELAY_BARS - 1:
            continue
        samples.append(
            float(
                vectorized_log_factors(
                    engine, candidate_signals, candidate_sides, policy.hold_bars, cfg
                ).sum()
            )
        )
    values = np.asarray(samples)
    return {
        "samples": int(len(values)),
        "attempts": attempts,
        "same_trade_count": int(len(signals)),
        "preserved": "calendar-year counts, side labels, and exact 7d-block offsets",
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
    stats: dict[str, dict[str, Any]], stress: dict[str, Any], significance: dict[str, Any]
) -> dict[str, bool]:
    years = [stats[name] for name in YEAR_WINDOWS]
    halves = [stats[name] for name in HALF_WINDOWS]
    combined = stats["combined_2020_2022"]
    return {
        "every_calendar_year_absolute_return_positive": all(
            row["absolute_return_pct"] > 0 for row in years
        ),
        "positive_half_years_at_least_5_of_6": sum(
            row["absolute_return_pct"] > 0 for row in halves
        )
        >= 5,
        "strict_mdd_each_year_at_most_12": all(
            row["strict_mdd_pct"] <= 12.0 for row in years
        ),
        "combined_cagr_to_strict_mdd_at_least_2": combined[
            "cagr_to_strict_mdd"
        ]
        >= 2.0,
        "combined_trades_at_least_600": combined["trades"] >= 600,
        "each_calendar_year_trades_at_least_150": all(
            row["trades"] >= 150 for row in years
        ),
        "eight_bp_notional_side_stress_positive": stress["absolute_return_pct"] > 0,
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
    signals, sides = policy_clock(clocks, policy.policy_id)
    primary = trades_from_arrays(engine, signals, sides, hold_bars=policy.hold_bars)
    stats = stats_windows(primary, cfg, cfg.base_cost_notional_per_side)
    combined = window_trades(primary, *WINDOWS["combined_2020_2022"])
    stress = strict_equity_stats(
        combined,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        leverage=cfg.leverage,
        cost_notional_per_side=cfg.stress_cost_notional_per_side,
    )
    controls: dict[str, Any] = {}
    other_side_column = (
        "um_source_side" if policy.route == "spot_preload_um_echo" else "spot_source_side"
    )
    primary_controls = {
        "direction_flip": trades_from_arrays(
            engine, signals, -sides, hold_bars=policy.hold_bars
        ),
        "route_side_swap": trades_from_arrays(
            engine,
            signals,
            features.loc[signals, other_side_column].to_numpy(np.int8),
            hold_bars=policy.hold_bars,
        ),
        "delay_1_bar": trades_from_arrays(
            engine, signals, sides, hold_bars=policy.hold_bars, extra_delay_bars=1
        ),
        "delay_12_bars": trades_from_arrays(
            engine, signals, sides, hold_bars=policy.hold_bars, extra_delay_bars=12
        ),
    }
    for mode in (
        "aggregate_flow_without_crossed_clock",
        "same_venue_preload_only",
        "same_venue_echo_only",
    ):
        control_signals, control_sides = generated_control_clock(features, policy, mode)
        primary_controls[mode] = trades_from_arrays(
            engine, control_signals, control_sides, hold_bars=policy.hold_bars
        )
    for name, trades in primary_controls.items():
        controls[name] = strict_equity_stats(
            window_trades(trades, *WINDOWS["combined_2020_2022"]),
            start=WINDOWS["combined_2020_2022"][0],
            end=WINDOWS["combined_2020_2022"][1],
            leverage=cfg.leverage,
            cost_notional_per_side=cfg.base_cost_notional_per_side,
        )
    significance = weekly_cluster_signflip(combined, cfg, policy_number)
    random_control = time_of_week_block_random(
        engine, signals, sides, policy, cfg, policy_number
    )
    gates = selection_gates(stats, stress, significance)
    return {
        "policy": asdict(policy),
        "frozen_clock_events": int(len(signals)),
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
        "# CVTT v2 2020–2022 selection",
        "",
        "> 2020–2022만 열었고 2023은 봉인 상태다. 모든 CAGR은 무거래 시간을 포함한다.",
        "",
        "| Rank | Policy | Route | Hold | 절대수익 | CAGR | strict MDD | CAGR/MDD | 거래 | 판정 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, trial in enumerate(payload["trials"], start=1):
        policy = trial["policy"]
        stats = trial["stats"]["combined_2020_2022"]
        lines.append(
            f"| {rank} | {policy['policy_id']} | {policy['route']} | {policy['hold_bars']} | "
            f"{stats['absolute_return_pct']:.3f}% | {stats['cagr_pct']:.3f}% | "
            f"{stats['strict_mdd_pct']:.3f}% | {stats['cagr_to_strict_mdd']:.3f} | "
            f"{stats['trades']} | {'PASS' if trial['passes_selection'] else 'REJECT'} |"
        )
    lines.extend(
        [
            "",
            "## 판정",
            "",
            f"- 상태: **{payload['decision']}**",
            f"- 통과 정책 수: {payload['passing_policies']}",
            f"- 선택 정책: `{payload.get('selected_policy_id')}`",
            "- strict MDD는 global/pre-entry HWM, 보유 중 favorable-before-adverse OHLC, "
            "진입·가상청산 비용, funding debit/credit을 포함한다.",
            "- 정책 선택 결과를 별도 커밋하기 전에는 2023을 열지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_config(cfg: Config) -> None:
    if cfg.leverage != 0.5:
        raise RuntimeError("CVTT v2 leverage differs from preregistration")
    if cfg.base_cost_notional_per_side != 0.0006 or cfg.stress_cost_notional_per_side != 0.0008:
        raise RuntimeError("CVTT v2 costs differ from preregistration")
    if cfg.signflip_samples != 5_000 or cfg.random_control_samples != 5_000:
        raise RuntimeError("CVTT v2 control samples differ from preregistration")
    for path in (cfg.output, cfg.policy_output, cfg.docs_output):
        if Path(path).exists():
            raise RuntimeError(f"CVTT v2 append-only output exists: {path}")


def run(cfg: Config) -> dict[str, Any]:
    validate_config(cfg)
    clean_head = require_clean_tracked_tree()
    attestation = {
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
        "protocol_version": "cross_venue_temporal_torsion_selection_v2",
        "outcomes_opened": True,
        "opened_window": ["2020-01-01", SELECTION_END],
        "holdout_2023_opened": False,
        "future_2024_plus_opened": False,
        "source_record": source_record,
        "execution_contract": {
            "entry": "USD-M open at signal bucket t+10m",
            "entry_delay_bars_from_bucket_open": ENTRY_DELAY_BARS,
            "fixed_hold_bars": True,
            "leverage": cfg.leverage,
            "base_cost_notional_per_side": cfg.base_cost_notional_per_side,
            "stress_cost_notional_per_side": cfg.stress_cost_notional_per_side,
            "funding": "exact source milliseconds; entry_time < funding_time <= exit_time",
            "strict_mdd": (
                "global/pre-entry HWM plus favorable-before-adverse held OHLC, funding "
                "credits at favorable HWM, funding debits at adverse path, and entry/"
                "hypothetical-liquidation costs"
            ),
            "cagr_clock": "full calendar including idle periods",
        },
        "multiple_testing": {
            "policies": len(policy_grid()),
            "weekly_cluster_signflip_samples_each": cfg.signflip_samples,
            "adjustment": "Bonferroni",
            "familywise_p_max": 0.10,
        },
        "controls": {
            "diagnostic_not_selection_gates": True,
            "time_of_week_random": (
                "permute complete 7d event blocks within year while retaining side "
                "labels, exact within-week offsets, count, and nonoverlap"
            ),
        },
        "trials": trials,
        "passing_policies": len(passing),
        "selected_policy_id": selected["policy"]["policy_id"] if selected else None,
        "decision": "selected_pre_2023" if selected else "rejected_before_2023_holdout",
        "pre_outcome_selector_attestation": attestation,
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
        policy_core = {
            "protocol_version": "cross_venue_temporal_torsion_frozen_policy_v2",
            "policy": selected["policy"],
            "selection_result_path": cfg.output,
            "selection_result_sha256": sha256_file(cfg.output),
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
