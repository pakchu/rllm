"""Freeze and evaluate a causal BTC chain-activity impulse alpha.

The information clock is a completed Coin Metrics BTC network day, not a
derivatives or price-derived event.  Selection is physically truncated before
2024.  The selected policy may be replayed on 2024+ only when the committed
pre-2024 manifest reproduces exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.strict_bar_backtest import _trade_stats


SELECTION_END = "2024-01-01"
FULL_CUTOFF = "2026-06-02"
FIT_START = "2021-03-01"
FIT_END = "2023-01-01"
WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2021": (FIT_START, "2022-01-01"),
    "fit_2022": ("2022-01-01", FIT_END),
    "select_2023_h1": (FIT_END, "2023-07-01"),
    "select_2023_h2": ("2023-07-01", SELECTION_END),
    "select_2023": (FIT_END, SELECTION_END),
    "test_2024": (SELECTION_END, "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026h1": ("2026-01-01", FULL_CUTOFF),
    "oos_2024_2026h1": (SELECTION_END, FULL_CUTOFF),
    "all_2021_2026h1": (FIT_START, FULL_CUTOFF),
}

EVENT_FEATURES = (
    "activity_shock_1d",
    "activity_shock_7d",
)
RULES = (
    "absorption_long",
    "confirmation_long",
    "exhaustion_short",
    "failure_short",
    "momentum",
    "reversal",
)
TAILS = (0.05, 0.10, 0.15, 0.20, 0.25)
HOLD_DAYS = (1, 3, 7)
EXPECTED_POLICY = {
    "event": "activity_shock_1d",
    "tail": 0.10,
    "rule": "momentum",
    "hold_days": 7,
}


@dataclass(frozen=True)
class Config:
    input_csv: str
    funding_csv: str
    network_csv: str
    output: str
    manifest_output: str
    docs_output: str = ""
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0012
    open_oos: bool = False
    random_control_count: int = 200
    random_seed: int = 20260716


def _read_before(path: str, date_column: str, cutoff: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="infer", chunksize=100_000):
        parsed = pd.to_datetime(chunk[date_column], utc=True, errors="raise", format="mixed").dt.tz_convert(None)
        keep = parsed < pd.Timestamp(cutoff)
        if keep.any():
            part = chunk.loc[keep].copy()
            part[date_column] = parsed.loc[keep]
            chunks.append(part)
    if not chunks:
        raise ValueError(f"no {date_column} rows before {cutoff}: {path}")
    return pd.concat(chunks, ignore_index=True)


def frame_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    for column in canonical:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].astype("datetime64[ns]").astype("int64")
    digest = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(digest.tobytes()).hexdigest()


def json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def rolling_z(values: pd.Series, window: int = 180) -> pd.Series:
    minimum = window // 2
    mean = values.rolling(window, min_periods=minimum).mean()
    std = values.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def load_network(path: str, *, cutoff: str) -> pd.DataFrame:
    frame = _read_before(path, "observation_date", cutoff)
    required = {
        "observation_date",
        "available_at",
        "AdrActCnt",
        "TxCnt",
        "TxTfrCnt",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"network source missing columns: {sorted(missing)}")
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="raise", format="mixed"
    ).dt.tz_convert(None)
    frame = frame.sort_values("observation_date").drop_duplicates("observation_date", keep="last").reset_index(drop=True)
    if (frame["available_at"] < frame["observation_date"] + pd.Timedelta(days=1)).any():
        raise RuntimeError("network day was available before the UTC day completed")
    numeric = sorted(required - {"observation_date", "available_at"})
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame[["AdrActCnt", "TxCnt", "TxTfrCnt"]] <= 0.0).any().any():
        raise ValueError("network source contains a non-positive core metric")
    return frame


def load_market_and_funding(cfg: Config, *, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = _read_before(cfg.input_csv, "date", cutoff)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    intervals = market["date"].diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("execution market is not a complete 5-minute grid")
    funding = _read_before(cfg.funding_csv, "date", cutoff)[["date", "funding_rate"]].copy()
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    funding = funding.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return market, funding


def execution_config(cfg: Config) -> ExecutionConfig:
    return ExecutionConfig(
        input_csv=cfg.input_csv,
        metrics_csv="",
        funding_csv=cfg.funding_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )


def build_daily_features(network: pd.DataFrame, engine: ExecutionEngine) -> pd.DataFrame:
    frame = network[["observation_date", "available_at"]].copy()
    addresses = np.log(network["AdrActCnt"])
    transactions = np.log(network["TxCnt"])
    transfers = np.log(network["TxTfrCnt"])
    activity_1d = (addresses.diff() + transactions.diff() + transfers.diff()) / 3.0
    activity_7d = (addresses.diff(7) + transactions.diff(7) + transfers.diff(7)) / 3.0
    frame["activity_shock_1d"] = rolling_z(activity_1d)
    frame["activity_shock_7d"] = rolling_z(activity_7d)

    market_dates = engine.dates.to_numpy(dtype="datetime64[ns]")
    anchors = np.searchsorted(
        market_dates,
        frame["available_at"].to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    valid = anchors < len(market_dates)
    frame = frame.loc[valid].copy().reset_index(drop=True)
    frame["anchor"] = anchors[valid].astype(np.int64)
    frame["anchor_date"] = engine.dates.iloc[frame["anchor"].to_numpy()].to_numpy()
    if (frame["anchor_date"] < frame["available_at"]).any():
        raise RuntimeError("network observation mapped before provider completion")
    close = pd.to_numeric(engine.market["close"], errors="raise").to_numpy(float)
    anchor = frame["anchor"].to_numpy(int)
    for hours in (24, 72):
        previous = anchor - hours * 12
        values = np.full(len(frame), np.nan)
        usable = previous >= 0
        values[usable] = np.log(close[anchor[usable]] / close[previous[usable]])
        frame[f"price_ret_{hours}h"] = values
    return frame.replace([np.inf, -np.inf], np.nan)


def event_onset(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    return active & ~np.r_[False, active[:-1]]


def policy_masks(
    frame: pd.DataFrame,
    *,
    event: str,
    threshold: float,
    rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    event_value = pd.to_numeric(frame[event], errors="coerce").to_numpy(float)
    active = event_onset(np.isfinite(event_value) & (event_value >= threshold))
    ret24 = frame["price_ret_24h"].to_numpy(float)
    ret72 = frame["price_ret_72h"].to_numpy(float)
    finite = np.isfinite(ret24) & np.isfinite(ret72)
    zeros = np.zeros(len(frame), dtype=bool)
    if rule == "absorption_long":
        long_active, short_active = active & finite & (ret24 < 0.0) & (ret72 <= 0.0), zeros
    elif rule == "confirmation_long":
        long_active, short_active = active & finite & (ret24 > 0.0) & (ret72 >= 0.0), zeros
    elif rule == "exhaustion_short":
        long_active, short_active = zeros, active & finite & (ret24 > 0.0) & (ret72 >= 0.0)
    elif rule == "failure_short":
        long_active, short_active = zeros, active & finite & (ret24 < 0.0) & (ret72 <= 0.0)
    elif rule == "momentum":
        long_active, short_active = active & finite & (ret24 > 0.0), active & finite & (ret24 < 0.0)
    elif rule == "reversal":
        long_active, short_active = active & finite & (ret24 < 0.0), active & finite & (ret24 > 0.0)
    else:
        raise ValueError(f"unknown rule: {rule}")
    return long_active, short_active


def schedule_policy(
    engine: ExecutionEngine,
    frame: pd.DataFrame,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    hold_days: int,
    start: str,
    end: str,
) -> list[Trade]:
    if np.any(long_active & short_active):
        raise RuntimeError("policy emitted conflicting sides")
    period = ((engine.dates >= pd.Timestamp(start)) & (engine.dates < pd.Timestamp(end))).to_numpy(bool)
    anchors = frame["anchor"].to_numpy(int)
    trades: list[Trade] = []
    next_allowed = 0
    for index in np.flatnonzero(long_active | short_active):
        signal = int(anchors[index])
        if signal < next_allowed or not period[signal]:
            continue
        side = 1 if bool(long_active[index]) else -1
        trade = engine.trade_at(signal, side, int(hold_days) * 288, 10_000, 10_000)
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def policy_schedules(
    engine: ExecutionEngine,
    frame: pd.DataFrame,
    policy: dict[str, Any],
    *,
    windows: tuple[str, ...],
) -> dict[str, list[Trade]]:
    long_active, short_active = policy_masks(
        frame,
        event=policy["event"],
        threshold=float(policy["threshold"]),
        rule=policy["rule"],
    )
    return {
        name: schedule_policy(
            engine,
            frame,
            long_active,
            short_active,
            hold_days=int(policy["hold_days"]),
            start=WINDOWS[name][0],
            end=WINDOWS[name][1],
        )
        for name in windows
    }


def schedule_stats(
    schedules: dict[str, list[Trade]],
    cfg: Config,
    *,
    cost_rate: float | None = None,
) -> dict[str, dict[str, Any]]:
    exec_cfg = execution_config(cfg)
    result: dict[str, dict[str, Any]] = {}
    for name, trades in schedules.items():
        stats = equity_stats(
            trades,
            start=WINDOWS[name][0],
            end=WINDOWS[name][1],
            cfg=exec_cfg,
            cost_rate=cost_rate,
        )
        stats["schedule_hash"] = _schedule_hash(trades)
        result[name] = stats
    return result


def net_trade_returns(trades: list[Trade], cfg: Config, *, cost_rate: float | None = None) -> list[float]:
    cost = cfg.fee_rate + cfg.slippage_rate if cost_rate is None else cost_rate
    edge = 1.0 - cfg.leverage * cost
    return [edge * trade.price_factor * trade.funding_factor * edge - 1.0 for trade in trades]


def search_selection(
    engine: ExecutionEngine,
    frame: pd.DataFrame,
    cfg: Config,
) -> list[dict[str, Any]]:
    fit = (
        (frame["observation_date"] >= pd.Timestamp(FIT_START))
        & (frame["observation_date"] < pd.Timestamp(FIT_END))
    ).to_numpy(bool)
    rows: list[dict[str, Any]] = []
    select_windows = ("fit_2021", "fit_2022", "select_2023_h1", "select_2023_h2", "select_2023")
    for event in EVENT_FEATURES:
        values = frame[event].to_numpy(float)
        reference = values[fit & np.isfinite(values)]
        if len(reference) < 400:
            raise RuntimeError(f"insufficient threshold history for {event}: {len(reference)}")
        for tail in TAILS:
            threshold = float(np.quantile(reference, 1.0 - tail))
            for rule in RULES:
                for hold_days in HOLD_DAYS:
                    policy = {
                        "event": event,
                        "tail": tail,
                        "threshold": threshold,
                        "rule": rule,
                        "hold_days": hold_days,
                    }
                    schedules = policy_schedules(engine, frame, policy, windows=select_windows)
                    stats = schedule_stats(schedules, cfg)
                    blocks = [stats[name] for name in select_windows[:4]]
                    eligible = min(row["trades"] for row in blocks) >= 4 and all(
                        row["absolute_return_pct"] > 0.0 for row in blocks
                    )
                    rows.append(
                        {
                            **policy,
                            "eligible": eligible,
                            "minimum_block_ratio": (
                                min(row["cagr_to_strict_mdd"] for row in blocks) if eligible else -999.0
                            ),
                            "stats": stats,
                        }
                    )
    rows.sort(
        key=lambda row: (
            row["eligible"],
            row["minimum_block_ratio"],
            row["stats"]["select_2023"]["cagr_to_strict_mdd"],
        ),
        reverse=True,
    )
    return rows


def shifted_masks(mask: np.ndarray, lag: int) -> np.ndarray:
    shifted = np.roll(mask, lag)
    if lag > 0:
        shifted[:lag] = False
    elif lag < 0:
        shifted[lag:] = False
    return shifted


def selection_controls(
    engine: ExecutionEngine,
    frame: pd.DataFrame,
    policy: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    long_active, short_active = policy_masks(
        frame,
        event=policy["event"],
        threshold=float(policy["threshold"]),
        rule=policy["rule"],
    )
    event = long_active | short_active
    ret24 = frame["price_ret_24h"].to_numpy(float)
    finite = np.isfinite(ret24)
    windows = ("fit_2021", "fit_2022", "select_2023_h1", "select_2023_h2", "select_2023")

    def stats_from_masks(long_mask: np.ndarray, short_mask: np.ndarray) -> dict[str, dict[str, Any]]:
        schedules = {
            name: schedule_policy(
                engine,
                frame,
                long_mask,
                short_mask,
                hold_days=int(policy["hold_days"]),
                start=WINDOWS[name][0],
                end=WINDOWS[name][1],
            )
            for name in windows
        }
        return schedule_stats(schedules, cfg)

    price_only = stats_from_masks(finite & (ret24 > 0.0), finite & (ret24 < 0.0))
    shifts: dict[str, Any] = {}
    for lag in (-28, -14, -7, 7, 14, 28):
        shifted = shifted_masks(event, lag)
        shifts[str(lag)] = stats_from_masks(shifted & finite & (ret24 > 0.0), shifted & finite & (ret24 < 0.0))

    rng = np.random.default_rng(cfg.random_seed)
    years = frame["observation_date"].dt.year.to_numpy()
    random_min_ratios: list[float] = []
    random_sum_returns: list[float] = []
    random_all_positive: list[bool] = []
    blocks = windows[:4]
    for _ in range(cfg.random_control_count):
        random_event = np.zeros(len(frame), dtype=bool)
        for year in (2021, 2022, 2023):
            pool = np.flatnonzero((years == year) & finite)
            count = int((event & (years == year)).sum())
            if count:
                random_event[rng.choice(pool, size=count, replace=False)] = True
        stats = stats_from_masks(
            random_event & finite & (ret24 > 0.0),
            random_event & finite & (ret24 < 0.0),
        )
        random_min_ratios.append(min(stats[name]["cagr_to_strict_mdd"] for name in blocks))
        random_sum_returns.append(sum(stats[name]["absolute_return_pct"] for name in blocks))
        random_all_positive.append(all(stats[name]["absolute_return_pct"] > 0.0 for name in blocks))
    return {
        "price_only": price_only,
        "event_day_shifts": shifts,
        "random_clock": {
            "count": cfg.random_control_count,
            "seed": cfg.random_seed,
            "all_blocks_positive_fraction": float(np.mean(random_all_positive)),
            "minimum_block_ratio_quantiles": {
                str(q): float(np.quantile(random_min_ratios, q)) for q in (0.5, 0.9, 0.95, 0.99)
            },
            "sum_block_return_quantiles": {
                str(q): float(np.quantile(random_sum_returns, q)) for q in (0.5, 0.9, 0.95, 0.99)
            },
        },
    }


def policy_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("event", "tail", "threshold", "rule", "hold_days")}


def render_selection_docs(payload: dict[str, Any]) -> str:
    policy = payload["selected_policy"]
    lines = [
        "# Chain activity impulse momentum — pre-2024 selection",
        "",
        "## Frozen mechanism",
        "",
        "A completed BTC network day is converted into the mean one-day log change of active addresses, "
        "transactions and transfers. When its 180-day rolling z-score enters the fitted upper tail, the policy "
        "trades in the direction of the already-completed prior 24-hour BTC move for seven days.",
        "",
        f"- event: `{policy['event']}`",
        f"- fit tail: `{policy['tail']}`; threshold `{policy['threshold']:.12f}`",
        f"- direction: `{policy['rule']}`",
        f"- hold: `{policy['hold_days']}` days",
        "- source availability: Coin Metrics `AssetEODCompletionTime`; signal uses a completed 5-minute bar after availability; fill is the next 5-minute open.",
        "- execution: 0.5x, 6 bp of notional per side, realized funding, full-calendar CAGR, intratrade strict MDD.",
        "",
        "## Selection evidence",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in policy["stats"].items():
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    random = payload["controls"]["random_clock"]
    lines.extend(
        [
            "",
            "## Falsification controls",
            "",
            f"- price-only seven-day momentum is reported separately and does not reproduce the candidate's all-block result.",
            f"- ±7/14/28-day event-clock shifts are reported; none is used to alter the frozen rule.",
            f"- `{random['count']}` year-stratified matched random clocks: all-block-positive fraction "
            f"`{random['all_blocks_positive_fraction']:.4f}`; q99 minimum-block ratio "
            f"`{random['minimum_block_ratio_quantiles']['0.99']:.4f}`.",
            "",
            "## Integrity boundary",
            "",
            "- No exchange-address-labelled metric is used.",
            "- The formal search opened 180 pre-2024 cells, so the family still has multiple-testing risk. Earlier exploratory prototypes also inspected other network transforms.",
            "- 2024+ BTC outcomes are globally research-seen in this repository; they are not pristine human holdout. "
            "The manifest still prevents this new data family from being re-ranked after its 2024+ replay.",
            "- Promotion requires positive test/eval/holdout performance, doubled-cost survival, and measured trade/PnL orthogonality.",
            "",
            "Official source: https://gitbook-docs.coinmetrics.io/access-our-data/api",
            "",
        ]
    )
    return "\n".join(lines)


def run_selection(cfg: Config) -> dict[str, Any]:
    market, funding = load_market_and_funding(cfg, cutoff=SELECTION_END)
    network = load_network(cfg.network_csv, cutoff=SELECTION_END)
    exec_cfg = execution_config(cfg)
    engine = ExecutionEngine(market, funding, exec_cfg)
    features = build_daily_features(network, engine)
    rows = search_selection(engine, features, cfg)
    selected = rows[0]
    for key, expected in EXPECTED_POLICY.items():
        if selected[key] != expected:
            raise RuntimeError(f"selection drift for {key}: {selected[key]!r} != {expected!r}")
    controls = selection_controls(engine, features, selected, cfg)
    selected_identity = policy_identity(selected)
    selected_schedules = policy_schedules(
        engine,
        features,
        selected_identity,
        windows=("fit_2021", "fit_2022", "select_2023_h1", "select_2023_h2", "select_2023"),
    )
    manifest_core = {
        "phase": "pre_2024_selection",
        "selection_end": SELECTION_END,
        "policy": selected_identity,
        "policy_hash": json_hash(selected_identity),
        "source_prefix_hashes": {
            "market": frame_hash(market),
            "funding": frame_hash(funding),
            "network": frame_hash(network),
            "features": frame_hash(features),
        },
        "schedule_hashes": {name: _schedule_hash(trades) for name, trades in selected_schedules.items()},
        "tested_cells": len(rows),
    }
    manifest = {
        **manifest_core,
        "manifest_hash": json_hash(manifest_core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(cfg.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.manifest_output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    payload = {
        "mode": "pre_2024_selection",
        "config": asdict(cfg),
        "protocol": {
            "threshold_fit": [FIT_START, FIT_END],
            "selection": [FIT_END, SELECTION_END],
            "future_opened": False,
            "network_availability": "AssetEODCompletionTime, then one completed 5m signal bar",
            "entry": "next 5m open",
            "realized_funding": True,
            "strict_mdd": "intratrade favorable-before-adverse high-water path",
            "full_calendar_cagr": True,
            "globally_research_seen_future": True,
        },
        "tested_cells": len(rows),
        "eligible_cells": sum(row["eligible"] for row in rows),
        "selected_policy": selected,
        "top_10": rows[:10],
        "controls": controls,
        "manifest_hash": manifest["manifest_hash"],
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if cfg.docs_output:
        Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.docs_output).write_text(render_selection_docs(payload))
    return payload


def run_oos(cfg: Config) -> dict[str, Any]:
    manifest = json.loads(Path(cfg.manifest_output).read_text())
    expected_hash = manifest.get("manifest_hash")
    manifest_core = {key: value for key, value in manifest.items() if key not in {"manifest_hash", "created_at"}}
    if json_hash(manifest_core) != expected_hash:
        raise RuntimeError("selection manifest hash mismatch")

    selection_market, selection_funding = load_market_and_funding(cfg, cutoff=SELECTION_END)
    selection_network = load_network(cfg.network_csv, cutoff=SELECTION_END)
    selection_engine = ExecutionEngine(selection_market, selection_funding, execution_config(cfg))
    selection_features = build_daily_features(selection_network, selection_engine)
    source_hashes = {
        "market": frame_hash(selection_market),
        "funding": frame_hash(selection_funding),
        "network": frame_hash(selection_network),
        "features": frame_hash(selection_features),
    }
    if source_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefix changed before OOS replay")
    replay = policy_schedules(
        selection_engine,
        selection_features,
        manifest["policy"],
        windows=tuple(manifest["schedule_hashes"]),
    )
    replay_hashes = {name: _schedule_hash(trades) for name, trades in replay.items()}
    if replay_hashes != manifest["schedule_hashes"]:
        raise RuntimeError("pre-2024 schedule changed before OOS replay")

    market, funding = load_market_and_funding(cfg, cutoff=FULL_CUTOFF)
    network = load_network(cfg.network_csv, cutoff=FULL_CUTOFF)
    if network["observation_date"].max() < pd.Timestamp("2026-05-31"):
        raise RuntimeError("OOS network source does not cover the frozen 2026 holdout")
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    features = build_daily_features(network, engine)
    windows = (
        "fit_2021",
        "fit_2022",
        "select_2023_h1",
        "select_2023_h2",
        "select_2023",
        "test_2024",
        "eval_2025",
        "holdout_2026h1",
        "oos_2024_2026h1",
        "all_2021_2026h1",
    )
    schedules = policy_schedules(engine, features, manifest["policy"], windows=windows)
    stats = schedule_stats(schedules, cfg)
    stress = schedule_stats(schedules, cfg, cost_rate=cfg.stress_cost_rate)
    significance = {
        name: _trade_stats(net_trade_returns(trades, cfg)) for name, trades in schedules.items()
    }
    payload = {
        "mode": "frozen_oos_replay",
        "config": asdict(cfg),
        "manifest_hash": expected_hash,
        "future_did_not_rerank": True,
        "globally_research_seen_future": True,
        "policy": manifest["policy"],
        "stats": stats,
        "double_cost_stats": stress,
        "trade_statistics": significance,
        "performance_pass": (
            stats["test_2024"]["absolute_return_pct"] > 0.0
            and stats["eval_2025"]["absolute_return_pct"] > 0.0
            and stats["holdout_2026h1"]["absolute_return_pct"] > 0.0
            and stats["oos_2024_2026h1"]["cagr_to_strict_mdd"] >= 3.0
            and stress["oos_2024_2026h1"]["absolute_return_pct"] > 0.0
        ),
    }
    Path(cfg.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--network-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--docs-output", default="")
    parser.add_argument("--leverage", type=float, default=Config.leverage)
    parser.add_argument("--fee-rate", type=float, default=Config.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=Config.slippage_rate)
    parser.add_argument("--stress-cost-rate", type=float, default=Config.stress_cost_rate)
    parser.add_argument("--open-oos", action="store_true")
    parser.add_argument("--random-control-count", type=int, default=Config.random_control_count)
    parser.add_argument("--random-seed", type=int, default=Config.random_seed)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    result = run_oos(cfg) if cfg.open_oos else run_selection(cfg)
    summary = {
        "mode": result["mode"],
        "policy": result.get("policy", result.get("selected_policy", {})),
        "performance_pass": result.get("performance_pass"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
