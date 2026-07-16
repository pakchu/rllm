"""Select a Wikimedia attention-divergence policy without opening 2023+.

Only the preregistered 14-policy family is evaluated.  Market and funding
inputs are physically truncated before 2023, and the page-view source manifest
must prove it ends on 2022-12-31.  A passing policy is frozen for a later,
single 2023 holdout replay; this module never reads that holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.export_wikimedia_attention_source import sha256_file
from training.preregister_wikimedia_attention_divergence_alpha import (
    ARTICLES,
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    Policy,
    SELECTION_END,
    canonical_hash,
    policy_grid,
    validate_manifest as validate_preregistration,
)
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
)


DEFAULT_INPUT = "data/wikimedia_alpha_btcusdt_5m_2020_2022.csv.gz"
DEFAULT_FUNDING = "data/wikimedia_alpha_funding_2020_2022.csv.gz"
DEFAULT_MARKET_PREFIX_MANIFEST = (
    "results/wikimedia_attention_selection_market_prefix_manifest_2026-07-16.json"
)
DEFAULT_ATTENTION = "data/wikimedia_crypto_attention_daily_2020_2022.csv.gz"
DEFAULT_SOURCE_MANIFEST = (
    "results/wikimedia_crypto_attention_source_manifest_2020_2022_2026-07-16.json"
)
DEFAULT_OUTPUT = "results/wikimedia_attention_divergence_selection_2026-07-16.json"
DEFAULT_POLICY_OUTPUT = (
    "results/wikimedia_attention_divergence_frozen_policy_2026-07-16.json"
)
DEFAULT_DOCS = "docs/wikimedia-attention-divergence-selection-2026-07-16.md"

WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2020": ("2020-01-01", "2021-01-01"),
    "fit_2021": ("2021-01-01", "2022-01-01"),
    "selection_2022": ("2022-01-01", SELECTION_END),
    "combined_2020_2022": ("2020-01-01", SELECTION_END),
}


@dataclass(frozen=True)
class Config:
    input_csv: str = DEFAULT_INPUT
    funding_csv: str = DEFAULT_FUNDING
    attention_csv: str = DEFAULT_ATTENTION
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    market_prefix_manifest: str = DEFAULT_MARKET_PREFIX_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    output: str = DEFAULT_OUTPUT
    policy_output: str = DEFAULT_POLICY_OUTPUT
    docs_output: str = DEFAULT_DOCS
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    random_controls: int = 5_000
    random_seed: int = 20260716


def resolve_existing(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / candidate
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def frame_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    for column in canonical:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].astype("datetime64[ns]").astype("int64")
    digest = pd.util.hash_pandas_object(canonical, index=False).to_numpy(np.uint64)
    return hashlib.sha256(digest.tobytes()).hexdigest()


def load_selection_sources(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prereg_path = resolve_existing(cfg.preregistration)
    prereg = json.loads(prereg_path.read_text())
    validate_preregistration(prereg)
    source_manifest_path = resolve_existing(cfg.source_manifest)
    source_manifest = json.loads(source_manifest_path.read_text())
    core = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"manifest_hash", "retrieved_at"}
    }
    if canonical_hash(core) != source_manifest.get("manifest_hash"):
        raise RuntimeError("Wikimedia source manifest hash mismatch")
    if source_manifest.get("end") != "2022-12-31":
        raise RuntimeError("Wikimedia selection source does not end on 2022-12-31")
    if source_manifest.get("future_data_requested") is not False:
        raise RuntimeError("Wikimedia source manifest opened future data")
    if source_manifest.get("preregistration_manifest_hash") != prereg["manifest_hash"]:
        raise RuntimeError("Wikimedia source and preregistration hashes disagree")

    market_manifest_path = resolve_existing(cfg.market_prefix_manifest)
    market_manifest = json.loads(market_manifest_path.read_text())
    market_core = {
        key: value
        for key, value in market_manifest.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(market_core) != market_manifest.get("manifest_hash"):
        raise RuntimeError("Wikimedia market prefix manifest hash mismatch")
    if market_manifest.get("future_outcomes_opened") is not False:
        raise RuntimeError("Wikimedia market prefix opened future outcomes")
    if market_manifest.get("cutoff_exclusive") != SELECTION_END:
        raise RuntimeError("Wikimedia market prefix cutoff mismatch")
    if market_manifest.get("preregistration_manifest_hash") != prereg["manifest_hash"]:
        raise RuntimeError("market prefix and preregistration hashes disagree")

    attention_path = resolve_existing(cfg.attention_csv)
    if sha256_file(attention_path) != source_manifest.get("output_sha256"):
        raise RuntimeError("Wikimedia attention source file hash mismatch")
    attention = pd.read_csv(attention_path, compression="infer")
    attention["date"] = pd.to_datetime(attention["date"], errors="raise")
    if len(attention) == 0 or attention["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("attention source crossed the 2023 holdout boundary")
    if attention["date"].duplicated().any():
        raise RuntimeError("attention source has duplicate UTC dates")

    market_path = resolve_existing(cfg.input_csv)
    if sha256_file(market_path) != market_manifest["market"]["sha256"]:
        raise RuntimeError("market prefix file hash mismatch")
    market = pd.read_csv(
        market_path,
        compression="infer",
        usecols=["date", "open", "high", "low", "close"],
    )
    market["date"] = pd.to_datetime(
        market["date"], utc=True, errors="raise", format="mixed"
    ).dt.tz_convert(None)
    market = market.sort_values("date").reset_index(drop=True)
    if market["date"].duplicated().any() or market["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("market selection prefix is not unique and sealed")
    intervals = market["date"].diff().dropna()
    if not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("market selection prefix is not a complete 5-minute grid")
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="raise")
        if not np.isfinite(market[column]).all() or (market[column] <= 0.0).any():
            raise ValueError(f"invalid market prices: {column}")

    funding_path = resolve_existing(cfg.funding_csv)
    if sha256_file(funding_path) != market_manifest["funding"]["sha256"]:
        raise RuntimeError("funding prefix file hash mismatch")
    funding = pd.read_csv(
        funding_path, compression="infer", usecols=["date", "funding_rate"]
    )
    funding["date"] = pd.to_datetime(
        funding["date"], utc=True, errors="raise", format="mixed"
    ).dt.tz_convert(None)
    funding = funding.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    if not np.isfinite(funding["funding_rate"]).all():
        raise ValueError("invalid funding rate")
    if funding["date"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("funding selection prefix crossed 2023")

    source_record = {
        "preregistration": {
            "path": str(prereg_path),
            "file_sha256": sha256_file(prereg_path),
            "manifest_hash": prereg["manifest_hash"],
        },
        "attention_manifest": {
            "path": str(source_manifest_path),
            "file_sha256": sha256_file(source_manifest_path),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "market_prefix_manifest": {
            "path": str(market_manifest_path),
            "file_sha256": sha256_file(market_manifest_path),
            "manifest_hash": market_manifest["manifest_hash"],
        },
        "attention": {
            "path": str(attention_path),
            "file_sha256": sha256_file(attention_path),
            "rows": int(len(attention)),
            "prefix_hash": frame_hash(attention),
        },
        "market": {
            "path": str(market_path),
            "rows": int(len(market)),
            "prefix_hash": frame_hash(market),
            "max_date": str(market["date"].max()),
        },
        "funding": {
            "path": str(funding_path),
            "rows": int(len(funding)),
            "prefix_hash": frame_hash(funding),
            "max_date": str(funding["date"].max()),
        },
    }
    return market, funding, attention, source_record


def lagged_robust_zscore(
    values: pd.Series, *, window: int = 90, minimum: int = 45
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1)
    median = prior.rolling(window, min_periods=minimum).median()
    mad = prior.rolling(window, min_periods=minimum).apply(
        lambda sample: float(np.median(np.abs(sample - np.median(sample)))), raw=True
    )
    scale = (1.4826 * mad).where(mad > 1e-12)
    return (numeric - median) / scale


def build_daily_features(attention: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "source_complete",
        "project_user_views",
        *(f"{article.lower()}_views" for article in ARTICLES),
        *(f"{article.lower()}_per_million" for article in ARTICLES),
    }
    missing = required.difference(attention.columns)
    if missing:
        raise ValueError(f"attention source missing columns: {sorted(missing)}")
    daily_market = market.loc[
        (market["date"].dt.hour == 23) & (market["date"].dt.minute == 55),
        ["date", "close"],
    ].copy()
    daily_market["date"] = daily_market["date"].dt.normalize()
    expected_days = market["date"].dt.normalize().drop_duplicates()
    if len(daily_market) != len(expected_days):
        raise RuntimeError("market is missing a completed 23:55 UTC close")
    frame = attention.copy().sort_values("date").reset_index(drop=True)
    frame = frame.merge(daily_market, on="date", how="left", validate="one_to_one")
    normalized_columns = [f"{article.lower()}_per_million" for article in ARTICLES]
    normalized = frame[normalized_columns].apply(pd.to_numeric, errors="coerce")
    frame["broad_attention"] = normalized.sum(axis=1, min_count=len(normalized_columns))
    article_columns = [f"{article.lower()}_views" for article in ARTICLES]
    raw = frame[article_columns].apply(pd.to_numeric, errors="coerce")
    raw_total = raw.sum(axis=1, min_count=len(article_columns)).replace(0.0, np.nan)
    frame["bitcoin_share"] = raw["bitcoin_views"] / raw_total
    frame["broad_attention_z"] = lagged_robust_zscore(np.log1p(frame["broad_attention"]))
    frame["bitcoin_share_z"] = lagged_robust_zscore(frame["bitcoin_share"])
    log_close = np.log(pd.to_numeric(frame["close"], errors="coerce"))
    frame["price_return_1d"] = log_close - log_close.shift(1)
    frame["price_return_3d"] = log_close - log_close.shift(3)
    frame["anchor_date"] = frame["date"] + pd.Timedelta(days=2, hours=12, minutes=5)
    complete = pd.to_numeric(frame["source_complete"], errors="coerce").eq(1)
    feature_columns = [
        "broad_attention_z",
        "bitcoin_share_z",
        "price_return_1d",
        "price_return_3d",
    ]
    frame.loc[~complete, feature_columns] = np.nan
    return frame


def policy_events(features: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    price_column = f"price_return_{policy.price_horizon_days}d"
    price = features[price_column].to_numpy(float)
    broad = features["broad_attention_z"].to_numpy(float)
    share = features["bitcoin_share_z"].to_numpy(float)
    finite = np.isfinite(price) & np.isfinite(broad)
    if policy.family == "broad_attention_reversal":
        active = finite & (broad >= policy.attention_threshold) & (
            np.abs(price) >= policy.price_threshold
        )
        side = -np.sign(price)
    elif policy.family == "bitcoin_share_reversal":
        active = finite & np.isfinite(share) & (share >= policy.attention_threshold)
        active &= broad >= 1.0
        active &= np.abs(price) >= policy.price_threshold
        side = -np.sign(price)
    elif policy.family == "silent_impulse_continuation":
        active = finite & (broad <= policy.attention_threshold) & (
            np.abs(price) >= policy.price_threshold
        )
        side = np.sign(price)
    else:
        raise ValueError(f"unknown policy family: {policy.family}")
    active &= side != 0.0
    return pd.DataFrame(
        {
            "observation_date": features.loc[active, "date"].to_numpy(),
            "anchor_date": features.loc[active, "anchor_date"].to_numpy(),
            "side": side[active].astype(np.int8),
            "broad_attention_z": broad[active],
            "bitcoin_share_z": share[active],
            "price_return": price[active],
        }
    ).sort_values("anchor_date").reset_index(drop=True)


def build_schedule(
    engine: ExecutionEngine,
    events: pd.DataFrame,
    policy: Policy,
    *,
    start: str,
    end: str,
    invert: bool = False,
    delay_days: int = 0,
) -> list[Trade]:
    date_to_position = {date: i for i, date in enumerate(engine.dates)}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    trades: list[Trade] = []
    next_allowed = 0
    for row in events.itertuples(index=False):
        anchor = pd.Timestamp(row.anchor_date) + pd.Timedelta(days=delay_days)
        signal = date_to_position.get(anchor)
        if signal is None or signal < next_allowed:
            continue
        side = -int(row.side) if invert else int(row.side)
        trade = engine.trade_at(
            signal,
            side,
            int(policy.hold_days * 288),
            10_000,
            10_000,
        )
        if trade is None:
            continue
        entry_date = engine.dates.iloc[trade.entry_position]
        exit_date = engine.dates.iloc[trade.exit_position]
        if entry_date < start_ts or exit_date >= end_ts:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def execution_config(cfg: Config) -> ExecutionConfig:
    return ExecutionConfig(
        input_csv=str(resolve_existing(cfg.input_csv)),
        metrics_csv="",
        funding_csv=str(resolve_existing(cfg.funding_csv)),
        output=cfg.output,
        manifest_output=cfg.policy_output,
        exclude_from=SELECTION_END,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )


def strict_equity_stats(
    trades: Iterable[Trade],
    *,
    start: str,
    end: str,
    cfg: ExecutionConfig,
    cost_rate: float | None = None,
) -> dict[str, Any]:
    """Include entry and hypothetical liquidation costs in strict MDD."""
    cost = float(cfg.fee_rate + cfg.slippage_rate if cost_rate is None else cost_rate)
    side_factor = 1.0 - float(cfg.leverage) * cost
    equity = peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_returns: list[float] = []
    sides: list[int] = []
    for trade in trades:
        entry_equity = equity
        favorable_equity = equity * side_factor * trade.favorable_price_factor
        liquidation_equity = (
            equity
            * side_factor
            * trade.funding_debit_factor
            * trade.adverse_price_factor
            * side_factor
        )
        intratrade_peak = max(peak, favorable_equity)
        strict_mdd = max(strict_mdd, 1.0 - liquidation_equity / intratrade_peak)
        peak = intratrade_peak
        equity *= side_factor * trade.price_factor * trade.funding_factor * side_factor
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        net_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(trade.gross_return)
        sides.append(trade.side)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    returns = np.asarray(net_returns, dtype=float)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": int(len(returns)),
        "longs": int(sum(side > 0 for side in sides)),
        "shorts": int(sum(side < 0 for side in sides)),
        "mean_net_bps": float(returns.mean() * 10_000.0) if len(returns) else 0.0,
        "mean_gross_bps": float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0,
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
    }


def evaluate_policy(
    engine: ExecutionEngine,
    features: pd.DataFrame,
    policy: Policy,
    cfg: Config,
) -> dict[str, Any]:
    events = policy_events(features, policy)
    schedules = {
        name: build_schedule(engine, events, policy, start=start, end=end)
        for name, (start, end) in WINDOWS.items()
    }
    exec_cfg = execution_config(cfg)
    stats = {
        name: strict_equity_stats(
            trades, start=WINDOWS[name][0], end=WINDOWS[name][1], cfg=exec_cfg
        )
        for name, trades in schedules.items()
    }
    combined = schedules["combined_2020_2022"]
    double_cost = strict_equity_stats(
        combined,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        cfg=exec_cfg,
        cost_rate=2.0 * (cfg.fee_rate + cfg.slippage_rate),
    )
    inverted = build_schedule(
        engine,
        events,
        policy,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        invert=True,
    )
    inverted_stats = strict_equity_stats(
        inverted,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        cfg=exec_cfg,
    )
    delayed = build_schedule(
        engine,
        events,
        policy,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        delay_days=1,
    )
    delayed_stats = strict_equity_stats(
        delayed,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        cfg=exec_cfg,
    )
    gates = selection_gates(stats, double_cost, inverted_stats)
    return {
        "policy": asdict(policy),
        "raw_events": int(len(events)),
        "stats": stats,
        "double_cost": double_cost,
        "inverted_side": inverted_stats,
        "one_day_later_entry": delayed_stats,
        "schedule_hashes": {name: _schedule_hash(value) for name, value in schedules.items()},
        "selection_gates": gates,
        "passes_selection": bool(all(gates.values())),
    }


def selection_gates(
    stats: dict[str, dict[str, Any]],
    double_cost: dict[str, Any],
    inverted: dict[str, Any],
) -> dict[str, bool]:
    combined = stats["combined_2020_2022"]
    selection = stats["selection_2022"]
    years = [stats[name] for name in ("fit_2020", "fit_2021", "selection_2022")]
    return {
        "combined_absolute_return_positive": combined["absolute_return_pct"] > 0.0,
        "combined_ratio_at_least_2": combined["cagr_to_strict_mdd"] >= 2.0,
        "selection_2022_absolute_return_positive": selection["absolute_return_pct"] > 0.0,
        "selection_2022_ratio_at_least_2": selection["cagr_to_strict_mdd"] >= 2.0,
        "every_calendar_year_absolute_return_positive": all(
            row["absolute_return_pct"] > 0.0 for row in years
        ),
        "combined_trades_at_least_18": combined["trades"] >= 18,
        "each_calendar_year_trades_at_least_4": all(row["trades"] >= 4 for row in years),
        "combined_strict_mdd_at_most_15": combined["strict_mdd_pct"] <= 15.0,
        "each_calendar_year_strict_mdd_at_most_15": all(
            row["strict_mdd_pct"] <= 15.0 for row in years
        ),
        "double_cost_combined_positive": double_cost["absolute_return_pct"] > 0.0,
        "double_cost_combined_ratio_at_least_1_5": double_cost["cagr_to_strict_mdd"] >= 1.5,
        "inverted_side_combined_nonpositive": inverted["absolute_return_pct"] <= 0.0,
    }


def rank_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    stats = trial["stats"]
    minimum_year_ratio = min(
        stats[name]["cagr_to_strict_mdd"]
        for name in ("fit_2020", "fit_2021", "selection_2022")
    )
    policy = trial["policy"]
    policy_tuple = (
        str(policy["family"]),
        float(policy["attention_threshold"]),
        int(policy["price_horizon_days"]),
        float(policy["price_threshold"]),
        int(policy["hold_days"]),
    )
    return (
        -float(minimum_year_ratio),
        -float(stats["combined_2020_2022"]["cagr_to_strict_mdd"]),
        *policy_tuple,
    )


def _eligible_daily_anchors(
    engine: ExecutionEngine, *, start: str, end: str, hold_days: int
) -> np.ndarray:
    dates = pd.Series(engine.dates)
    mask = (
        (dates >= pd.Timestamp(start))
        & (dates < pd.Timestamp(end) - pd.Timedelta(days=int(hold_days), minutes=5))
        & dates.dt.hour.eq(12)
        & dates.dt.minute.eq(5)
    )
    return np.flatnonzero(mask.to_numpy(bool))


def random_same_count_control(
    engine: ExecutionEngine,
    policy: Policy,
    candidate: list[Trade],
    cfg: Config,
) -> dict[str, Any]:
    if not candidate or cfg.random_controls <= 0:
        return {"samples": 0, "p_value_ratio": None, "positive_fraction": None}
    anchors = _eligible_daily_anchors(
        engine,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        hold_days=policy.hold_days,
    )
    sides = np.asarray([trade.side for trade in candidate], dtype=np.int8)
    observed = strict_equity_stats(
        candidate,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
        cfg=execution_config(cfg),
    )["cagr_to_strict_mdd"]
    rng = np.random.default_rng(cfg.random_seed)
    ratios: list[float] = []
    positives = 0
    hold = int(policy.hold_days * 288)
    minimum_gap = hold + 1
    for _ in range(cfg.random_controls):
        chosen: list[int] = []
        for raw in rng.permutation(anchors):
            value = int(raw)
            if all(abs(value - prior) >= minimum_gap for prior in chosen):
                chosen.append(value)
                if len(chosen) == len(sides):
                    break
        if len(chosen) != len(sides):
            continue
        chosen.sort()
        shuffled_sides = rng.permutation(sides)
        trades: list[Trade] = []
        for signal, side in zip(chosen, shuffled_sides):
            trade = engine.trade_at(signal, int(side), hold, 10_000, 10_000)
            if trade is not None and engine.dates.iloc[trade.exit_position] < pd.Timestamp(SELECTION_END):
                trades.append(trade)
        if len(trades) != len(candidate):
            continue
        stats = strict_equity_stats(
            trades,
            start=WINDOWS["combined_2020_2022"][0],
            end=WINDOWS["combined_2020_2022"][1],
            cfg=execution_config(cfg),
        )
        ratios.append(float(stats["cagr_to_strict_mdd"]))
        positives += int(stats["absolute_return_pct"] > 0.0)
    values = np.asarray(ratios, dtype=float)
    return {
        "samples": int(len(values)),
        "p_value_ratio": float((1 + np.sum(values >= observed)) / (1 + len(values))) if len(values) else None,
        "positive_fraction": float(positives / len(values)) if len(values) else None,
        "ratio_q95": float(np.quantile(values, 0.95)) if len(values) else None,
        "observed_ratio": float(observed),
        "same_trade_count": int(len(candidate)),
        "same_side_counts": {
            "long": int(np.sum(sides > 0)),
            "short": int(np.sum(sides < 0)),
        },
    }


def write_docs(payload: dict[str, Any], path: str) -> None:
    selected = payload["selected_diagnostic"]
    stats = selected["stats"]
    random_control = payload["random_same_side_same_count_control"]
    lines = [
        "# Wikimedia Attention-Divergence — Selection Result",
        "",
        f"- Status: **{payload['decision']}**",
        f"- Policies opened: {payload['protocol']['policies_opened']} (preregistered only)",
        "- Data opened: Wikimedia + BTC/funding through 2022-12-31 only; 2023 and 2024+ remain sealed.",
        f"- Diagnostic policy: `{json.dumps(selected['policy'], sort_keys=True)}`",
        "",
        "## Diagnostic policy statistics",
        "",
        "| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Long/Short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("fit_2020", "fit_2021", "selection_2022", "combined_2020_2022"):
        row = stats[name]
        lines.append(
            f"| {name} | {row['absolute_return_pct']:.4f}% | {row['cagr_pct']:.4f}% | "
            f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | "
            f"{row['trades']} | {row['longs']}/{row['shorts']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision_reason"],
            "",
            f"Same-count/same-side random control: {random_control['samples']} samples, "
            f"ratio p={random_control['p_value_ratio']:.4f}, random-positive fraction="
            f"{random_control['positive_fraction']:.4f}.",
            "The random control is a selection diagnostic; the preregistered Bonferroni weekly "
            "block-bootstrap gate belongs to the still-sealed 2023 holdout phase.",
            "",
            "Historical Wikimedia snapshots do not prove point-in-time publication; even a future "
            "passing result would require retrieval-timestamped forward shadow evidence.",
            "",
            "A passing selection policy is not an alpha yet. It may open exactly one frozen 2023 holdout; "
            "2024+ remains sealed until the holdout gates pass.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def run(cfg: Config) -> dict[str, Any]:
    market, funding, attention, source_record = load_selection_sources(cfg)
    features = build_daily_features(attention, market)
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    trials = [evaluate_policy(engine, features, policy, cfg) for policy in policy_grid()]
    ordered = sorted(trials, key=rank_key)
    passing = [trial for trial in ordered if trial["passes_selection"]]
    selected = passing[0] if passing else ordered[0]
    policy = Policy(**selected["policy"])
    selected_events = policy_events(features, policy)
    selected_schedule = build_schedule(
        engine,
        selected_events,
        policy,
        start=WINDOWS["combined_2020_2022"][0],
        end=WINDOWS["combined_2020_2022"][1],
    )
    random_control = random_same_count_control(engine, policy, selected_schedule, cfg)
    decision = "selection_passed_holdout_still_sealed" if passing else "rejected_before_holdout"
    reason = (
        "At least one preregistered policy passed every 2020-2022 selection gate; the highest frozen rank may proceed to a single 2023 holdout."
        if passing
        else "No preregistered policy passed every 2020-2022 selection gate. The 2023 holdout and all 2024+ data remain unopened."
    )
    payload_core: dict[str, Any] = {
        "protocol_version": "wikimedia_attention_divergence_selection_v1",
        "decision": decision,
        "decision_reason": reason,
        "protocol": {
            "outcomes_opened": True,
            "policies_opened": len(trials),
            "policy_family_matches_preregistration": [trial["policy"] for trial in trials]
            == [asdict(policy) for policy in policy_grid()],
            "selection_cutoff_exclusive": SELECTION_END,
            "holdout_2023_opened": False,
            "future_2024_plus_opened": False,
            "historical_attention_is_point_in_time": False,
            "promotion_requires_forward_shadow": True,
        },
        "source_record": source_record,
        "feature_contract": {
            "attention_baseline": "strictly prior 90 days, minimum 45, median/MAD",
            "pageview_availability_anchor": "D+2 12:05 UTC",
            "execution": "next 5m open",
            "missing": "fail closed",
        },
        "selected_diagnostic": selected,
        "passing_policies": int(len(passing)),
        "all_trials": ordered,
        "random_same_side_same_count_control": random_control,
        "multiple_testing_contract": {
            "selection_random_control_role": "diagnostic_only_as_preregistered_controls",
            "familywise_gate_phase": "sealed_2023_holdout",
            "familywise_hypotheses": len(policy_grid()),
            "holdout_familywise_method": "Bonferroni weekly block bootstrap",
        },
    }
    payload = {
        **payload_core,
        "result_hash": canonical_hash(payload_core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    policy_output = Path(cfg.policy_output)
    if passing:
        frozen_core = {
            "protocol_version": "wikimedia_attention_divergence_frozen_policy_v1",
            "selection_passed": True,
            "holdout_2023_opened": False,
            "future_2024_plus_opened": False,
            "policy": selected["policy"],
            "selection_result": str(output),
            "selection_result_sha256": sha256_file(output),
            "selection_result_hash": payload["result_hash"],
            "source_prefix_hashes": {
                key: value.get("prefix_hash", value.get("manifest_hash"))
                for key, value in source_record.items()
            },
            "schedule_hashes": selected["schedule_hashes"],
            "selection_stats_hash": canonical_hash(selected["stats"]),
        }
        frozen = {
            **frozen_core,
            "manifest_hash": canonical_hash(frozen_core),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        policy_output.parent.mkdir(parents=True, exist_ok=True)
        policy_output.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    elif policy_output.exists():
        raise RuntimeError("stale frozen policy exists despite failed selection")
    write_docs(payload, cfg.docs_output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name in (
        "input_csv",
        "funding_csv",
        "attention_csv",
        "source_manifest",
        "market_prefix_manifest",
        "preregistration",
        "output",
        "policy_output",
        "docs_output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", default=getattr(defaults, name))
    parser.add_argument("--random-controls", type=int, default=defaults.random_controls)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    selected = payload["selected_diagnostic"]
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "passing_policies": payload["passing_policies"],
                "selected_policy": selected["policy"],
                "stats": selected["stats"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
