"""Freeze and evaluate a causal stablecoin-supply breadth absorption alpha.

Selection is physically truncated before 2024.  A fixed basket of chain-level
stablecoin supplies is used instead of revision-prone composite assets.  Signals
are allowed only when every component was completed one to three days after its
observation day; fills occur at the next 5-minute open after a completed signal
bar.  A committed manifest must reproduce before 2024+ can be opened.
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

from training.download_coinmetrics_stablecoin_supply_daily import ASSETS
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
FIT_START = "2021-06-01"
FIT_END = "2023-01-01"
WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2021h2": (FIT_START, "2022-01-01"),
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
FEATURES = (
    "supply_growth_7d_z",
    "usdt_share_change_7d_z",
    "breadth_7d_z",
    "supply_growth_30d_z",
    "usdt_share_change_30d_z",
    "breadth_30d_z",
    "supply_accel_7_30_z",
)
RULES = (
    "direct",
    "inverse",
    "confirm",
    "absorb",
    "price_momentum",
    "price_reversal",
    "positive_long",
    "negative_short",
)
TAILS = (0.10, 0.20, 0.30)
HOLD_DAYS = (1, 3, 7, 14)
EXPECTED_POLICY = {
    "feature": "breadth_7d_z",
    "tail": 0.30,
    "rule": "absorb",
    "hold_days": 7,
}
ACCOUNTING_VERSION = "stablecoin_supply_breadth_execution_v1"
HISTORICAL_VINTAGE_VERIFIED = False


@dataclass(frozen=True)
class Config:
    input_csv: str
    funding_csv: str
    stablecoin_csv: str
    output: str
    manifest_output: str
    docs_output: str = ""
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0012
    open_oos: bool = False
    random_control_count: int = 2_000
    random_seed: int = 20260716


def _read_before(path: str, date_column: str, cutoff: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="infer", chunksize=100_000):
        parsed = pd.to_datetime(
            chunk[date_column], utc=True, errors="raise", format="mixed"
        ).dt.tz_convert(None)
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


def rolling_z(values: pd.Series, window: int = 180, minimum: int = 90) -> pd.Series:
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def fit_reference_mask(frame: pd.DataFrame) -> np.ndarray:
    anchor = pd.to_datetime(frame["anchor_date"])
    return ((anchor >= pd.Timestamp(FIT_START)) & (anchor < pd.Timestamp(FIT_END))).to_numpy(bool)


def execution_contract(cfg: Config) -> dict[str, Any]:
    return {
        "accounting_version": ACCOUNTING_VERSION,
        "leverage": cfg.leverage,
        "fee_rate": cfg.fee_rate,
        "slippage_rate": cfg.slippage_rate,
        "stress_cost_rate": cfg.stress_cost_rate,
        "entry_delay_bars": 1,
        "take_bps": 10_000,
        "stop_bps": 10_000,
        "realized_funding": True,
        "strict_mdd": "intratrade_favorable_before_adverse_high_water",
        "full_calendar_cagr": True,
    }


def vintage_contract() -> dict[str, Any]:
    return {
        "provider_series": "Coin Metrics reviewed latest-snapshot SplyCur",
        "point_in_time_value_vintage_verified": HISTORICAL_VINTAGE_VERIFIED,
        "completion_time_is_not_value_vintage": True,
        "promotion_eligible": False,
        "required_repair": (
            "reconstruct immutable chain supplies at historical blocks or accumulate a forward-only "
            "versioned snapshot archive before promotion"
        ),
    }


def validate_frozen_manifest(manifest: dict[str, Any], cfg: Config) -> str:
    expected_hash = manifest.get("manifest_hash")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if json_hash(core) != expected_hash:
        raise RuntimeError("selection manifest hash mismatch")
    if manifest.get("fixed_asset_universe") != list(ASSETS):
        raise RuntimeError("stablecoin fixed universe differs from the frozen manifest")
    if manifest.get("execution_contract") != execution_contract(cfg):
        raise RuntimeError("OOS execution economics differ from the frozen manifest")
    if manifest.get("vintage_contract") != vintage_contract():
        raise RuntimeError("stablecoin vintage contract differs from the frozen manifest")
    return str(expected_hash)


def validate_selection_replay(
    manifest: dict[str, Any],
    *,
    source_prefix_hashes: dict[str, str],
    schedule_hashes: dict[str, str],
    stats: dict[str, dict[str, Any]],
) -> None:
    if source_prefix_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefix changed before OOS replay")
    if schedule_hashes != manifest["schedule_hashes"]:
        raise RuntimeError("pre-2024 schedule changed before OOS replay")
    if json_hash(stats) != manifest["selection_stats_hash"]:
        raise RuntimeError("pre-2024 performance accounting changed before OOS replay")


def load_stablecoin(path: str, *, cutoff: str) -> pd.DataFrame:
    frame = _read_before(path, "observation_date", cutoff)
    required = {"asset", "observation_date", "available_at", "supply"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"stablecoin source missing columns: {sorted(missing)}")
    frame["asset"] = frame["asset"].astype(str).str.lower()
    observed_assets = set(frame["asset"].unique())
    if observed_assets != set(ASSETS):
        raise ValueError(
            f"stablecoin fixed universe mismatch: expected={sorted(ASSETS)} observed={sorted(observed_assets)}"
        )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="raise", format="mixed"
    ).dt.tz_convert(None)
    frame["supply"] = pd.to_numeric(frame["supply"], errors="raise")
    if not np.isfinite(frame["supply"]).all() or (frame["supply"] <= 0.0).any():
        raise ValueError("stablecoin source contains a non-positive or non-finite supply")
    if (frame["available_at"] < frame["observation_date"] + pd.Timedelta(days=1)).any():
        raise RuntimeError("stablecoin day was available before the UTC day completed")
    frame = frame.sort_values(["observation_date", "asset"]).drop_duplicates(
        ["observation_date", "asset"], keep="last"
    )
    counts = frame.groupby("observation_date")["asset"].nunique()
    if not counts.eq(len(ASSETS)).all():
        bad = counts[counts != len(ASSETS)].index.min()
        raise RuntimeError(f"stablecoin source has an incomplete fixed-basket day: {bad}")
    expected_days = pd.date_range(
        frame["observation_date"].min(), frame["observation_date"].max(), freq="1D"
    )
    if not frame["observation_date"].drop_duplicates().reset_index(drop=True).equals(
        pd.Series(expected_days)
    ):
        raise RuntimeError("stablecoin source is not a complete daily grid")
    return frame.reset_index(drop=True)


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


def build_daily_features(source: pd.DataFrame, engine: ExecutionEngine) -> pd.DataFrame:
    supply = source.pivot(index="observation_date", columns="asset", values="supply")[list(ASSETS)]
    available = source.pivot(
        index="observation_date", columns="asset", values="available_at"
    )[list(ASSETS)]
    lag_days = available.sub(pd.Series(available.index, index=available.index), axis=0)
    lag_days = lag_days.apply(lambda column: column.dt.total_seconds() / 86_400.0)
    timely = supply.notna().all(axis=1) & lag_days.ge(1.0).all(axis=1) & lag_days.le(3.0).all(axis=1)

    frame = pd.DataFrame(index=supply.index)
    frame["available_at"] = available.max(axis=1)
    frame["valid_today"] = timely
    total = supply.sum(axis=1)
    usdt = supply[["usdt_eth", "usdt_trx", "usdt_omni"]].sum(axis=1)
    log_supply = np.log(supply)
    for horizon in (7, 30):
        endpoints_timely = timely & timely.shift(horizon, fill_value=False)
        frame[f"supply_growth_{horizon}d"] = np.log(total).diff(horizon).where(endpoints_timely)
        frame[f"usdt_share_change_{horizon}d"] = (usdt / total).diff(horizon).where(
            endpoints_timely
        )
        breadth = (log_supply.diff(horizon) > 0.0).mean(axis=1) - 0.5
        frame[f"breadth_{horizon}d"] = breadth.where(endpoints_timely)
    frame["supply_accel_7_30"] = (
        frame["supply_growth_7d"] - frame["supply_growth_30d"] * 7.0 / 30.0
    )
    raw_features = [
        "supply_growth_7d",
        "usdt_share_change_7d",
        "breadth_7d",
        "supply_growth_30d",
        "usdt_share_change_30d",
        "breadth_30d",
        "supply_accel_7_30",
    ]
    for feature in raw_features:
        frame[f"{feature}_z"] = rolling_z(frame[feature])

    frame = frame.reset_index()
    market_dates = engine.dates.to_numpy(dtype="datetime64[ns]")
    anchors = np.searchsorted(
        market_dates, frame["available_at"].to_numpy(dtype="datetime64[ns]"), side="left"
    )
    valid_anchor = anchors < len(market_dates)
    frame = frame.loc[valid_anchor].copy().reset_index(drop=True)
    frame["anchor"] = anchors[valid_anchor].astype(np.int64)
    frame["anchor_date"] = engine.dates.iloc[frame["anchor"].to_numpy()].to_numpy()
    if (frame["anchor_date"] < frame["available_at"]).any():
        raise RuntimeError("stablecoin observation mapped before provider completion")
    close = pd.to_numeric(engine.market["close"], errors="raise").to_numpy(float)
    anchor = frame["anchor"].to_numpy(int)
    previous = anchor - 24 * 12
    values = np.full(len(frame), np.nan)
    usable = previous >= 0
    values[usable] = np.log(close[anchor[usable]] / close[previous[usable]])
    frame["price_ret_24h"] = values
    return frame.replace([np.inf, -np.inf], np.nan)


def event_onset(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    return active & ~np.r_[False, active[:-1]]


def signed_event_masks(
    frame: pd.DataFrame, *, feature: str, lower_threshold: float, upper_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
    positive = event_onset(np.isfinite(values) & (values >= upper_threshold))
    negative = event_onset(np.isfinite(values) & (values <= lower_threshold))
    return positive, negative


def policy_masks(
    frame: pd.DataFrame,
    *,
    feature: str,
    lower_threshold: float,
    upper_threshold: float,
    rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    positive, negative = signed_event_masks(
        frame,
        feature=feature,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )
    price = frame["price_ret_24h"].to_numpy(float)
    finite = np.isfinite(price)
    zero = np.zeros(len(frame), dtype=bool)
    if rule == "direct":
        return positive, negative
    if rule == "inverse":
        return negative, positive
    if rule == "confirm":
        return positive & finite & (price > 0.0), negative & finite & (price < 0.0)
    if rule == "absorb":
        return positive & finite & (price < 0.0), negative & finite & (price > 0.0)
    if rule == "price_momentum":
        event = positive | negative
        return event & finite & (price > 0.0), event & finite & (price < 0.0)
    if rule == "price_reversal":
        event = positive | negative
        return event & finite & (price < 0.0), event & finite & (price > 0.0)
    if rule == "positive_long":
        return positive, zero
    if rule == "negative_short":
        return zero, negative
    raise ValueError(f"unknown rule: {rule}")


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
        feature=policy["feature"],
        lower_threshold=float(policy["lower_threshold"]),
        upper_threshold=float(policy["upper_threshold"]),
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
    schedules: dict[str, list[Trade]], cfg: Config, *, cost_rate: float | None = None
) -> dict[str, dict[str, Any]]:
    exec_cfg = execution_config(cfg)
    result: dict[str, dict[str, Any]] = {}
    for name, trades in schedules.items():
        stats = equity_stats(
            trades, start=WINDOWS[name][0], end=WINDOWS[name][1], cfg=exec_cfg, cost_rate=cost_rate
        )
        stats["schedule_hash"] = _schedule_hash(trades)
        result[name] = stats
    return result


def net_trade_returns(
    trades: list[Trade], cfg: Config, *, cost_rate: float | None = None
) -> list[float]:
    cost = cfg.fee_rate + cfg.slippage_rate if cost_rate is None else cost_rate
    edge = 1.0 - cfg.leverage * cost
    return [edge * trade.price_factor * trade.funding_factor * edge - 1.0 for trade in trades]


def search_selection(
    engine: ExecutionEngine, frame: pd.DataFrame, cfg: Config
) -> list[dict[str, Any]]:
    fit = fit_reference_mask(frame)
    rows: list[dict[str, Any]] = []
    selection_windows = (
        "fit_2021h2",
        "fit_2022",
        "select_2023_h1",
        "select_2023_h2",
        "select_2023",
    )
    for feature in FEATURES:
        values = frame[feature].to_numpy(float)
        reference = values[fit & np.isfinite(values)]
        if len(reference) < 250:
            raise RuntimeError(f"insufficient threshold history for {feature}: {len(reference)}")
        for tail in TAILS:
            lower, upper = np.quantile(reference, [tail, 1.0 - tail])
            for rule in RULES:
                for hold_days in HOLD_DAYS:
                    policy = {
                        "feature": feature,
                        "tail": tail,
                        "lower_threshold": float(lower),
                        "upper_threshold": float(upper),
                        "rule": rule,
                        "hold_days": hold_days,
                    }
                    schedules = policy_schedules(engine, frame, policy, windows=selection_windows)
                    stats = schedule_stats(schedules, cfg)
                    blocks = [stats[name] for name in selection_windows[:4]]
                    eligible = min(row["trades"] for row in blocks) >= 4 and all(
                        row["absolute_return_pct"] > 0.0 for row in blocks
                    )
                    rows.append(
                        {
                            **policy,
                            "eligible": eligible,
                            "minimum_block_ratio": (
                                min(row["cagr_to_strict_mdd"] for row in blocks)
                                if eligible
                                else -999.0
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
    engine: ExecutionEngine, frame: pd.DataFrame, policy: dict[str, Any], cfg: Config
) -> dict[str, Any]:
    positive, negative = signed_event_masks(
        frame,
        feature=policy["feature"],
        lower_threshold=float(policy["lower_threshold"]),
        upper_threshold=float(policy["upper_threshold"]),
    )
    price = frame["price_ret_24h"].to_numpy(float)
    finite = np.isfinite(price)
    windows = ("fit_2021h2", "fit_2022", "select_2023_h1", "select_2023_h2", "select_2023")

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

    mappings: dict[str, Any] = {}
    for rule in RULES:
        long_mask, short_mask = policy_masks(
            frame,
            feature=policy["feature"],
            lower_threshold=float(policy["lower_threshold"]),
            upper_threshold=float(policy["upper_threshold"]),
            rule=rule,
        )
        mappings[rule] = stats_from_masks(long_mask, short_mask)
    price_only_reversal = stats_from_masks(finite & (price < 0.0), finite & (price > 0.0))
    shifts: dict[str, Any] = {}
    for lag in (-28, -14, -7, 7, 14, 28):
        shifted_positive = shifted_masks(positive, lag)
        shifted_negative = shifted_masks(negative, lag)
        shifts[str(lag)] = stats_from_masks(
            shifted_positive & finite & (price < 0.0),
            shifted_negative & finite & (price > 0.0),
        )

    rng = np.random.default_rng(cfg.random_seed)
    years = frame["observation_date"].dt.year.to_numpy()
    values = frame[policy["feature"]].to_numpy(float)
    pool_ok = finite & np.isfinite(values)
    blocks = windows[:4]
    random_min_ratios: list[float] = []
    random_sum_returns: list[float] = []
    random_all_positive: list[bool] = []
    for _ in range(cfg.random_control_count):
        random_positive = np.zeros(len(frame), dtype=bool)
        random_negative = np.zeros(len(frame), dtype=bool)
        for year in (2021, 2022, 2023):
            pool = np.flatnonzero((years == year) & pool_ok)
            positive_count = int((positive & (years == year)).sum())
            negative_count = int((negative & (years == year)).sum())
            if positive_count + negative_count:
                chosen = rng.choice(pool, size=positive_count + negative_count, replace=False)
                random_positive[chosen[:positive_count]] = True
                random_negative[chosen[positive_count:]] = True
        stats = stats_from_masks(
            random_positive & finite & (price < 0.0),
            random_negative & finite & (price > 0.0),
        )
        random_min_ratios.append(min(stats[name]["cagr_to_strict_mdd"] for name in blocks))
        random_sum_returns.append(sum(stats[name]["absolute_return_pct"] for name in blocks))
        random_all_positive.append(all(stats[name]["absolute_return_pct"] > 0.0 for name in blocks))

    candidate_schedules = policy_schedules(engine, frame, policy, windows=blocks)
    candidate_stats = schedule_stats(candidate_schedules, cfg)
    candidate_min = min(candidate_stats[name]["cagr_to_strict_mdd"] for name in blocks)
    candidate_sum = sum(candidate_stats[name]["absolute_return_pct"] for name in blocks)
    random_min = np.asarray(random_min_ratios, dtype=float)
    random_sum = np.asarray(random_sum_returns, dtype=float)
    random_positive_flag = np.asarray(random_all_positive, dtype=bool)
    return {
        "mapping_ablation": mappings,
        "price_only_reversal": price_only_reversal,
        "event_day_shifts": shifts,
        "double_cost": schedule_stats(candidate_schedules, cfg, cost_rate=cfg.stress_cost_rate),
        "random_clock": {
            "count": cfg.random_control_count,
            "seed": cfg.random_seed,
            "all_blocks_positive_fraction": float(np.mean(random_positive_flag)),
            "candidate_minimum_block_ratio": float(candidate_min),
            "candidate_sum_block_return_pct": float(candidate_sum),
            "empirical_p_minimum_block_ratio": float(
                (1 + np.sum(random_min >= candidate_min)) / (1 + len(random_min))
            ),
            "empirical_p_positive_and_sum_return": float(
                (1 + np.sum(random_positive_flag & (random_sum >= candidate_sum)))
                / (1 + len(random_sum))
            ),
            "minimum_block_ratio_quantiles": {
                str(q): float(np.quantile(random_min, q)) for q in (0.5, 0.9, 0.95, 0.99)
            },
            "sum_block_return_quantiles": {
                str(q): float(np.quantile(random_sum, q)) for q in (0.5, 0.9, 0.95, 0.99)
            },
        },
    }


def policy_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "feature",
            "tail",
            "lower_threshold",
            "upper_threshold",
            "rule",
            "hold_days",
        )
    }


def source_quality(source: pd.DataFrame) -> dict[str, Any]:
    lag = (
        source["available_at"] - source["observation_date"]
    ).dt.total_seconds() / 86_400.0
    by_asset: dict[str, Any] = {}
    for asset in ASSETS:
        values = lag[source["asset"] == asset]
        by_asset[asset] = {
            "rows": int(len(values)),
            "timely_1_to_3_days": int(values.between(1.0, 3.0).sum()),
            "maximum_lag_days": float(values.max()),
        }
    return {"fixed_assets": list(ASSETS), "by_asset": by_asset}


def render_selection_docs(payload: dict[str, Any]) -> str:
    policy = payload["selected_policy"]
    lines = [
        "# Stablecoin supply breadth absorption — pre-2024 selection",
        "",
        "## Frozen mechanism",
        "",
        "The feature counts how many members of a fixed chain-specific stablecoin basket increased supply over seven days. "
        "A prior-only 180-day z-score converts that breadth into a sparse event clock. A broad expansion after a completed "
        "BTC decline is interpreted as cash absorption and goes long; a broad contraction after a completed BTC rally is "
        "interpreted as fragile liquidity and goes short.",
        "",
        f"- feature: `{policy['feature']}`",
        f"- fitted tails: `{policy['tail']}`; lower `{policy['lower_threshold']:.12f}`, upper `{policy['upper_threshold']:.12f}`",
        f"- direction rule: `{policy['rule']}`",
        f"- hold: `{policy['hold_days']}` days, no overlap",
        f"- fixed basket: `{', '.join(ASSETS)}`; composite `usdt`/`usdc` excluded",
        "- availability: all component rows must complete 1–3 days after observation; first 5-minute bar at/after the latest completion is the signal bar; next open fills.",
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
            "- Direct, inverse, confirmation, unconditional price-momentum/reversal, and one-sided mappings are reported without altering the frozen choice.",
            "- An every-valid-day price-reversal control and ±7/14/28-day event-clock shifts are reported.",
            f"- `{random['count']}` year- and event-sign-matched random clocks: all-block-positive fraction "
            f"`{random['all_blocks_positive_fraction']:.4f}`; q99 minimum-block ratio "
            f"`{random['minimum_block_ratio_quantiles']['0.99']:.4f}`; empirical p(minimum-block ratio) "
            f"`{random['empirical_p_minimum_block_ratio']:.6f}`; empirical p(positive blocks and summed return) "
            f"`{random['empirical_p_positive_and_sum_return']:.6f}`.",
            "",
            "## Integrity boundary",
            "",
            f"- The exploratory family opened `{payload['tested_cells']}` pre-2024 cells. Random-clock p-values are diagnostics, not family-wise correction.",
            "- The trade-return p-value and random-clock p-values are descriptive post-selection diagnostics; none is selection-adjusted across the 672-cell family.",
            "- Coin Metrics `SplyCur` can be revised. Composite assets and rows completed more than three days late are excluded from event generation, but `AssetEODCompletionTime` is not a value-vintage archive. A committed prefix hash detects later changes and prevents reranking; it does not prove the latest snapshot equals the value published historically.",
            "- 2024+ BTC outcomes are globally research-seen in this repository. The freeze prevents reranking this new family after replay, but the future is not a pristine human holdout.",
            "- This candidate is therefore a non-promotable historical hypothesis. Promotion requires immutable block-height reconstruction or forward-only versioned snapshots, positive test/eval/holdout performance, doubled-cost survival, and actual trade/PnL orthogonality.",
            "",
            "Official sources:",
            "- https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/supply/current-supply",
            "- https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/availability/asseteodcompletiontime",
            "- https://gitbook-docs.coinmetrics.io/access-our-data/api",
            "",
        ]
    )
    return "\n".join(lines)


def run_selection(cfg: Config) -> dict[str, Any]:
    market, funding = load_market_and_funding(cfg, cutoff=SELECTION_END)
    stablecoin = load_stablecoin(cfg.stablecoin_csv, cutoff=SELECTION_END)
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    features = build_daily_features(stablecoin, engine)
    rows = search_selection(engine, features, cfg)
    selected = rows[0]
    for key, expected in EXPECTED_POLICY.items():
        if selected[key] != expected:
            raise RuntimeError(f"selection drift for {key}: {selected[key]!r} != {expected!r}")
    controls = selection_controls(engine, features, selected, cfg)
    identity = policy_identity(selected)
    selection_windows = (
        "fit_2021h2",
        "fit_2022",
        "select_2023_h1",
        "select_2023_h2",
        "select_2023",
    )
    schedules = policy_schedules(engine, features, identity, windows=selection_windows)
    manifest_core = {
        "phase": "pre_2024_selection",
        "selection_end": SELECTION_END,
        "fixed_asset_universe": list(ASSETS),
        "vintage_contract": vintage_contract(),
        "policy": identity,
        "policy_hash": json_hash(identity),
        "source_prefix_hashes": {
            "market": frame_hash(market),
            "funding": frame_hash(funding),
            "stablecoin": frame_hash(stablecoin),
            "features": frame_hash(features),
        },
        "schedule_hashes": {name: _schedule_hash(trades) for name, trades in schedules.items()},
        "execution_contract": execution_contract(cfg),
        "selection_stats_hash": json_hash(selected["stats"]),
        "search_space": {
            "features": list(FEATURES),
            "tails": list(TAILS),
            "rules": list(RULES),
            "hold_days": list(HOLD_DAYS),
        },
        "tested_cells": len(rows),
    }
    manifest = {
        **manifest_core,
        "manifest_hash": json_hash(manifest_core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(cfg.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.manifest_output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    candidate_trades = [trade for name in selection_windows[:4] for trade in schedules[name]]
    payload = {
        "mode": "pre_2024_selection",
        "config": asdict(cfg),
        "protocol": {
            "threshold_fit": [FIT_START, FIT_END],
            "selection": [FIT_END, SELECTION_END],
            "future_opened": False,
            "availability": "all fixed-basket component completion lags in [1d,3d], then one completed 5m signal bar",
            "entry": "next 5m open",
            "realized_funding": True,
            "strict_mdd": "intratrade favorable-before-adverse high-water path",
            "full_calendar_cagr": True,
            "globally_research_seen_future": True,
            "composite_assets_excluded": ["usdt", "usdc"],
        },
        "promotion_eligibility": {
            "eligible": False,
            "blockers": [
                "historical SplyCur is a reviewed latest snapshot, not a point-in-time value vintage",
                "actual trade/PnL orthogonality is pending OOS performance survival",
            ],
        },
        "source_quality": source_quality(stablecoin),
        "tested_cells": len(rows),
        "eligible_cells": sum(row["eligible"] for row in rows),
        "selected_policy": selected,
        "top_10": rows[:10],
        "controls": controls,
        "post_selection_descriptive_trade_statistics": _trade_stats(
            net_trade_returns(candidate_trades, cfg)
        ),
        "manifest_hash": manifest["manifest_hash"],
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if cfg.docs_output:
        Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.docs_output).write_text(render_selection_docs(payload))
    return payload


def run_oos(cfg: Config) -> dict[str, Any]:
    manifest = json.loads(Path(cfg.manifest_output).read_text())
    expected_hash = validate_frozen_manifest(manifest, cfg)

    selection_market, selection_funding = load_market_and_funding(cfg, cutoff=SELECTION_END)
    selection_stablecoin = load_stablecoin(cfg.stablecoin_csv, cutoff=SELECTION_END)
    selection_engine = ExecutionEngine(
        selection_market, selection_funding, execution_config(cfg)
    )
    selection_features = build_daily_features(selection_stablecoin, selection_engine)
    prefix_hashes = {
        "market": frame_hash(selection_market),
        "funding": frame_hash(selection_funding),
        "stablecoin": frame_hash(selection_stablecoin),
        "features": frame_hash(selection_features),
    }
    replay = policy_schedules(
        selection_engine,
        selection_features,
        manifest["policy"],
        windows=tuple(manifest["schedule_hashes"]),
    )
    validate_selection_replay(
        manifest,
        source_prefix_hashes=prefix_hashes,
        schedule_hashes={name: _schedule_hash(trades) for name, trades in replay.items()},
        stats=schedule_stats(replay, cfg),
    )

    market, funding = load_market_and_funding(cfg, cutoff=FULL_CUTOFF)
    stablecoin = load_stablecoin(cfg.stablecoin_csv, cutoff=FULL_CUTOFF)
    coverage = stablecoin.groupby("asset")["observation_date"].max()
    if (coverage < pd.Timestamp("2026-05-31")).any():
        raise RuntimeError(f"OOS stablecoin source lacks frozen holdout coverage: {coverage.to_dict()}")
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    features = build_daily_features(stablecoin, engine)
    windows = tuple(WINDOWS)
    schedules = policy_schedules(engine, features, manifest["policy"], windows=windows)
    stats = schedule_stats(schedules, cfg)
    stress = schedule_stats(schedules, cfg, cost_rate=cfg.stress_cost_rate)
    significance = {
        name: _trade_stats(net_trade_returns(trades, cfg)) for name, trades in schedules.items()
    }
    raw_performance_pass = (
        stats["test_2024"]["absolute_return_pct"] > 0.0
        and stats["eval_2025"]["absolute_return_pct"] > 0.0
        and stats["holdout_2026h1"]["absolute_return_pct"] > 0.0
        and stats["oos_2024_2026h1"]["cagr_to_strict_mdd"] >= 3.0
        and stress["oos_2024_2026h1"]["absolute_return_pct"] > 0.0
    )
    payload = {
        "mode": "frozen_oos_replay",
        "config": asdict(cfg),
        "manifest_hash": expected_hash,
        "future_did_not_rerank": True,
        "globally_research_seen_future": True,
        "policy": manifest["policy"],
        "source_quality": source_quality(stablecoin),
        "stats": stats,
        "double_cost_stats": stress,
        "trade_statistics": significance,
        "raw_performance_pass": raw_performance_pass,
        "historical_vintage_verified": HISTORICAL_VINTAGE_VERIFIED,
        "orthogonality_pass": None,
        "performance_pass": False,
        "promotion_pass": False,
        "promotion_blockers": [
            "historical SplyCur point-in-time value vintage is not verified",
            "actual trade/PnL orthogonality has not been audited",
        ],
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--stablecoin-csv", required=True)
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
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "policy": result.get("policy", result.get("selected_policy", {})),
                "performance_pass": result.get("performance_pass"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
