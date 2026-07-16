"""Freeze and evaluate an EM-FX coherent-pressure BTC reversal alpha.

Selection is physically truncated before 2024. Daily closes from four unused
USD crosses are standardized using prior sessions only and collapsed into a
common-mode pressure score. Extreme coherent EM-FX moves mark BTC reversal
opportunities; the next 5-minute open is used after the full UTC FX day is known.
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

from training.export_emfx_daily_from_postgres import SYMBOLS
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
FIT_START = "2021-01-01"
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
MIN_OBSERVATIONS = {
    "USDAUD": 1_000,
    "USDCNY": 200,
    "USDHKD": 700,
    "USDINR": 600,
    "USDMXN": 1_000,
}
RISK_SYMBOLS = ("USDAUD", "USDCNY", "USDINR", "USDMXN")
FEATURES = tuple(
    feature
    for horizon in (1, 5)
    for feature in (
        f"em_stress_{horizon}d",
        f"em_median_{horizon}d",
        f"em_coherence_{horizon}d",
        f"em_coherent_pressure_{horizon}d",
        f"em_breadth_{horizon}d_z",
        f"em_dispersion_{horizon}d",
        f"asia_stress_{horizon}d",
        f"carry_stress_{horizon}d",
        f"asia_carry_gap_{horizon}d",
    )
) + ("hkd_band_z",)
RULES = (
    "direct",
    "risk_inverse",
    "risk_confirm",
    "risk_divergence",
    "price_momentum",
    "price_reversal",
    "riskoff_short",
    "riskon_long",
)
TAILS = (0.10, 0.20, 0.30)
HOLD_DAYS = (1, 3, 7)
EXPECTED_POLICY = {
    "feature": "em_coherent_pressure_1d",
    "tail": 0.20,
    "rule": "price_reversal",
    "hold_days": 7,
}
ACCOUNTING_VERSION = "emfx_coherent_pressure_execution_v1"
MIN_SELECTION_YEAR_RATIO = 2.0


@dataclass(frozen=True)
class Config:
    input_csv: str
    funding_csv: str
    emfx_csv: str
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
    allow_backfilled_emfx: bool = False


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


def rolling_z(values: pd.Series, window: int = 252, minimum: int = 126) -> pd.Series:
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


def source_vintage_contract(cfg: Config) -> dict[str, Any]:
    return {
        "allow_backfilled_emfx": cfg.allow_backfilled_emfx,
        "semantic_availability": "UTC day d plus five minutes, then one completed 5m bar",
        "database_snapshot_is_point_in_time": False,
        "promotion_requires_live_forward_validation": True,
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
    if manifest.get("fixed_symbol_universe") != list(SYMBOLS):
        raise RuntimeError("EM-FX fixed universe differs from the frozen manifest")
    if manifest.get("execution_contract") != execution_contract(cfg):
        raise RuntimeError("OOS execution economics differ from the frozen manifest")
    if manifest.get("source_vintage_contract") != source_vintage_contract(cfg):
        raise RuntimeError("OOS EM-FX source-vintage mode differs from the frozen manifest")
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


def load_emfx(path: str, *, cutoff: str, allow_backfilled: bool = False) -> pd.DataFrame:
    frame = _read_before(path, "observation_date", cutoff)
    required = {
        "symbol",
        "observation_date",
        "observations",
        "close",
        "last_ts",
        "max_updated_at",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"EM-FX source missing columns: {sorted(missing)}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    observed_symbols = set(frame["symbol"].unique())
    if observed_symbols != set(SYMBOLS):
        raise ValueError(
            f"EM-FX fixed universe mismatch: expected={sorted(SYMBOLS)} observed={sorted(observed_symbols)}"
        )
    for column in ("last_ts", "max_updated_at"):
        frame[column] = pd.to_datetime(
            frame[column], utc=True, errors="raise", format="mixed"
        ).dt.tz_convert(None)
    frame["observations"] = pd.to_numeric(frame["observations"], errors="raise").astype(int)
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if (frame["observations"] <= 0).any():
        raise ValueError("EM-FX observations must be positive")
    if not np.isfinite(frame["close"]).all() or (frame["close"] <= 0.0).any():
        raise ValueError("EM-FX close must be positive and finite")
    if (
        (frame["last_ts"] < frame["observation_date"])
        | (frame["last_ts"] >= frame["observation_date"] + pd.Timedelta(days=1))
    ).any():
        raise RuntimeError("EM-FX quote falls outside its labelled UTC day")
    duplicate = frame.duplicated(["observation_date", "symbol"], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, ["observation_date", "symbol"]].head().to_dict("records")
        raise RuntimeError(f"duplicate EM-FX UTC day/symbol rows: {sample}")
    semantic_available_at = frame["observation_date"] + pd.Timedelta(days=1, minutes=5)
    backfilled = frame["max_updated_at"] > semantic_available_at
    if backfilled.any() and not allow_backfilled:
        raise RuntimeError(
            "EM-FX database snapshot is backfilled rather than point-in-time; "
            "pass --allow-backfilled-emfx only for explicitly labelled research"
        )
    frame["backfilled_after_semantic_availability"] = backfilled
    return frame.sort_values(["observation_date", "symbol"]).reset_index(drop=True)


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
    close = source.pivot(index="observation_date", columns="symbol", values="close")[list(SYMBOLS)]
    counts = source.pivot(
        index="observation_date", columns="symbol", values="observations"
    )[list(SYMBOLS)]
    valid = close.notna().all(axis=1)
    for symbol, minimum in MIN_OBSERVATIONS.items():
        valid &= counts[symbol] >= minimum
    close = close.loc[valid].copy()
    if len(close) < 700:
        raise RuntimeError(f"insufficient complete EM-FX sessions: {len(close)}")
    log_close = np.log(close)
    frame = pd.DataFrame(index=close.index)
    frame["available_at"] = close.index + pd.Timedelta(days=1, minutes=5)
    for horizon in (1, 5):
        returns = log_close.diff(horizon)
        standardized = pd.DataFrame(
            {symbol: rolling_z(returns[symbol]) for symbol in SYMBOLS}, index=close.index
        )
        core = standardized[list(RISK_SYMBOLS)]
        complete = core.notna().all(axis=1)
        mean = core.mean(axis=1).where(complete)
        median = core.median(axis=1).where(complete)
        rms = np.sqrt(core.pow(2).mean(axis=1)).where(complete).replace(0.0, np.nan)
        coherence = (mean / rms).where(complete)
        frame[f"em_stress_{horizon}d"] = mean
        frame[f"em_median_{horizon}d"] = median
        frame[f"em_coherence_{horizon}d"] = coherence
        frame[f"em_coherent_pressure_{horizon}d"] = mean * coherence.abs()
        raw_breadth = (returns[list(RISK_SYMBOLS)] > 0.0).mean(axis=1).where(
            returns[list(RISK_SYMBOLS)].notna().all(axis=1)
        ) - 0.5
        frame[f"em_breadth_{horizon}d_z"] = rolling_z(raw_breadth)
        frame[f"em_dispersion_{horizon}d"] = core.std(axis=1, ddof=0).where(complete)
        frame[f"asia_stress_{horizon}d"] = (
            (standardized["USDCNY"] + standardized["USDINR"]) / 2.0
        ).where(complete)
        frame[f"carry_stress_{horizon}d"] = (
            (standardized["USDAUD"] + standardized["USDMXN"]) / 2.0
        ).where(complete)
        frame[f"asia_carry_gap_{horizon}d"] = (
            frame[f"asia_stress_{horizon}d"] - frame[f"carry_stress_{horizon}d"]
        )
    frame["hkd_band_z"] = rolling_z(close["USDHKD"])
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
        raise RuntimeError("EM-FX observation mapped before the full UTC day was available")
    market_close = pd.to_numeric(engine.market["close"], errors="raise").to_numpy(float)
    anchor = frame["anchor"].to_numpy(int)
    previous = anchor - 24 * 12
    price_return = np.full(len(frame), np.nan)
    usable = previous >= 0
    price_return[usable] = np.log(market_close[anchor[usable]] / market_close[previous[usable]])
    frame["price_ret_24h"] = price_return
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
    if rule == "risk_inverse":
        return negative, positive
    if rule == "risk_confirm":
        return negative & finite & (price > 0.0), positive & finite & (price < 0.0)
    if rule == "risk_divergence":
        return negative & finite & (price < 0.0), positive & finite & (price > 0.0)
    if rule == "price_momentum":
        event = positive | negative
        return event & finite & (price > 0.0), event & finite & (price < 0.0)
    if rule == "price_reversal":
        event = positive | negative
        return event & finite & (price < 0.0), event & finite & (price > 0.0)
    if rule == "riskoff_short":
        return zero, positive
    if rule == "riskon_long":
        return negative, zero
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


def selection_eligible(stats: dict[str, dict[str, Any]]) -> bool:
    """Require positive subperiods plus a usable full 2023 risk-adjusted result."""
    blocks = [
        stats["fit_2021"],
        stats["fit_2022"],
        stats["select_2023_h1"],
        stats["select_2023_h2"],
    ]
    return (
        min(row["trades"] for row in blocks) >= 5
        and all(row["absolute_return_pct"] > 0.0 for row in blocks)
        and stats["select_2023"]["cagr_to_strict_mdd"] >= MIN_SELECTION_YEAR_RATIO
    )


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
        "fit_2021",
        "fit_2022",
        "select_2023_h1",
        "select_2023_h2",
        "select_2023",
    )
    for feature in FEATURES:
        values = frame[feature].to_numpy(float)
        reference = values[fit & np.isfinite(values)]
        if len(reference) < 300:
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
                    eligible = selection_eligible(stats)
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
    event = positive | negative
    price = frame["price_ret_24h"].to_numpy(float)
    finite = np.isfinite(price)
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
    for lag in (-14, -7, -3, -1, 1, 3, 7, 14):
        shifted = shifted_masks(event, lag)
        shifts[str(lag)] = stats_from_masks(
            shifted & finite & (price < 0.0), shifted & finite & (price > 0.0)
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
        random_event = np.zeros(len(frame), dtype=bool)
        for year in (2021, 2022, 2023):
            pool = np.flatnonzero((years == year) & pool_ok)
            count = int((event & (years == year)).sum())
            if count:
                random_event[rng.choice(pool, size=count, replace=False)] = True
        stats = stats_from_masks(
            random_event & finite & (price < 0.0),
            random_event & finite & (price > 0.0),
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
    random_positive = np.asarray(random_all_positive, dtype=bool)
    return {
        "mapping_ablation": mappings,
        "price_only_reversal": price_only_reversal,
        "event_day_shifts": shifts,
        "double_cost": schedule_stats(candidate_schedules, cfg, cost_rate=cfg.stress_cost_rate),
        "random_clock": {
            "count": cfg.random_control_count,
            "seed": cfg.random_seed,
            "all_blocks_positive_fraction": float(np.mean(random_positive)),
            "candidate_minimum_block_ratio": float(candidate_min),
            "candidate_sum_block_return_pct": float(candidate_sum),
            "empirical_p_minimum_block_ratio": float(
                (1 + np.sum(random_min >= candidate_min)) / (1 + len(random_min))
            ),
            "empirical_p_positive_and_sum_return": float(
                (1 + np.sum(random_positive & (random_sum >= candidate_sum)))
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


def complete_session_mask(source: pd.DataFrame) -> pd.Series:
    counts = source.pivot(
        index="observation_date", columns="symbol", values="observations"
    )[list(SYMBOLS)]
    complete = counts.notna().all(axis=1)
    for symbol, minimum in MIN_OBSERVATIONS.items():
        complete &= counts[symbol] >= minimum
    return complete


def source_quality(source: pd.DataFrame) -> dict[str, Any]:
    complete = complete_session_mask(source)
    complete_dates = complete.index[complete]
    return {
        "fixed_symbols": list(SYMBOLS),
        "minimum_observations": MIN_OBSERVATIONS,
        "raw_rows": int(len(source)),
        "complete_sessions": int(len(complete_dates)),
        "complete_session_range": [
            str(complete_dates.min()),
            str(complete_dates.max()),
        ],
        "complete_sessions_by_year": {
            str(year): int((complete_dates.year == year).sum())
            for year in sorted(set(complete_dates.year))
        },
        "max_database_updated_at": str(source["max_updated_at"].max()),
        "rows_backfilled_after_semantic_availability": int(
            source["backfilled_after_semantic_availability"].sum()
        ),
        "database_snapshot_is_point_in_time": not bool(
            source["backfilled_after_semantic_availability"].any()
        ),
    }


def render_selection_docs(payload: dict[str, Any]) -> str:
    policy = payload["selected_policy"]
    lines = [
        "# EM-FX coherent pressure reversal — pre-2024 selection",
        "",
        "## Frozen mechanism",
        "",
        "AUD, CNY, INR and MXN one-session USD returns are standardized with prior-only 252-session histories. "
        "Their equal-weight mean is multiplied by absolute common-mode coherence, suppressing idiosyncratic FX moves. "
        "When this score enters either fitted tail, the policy fades the already-completed BTC 24-hour move for seven days.",
        "",
        f"- feature: `{policy['feature']}`",
        f"- fitted tails: `{policy['tail']}`; lower `{policy['lower_threshold']:.12f}`, upper `{policy['upper_threshold']:.12f}`",
        f"- direction rule: `{policy['rule']}`",
        f"- hold: `{policy['hold_days']}` days, no overlap",
        "- fixed FX panel: `USDAUD, USDCNY, USDHKD, USDINR, USDMXN`; HKD is tested separately and is not part of the chosen common factor.",
        "- semantic availability: complete UTC day plus five minutes; completed 5-minute signal bar; next-open fill.",
        "- source vintage: the local PostgreSQL rows were historically backfilled, not captured in a point-in-time database snapshot. The values are fixed-panel timestamped quotes, but promotion requires live forward validation.",
        "- execution: 0.5x, 6 bp per side, realized funding, full-calendar CAGR, intratrade strict MDD.",
        "",
        "## Selection evidence",
        "",
        f"The `{payload['tested_cells']}`-cell family first requires at least five trades and positive absolute return in each 2021/2022/2023H1/2023H2 block, plus full-2023 CAGR/strict-MDD of at least `{MIN_SELECTION_YEAR_RATIO:.1f}`. Among eligible cells, selection maximizes the minimum subperiod ratio before the full-2023 tie-break.",
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
            "- Every-valid-FX-day BTC reversal loses in every core block.",
            "- Direct/risk mappings and ±1/3/7/14 valid-session clock shifts are reported without altering the frozen choice.",
            f"- `{random['count']}` year- and count-matched random clocks: all-block-positive fraction "
            f"`{random['all_blocks_positive_fraction']:.4f}`; q99 minimum-block ratio "
            f"`{random['minimum_block_ratio_quantiles']['0.99']:.4f}`; empirical p(minimum-block ratio) "
            f"`{random['empirical_p_minimum_block_ratio']:.6f}`; empirical p(positive blocks and summed return) "
            f"`{random['empirical_p_positive_and_sum_return']:.6f}`.",
            "",
            "## Integrity boundary",
            "",
            f"- The exploratory family opened `{payload['tested_cells']}` pre-2024 cells. Trade and random-clock p-values are descriptive post-selection diagnostics, not family-wise correction.",
            "- PostgreSQL ingestion timestamps are later backfills. Inputs are timestamped market quotes with a fixed symbol panel rather than a reconstructed composition, but this is explicitly not point-in-time source evidence. Source and feature prefixes are hash-frozen before future replay.",
            "- 2024+ BTC outcomes are globally research-seen in this repository; the manifest prevents reranking this family but does not create a pristine human holdout.",
            "- Promotion requires positive frozen test/eval/holdout performance, doubled-cost survival, actual trade/PnL orthogonality, and a live point-in-time forward window.",
            "",
        ]
    )
    return "\n".join(lines)


def run_selection(cfg: Config) -> dict[str, Any]:
    market, funding = load_market_and_funding(cfg, cutoff=SELECTION_END)
    emfx = load_emfx(
        cfg.emfx_csv, cutoff=SELECTION_END, allow_backfilled=cfg.allow_backfilled_emfx
    )
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    features = build_daily_features(emfx, engine)
    rows = search_selection(engine, features, cfg)
    selected = rows[0]
    for key, expected in EXPECTED_POLICY.items():
        if selected[key] != expected:
            raise RuntimeError(f"selection drift for {key}: {selected[key]!r} != {expected!r}")
    controls = selection_controls(engine, features, selected, cfg)
    identity = policy_identity(selected)
    selection_windows = (
        "fit_2021",
        "fit_2022",
        "select_2023_h1",
        "select_2023_h2",
        "select_2023",
    )
    schedules = policy_schedules(engine, features, identity, windows=selection_windows)
    manifest_core = {
        "phase": "pre_2024_selection",
        "selection_end": SELECTION_END,
        "fixed_symbol_universe": list(SYMBOLS),
        "minimum_daily_observations": MIN_OBSERVATIONS,
        "policy": identity,
        "policy_hash": json_hash(identity),
        "source_prefix_hashes": {
            "market": frame_hash(market),
            "funding": frame_hash(funding),
            "emfx": frame_hash(emfx),
            "features": frame_hash(features),
        },
        "schedule_hashes": {name: _schedule_hash(trades) for name, trades in schedules.items()},
        "execution_contract": execution_contract(cfg),
        "source_vintage_contract": source_vintage_contract(cfg),
        "selection_stats_hash": json_hash(selected["stats"]),
        "search_space": {
            "features": list(FEATURES),
            "tails": list(TAILS),
            "rules": list(RULES),
            "hold_days": list(HOLD_DAYS),
            "eligibility": {
                "minimum_trades_per_subperiod": 5,
                "all_subperiod_absolute_returns_positive": True,
                "minimum_select_2023_cagr_to_strict_mdd": MIN_SELECTION_YEAR_RATIO,
            },
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
            "availability": "complete UTC FX day plus five minutes, then one completed 5m signal bar",
            "entry": "next 5m open",
            "realized_funding": True,
            "strict_mdd": "intratrade favorable-before-adverse high-water path",
            "full_calendar_cagr": True,
            "globally_research_seen_future": True,
            "database_snapshot_is_point_in_time": False,
            "live_forward_validation_required": True,
        },
        "source_quality": source_quality(emfx),
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
    selection_emfx = load_emfx(
        cfg.emfx_csv, cutoff=SELECTION_END, allow_backfilled=cfg.allow_backfilled_emfx
    )
    selection_engine = ExecutionEngine(
        selection_market, selection_funding, execution_config(cfg)
    )
    selection_features = build_daily_features(selection_emfx, selection_engine)
    prefix_hashes = {
        "market": frame_hash(selection_market),
        "funding": frame_hash(selection_funding),
        "emfx": frame_hash(selection_emfx),
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
    emfx = load_emfx(
        cfg.emfx_csv, cutoff=FULL_CUTOFF, allow_backfilled=cfg.allow_backfilled_emfx
    )
    coverage = emfx.groupby("symbol")["observation_date"].max()
    if (coverage < pd.Timestamp("2026-05-29")).any():
        raise RuntimeError(f"OOS EM-FX source lacks frozen holdout coverage: {coverage.to_dict()}")
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    features = build_daily_features(emfx, engine)
    schedules = policy_schedules(engine, features, manifest["policy"], windows=tuple(WINDOWS))
    stats = schedule_stats(schedules, cfg)
    stress = schedule_stats(schedules, cfg, cost_rate=cfg.stress_cost_rate)
    significance = {
        name: _trade_stats(net_trade_returns(trades, cfg)) for name, trades in schedules.items()
    }
    complete = complete_session_mask(emfx)
    complete_dates = complete.index[complete]
    source_coverage_pass = bool(
        len(complete_dates) and complete_dates.max() >= pd.Timestamp("2026-05-29")
    )
    window_performance_pass = (
        stats["test_2024"]["absolute_return_pct"] > 0.0
        and stats["eval_2025"]["absolute_return_pct"] > 0.0
        and stats["holdout_2026h1"]["absolute_return_pct"] > 0.0
        and stats["oos_2024_2026h1"]["cagr_to_strict_mdd"] >= 3.0
        and stress["oos_2024_2026h1"]["absolute_return_pct"] > 0.0
    )
    raw_performance_pass = source_coverage_pass and window_performance_pass
    blockers: list[str] = []
    if not source_coverage_pass:
        blockers.append(
            "complete fixed-panel FX sessions end before 2026-05-29; 2025/2026 statistics are partial"
        )
    if not window_performance_pass:
        blockers.append("frozen OOS performance gate failed")
    blockers.append("backfilled EM-FX snapshot requires live point-in-time forward validation")
    if raw_performance_pass:
        blockers.append("actual trade/PnL orthogonality has not been audited")
    payload = {
        "mode": "frozen_oos_replay",
        "config": asdict(cfg),
        "manifest_hash": expected_hash,
        "future_did_not_rerank": True,
        "globally_research_seen_future": True,
        "policy": manifest["policy"],
        "source_quality": source_quality(emfx),
        "source_coverage_pass": source_coverage_pass,
        "window_performance_pass": window_performance_pass,
        "stats": stats,
        "double_cost_stats": stress,
        "trade_statistics": significance,
        "raw_performance_pass": raw_performance_pass,
        "orthogonality_pass": None,
        "promotion_pass": False,
        "promotion_blockers": blockers,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--emfx-csv", required=True)
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
    parser.add_argument("--allow-backfilled-emfx", action="store_true")
    return Config(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    result = run_oos(cfg) if cfg.open_oos else run_selection(cfg)
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "policy": result.get("policy", result.get("selected_policy", {})),
                "raw_performance_pass": result.get("raw_performance_pass"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
