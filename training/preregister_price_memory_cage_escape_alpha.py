"""Freeze outcome-blind support for the BTC price-memory cage-escape battery.

The event is deliberately conjunctive rather than a linear blend of weak
features.  A completed 30-day occupation-saddle crossing must agree with the
direction of multiple persistent barriers crossed during the next completed
hour, and completed quote-volume flow must confirm that direction.  No future
return is read by this module.
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

from training.search_occupation_saddle_escape_alpha import build_saddle_state
from training.search_persistent_barrier_annihilation_alpha import (
    persistent_barrier_features,
)


SELECTION_END = pd.Timestamp("2024-01-01")
FIT_START = pd.Timestamp("2020-10-15")
PERSISTENCE_HORIZONS = (576, 2016)  # 2d and 7d on a 5-minute grid.
HOLD_HOURS = (12, 24)
MIN_SADDLES_IN_PROFILE = 2
MIN_CROSSED_BARRIERS = 3
VOLUME_CLOCK_FRACTION = 0.25


@dataclass(frozen=True)
class Candidate:
    persistence_horizon_bars: int
    hold_hours: int

    @property
    def name(self) -> str:
        horizon_days = self.persistence_horizon_bars // 288
        return f"price_memory_cage_escape_{horizon_days}d_h{self.hold_hours}"


CANDIDATES = tuple(
    Candidate(horizon, hold)
    for horizon in PERSISTENCE_HORIZONS
    for hold in HOLD_HOURS
)


@dataclass(frozen=True)
class Config:
    input_csv: str = (
        "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    output: str = "results/price_memory_cage_escape_support_2026-07-19.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def implementation_path() -> str:
    source = Path(__file__).resolve()
    return str(source.relative_to(source.parents[1]))


def load_market(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    required = {
        "date",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
    }
    market = pd.read_csv(
        path,
        compression="infer",
        usecols=lambda column: column in required,
    )
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ValueError(f"market source missing columns: {missing}")
    market["date"] = (
        pd.to_datetime(market["date"], utc=True, errors="raise")
        .dt.tz_convert(None)
    )
    market = market.sort_values("date").reset_index(drop=True)
    if market.empty or market["date"].duplicated().any():
        raise ValueError("market source is empty or has duplicate timestamps")
    dates = market["date"]
    expected_last = SELECTION_END - pd.Timedelta("5min")
    if (
        dates.min() != pd.Timestamp("2020-01-01")
        or dates.max() != expected_last
    ):
        raise ValueError("market source is not the sealed 2020-2023 interval")
    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="5min")
    if not dates.equals(pd.Series(expected, name="date")):
        raise ValueError("market source is not a complete five-minute grid")
    numeric = market[list(required - {"date"})].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("market source contains non-finite required values")
    if (numeric[["close"]] <= 0.0).any().any():
        raise ValueError("market source contains non-positive prices")
    if (numeric[["quote_asset_volume", "taker_buy_quote"]] < 0.0).any().any():
        raise ValueError("market source contains negative quote flow")
    if (numeric["taker_buy_quote"] > numeric["quote_asset_volume"] + 1e-8).any():
        raise ValueError("taker-buy quote exceeds total quote volume")
    market.loc[:, numeric.columns] = numeric
    return market, dates


def volume_clock_flow_speed(
    market: pd.DataFrame,
    *,
    fraction: float = VOLUME_CLOCK_FRACTION,
) -> np.ndarray:
    """Return causal signed-flow speed over a trailing notional clock.

    The notional target is the strictly preceding completed 24-hour quote
    volume.  The interval ending at the current completed bar is then found by
    cumulative quote volume, and signed taker flow is divided by both interval
    notional and elapsed bars.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("volume-clock fraction must be in (0, 1]")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").clip(lower=0.0)
    buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce").clip(lower=0.0)
    signed = 2.0 * buy - quote
    target = quote.rolling(288, min_periods=288).sum().shift(1).to_numpy(float) * fraction
    cumulative_quote = np.r_[0.0, np.cumsum(quote.to_numpy(float))]
    cumulative_signed = np.r_[0.0, np.cumsum(signed.to_numpy(float))]
    index = np.arange(len(market))
    target_level = cumulative_quote[1:] - np.nan_to_num(target, nan=np.inf)
    start = np.searchsorted(cumulative_quote, target_level, side="left").clip(
        0, len(market) - 1
    )
    duration = (index - start).astype(float)
    valid = np.isfinite(target) & (start < index) & (duration > 0.0)
    notional = cumulative_quote[index + 1] - cumulative_quote[start]
    signed_flow = cumulative_signed[index + 1] - cumulative_signed[start]
    output = np.full(len(market), np.nan)
    output[valid] = (
        signed_flow[valid] / notional[valid] / duration[valid]
    )
    output[~np.isfinite(output)] = np.nan
    return output


def combine_event_clock(
    occupation: pd.DataFrame,
    persistence: pd.DataFrame,
    flow_speed: np.ndarray,
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Align completed features and return the fixed continuation event.

    At an hourly row ``t`` the preceding row ``t-5m`` contains the completed
    minute-55 occupation crossing.  The persistent traversal and flow state at
    ``t`` become available when that five-minute bar closes.  Execution is
    therefore no earlier than the next row, ``t+5m``.
    """
    if not (len(occupation) == len(persistence) == len(flow_speed) == len(dates)):
        raise ValueError("feature lengths differ")
    occupation_side = occupation["side"].shift(1, fill_value=0).to_numpy(np.int8)
    occupation_depth = pd.to_numeric(
        occupation["barrier_depth"].shift(1), errors="coerce"
    ).to_numpy(float)
    saddle_count = occupation["saddle_count"].shift(1, fill_value=0).to_numpy(int)
    persistence_side = persistence["side"].to_numpy(np.int8)
    barrier_count = persistence["barrier_count"].to_numpy(int)
    flow_side = np.sign(np.nan_to_num(flow_speed, nan=0.0)).astype(np.int8)
    hourly = dates.dt.minute.eq(0).to_numpy(bool)
    active = (
        hourly
        & np.isfinite(occupation_depth)
        & (saddle_count >= MIN_SADDLES_IN_PROFILE)
        & (barrier_count >= MIN_CROSSED_BARRIERS)
        & (occupation_side != 0)
        & (occupation_side == persistence_side)
        & (occupation_side == flow_side)
    )
    side = np.where(active, occupation_side, 0).astype(np.int8)
    return active, side


def nonoverlapping_schedule(
    dates: pd.Series,
    active: np.ndarray,
    side: np.ndarray,
    *,
    hold_hours: int,
    start: pd.Timestamp = FIT_START,
    end: pd.Timestamp = SELECTION_END,
) -> pd.DataFrame:
    if hold_hours <= 0:
        raise ValueError("hold_hours must be positive")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError("schedule end must follow start")
    rows: list[dict[str, Any]] = []
    next_allowed = start
    for position in np.flatnonzero(np.asarray(active, dtype=bool)):
        signal_bar_open = dates.iloc[position]
        feature_available = signal_bar_open + pd.Timedelta("5min")
        entry_time = feature_available
        exit_time = entry_time + pd.Timedelta(hours=hold_hours)
        action = int(side[position])
        if (
            entry_time < start
            or entry_time < next_allowed
            or exit_time >= end
            or action not in (-1, 1)
        ):
            continue
        rows.append(
            {
                "signal_bar_open": str(signal_bar_open),
                "feature_available": str(feature_available),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "side": action,
            }
        )
        next_allowed = exit_time
    return pd.DataFrame(
        rows,
        columns=[
            "signal_bar_open",
            "feature_available",
            "entry_time",
            "exit_time",
            "side",
        ],
    )


def period_support(schedule: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    if schedule.empty:
        subset = schedule.copy()
    else:
        entry = pd.to_datetime(schedule["entry_time"])
        subset = schedule.loc[(entry >= start) & (entry < end)].copy()
    total = len(subset)
    if not total:
        return {
            "total": 0,
            "longs": 0,
            "shorts": 0,
            "by_month": {},
            "max_month_fraction": 1.0,
        }
    entry = pd.to_datetime(subset["entry_time"])
    months = entry.dt.to_period("M").value_counts().sort_index()
    return {
        "total": int(total),
        "longs": int((subset["side"] > 0).sum()),
        "shorts": int((subset["side"] < 0).sum()),
        "by_month": {str(key): int(value) for key, value in months.items()},
        "max_month_fraction": float(months.max() / total),
    }


def support_summary(schedule: pd.DataFrame) -> dict[str, Any]:
    windows = {
        "fit": ("2020-10-15", "2023-01-01"),
        "fit_2020q4": ("2020-10-15", "2021-01-01"),
        "fit_2021h1": ("2021-01-01", "2021-07-01"),
        "fit_2021h2": ("2021-07-01", "2022-01-01"),
        "fit_2022h1": ("2022-01-01", "2022-07-01"),
        "fit_2022h2": ("2022-07-01", "2023-01-01"),
        "select_2023": ("2023-01-01", "2024-01-01"),
        "select_2023h1": ("2023-01-01", "2023-07-01"),
        "select_2023h2": ("2023-07-01", "2024-01-01"),
    }
    return {
        name: period_support(schedule, start, end)
        for name, (start, end) in windows.items()
    }


def windowed_support_summary(
    dates: pd.Series,
    active: np.ndarray,
    side: np.ndarray,
    *,
    hold_hours: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Mirror split-contained evaluator scheduling in every support window."""
    windows = {
        "fit": ("2020-10-15", "2023-01-01"),
        "fit_2020q4": ("2020-10-15", "2021-01-01"),
        "fit_2021h1": ("2021-01-01", "2021-07-01"),
        "fit_2021h2": ("2021-07-01", "2022-01-01"),
        "fit_2022h1": ("2022-01-01", "2022-07-01"),
        "fit_2022h2": ("2022-07-01", "2023-01-01"),
        "select_2023": ("2023-01-01", "2024-01-01"),
        "select_2023h1": ("2023-01-01", "2023-07-01"),
        "select_2023h2": ("2023-07-01", "2024-01-01"),
    }
    support: dict[str, Any] = {}
    schedule_hashes: dict[str, str] = {}
    for name, (start, end) in windows.items():
        schedule = nonoverlapping_schedule(
            dates,
            active,
            side,
            hold_hours=hold_hours,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
        )
        support[name] = period_support(schedule, start, end)
        schedule_hashes[name] = canonical_hash(schedule.to_dict(orient="records"))
    return support, schedule_hashes


def support_gates(summary: dict[str, Any]) -> dict[str, bool]:
    fit = summary["fit"]
    select = summary["select_2023"]

    def each_side_at_least(row: dict[str, Any], fraction: float) -> bool:
        total = int(row["total"])
        return total > 0 and min(int(row["longs"]), int(row["shorts"])) / total >= fraction

    return {
        "fit_at_least_48": int(fit["total"]) >= 48,
        "select_2023_at_least_24": int(select["total"]) >= 24,
        "each_full_fit_half_at_least_6": min(
            int(summary[name]["total"])
            for name in ("fit_2021h1", "fit_2021h2", "fit_2022h1", "fit_2022h2")
        ) >= 6,
        "each_2023_half_at_least_8": min(
            int(summary["select_2023h1"]["total"]),
            int(summary["select_2023h2"]["total"]),
        ) >= 8,
        "fit_each_side_at_least_20pct": each_side_at_least(fit, 0.20),
        "select_each_side_at_least_20pct": each_side_at_least(select, 0.20),
        "fit_max_month_at_most_20pct": float(fit["max_month_fraction"]) <= 0.20,
        "select_max_month_at_most_25pct": float(select["max_month_fraction"]) <= 0.25,
    }


def build_report(cfg: Config) -> dict[str, Any]:
    market, dates = load_market(cfg.input_csv)
    occupation = build_saddle_state(market, dates, mode="joint")
    flow_speed = volume_clock_flow_speed(market)
    candidate_rows: list[dict[str, Any]] = []
    raw_events: dict[int, int] = {}
    for horizon in PERSISTENCE_HORIZONS:
        persistence = persistent_barrier_features(
            np.log(market["close"].to_numpy(float)), dates, horizon=horizon
        )
        active, side = combine_event_clock(occupation, persistence, flow_speed, dates)
        raw_events[horizon] = int(active.sum())
        event_clock = [
            {"signal_bar_open": str(dates.iloc[position]), "side": int(side[position])}
            for position in np.flatnonzero(active)
        ]
        for candidate in (item for item in CANDIDATES if item.persistence_horizon_bars == horizon):
            support, schedule_hashes = windowed_support_summary(
                dates,
                active,
                side,
                hold_hours=candidate.hold_hours,
            )
            gates = support_gates(support)
            candidate_rows.append(
                {
                    "candidate": asdict(candidate),
                    "name": candidate.name,
                    "clock_hash": canonical_hash(event_clock),
                    "schedule_hashes": schedule_hashes,
                    "support": support,
                    "gates": gates,
                    "passes_support": bool(all(gates.values())),
                }
            )
    stable = {
        "protocol": {
            "outcomes_opened": False,
            "market_end_exclusive": str(SELECTION_END),
            "fit_start": str(FIT_START),
            "occupation": "prior 30 UTC days, daily frozen, joint time/quote-volume density",
            "persistence": "prior frozen 2d or 7d local-extrema spectrum; current completed-hour traversal",
            "flow": "signed taker flow speed over 0.25x strictly-prior 24h quote notional",
            "event": (
                "occupation crossing, persistence traversal and volume flow share direction; "
                f"profile saddles>={MIN_SADDLES_IN_PROFILE}; crossed barriers>={MIN_CROSSED_BARRIERS}"
            ),
            "entry": "next five-minute open after all completed features",
            "exit": "fixed elapsed hold; global non-overlap per candidate",
            "candidate_count": len(CANDIDATES),
            "direction": "constraint-release continuation; fixed before returns",
            "direction_repair_allowed": False,
            "threshold_repair_after_returns_allowed": False,
            "2024_plus_opened": False,
        },
        "source": {"path": cfg.input_csv, "sha256": sha256_file(cfg.input_csv)},
        "implementation": {
            "path": implementation_path(),
            "sha256": sha256_file(__file__),
        },
        "raw_events_by_horizon": {str(key): value for key, value in raw_events.items()},
        "candidates": candidate_rows,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **stable,
        "support_freeze_hash": canonical_hash(stable),
    }


def write_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--output", default=Config.output)
    cfg = Config(**vars(parser.parse_args()))
    report = build_report(cfg)
    write_exclusive(cfg.output, report)
    print(
        json.dumps(
            {
                "candidates": len(report["candidates"]),
                "support_passes": sum(
                    item["passes_support"] for item in report["candidates"]
                ),
                "raw_events_by_horizon": report["raw_events_by_horizon"],
                "support_freeze_hash": report["support_freeze_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
