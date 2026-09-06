#!/usr/bin/env python3
"""One-shot evaluator for the frozen 1x RV20 low-participation shock reversal with causal breakeven protection.

The pre-2024-selected Q60/hold12/stop4/take4 two-sided cell is replayed
exactly at 1.0x with a 1% favorable-excursion breakeven rule that activates
only on the next 5-minute bar. Only its 2024 Gross9 weight may be selected;
future values remain closed unless selection succeeds, then may only veto.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.gross9_structural_clock_primitives import (
    attach_open_interest,
    load_market,
)
import training.audit_gross9_pullback_premium_overheat_marginal as gross9
import training.portfolio_opt_added_alpha_update as portfolio
from training.audit_rank7_fresh_kimchi_fixed_portfolio import subaccount_bar_path
from training.search_inventory_purge_reclaim_alpha import Config as ExecutionConfig


PREREGISTRATION = Path(
    "results/rv20_low_participation_shock_reversal_breakeven1_fixed1x_preregistration_2026-08-08.json"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "f2738dca6a29c697aea9a05a45cd4b4a42d1ebef8476c02fa100f63424ab68ae"
)
OUTPUT = Path("results/rv20_low_participation_shock_reversal_breakeven1_fixed1x_evaluation_2026-08-08.json")
REPORT = Path("docs/rv20-low-participation-shock-reversal-breakeven1-fixed1x-evaluation-2026-08-08.md")
ATTEMPT = Path(
    "results/rv20_low_participation_shock_reversal_breakeven1_fixed1x_evaluation_attempt_consumed_2026-08-08.json"
)

TRAIN = (pd.Timestamp("2020-09-01"), pd.Timestamp("2024-01-01"))
CONFIRMATION = (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"))
FUTURE_WINDOWS = {
    "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    "2026h1": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01")),
    "2025_2026h1": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-06-01")),
}
FEATURE_WINDOW = 2160
RV_WINDOW = 20
RV_REFERENCE_WINDOW = 756
ENTRY_COST = 0.0006
STRESS_COST = 0.0010
ABSOLUTE_HOUR_RETURN_QUANTILE = 0.90
HOUR_RANGE_QUANTILE = 0.90
CAPITULATION_TAKER_FRACTION = 0.48
OI_CHANGE_UPPER_QUANTILE = 0.50
CANDIDATE = "rv20_low_participation_shock_reversal_breakeven1_fixed1x"
BASELINE_WEIGHTS = dict(gross9.BASELINE_WEIGHTS)
BASELINE_GROSS = float(sum(BASELINE_WEIGHTS.values()))
AUTHORITATIVE_GROSS9_MARKET = (
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
)
AUTHORITATIVE_GROSS9_WIDE_OI = (
    "/home/pakchu/rllm/data/"
    "cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
)
INPUT_SHA256 = {
    "market": "b5d6f4b2edb4d3a7f68dc84097a5c3caf6947b2cf77ceb6a3e621d220068a282",
    "open_interest": "dc73ba1d080bd4c4a493e315467ca998399a55914041c67c6ae08db84266ded8",
    "funding": "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
    "gross9_anchor": "329878d90b6cd9c731eb4871ac041256f95f03c14dd261ada681d3a370709875",
    "pre2024_selection": "8cb0cdd20232885575c70de1e150a706254489289d310052f3372c70ce61d934",
}
GROSS9_CONTEXT_SHA256 = {
    AUTHORITATIVE_GROSS9_MARKET: "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
    AUTHORITATIVE_GROSS9_WIDE_OI: "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz": "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
    "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json": "006f82e1f0affad9f96a08a6c600542feec4a0e1198ed99b8630627de4913450",
    "results/portfolio_rank7_capacity_update_2026-07-28.json": "7260ba91698ac31838558d1af11701e5e251f9b49749a573c8d38610d5388756",
    "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json": "e079f7a70d4e5eea7de962cf5daad93fd634fdf5779854d1783f83a837dc41ab",
    "training/audit_gross9_fixed_candidate_state_substitution.py": "b727d3fb0c45801e839265e0595be457a7fb204b8f64b92247cb82d50e49f4c2",
    "training/portfolio_opt_added_alpha_update.py": "0bbe02d60ed7393e704a72c65d92d2a952eafee0d5a3d32310a782b80e2ae901",
    "training/audit_rank7_fresh_kimchi_fixed_portfolio.py": "a311492a47dd63647130d3eeb7a6740de23c12b0e6af42f8f5e81f7b0fb41345",
}

@dataclass(frozen=True, order=True)
class Cell:
    max_hold_hours: int
    rv20_regime_quantile: float
    side_mode: str
    stop_loss_pct: float
    take_profit_pct: float


FROZEN_CELL = Cell(12, 0.60, "both", 0.04, 0.04)
FIXED_LEVERAGE = 1.0
BREAKEVEN_ACTIVATION_PCT = 0.01
BREAKEVEN_LOCK_PCT = 0.0


@dataclass(frozen=True)
class Trade:
    signal_position: int
    entry_position: int
    exit_position: int
    side: int
    exit_kind: str
    entry_time: str
    exit_time: str
    price_factor: float
    funding_factor: float
    favorable_price_factor: float
    adverse_price_factor: float
    funding_debit_factor: float = 1.0

    @property
    def gross_return(self) -> float:
        """Compatibility sign used by the authenticated Gross9 path adapter."""
        return float(self.price_factor) - 1.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    """Authenticate the immutable preregistration before decoding it."""
    observed = sha256_file(path)
    if observed != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration hash drifted: {observed} != "
            f"{EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "rllm.rv20-low-participation-shock-reversal-breakeven1-fixed1x.preregistration.v1":
        raise RuntimeError("unexpected RV20 preregistration schema")
    if payload.get("status") != "FROZEN_BEFORE_2024_CONFIRMATION_INSPECTION":
        raise RuntimeError("RV20 preregistration is not frozen")
    if payload["one_shot"] != {
        "candidate_value_open_count_max": 1,
        "repair_allowed": False,
        "resume_allowed": False,
        "retry_allowed": False,
    }:
        raise RuntimeError("one-shot contract drifted")
    if payload.get("frozen_candidate") != {
        "absolute_hour_return_quantile": ABSOLUTE_HOUR_RETURN_QUANTILE,
        "capitulation_taker_fraction": CAPITULATION_TAKER_FRACTION,
        "hour_range_quantile": HOUR_RANGE_QUANTILE,
        "breakeven_activation_pct": BREAKEVEN_ACTIVATION_PCT,
        "breakeven_lock_pct": BREAKEVEN_LOCK_PCT,
        "leverage": FIXED_LEVERAGE,
        "max_hold_hours": FROZEN_CELL.max_hold_hours,
        "oi_change_upper_quantile": OI_CHANGE_UPPER_QUANTILE,
        "rv20_regime_quantile": FROZEN_CELL.rv20_regime_quantile,
        "side_mode": FROZEN_CELL.side_mode,
        "stop_loss_pct": FROZEN_CELL.stop_loss_pct,
        "take_profit_pct": FROZEN_CELL.take_profit_pct,
    }:
        raise RuntimeError("frozen candidate contract drifted")
    if payload.get("selection", {}).get("train_replay_only") is not True:
        raise RuntimeError("train replay contract drifted")
    if payload.get("selection", {}).get("portfolio_weight_grid") != [0.25, 0.5, 0.75]:
        raise RuntimeError("weight grid drifted")
    source = payload.get("contamination", {})
    if (
        payload.get("inputs", {}).get("pre2024_selection")
        != source.get("source_selection_artifact")
        or source.get("source_selection_sha256")
        != INPUT_SHA256["pre2024_selection"]
        or source.get("source_verdict") != "REJECT_NO_STRUCTURAL_TOP1"
    ):
        raise RuntimeError("pre2024 selection identity drifted")
    return payload


def frozen_grid(preregistration: Mapping[str, Any]) -> tuple[Cell, ...]:
    """Compatibility surface exposing exactly the preregistered fixed cell."""
    del preregistration
    return (FROZEN_CELL,)


def shifted_rolling_quantile(
    values: pd.Series, quantile: float, window: int
) -> pd.Series:
    """Quantile of the preceding ``window`` finite observations only."""
    numeric = pd.to_numeric(values, errors="coerce")
    # rolling counts rows, so evaluate on the compact finite clock and map back.
    finite = numeric.dropna()
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    result.loc[finite.index] = (
        finite.rolling(window, min_periods=window)
        .quantile(float(quantile), interpolation="linear")
        .shift(1)
    )
    return result


def _require_regular_5m_hour(group: pd.DataFrame) -> bool:
    if len(group) != 12:
        return False
    dates = pd.DatetimeIndex(group["date"])
    return bool(
        dates[0].minute == 0
        and np.all(np.diff(dates.view("i8")) == pd.Timedelta("5min").value)
    )


def build_hourly_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build completed-hour features; incomplete hours are absent/inactive."""
    required = {
        "date", "open", "high", "low", "close", "quote_asset_volume",
        "taker_buy_quote", "open_interest",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"market lacks hourly feature columns: {missing}")
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None)
    frame = (
        frame.sort_values("date").drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    frame["hour"] = frame["date"].dt.floor("h")
    rows: list[dict[str, Any]] = []
    for hour, group in frame.groupby("hour", sort=True):
        if not _require_regular_5m_hour(group):
            continue
        quote = pd.to_numeric(group["quote_asset_volume"], errors="coerce")
        taker = pd.to_numeric(group["taker_buy_quote"], errors="coerce")
        values = {
            name: pd.to_numeric(group[name], errors="coerce")
            for name in ("open", "high", "low", "close", "open_interest")
        }
        if (
            quote.isna().any()
            or taker.isna().any()
            or any(series.isna().any() for series in values.values())
        ):
            continue
        quote_sum = float(quote.sum())
        oi = float(values["open_interest"].iloc[-1])
        rows.append(
            {
                "date": pd.Timestamp(hour),
                "signal_position": int(group.index[-1]),
                "hour_return": math.log(
                    float(values["close"].iloc[-1]) / float(values["open"].iloc[0])
                ),
                "hour_range": math.log(
                    float(values["high"].max()) / float(values["low"].min())
                ),
                "taker_fraction": (
                    float(taker.sum()) / quote_sum
                    if quote_sum > 0.0 and np.isfinite(quote_sum)
                    else np.nan
                ),
                "open_interest": oi if oi > 0.0 else np.nan,
            }
        )
    hourly = pd.DataFrame(rows)
    if hourly.empty:
        return hourly
    full_clock = pd.date_range(
        hourly["date"].iloc[0], hourly["date"].iloc[-1], freq="h"
    )
    hourly = hourly.set_index("date").reindex(full_clock)
    hourly.index.name = "date"
    hourly = hourly.reset_index()
    hourly["oi_change_1h"] = np.log(
        hourly["open_interest"] / hourly["open_interest"].shift(1)
    )
    return hourly.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def build_rv20_daily_clock(market: pd.DataFrame) -> pd.DataFrame:
    """Causal daily RV20 with selectable shifted Q60/Q75 reference clocks."""
    frame = market.loc[:, ["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["day"] = frame["date"].dt.normalize()
    groups = frame.groupby("day", sort=True)
    daily = groups.tail(1).set_index("day")
    # A daily reference exists only after that UTC day has actually completed.
    complete = groups["date"].max().dt.hour.eq(23) & groups["date"].max().dt.minute.eq(55)
    daily = daily.loc[complete[complete].index]
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    returns = np.log(daily["close"] / daily["close"].shift(1))
    # At day D, only returns ending on D-1 and earlier are usable.
    rv20 = np.sqrt(365.0 * returns.pow(2).shift(1).rolling(20, min_periods=20).mean())
    return pd.DataFrame({
        "rv20": rv20,
        "rv20_q60": shifted_rolling_quantile(
            rv20, 0.60, RV_REFERENCE_WINDOW
        ),
        "rv20_q75": shifted_rolling_quantile(
            rv20, 0.75, RV_REFERENCE_WINDOW
        ),
    })


def attach_causal_thresholds(
    hourly: pd.DataFrame, daily_clock: pd.DataFrame, cells: Sequence[Cell]
) -> pd.DataFrame:
    output = hourly.copy()
    days = pd.DatetimeIndex(output["date"]).normalize()
    output["rv20"] = daily_clock["rv20"].reindex(days).to_numpy(float)
    for quantile in sorted({cell.rv20_regime_quantile for cell in cells}):
        column = f"rv20_q{round(quantile * 100)}"
        output[column] = daily_clock[column].reindex(days).to_numpy(float)
    specs = {
        ("abs_return", ABSOLUTE_HOUR_RETURN_QUANTILE),
        ("range", HOUR_RANGE_QUANTILE),
        ("oi", OI_CHANGE_UPPER_QUANTILE),
    }
    source = {
        "abs_return": output["hour_return"].abs(),
        "range": output["hour_range"],
        "oi": output["oi_change_1h"],
    }
    for name, quantile in sorted(specs):
        output[f"q_{name}_{quantile:g}"] = shifted_rolling_quantile(
            source[name], quantile, FEATURE_WINDOW
        )
    return output


def cell_signal_sides(hourly: pd.DataFrame, cell: Cell) -> np.ndarray:
    if cell.side_mode not in {"both", "long_only", "short_only"}:
        raise ValueError(f"invalid side_mode: {cell.side_mode}")
    absolute = hourly[f"q_abs_return_{ABSOLUTE_HOUR_RETURN_QUANTILE:g}"]
    range_threshold = hourly[f"q_range_{HOUR_RANGE_QUANTILE:g}"]
    oi_threshold = hourly[f"q_oi_{OI_CHANGE_UPPER_QUANTILE:g}"]
    regime_threshold = hourly[
        f"rv20_q{round(cell.rv20_regime_quantile * 100)}"
    ]
    common = (
        (hourly["rv20"] >= regime_threshold).fillna(False)
        & (hourly["hour_return"].abs() >= absolute)
        & (hourly["hour_range"] >= range_threshold)
        & (hourly["oi_change_1h"] <= oi_threshold)
    )
    long = common & (hourly["hour_return"] < 0.0) & (
        hourly["taker_fraction"] <= CAPITULATION_TAKER_FRACTION
    )
    short = common & (hourly["hour_return"] > 0.0) & (
        hourly["taker_fraction"] >= 1.0 - CAPITULATION_TAKER_FRACTION
    )
    if cell.side_mode == "long_only":
        short = pd.Series(False, index=hourly.index)
    elif cell.side_mode == "short_only":
        long = pd.Series(False, index=hourly.index)
    return np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)


def _funding_factors(
    funding: pd.DataFrame, entry: pd.Timestamp, exit_: pd.Timestamp, side: int,
    leverage: float,
) -> tuple[float, float]:
    if funding.empty:
        return 1.0, 1.0
    dates = pd.DatetimeIndex(pd.to_datetime(funding["date"], utc=True)).tz_convert(None)
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    selected = rates[(dates >= entry) & (dates <= exit_)]
    factors = 1.0 - float(leverage) * int(side) * selected
    if not np.isfinite(factors).all() or np.any(factors <= 0.0):
        raise RuntimeError("invalid realized funding factor")
    if not len(factors):
        return 1.0, 1.0
    cumulative = np.cumprod(factors, dtype=float)
    return float(cumulative[-1]), float(min(1.0, cumulative.min()))


def execute_trade(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    signal_position: int,
    side: int,
    cell: Cell,
    *,
    leverage: float = 1.0,
) -> Trade | None:
    """Enter next open; apply each bar's pre-fixed stop adverse-first."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    entry = int(signal_position) + 1
    horizon = entry + int(cell.max_hold_hours) * 12
    if entry >= len(market) or horizon >= len(market):
        return None
    last = horizon - 1
    entry_price = float(market.iloc[entry]["open"])
    initial_stop = entry_price * (1.0 - side * cell.stop_loss_pct)
    breakeven_stop = entry_price * (1.0 + side * BREAKEVEN_LOCK_PCT)
    activation = entry_price * (1.0 + side * BREAKEVEN_ACTIVATION_PCT)
    take = entry_price * (1.0 + side * cell.take_profit_pct)
    favorable = entry_price
    adverse = entry_price
    breakeven_from: int | None = None
    exit_position = horizon
    exit_kind = "time"
    exit_price = float(market.iloc[horizon]["open"])
    for position in range(entry, last + 1):
        # The stop is frozen before this bar opens. A favorable touch on this
        # bar can only alter the stop beginning with the following bar.
        stop = (
            breakeven_stop
            if breakeven_from is not None and position >= breakeven_from
            else initial_stop
        )
        open_price = float(market.iloc[position]["open"])
        high = float(market.iloc[position]["high"])
        low = float(market.iloc[position]["low"])
        gap_stop = open_price <= stop if side > 0 else open_price >= stop
        gap_take = open_price >= take if side > 0 else open_price <= take
        if gap_stop:
            exit_position, exit_kind, exit_price = position, "gap_stop", open_price
            adverse = open_price
            break
        if gap_take:
            exit_position, exit_kind, exit_price = position, "gap_take", open_price
            favorable = open_price
            break
        favorable = max(favorable, high) if side > 0 else min(favorable, low)
        adverse = min(adverse, low) if side > 0 else max(adverse, high)
        stop_hit = low <= stop if side > 0 else high >= stop
        take_hit = high >= take if side > 0 else low <= take
        if stop_hit:  # deliberately before take_hit and before arming
            exit_position, exit_kind, exit_price = position, "stop", stop
            adverse = stop
            break
        if take_hit:
            exit_position, exit_kind, exit_price = position, "take", take
            break
        activation_hit = high >= activation if side > 0 else low <= activation
        if breakeven_from is None and activation_hit:
            breakeven_from = position + 1
    entry_time = pd.Timestamp(market.iloc[entry]["date"])
    exit_time = pd.Timestamp(market.iloc[exit_position]["date"])
    gross = side * (exit_price / entry_price - 1.0)
    funding_factor, funding_debit_factor = _funding_factors(
        funding, entry_time, exit_time, side, leverage
    )
    return Trade(
        signal_position=int(signal_position), entry_position=entry,
        exit_position=int(exit_position), side=int(side), exit_kind=exit_kind,
        entry_time=entry_time.isoformat(), exit_time=exit_time.isoformat(),
        price_factor=max(0.0, 1.0 + leverage * gross),
        funding_factor=funding_factor,
        favorable_price_factor=max(
            0.0, 1.0 + leverage * side * (favorable / entry_price - 1.0)
        ),
        adverse_price_factor=max(
            0.0, 1.0 + leverage * side * (adverse / entry_price - 1.0)
        ),
        funding_debit_factor=funding_debit_factor,
    )


def execute_schedule(
    market: pd.DataFrame, funding: pd.DataFrame, hourly: pd.DataFrame, cell: Cell,
    *, leverage: float = 1.0,
) -> list[Trade]:
    sides = cell_signal_sides(hourly, cell)
    trades: list[Trade] = []
    next_entry = 0
    for row, side in zip(hourly.itertuples(index=False), sides):
        if not side or pd.isna(row.signal_position):
            continue
        if int(row.signal_position) + 1 < next_entry:
            continue
        trade = execute_trade(
            market, funding, int(row.signal_position), int(side), cell,
            leverage=leverage,
        )
        if trade is not None:
            trades.append(trade)
            next_entry = trade.exit_position + 1
    return trades


def execute_schedule_window(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    hourly: pd.DataFrame,
    cell: Cell,
    *,
    start: Any,
    end: Any,
    leverage: float = 1.0,
) -> list[Trade]:
    """Schedule one window from flat state without cross-boundary suppression."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    sides = cell_signal_sides(hourly, cell)
    trades: list[Trade] = []
    next_entry = 0
    for row, side in zip(hourly.itertuples(index=False), sides):
        if not side or pd.isna(row.signal_position):
            continue
        entry_position = int(row.signal_position) + 1
        if entry_position >= len(market):
            continue
        entry_time = pd.Timestamp(market.iloc[entry_position]["date"])
        if entry_time < start_ts or entry_time >= end_ts:
            continue
        if entry_position < next_entry:
            continue
        trade = execute_trade(
            market, funding, int(row.signal_position), int(side), cell,
            leverage=leverage,
        )
        if trade is None:
            continue
        if pd.Timestamp(trade.exit_time) >= end_ts:
            break
        trades.append(trade)
        next_entry = trade.exit_position + 1
    return trades


def years_between(start: Any, end: Any) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (
        365.25 * 86400.0
    )


def strict_metrics(
    trades: Iterable[Trade], *, start: Any, end: Any, cost_rate: float,
    leverage: float = 1.0,
) -> dict[str, Any]:
    """Full-calendar CAGR and global favorable-before-adverse strict MDD."""
    cost_factor = 1.0 - float(leverage) * float(cost_rate)
    if cost_factor <= 0.0:
        raise ValueError("non-positive cost factor")
    equity = peak = 1.0
    mdd = 0.0
    selected = [
        trade for trade in trades
        if pd.Timestamp(start) <= pd.Timestamp(trade.entry_time) < pd.Timestamp(end)
        and pd.Timestamp(trade.exit_time) < pd.Timestamp(end)
    ]
    for trade in selected:
        favorable = equity * cost_factor * trade.favorable_price_factor
        intratrade_peak = max(peak, favorable)
        adverse = (
            equity * cost_factor * trade.funding_debit_factor
            * trade.adverse_price_factor
        )
        mdd = max(mdd, 1.0 - adverse / max(intratrade_peak, 1e-15))
        peak = intratrade_peak
        equity *= (
            cost_factor * trade.price_factor * trade.funding_factor * cost_factor
        )
        mdd = max(mdd, 1.0 - equity / max(peak, 1e-15))
        peak = max(peak, equity)
    years = years_between(start, end)
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else -100.0
    mdd_pct = mdd * 100.0
    return {
        "absolute_return_pct": (equity - 1.0) * 100.0,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd_pct,
        "cagr_to_strict_mdd": cagr / mdd_pct if mdd_pct > 1e-12 else 0.0,
        "trades": len(selected),
        "longs": sum(t.side > 0 for t in selected),
        "shorts": sum(t.side < 0 for t in selected),
        "active_weeks": len({pd.Timestamp(t.entry_time).to_period("W") for t in selected}),
    }


def select_weight(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passing = [row for row in rows if row.get("pass") is True]
    return max(
        passing,
        key=lambda row: (
            float(row["same_gross_ratio_improvement"]),
            -float(row["weight"]),
        ),
    ) if passing else None


def frozen_future_veto(
    frozen: Mapping[str, Any], checks: Mapping[str, bool]
) -> dict[str, Any]:
    """Apply future gates to rank-1 only; never accept an alternatives list."""
    if any(key in frozen for key in ("rank", "replacement", "alternatives")):
        raise RuntimeError("future input attempted rerank or repair")
    passed = bool(checks) and all(value is True for value in checks.values())
    return {
        "frozen_candidate": dict(frozen),
        "checks": {str(key): bool(value) for key, value in sorted(checks.items())},
        "passed": passed,
        "action": "PROMOTE" if passed else "VETO_FROZEN_TOP1",
        "repair_attempted": False,
        "rerank_attempted": False,
    }


def exact_entry_jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a = {str(pd.Timestamp(value)) for value in left}
    b = {str(pd.Timestamp(value)) for value in right}
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def max_per_sleeve_entry_jaccard(
    candidate: Iterable[Any], gross9_entries: Mapping[str, Iterable[Any]]
) -> dict[str, Any]:
    values = {
        sleeve: exact_entry_jaccard(candidate, entries)
        for sleeve, entries in sorted(gross9_entries.items())
    }
    return {"per_sleeve": values, "maximum": max(values.values(), default=0.0)}


def persistent_long_vol_control(
    market: pd.DataFrame, funding: pd.DataFrame, daily_clock: pd.DataFrame,
    *, regime_quantile: float, leverage: float, cost_rate: float,
) -> list[Trade]:
    """Long each day active under the frozen cell's RV20 regime quantile."""
    threshold_column = f"rv20_q{round(regime_quantile * 100)}"
    if threshold_column not in daily_clock:
        raise ValueError(f"daily clock lacks selected regime: {threshold_column}")
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"], utc=True)).tz_convert(None)
    days = dates.normalize()
    trades: list[Trade] = []
    for day, positions in pd.Series(np.arange(len(market)), index=days).groupby(level=0):
        if day not in daily_clock.index:
            continue
        row = daily_clock.loc[day]
        if not (
            np.isfinite(row[threshold_column])
            and row["rv20"] >= row[threshold_column]
        ):
            continue
        pos = positions.to_numpy(int)
        entry, exit_ = int(pos[0]), int(pos[-1])
        entry_price = float(market.iloc[entry]["open"])
        high = float(market.iloc[pos]["high"].max())
        low = float(market.iloc[pos]["low"].min())
        close = float(market.iloc[exit_]["close"])
        funding_factor, funding_debit_factor = _funding_factors(
            funding, dates[entry], dates[exit_], 1, leverage
        )
        trades.append(Trade(
            signal_position=entry - 1, entry_position=entry, exit_position=exit_,
            side=1, exit_kind="day_close", entry_time=dates[entry].isoformat(),
            exit_time=dates[exit_].isoformat(),
            price_factor=1.0 + leverage * (close / entry_price - 1.0),
            funding_factor=funding_factor,
            favorable_price_factor=1.0 + leverage * (high / entry_price - 1.0),
            adverse_price_factor=max(0.0, 1.0 + leverage * (low / entry_price - 1.0)),
            funding_debit_factor=funding_debit_factor,
        ))
    return trades


def control_decomposition(
    candidate_metrics: Mapping[str, Any], control_metrics: Mapping[str, Any]
) -> dict[str, float]:
    candidate_log = math.log1p(float(candidate_metrics["absolute_return_pct"]) / 100.0)
    control_log = math.log1p(float(control_metrics["absolute_return_pct"]) / 100.0)
    return {
        "candidate_log_return": candidate_log,
        "control_log_return": control_log,
        "log_return_residual": candidate_log - control_log,
    }


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict", "NOT_RUN")
    lines = [
        "# RV20 Low-Participation Shock Reversal Breakeven1 Fixed1x Evaluation\n\n"
        f"- Preregistration SHA-256: `{EXPECTED_PREREGISTRATION_SHA256}`\n"
        f"- Verdict: **{verdict}**\n"
        "- Selection: exact train replay, then 2024 Gross9 weight only.\n"
        "- Future use: veto only; no repair, rerank, rank-2, or threshold change.\n"
    ]
    top = payload.get("frozen_structural_top1")
    if isinstance(top, Mapping):
        lines.extend(["\n## Frozen selection\n\n", "```json\n"])
        lines.append(canonical_json({
            "cell": top.get("cell"),
            "leverage": payload.get("fixed_leverage", FIXED_LEVERAGE),
            "weight": (payload.get("selected_weight") or {}).get("weight"),
        }))
        lines.append("```\n")
    future = payload.get("future_veto")
    if isinstance(future, Mapping):
        lines.extend(["\n## Frozen future veto checks\n\n"])
        for name, passed in sorted(future.get("checks", {}).items()):
            lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**\n")
    lines.extend([
        "\n## Protocol integrity\n\n",
        f"- Structural cells tested: `{payload.get('tested_structural_cells', 0)}`\n",
        f"- Fixed leverage: `{payload.get('fixed_leverage', FIXED_LEVERAGE)}`\n",
        f"- Future used for ranking: `{payload.get('future_used_for_ranking', False)}`\n",
        f"- Repair attempted: `{payload.get('repair_attempted', False)}`\n",
        f"- Rerank attempted: `{payload.get('rerank_attempted', False)}`\n",
    ])
    return "".join(lines)


def write_outputs(payload: Mapping[str, Any], output: Path, report: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    for path, text in ((output, canonical_json(payload)), (report, render_markdown(payload))):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite frozen output: {path}")
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            raw = text.encode("utf-8")
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        os.chmod(path, 0o444)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def consume_one_shot(marker: Path, output: Path, report: Path) -> None:
    """Durably consume the sole economic-open authority before decoding values."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    for path in (marker, output, report):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"economic one-shot already consumed: {path}")
    payload = canonical_json({
        "candidate": CANDIDATE,
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "retry_allowed": False,
        "resume_allowed": False,
        "state": "ECONOMIC_OPEN_CONSUMED",
    }).encode("utf-8")
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def authenticate_inputs(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every frozen input before any CSV/JSON value is decoded."""
    configured = {str(k): str(v) for k, v in preregistration["inputs"].items()}
    if set(configured) != set(INPUT_SHA256):
        raise RuntimeError("preregistered input set drifted")
    records: dict[str, Any] = {}
    for name in sorted(INPUT_SHA256):
        path = configured[name]
        observed = sha256_file(path)
        expected = INPUT_SHA256[name]
        if observed != expected:
            raise RuntimeError(
                f"input hash drifted for {name}: {observed} != {expected}"
            )
        records[name] = {"path": path, "sha256": observed, "validated": True}
    context_records: dict[str, Any] = {}
    for path, expected in GROSS9_CONTEXT_SHA256.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Gross9 context input drifted for {path}: {observed} != {expected}"
            )
        context_records[path] = {"sha256": observed, "validated": True}
    records["gross9_context_dependencies"] = context_records
    return records


def load_inputs(preregistration: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    authenticate_inputs(preregistration)
    inputs = preregistration["inputs"]
    market = load_market(inputs["market"])
    oi = pd.read_csv(inputs["open_interest"], compression="infer")
    market = attach_open_interest(market, oi)
    funding = pd.read_csv(inputs["funding"], compression="infer")
    if "funding_time" in funding and "date" not in funding:
        funding = funding.rename(columns={"funding_time": "date"})
    funding["date"] = pd.to_datetime(funding["date"], utc=True).dt.tz_convert(None)
    return market, funding.loc[:, ["date", "funding_rate"]]


def extend_gross9_context(
    gross_market: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    events: Sequence[Mapping[str, Any]],
    complete_market: pd.DataFrame,
    *,
    split_bounds: Mapping[str, tuple[Any, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Authenticate an exact Gross9 prefix and zero-pad it to G9CB12."""
    if len(gross_market) > len(complete_market):
        target_size = len(complete_market)
        original_size = len(gross_market)
        cropped_events: list[dict[str, Any]] = []
        for source in events:
            event = dict(source)
            for field in ("ret", "adv", "fav", "low", "high"):
                values = np.asarray(source[field], dtype=np.float64)
                if len(values) != original_size:
                    raise RuntimeError(f"Gross9 event length drifted: {field}")
                event[field] = values[:target_size].copy()
            cropped_events.append(event)
        masks = {
            split: np.asarray(mask, dtype=bool)[:target_size].copy()
            for split, mask in masks.items()
        }
        gross_market = gross_market.iloc[:target_size].reset_index(drop=True)
        events = cropped_events
    prefix = complete_market.iloc[: len(gross_market)]
    for column in ("date", "open", "high", "low", "close"):
        left = (
            pd.to_datetime(gross_market[column]).to_numpy()
            if column == "date"
            else pd.to_numeric(gross_market[column], errors="raise").to_numpy(float)
        )
        right = (
            pd.to_datetime(prefix[column]).to_numpy()
            if column == "date"
            else pd.to_numeric(prefix[column], errors="raise").to_numpy(float)
        )
        equal = (
            np.array_equal(left, right)
            if column == "date"
            else (
                np.isfinite(left).all()
                and np.isfinite(right).all()
                and np.allclose(left, right, rtol=0.0, atol=1e-10, equal_nan=False)
            )
        )
        if not equal:
            raise RuntimeError(f"Gross9 is not an authenticated G9CB12 prefix: {column}")
    size = len(complete_market)
    padded_events: list[dict[str, Any]] = []
    for source in events:
        event = dict(source)
        for field in ("ret", "adv", "fav", "low", "high"):
            values = np.asarray(source[field], dtype=np.float64)
            if len(values) != len(gross_market):
                raise RuntimeError(f"Gross9 event length drifted: {field}")
            event[field] = np.pad(values, (0, size - len(values)))
        padded_events.append(event)
    dates = pd.DatetimeIndex(pd.to_datetime(complete_market["date"]))
    padded_masks: dict[str, np.ndarray] = {}
    bounds = split_bounds or portfolio.SPLIT_BOUNDS
    for split, (start, end) in bounds.items():
        if split not in masks:
            raise RuntimeError(f"Gross9 context lacks split {split}")
        expected_prefix = (
            (pd.DatetimeIndex(pd.to_datetime(gross_market["date"])) >= pd.Timestamp(start))
            & (pd.DatetimeIndex(pd.to_datetime(gross_market["date"])) < pd.Timestamp(end))
        )
        if not np.array_equal(np.asarray(masks[split], dtype=bool), expected_prefix):
            raise RuntimeError(f"Gross9 split mask drifted: {split}")
        padded_masks[split] = np.asarray(
            (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)), dtype=bool
        )
    return complete_market, padded_masks, padded_events


def _execution_config(
    preregistration: Mapping[str, Any], *, leverage: float, cost_rate: float
) -> ExecutionConfig:
    inputs = preregistration["inputs"]
    return ExecutionConfig(
        input_csv=str(inputs["market"]),
        metrics_csv=str(inputs["open_interest"]),
        funding_csv=str(inputs["funding"]),
        output="/tmp/no_write_rv20_low_participation_shock_reversal_breakeven1_fixed1x.json",
        manifest_output="/tmp/no_write_rv20_low_participation_shock_reversal_breakeven1_fixed1x_manifest.json",
        leverage=float(leverage),
        fee_rate=float(cost_rate),
        slippage_rate=0.0,
    )


def _install_candidate_sleeve() -> None:
    if CANDIDATE not in portfolio.SLEEVES:
        portfolio.SLEEVES = (*portfolio.SLEEVES, CANDIDATE)
    portfolio.FAMILIES[CANDIDATE] = CANDIDATE


def _split_trades(
    trades: Iterable[Trade], start: Any, end: Any
) -> list[Trade]:
    return [
        trade for trade in trades
        if pd.Timestamp(start) <= pd.Timestamp(trade.entry_time) < pd.Timestamp(end)
        and pd.Timestamp(trade.exit_time) < pd.Timestamp(end)
    ]


def candidate_events(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    trades: Sequence[Trade],
    preregistration: Mapping[str, Any],
    *,
    leverage: float,
    cost_rate: float,
    hold_bars: int,
    splits: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Build independent, split-contained candidate subaccount paths."""
    cfg = _execution_config(
        preregistration, leverage=leverage, cost_rate=cost_rate
    )
    events = []
    zeros = np.zeros(len(market), dtype=np.float64)
    selected_splits = tuple(splits) if splits is not None else tuple(portfolio.SPLIT_BOUNDS)
    for split in selected_splits:
        start, configured_end = portfolio.SPLIT_BOUNDS[split]
        contained = _split_trades(trades, start, configured_end)
        dates = pd.DatetimeIndex(pd.to_datetime(market["date"]))
        available_end = dates[-1] + pd.Timedelta("5min")
        end = min(pd.Timestamp(configured_end), available_end)
        if not contained:
            events.append({
                "split": split, "sleeve": CANDIDATE, "side": "mixed",
                "signal_pos": int(dates.searchsorted(pd.Timestamp(start))),
                "date": str(pd.Timestamp(start)),
                "ret": zeros.copy(), "adv": zeros.copy(), "fav": zeros.copy(),
                "low": zeros.copy(), "high": zeros.copy(), "trade_count": 0,
                "win_count": 0, "entry_positions": [],
            })
            continue
        path = subaccount_bar_path(
            market, funding, contained, cfg,
            start=str(start), end=str(end),
            hold_bars=lambda _trade: int(hold_bars),
        )
        event = portfolio.path_event(
            market, path, split=split, sleeve=CANDIDATE,
            trades=contained,
        )
        events.append(event)
    return events


def same_gross_weights(weight: float) -> tuple[dict[str, float], dict[str, float]]:
    combined = {**BASELINE_WEIGHTS, CANDIDATE: float(weight)}
    comparator = {
        sleeve: value * (BASELINE_GROSS + float(weight)) / BASELINE_GROSS
        for sleeve, value in BASELINE_WEIGHTS.items()
    }
    return combined, comparator


def _portfolio_metric(
    arrays: Mapping[str, Mapping[str, Any]], split: str,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    start, end = {
        "train": TRAIN,
        "test2024": CONFIRMATION,
        "eval2025": FUTURE_WINDOWS["2025"],
        "ytd2026": FUTURE_WINDOWS["2026h1"],
    }[split]
    return portfolio.strict_metric(
        arrays[split], years_between(start, end), dict(weights)
    )


def _concat_arrays(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return gross9.concat_array_data(parts)


def _same_gross_row(
    weight: float,
    normal_arrays: Mapping[str, Mapping[str, Any]],
    stress_arrays: Mapping[str, Mapping[str, Any]],
    baseline_arrays: Mapping[str, Mapping[str, Any]],
    split: str,
) -> dict[str, Any]:
    combined_weights, comparator_weights = same_gross_weights(weight)
    baseline = _portfolio_metric(baseline_arrays, split, BASELINE_WEIGHTS)
    combined = _portfolio_metric(normal_arrays, split, combined_weights)
    comparator = _portfolio_metric(baseline_arrays, split, comparator_weights)
    standalone = _portfolio_metric(normal_arrays, split, {CANDIDATE: 1.0})
    standalone_stress = _portfolio_metric(stress_arrays, split, {CANDIDATE: 1.0})
    return {
        "weight": float(weight), "baseline": baseline, "combined": combined,
        "same_gross_comparator": comparator, "standalone": standalone,
        "standalone_stress": standalone_stress,
        "same_gross_ratio_improvement": float(
            combined["cagr_to_strict_mdd"] - comparator["cagr_to_strict_mdd"]
        ),
        "strict_mdd_worsening_pct": float(
            combined["strict_mdd_pct"] - baseline["strict_mdd_pct"]
        ),
    }


def _control_report(
    candidate: Sequence[Trade], control: Sequence[Trade], *, start: Any, end: Any,
    leverage: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, cost in (("full_calendar", ENTRY_COST), ("rv20_stress", STRESS_COST)):
        candidate_metric = strict_metrics(
            candidate, start=start, end=end, cost_rate=cost, leverage=leverage
        )
        control_metric = strict_metrics(
            control, start=start, end=end, cost_rate=cost, leverage=leverage
        )
        output[label] = {
            "candidate": candidate_metric,
            "control": control_metric,
            **control_decomposition(candidate_metric, control_metric),
        }
    return output


def residual_report_passes(report: Mapping[str, Mapping[str, Any]]) -> bool:
    """Require positive normal and stress residuals for the same window."""
    return bool(
        report["full_calendar"]["log_return_residual"] > 0.0
        and report["rv20_stress"]["log_return_residual"] > 0.0
    )


def require_exact_train_replay(
    preregistration: Mapping[str, Any],
    normal: Mapping[str, Any],
    stress: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the fixed 1x replay exactly matches the preregistration."""
    expected = dict(preregistration["promotion_gates"]["train_replay"])
    observed = {
        "absolute_return_pct": normal["absolute_return_pct"],
        "active_weeks": normal["active_weeks"],
        "cagr_pct": normal["cagr_pct"],
        "cagr_to_strict_mdd": normal["cagr_to_strict_mdd"],
        "stress_absolute_return_pct": stress["absolute_return_pct"],
        "strict_mdd_pct": normal["strict_mdd_pct"],
        "trades": normal["trades"],
    }
    if observed != expected:
        raise RuntimeError(f"fixed train replay drifted: {observed} != {expected}")
    return observed


def _market_before(market: pd.DataFrame, end: Any) -> pd.DataFrame:
    dates = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    return market.loc[dates < pd.Timestamp(end)].reset_index(drop=True)


def evaluate_frozen_protocol(
    preregistration: Mapping[str, Any],
    market: pd.DataFrame,
    funding: pd.DataFrame,
) -> dict[str, Any]:
    """Replay one fixed candidate, select on 2024, then conditionally veto."""
    leverage = FIXED_LEVERAGE
    top_cell = FROZEN_CELL
    pre2025_market = _market_before(market, CONFIRMATION[1])
    pre2025_funding = funding.loc[
        pd.to_datetime(funding["date"]) < CONFIRMATION[1]
    ].reset_index(drop=True)
    pre2025_daily = build_rv20_daily_clock(pre2025_market)
    pre2025_hourly = attach_causal_thresholds(
        build_hourly_features(pre2025_market), pre2025_daily, (top_cell,)
    )
    train_trades = execute_schedule_window(
        pre2025_market, pre2025_funding, pre2025_hourly, top_cell,
        start=TRAIN[0], end=TRAIN[1], leverage=leverage,
    )
    train_normal = strict_metrics(
        train_trades, start=TRAIN[0], end=TRAIN[1], cost_rate=ENTRY_COST,
        leverage=leverage,
    )
    train_stress = strict_metrics(
        train_trades, start=TRAIN[0], end=TRAIN[1], cost_rate=STRESS_COST,
        leverage=leverage,
    )
    train_replay = require_exact_train_replay(
        preregistration, train_normal, train_stress
    )
    confirmation_trades = execute_schedule_window(
        pre2025_market, pre2025_funding, pre2025_hourly, top_cell,
        start=CONFIRMATION[0], end=CONFIRMATION[1], leverage=leverage,
    )
    selection_trades = sorted(
        [*train_trades, *confirmation_trades],
        key=lambda trade: (trade.entry_position, trade.exit_position),
    )
    cfg = gross9.Config(
        market_csv=AUTHORITATIVE_GROSS9_MARKET,
        market_with_oi_csv=AUTHORITATIVE_GROSS9_WIDE_OI,
        funding_csv=str(preregistration["inputs"]["funding"]),
        gross9_anchor=str(preregistration["inputs"]["gross9_anchor"]),
    )
    gross_market, masks, baseline_events, source_meta = gross9.build_selection_context(cfg)
    gross_market, masks, baseline_events = extend_gross9_context(
        gross_market, masks, baseline_events, pre2025_market,
        split_bounds={name: portfolio.SPLIT_BOUNDS[name] for name in masks},
    )
    _install_candidate_sleeve()
    baseline_arrays = portfolio.split_arrays(baseline_events, gross_market, masks)
    authoritative = gross9.validate_authoritative_gross9(
        cfg, baseline_arrays, tuple(masks)
    )
    normal_events = list(baseline_events) + candidate_events(
        pre2025_market, pre2025_funding, selection_trades, preregistration,
        leverage=leverage, cost_rate=ENTRY_COST,
        hold_bars=top_cell.max_hold_hours * 12,
        splits=masks,
    )
    stress_events = list(baseline_events) + candidate_events(
        pre2025_market, pre2025_funding, selection_trades, preregistration,
        leverage=leverage, cost_rate=STRESS_COST,
        hold_bars=top_cell.max_hold_hours * 12,
        splits=masks,
    )
    normal_arrays = portfolio.split_arrays(normal_events, pre2025_market, masks)
    stress_arrays = portfolio.split_arrays(stress_events, pre2025_market, masks)
    jaccard = max_per_sleeve_entry_jaccard(
        [trade.entry_time for trade in confirmation_trades],
        {
            sleeve: [pre2025_market.iloc[int(pos)]["date"] for pos in baseline_arrays["test2024"]["entry_positions"][sleeve]]
            for sleeve in BASELINE_WEIGHTS
        },
    )
    selection_control = persistent_long_vol_control(
        pre2025_market, pre2025_funding, pre2025_daily,
        regime_quantile=0.75, leverage=leverage, cost_rate=ENTRY_COST,
    )
    selection_residuals = {
        "train": _control_report(
            selection_trades, selection_control, start=TRAIN[0], end=TRAIN[1],
            leverage=leverage,
        ),
        "2024": _control_report(
            selection_trades, selection_control,
            start=CONFIRMATION[0], end=CONFIRMATION[1], leverage=leverage,
        ),
    }
    confirmation_gate = preregistration["promotion_gates"]["confirmation_2024"]
    weight_rows = []
    for weight in map(float, preregistration["selection"]["portfolio_weight_grid"]):
        row = _same_gross_row(
            weight, normal_arrays, stress_arrays, baseline_arrays, "test2024"
        )
        standalone = row["standalone"]
        row["checks"] = {
            "standalone_cagr": standalone["cagr_pct"] >= confirmation_gate["standalone_cagr_pct_min"],
            "standalone_ratio": standalone["cagr_to_strict_mdd"] >= confirmation_gate["cagr_to_strict_mdd_min"],
            "standalone_mdd": standalone["strict_mdd_pct"] <= confirmation_gate["strict_mdd_pct_max"],
            "candidate_trades": standalone["trades"] >= confirmation_gate["candidate_trades_min"],
            "stress_positive": row["standalone_stress"]["absolute_return_pct"] > 0.0,
            "same_gross_improvement": row["same_gross_ratio_improvement"] >= confirmation_gate["same_gross_ratio_improvement_min"],
            "mdd_no_worsening": row["strict_mdd_worsening_pct"] <= confirmation_gate["strict_mdd_worsening_pct_max"],
            "entry_jaccard": jaccard["maximum"] <= confirmation_gate["max_entry_jaccard"],
            "persistent_long_vol_train": residual_report_passes(selection_residuals["train"]),
            "persistent_long_vol_2024": residual_report_passes(selection_residuals["2024"]),
        }
        row["pass"] = bool(all(row["checks"].values()))
        weight_rows.append(row)
    selected_weight = select_weight(weight_rows)
    if selected_weight is None:
        return {
            "schema": "rllm.rv20-low-participation-shock-reversal-breakeven1-fixed1x.evaluation.v1",
            "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
            "tested_structural_cells": 1,
            "frozen_structural_top1": {
                "cell": asdict(top_cell), "train": train_normal,
                "train_stress": train_stress, "train_replay_exact": True,
            },
            "fixed_leverage": leverage, "train_replay": train_replay,
            "gross9_selection": {"authoritative": authoritative, "source_meta": source_meta},
            "entry_jaccard_2024": jaccard, "weight_rows_2024": weight_rows,
            "selected_weight": None, "selection_persistent_long_vol": selection_residuals,
            "future_veto": None, "future_opened": False,
            "future_used_for_ranking": False, "repair_attempted": False,
            "rerank_attempted": False, "verdict": "REJECT_NO_2024_WEIGHT",
        }

    # This is the sole transition that opens post-2024 candidate/Gross9 values.
    weight = float(selected_weight["weight"])
    full_daily = build_rv20_daily_clock(market)
    full_hourly = attach_causal_thresholds(
        build_hourly_features(market), full_daily, (top_cell,)
    )
    future_schedules = {
        name: execute_schedule_window(
            market, funding, full_hourly, top_cell, start=bounds[0], end=bounds[1],
            leverage=leverage,
        )
        for name, bounds in FUTURE_WINDOWS.items()
        if name != "2025_2026h1"
    }
    all_trades = sorted(
        [*selection_trades, *future_schedules["2025"], *future_schedules["2026h1"]],
        key=lambda trade: (trade.entry_position, trade.exit_position),
    )
    full_gross_market, full_masks, full_baseline_events, full_source_meta = (
        gross9.build_full_context(cfg)
    )
    full_gross_market, full_masks, full_baseline_events = extend_gross9_context(
        full_gross_market, full_masks, full_baseline_events, market
    )
    full_baseline_arrays = portfolio.split_arrays(
        full_baseline_events, full_gross_market, full_masks
    )
    full_authoritative = gross9.validate_authoritative_gross9(
        cfg, full_baseline_arrays, tuple(portfolio.SPLIT_BOUNDS)
    )
    full_normal_events = list(full_baseline_events) + candidate_events(
        market, funding, all_trades, preregistration, leverage=leverage,
        cost_rate=ENTRY_COST, hold_bars=top_cell.max_hold_hours * 12,
    )
    full_stress_events = list(full_baseline_events) + candidate_events(
        market, funding, all_trades, preregistration, leverage=leverage,
        cost_rate=STRESS_COST, hold_bars=top_cell.max_hold_hours * 12,
    )
    full_normal_arrays = portfolio.split_arrays(full_normal_events, market, full_masks)
    full_stress_arrays = portfolio.split_arrays(full_stress_events, market, full_masks)
    future_rows: dict[str, Any] = {}
    each_gate = preregistration["promotion_gates"]["future_each_window"]
    split_names = {"2025": "eval2025", "2026h1": "ytd2026"}
    for name, split in split_names.items():
        row = _same_gross_row(
            weight, full_normal_arrays, full_stress_arrays,
            full_baseline_arrays, split
        )
        row["checks"] = {
            "standalone_positive": row["standalone"]["absolute_return_pct"] > 0.0,
            "stress_positive": row["standalone_stress"]["absolute_return_pct"] > 0.0,
            "same_gross_improvement": row["same_gross_ratio_improvement"] >= each_gate["same_gross_ratio_improvement_min"],
            "mdd_no_worsening": row["strict_mdd_worsening_pct"] <= each_gate["strict_mdd_worsening_pct_max"],
        }
        row["pass"] = bool(all(row["checks"].values()))
        future_rows[name] = row
    combined_weights, comparator_weights = same_gross_weights(weight)
    normal_combined = _concat_arrays([full_normal_arrays["eval2025"], full_normal_arrays["ytd2026"]])
    stress_combined = _concat_arrays([full_stress_arrays["eval2025"], full_stress_arrays["ytd2026"]])
    base_combined = _concat_arrays([full_baseline_arrays["eval2025"], full_baseline_arrays["ytd2026"]])
    start, end = FUTURE_WINDOWS["2025_2026h1"]
    combined = portfolio.strict_metric(normal_combined, years_between(start, end), combined_weights)
    comparator = portfolio.strict_metric(base_combined, years_between(start, end), comparator_weights)
    baseline = portfolio.strict_metric(base_combined, years_between(start, end), BASELINE_WEIGHTS)
    standalone = portfolio.strict_metric(normal_combined, years_between(start, end), {CANDIDATE: 1.0})
    standalone_stress = portfolio.strict_metric(stress_combined, years_between(start, end), {CANDIDATE: 1.0})
    statistics = gross9.paired_statistics(
        gross9.paired_weekly_effects(normal_combined, combined_weights, comparator_weights)
    )
    gate = preregistration["promotion_gates"]["combined_2025_2026h1"]
    improvement = combined["cagr_to_strict_mdd"] - comparator["cagr_to_strict_mdd"]
    reduction = baseline["strict_mdd_pct"] - combined["strict_mdd_pct"]
    retention = combined["absolute_return_pct"] / baseline["absolute_return_pct"]
    combined_checks = {
        "candidate_trades": standalone["trades"] >= gate["candidate_trades_min"],
        "active_weeks": statistics["active_weeks"] >= gate["active_weeks_min"],
        "return_retention": retention >= gate["return_retention_vs_unscaled_gross9_min"],
        "same_gross_improvement": improvement >= gate["same_gross_ratio_improvement_min"],
        "mdd_reduction": reduction >= gate["strict_mdd_reduction_pct_min"],
        "sign_flip": statistics["sign_flip_pvalue"] <= gate["sign_flip_pvalue_max"],
        "bootstrap": statistics["bootstrap_90pct_lower_mean"] > 0.0,
    }
    future_rows["2025_2026h1"] = {
        "baseline": baseline, "combined": combined,
        "same_gross_comparator": comparator, "standalone": standalone,
        "standalone_stress": standalone_stress,
        "same_gross_ratio_improvement": improvement,
        "strict_mdd_reduction_pct": reduction,
        "return_retention_vs_unscaled_gross9": retention,
        "paired_weekly": statistics, "checks": combined_checks,
        "pass": bool(all(combined_checks.values())),
    }
    full_control = persistent_long_vol_control(
        market, funding, full_daily, regime_quantile=0.75,
        leverage=leverage, cost_rate=ENTRY_COST,
    )
    residuals = {
        **selection_residuals,
        "2025": _control_report(all_trades, full_control, start=FUTURE_WINDOWS["2025"][0], end=FUTURE_WINDOWS["2025"][1], leverage=leverage),
        "2026h1": _control_report(all_trades, full_control, start=FUTURE_WINDOWS["2026h1"][0], end=FUTURE_WINDOWS["2026h1"][1], leverage=leverage),
        "2025_2026h1": _control_report(all_trades, full_control, start=start, end=end, leverage=leverage),
    }
    residual_checks = {
        name: {
            "full_calendar_positive": report["full_calendar"]["log_return_residual"] > 0.0,
            "rv20_stress_positive": report["rv20_stress"]["log_return_residual"] > 0.0,
        }
        for name, report in residuals.items()
    }
    checks = {
        **{f"future_{name}": bool(row["pass"]) for name, row in future_rows.items()},
        **{f"persistent_long_vol_{name}": residual_report_passes(residuals[name]) for name in ("2025", "2026h1", "2025_2026h1")},
    }
    future = frozen_future_veto(
        {"cell": asdict(top_cell), "leverage": leverage, "weight": weight}, checks
    )
    future["windows"] = future_rows
    future["persistent_long_vol"] = {"reports": residuals, "checks": residual_checks}
    return {
        "schema": "rllm.rv20-low-participation-shock-reversal-breakeven1-fixed1x.evaluation.v1",
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "tested_structural_cells": 1,
        "frozen_structural_top1": {
            "cell": asdict(top_cell), "train": train_normal,
            "train_stress": train_stress, "train_replay_exact": True,
        },
        "fixed_leverage": leverage, "train_replay": train_replay,
        "gross9_selection": {"authoritative": authoritative, "source_meta": source_meta},
        "gross9_future": {"authoritative": full_authoritative, "source_meta": full_source_meta},
        "entry_jaccard_2024": jaccard, "weight_rows_2024": weight_rows,
        "selected_weight": selected_weight,
        "selection_persistent_long_vol": selection_residuals,
        "future_veto": future, "future_opened": True,
        "future_used_for_ranking": False, "repair_attempted": False,
        "rerank_attempted": False, "verdict": future["action"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    preregistration = load_preregistration(args.preregistration)
    input_identity = authenticate_inputs(preregistration)
    if args.validate_only:
        print(canonical_json({
            "grid_cardinality": len(frozen_grid(preregistration)),
            "inputs": input_identity,
            "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
            "status": "VALID",
        }), end="")
        return
    consume_one_shot(ATTEMPT, Path(args.output), Path(args.report))
    market, funding = load_inputs(preregistration)
    payload = evaluate_frozen_protocol(preregistration, market, funding)
    payload["input_identity"] = input_identity
    write_outputs(payload, Path(args.output), Path(args.report))
    print(canonical_json({
        "output": str(args.output), "report": str(args.report),
        "verdict": payload["verdict"],
    }), end="")


if __name__ == "__main__":
    main()
