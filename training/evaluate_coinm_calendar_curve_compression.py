"""Frozen strict evaluator for the COIN-M calendar-curve compression alpha."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.preregister_coinm_calendar_curve_compression import (
    CANDIDATE,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_SHA256,
    FIT_END,
    FIT_START,
    SELECTION_END,
    build_signal_state,
    canonical_hash,
    candidate_clock,
    load_source,
    nonoverlapping_schedule,
    schedule_window,
    support_gates,
)


SUPPORT_COMMIT = "68b8c159845c18c533fd3f90fdb8a854af90a1a7"
STATIC_INPUT_SHA256 = {
    "training/preregister_coinm_calendar_curve_compression.py": (
        "a3f6ce9991c2b9a63c0c8a79c70bf6bf0005d1972afff760bb3317e2bf4d135d"
    ),
    "docs/coinm-calendar-curve-compression-preregistration-2026-07-19.md": (
        "adaf26406374aca150831cc1430b0e0b75d9b9263e965df5a5ee78dcb546aca1"
    ),
    "docs/coinm-calendar-curve-compression-support-freeze-2026-07-19.md": (
        "e003873c1f183cd62aae0242120fc1bf0a4bb59b864667e7ebdb0265c349fd5a"
    ),
    "results/coinm_calendar_curve_compression_support_2026-07-19.json": (
        "1377015da4f12bb90441d5f3f3bfbf1788ca0416b75bb87ec3b62dc25d6b0dfc"
    ),
    "training/evaluate_metaorder_fragmentation_impact_curvature.py": (
        "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
    ),
}
SUPPORT_RESULT = Path(
    "results/coinm_calendar_curve_compression_support_2026-07-19.json"
)
EVALUATOR_SOURCE = Path("training/evaluate_coinm_calendar_curve_compression.py")
EVALUATOR_FREEZE = Path(
    "results/coinm_calendar_curve_compression_evaluator_freeze_2026-07-19.json"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "fit": ("2020-07-15", "2023-01-01"),
    "fit_2020_partial": ("2020-07-15", "2021-01-01"),
    "fit_2021h1": ("2021-01-01", "2021-07-01"),
    "fit_2021h2": ("2021-07-01", "2022-01-01"),
    "fit_2022h1": ("2022-01-01", "2022-07-01"),
    "fit_2022h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023h1": ("2023-01-01", "2023-07-01"),
    "select_2023h2": ("2023-07-01", "2024-01-01"),
}
FULL_WINDOWS = ("fit", "select_2023")
SELECT_HALVES = ("select_2023h1", "select_2023h2")
EVALUATOR_SEALED_WINDOWS = (
    "fit",
    "select_2023",
    "2024",
    "2025",
    "2026_ytd",
)


@dataclass(frozen=True)
class EvaluationConfig:
    source_csv: str = (
        "data/binance_coinm_quarterly_strip_pre2024_v2/"
        "BTCUSD_front_next_quarterly_5m_20200701T0000_20231231T2350.csv.gz"
    )
    manifest_json: str = (
        "data/binance_coinm_quarterly_strip_pre2024_v2/build_manifest.json"
    )
    output: str = (
        "results/coinm_calendar_curve_compression_pre2024_selection_2026-07-19.json"
    )
    freeze_output: str = str(EVALUATOR_FREEZE)
    contract_face_usd: float = 100.0
    cost_rate_per_leg_per_side: float = 0.0006
    stress_cost_rate_per_leg_per_side: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_719
    delay_1h_bars: int = 12
    delay_24h_bars: int = 288


@dataclass(frozen=True)
class PairTrade:
    confirmation_position: int
    entry_position: int
    exit_position: int
    front_symbol: str
    next_symbol: str
    front_side: int
    next_side: int
    front_entry: float
    front_exit: float
    next_entry: float
    next_exit: float
    front_highs: tuple[float, ...]
    front_lows: tuple[float, ...]
    next_highs: tuple[float, ...]
    next_lows: tuple[float, ...]
    entry_date: str

    @property
    def gross_curve_return(self) -> float:
        return pair_curve_return(
            self.front_entry,
            self.front_exit,
            self.front_side,
            self.next_entry,
            self.next_exit,
            self.next_side,
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if bool(pd.isna(parsed)):
        raise ValueError("timestamp cannot be NaT")
    return cast(pd.Timestamp, parsed)


def _write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _stable_artifact_hash(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "freeze_hash", "result_hash"}
    }
    return canonical_hash(stable)


def _require_canonical_config(cfg: EvaluationConfig) -> None:
    if asdict(cfg) != asdict(EvaluationConfig()):
        raise ValueError("all evaluator paths and protocol parameters are frozen")
    if CANDIDATE.total_gross != 0.5 or CANDIDATE.hold_bars != 144:
        raise ValueError("candidate sizing or hold changed after preregistration")


def _verify_static_dependencies() -> dict[str, Any]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen dependency changed: {path}")
    support = _read_json(SUPPORT_RESULT)
    stable_support = {
        key: value
        for key, value in support.items()
        if key not in {"created_at", "manifest_hash"}
    }
    if support.get("manifest_hash") != canonical_hash(stable_support):
        raise ValueError("support manifest hash changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("support stage opened outcomes")
    if support.get("post_entry_returns_computed") is not False:
        raise ValueError("support stage computed post-entry returns")
    if support.get("passes_support") is not True:
        raise ValueError("support stage did not pass")
    if support.get("candidate") != asdict(CANDIDATE):
        raise ValueError("candidate changed after support freeze")
    source = support.get("source", {})
    if source.get("csv_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("support source identity changed")
    if source.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("support source manifest identity changed")
    if _sha256(source["csv"]) != EXPECTED_SOURCE_SHA256:
        raise ValueError("support source bytes changed")
    if _sha256(source["manifest"]) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("support source manifest bytes changed")
    return support


def freeze_evaluator(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_config(cfg)
    support = _verify_static_dependencies()
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists; evaluator cannot be frozen")
    if Path(cfg.freeze_output).exists():
        raise ValueError("evaluator freeze already exists and cannot be replaced")
    if _sha256(cfg.source_csv) != EXPECTED_SOURCE_SHA256:
        raise ValueError("outcome source identity changed")
    if _sha256(cfg.manifest_json) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("outcome source manifest changed")
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "support_commit": SUPPORT_COMMIT,
        "support_manifest_hash": support["manifest_hash"],
        "support_artifact_sha256": _sha256(SUPPORT_RESULT),
        "evaluation_source": str(EVALUATOR_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": list(EVALUATOR_SEALED_WINDOWS),
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    report["freeze_hash"] = _stable_artifact_hash(report)
    _write_json_exclusive(cfg.freeze_output, report)
    return report


def verify_evaluator_freeze(
    cfg: EvaluationConfig,
    support: dict[str, Any],
) -> dict[str, Any]:
    _require_canonical_config(cfg)
    freeze = _read_json(cfg.freeze_output)
    if freeze.get("freeze_hash") != _stable_artifact_hash(freeze):
        raise ValueError("evaluator freeze hash changed")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("evaluator source changed after freeze")
    if freeze.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("evaluator freeze support commit changed")
    if freeze.get("support_manifest_hash") != support.get("manifest_hash"):
        raise ValueError("evaluator freeze support manifest changed")
    if freeze.get("support_artifact_sha256") != _sha256(SUPPORT_RESULT):
        raise ValueError("evaluator freeze support artifact changed")
    if freeze.get("config") != asdict(cfg):
        raise ValueError("evaluation config changed after freeze")
    if freeze.get("opened_windows") != [] or freeze.get("mutable_parameters") != []:
        raise ValueError("evaluator freeze is not sealed")
    if freeze.get("candidate_returns_computed_before_freeze") is not False:
        raise ValueError("candidate returns were computed before evaluator freeze")
    if freeze.get("simulation_run") is not False:
        raise ValueError("evaluator freeze ran a simulation")
    if freeze.get("sealed_windows") != list(EVALUATOR_SEALED_WINDOWS):
        raise ValueError("evaluator freeze sealed windows changed")
    if _sha256(cfg.source_csv) != EXPECTED_SOURCE_SHA256:
        raise ValueError("frozen outcome source changed")
    if _sha256(cfg.manifest_json) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("frozen outcome manifest changed")
    return freeze


def _rebuild_schedules(
    source: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    state = build_signal_state(source)
    active, side = candidate_clock(source, state)
    fit = nonoverlapping_schedule(
        source, state, active, side, start=FIT_START, end=FIT_END
    )
    selection = nonoverlapping_schedule(
        source, state, active, side, start=FIT_END, end=SELECTION_END
    )
    schedules = {
        "fit": fit,
        "fit_2020_partial": schedule_window(
            fit, FIT_START, _timestamp("2021-01-01")
        ),
        "fit_2021": schedule_window(
            fit, _timestamp("2021-01-01"), _timestamp("2022-01-01")
        ),
        "fit_2022": schedule_window(
            fit, _timestamp("2022-01-01"), FIT_END
        ),
        "fit_2021h1": schedule_window(
            fit, _timestamp("2021-01-01"), _timestamp("2021-07-01")
        ),
        "fit_2021h2": schedule_window(
            fit, _timestamp("2021-07-01"), _timestamp("2022-01-01")
        ),
        "fit_2022h1": schedule_window(
            fit, _timestamp("2022-01-01"), _timestamp("2022-07-01")
        ),
        "fit_2022h2": schedule_window(
            fit, _timestamp("2022-07-01"), FIT_END
        ),
        "select_2023": selection,
        "select_2023h1": schedule_window(
            selection, FIT_END, _timestamp("2023-07-01")
        ),
        "select_2023h2": schedule_window(
            selection, _timestamp("2023-07-01"), SELECTION_END
        ),
    }
    schedules["select_2023_h1"] = schedules["select_2023h1"]
    schedules["select_2023_h2"] = schedules["select_2023h2"]
    return schedules, active, side


def _verify_rebuilt_support(
    source: pd.DataFrame,
    schedules: dict[str, pd.DataFrame],
    support: dict[str, Any],
) -> None:
    state = build_signal_state(source)
    active, _ = candidate_clock(source, state)
    rebuilt_support = {
        "raw_confirmation_events": int(np.asarray(active, dtype=bool).sum()),
        "windows": {},
        "schedule_hashes": {},
    }
    from training.preregister_coinm_calendar_curve_compression import period_support

    for name in support["support"]["windows"]:
        schedule = schedules[name]
        rebuilt_support["windows"][name] = period_support(schedule)
        rebuilt_support["schedule_hashes"][name] = canonical_hash(
            schedule.to_dict(orient="records")
        )
    if rebuilt_support != support["support"]:
        raise ValueError("frozen support or schedule hashes changed")
    if support_gates(rebuilt_support) != support["support_gates"]:
        raise ValueError("frozen support gates changed")


def _load_outcomes(cfg: EvaluationConfig) -> pd.DataFrame:
    if _sha256(cfg.source_csv) != EXPECTED_SOURCE_SHA256:
        raise ValueError("outcome source identity changed")
    columns = [
        "signal_bar_open_utc",
        "front_symbol",
        "next_symbol",
        "front_open",
        "front_high",
        "front_low",
        "front_close",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
    ]
    outcome = pd.read_csv(
        cfg.source_csv,
        compression="infer",
        usecols=lambda column: str(column) in columns,
    )
    outcome["signal_bar_open_utc"] = (
        pd.to_datetime(outcome["signal_bar_open_utc"], utc=True, errors="raise")
        .dt.tz_convert(None)
    )
    dates = cast(pd.Series, outcome["signal_bar_open_utc"])
    if (
        outcome.empty
        or dates.duplicated().any()
        or dates.iloc[0] != _timestamp("2020-07-01")
        or dates.iloc[-1] != _timestamp("2023-12-31 23:50")
    ):
        raise ValueError("outcome source is not the exact sealed interval")
    expected = pd.Series(
        pd.date_range(dates.iloc[0], dates.iloc[-1], freq="5min"),
        name="signal_bar_open_utc",
    )
    if not dates.equals(expected):
        raise ValueError("outcome source is not a complete five-minute grid")
    for leg in ("front", "next"):
        price_columns = [f"{leg}_{field}" for field in ("open", "high", "low", "close")]
        prices = outcome[price_columns].apply(pd.to_numeric, errors="coerce")
        finite_count = np.isfinite(prices.to_numpy(float)).sum(axis=1)
        if ((finite_count != 0) & (finite_count != 4)).any():
            raise ValueError(f"partial {leg} OHLC row in outcome source")
        complete = finite_count == 4
        if (prices.loc[complete] <= 0.0).any().any():
            raise ValueError(f"non-positive {leg} outcome price")
        high = prices.loc[complete, f"{leg}_high"]
        low = prices.loc[complete, f"{leg}_low"]
        endpoints = prices.loc[complete, [f"{leg}_open", f"{leg}_close"]]
        if (
            (high < endpoints.max(axis=1)).any()
            or (low > endpoints.min(axis=1)).any()
            or (high < low).any()
        ):
            raise ValueError(f"{leg} outcome source violates OHLC invariants")
        outcome.loc[:, price_columns] = prices
    return outcome


def inverse_usd_return(entry_price: float, mark_price: float, side: int) -> float:
    """Exact USD return per fixed USD face for an inverse contract."""
    if entry_price <= 0.0 or mark_price <= 0.0:
        raise ValueError("inverse prices must be positive")
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    coin_pnl_per_face = side * (1.0 / entry_price - 1.0 / mark_price)
    return float(coin_pnl_per_face * mark_price)


def pair_curve_return(
    front_entry: float,
    front_mark: float,
    front_side: int,
    next_entry: float,
    next_mark: float,
    next_side: int,
) -> float:
    if front_side != -next_side:
        raise ValueError("calendar-spread legs must have opposite sides")
    return float(
        inverse_usd_return(front_entry, front_mark, front_side)
        + inverse_usd_return(next_entry, next_mark, next_side)
    )


def _build_trades(
    outcome: pd.DataFrame,
    positions: dict[pd.Timestamp, int],
    schedule: pd.DataFrame,
    *,
    flip: bool = False,
) -> list[PairTrade]:
    trades: list[PairTrade] = []
    dates = cast(pd.Series, outcome["signal_bar_open_utc"])
    for row in schedule.to_dict(orient="records"):
        confirmation_time = _timestamp(row["confirmation_bar_open"])
        confirmation_position = positions.get(confirmation_time)
        if confirmation_position is None:
            raise ValueError("frozen confirmation time is absent from outcome grid")
        entry_position = confirmation_position + 1
        exit_position = entry_position + CANDIDATE.hold_bars
        if exit_position >= len(outcome):
            raise ValueError("frozen pair trade exceeds outcome horizon")
        if dates.iloc[entry_position] != _timestamp(row["entry_time"]):
            raise ValueError("pair entry differs from frozen next-open entry")
        if dates.iloc[exit_position] != _timestamp(row["exit_time"]):
            raise ValueError("pair exit differs from frozen fixed hold")
        front_symbol = str(row["front_symbol"])
        next_symbol = str(row["next_symbol"])
        held = outcome.iloc[entry_position : exit_position + 1]
        if not held["front_symbol"].astype(str).eq(front_symbol).all():
            raise ValueError("front leg crosses a contract transition")
        if not held["next_symbol"].astype(str).eq(next_symbol).all():
            raise ValueError("next leg crosses a contract transition")
        front_entry = float(outcome.iloc[entry_position]["front_open"])
        front_exit = float(outcome.iloc[exit_position]["front_open"])
        next_entry = float(outcome.iloc[entry_position]["next_open"])
        next_exit = float(outcome.iloc[exit_position]["next_open"])
        held_bars = outcome.iloc[entry_position:exit_position]
        arrays = {
            name: held_bars[name].to_numpy(float)
            for name in ("front_high", "front_low", "next_high", "next_low")
        }
        required = np.concatenate(
            [
                np.asarray([front_entry, front_exit, next_entry, next_exit]),
                *arrays.values(),
            ]
        )
        if not np.isfinite(required).all() or (required <= 0.0).any():
            raise ValueError("scheduled pair trade crosses missing outcome prices")
        front_side = int(row["front_side"])
        next_side = int(row["next_side"])
        if flip:
            front_side = -front_side
            next_side = -next_side
        if front_side != -next_side or front_side not in (-1, 1):
            raise ValueError("frozen pair sides are not opposite")
        trades.append(
            PairTrade(
                confirmation_position=confirmation_position,
                entry_position=entry_position,
                exit_position=exit_position,
                front_symbol=front_symbol,
                next_symbol=next_symbol,
                front_side=front_side,
                next_side=next_side,
                front_entry=front_entry,
                front_exit=front_exit,
                next_entry=next_entry,
                next_exit=next_exit,
                front_highs=tuple(float(value) for value in arrays["front_high"]),
                front_lows=tuple(float(value) for value in arrays["front_low"]),
                next_highs=tuple(float(value) for value in arrays["next_high"]),
                next_lows=tuple(float(value) for value in arrays["next_low"]),
                entry_date=str(dates.iloc[entry_position]),
            )
        )
    return trades


def delayed_schedule(
    schedule: pd.DataFrame,
    source: pd.DataFrame,
    *,
    bars: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Shift the frozen clock uniformly without re-detecting curve events."""
    if bars <= 0:
        raise ValueError("delay bars must be positive")
    dates = cast(pd.Series, source["signal_bar_open_utc"])
    positions = {
        _timestamp(timestamp): position for position, timestamp in enumerate(dates)
    }
    start_time = _timestamp(start)
    end_time = _timestamp(end)
    rows: list[dict[str, Any]] = []
    hold_hours = CANDIDATE.hold_bars / 12.0
    for row in schedule.to_dict(orient="records"):
        original = positions.get(_timestamp(row["confirmation_bar_open"]))
        if original is None:
            raise ValueError("frozen delayed-control confirmation is absent")
        destination = original + bars
        if destination >= len(source):
            continue
        target = source.iloc[destination]
        entry_time = _timestamp(target["trade_earliest_time_utc"])
        exit_time = entry_time + pd.Timedelta(minutes=5 * CANDIDATE.hold_bars)
        delivery_required = 12.0 + hold_hours + 5.0 / 60.0
        same_symbols = (
            str(target["front_symbol"]) == str(row["front_symbol"])
            and str(target["next_symbol"]) == str(row["next_symbol"])
        )
        if (
            entry_time < start_time
            or exit_time >= end_time
            or not bool(target["feature_valid"])
            or not same_symbols
            or float(target["front_hours_to_delivery"]) < delivery_required
            or float(target["next_hours_to_delivery"]) < delivery_required
        ):
            continue
        rows.append(
            {
                "shock_bar_open": str(source.iloc[destination - 1]["signal_bar_open_utc"]),
                "confirmation_bar_open": str(target["signal_bar_open_utc"]),
                "feature_available": str(target["feature_available_time_utc"]),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "front_symbol": str(row["front_symbol"]),
                "next_symbol": str(row["next_symbol"]),
                "front_side": int(row["front_side"]),
                "next_side": int(row["next_side"]),
                "hold_bars": CANDIDATE.hold_bars,
            }
        )
    delayed = pd.DataFrame(rows, columns=schedule.columns)
    if len(delayed) > 1:
        entries = pd.to_datetime(delayed["entry_time"]).to_numpy(dtype="datetime64[ns]")
        exits = pd.to_datetime(delayed["exit_time"]).to_numpy(dtype="datetime64[ns]")
        if (entries[1:] < exits[:-1]).any():
            raise ValueError("uniform delay unexpectedly created overlap")
    return delayed


def _pair_mark_return(
    trade: PairTrade, front_mark: float, next_mark: float
) -> float:
    return pair_curve_return(
        trade.front_entry,
        front_mark,
        trade.front_side,
        trade.next_entry,
        next_mark,
        trade.next_side,
    )


def strict_equity_stats(
    trades: Iterable[PairTrade],
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
    cost_rate: float | None = None,
) -> dict[str, Any]:
    """Full-calendar equity with barwise independent-leg strict extremes."""
    cost = (
        cfg.cost_rate_per_leg_per_side if cost_rate is None else float(cost_rate)
    )
    if cfg.contract_face_usd != 100.0:
        raise ValueError("BTCUSD quarterly face must remain USD 100")
    if not 0.0 <= cost < 1.0 or not 0.0 < CANDIDATE.total_gross <= 10.0:
        raise ValueError("invalid gross or cost")
    leg_weight = CANDIDATE.total_gross / 2.0
    one_side_pair_cost = CANDIDATE.total_gross * cost
    equity = peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_curve_returns: list[float] = []
    gross_account_returns: list[float] = []
    entry_dates: list[str] = []
    next_sides: list[int] = []
    for trade in trades:
        lengths = {
            len(trade.front_highs),
            len(trade.front_lows),
            len(trade.next_highs),
            len(trade.next_lows),
        }
        if lengths != {CANDIDATE.hold_bars}:
            raise ValueError("pair trade does not contain the fixed held path")
        entry_equity = equity
        entry_mark = entry_equity * max(0.0, 1.0 - one_side_pair_cost)
        strict_mdd = max(strict_mdd, 1.0 - entry_mark / peak)
        for bar in range(CANDIDATE.hold_bars):
            if trade.front_side > 0:
                front_favorable = trade.front_highs[bar]
                front_adverse = trade.front_lows[bar]
            else:
                front_favorable = trade.front_lows[bar]
                front_adverse = trade.front_highs[bar]
            if trade.next_side > 0:
                next_favorable = trade.next_highs[bar]
                next_adverse = trade.next_lows[bar]
            else:
                next_favorable = trade.next_lows[bar]
                next_adverse = trade.next_highs[bar]
            favorable_curve = _pair_mark_return(
                trade, front_favorable, next_favorable
            )
            favorable_factor = (
                1.0 + leg_weight * favorable_curve - one_side_pair_cost
            )
            intratrade_peak = max(
                peak, entry_equity * max(0.0, favorable_factor)
            )
            adverse_curve = _pair_mark_return(trade, front_adverse, next_adverse)
            adverse_factor = (
                1.0 + leg_weight * adverse_curve - 2.0 * one_side_pair_cost
            )
            adverse_equity = entry_equity * max(0.0, adverse_factor)
            strict_mdd = max(strict_mdd, 1.0 - adverse_equity / intratrade_peak)
            peak = intratrade_peak
        gross_curve = trade.gross_curve_return
        gross_account = leg_weight * gross_curve
        realized_factor = 1.0 + gross_account - 2.0 * one_side_pair_cost
        equity = entry_equity * max(0.0, realized_factor)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        net_returns.append(equity / entry_equity - 1.0)
        gross_curve_returns.append(gross_curve)
        gross_account_returns.append(gross_account)
        entry_dates.append(trade.entry_date)
        next_sides.append(trade.next_side)
        if equity <= 0.0:
            break
    years = (_timestamp(end) - _timestamp(start)).total_seconds() / (
        365.25 * 86_400.0
    )
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    net = np.asarray(net_returns, dtype=float)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": int(len(net)),
        "long_next": int(sum(side > 0 for side in next_sides)),
        "short_next": int(sum(side < 0 for side in next_sides)),
        "mean_net_bps": float(net.mean() * 10_000.0) if len(net) else 0.0,
        "mean_gross_account_bps": (
            float(np.mean(gross_account_returns) * 10_000.0)
            if gross_account_returns
            else 0.0
        ),
        "mean_gross_curve_compression_bps": (
            float(np.mean(gross_curve_returns) * 10_000.0)
            if gross_curve_returns
            else 0.0
        ),
        "win_rate": float((net > 0.0).mean()) if len(net) else 0.0,
        "wall_clock_years": float(years),
        "net_trade_returns": [float(value) for value in net],
        "entry_dates": entry_dates,
    }


def _public_stats(
    stats: dict[str, Any], *, cluster: bool, cfg: EvaluationConfig
) -> dict[str, Any]:
    output = {
        key: value
        for key, value in stats.items()
        if key not in {"net_trade_returns", "entry_dates"}
    }
    if cluster:
        output["weekly_cluster_sign_flip"] = weekly_cluster_sign_flip(
            stats["net_trade_returns"],
            stats["entry_dates"],
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
    return output


def _stats(
    trades: list[PairTrade],
    window: str,
    cfg: EvaluationConfig,
    *,
    cost_rate: float | None = None,
    cluster: bool = False,
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    raw = strict_equity_stats(
        trades, start=start, end=end, cfg=cfg, cost_rate=cost_rate
    )
    return _public_stats(raw, cluster=cluster, cfg=cfg)


def selection_gates(
    windows: dict[str, dict[str, Any]],
    stress: dict[str, dict[str, Any]],
    direction_flip: dict[str, dict[str, Any]],
    delay_1h: dict[str, dict[str, Any]],
    delay_24h: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    fit = windows["fit"]
    select = windows["select_2023"]
    return {
        "fit_absolute_return_positive": fit["absolute_return_pct"] > 0.0,
        "select_absolute_return_positive": select["absolute_return_pct"] > 0.0,
        "fit_cagr_to_strict_mdd_at_least_3": fit["cagr_to_strict_mdd"] >= 3.0,
        "select_cagr_to_strict_mdd_at_least_3": select["cagr_to_strict_mdd"] >= 3.0,
        "fit_strict_mdd_at_most_15pct": fit["strict_mdd_pct"] <= 15.0,
        "select_strict_mdd_at_most_15pct": select["strict_mdd_pct"] <= 15.0,
        "fit_trades_at_least_150": fit["trades"] >= 150,
        "select_trades_at_least_50": select["trades"] >= 50,
        "each_select_half_positive": min(
            windows[name]["absolute_return_pct"] for name in SELECT_HALVES
        )
        > 0.0,
        "each_select_half_at_least_25_trades": min(
            windows[name]["trades"] for name in SELECT_HALVES
        )
        >= 25,
        "fit_mean_gross_curve_at_least_12bp": fit[
            "mean_gross_curve_compression_bps"
        ]
        >= 12.0,
        "select_mean_gross_curve_at_least_12bp": select[
            "mean_gross_curve_compression_bps"
        ]
        >= 12.0,
        "fit_weekly_cluster_p_below_0_10": fit["weekly_cluster_sign_flip"][
            "p_value_one_sided"
        ]
        < 0.10,
        "select_weekly_cluster_p_below_0_10": select[
            "weekly_cluster_sign_flip"
        ]["p_value_one_sided"]
        < 0.10,
        "fit_10bp_per_leg_side_stress_positive": stress["fit"][
            "absolute_return_pct"
        ]
        > 0.0,
        "select_10bp_per_leg_side_stress_positive": stress["select_2023"][
            "absolute_return_pct"
        ]
        > 0.0,
        "fit_direction_flip_negative": direction_flip["fit"][
            "absolute_return_pct"
        ]
        < 0.0,
        "select_direction_flip_negative": direction_flip["select_2023"][
            "absolute_return_pct"
        ]
        < 0.0,
        "fit_beats_1h_delay": fit["absolute_return_pct"]
        > delay_1h["fit"]["absolute_return_pct"],
        "select_beats_1h_delay": select["absolute_return_pct"]
        > delay_1h["select_2023"]["absolute_return_pct"],
        "fit_beats_24h_delay": fit["absolute_return_pct"]
        > delay_24h["fit"]["absolute_return_pct"],
        "select_beats_24h_delay": select["absolute_return_pct"]
        > delay_24h["select_2023"]["absolute_return_pct"],
    }


def evaluate(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_config(cfg)
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists and cannot be replaced")
    support = _verify_static_dependencies()
    freeze = verify_evaluator_freeze(cfg, support)
    source = load_source(cfg.source_csv)
    schedules, _, _ = _rebuild_schedules(source)
    _verify_rebuilt_support(source, schedules, support)
    outcome = _load_outcomes(cfg)
    if not outcome["signal_bar_open_utc"].equals(source["signal_bar_open_utc"]):
        raise ValueError("support and outcome clocks differ")
    positions = {
        _timestamp(timestamp): position
        for position, timestamp in enumerate(outcome["signal_bar_open_utc"])
    }
    windows: dict[str, dict[str, Any]] = {}
    for name in WINDOWS:
        schedule = schedules[name]
        trades = _build_trades(outcome, positions, schedule)
        windows[name] = _stats(trades, name, cfg, cluster=name in FULL_WINDOWS)

    stress: dict[str, dict[str, Any]] = {}
    direction_flip: dict[str, dict[str, Any]] = {}
    delay_1h: dict[str, dict[str, Any]] = {}
    delay_24h: dict[str, dict[str, Any]] = {}
    control_hashes: dict[str, dict[str, str]] = {"delay_1h": {}, "delay_24h": {}}
    for name in FULL_WINDOWS:
        base_schedule = schedules[name]
        base_trades = _build_trades(outcome, positions, base_schedule)
        flip_trades = _build_trades(outcome, positions, base_schedule, flip=True)
        hour_schedule = delayed_schedule(
            base_schedule,
            source,
            bars=cfg.delay_1h_bars,
            start=WINDOWS[name][0],
            end=WINDOWS[name][1],
        )
        day_schedule = delayed_schedule(
            base_schedule,
            source,
            bars=cfg.delay_24h_bars,
            start=WINDOWS[name][0],
            end=WINDOWS[name][1],
        )
        hour_trades = _build_trades(outcome, positions, hour_schedule)
        day_trades = _build_trades(outcome, positions, day_schedule)
        control_hashes["delay_1h"][name] = canonical_hash(
            hour_schedule.to_dict(orient="records")
        )
        control_hashes["delay_24h"][name] = canonical_hash(
            day_schedule.to_dict(orient="records")
        )
        stress[name] = _stats(
            base_trades,
            name,
            cfg,
            cost_rate=cfg.stress_cost_rate_per_leg_per_side,
        )
        direction_flip[name] = _stats(flip_trades, name, cfg)
        delay_1h[name] = _stats(hour_trades, name, cfg)
        delay_24h[name] = _stats(day_trades, name, cfg)

    gates = selection_gates(windows, stress, direction_flip, delay_1h, delay_24h)
    candidate = {
        "candidate": asdict(CANDIDATE),
        "windows": windows,
        "stress_10bp_per_leg_per_side": stress,
        "direction_flip": direction_flip,
        "delay_1h": delay_1h,
        "delay_24h": delay_24h,
        "control_schedule_hashes": control_hashes,
        "gates": gates,
        "passes_selection": bool(all(gates.values())),
    }
    stable: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "support_commit": SUPPORT_COMMIT,
            "support_manifest_hash": support["manifest_hash"],
            "evaluator_freeze_hash": freeze["freeze_hash"],
            "opened_windows": ["fit", "select_2023"],
            "sealed_windows": ["2024", "2025", "2026_ytd"],
            "full_calendar_cagr": True,
            "strict_mdd": (
                "global/pre-entry HWM; entry cost; each held 5m bar marks both "
                "legs at independent favorable extremes before independent adverse "
                "extremes with hypothetical exit cost; realized two-sided costs"
            ),
            "inverse_ledger": (
                "fractional equal-USD-face inverse legs marked back to USD; no funding"
            ),
            "integer_contract_rounding_modeled": False,
            "btc_collateral_beta_modeled": False,
            "order_book_impact_modeled": False,
            "post_selection_parameter_repair_allowed": False,
        },
        "config": asdict(cfg),
        "candidates_evaluated": 1,
        "candidates_passing": int(candidate["passes_selection"]),
        "winner": candidate if candidate["passes_selection"] else None,
        "advance_to_2024_test": bool(candidate["passes_selection"]),
        "candidate": candidate,
    }
    report = {**stable, "result_hash": _stable_artifact_hash(stable)}
    _write_json_exclusive(cfg.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default=EvaluationConfig.source_csv)
    parser.add_argument("--manifest-json", default=EvaluationConfig.manifest_json)
    parser.add_argument("--output", default=EvaluationConfig.output)
    parser.add_argument("--freeze-output", default=EvaluationConfig.freeze_output)
    parser.add_argument("--freeze-only", action="store_true")
    args = vars(parser.parse_args())
    freeze_only = args.pop("freeze_only")
    cfg = EvaluationConfig(**args)
    report = freeze_evaluator(cfg) if freeze_only else evaluate(cfg)
    keys = (
        ["freeze_hash"]
        if freeze_only
        else ["candidates_evaluated", "candidates_passing", "advance_to_2024_test"]
    )
    print(json.dumps({key: report[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
