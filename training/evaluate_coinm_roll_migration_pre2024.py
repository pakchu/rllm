"""Frozen strict pre-2024 evaluator for COIN-M quarterly roll events."""
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
from training.preregister_coinm_roll_migration_alpha import (
    CANDIDATES,
    Candidate,
    build_signal_state,
    candidate_clock,
    canonical_hash,
    load_source,
    nonoverlapping_schedule,
    windowed_support_summary,
)


SUPPORT_COMMIT = "9a56e01773bccf2e70f0b467977f5e82bf08181e"
STATIC_INPUT_SHA256 = {
    "training/preregister_coinm_roll_migration_alpha.py": (
        "e8872482d6382e51eb0d80400c662ac1c4bd5626a8653b14b6489d8b2df8b3a3"
    ),
    "docs/coinm-roll-migration-preregistration-2026-07-19.md": (
        "fbd4e4fde3c3ba1669050164f7e2df6873dab8aca39d7c679f0da3c723a290c8"
    ),
    "results/coinm_roll_migration_support_2026-07-19.json": (
        "78e2dab97f046fc32b70548c759df5c6dd7125cf42d58381e5b744af550b2dbd"
    ),
    "training/evaluate_metaorder_fragmentation_impact_curvature.py": (
        "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
    ),
}
SOURCE_SHA256 = "d107b6dee3f8d1012110db4744cb36d3e7e7fc36a1f93cc17f5ce4c92ab461f3"
MANIFEST_SHA256 = "cdb1ea8f175b0edebf36373aa3231de0a9026413ee7bb3bf4ee602b5abe2db2e"
SUPPORT_RESULT = Path("results/coinm_roll_migration_support_2026-07-19.json")
EVALUATOR_SOURCE = Path("training/evaluate_coinm_roll_migration_pre2024.py")
EVALUATOR_FREEZE = Path(
    "results/coinm_roll_migration_evaluator_freeze_2026-07-19.json"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "fit": ("2020-07-15", "2023-01-01"),
    "fit_2020h2": ("2020-07-15", "2021-01-01"),
    "fit_2021h1": ("2021-01-01", "2021-07-01"),
    "fit_2021h2": ("2021-07-01", "2022-01-01"),
    "fit_2022h1": ("2022-01-01", "2022-07-01"),
    "fit_2022h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023h1": ("2023-01-01", "2023-07-01"),
    "select_2023h2": ("2023-07-01", "2024-01-01"),
}
FULL_WINDOWS = ("fit", "select_2023")
FIT_HALVES = (
    "fit_2020h2",
    "fit_2021h1",
    "fit_2021h2",
    "fit_2022h1",
    "fit_2022h2",
)
SELECT_HALVES = ("select_2023h1", "select_2023h2")


@dataclass(frozen=True)
class EvaluationConfig:
    source_csv: str = (
        "data/binance_coinm_quarterly_strip_pre2024/"
        "BTCUSD_front_next_quarterly_5m_20200701T0000_20231231T2350.csv.gz"
    )
    manifest_json: str = (
        "data/binance_coinm_quarterly_strip_pre2024/build_manifest.json"
    )
    output: str = "results/coinm_roll_migration_pre2024_selection_2026-07-19.json"
    freeze_output: str = str(EVALUATOR_FREEZE)
    leverage: float = 0.5
    contract_face_usd: float = 100.0
    cost_rate_per_side: float = 0.0006
    stress_cost_rate_per_side: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_719
    delay_1h_bars: int = 12
    delay_24h_bars: int = 288


@dataclass(frozen=True)
class InverseTrade:
    signal_position: int
    entry_position: int
    exit_position: int
    side: int
    traded_leg: str
    symbol: str
    entry_price: float
    exit_price: float
    favorable_price: float
    adverse_price: float
    entry_date: str

    @property
    def gross_return(self) -> float:
        return inverse_usd_return(self.entry_price, self.exit_price, self.side)


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
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _stable_artifact_hash(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "freeze_hash", "result_hash"}
    }
    return canonical_hash(stable)


def _require_canonical_artifact_paths(cfg: EvaluationConfig) -> None:
    if asdict(cfg) != asdict(EvaluationConfig()):
        raise ValueError("all evaluator paths and protocol parameters are frozen")


def _support_stable_payload(support: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in support.items()
        if key not in {"schema_version", "created_at", "support_freeze_hash"}
    }


def _verify_static_dependencies() -> dict[str, Any]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen dependency changed: {path}")
    support = _read_json(SUPPORT_RESULT)
    if support.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("support stage opened outcomes")
    if support.get("support_freeze_hash") != canonical_hash(
        _support_stable_payload(support)
    ):
        raise ValueError("support freeze hash changed")
    if support.get("source", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("support source identity changed")
    if support.get("source_manifest", {}).get("sha256") != MANIFEST_SHA256:
        raise ValueError("support manifest identity changed")
    if _sha256(support["source"]["path"]) != SOURCE_SHA256:
        raise ValueError("support source bytes changed after freeze")
    if _sha256(support["source_manifest"]["path"]) != MANIFEST_SHA256:
        raise ValueError("support manifest bytes changed after freeze")
    expected_candidates = [asdict(candidate) for candidate in CANDIDATES]
    if [item.get("candidate") for item in support["candidates"]] != expected_candidates:
        raise ValueError("candidate definitions changed after support freeze")
    if sum(item.get("passes_support") is True for item in support["candidates"]) != 2:
        raise ValueError("unexpected supported-candidate count")
    return support


def freeze_evaluator(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_artifact_paths(cfg)
    support = _verify_static_dependencies()
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists; evaluator cannot be frozen")
    if Path(cfg.freeze_output).exists():
        raise ValueError("evaluator freeze already exists and cannot be replaced")
    if _sha256(cfg.source_csv) != SOURCE_SHA256:
        raise ValueError("outcome source identity changed")
    if _sha256(cfg.manifest_json) != MANIFEST_SHA256:
        raise ValueError("outcome source manifest changed")
    core = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "support_commit": SUPPORT_COMMIT,
        "support_freeze_hash": support["support_freeze_hash"],
        "evaluation_source": str(EVALUATOR_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "source_sha256": SOURCE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": [
            "fit",
            "select_2023",
            "test_2024",
            "eval_2025",
            "holdout_2026",
        ],
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    core["freeze_hash"] = _stable_artifact_hash(core)
    _write_json_exclusive(cfg.freeze_output, core)
    return core


def verify_evaluator_freeze(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_artifact_paths(cfg)
    freeze = _read_json(cfg.freeze_output)
    if freeze.get("freeze_hash") != _stable_artifact_hash(freeze):
        raise ValueError("evaluator freeze hash changed")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("evaluator source changed after freeze")
    if freeze.get("config") != asdict(cfg):
        raise ValueError("evaluation config changed after freeze")
    if freeze.get("opened_windows") != [] or freeze.get("mutable_parameters") != []:
        raise ValueError("evaluator freeze is not sealed")
    if freeze.get("candidate_returns_computed_before_freeze") is not False:
        raise ValueError("candidate returns were computed before evaluator freeze")
    if freeze.get("simulation_run") is not False:
        raise ValueError("evaluator freeze ran a simulation")
    if _sha256(cfg.source_csv) != SOURCE_SHA256:
        raise ValueError("frozen outcome source changed")
    if _sha256(cfg.manifest_json) != MANIFEST_SHA256:
        raise ValueError("frozen outcome manifest changed")
    return freeze


def _supported_candidates(support: dict[str, Any]) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    return [
        by_name[item["candidate"]["name"]]
        for item in support["candidates"]
        if item.get("passes_support") is True
    ]


def _event_clock_hash(
    source: pd.DataFrame,
    active: np.ndarray,
    side: np.ndarray,
    candidate: Candidate,
) -> str:
    rows = [
        {
            "signal_bar_open": str(source.iloc[position]["signal_bar_open_utc"]),
            "side": int(side[position]),
            "symbol": str(source.iloc[position][f"{candidate.traded_leg}_symbol"]),
        }
        for position in np.flatnonzero(active)
    ]
    return canonical_hash(rows)


def _rebuild_event_clocks(
    cfg: EvaluationConfig,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    source = load_source(cfg.source_csv)
    state = build_signal_state(source)
    clocks = {
        candidate.name: candidate_clock(source, state, candidate)
        for candidate in CANDIDATES
    }
    return source, clocks


def _verify_rebuilt_support(
    source: pd.DataFrame,
    clocks: dict[str, tuple[np.ndarray, np.ndarray]],
    support: dict[str, Any],
) -> None:
    support_by_name = {
        item["candidate"]["name"]: item for item in support["candidates"]
    }
    for candidate in _supported_candidates(support):
        active, side = clocks[candidate.name]
        frozen = support_by_name[candidate.name]
        if _event_clock_hash(source, active, side, candidate) != frozen["clock_hash"]:
            raise ValueError(f"frozen event clock changed: {candidate.name}")
        rebuilt_support, schedule_hashes = windowed_support_summary(
            source, active, side, candidate
        )
        if rebuilt_support != frozen["support"]:
            raise ValueError(f"frozen support changed: {candidate.name}")
        if schedule_hashes != frozen["schedule_hashes"]:
            raise ValueError(f"frozen schedules changed: {candidate.name}")


def _load_outcomes(cfg: EvaluationConfig) -> pd.DataFrame:
    if _sha256(cfg.source_csv) != SOURCE_SHA256:
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
        usecols=lambda column: column in columns,
    )
    outcome["signal_bar_open_utc"] = (
        pd.to_datetime(outcome["signal_bar_open_utc"], utc=True, errors="raise")
        .dt.tz_convert(None)
    )
    dates = outcome["signal_bar_open_utc"]
    if (
        outcome.empty
        or dates.duplicated().any()
        or dates.iloc[0] != pd.Timestamp("2020-07-01")
        or dates.iloc[-1] != pd.Timestamp("2023-12-31 23:50")
    ):
        raise ValueError("outcome source is not the exact sealed interval")
    if not dates.equals(
        pd.Series(
            pd.date_range(dates.iloc[0], dates.iloc[-1], freq="5min"),
            name="signal_bar_open_utc",
        )
    ):
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
        if (
            prices.loc[complete, f"{leg}_high"]
            < prices.loc[complete, [f"{leg}_open", f"{leg}_close"]].max(axis=1)
        ).any() or (
            prices.loc[complete, f"{leg}_low"]
            > prices.loc[complete, [f"{leg}_open", f"{leg}_close"]].min(axis=1)
        ).any() or (
            prices.loc[complete, f"{leg}_high"]
            < prices.loc[complete, f"{leg}_low"]
        ).any():
            raise ValueError(f"{leg} outcome source violates OHLC invariants")
        outcome.loc[:, price_columns] = prices
    return outcome


def inverse_coin_pnl(
    contracts: float,
    face_usd: float,
    entry_price: float,
    mark_price: float,
    side: int,
) -> float:
    """Exact inverse-contract PnL in BTC."""
    if contracts < 0.0 or face_usd <= 0.0 or entry_price <= 0.0 or mark_price <= 0.0:
        raise ValueError("invalid inverse contract inputs")
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    return float(side * contracts * face_usd * (1.0 / entry_price - 1.0 / mark_price))


def inverse_usd_return(entry_price: float, mark_price: float, side: int) -> float:
    """USD PnL per fixed USD face after converting inverse coin PnL at mark."""
    coin_pnl = inverse_coin_pnl(1.0, 1.0, entry_price, mark_price, side)
    return float(coin_pnl * mark_price)


def _build_trades(
    outcome: pd.DataFrame,
    positions: dict[pd.Timestamp, int],
    schedule: pd.DataFrame,
    candidate: Candidate,
    *,
    flip: bool = False,
) -> list[InverseTrade]:
    trades: list[InverseTrade] = []
    dates = outcome["signal_bar_open_utc"]
    leg = candidate.traded_leg
    for row in schedule.to_dict(orient="records"):
        signal_time = _timestamp(row["signal_bar_open"])
        signal_position = positions.get(signal_time)
        if signal_position is None:
            raise ValueError("frozen signal time is absent from outcome grid")
        entry_position = signal_position + 1
        exit_position = entry_position + candidate.hold_bars
        if exit_position >= len(outcome):
            raise ValueError("frozen trade exceeds outcome horizon")
        entry_time = dates.iloc[entry_position]
        exit_time = dates.iloc[exit_position]
        if entry_time != _timestamp(row["feature_available"]):
            raise ValueError("execution entry differs from feature availability")
        if entry_time != _timestamp(row["entry_time"]):
            raise ValueError("execution entry differs from frozen next-open entry")
        if exit_time != _timestamp(row["exit_time"]):
            raise ValueError("execution exit differs from frozen elapsed hold")
        symbol = str(row["symbol"])
        held_symbols = outcome.loc[
            entry_position:exit_position, f"{leg}_symbol"
        ].astype(str)
        if not held_symbols.eq(symbol).all():
            raise ValueError("trade crosses a contract-symbol transition")
        entry_price = float(outcome.iloc[entry_position][f"{leg}_open"])
        exit_price = float(outcome.iloc[exit_position][f"{leg}_open"])
        held = outcome.iloc[entry_position:exit_position]
        highs = held[f"{leg}_high"].to_numpy(float)
        lows = held[f"{leg}_low"].to_numpy(float)
        required = np.r_[entry_price, exit_price, highs, lows]
        if not np.isfinite(required).all() or (required <= 0.0).any():
            raise ValueError("scheduled trade crosses missing outcome prices")
        side = -int(row["side"]) if flip else int(row["side"])
        if side > 0:
            favorable_price = max(entry_price, float(highs.max()))
            adverse_price = min(entry_price, float(lows.min()))
        else:
            favorable_price = min(entry_price, float(lows.min()))
            adverse_price = max(entry_price, float(highs.max()))
        trades.append(
            InverseTrade(
                signal_position=signal_position,
                entry_position=entry_position,
                exit_position=exit_position,
                side=side,
                traded_leg=leg,
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                favorable_price=favorable_price,
                adverse_price=adverse_price,
                entry_date=str(entry_time),
            )
        )
    return trades


def delayed_schedule(
    schedule: pd.DataFrame,
    source: pd.DataFrame,
    candidate: Candidate,
    *,
    bars: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Shift the frozen non-overlapping event set without re-scheduling it."""
    if bars <= 0:
        raise ValueError("delay bars must be positive")
    dates = source["signal_bar_open_utc"]
    positions = {
        _timestamp(timestamp): position for position, timestamp in enumerate(dates)
    }
    start_time = _timestamp(start)
    end_time = _timestamp(end)
    rows: list[dict[str, Any]] = []
    for row in schedule.to_dict(orient="records"):
        original_position = positions.get(_timestamp(row["signal_bar_open"]))
        if original_position is None:
            raise ValueError("frozen delayed-control source signal is absent")
        destination = original_position + bars
        if destination >= len(source):
            continue
        destination_row = source.iloc[destination]
        signal_time = _timestamp(destination_row["signal_bar_open_utc"])
        entry_time = _timestamp(destination_row["trade_earliest_time_utc"])
        exit_time = entry_time + pd.Timedelta(minutes=candidate.hold_minutes)
        symbol = str(destination_row[f"{candidate.traded_leg}_symbol"])
        hold_hours = candidate.hold_bars * 5.0 / 60.0
        front_dte = float(destination_row["front_hours_to_delivery"])
        next_dte = float(destination_row["next_hours_to_delivery"])
        delivery_safe = (
            front_dte >= 12.0 + hold_hours + 5.0 / 60.0
            and next_dte >= 12.0 + hold_hours + 5.0 / 60.0
            and front_dte <= 45.0 * 24.0
        )
        if (
            signal_time < start_time
            or entry_time < start_time
            or exit_time >= end_time
            or not bool(destination_row["feature_valid"])
            or symbol != str(row["symbol"])
            or not delivery_safe
        ):
            continue
        rows.append(
            {
                "signal_bar_open": str(signal_time),
                "feature_available": str(
                    destination_row["feature_available_time_utc"]
                ),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "side": int(row["side"]),
                "traded_leg": candidate.traded_leg,
                "symbol": symbol,
            }
        )
    delayed = pd.DataFrame(rows, columns=schedule.columns)
    if len(delayed) > 1:
        entries = pd.to_datetime(delayed["entry_time"]).to_numpy(dtype="datetime64[ns]")
        exits = pd.to_datetime(delayed["exit_time"]).to_numpy(dtype="datetime64[ns]")
        if np.isnat(entries).any() or np.isnat(exits).any():
            raise ValueError("delayed schedule contains NaT")
        if (entries[1:] < exits[:-1]).any():
            raise ValueError("uniform delay unexpectedly created schedule overlap")
    return delayed


def _schedule_hash(schedule: pd.DataFrame) -> str:
    return canonical_hash(schedule.to_dict(orient="records"))


def _window_schedule(
    source: pd.DataFrame,
    active: np.ndarray,
    side: np.ndarray,
    candidate: Candidate,
    window: str,
) -> pd.DataFrame:
    start, end = WINDOWS[window]
    return nonoverlapping_schedule(
        source,
        active,
        side,
        candidate,
        start=_timestamp(start),
        end=_timestamp(end),
    )


def _window_trades(
    source: pd.DataFrame,
    outcome: pd.DataFrame,
    positions: dict[pd.Timestamp, int],
    active: np.ndarray,
    side: np.ndarray,
    candidate: Candidate,
    window: str,
    *,
    flip: bool = False,
) -> tuple[list[InverseTrade], pd.DataFrame]:
    schedule = _window_schedule(source, active, side, candidate, window)
    return (
        _build_trades(outcome, positions, schedule, candidate, flip=flip),
        schedule,
    )


def _trade_factor(
    trade: InverseTrade,
    mark_price: float,
    cfg: EvaluationConfig,
    cost_rate: float,
    *,
    include_exit_cost: bool,
) -> float:
    gross = inverse_usd_return(trade.entry_price, mark_price, trade.side)
    cost_legs = 2 if include_exit_cost else 1
    return float(1.0 + cfg.leverage * gross - cfg.leverage * cost_rate * cost_legs)


def strict_equity_stats(
    trades: Iterable[InverseTrade],
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
    cost_rate: float | None = None,
) -> dict[str, Any]:
    """Full-calendar USD equity and conservative held-path strict MDD.

    Research exposure is fractional fixed-USD-face contracts, so each trade is
    exactly ``leverage`` times pre-entry USD equity.  The exact inverse PnL is
    first accrued in coin and marked back to USD.  Entry cost is paid before a
    favorable HWM; the adverse state and realized exit include both cost legs.
    Favorable-before-adverse ordering is conservative when bar ordering is
    unknowable.
    """
    cost = cfg.cost_rate_per_side if cost_rate is None else float(cost_rate)
    if cfg.contract_face_usd != 100.0:
        raise ValueError("BTCUSD quarterly contract face must remain USD 100")
    if not 0.0 <= cost < 1.0 or not 0.0 < cfg.leverage <= 10.0:
        raise ValueError("invalid leverage or cost")
    equity = peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_returns: list[float] = []
    sides: list[int] = []
    entry_dates: list[str] = []
    for trade in trades:
        entry_equity = equity
        entry_factor = 1.0 - cfg.leverage * cost
        favorable_factor = _trade_factor(
            trade,
            trade.favorable_price,
            cfg,
            cost,
            include_exit_cost=False,
        )
        adverse_factor = _trade_factor(
            trade,
            trade.adverse_price,
            cfg,
            cost,
            include_exit_cost=True,
        )
        realized_factor = _trade_factor(
            trade,
            trade.exit_price,
            cfg,
            cost,
            include_exit_cost=True,
        )
        entry_mark = entry_equity * max(0.0, entry_factor)
        strict_mdd = max(strict_mdd, 1.0 - entry_mark / peak)
        intratrade_peak = max(peak, entry_equity * max(0.0, favorable_factor))
        adverse_equity = entry_equity * max(0.0, adverse_factor)
        strict_mdd = max(strict_mdd, 1.0 - adverse_equity / intratrade_peak)
        peak = intratrade_peak
        equity = entry_equity * max(0.0, realized_factor)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        net_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(trade.gross_return)
        sides.append(trade.side)
        entry_dates.append(trade.entry_date)
        if equity <= 0.0:
            break
    years = (
        pd.Timestamp(end) - pd.Timestamp(start)
    ).total_seconds() / (365.25 * 86_400.0)
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
        "longs": int(sum(value > 0 for value in sides)),
        "shorts": int(sum(value < 0 for value in sides)),
        "mean_net_bps": float(returns.mean() * 10_000.0) if len(returns) else 0.0,
        "mean_gross_bps": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "wall_clock_years": float(years),
        "net_trade_returns": [float(value) for value in returns],
        "entry_dates": entry_dates,
    }


def _public_stats(stats: dict[str, Any], *, cluster: bool, cfg: EvaluationConfig) -> dict[str, Any]:
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
    trades: list[InverseTrade],
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
        "fit_trades_at_least_400": fit["trades"] >= 400,
        "select_trades_at_least_100": select["trades"] >= 100,
        "fit_mean_net_positive": fit["mean_net_bps"] > 0.0,
        "select_mean_net_positive": select["mean_net_bps"] > 0.0,
        "fit_weekly_cluster_p_below_0_10": fit["weekly_cluster_sign_flip"][
            "p_value_one_sided"
        ]
        < 0.10,
        "select_weekly_cluster_p_below_0_10": select[
            "weekly_cluster_sign_flip"
        ]["p_value_one_sided"]
        < 0.10,
        "at_least_4_of_5_fit_halves_positive": sum(
            windows[name]["absolute_return_pct"] > 0.0 for name in FIT_HALVES
        )
        >= 4,
        "each_fit_half_at_least_50_trades": min(
            windows[name]["trades"] for name in FIT_HALVES
        )
        >= 50,
        "each_select_half_positive": min(
            windows[name]["absolute_return_pct"] for name in SELECT_HALVES
        )
        > 0.0,
        "each_select_half_at_least_40_trades": min(
            windows[name]["trades"] for name in SELECT_HALVES
        )
        >= 40,
        "fit_10bp_per_side_stress_positive": stress["fit"][
            "absolute_return_pct"
        ]
        > 0.0,
        "select_10bp_per_side_stress_positive": stress["select_2023"][
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


def winner_sort_key(row: dict[str, Any]) -> tuple[float, float, float, int, str]:
    windows = row["windows"]
    return (
        -min(
            windows["fit"]["cagr_to_strict_mdd"],
            windows["select_2023"]["cagr_to_strict_mdd"],
        ),
        -min(windows[name]["absolute_return_pct"] for name in SELECT_HALVES),
        -min(windows["fit"]["mean_net_bps"], windows["select_2023"]["mean_net_bps"]),
        -(windows["fit"]["trades"] + windows["select_2023"]["trades"]),
        row["name"],
    )


def evaluate(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_artifact_paths(cfg)
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists and cannot be replaced")
    support = _verify_static_dependencies()
    freeze = verify_evaluator_freeze(cfg)
    source, clocks = _rebuild_event_clocks(cfg)
    _verify_rebuilt_support(source, clocks, support)
    outcome = _load_outcomes(cfg)
    if not outcome["signal_bar_open_utc"].equals(source["signal_bar_open_utc"]):
        raise ValueError("signal and outcome clocks differ")
    positions = {
        _timestamp(timestamp): position
        for position, timestamp in enumerate(outcome["signal_bar_open_utc"])
    }

    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    support_by_name = {
        item["candidate"]["name"]: item for item in support["candidates"]
    }
    for candidate in _supported_candidates(support):
        active, side = clocks[candidate.name]
        windows: dict[str, dict[str, Any]] = {}
        for name in WINDOWS:
            trades, _ = _window_trades(
                source, outcome, positions, active, side, candidate, name
            )
            windows[name] = _stats(
                trades, name, cfg, cluster=name in FULL_WINDOWS
            )

        stress: dict[str, dict[str, Any]] = {}
        direction_flip: dict[str, dict[str, Any]] = {}
        delay_1h: dict[str, dict[str, Any]] = {}
        delay_24h: dict[str, dict[str, Any]] = {}
        control_schedule_hashes = {"delay_1h": {}, "delay_24h": {}}
        for name in FULL_WINDOWS:
            base_trades, base_schedule = _window_trades(
                source, outcome, positions, active, side, candidate, name
            )
            one_hour_schedule = delayed_schedule(
                base_schedule,
                source,
                candidate,
                bars=cfg.delay_1h_bars,
                start=WINDOWS[name][0],
                end=WINDOWS[name][1],
            )
            day_schedule = delayed_schedule(
                base_schedule,
                source,
                candidate,
                bars=cfg.delay_24h_bars,
                start=WINDOWS[name][0],
                end=WINDOWS[name][1],
            )
            flip_trades = _build_trades(
                outcome,
                positions,
                base_schedule,
                candidate,
                flip=True,
            )
            one_hour_trades = _build_trades(
                outcome,
                positions,
                one_hour_schedule,
                candidate,
            )
            day_trades = _build_trades(
                outcome,
                positions,
                day_schedule,
                candidate,
            )
            control_schedule_hashes["delay_1h"][name] = _schedule_hash(
                one_hour_schedule
            )
            control_schedule_hashes["delay_24h"][name] = _schedule_hash(
                day_schedule
            )
            stress[name] = _stats(
                base_trades,
                name,
                cfg,
                cost_rate=cfg.stress_cost_rate_per_side,
            )
            direction_flip[name] = _stats(flip_trades, name, cfg)
            delay_1h[name] = _stats(one_hour_trades, name, cfg)
            delay_24h[name] = _stats(day_trades, name, cfg)

        gates = selection_gates(
            windows, stress, direction_flip, delay_1h, delay_24h
        )
        frozen = support_by_name[candidate.name]
        row = {
            "candidate": asdict(candidate),
            "name": candidate.name,
            "clock_hash": frozen["clock_hash"],
            "windows": windows,
            "stress_10bp_per_side": stress,
            "direction_flip": direction_flip,
            "delay_1h": delay_1h,
            "delay_24h": delay_24h,
            "control_schedule_hashes": control_schedule_hashes,
            "gates": gates,
            "passes_selection": bool(all(gates.values())),
        }
        rows.append(row)
        if row["passes_selection"]:
            eligible.append(row)

    eligible.sort(key=winner_sort_key)
    winner = eligible[0] if eligible else None
    stable = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "support_commit": SUPPORT_COMMIT,
            "support_freeze_hash": support["support_freeze_hash"],
            "evaluator_freeze_hash": freeze["freeze_hash"],
            "opened_windows": ["fit", "select_2023"],
            "sealed_windows": ["test_2024", "eval_2025", "holdout_2026"],
            "full_calendar_cagr": True,
            "strict_mdd": (
                "global/pre-entry HWM plus entry-cost/favorable-before-adverse held "
                "five-minute path, hypothetical exit cost, and realized two-sided cost"
            ),
            "inverse_ledger": (
                "exact coin PnL converted to USD at each mark; fractional fixed-USD-face "
                "research contracts; no delivery-futures funding"
            ),
            "production_rounding_modeled": False,
            "btc_collateral_beta_modeled": False,
            "post_selection_parameter_repair_allowed": False,
        },
        "config": asdict(cfg),
        "candidates_evaluated": len(rows),
        "candidates_passing": len(eligible),
        "winner": winner,
        "advance_to_2024_test": winner is not None,
        "candidates": rows,
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
