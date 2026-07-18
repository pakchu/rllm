"""Frozen strict pre-2024 evaluator for price-memory cage escape events."""
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

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.preregister_price_memory_cage_escape_alpha import (
    CANDIDATES,
    Candidate,
    PERSISTENCE_HORIZONS,
    build_saddle_state,
    canonical_hash,
    combine_event_clock,
    load_market,
    nonoverlapping_schedule,
    persistent_barrier_features,
    volume_clock_flow_speed,
    windowed_support_summary,
)
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
)


SUPPORT_COMMIT = "7acc3f7bc5d97afa2a642e6e848ef276a55a627a"
STATIC_INPUT_SHA256 = {
    "training/preregister_price_memory_cage_escape_alpha.py": "46cc84e5e51159962a9e0d443f74bae77b42da90d9dc020257a8956b67abe8f7",
    "docs/price-memory-cage-escape-preregistration-2026-07-19.md": "2396aa6c0c4f22cc082d59ab8ef08e9be9572b9450525dbeb5fc0805b4504b83",
    "results/price_memory_cage_escape_support_2026-07-19.json": "418664310a00aa7b22b4262dacb2918fabc576e2589e071b1309dfe150e6bfd8",
    "training/search_occupation_saddle_escape_alpha.py": "37bbb4d0799f227539900d359fe4af78258c840bb3e025d98d4c29d1dc92d112",
    "training/search_persistent_barrier_annihilation_alpha.py": "c62870da2657a4cb049bef9d67ef593059892c4e3d44061cdc3df6519715730f",
    "training/search_inventory_purge_reclaim_alpha.py": "5d8d4df7ea79790afb919bbb481d11de33ecba5768f6e26feb1f7667cd947d65",
    "training/evaluate_metaorder_fragmentation_impact_curvature.py": "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3",
}
MARKET_SOURCE_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
FUNDING_SOURCE_SHA256 = "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"

SUPPORT_RESULT = Path("results/price_memory_cage_escape_support_2026-07-19.json")
EVALUATOR_SOURCE = Path("training/evaluate_price_memory_cage_escape_pre2024.py")
EVALUATOR_FREEZE = Path(
    "results/price_memory_cage_escape_evaluator_freeze_2026-07-19.json"
)

WINDOWS: dict[str, tuple[str, str]] = {
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
FULL_WINDOWS = ("fit", "select_2023")
FIT_HALVES = ("fit_2021h1", "fit_2021h2", "fit_2022h1", "fit_2022h2")
SELECT_HALVES = ("select_2023h1", "select_2023h2")


@dataclass(frozen=True)
class EvaluationConfig:
    market_csv: str = (
        "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    funding_csv: str = (
        "/home/pakchu/rllm/results/binance_um_btcusdt_realized_funding_2020_2023.csv"
    )
    output: str = "results/price_memory_cage_escape_pre2024_selection_2026-07-19.json"
    freeze_output: str = str(EVALUATOR_FREEZE)
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_719
    delay_1h_bars: int = 12
    delay_24h_bars: int = 288


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


def _write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _require_canonical_artifact_paths(cfg: EvaluationConfig) -> None:
    if cfg.output != EvaluationConfig.output:
        raise ValueError("selection output path is frozen")
    if cfg.freeze_output != EvaluationConfig.freeze_output:
        raise ValueError("evaluator freeze path is frozen")


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
    if support.get("source", {}).get("sha256") != MARKET_SOURCE_SHA256:
        raise ValueError("support source identity changed")
    if _sha256(support["source"]["path"]) != MARKET_SOURCE_SHA256:
        raise ValueError("support source bytes changed after freeze")
    expected_candidates = [asdict(candidate) for candidate in CANDIDATES]
    if [item.get("candidate") for item in support["candidates"]] != expected_candidates:
        raise ValueError("candidate grid changed after support freeze")
    if sum(item.get("passes_support") is True for item in support["candidates"]) != 3:
        raise ValueError("unexpected supported-candidate count")
    return support


def freeze_evaluator(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_artifact_paths(cfg)
    support = _verify_static_dependencies()
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists; evaluator cannot be frozen now")
    if Path(cfg.freeze_output).exists():
        raise ValueError("evaluator freeze already exists and cannot be replaced")
    if _sha256(cfg.market_csv) != MARKET_SOURCE_SHA256:
        raise ValueError("market outcome source identity changed")
    if _sha256(cfg.funding_csv) != FUNDING_SOURCE_SHA256:
        raise ValueError("funding outcome source identity changed")
    core = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "support_commit": SUPPORT_COMMIT,
        "support_freeze_hash": support["support_freeze_hash"],
        "evaluation_source": str(EVALUATOR_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "market_source_sha256": MARKET_SOURCE_SHA256,
        "funding_source_sha256": FUNDING_SOURCE_SHA256,
        "opened_windows": [],
        "sealed_windows": ["fit", "select_2023", "test_2024", "eval_2025", "holdout_2026"],
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    core["freeze_hash"] = canonical_hash(core)
    _write_json_exclusive(cfg.freeze_output, core)
    return core


def verify_evaluator_freeze(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_artifact_paths(cfg)
    freeze = _read_json(cfg.freeze_output)
    stable = {key: value for key, value in freeze.items() if key != "freeze_hash"}
    if freeze.get("freeze_hash") != canonical_hash(stable):
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
    for path, expected in (
        (cfg.market_csv, MARKET_SOURCE_SHA256),
        (cfg.funding_csv, FUNDING_SOURCE_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    return freeze


def _validated_funding_frame(
    raw: pd.DataFrame,
    *,
    start: str = "2020-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    required = {"funding_time_utc", "symbol", "funding_rate"}
    if not required.issubset(raw.columns):
        raise ValueError("funding source schema changed")
    timestamps = pd.to_datetime(raw["funding_time_utc"], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("funding source contains duplicate timestamps")
    if not raw["symbol"].eq("BTCUSDT").all():
        raise ValueError("funding source contains an unexpected symbol")
    order = np.argsort(timestamps.to_numpy())
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    rates = pd.to_numeric(raw["funding_rate"], errors="raise").iloc[order].reset_index(drop=True)
    expected = pd.date_range(start, end, freq="8h", inclusive="left", tz="UTC")
    if len(timestamps) != len(expected):
        raise ValueError("funding source does not cover every expected eight-hour event")
    offsets_ms = (pd.DatetimeIndex(timestamps).asi8 - expected.asi8).astype(float) / 1_000_000.0
    if np.abs(offsets_ms).max(initial=0.0) > 60_000.0:
        raise ValueError("funding timestamp exceeds the frozen one-minute grid tolerance")
    if not np.isfinite(rates.to_numpy(float)).all():
        raise ValueError("funding source contains non-finite rates")
    return pd.DataFrame(
        {"date": expected.tz_convert(None), "funding_rate": rates.to_numpy(float)}
    )


def _load_outcomes(cfg: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["date", "open", "high", "low", "close"]
    market = pd.read_csv(
        cfg.market_csv,
        compression="infer",
        usecols=columns,
        parse_dates=["date"],
    ).sort_values("date").reset_index(drop=True)
    expected_first = pd.Timestamp("2020-01-01")
    expected_last = pd.Timestamp("2023-12-31 23:55")
    if (
        market.empty
        or market["date"].duplicated().any()
        or market["date"].iloc[0] != expected_first
        or market["date"].iloc[-1] != expected_last
    ):
        raise ValueError("market outcome source is not the exact sealed interval")
    expected = pd.date_range(expected_first, expected_last, freq="5min")
    if not market["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("market outcome source is not a gapless five-minute grid")
    prices = market[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="raise"
    )
    values = prices.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("market outcome source contains invalid prices")
    if (
        (prices["high"] < prices[["open", "close"]].max(axis=1)).any()
        or (prices["low"] > prices[["open", "close"]].min(axis=1)).any()
        or (prices["high"] < prices["low"]).any()
    ):
        raise ValueError("market outcome source violates OHLC invariants")
    market.loc[:, ["open", "high", "low", "close"]] = prices
    funding = _validated_funding_frame(pd.read_csv(cfg.funding_csv))
    return market, funding


def _supported_candidates(support: dict[str, Any]) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    return [
        by_name[item["name"]]
        for item in support["candidates"]
        if item.get("passes_support") is True
    ]


def _rebuild_event_clocks(
    cfg: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.Series, dict[int, tuple[np.ndarray, np.ndarray]]]:
    market, dates = load_market(cfg.market_csv)
    occupation = build_saddle_state(market, dates, mode="joint")
    flow_speed = volume_clock_flow_speed(market)
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    log_close = np.log(market["close"].to_numpy(float))
    for horizon in PERSISTENCE_HORIZONS:
        persistence = persistent_barrier_features(log_close, dates, horizon=horizon)
        output[horizon] = combine_event_clock(
            occupation, persistence, flow_speed, dates
        )
    return market, dates, output


def _event_clock_hash(dates: pd.Series, active: np.ndarray, side: np.ndarray) -> str:
    rows = [
        {"signal_bar_open": str(dates.iloc[position]), "side": int(side[position])}
        for position in np.flatnonzero(active)
    ]
    return canonical_hash(rows)


def delay_clock(
    active: np.ndarray, side: np.ndarray, bars: int
) -> tuple[np.ndarray, np.ndarray]:
    if bars <= 0:
        raise ValueError("delay bars must be positive")
    active = np.asarray(active, dtype=bool)
    side = np.asarray(side, dtype=np.int8)
    delayed_active = np.zeros(len(active), dtype=bool)
    delayed_side = np.zeros(len(side), dtype=np.int8)
    if bars < len(active):
        delayed_active[bars:] = active[:-bars]
        delayed_side[bars:] = side[:-bars]
    delayed_side[~delayed_active] = 0
    return delayed_active, delayed_side


def delayed_window_clock(
    active: np.ndarray,
    side: np.ndarray,
    dates: pd.Series,
    *,
    bars: int,
    start: str,
    end: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Delay only events whose original and delayed clocks stay in one split."""
    active = np.asarray(active, dtype=bool)
    side = np.asarray(side, dtype=np.int8)
    if not (len(active) == len(side) == len(dates)):
        raise ValueError("clock lengths differ")
    if bars <= 0:
        raise ValueError("delay bars must be positive")
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    original_in_window = (
        (dates >= start_time) & (dates < end_time)
    ).to_numpy(bool)
    sources = np.flatnonzero(active & original_in_window)
    destinations = sources + bars
    keep = destinations < len(active)
    sources = sources[keep]
    destinations = destinations[keep]
    if len(destinations):
        delayed_dates = dates.iloc[destinations].to_numpy(dtype="datetime64[ns]")
        keep = (
            (delayed_dates >= np.datetime64(start_time))
            & (delayed_dates < np.datetime64(end_time))
        )
        sources = sources[keep]
        destinations = destinations[keep]
    delayed_active = np.zeros(len(active), dtype=bool)
    delayed_side = np.zeros(len(side), dtype=np.int8)
    delayed_active[destinations] = True
    delayed_side[destinations] = side[sources]
    return delayed_active, delayed_side


def _build_trades(
    engine: ExecutionEngine,
    market_positions: dict[pd.Timestamp, int],
    schedule: pd.DataFrame,
    candidate: Candidate,
    *,
    flip: bool = False,
) -> list[Trade]:
    trades: list[Trade] = []
    for row in schedule.itertuples(index=False):
        signal_time = pd.Timestamp(row.signal_bar_open)
        signal_position = market_positions.get(signal_time)
        if signal_position is None:
            raise ValueError("frozen signal time is absent from market grid")
        side = -int(row.side) if flip else int(row.side)
        trade = engine.trade_at(
            signal_position,
            side,
            candidate.hold_hours * 12,
            1_000_000,
            1_000_000,
        )
        if trade is None:
            raise ValueError("frozen schedule exceeds market horizon")
        entry_time = pd.Timestamp(engine.dates.iloc[trade.entry_position])
        exit_time = pd.Timestamp(engine.dates.iloc[trade.exit_position])
        if entry_time != pd.Timestamp(row.feature_available):
            raise ValueError("execution entry differs from feature availability")
        if entry_time != pd.Timestamp(row.entry_time):
            raise ValueError("execution entry differs from frozen next-open entry")
        if exit_time != pd.Timestamp(row.exit_time):
            raise ValueError("execution exit differs from frozen elapsed hold")
        trades.append(trade)
    return trades


def _window_trades(
    dates: pd.Series,
    active: np.ndarray,
    side: np.ndarray,
    candidate: Candidate,
    window: str,
    engine: ExecutionEngine,
    market_positions: dict[pd.Timestamp, int],
    *,
    flip: bool = False,
) -> tuple[list[Trade], pd.DataFrame]:
    start, end = WINDOWS[window]
    schedule = nonoverlapping_schedule(
        dates,
        active,
        side,
        hold_hours=candidate.hold_hours,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    return (
        _build_trades(
            engine,
            market_positions,
            schedule,
            candidate,
            flip=flip,
        ),
        schedule,
    )


def _net_trade_returns(
    trades: Iterable[Trade],
    cfg: EvaluationConfig,
    *,
    cost_rate: float | None = None,
) -> list[float]:
    cost = cfg.fee_rate + cfg.slippage_rate if cost_rate is None else float(cost_rate)
    execution = 1.0 - cfg.leverage * cost
    return [
        float(execution * trade.price_factor * trade.funding_factor * execution - 1.0)
        for trade in trades
    ]


def strict_equity_stats(
    trades: Iterable[Trade],
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
    cost_rate: float | None = None,
) -> dict[str, Any]:
    """Full-calendar equity with conservative two-leg intratrade cost MDD.

    The favorable high-water applies the paid entry cost.  The adverse state
    assumes the exit cost is then paid at the adverse mark, so a strict held
    drawdown cannot be understated by omitting the eventual second cost leg.
    For worst-ordering robustness, realized funding credits are placed before
    the favorable mark while realized funding debits are placed before the
    adverse mark.
    """
    cost = float(cfg.fee_rate + cfg.slippage_rate if cost_rate is None else cost_rate)
    per_side_factor = 1.0 - float(cfg.leverage) * cost
    if not 0.0 < per_side_factor <= 1.0:
        raise ValueError("invalid per-side execution factor")
    equity = peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_returns: list[float] = []
    sides: list[int] = []
    for trade in trades:
        entry_equity = equity
        if not 0.0 < trade.funding_debit_factor <= 1.0:
            raise ValueError("invalid funding debit factor")
        funding_credit_factor = trade.funding_factor / trade.funding_debit_factor
        if not np.isfinite(funding_credit_factor) or funding_credit_factor < 1.0:
            raise ValueError("invalid funding credit factor")
        favorable_factor = (
            per_side_factor
            * funding_credit_factor
            * trade.favorable_price_factor
        )
        adverse_factor = (
            per_side_factor
            * trade.funding_debit_factor
            * trade.adverse_price_factor
            * per_side_factor
        )
        intratrade_peak = max(peak, equity * favorable_factor)
        strict_mdd = max(
            strict_mdd,
            1.0 - equity * adverse_factor / intratrade_peak,
        )
        peak = intratrade_peak
        equity *= (
            per_side_factor
            * trade.price_factor
            * trade.funding_factor
            * per_side_factor
        )
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        net_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(trade.gross_return)
        sides.append(trade.side)
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
        "mean_gross_bps": float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0,
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "wall_clock_years": float(years),
    }


def _stats(
    trades: list[Trade],
    window: str,
    cfg: EvaluationConfig,
    *,
    cost_rate: float | None = None,
    cluster: bool = False,
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    output = strict_equity_stats(
        trades,
        start=start,
        end=end,
        cfg=cfg,
        cost_rate=cost_rate,
    )
    if cluster:
        output["weekly_cluster_sign_flip"] = weekly_cluster_sign_flip(
            _net_trade_returns(trades, cfg, cost_rate=cost_rate),
            [trade.entry_date for trade in trades],
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
    return output


def selection_gates(
    windows: dict[str, dict[str, Any]],
    stress: dict[str, dict[str, Any]],
    direction_flip: dict[str, dict[str, Any]],
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
        "fit_trades_at_least_48": fit["trades"] >= 48,
        "select_trades_at_least_24": select["trades"] >= 24,
        "fit_mean_gross_at_least_25bp": fit["mean_gross_bps"] >= 25.0,
        "select_mean_gross_at_least_25bp": select["mean_gross_bps"] >= 25.0,
        "fit_weekly_cluster_p_below_0_10": fit["weekly_cluster_sign_flip"][
            "p_value_one_sided"
        ] < 0.10,
        "select_weekly_cluster_p_below_0_10": select[
            "weekly_cluster_sign_flip"
        ]["p_value_one_sided"] < 0.10,
        "at_least_3_of_4_fit_halves_positive": sum(
            windows[name]["absolute_return_pct"] > 0.0 for name in FIT_HALVES
        ) >= 3,
        "each_fit_half_at_least_6_trades": min(
            windows[name]["trades"] for name in FIT_HALVES
        ) >= 6,
        "each_select_half_positive": min(
            windows[name]["absolute_return_pct"] for name in SELECT_HALVES
        ) > 0.0,
        "each_select_half_at_least_8_trades": min(
            windows[name]["trades"] for name in SELECT_HALVES
        ) >= 8,
        "fit_10bp_per_side_stress_positive": stress["fit"]["absolute_return_pct"] > 0.0,
        "select_10bp_per_side_stress_positive": stress["select_2023"][
            "absolute_return_pct"
        ] > 0.0,
        "fit_direction_flip_negative": direction_flip["fit"]["absolute_return_pct"] < 0.0,
        "select_direction_flip_negative": direction_flip["select_2023"][
            "absolute_return_pct"
        ] < 0.0,
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
        -min(
            windows["fit"]["mean_net_bps"],
            windows["select_2023"]["mean_net_bps"],
        ),
        -(windows["fit"]["trades"] + windows["select_2023"]["trades"]),
        row["name"],
    )


def evaluate(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_artifact_paths(cfg)
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists and cannot be replaced")
    support = _verify_static_dependencies()
    freeze = verify_evaluator_freeze(cfg)
    _, feature_dates, clocks = _rebuild_event_clocks(cfg)

    support_by_name = {item["name"]: item for item in support["candidates"]}
    for candidate in _supported_candidates(support):
        active, side = clocks[candidate.persistence_horizon_bars]
        support_item = support_by_name[candidate.name]
        if _event_clock_hash(feature_dates, active, side) != support_item["clock_hash"]:
            raise ValueError(f"frozen event clock changed: {candidate.name}")
        rebuilt_support, schedule_hashes = windowed_support_summary(
            feature_dates,
            active,
            side,
            hold_hours=candidate.hold_hours,
        )
        if rebuilt_support != support_item["support"]:
            raise ValueError(f"frozen support counts changed: {candidate.name}")
        if schedule_hashes != support_item["schedule_hashes"]:
            raise ValueError(f"frozen schedules changed: {candidate.name}")

    market, funding = _load_outcomes(cfg)
    engine = ExecutionEngine(market, funding, cfg)
    market_positions = {
        timestamp: position for position, timestamp in enumerate(engine.dates)
    }
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate in _supported_candidates(support):
        active, side = clocks[candidate.persistence_horizon_bars]
        windows: dict[str, dict[str, Any]] = {}
        for name in WINDOWS:
            trades, _ = _window_trades(
                feature_dates,
                active,
                side,
                candidate,
                name,
                engine,
                market_positions,
            )
            windows[name] = _stats(
                trades,
                name,
                cfg,
                cluster=name in FULL_WINDOWS,
            )

        stress: dict[str, dict[str, Any]] = {}
        direction_flip: dict[str, dict[str, Any]] = {}
        delay_1h: dict[str, dict[str, Any]] = {}
        delay_24h: dict[str, dict[str, Any]] = {}
        for name in FULL_WINDOWS:
            start, end = WINDOWS[name]
            one_hour_active, one_hour_side = delayed_window_clock(
                active,
                side,
                feature_dates,
                bars=cfg.delay_1h_bars,
                start=start,
                end=end,
            )
            day_active, day_side = delayed_window_clock(
                active,
                side,
                feature_dates,
                bars=cfg.delay_24h_bars,
                start=start,
                end=end,
            )
            base_trades, _ = _window_trades(
                feature_dates,
                active,
                side,
                candidate,
                name,
                engine,
                market_positions,
            )
            flip_trades, _ = _window_trades(
                feature_dates,
                active,
                side,
                candidate,
                name,
                engine,
                market_positions,
                flip=True,
            )
            one_hour_trades, _ = _window_trades(
                feature_dates,
                one_hour_active,
                one_hour_side,
                candidate,
                name,
                engine,
                market_positions,
            )
            day_trades, _ = _window_trades(
                feature_dates,
                day_active,
                day_side,
                candidate,
                name,
                engine,
                market_positions,
            )
            stress[name] = _stats(
                base_trades,
                name,
                cfg,
                cost_rate=cfg.stress_cost_rate,
            )
            direction_flip[name] = _stats(flip_trades, name, cfg)
            delay_1h[name] = _stats(one_hour_trades, name, cfg)
            delay_24h[name] = _stats(day_trades, name, cfg)

        gates = selection_gates(windows, stress, direction_flip, delay_24h)
        support_item = support_by_name[candidate.name]
        row = {
            "candidate": asdict(candidate),
            "name": candidate.name,
            "clock_hash": support_item["clock_hash"],
            "windows": windows,
            "stress_10bp_per_side": stress,
            "direction_flip": direction_flip,
            "delay_1h": delay_1h,
            "delay_24h": delay_24h,
            "gates": gates,
            "passes_selection": bool(all(gates.values())),
        }
        rows.append(row)
        if row["passes_selection"]:
            eligible.append(row)

    eligible.sort(key=winner_sort_key)
    winner = eligible[0] if eligible else None
    stable_report = {
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
                "global/pre-entry HWM plus funding-credit/favorable-before-funding-debit/"
                "adverse held 5m path and two-sided cost"
            ),
            "selection_order": (
                "minimum fit/select ratio, worst 2023-half return, minimum mean net bp, "
                "combined trades, name"
            ),
            "post_selection_parameter_repair_allowed": False,
        },
        "config": asdict(cfg),
        "candidates_evaluated": len(rows),
        "candidates_passing": len(eligible),
        "winner": winner,
        "advance_to_2024_test": winner is not None,
        "candidates": rows,
    }
    report = {**stable_report, "result_hash": canonical_hash(stable_report)}
    _write_json_exclusive(cfg.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-csv", default=EvaluationConfig.market_csv)
    parser.add_argument("--funding-csv", default=EvaluationConfig.funding_csv)
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
