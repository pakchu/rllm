"""Frozen strict 2023H2 selector for the BTC BVOL/DVOL disagreement battery."""
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

from training.evaluate_metaorder_fragmentation_impact_curvature import weekly_cluster_sign_flip
from training.preregister_cross_venue_vol_disagreement_alpha import (
    CANDIDATES,
    Candidate,
    build_thresholds,
    candidate_clock,
    canonical_hash,
    load_source,
    nonoverlapping_schedule,
)
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade, equity_stats


SUPPORT_COMMIT = "43771e90b4fc03bd7041b57d3e75494322db0116"
STATIC_INPUT_SHA256 = {
    "training/preregister_cross_venue_vol_disagreement_alpha.py": "ecd0e5748d60a2d5f358209e909f94803628c88264b06bfa63f833a9426861b3",
    "docs/cross-venue-vol-disagreement-alpha-preregistration-2026-07-19.md": "835427082cd05430eaacb7623d7cb629ec272ed2a3a7661ace19babf77b295a6",
    "results/cross_venue_vol_disagreement_support_2026-07-19.json": "cc7d3b8c123ebaccf3d048f67d38609533fc7dd67f5c6b30a63e7fd3f0bea0fc",
    "training/search_inventory_purge_reclaim_alpha.py": "5d8d4df7ea79790afb919bbb481d11de33ecba5768f6e26feb1f7667cd947d65",
    "training/evaluate_metaorder_fragmentation_impact_curvature.py": "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3",
}
FEATURE_SOURCE_SHA256 = "b313360c8c1f7acc5f744a96efdfa9d6aeecb9f5d9340abc500748410a92f9a3"
MARKET_SOURCE_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
FUNDING_SOURCE_SHA256 = "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"

SUPPORT_RESULT = Path("results/cross_venue_vol_disagreement_support_2026-07-19.json")
EVALUATOR_SOURCE = Path("training/evaluate_cross_venue_vol_disagreement_pre2024.py")
EVALUATOR_FREEZE = Path(
    "results/cross_venue_vol_disagreement_evaluator_freeze_2026-07-19.json"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "select_2023_q3": ("2023-07-01", "2023-10-01"),
    "select_2023_q4": ("2023-10-01", "2024-01-01"),
}


@dataclass(frozen=True)
class EvaluationConfig:
    feature_csv: str = (
        "/home/pakchu/rllm/data/cross_venue_vol_disagreement_btc/"
        "BTC_cross_venue_vol_disagreement_1h_pre2024.csv.gz"
    )
    market_csv: str = (
        "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    funding_csv: str = (
        "/home/pakchu/rllm/results/binance_um_btcusdt_realized_funding_2020_2023.csv"
    )
    output: str = "results/cross_venue_vol_disagreement_pre2024_selection_2026-07-19.json"
    freeze_output: str = str(EVALUATOR_FREEZE)
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_719


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


def _verify_static_dependencies() -> dict[str, Any]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen dependency changed: {path}")
    support = _read_json(SUPPORT_RESULT)
    if support.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("support stage opened outcomes")
    stable = {
        "protocol": support["protocol"],
        "source": support["source"],
        "candidates": support["candidates"],
    }
    if support.get("support_freeze_hash") != canonical_hash(stable):
        raise ValueError("support freeze hash changed")
    if support.get("source", {}).get("sha256") != FEATURE_SOURCE_SHA256:
        raise ValueError("feature source identity changed in support artifact")
    if _sha256(support["source"]["path"]) != FEATURE_SOURCE_SHA256:
        raise ValueError("feature source bytes changed after support freeze")
    expected_candidates = [asdict(candidate) for candidate in CANDIDATES]
    if [item.get("candidate") for item in support["candidates"]] != expected_candidates:
        raise ValueError("candidate grid changed after support freeze")
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
        "feature_source_sha256": FEATURE_SOURCE_SHA256,
        "market_source_sha256": MARKET_SOURCE_SHA256,
        "funding_source_sha256": FUNDING_SOURCE_SHA256,
        "opened_windows": [],
        "sealed_windows": ["select_2023_h2", "test_2024", "eval_2025", "holdout_2026"],
        "pre_freeze_outcome_source_schema_validation": {
            "performed": True,
            "reported_to_selector_author": [
                "market row count and first/last timestamp",
                "funding row count and first/last timestamp",
                "funding missing-value count",
            ],
            "price_or_funding_values_reported": False,
            "candidate_trade_returns_computed": False,
        },
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
    validation = freeze.get("pre_freeze_outcome_source_schema_validation", {})
    if validation.get("price_or_funding_values_reported") is not False:
        raise ValueError("pre-freeze validation disclosed outcome values")
    if validation.get("candidate_trade_returns_computed") is not False:
        raise ValueError("candidate returns were computed before evaluator freeze")
    if freeze.get("simulation_run") is not False:
        raise ValueError("evaluator freeze ran a simulation")
    for path, expected in (
        (cfg.feature_csv, FEATURE_SOURCE_SHA256),
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
    offsets_ms = (
        pd.DatetimeIndex(timestamps).asi8 - expected.asi8
    ).astype(float) / 1_000_000.0
    if np.abs(offsets_ms).max(initial=0.0) > 60_000.0:
        raise ValueError("funding timestamp exceeds the frozen one-minute grid tolerance")
    if not np.isfinite(rates.to_numpy(float)).all():
        raise ValueError("funding source contains non-finite rates")
    return pd.DataFrame(
        {
            "date": expected.tz_convert(None),
            "funding_rate": rates.to_numpy(float),
        }
    )


def _load_outcomes(cfg: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(cfg.market_csv, compression="infer", parse_dates=["date"])
    market = market.sort_values("date").reset_index(drop=True)
    if market.empty or market["date"].max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("market source is not physically truncated before 2024")
    expected = pd.date_range(market["date"].iloc[0], market["date"].iloc[-1], freq="5min")
    if not market["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("market source is not a gapless five-minute grid")
    prices = market[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
    values = prices.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("market source contains invalid prices")
    if (
        (prices["high"] < prices[["open", "close"]].max(axis=1)).any()
        or (prices["low"] > prices[["open", "close"]].min(axis=1)).any()
        or (prices["high"] < prices["low"]).any()
    ):
        raise ValueError("market source violates OHLC invariants")
    market[["open", "high", "low", "close"]] = prices

    raw = pd.read_csv(cfg.funding_csv)
    funding = _validated_funding_frame(raw)
    return market, funding


def _supported_candidates(support: dict[str, Any]) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    return [
        by_name[item["name"]]
        for item in support["candidates"]
        if item.get("passes_support") is True
    ]


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
        signal_time = pd.Timestamp(row.signal_time)
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
        if pd.Timestamp(engine.dates.iloc[trade.entry_position]) != pd.Timestamp(row.entry_time):
            raise ValueError("execution entry differs from frozen five-minute delay")
        if pd.Timestamp(engine.dates.iloc[trade.exit_position]) != pd.Timestamp(row.exit_time):
            raise ValueError("execution exit differs from frozen elapsed hold")
        trades.append(trade)
    return trades


def _window_trades(
    trades: Iterable[Trade], market_dates: pd.Series, window: str
) -> list[Trade]:
    start, end = (pd.Timestamp(value) for value in WINDOWS[window])
    return [
        trade
        for trade in trades
        if start <= market_dates.iloc[trade.signal_position] < end
        and start <= market_dates.iloc[trade.entry_position] < end
        and start < market_dates.iloc[trade.exit_position] <= end
    ]


def _net_trade_returns(
    trades: Iterable[Trade], cfg: EvaluationConfig, *, cost_rate: float | None = None
) -> list[float]:
    cost = cfg.fee_rate + cfg.slippage_rate if cost_rate is None else float(cost_rate)
    execution = 1.0 - cfg.leverage * cost
    return [
        float(execution * trade.price_factor * trade.funding_factor * execution - 1.0)
        for trade in trades
    ]


def _stats(
    trades: list[Trade], window: str, cfg: EvaluationConfig, *, cost_rate: float | None = None
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    output = equity_stats(trades, start=start, end=end, cfg=cfg, cost_rate=cost_rate)
    output["weekly_cluster_sign_flip"] = weekly_cluster_sign_flip(
        _net_trade_returns(trades, cfg, cost_rate=cost_rate),
        [trade.entry_date for trade in trades],
        permutations=cfg.cluster_permutations,
        seed=cfg.cluster_seed,
    )
    return output


def selection_gates(
    full: dict[str, Any],
    q3: dict[str, Any],
    q4: dict[str, Any],
    stress: dict[str, Any],
    flip: dict[str, Any],
) -> dict[str, bool]:
    return {
        "absolute_return_positive": full["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": full["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": full["strict_mdd_pct"] <= 15.0,
        "trades_at_least_24": full["trades"] >= 24,
        "mean_gross_at_least_25bp": full["mean_gross_bps"] >= 25.0,
        "weekly_cluster_p_below_0_10": full["weekly_cluster_sign_flip"][
            "p_value_one_sided"
        ] < 0.10,
        "each_quarter_positive": min(q3["absolute_return_pct"], q4["absolute_return_pct"]) > 0.0,
        "each_quarter_at_least_8_trades": min(q3["trades"], q4["trades"]) >= 8,
        "ten_bp_per_side_stress_positive": stress["absolute_return_pct"] > 0.0,
        "direction_flip_absolute_return_negative": flip["absolute_return_pct"] < 0.0,
    }


def winner_sort_key(row: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        -min(row["q3"]["absolute_return_pct"], row["q4"]["absolute_return_pct"]),
        -row["selection"]["cagr_to_strict_mdd"],
        -row["selection"]["mean_net_bps"],
        -row["selection"]["trades"],
        row["name"],
    )


def evaluate(cfg: EvaluationConfig) -> dict[str, Any]:
    _require_canonical_artifact_paths(cfg)
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists and cannot be replaced")
    support = _verify_static_dependencies()
    freeze = verify_evaluator_freeze(cfg)
    features = load_source(cfg.feature_csv)
    thresholds = build_thresholds(features)
    market, funding = _load_outcomes(cfg)
    engine = ExecutionEngine(market, funding, cfg)
    market_positions = {timestamp: position for position, timestamp in enumerate(engine.dates)}

    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate in _supported_candidates(support):
        onset, side = candidate_clock(features, thresholds, candidate)
        schedule = nonoverlapping_schedule(
            features, onset, side, hold_hours=candidate.hold_hours
        )
        support_item = next(item for item in support["candidates"] if item["name"] == candidate.name)
        if canonical_hash(schedule.to_dict(orient="records")) != support_item["clock_hash"]:
            raise ValueError(f"frozen clock changed: {candidate.name}")
        trades = _build_trades(engine, market_positions, schedule, candidate)
        flipped = _build_trades(engine, market_positions, schedule, candidate, flip=True)
        windowed = {
            name: _window_trades(trades, engine.dates, name) for name in WINDOWS
        }
        flip_windowed = _window_trades(flipped, engine.dates, "select_2023_h2")
        selection = _stats(windowed["select_2023_h2"], "select_2023_h2", cfg)
        q3 = _stats(windowed["select_2023_q3"], "select_2023_q3", cfg)
        q4 = _stats(windowed["select_2023_q4"], "select_2023_q4", cfg)
        stress = _stats(
            windowed["select_2023_h2"],
            "select_2023_h2",
            cfg,
            cost_rate=cfg.stress_cost_rate,
        )
        flip_stats = _stats(flip_windowed, "select_2023_h2", cfg)
        gates = selection_gates(selection, q3, q4, stress, flip_stats)
        row = {
            "candidate": asdict(candidate),
            "name": candidate.name,
            "clock_hash": support_item["clock_hash"],
            "selection": selection,
            "q3": q3,
            "q4": q4,
            "stress_10bp_per_side": stress,
            "direction_flip": flip_stats,
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
            "opened_windows": ["select_2023_h2"],
            "sealed_windows": ["test_2024", "eval_2025", "holdout_2026"],
            "full_calendar_cagr": True,
            "strict_mdd": "global HWM plus held 5m high/low path, two-sided cost, funding debit",
            "selection_order": "worst-quarter absolute return, full ratio, mean net bp, trades, name",
            "post_selection_parameter_repair_allowed": False,
        },
        "config": asdict(cfg),
        "candidates_evaluated": len(rows),
        "candidates_passing": len(eligible),
        "winner": None if winner is None else {
            "candidate": winner["candidate"],
            "name": winner["name"],
            "clock_hash": winner["clock_hash"],
            "selection": winner["selection"],
            "q3": winner["q3"],
            "q4": winner["q4"],
            "stress_10bp_per_side": winner["stress_10bp_per_side"],
            "direction_flip": winner["direction_flip"],
        },
        "advance_to_2024_test": winner is not None,
        "candidates": rows,
    }
    report = {**stable_report, "result_hash": canonical_hash(stable_report)}
    _write_json_exclusive(cfg.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", default=EvaluationConfig.feature_csv)
    parser.add_argument("--market-csv", default=EvaluationConfig.market_csv)
    parser.add_argument("--funding-csv", default=EvaluationConfig.funding_csv)
    parser.add_argument("--output", default=EvaluationConfig.output)
    parser.add_argument("--freeze-output", default=EvaluationConfig.freeze_output)
    parser.add_argument("--freeze-only", action="store_true")
    args = vars(parser.parse_args())
    freeze_only = args.pop("freeze_only")
    cfg = EvaluationConfig(**args)
    report = freeze_evaluator(cfg) if freeze_only else evaluate(cfg)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    ["freeze_hash"]
                    if freeze_only
                    else ["candidates_evaluated", "candidates_passing", "advance_to_2024_test"]
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
