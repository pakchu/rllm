"""Frozen pre-2024 evaluator for the minute packet topology battery."""
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
from training.preregister_minute_packet_topology_alpha import (
    CANDIDATES,
    Candidate,
    build_thresholds,
    candidate_clock,
    canonical_hash,
    load_source,
    nonoverlapping_schedule,
)
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    equity_stats,
)


PREREGISTRATION_COMMIT = "2d9d0a5"
PREREGISTRATION_SOURCE = Path("training/preregister_minute_packet_topology_alpha.py")
PREREGISTRATION_SOURCE_SHA256 = "4aeb44d58ac37a4133991f39d545bb113163059cba4ceaa0db9564f58eb8712b"
PREREGISTRATION_DOCUMENT = Path("docs/minute-packet-topology-alpha-preregistration-2026-07-19.md")
PREREGISTRATION_DOCUMENT_SHA256 = "d017cebcbebbf15bf7cf3384a6601bad8585b772380c0d6f7d217d8d2b7aebe3"
SUPPORT_RESULT = Path("results/minute_packet_topology_support_2026-07-19.json")
SUPPORT_RESULT_SHA256 = "3ba017cbd1145b09b0bc3cc58b74a732fd57445304b7d884b52f9d50bed03f7c"
EXECUTION_SOURCE = Path("training/search_inventory_purge_reclaim_alpha.py")
EXECUTION_SOURCE_SHA256 = "5d8d4df7ea79790afb919bbb481d11de33ecba5768f6e26feb1f7667cd947d65"
CLUSTER_SOURCE = Path("training/evaluate_metaorder_fragmentation_impact_curvature.py")
CLUSTER_SOURCE_SHA256 = "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
EVALUATION_SOURCE = Path("training/evaluate_minute_packet_topology_pre2024.py")
EVALUATION_FREEZE = Path("results/minute_packet_topology_evaluator_freeze_2026-07-19.json")

WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "train_2020": ("2020-01-01", "2021-01-01"),
    "train_2021": ("2021-01-01", "2022-01-01"),
    "train_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}


@dataclass(frozen=True)
class EvaluationConfig:
    feature_csv: str = (
        "/home/pakchu/rllm/data/binance_cross_venue_minute_dispersion_btc/"
        "BTCUSDT_cross_venue_minute_dispersion_5m_2020-01_2023-12.csv.gz"
    )
    market_csv: str = (
        "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    funding_csv: str = (
        "/home/pakchu/rllm/results/binance_um_btcusdt_realized_funding_2020_2023.csv"
    )
    output: str = "results/minute_packet_topology_pre2024_selection_2026-07-19.json"
    freeze_output: str = str(EVALUATION_FREEZE)
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
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_static_dependencies() -> dict[str, Any]:
    frozen = (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (CLUSTER_SOURCE, CLUSTER_SOURCE_SHA256),
    )
    for path, expected in frozen:
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"frozen dependency changed: {path}: {observed}")
    support = _read_json(SUPPORT_RESULT)
    if support.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("support stage already opened outcomes")
    if support.get("source", {}).get("sha256") != _sha256(support["source"]["path"]):
        raise ValueError("minute packet feature source changed after support freeze")
    expected_candidates = [asdict(candidate) for candidate in CANDIDATES]
    observed_candidates = [item.get("candidate") for item in support.get("candidates", [])]
    if observed_candidates != expected_candidates:
        raise ValueError("candidate grid changed after support freeze")
    return support


def freeze_evaluator(cfg: EvaluationConfig) -> dict[str, Any]:
    support = _verify_static_dependencies()
    if Path(cfg.output).exists():
        raise ValueError("selection result already exists; evaluator cannot be frozen now")
    freeze = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outcomes_opened": False,
        "opened_windows": [],
        "sealed_windows": [*WINDOWS, "test_2024", "eval_2025", "holdout_2026"],
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "evaluation_source": str(EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
        "config": asdict(cfg),
        "feature_source_sha256": support["source"]["sha256"],
        "market_source_sha256": _sha256(cfg.market_csv),
        "funding_source_sha256": _sha256(cfg.funding_csv),
        "returns_prices_or_funding_parsed": False,
        "mutable_parameters": [],
    }
    output = Path(cfg.freeze_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    return freeze


def verify_evaluator_freeze(cfg: EvaluationConfig) -> dict[str, Any]:
    freeze = _read_json(cfg.freeze_output)
    if freeze.get("outcomes_opened") is not False or freeze.get("opened_windows") != []:
        raise ValueError("evaluator was not frozen before outcomes")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("evaluator source changed after freeze")
    if freeze.get("config") != asdict(cfg):
        raise ValueError("evaluation config changed after freeze")
    if freeze.get("market_source_sha256") != _sha256(cfg.market_csv):
        raise ValueError("market outcome source changed after freeze")
    if freeze.get("funding_source_sha256") != _sha256(cfg.funding_csv):
        raise ValueError("funding outcome source changed after freeze")
    if freeze.get("returns_prices_or_funding_parsed") is not False:
        raise ValueError("freeze stage parsed outcomes")
    return freeze


def _load_outcomes(cfg: EvaluationConfig, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(cfg.market_csv, compression="infer", parse_dates=["date"])
    market = market.sort_values("date").reset_index(drop=True)
    if market.empty or market["date"].max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("market outcome source is not physically truncated before 2024")
    if not market["date"].equals(features["date"]):
        raise ValueError("official market and minute packet source grids differ")
    required = ["open", "high", "low", "close"]
    numeric = market[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all() or (numeric <= 0.0).any().any():
        raise ValueError("market outcome source contains invalid OHLC")

    funding_raw = pd.read_csv(cfg.funding_csv)
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(
                funding_raw["funding_time_utc"], utc=True, errors="raise"
            ).dt.tz_convert(None),
            "funding_rate": pd.to_numeric(funding_raw["funding_rate"], errors="raise"),
        }
    ).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if funding.empty or funding["date"].max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("funding outcome source is not physically truncated before 2024")
    if not np.isfinite(funding["funding_rate"].to_numpy(float)).all():
        raise ValueError("funding source contains non-finite rates")
    return market, funding


def _supported_candidates(support: dict[str, Any]) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    names = [
        item["name"]
        for item in support["candidates"]
        if item.get("passes_support") is True
    ]
    return [by_name[name] for name in names]


def _build_trades(
    engine: ExecutionEngine,
    schedule: pd.DataFrame,
    *,
    hold_bars: int,
    flip: bool = False,
) -> list[Trade]:
    trades: list[Trade] = []
    for row in schedule.itertuples(index=False):
        side = -int(row.side) if flip else int(row.side)
        trade = engine.trade_at(
            int(row.signal_position),
            side,
            int(hold_bars),
            1_000_000,
            1_000_000,
        )
        if trade is None:
            raise ValueError("frozen schedule exceeds the market outcome grid")
        if (
            trade.entry_position != int(row.entry_position)
            or trade.exit_position != int(row.exit_position)
        ):
            raise ValueError("execution path changed the fixed-hold schedule")
        trades.append(trade)
    return trades


def _inside_window(trade: Trade, dates: pd.Series, start: str, end: str) -> bool:
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end)
    return (
        lower <= dates.iloc[trade.signal_position] < upper
        and lower <= dates.iloc[trade.entry_position] < upper
        and lower <= dates.iloc[trade.exit_position] < upper
    )


def _window_trades(
    trades: Iterable[Trade],
    dates: pd.Series,
    window: str,
) -> list[Trade]:
    start, end = WINDOWS[window]
    return [trade for trade in trades if _inside_window(trade, dates, start, end)]


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


def _stats(
    trades: list[Trade],
    window: str,
    cfg: EvaluationConfig,
    *,
    include_cluster: bool,
    cost_rate: float | None = None,
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    output = equity_stats(trades, start=start, end=end, cfg=cfg, cost_rate=cost_rate)
    if include_cluster:
        output["weekly_cluster_sign_flip"] = weekly_cluster_sign_flip(
            _net_trade_returns(trades, cfg, cost_rate=cost_rate),
            [trade.entry_date for trade in trades],
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
    return output


def _train_coarse_gates(stats: dict[str, Any]) -> dict[str, bool]:
    return {
        "absolute_return_positive": stats["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_1_5": stats["cagr_to_strict_mdd"] >= 1.5,
        "strict_mdd_at_most_20": stats["strict_mdd_pct"] <= 20.0,
        "trades_at_least_100": stats["trades"] >= 100,
    }


def _selection_gates(
    full: dict[str, Any],
    h1: dict[str, Any],
    h2: dict[str, Any],
    stress: dict[str, Any],
) -> dict[str, bool]:
    return {
        "absolute_return_positive": full["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": full["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15": full["strict_mdd_pct"] <= 15.0,
        "each_half_absolute_return_positive": min(
            h1["absolute_return_pct"], h2["absolute_return_pct"]
        ) > 0.0,
        "each_half_trades_at_least_20": min(h1["trades"], h2["trades"]) >= 20,
        "ten_bp_stress_positive": stress["absolute_return_pct"] > 0.0,
        "weekly_cluster_p_below_0_10": full["weekly_cluster_sign_flip"][
            "p_value_one_sided"
        ] < 0.10,
    }


def evaluate(cfg: EvaluationConfig) -> dict[str, Any]:
    support = _verify_static_dependencies()
    freeze = verify_evaluator_freeze(cfg)
    features = load_source(cfg.feature_csv)
    if _sha256(cfg.feature_csv) != support["source"]["sha256"]:
        raise ValueError("feature source differs from support freeze")
    thresholds = build_thresholds(features)
    market, funding = _load_outcomes(cfg, features)
    engine = ExecutionEngine(market, funding, cfg)
    dates = market["date"]

    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate in _supported_candidates(support):
        onset, side = candidate_clock(features, thresholds, candidate)
        schedule = nonoverlapping_schedule(
            features, onset, side, hold_bars=candidate.hold_bars
        )
        support_item = next(item for item in support["candidates"] if item["name"] == candidate.name)
        if canonical_hash(schedule.to_dict(orient="records")) != support_item["clock_hash"]:
            raise ValueError(f"frozen clock changed: {candidate.name}")
        trades = _build_trades(engine, schedule, hold_bars=candidate.hold_bars)
        train_trades = _window_trades(trades, dates, "train")
        train = _stats(train_trades, "train", cfg, include_cluster=False)
        coarse = _train_coarse_gates(train)
        row: dict[str, Any] = {
            "candidate": asdict(candidate),
            "name": candidate.name,
            "clock_hash": support_item["clock_hash"],
            "train": train,
            "train_coarse_gates": coarse,
            "train_passes": False,
            "selection_opened": False,
        }
        if all(coarse.values()):
            train = _stats(train_trades, "train", cfg, include_cluster=True)
            row["train"] = train
            cluster_pass = train["weekly_cluster_sign_flip"]["p_value_one_sided"] < 0.10
            row["train_cluster_gate"] = cluster_pass
            row["train_passes"] = bool(cluster_pass)
        if row["train_passes"]:
            row["selection_opened"] = True
            windowed = {
                name: _window_trades(trades, dates, name)
                for name in (
                    "select_2023",
                    "select_2023_h1",
                    "select_2023_h2",
                )
            }
            full = _stats(windowed["select_2023"], "select_2023", cfg, include_cluster=True)
            h1 = _stats(windowed["select_2023_h1"], "select_2023_h1", cfg, include_cluster=False)
            h2 = _stats(windowed["select_2023_h2"], "select_2023_h2", cfg, include_cluster=False)
            stress = _stats(
                windowed["select_2023"],
                "select_2023",
                cfg,
                include_cluster=False,
                cost_rate=cfg.stress_cost_rate,
            )
            gates = _selection_gates(full, h1, h2, stress)
            row.update(
                {
                    "select_2023": full,
                    "select_2023_h1": h1,
                    "select_2023_h2": h2,
                    "select_2023_stress_10bp": stress,
                    "selection_gates": gates,
                    "selection_passes": bool(all(gates.values())),
                }
            )
            if row["selection_passes"]:
                ratios = [
                    train["cagr_to_strict_mdd"],
                    full["cagr_to_strict_mdd"],
                    h1["cagr_to_strict_mdd"],
                    h2["cagr_to_strict_mdd"],
                ]
                row["selection_score"] = float(min(ratios))
                eligible.append(row)
        rows.append(row)

    eligible.sort(
        key=lambda row: (
            row["selection_score"],
            -row["select_2023"]["strict_mdd_pct"],
            row["name"],
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    selected_control: dict[str, Any] | None = None
    if selected is not None:
        candidate = Candidate(**selected["candidate"])
        onset, side = candidate_clock(features, thresholds, candidate)
        schedule = nonoverlapping_schedule(features, onset, side, hold_bars=candidate.hold_bars)
        flipped = _build_trades(
            engine, schedule, hold_bars=candidate.hold_bars, flip=True
        )
        selected_control = {
            window: _stats(
                _window_trades(flipped, dates, window),
                window,
                cfg,
                include_cluster=False,
            )
            for window in ("train", "select_2023")
        }

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "pre2024_only": True,
            "oos_opened": False,
            "full_calendar_cagr": True,
            "strict_mdd": "global/pre-entry HWM plus favorable-before-adverse held 5m OHLC",
            "funding": "realized Binance USD-M settlements through held interval",
            "costs": {"base_per_side": 0.0006, "stress_per_side": 0.0010},
        },
        "config": asdict(cfg),
        "freeze_sha256": _sha256(cfg.freeze_output),
        "freeze": freeze,
        "supported_candidate_count": len(_supported_candidates(support)),
        "train_pass_count": int(sum(row["train_passes"] for row in rows)),
        "selection_pass_count": len(eligible),
        "decision": "candidate_frozen_before_oos" if selected else "rejected_before_oos",
        "selected_policy": selected["candidate"] if selected else None,
        "selected_name": selected["name"] if selected else None,
        "selected_direction_flip_control": selected_control,
        "candidates": rows,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", default=EvaluationConfig.feature_csv)
    parser.add_argument("--market-csv", default=EvaluationConfig.market_csv)
    parser.add_argument("--funding-csv", default=EvaluationConfig.funding_csv)
    parser.add_argument("--output", default=EvaluationConfig.output)
    parser.add_argument("--freeze-output", default=EvaluationConfig.freeze_output)
    parser.add_argument("--freeze", action="store_true")
    args = vars(parser.parse_args())
    freeze = bool(args.pop("freeze"))
    cfg = EvaluationConfig(**args)
    if freeze:
        result = freeze_evaluator(cfg)
        summary = {
            "freeze_output": cfg.freeze_output,
            "evaluation_source_sha256": result["evaluation_source_sha256"],
            "outcomes_opened": result["outcomes_opened"],
        }
    else:
        result = evaluate(cfg)
        summary = {
            "output": cfg.output,
            "decision": result["decision"],
            "train_pass_count": result["train_pass_count"],
            "selection_pass_count": result["selection_pass_count"],
            "selected_name": result["selected_name"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
