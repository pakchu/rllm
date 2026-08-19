"""Backtest positional SKIP/TP4/TP12 predictions on frozen PPOSM decisions."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_pposm_state_router_data as builder
from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade, equity_stats
from training import search_pullback_premium_overheat_state_machine_alpha as pposm
from training.evaluate_metaorder_fragmentation_impact_curvature import weekly_cluster_sign_flip

DEFAULT_OOS_DATA = Path("data/pposm_state_router_oos_2024_2026.jsonl")
DEFAULT_PREDICTIONS = Path("results/pposm_state_router_oos_predictions.jsonl")
DEFAULT_OUTPUT = Path("results/pposm_state_router_backtest.json")

REPORT_WINDOWS = tuple(item for item in builder.SPLIT_WINDOWS if item[0] == "oos")
COMBINED_WINDOW = ("combined_2024_2026_06_02", "2024-01-01", "2026-06-02")
COSTS = {"base_6bp": 0.0006, "stress_10bp": 0.0010}


@dataclass(frozen=True)
class Config:
    manifest: Path = builder.DEFAULT_MANIFEST
    oos_data: Path = DEFAULT_OOS_DATA
    predictions: Path = DEFAULT_PREDICTIONS
    output: Path = DEFAULT_OUTPUT
    signflip_permutations: int = 100_000
    signflip_seed: int = 20_260_819


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {number} in {path} is not an object")
        rows.append(value)
    return rows


def parse_route(value: str) -> str:
    raw = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).split()
    matches = [route for route in builder.ROUTES if route in raw]
    if len(matches) != 1:
        raise ValueError(f"prediction must contain exactly one route token: {value!r}")
    return matches[0]


def load_positional_predictions(path: str | Path, *, expected_length: int) -> list[str]:
    rows = _load_jsonl(path)
    if len(rows) != expected_length:
        raise ValueError(
            f"positional prediction length mismatch: expected {expected_length}, observed {len(rows)}"
        )
    return [parse_route(str(row.get("prediction", ""))) for row in rows]


def lock_oos_rows(
    rows: Sequence[dict[str, Any]], positions: dict[str, tuple[int, ...]]
) -> tuple[int, ...]:
    expected_positions = tuple(
        signal for _, window, _, _ in REPORT_WINDOWS for signal in positions[window]
    )
    expected_ids = [
        builder.signal_identity(window, signal)
        for _, window, _, _ in REPORT_WINDOWS
        for signal in positions[window]
    ]
    observed_ids: list[str] = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("identity"), str):
            raise ValueError(f"OOS row {index} has no string metadata.identity")
        observed_ids.append(metadata["identity"])
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("OOS row identities are not unique")
    if observed_ids != expected_ids:
        raise ValueError("OOS rows do not positionally match all frozen active decisions")
    return expected_positions


def deterministic_routes(state: pd.DataFrame, signals: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        builder.route_label(
            capitulation=bool(state.iloc[signal]["capitulation"]),
            overheat=bool(state.iloc[signal]["overheat"]),
        )
        for signal in signals
    )


def apply_routes(
    engine: ExecutionEngine,
    signals: Sequence[int],
    routes: Sequence[str],
    *,
    start: str,
    end: str,
) -> tuple[Trade, ...]:
    """Consume every route positionally, then apply lifecycle sequentially."""

    if len(signals) != len(routes):
        raise ValueError("signals and routes must have equal length")
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    take_bps = {"TP4": 400, "TP12": 1_200}
    for signal, route in zip(signals, routes, strict=True):
        signal = int(signal)
        if route not in builder.ROUTES:
            raise ValueError(f"unsupported route: {route}")
        if signal < next_allowed or route == "SKIP":
            continue
        trade = engine.trade_at(
            signal,
            int(pposm.SPEC["side"]),
            int(pposm.SPEC["hold_bars"]),
            take_bps[route],
            int(pposm.SPEC["stop_bps"]),
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    if any(right.entry_position <= left.exit_position for left, right in zip(trades, trades[1:])):
        raise RuntimeError("route simulation produced overlapping trades")
    return tuple(trades)


def _trade_key(trade: Trade, route: str) -> tuple[int, int, int, str]:
    return (trade.signal_position, trade.entry_position, trade.exit_position, route)


def _scheduled_keys(
    trades: Sequence[Trade], route_by_signal: dict[int, str]
) -> list[tuple[int, int, int, str]]:
    return [_trade_key(trade, route_by_signal[trade.signal_position]) for trade in trades]


def _economics(
    trades: Sequence[Trade], *, start: str, end: str, strategy_cfg: Any,
    signflip_permutations: int, signflip_seed: int,
) -> dict[str, Any]:
    output = {}
    for name, cost in COSTS.items():
        one_side = 1.0 - float(strategy_cfg.leverage) * cost
        returns = [
            float(one_side * trade.price_factor * trade.funding_factor * one_side - 1.0)
            for trade in trades
        ]
        output[name] = {
            "one_side_cost_rate": cost,
            "equity_stats": equity_stats(
                trades, start=start, end=end, cfg=strategy_cfg, cost_rate=cost
            ),
            "one_sided_utc_week_sign_flip": weekly_cluster_sign_flip(
                returns,
                [trade.entry_date for trade in trades],
                permutations=signflip_permutations,
                seed=signflip_seed,
            ),
        }
    return output


def _agreement(predicted: Sequence[str], baseline: Sequence[str]) -> dict[str, Any]:
    if len(predicted) != len(baseline):
        raise ValueError("agreement streams must have equal length")
    matches = sum(left == right for left, right in zip(predicted, baseline, strict=True))
    confusion = Counter(
        f"baseline={truth}|prediction={prediction}"
        for prediction, truth in zip(predicted, baseline, strict=True)
    )
    return {
        "decisions": len(baseline),
        "matching_decisions": matches,
        "decision_agreement_rate": matches / len(baseline) if baseline else 1.0,
        "confusion": dict(sorted(confusion.items())),
    }


def _window_report(
    engine: ExecutionEngine,
    signals: Sequence[int],
    predictions: Sequence[str],
    baseline_routes: Sequence[str],
    *,
    start: str,
    end: str,
    strategy_cfg: Any,
    signflip_permutations: int,
    signflip_seed: int,
) -> tuple[dict[str, Any], tuple[Trade, ...], tuple[Trade, ...]]:
    predicted = apply_routes(engine, signals, predictions, start=start, end=end)
    baseline = apply_routes(engine, signals, baseline_routes, start=start, end=end)
    predicted_map = dict(zip(signals, predictions, strict=True))
    baseline_map = dict(zip(signals, baseline_routes, strict=True))
    predicted_keys = _scheduled_keys(predicted, predicted_map)
    baseline_keys = _scheduled_keys(baseline, baseline_map)
    common = len(set(predicted_keys).intersection(baseline_keys))
    agreement = _agreement(predictions, baseline_routes)
    agreement["schedule"] = {
        "exact_sequence_match": predicted_keys == baseline_keys,
        "baseline_trade_count": len(baseline_keys),
        "predicted_trade_count": len(predicted_keys),
        "common_trade_count": common,
        "recall_vs_baseline": common / len(baseline_keys) if baseline_keys else 1.0,
        "precision_vs_baseline": common / len(predicted_keys) if predicted_keys else (1.0 if not baseline_keys else 0.0),
    }
    return (
        {
            "start": start,
            "end_exclusive": end,
            "route_counts": {
                "baseline": dict(sorted(Counter(baseline_routes).items())),
                "predicted": dict(sorted(Counter(predictions).items())),
            },
            "agreement": agreement,
            "economics": {
                "baseline": _economics(baseline, start=start, end=end, strategy_cfg=strategy_cfg, signflip_permutations=signflip_permutations, signflip_seed=signflip_seed),
                "predicted": _economics(predicted, start=start, end=end, strategy_cfg=strategy_cfg, signflip_permutations=signflip_permutations, signflip_seed=signflip_seed),
            },
        },
        baseline,
        predicted,
    )


def backtest(cfg: Config) -> dict[str, Any]:
    manifest, strategy_cfg = builder.frozen.load_frozen_manifest(cfg.manifest)
    market, funding, state, active = builder.replay_frozen_decisions(manifest, strategy_cfg)
    positions = builder.decision_positions(market, active)
    rows = _load_jsonl(cfg.oos_data)
    all_signals = lock_oos_rows(rows, positions)
    predictions = load_positional_predictions(cfg.predictions, expected_length=len(all_signals))
    engine_cfg = _execution_config(strategy_cfg, strategy_cfg.leverage)
    engine = ExecutionEngine(market, funding, engine_cfg)

    reports: dict[str, Any] = {}
    combined_baseline: list[Trade] = []
    combined_predicted: list[Trade] = []
    combined_baseline_routes: list[str] = []
    offset = 0
    for _, window, start, end in REPORT_WINDOWS:
        signals = positions[window]
        routes = deterministic_routes(state, signals)
        window_predictions = predictions[offset : offset + len(signals)]
        report, baseline_trades, predicted_trades = _window_report(
            engine,
            signals,
            window_predictions,
            routes,
            start=start,
            end=end,
            strategy_cfg=engine_cfg,
            signflip_permutations=cfg.signflip_permutations,
            signflip_seed=cfg.signflip_seed,
        )
        reports[window] = report
        combined_baseline.extend(baseline_trades)
        combined_predicted.extend(predicted_trades)
        combined_baseline_routes.extend(routes)
        offset += len(signals)
    if offset != len(predictions):
        raise RuntimeError("not all positional predictions were consumed")

    combined_name, combined_start, combined_end = COMBINED_WINDOW
    predicted_map = dict(zip(all_signals, predictions, strict=True))
    baseline_map = dict(zip(all_signals, combined_baseline_routes, strict=True))
    predicted_keys = _scheduled_keys(combined_predicted, predicted_map)
    baseline_keys = _scheduled_keys(combined_baseline, baseline_map)
    common = len(set(predicted_keys).intersection(baseline_keys))
    combined_agreement = _agreement(predictions, combined_baseline_routes)
    combined_agreement["schedule"] = {
        "exact_sequence_match": predicted_keys == baseline_keys,
        "baseline_trade_count": len(baseline_keys),
        "predicted_trade_count": len(predicted_keys),
        "common_trade_count": common,
        "recall_vs_baseline": common / len(baseline_keys) if baseline_keys else 1.0,
        "precision_vs_baseline": common / len(predicted_keys) if predicted_keys else (1.0 if not baseline_keys else 0.0),
    }
    reports[combined_name] = {
        "start": combined_start,
        "end_exclusive": combined_end,
        "route_counts": {
            "baseline": dict(sorted(Counter(combined_baseline_routes).items())),
            "predicted": dict(sorted(Counter(predictions).items())),
        },
        "agreement": combined_agreement,
        "economics": {
            "baseline": _economics(combined_baseline, start=combined_start, end=combined_end, strategy_cfg=engine_cfg, signflip_permutations=cfg.signflip_permutations, signflip_seed=cfg.signflip_seed),
            "predicted": _economics(combined_predicted, start=combined_start, end=combined_end, strategy_cfg=engine_cfg, signflip_permutations=cfg.signflip_permutations, signflip_seed=cfg.signflip_seed),
        },
    }
    output = {
        "protocol": "pposm_state_router_positional_backtest_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "invariants": {
            "all_active_decisions_consumed_positionally": True,
            "entry_rule": "exact_next_5m_open",
            "lifecycle": "TP_or_48h_cap",
            "non_overlapping": True,
            "future_return_used_for_route": False,
        },
        "windows": reports,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=builder.DEFAULT_MANIFEST)
    parser.add_argument("--oos-data", type=Path, default=DEFAULT_OOS_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signflip-permutations", type=int, default=100_000)
    parser.add_argument("--signflip-seed", type=int, default=20_260_819)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(backtest(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
