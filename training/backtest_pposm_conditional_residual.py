"""Backtest frozen residual-router routes against an ALWAYS-TP4 control."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_pposm_counterfactual_action_data as counterfactual
from training import build_pposm_residual_action_data as residual
from training import build_pposm_state_router_data as state_builder
from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.evaluate_metaorder_fragmentation_impact_curvature import weekly_cluster_sign_flip
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade, equity_stats

DEFAULT_OOS_DATA = Path(
    "data/pposm_residual_action_oos_2024_2026_2026-09-02.jsonl"
)
DEFAULT_ROUTE_PREDICTIONS = Path(
    "results/pposm_conditional_residual_oos_predictions_2026-09-02.jsonl"
)
DEFAULT_OUTPUT = Path("results/pposm_conditional_residual_backtest_2026-09-02.json")
REPORT_WINDOWS = tuple(item for item in state_builder.SPLIT_WINDOWS if item[0] == "oos")
COMBINED_WINDOW = ("combined_2024_2026_06_02", "2024-01-01", "2026-06-02")
COSTS = {"base_6bp": 0.0006, "stress_10bp": 0.0010}
ROUTES = (residual.DEFAULT_ACTION, *residual.CANDIDATE_ACTIONS)


@dataclass(frozen=True)
class Config:
    manifest: Path = counterfactual.DEFAULT_MANIFEST
    oos_data: Path = DEFAULT_OOS_DATA
    route_predictions: Path = DEFAULT_ROUTE_PREDICTIONS
    output: Path = DEFAULT_OUTPUT
    signflip_permutations: int = 100_000
    signflip_seed: int = 20_260_902


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {number} in {path} is not an object")
        rows.append(value)
    return rows


def parse_route(value: str) -> str:
    tokens = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).split()
    matches = [route for route in ROUTES if route in tokens]
    if len(matches) != 1:
        raise ValueError(f"prediction must contain exactly one route token: {value!r}")
    return matches[0]


def lock_pair_rows(
    pair_rows: Sequence[dict[str, Any]], positions: dict[str, tuple[int, ...]]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    expected_signals: list[int] = []
    expected_base_ids: list[str] = []
    expected_pair_ids: list[str] = []
    for _, window, _, _ in REPORT_WINDOWS:
        for signal in positions[window]:
            base = counterfactual.signal_identity(window, signal)
            expected_signals.append(int(signal))
            expected_base_ids.append(base)
            expected_pair_ids.extend(
                residual.residual_identity(base, candidate)
                for candidate in residual.CANDIDATE_ACTIONS
            )

    observed: list[str] = []
    for index, row in enumerate(pair_rows):
        metadata = row.get("metadata")
        identity = metadata.get("identity") if isinstance(metadata, dict) else None
        if not isinstance(identity, str):
            raise ValueError(f"pair row {index} has no metadata.identity")
        observed.append(identity)
    if observed != expected_pair_ids:
        raise ValueError(
            "OOS residual pair rows do not positionally match frozen active decisions"
        )
    return tuple(expected_signals), tuple(expected_base_ids)


def load_routes(
    path: str | Path, *, expected_base_ids: Sequence[str]
) -> tuple[str, ...]:
    rows = _load_jsonl(path)
    if len(rows) != len(expected_base_ids):
        raise ValueError(
            "route prediction length mismatch: "
            f"expected {len(expected_base_ids)}, observed {len(rows)}"
        )
    routes: list[str] = []
    for index, (row, expected_identity) in enumerate(
        zip(rows, expected_base_ids, strict=True)
    ):
        identity = row.get("base_identity", row.get("identity"))
        if identity != expected_identity:
            raise ValueError(
                f"route prediction {index} identity does not match frozen decision"
            )
        routes.append(parse_route(str(row.get("prediction", ""))))
    return tuple(routes)


def _apply_routes(
    engine: ExecutionEngine,
    signals: Sequence[int],
    routes: Sequence[str],
    *,
    start: str,
    end: str,
    spec: dict[str, Any],
) -> tuple[Trade, ...]:
    if len(signals) != len(routes):
        raise ValueError("signals and routes must have equal length")
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
        bool
    )
    trades: list[Trade] = []
    next_allowed = 0
    take_bps = {
        "TP4": int(spec["capitulation_take_bps"]),
        "TP12": int(spec["normal_take_bps"]),
    }
    for signal, route in zip(signals, routes, strict=True):
        signal = int(signal)
        if route not in ROUTES:
            raise ValueError(f"unsupported route: {route}")
        if signal < next_allowed or route == "SKIP":
            continue
        trade = engine.trade_at(
            signal,
            int(spec["side"]),
            int(spec["hold_bars"]),
            take_bps[route],
            int(spec["stop_bps"]),
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    if any(
        right.entry_position <= left.exit_position
        for left, right in zip(trades, trades[1:])
    ):
        raise RuntimeError("route simulation produced overlapping trades")
    return tuple(trades)


def _economics(
    trades: Sequence[Trade],
    *,
    start: str,
    end: str,
    cfg: Any,
    signflip_permutations: int,
    signflip_seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, cost in COSTS.items():
        one_side = 1.0 - float(cfg.leverage) * cost
        returns = [
            float(one_side * trade.price_factor * trade.funding_factor * one_side - 1.0)
            for trade in trades
        ]
        output[name] = {
            "one_side_cost_rate": cost,
            "equity_stats": equity_stats(
                trades, start=start, end=end, cfg=cfg, cost_rate=cost
            ),
            "one_sided_utc_week_sign_flip": weekly_cluster_sign_flip(
                returns,
                [trade.entry_date for trade in trades],
                permutations=signflip_permutations,
                seed=signflip_seed,
            ),
        }
    return output


def _replay_digest(
    *, signals: Sequence[int], routes: Sequence[str], trades: Sequence[Trade]
) -> str:
    payload = {
        "signals": [int(value) for value in signals],
        "routes": list(routes),
        "trades": [
            {
                "signal": int(trade.signal_position),
                "entry": int(trade.entry_position),
                "exit": int(trade.exit_position),
            }
            for trade in trades
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_report(
    engine: ExecutionEngine,
    signals: Sequence[int],
    predicted_routes: Sequence[str],
    *,
    start: str,
    end: str,
    cfg: Any,
    signflip_permutations: int,
    signflip_seed: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    baseline_routes = tuple("TP4" for _ in signals)
    predicted_trades = _apply_routes(
        engine, signals, predicted_routes, start=start, end=end, spec=spec
    )
    baseline_trades = _apply_routes(
        engine, signals, baseline_routes, start=start, end=end, spec=spec
    )
    matches = sum(
        predicted == baseline
        for predicted, baseline in zip(predicted_routes, baseline_routes, strict=True)
    )
    return {
        "start": start,
        "end_exclusive": end,
        "route_counts": {
            "baseline": dict(sorted(Counter(baseline_routes).items())),
            "predicted": dict(sorted(Counter(predicted_routes).items())),
        },
        "agreement": {
            "decisions": len(signals),
            "matching_decisions": matches,
            "decision_agreement_rate": matches / len(signals) if signals else 1.0,
        },
        "trade_counts": {
            "baseline": len(baseline_trades),
            "predicted": len(predicted_trades),
        },
        "economics": {
            "baseline": _economics(
                baseline_trades,
                start=start,
                end=end,
                cfg=cfg,
                signflip_permutations=signflip_permutations,
                signflip_seed=signflip_seed,
            ),
            "predicted": _economics(
                predicted_trades,
                start=start,
                end=end,
                cfg=cfg,
                signflip_permutations=signflip_permutations,
                signflip_seed=signflip_seed,
            ),
        },
        "replay_digest": {
            "baseline": _replay_digest(
                signals=signals, routes=baseline_routes, trades=baseline_trades
            ),
            "predicted": _replay_digest(
                signals=signals, routes=predicted_routes, trades=predicted_trades
            ),
        },
    }


def backtest(cfg: Config) -> dict[str, Any]:
    manifest, strategy_cfg = counterfactual.frozen.load_frozen_manifest(cfg.manifest)
    market, funding, _, active = state_builder.replay_frozen_decisions(
        manifest, strategy_cfg
    )
    positions = state_builder.decision_positions(market, active)
    pair_rows = _load_jsonl(cfg.oos_data)
    all_signals, expected_base_ids = lock_pair_rows(pair_rows, positions)
    routes = load_routes(cfg.route_predictions, expected_base_ids=expected_base_ids)
    spec = manifest["spec"]
    engine_cfg = _execution_config(strategy_cfg, strategy_cfg.leverage)
    engine = ExecutionEngine(market, funding, engine_cfg)

    reports: dict[str, Any] = {}
    offset = 0
    for _, window, start, end in REPORT_WINDOWS:
        signals = positions[window]
        window_routes = routes[offset : offset + len(signals)]
        reports[window] = _window_report(
            engine,
            signals,
            window_routes,
            start=start,
            end=end,
            cfg=engine_cfg,
            signflip_permutations=cfg.signflip_permutations,
            signflip_seed=cfg.signflip_seed,
            spec=spec,
        )
        offset += len(signals)
    if offset != len(routes):
        raise RuntimeError("not all route predictions were consumed")

    combined_name, combined_start, combined_end = COMBINED_WINDOW
    reports[combined_name] = _window_report(
        engine,
        all_signals,
        routes,
        start=combined_start,
        end=combined_end,
        cfg=engine_cfg,
        signflip_permutations=cfg.signflip_permutations,
        signflip_seed=cfg.signflip_seed,
        spec=spec,
    )
    output = {
        "protocol": "pposm_conditional_residual_tp4_default_backtest_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "execution_spec": {
            key: spec[key]
            for key in (
                "side",
                "hold_bars",
                "stop_bps",
                "capitulation_take_bps",
                "normal_take_bps",
            )
        },
        "invariants": {
            "baseline": "ALWAYS_TP4",
            "entry_rule": "exact_next_5m_open",
            "lifecycle": "TP_or_48h_cap",
            "funding_applied": True,
            "non_overlapping": True,
            "all_active_oos_decisions_consumed_positionally": True,
            "future_return_used_for_route": False,
        },
        "route_counts": dict(sorted(Counter(routes).items())),
        "windows": reports,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=counterfactual.DEFAULT_MANIFEST)
    parser.add_argument("--oos-data", type=Path, default=DEFAULT_OOS_DATA)
    parser.add_argument(
        "--route-predictions", type=Path, default=DEFAULT_ROUTE_PREDICTIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signflip-permutations", type=int, default=100_000)
    parser.add_argument("--signflip-seed", type=int, default=20_260_902)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(backtest(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
