"""Strict veto-only backtest for frozen PPOSM SFT/RLVR gate predictions.

The OOS JSONL is not a source of trades.  It is an identity-bearing view of
the exact schedules replayed by :mod:`training.build_pposm_sft_rlvr_data`.
Predictions are consumed positionally and may only remove a frozen trade.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training import build_pposm_sft_rlvr_data as builder
from training.eval_text_label import parse_label
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.search_inventory_purge_reclaim_alpha import Trade, equity_stats

DEFAULT_OOS_DATA = Path("data/pposm_sft_rlvr_oos_2024_2026_2026-08-19.jsonl")
DEFAULT_PREDICTIONS = Path(
    "results/pposm_gate_sft_economic_rlvr_oos_predictions_2026-08-19.jsonl"
)
DEFAULT_OUTPUT = Path("results/pposm_sft_rlvr_gate_backtest_2026-08-19.json")

REPORT_WINDOWS: tuple[tuple[str, str, str, str], ...] = tuple(
    (window, window, start, end)
    for split, window, start, end in builder.SPLIT_WINDOWS
    if split == "oos"
)
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
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} in {path} is not an object")
        rows.append(value)
    return rows


def load_positional_predictions(
    path: str | Path, *, expected_length: int
) -> list[str]:
    """Load eval_text_label output without silently truncating positional zip."""

    rows = _load_jsonl(path)
    if len(rows) != expected_length:
        raise ValueError(
            "positional prediction length mismatch: "
            f"expected {expected_length}, observed {len(rows)}"
        )
    return [parse_label(str(row.get("prediction", "")), key="gate") for row in rows]


def _row_identities(rows: Sequence[dict[str, Any]]) -> list[str]:
    identities: list[str] = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("identity"), str):
            raise ValueError(f"OOS row {index} has no string metadata.identity")
        identities.append(metadata["identity"])
    if len(identities) != len(set(identities)):
        raise ValueError("OOS row identities are not unique")
    return identities


def freeze_oos_baseline(
    schedules: dict[str, Sequence[Trade]], rows: Sequence[dict[str, Any]]
) -> tuple[tuple[Trade, ...], dict[str, tuple[Trade, ...]]]:
    """Identity-lock JSONL rows to the complete replay before any vetoes."""

    by_window = {
        window: tuple(schedules[window]) for window, _, _, _ in REPORT_WINDOWS
    }
    baseline = tuple(trade for window, _, _, _ in REPORT_WINDOWS for trade in by_window[window])
    expected = [
        builder.trade_identity(window, trade)
        for window, _, _, _ in REPORT_WINDOWS
        for trade in by_window[window]
    ]
    observed = _row_identities(rows)
    if observed != expected:
        raise ValueError(
            "OOS rows do not positionally match exact frozen replay identities"
        )
    return baseline, by_window


def apply_veto_only(
    baseline: Sequence[Trade], predictions: Sequence[str]
) -> tuple[Trade, ...]:
    if len(predictions) != len(baseline):
        raise ValueError(
            "positional prediction length mismatch: "
            f"expected {len(baseline)}, observed {len(predictions)}"
        )
    selected = tuple(
        trade
        for trade, prediction in zip(baseline, predictions, strict=True)
        if prediction == "TRADE"
    )
    # Object identity makes the no-replacement invariant stronger than merely
    # comparing trade values: every selected item came from the frozen tuple.
    baseline_objects = {id(trade) for trade in baseline}
    if any(id(trade) not in baseline_objects for trade in selected):
        raise RuntimeError("veto gate introduced a replacement trade")
    return selected


def _net_returns(trades: Sequence[Trade], *, leverage: float, cost: float) -> list[float]:
    one_side = 1.0 - float(leverage) * float(cost)
    return [
        float(one_side * trade.price_factor * trade.funding_factor * one_side - 1.0)
        for trade in trades
    ]


def _return_retention(gated: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    denominator = float(baseline["absolute_return_pct"])
    if abs(denominator) <= 1e-15:
        return None
    return float(gated["absolute_return_pct"] / denominator)


def _portfolio_report(
    trades: Sequence[Trade],
    *,
    start: str,
    end: str,
    strategy_cfg: Any,
    cost: float,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    stats = equity_stats(trades, start=start, end=end, cfg=strategy_cfg, cost_rate=cost)
    returns = _net_returns(trades, leverage=strategy_cfg.leverage, cost=cost)
    signflip = weekly_cluster_sign_flip(
        returns,
        [trade.entry_date for trade in trades],
        permutations=permutations,
        seed=seed,
    )
    return {
        "selected_trade_count": len(trades),
        "equity_stats": stats,
        "one_sided_utc_week_sign_flip": signflip,
    }


def _window_report(
    baseline: Sequence[Trade],
    gated: Sequence[Trade],
    *,
    start: str,
    end: str,
    strategy_cfg: Any,
    cfg: Config,
) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    for cost_name, cost in COSTS.items():
        base_report = _portfolio_report(
            baseline,
            start=start,
            end=end,
            strategy_cfg=strategy_cfg,
            cost=cost,
            permutations=cfg.signflip_permutations,
            seed=cfg.signflip_seed,
        )
        gated_report = _portfolio_report(
            gated,
            start=start,
            end=end,
            strategy_cfg=strategy_cfg,
            cost=cost,
            permutations=cfg.signflip_permutations,
            seed=cfg.signflip_seed,
        )
        gated_report["return_retention_vs_baseline"] = _return_retention(
            gated_report["equity_stats"], base_report["equity_stats"]
        )
        costs[cost_name] = {
            "one_side_cost_rate": cost,
            "baseline": base_report,
            "gated": gated_report,
        }
    return {
        "start": start,
        "end_exclusive": end,
        "baseline_trade_count": len(baseline),
        "selected_trade_count": len(gated),
        "vetoed_trade_count": len(baseline) - len(gated),
        "costs": costs,
    }


def backtest(cfg: Config) -> dict[str, Any]:
    if cfg.signflip_permutations < 1:
        raise ValueError("signflip_permutations must be positive")
    manifest, strategy_cfg = builder.load_frozen_manifest(cfg.manifest)
    _, _, schedules = builder.replay_frozen_schedules(manifest, strategy_cfg)
    rows = _load_jsonl(cfg.oos_data)

    # The complete schedule is frozen and identity-checked before predictions
    # are even loaded.  This ordering is the core no-replacement contract.
    baseline, baseline_by_window = freeze_oos_baseline(schedules, rows)
    predictions = load_positional_predictions(
        cfg.predictions, expected_length=len(baseline)
    )
    gated = apply_veto_only(baseline, predictions)

    reports: dict[str, Any] = {}
    offset = 0
    gated_by_window: dict[str, tuple[Trade, ...]] = {}
    for window, _, start, end in REPORT_WINDOWS:
        window_baseline = baseline_by_window[window]
        window_predictions = predictions[offset : offset + len(window_baseline)]
        window_gated = apply_veto_only(window_baseline, window_predictions)
        gated_by_window[window] = window_gated
        reports[window] = _window_report(
            window_baseline,
            window_gated,
            start=start,
            end=end,
            strategy_cfg=strategy_cfg,
            cfg=cfg,
        )
        offset += len(window_baseline)
    if tuple(trade for window, _, _, _ in REPORT_WINDOWS for trade in gated_by_window[window]) != gated:
        raise RuntimeError("per-window veto schedules differ from combined veto schedule")

    combined_name, combined_start, combined_end = COMBINED_WINDOW
    reports[combined_name] = _window_report(
        baseline,
        gated,
        start=combined_start,
        end=combined_end,
        strategy_cfg=strategy_cfg,
        cfg=cfg,
    )
    output = {
        "protocol": "pposm_sft_rlvr_strict_veto_only_backtest_v1",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "prediction_counts": {
            "TRADE": predictions.count("TRADE"),
            "NO_TRADE": predictions.count("NO_TRADE"),
        },
        "invariants": {
            "full_baseline_frozen_before_prediction_load": True,
            "positional_prediction_length_exact": True,
            "row_identities_unique": True,
            "row_identities_equal_frozen_replay": True,
            "model_action": "veto_only",
            "replacement_allowed": False,
        },
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
    parser.add_argument("--manifest", type=Path, default=builder.DEFAULT_MANIFEST)
    parser.add_argument("--oos-data", type=Path, default=DEFAULT_OOS_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signflip-permutations", type=int, default=100_000)
    parser.add_argument("--signflip-seed", type=int, default=20_260_819)
    return parser.parse_args()


def main() -> None:
    report = backtest(Config(**vars(parse_args())))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
