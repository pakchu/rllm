"""Sweep simple conjunction gates over fixed event candidate rows.

This is a leakage-guarded regime-filter probe. Candidate rows already contain
signal-time feature snapshots. Primitive gates are formed from train quantiles,
then one- and two-gate conjunctions are selected using train+test only. Eval is
reported only after ranking.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_action_backtest import EconomicActionBacktestConfig, strict_backtest_actions
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class Gate:
    feature: str
    op: str
    threshold: float

    def match(self, row: dict[str, Any]) -> bool:
        val = row.get("_fs", {}).get(self.feature)
        if not isinstance(val, (int, float)) or not np.isfinite(float(val)):
            return False
        x = float(val)
        return x >= self.threshold if self.op == ">=" else x <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "threshold": self.threshold}


def load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        action = row["action"]
        rows.append(
            {
                **row,
                "prediction": {
                    "gate": "TRADE",
                    "family": action["family"],
                    "side": action["side"],
                    "hold_bars": action["hold_bars"],
                },
                "_fs": row.get("feature_snapshot", {}),
            }
        )
    return rows


def _pred_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"date": r["date"], "signal_pos": r["signal_pos"], "prediction": r["prediction"]} for r in rows]


def backtest(
    rows: list[dict[str, Any]],
    market: Any,
    leverage: float,
    *,
    annualization_start: str = "",
    annualization_end: str = "",
) -> dict[str, Any] | None:
    if not rows:
        return None
    return strict_backtest_actions(
        _pred_rows(rows),
        market,
        EconomicActionBacktestConfig(
            leverage=leverage,
            fee_rate=0.0004,
            slippage_rate=0.0001,
            entry_delay_bars=1,
            max_hold_bars=144,
            annualization_start=annualization_start,
            annualization_end=annualization_end,
        ),
    )


def filter_rows(rows: list[dict[str, Any]], gates: tuple[Gate, ...]) -> list[dict[str, Any]]:
    return [row for row in rows if all(g.match(row) for g in gates)]


def metric_score(result: dict[str, Any]) -> float:
    sim = result["sim"]
    stats = result["trade_stats"]
    p = float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
    return (
        float(sim["cagr_to_strict_mdd"])
        + 0.02 * float(sim["cagr_pct"])
        + 0.001 * float(sim["trade_entries"])
        - 0.5 * max(0.0, p - 0.1)
        - 0.25 * max(0.0, float(sim["strict_mdd_pct"]) - 15.0)
    )


def _numeric_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key, value in row.get("_fs", {}).items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                names.add(key)
    return sorted(names)


def make_primitives(train_rows: list[dict[str, Any]], quantiles: tuple[float, ...]) -> list[Gate]:
    gates: list[Gate] = []
    for feature in _numeric_feature_names(train_rows):
        vals = np.array([float(r["_fs"].get(feature)) for r in train_rows if isinstance(r["_fs"].get(feature), (int, float)) and np.isfinite(float(r["_fs"].get(feature)))], dtype=float)
        if vals.size < 50 or np.nanstd(vals) <= 1e-12:
            continue
        for q in quantiles:
            threshold = float(np.quantile(vals, q))
            for op in (">=", "<="):
                gates.append(Gate(feature=feature, op=op, threshold=threshold))
    return gates


def _period_bounds(rows: list[dict[str, Any]]) -> tuple[str, str]:
    dates = sorted(str(row["date"]) for row in rows if str(row.get("date", "")).strip())
    return (dates[0], dates[-1]) if dates else ("", "")


def evaluate_gate_set(
    gates: tuple[Gate, ...],
    sets: dict[str, list[dict[str, Any]]],
    market: Any,
    leverage: float,
    period_bounds: dict[str, tuple[str, str]],
) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for split, rows in sets.items():
        selected = filter_rows(rows, gates)
        start, end = period_bounds.get(split, ("", ""))
        result = backtest(selected, market, leverage, annualization_start=start, annualization_end=end)
        if result is None:
            return None
        out[split] = {"n_rows": len(selected), "sim": result["sim"], "trade_stats": result["trade_stats"]}
    return out


def _gate_key(gates: tuple[Gate, ...]) -> tuple[tuple[str, str, float], ...]:
    return tuple((g.feature, g.op, round(float(g.threshold), 12)) for g in gates)


def run(
    *,
    train_jsonl: str,
    test_jsonl: str,
    eval_jsonl: str,
    market_csv: str,
    output: str,
    max_primitives: int = 40,
    max_width: int = 2,
    leverage_grid: str = "0.5,1.0,1.25,1.5,2.0",
    side_filter: str = "",
    family_filter: str = "",
    min_train_trades: int = 120,
    min_test_trades: int = 20,
    train_start: str = "",
    train_end: str = "",
    test_start: str = "",
    test_end: str = "",
    eval_start: str = "",
    eval_end: str = "",
) -> dict[str, Any]:
    market = load_market_bars(market_csv)
    sets = {"train": load_rows(train_jsonl), "test": load_rows(test_jsonl), "eval": load_rows(eval_jsonl)}
    side_filter = str(side_filter).strip().upper()
    family_filter = str(family_filter).strip()
    if side_filter:
        sets = {split: [row for row in rows if str(row.get("action", {}).get("side", "")).upper() == side_filter] for split, rows in sets.items()}
    if family_filter:
        sets = {split: [row for row in rows if str(row.get("action", {}).get("family", "")) == family_filter] for split, rows in sets.items()}
    period_bounds = {
        "train": (train_start, train_end) if train_start and train_end else _period_bounds(sets["train"]),
        "test": (test_start, test_end) if test_start and test_end else _period_bounds(sets["test"]),
        "eval": (eval_start, eval_end) if eval_start and eval_end else _period_bounds(sets["eval"]),
    }
    primitives = make_primitives(sets["train"], (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))

    primitive_rows: list[dict[str, Any]] = []
    for gate in primitives:
        metrics = evaluate_gate_set((gate,), sets, market, 0.5, period_bounds)
        if metrics is None:
            continue
        if metrics["train"]["sim"]["trade_entries"] < int(min_train_trades) or metrics["test"]["sim"]["trade_entries"] < int(min_test_trades):
            continue
        if metrics["train"]["sim"]["cagr_pct"] <= 0 or metrics["test"]["sim"]["cagr_pct"] <= 0:
            continue
        if metrics["train"]["sim"]["strict_mdd_pct"] > 25 or metrics["test"]["sim"]["strict_mdd_pct"] > 15:
            continue
        primitive_rows.append({"gates": (gate,), "metrics": metrics, "score": metric_score(metrics["train"]) + 2.0 * metric_score(metrics["test"])})
    primitive_rows.sort(key=lambda row: row["score"], reverse=True)
    kept_primitives = [row["gates"][0] for row in primitive_rows[:max_primitives]]

    candidates: dict[tuple[tuple[str, str, float], ...], dict[str, Any]] = {}
    for width in range(1, int(max_width) + 1):
        for combo in itertools.combinations(kept_primitives, width):
            # avoid contradictory duplicate feature pairs; single-feature sweep already covers those surfaces.
            if len({g.feature for g in combo}) != len(combo):
                continue
            key = _gate_key(combo)
            metrics = evaluate_gate_set(combo, sets, market, 0.5, period_bounds)
            if metrics is None:
                continue
            if metrics["train"]["sim"]["trade_entries"] < int(min_train_trades) or metrics["test"]["sim"]["trade_entries"] < int(min_test_trades):
                continue
            if metrics["train"]["sim"]["cagr_pct"] <= 0 or metrics["test"]["sim"]["cagr_pct"] <= 0:
                continue
            if metrics["train"]["sim"]["strict_mdd_pct"] > 25 or metrics["test"]["sim"]["strict_mdd_pct"] > 15:
                continue
            candidates[key] = {
                "gates": [g.as_dict() for g in combo],
                "width": width,
                "train": metrics["train"],
                "test": metrics["test"],
                "eval": metrics["eval"],
                "selection_score": metric_score(metrics["train"]) + 2.0 * metric_score(metrics["test"]),
            }

    rows = sorted(candidates.values(), key=lambda row: row["selection_score"], reverse=True)
    leverages = [float(x.strip()) for x in str(leverage_grid).split(",") if x.strip()]
    for row in rows[:10]:
        gates = tuple(Gate(**g) for g in row["gates"])
        grid = []
        for lev in leverages:
            eval_rows = filter_rows(sets["eval"], gates)
            start, end = period_bounds.get("eval", ("", ""))
            result = backtest(eval_rows, market, lev, annualization_start=start, annualization_end=end)
            grid.append({"leverage": lev, "eval": {"n_rows": len(eval_rows), "sim": result["sim"], "trade_stats": result["trade_stats"]}})
        row["eval_leverage_grid"] = grid

    report = {
        "leakage_guard": {
            "primitive_thresholds_fit_on_train_feature_quantiles_only": True,
            "gate_sets_ranked_on_train_and_test_only": True,
            "eval_not_used_for_selection": True,
            "cagr_uses_full_configured_split_window_including_idle_time": True,
        },
        "filters": {"side_filter": side_filter, "family_filter": family_filter},
        "period_bounds": period_bounds,
        "min_trade_rules": {"train": int(min_train_trades), "test": int(min_test_trades)},
        "primitive_count": len(primitives),
        "kept_primitive_count": len(kept_primitives),
        "candidate_count": len(rows),
        "top": rows[:50],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep conjunction gates over fixed event candidate rows")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--market-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-primitives", type=int, default=40)
    parser.add_argument("--max-width", type=int, default=2)
    parser.add_argument("--leverage-grid", default="0.5,1.0,1.25,1.5,2.0")
    parser.add_argument("--side-filter", choices=["", "LONG", "SHORT"], default="")
    parser.add_argument("--family-filter", default="")
    parser.add_argument("--min-train-trades", type=int, default=120)
    parser.add_argument("--min-test-trades", type=int, default=20)
    parser.add_argument("--train-start", default="")
    parser.add_argument("--train-end", default="")
    parser.add_argument("--test-start", default="")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--eval-start", default="")
    parser.add_argument("--eval-end", default="")
    return parser.parse_args()


def main() -> None:
    report = run(**vars(parse_args()))
    print(json.dumps({"candidate_count": report["candidate_count"], "top": report["top"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
