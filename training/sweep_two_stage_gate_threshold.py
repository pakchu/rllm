"""Sweep gate-score thresholds for two-stage event-action validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_action_backtest import EconomicActionBacktestConfig, dedupe_signal_predictions, load_prediction_rows, strict_backtest_actions
from training.strict_bar_backtest import load_market_bars

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "NONE", "confidence": "HIGH"}


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def _compose(action_rows: list[dict[str, Any]], gate_scores: dict[tuple[str, int], float], threshold: float) -> list[dict[str, Any]]:
    out = []
    for row in action_rows:
        margin = gate_scores.get(_key(row), float("-inf"))
        pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
        if margin < threshold:
            pred = dict(NO_TRADE)
        else:
            pred["gate"] = "TRADE"
        out.append({**row, "prediction": pred, "gate_margin": margin, "gate_threshold": threshold})
    return out


def sweep_thresholds(*, gate_scores_jsonl: str, action_predictions_jsonl: str, market_csv: str, output: str, margin_key: str = "margin_sum_trade_minus_no_trade", thresholds: str = "") -> dict[str, Any]:
    gate_rows = _read(gate_scores_jsonl)
    action_rows = dedupe_signal_predictions(load_prediction_rows(action_predictions_jsonl))
    scores = {_key(r): float(r.get(margin_key, float("nan"))) for r in gate_rows}
    vals = np.asarray([scores[_key(r)] for r in action_rows if _key(r) in scores and np.isfinite(scores[_key(r)])], dtype=float)
    if vals.size == 0:
        raise ValueError("no overlapping finite gate scores")
    if thresholds:
        ths = [float(x) for x in thresholds.split(",") if x.strip()]
    else:
        qs = np.linspace(0.0, 1.0, 31)
        ths = sorted(set(float(x) for x in np.quantile(vals, qs)))
    market = load_market_bars(market_csv)
    cfg = EconomicActionBacktestConfig()
    reports = []
    best = None
    for th in ths:
        rows = _compose(action_rows, scores, th)
        bt = strict_backtest_actions(rows, market, cfg)
        sim = bt["sim"]
        item = {"threshold": th, "sim": sim, "trade_stats": bt["trade_stats"]}
        reports.append(item)
        if best is None or (float(sim["cagr_to_strict_mdd"]) if np.isfinite(sim["cagr_to_strict_mdd"]) else -1e9) > (float(best["sim"]["cagr_to_strict_mdd"]) if np.isfinite(best["sim"]["cagr_to_strict_mdd"]) else -1e9):
            best = item
    result = {"gate_scores_jsonl": str(Path(gate_scores_jsonl).resolve()), "action_predictions_jsonl": str(Path(action_predictions_jsonl).resolve()), "market_csv": str(Path(market_csv).resolve()), "margin_key": margin_key, "overlap_rows": int(vals.size), "best": best, "reports": reports, "leakage_guard": {"thresholds_selected_on_this_validation_only": True, "eval_not_used": True}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"best": best, "overlap_rows": int(vals.size)}, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep two-stage gate thresholds")
    p.add_argument("--gate-scores-jsonl", required=True)
    p.add_argument("--action-predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--margin-key", default="margin_sum_trade_minus_no_trade")
    p.add_argument("--thresholds", default="")
    return p.parse_args()


def main() -> None:
    sweep_thresholds(**vars(parse_args()))


if __name__ == "__main__":
    main()
