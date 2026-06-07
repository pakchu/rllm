"""Sweep live-safe filters for pressure prediction backtests."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from training.economic_pressure_backtest import run_pressure_backtest


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pred_pressure(row: dict[str, Any]) -> str:
    p = row.get("prediction", {})
    if isinstance(p, dict):
        return str(p.get("direction_pressure", "NO_TRADE_FAVORED"))
    try:
        return str(json.loads(str(p)).get("direction_pressure", "NO_TRADE_FAVORED"))
    except Exception:
        return "NO_TRADE_FAVORED"


def teacher_pressure(row: dict[str, Any]) -> str:
    t = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    return str(t.get("teacher_pressure", ""))


def teacher_conf(row: dict[str, Any]) -> float:
    t = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
    try:
        return float(t.get("teacher_confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def apply_filter(rows: list[dict[str, Any]], *, min_conf: float, require_agree: bool, allowed: set[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = deepcopy(row)
        p = pred_pressure(r)
        ok = p in allowed and teacher_conf(r) >= min_conf
        if require_agree:
            ok = ok and p == teacher_pressure(r)
        if not ok:
            r["prediction"] = {"direction_pressure": "NO_TRADE_FAVORED"}
        out.append(r)
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def run_filter_sweep(*, predictions_jsonl: str, market_csv: str, output: str, prefix: str, horizon_bars: int = 36, target_pct: float = 0.5, stop_pct: float = 0.6) -> dict[str, Any]:
    rows = load_jsonl(predictions_jsonl)
    configs = []
    for min_conf in [0.0, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55]:
        for require_agree in [False, True]:
            for allowed in [{"LONG_FAVORED", "SHORT_FAVORED"}, {"LONG_FAVORED"}, {"SHORT_FAVORED"}]:
                configs.append((min_conf, require_agree, allowed))
    results = []
    for min_conf, require_agree, allowed in configs:
        tag = f"conf{str(min_conf).replace('.', 'p')}_agree{int(require_agree)}_{'-'.join(sorted(a.split('_')[0].lower() for a in allowed))}"
        pred_path = f"{prefix}_{tag}_predictions.jsonl"
        bt_path = f"{prefix}_{tag}_backtest.json"
        filtered = apply_filter(rows, min_conf=min_conf, require_agree=require_agree, allowed=allowed)
        write_jsonl(pred_path, filtered)
        bt = run_pressure_backtest(predictions_jsonl=pred_path, market_csv=market_csv, output=bt_path, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct)
        results.append({"config": {"min_conf": min_conf, "require_agree": require_agree, "allowed": sorted(allowed)}, "prediction_rows": pred_path, "backtest_path": bt_path, "sim": bt["backtest"]["sim"], "trade_stats": bt["backtest"]["trade_stats"]})
    ranked = sorted(results, key=lambda r: (r["sim"]["cagr_to_strict_mdd"], r["trade_stats"].get("n_trades", 0)), reverse=True)
    report = {"predictions_jsonl": predictions_jsonl, "results": ranked, "top": ranked[:15], "leakage_guard": {"filters_use_prediction_and_train_teacher_confidence_only": True, "no_future_eval_labels_in_filter": True}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--target-pct", type=float, default=0.5)
    p.add_argument("--stop-pct", type=float, default=0.6)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_filter_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
