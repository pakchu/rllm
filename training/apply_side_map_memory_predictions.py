"""Apply side-map memory predictions to trade rows and strict backtest them."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.nested_score_geometry_transform_selection import _invert_prediction
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "SIDE_MAP_MEMORY", "confidence": "HIGH"}


@dataclass(frozen=True)
class ApplySideMapMemoryCfg:
    predictions_jsonl: str
    memory_eval_json: str
    method: str
    output_jsonl: str
    market_csv: str = ""
    backtest_output: str = ""
    start_month: str = "2026-01"
    end_month: str = "2026-05"
    trade_take_profit_pct: float = 3.0
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _month(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))[:7]


def apply(cfg: ApplySideMapMemoryCfg) -> dict[str, Any]:
    rows = [r for r in _read_jsonl(cfg.predictions_jsonl) if cfg.start_month <= _month(r) <= cfg.end_month]
    mem = json.loads(Path(cfg.memory_eval_json).read_text())
    month_pred = {str(r["month"]): str(r["predictions"][cfg.method]) for r in mem.get("predictions", [])}
    out = []
    counts: dict[str, int] = {}
    for row in rows:
        m = _month(row)
        mode = month_pred.get(m, "unreliable")
        counts[mode] = counts.get(mode, 0) + 1
        nr = dict(row)
        pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
        if mode == "normal":
            nr["prediction"] = pred
        elif mode == "inverse":
            nr["prediction"] = _invert_prediction(pred)
        else:
            nr["prediction"] = dict(NO_TRADE)
        nr["side_map_memory_prediction"] = mode
        out.append(nr)
    _write_jsonl(cfg.output_jsonl, out)
    report: dict[str, Any] = {"config": asdict(cfg), "rows": len(out), "counts": counts, "leakage_guard": {"uses_eval_truth_labels": False, "uses_memory_predictions": True}}
    if cfg.market_csv and cfg.backtest_output:
        bt = run_overlay(OnlineRiskOverlayConfig(
            predictions_jsonl=cfg.output_jsonl,
            market_csv=cfg.market_csv,
            output=cfg.backtest_output,
            leverage=float(cfg.leverage),
            fee_rate=float(cfg.fee_rate),
            slippage_rate=float(cfg.slippage_rate),
            entry_delay_bars=int(cfg.entry_delay_bars),
            trade_take_profit_pct=float(cfg.trade_take_profit_pct),
        ))
        report["backtest"] = {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply side-map memory predictions")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--memory-eval-json", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--market-csv", default="")
    p.add_argument("--backtest-output", default="")
    p.add_argument("--start-month", default=ApplySideMapMemoryCfg.start_month)
    p.add_argument("--end-month", default=ApplySideMapMemoryCfg.end_month)
    p.add_argument("--trade-take-profit-pct", type=float, default=ApplySideMapMemoryCfg.trade_take_profit_pct)
    return p.parse_args()


def main() -> None:
    print(json.dumps(apply(ApplySideMapMemoryCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
