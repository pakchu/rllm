"""Apply eval_text_json_key side_map predictions to event trade proposals."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.apply_side_map_memory_predictions import NO_TRADE, _write_jsonl
from training.nested_score_geometry_transform_selection import _invert_prediction
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class ApplyEventSideMapTextCfg:
    dataset_jsonl: str
    base_predictions_jsonl: str
    text_eval_json: str
    output_jsonl: str
    market_csv: str = ""
    backtest_output: str = ""
    trade_take_profit_pct: float = 3.0
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("date")), int(row.get("signal_pos", -1) or -1)


def _apply(row: dict[str, Any], side_map: str) -> dict[str, Any]:
    nr = dict(row)
    pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
    label = str(side_map).upper()
    if label == "NORMAL":
        nr["prediction"] = pred
    elif label == "INVERSE":
        nr["prediction"] = _invert_prediction(pred)
    else:
        nr["prediction"] = dict(NO_TRADE)
    nr["event_side_map_text_prediction"] = label
    return nr


def run(cfg: ApplyEventSideMapTextCfg) -> dict[str, Any]:
    eval_rows = _read_jsonl(cfg.dataset_jsonl)
    text = json.loads(Path(cfg.text_eval_json).read_text())
    preds = [str(p.get("prediction", "UNRELIABLE")).upper() for p in text.get("predictions", [])]
    if len(preds) != len(eval_rows):
        raise ValueError(f"prediction count mismatch: {len(preds)} vs {len(eval_rows)}")
    pred_by_key = {_key(row): pred for row, pred in zip(eval_rows, preds)}
    base = _read_jsonl(cfg.base_predictions_jsonl)
    out = [_apply(r, pred_by_key[_key(r)]) for r in base if _key(r) in pred_by_key]
    counts: dict[str, int] = {}
    for r in out:
        label = str(r.get("event_side_map_text_prediction"))
        counts[label] = counts.get(label, 0) + 1
    _write_jsonl(cfg.output_jsonl, out)
    report: dict[str, Any] = {"config": asdict(cfg), "rows": len(out), "counts": counts, "leakage_guard": {"uses_text_model_predictions_only": True}}
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
    p = argparse.ArgumentParser(description="Apply side_map text eval predictions")
    p.add_argument("--dataset-jsonl", required=True)
    p.add_argument("--base-predictions-jsonl", required=True)
    p.add_argument("--text-eval-json", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--market-csv", default="")
    p.add_argument("--backtest-output", default="")
    p.add_argument("--trade-take-profit-pct", type=float, default=ApplyEventSideMapTextCfg.trade_take_profit_pct)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(ApplyEventSideMapTextCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
