"""Compose gate + side eval reports into BUY/HOLD/SELL action-score report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.eval_vlm_policy import summarize_action_metrics

COMPOSED_LABELS = ("BUY", "HOLD", "SELL")
SIDE_TO_FINAL = {"LONG": "BUY", "SHORT": "SELL"}
FLOOR_SCORE = -1.0e9


def _load_report(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data.get("action_scores"), list):
        raise ValueError(f"report has no action_scores: {path}")
    return data


def _safe_scores(row: dict[str, Any]) -> dict[str, float]:
    src = row.get("adjusted_scores") or row.get("scores") or {}
    return {str(k): float(v) for k, v in src.items()}


def compose_gate_side_reports(
    *,
    gate_report_path: str,
    side_report_path: str,
    output_path: str,
    floor_score: float = FLOOR_SCORE,
    gate_margin_threshold: float = 0.0,
    side_weight: float = 1.0,
) -> dict[str, Any]:
    gate = _load_report(gate_report_path)
    side = _load_report(side_report_path)

    if str(gate.get("action_schema")) != "trade_gate":
        raise ValueError("gate report must use action_schema=trade_gate")
    if str(side.get("action_schema")) != "trade_side":
        raise ValueError("side report must use action_schema=trade_side")

    gate_rows = list(gate["action_scores"])
    side_rows = list(side["action_scores"])
    side_by_date = {str(row["date"]): row for row in side_rows}

    composed_rows: list[dict[str, Any]] = []
    targets: list[str] = []
    preds: list[str] = []

    for gate_row in gate_rows:
        date = str(gate_row["date"])
        gate_scores = _safe_scores(gate_row)
        side_row = side_by_date.get(date)
        side_scores = _safe_scores(side_row) if side_row is not None else {}

        trade_score = float(gate_scores.get("TRADE", floor_score))
        no_trade_score = float(gate_scores.get("NO_TRADE", floor_score))
        long_score = float(side_scores.get("LONG", floor_score))
        short_score = float(side_scores.get("SHORT", floor_score))
        trade_margin = float(trade_score - no_trade_score)

        if side_row is None or trade_margin < float(gate_margin_threshold):
            composed_scores = {
                "BUY": float(floor_score),
                "HOLD": float(no_trade_score),
                "SELL": float(floor_score),
            }
        else:
            composed_scores = {
                "BUY": float(trade_score + float(side_weight) * long_score),
                "HOLD": float(no_trade_score),
                "SELL": float(trade_score + float(side_weight) * short_score),
            }
        pred = max(COMPOSED_LABELS, key=lambda k: (composed_scores[k], -COMPOSED_LABELS.index(k)))

        gate_target = str(gate_row.get("target", "NO_TRADE"))
        if gate_target == "NO_TRADE":
            target = "HOLD"
        elif side_row is not None:
            target = SIDE_TO_FINAL.get(str(side_row.get("target", "")).upper(), "HOLD")
        else:
            target = "HOLD"

        composed_rows.append(
            {
                "date": date,
                "target": target,
                "next_return": float(gate_row.get("next_return", 0.0)),
                "pred": pred,
                "scores": composed_scores,
                "adjusted_scores": composed_scores,
            }
        )
        targets.append(target)
        preds.append(pred)

    metrics = summarize_action_metrics(targets=targets, predictions=preds, labels=COMPOSED_LABELS)
    report = {
        "composition": {
            "gate_report": str(Path(gate_report_path).resolve()),
            "side_report": str(Path(side_report_path).resolve()),
            "rule": "BUY=gate.TRADE+side.LONG, SELL=gate.TRADE+side.SHORT, HOLD=gate.NO_TRADE",
            "floor_score_for_missing_side": float(floor_score),
            "gate_margin_threshold": float(gate_margin_threshold),
            "side_weight": float(side_weight),
            "gate_rows": int(len(gate_rows)),
            "side_rows": int(len(side_rows)),
            "side_date_overlap": int(sum(1 for row in gate_rows if str(row['date']) in side_by_date)),
        },
        "action_schema": "buy_hold_sell",
        "metrics": metrics,
        "action_scores": composed_rows,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose gate and side reports into BUY/HOLD/SELL policy report.")
    parser.add_argument("--gate-report", type=str, required=True)
    parser.add_argument("--side-report", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/composed_gate_side_policy.json")
    parser.add_argument("--floor-score", type=float, default=FLOOR_SCORE)
    parser.add_argument("--gate-margin-threshold", type=float, default=0.0)
    parser.add_argument("--side-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = compose_gate_side_reports(
        gate_report_path=args.gate_report,
        side_report_path=args.side_report,
        output_path=args.output,
        floor_score=args.floor_score,
        gate_margin_threshold=args.gate_margin_threshold,
        side_weight=args.side_weight,
    )
    print(json.dumps({"metrics": out["metrics"], "composition": out["composition"]}, indent=2))


if __name__ == "__main__":
    main()
