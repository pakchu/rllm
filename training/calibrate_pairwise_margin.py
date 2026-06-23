"""Calibrate A/B margin offset for pairwise rank logprob scores.

This corrects positional label prior using a calibration split only.  Prediction
rule after calibration: predict A if margin_a_minus_b >= threshold else B.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PairwiseMarginCalibrationConfig:
    calibration_scores: str
    eval_scores: str
    output: str
    eval_predictions_output: str = ""
    objective: str = "accuracy"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _target(row: dict[str, Any]) -> str:
    t = str(row.get("target", "A")).strip().upper()
    return t if t in {"A", "B"} else "A"


def _margin(row: dict[str, Any]) -> float:
    return float(row.get("margin_a_minus_b", 0.0) or 0.0)


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    correct = 0
    pred_counts = {"A": 0, "B": 0}
    target_counts = {"A": 0, "B": 0}
    confusion: dict[str, int] = {}
    for row in rows:
        pred = "A" if _margin(row) >= float(threshold) else "B"
        target = _target(row)
        correct += int(pred == target)
        pred_counts[pred] += 1
        target_counts[target] += 1
        key = f"target={target}|pred={pred}"
        confusion[key] = confusion.get(key, 0) + 1
    return {"num_samples": len(rows), "threshold": float(threshold), "accuracy": correct / max(1, len(rows)), "target_counts": target_counts, "prediction_counts": pred_counts, "confusion": dict(sorted(confusion.items()))}


def choose_threshold(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    margins = sorted({_margin(row) for row in rows})
    candidates = [margins[0] - 1e-9]
    candidates.extend((a + b) / 2.0 for a, b in zip(margins, margins[1:]))
    candidates.append(margins[-1] + 1e-9)
    best_t = candidates[0]
    best_acc = -1.0
    # Prefer less aggressive offsets on ties.
    for t in candidates:
        acc = float(_metrics(rows, t)["accuracy"])
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and abs(t) < abs(best_t)):
            best_acc = acc
            best_t = float(t)
    return best_t


def _prediction_rows(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        pred = "A" if _margin(row) >= float(threshold) else "B"
        out.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "target": _target(row), "prediction": pred, "margin_a_minus_b": _margin(row), "threshold": float(threshold)})
    return out


def run(cfg: PairwiseMarginCalibrationConfig) -> dict[str, Any]:
    cal = read_jsonl(cfg.calibration_scores)
    ev = read_jsonl(cfg.eval_scores)
    threshold = choose_threshold(cal)
    report = {
        "config": asdict(cfg),
        "threshold_source": "calibration_scores_only",
        "threshold": threshold,
        "calibration_metrics": _metrics(cal, threshold),
        "eval_metrics": _metrics(ev, threshold),
        "leakage_guard": {"threshold_uses_eval_scores": False, "eval_scores_only_scored_after_threshold_fixed": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if cfg.eval_predictions_output:
        Path(cfg.eval_predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.eval_predictions_output).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in _prediction_rows(ev, threshold)))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate pairwise A/B margin offset")
    p.add_argument("--calibration-scores", required=True)
    p.add_argument("--eval-scores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-predictions-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PairwiseMarginCalibrationConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
