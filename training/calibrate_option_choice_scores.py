"""Fit simple option-score offsets for A/B/C option-choice predictions.

Use this only as a diagnostic unless calibration rows are strictly earlier or
separate from the validation/eval rows. By default the script reports in-sample
calibration; pass ``--validation-fraction`` to expose overfit sensitivity.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OPTIONS = ("A", "B", "C")


@dataclass(frozen=True)
class OptionCalibrationCfg:
    predictions_jsonl: str
    output_json: str
    offsets: str = "-8,-6,-4,-2,0,2,4,6,8"
    objective: str = "accuracy"
    validation_fraction: float = 0.0
    seed: int = 42


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _floats(csv: str) -> list[float]:
    return [float(x.strip()) for x in str(csv).split(",") if x.strip()]


def _split(rows: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frac = float(validation_fraction)
    if frac <= 0:
        return rows, []
    if not 0 < frac < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    idx = list(range(len(rows)))
    random.Random(int(seed)).shuffle(idx)
    n_val = max(1, int(round(len(rows) * frac)))
    val_idx = set(idx[:n_val])
    fit = [row for i, row in enumerate(rows) if i not in val_idx]
    val = [row for i, row in enumerate(rows) if i in val_idx]
    return fit, val


def _predict(row: dict[str, Any], offsets: dict[str, float]) -> str:
    scores = row.get("scores") or {}
    vals = {opt: float(scores.get(opt, -1e9)) + float(offsets.get(opt, 0.0)) for opt in OPTIONS}
    return max(vals.items(), key=lambda kv: kv[1])[0]


def _metrics(rows: list[dict[str, Any]], offsets: dict[str, float]) -> dict[str, Any]:
    counts = {opt: 0 for opt in OPTIONS}
    target_counts = {opt: 0 for opt in OPTIONS}
    correct_by_target = {opt: 0 for opt in OPTIONS}
    correct = 0
    for row in rows:
        target = str(row.get("target"))
        pred = _predict(row, offsets)
        if pred in counts:
            counts[pred] += 1
        if target in target_counts:
            target_counts[target] += 1
            correct_by_target[target] += int(pred == target)
        correct += int(pred == target)
    n = max(1, len(rows))
    pred_rates = {opt: counts[opt] / n for opt in OPTIONS}
    min_pred_rate = min(pred_rates.values())
    by_target = {opt: correct_by_target[opt] / max(1, target_counts[opt]) for opt in OPTIONS}
    return {
        "accuracy": correct / n,
        "balanced_accuracy": sum(by_target.values()) / len(OPTIONS),
        "correct": correct,
        "prediction_counts": counts,
        "target_counts": target_counts,
        "prediction_rates": pred_rates,
        "min_prediction_rate": min_pred_rate,
        "accuracy_by_target": by_target,
    }


def _objective(metrics: dict[str, Any], name: str) -> float:
    if name == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if name == "accuracy_minpred":
        return float(metrics["accuracy"]) + 0.1 * float(metrics["min_prediction_rate"])
    return float(metrics["accuracy"])


def run(cfg: OptionCalibrationCfg) -> dict[str, Any]:
    rows = _load(cfg.predictions_jsonl)
    fit_rows, val_rows = _split(rows, float(cfg.validation_fraction), int(cfg.seed))
    grid = _floats(cfg.offsets)
    candidates = []
    # Fix A offset at zero; scan relative B/C offsets to avoid redundant shifts.
    for b in grid:
        for c in grid:
            offsets = {"A": 0.0, "B": b, "C": c}
            fit_metrics = _metrics(fit_rows, offsets)
            validation_metrics = _metrics(val_rows, offsets) if val_rows else None
            candidates.append(
                {
                    "offsets": offsets,
                    "score": _objective(fit_metrics, cfg.objective),
                    "fit": fit_metrics,
                    "validation": validation_metrics,
                }
            )
    candidates.sort(
        key=lambda r: (
            float(r["score"]),
            float(r["fit"]["accuracy"]),
            float(r["fit"]["min_prediction_rate"]),
        ),
        reverse=True,
    )
    zero = {"A": 0.0, "B": 0.0, "C": 0.0}
    report = {
        "config": asdict(cfg),
        "rows": len(rows),
        "fit_rows": len(fit_rows),
        "validation_rows": len(val_rows),
        "baseline_fit": _metrics(fit_rows, zero),
        "baseline_validation": _metrics(val_rows, zero) if val_rows else None,
        "top": candidates[:20],
    }
    Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--offsets", default=OptionCalibrationCfg.offsets)
    p.add_argument("--objective", choices=["accuracy", "balanced_accuracy", "accuracy_minpred"], default=OptionCalibrationCfg.objective)
    p.add_argument("--validation-fraction", type=float, default=OptionCalibrationCfg.validation_fraction)
    p.add_argument("--seed", type=int, default=OptionCalibrationCfg.seed)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(OptionCalibrationCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
