"""Build compact pressure SFT rows augmented with train-only softmax teacher hints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.economic_compact_pressure_analyzer_sft_data import compact_pressure_prompt, compact_summary_from_prompt
from training.economic_opportunity_baseline import best_rows_by_signal, signal_feature_row
from training.economic_path_shape_data import PathTemplate, compute_path_shape
from training.economic_preference_sft_data import write_jsonl
from training.economic_pressure_learnability_sweep import LABELS, fit_softmax
from training.economic_value_baseline import FeatureSpace, load_jsonl


def pressure_labels(rows: list[dict[str, Any]], market: pd.DataFrame, *, horizon_bars: int, target_pct: float, stop_pct: float) -> list[str]:
    tpl = PathTemplate(horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, entry_delay_bars=1)
    labels = []
    for row in best_rows_by_signal(rows):
        shape = compute_path_shape(market, int(row.get("signal_pos", -1)), tpl)
        labels.append(str((shape or {}).get("direction_pressure", "NO_TRADE_FAVORED")))
    return labels


def fit_teacher(train_value_rows: list[dict[str, Any]], market: pd.DataFrame, *, horizon_bars: int, target_pct: float, stop_pct: float) -> tuple[FeatureSpace, np.ndarray]:
    train_best = best_rows_by_signal(train_value_rows)
    train_labels = pressure_labels(train_value_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct)
    fs = FeatureSpace(min_count=2)
    fs.fit([signal_feature_row(r) for r in train_best])
    x_train = fs.matrix([signal_feature_row(r) for r in train_best], fit_scale=True)
    y = np.array([LABELS.index(x) if x in LABELS else LABELS.index("NO_TRADE_FAVORED") for x in train_labels], dtype=np.int64)
    return fs, fit_softmax(x_train, y, num_classes=len(LABELS))


def teacher_predictions(value_rows: list[dict[str, Any]], fs: FeatureSpace, w: np.ndarray) -> list[dict[str, Any]]:
    best = best_rows_by_signal(value_rows)
    x = fs.matrix([signal_feature_row(r) for r in best])
    logits = x @ w
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    out = []
    for row, p in zip(best, probs):
        idx = int(np.argmax(p))
        out.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "teacher_pressure": LABELS[idx], "teacher_confidence": float(p[idx]), "teacher_probs": {LABELS[i]: float(p[i]) for i in range(len(LABELS))}})
    return out


def confidence_bucket(x: float) -> str:
    if x >= 0.55:
        return "HIGH"
    if x >= 0.42:
        return "MEDIUM"
    return "LOW"


def teacher_prompt(compact: dict[str, Any], teacher: dict[str, Any]) -> str:
    base = compact_pressure_prompt(compact)
    hint = {"teacher_pressure": teacher["teacher_pressure"], "teacher_confidence_bucket": confidence_bucket(float(teacher["teacher_confidence"]))}
    return base + "\n\nTrain-only structured teacher hint: " + json.dumps(hint, sort_keys=True, separators=(",", ":"))


def build_teacher_pressure_sft(
    *,
    train_value_jsonl: str,
    input_path_shape_jsonl: str,
    market_csv: str,
    output: str,
    summary_output: str = "",
    horizon_bars: int = 36,
    target_pct: float = 0.5,
    stop_pct: float = 0.6,
) -> dict[str, Any]:
    market = pd.read_csv(market_csv)
    train_values = load_jsonl(train_value_jsonl)
    fs, w = fit_teacher(train_values, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct)
    path_rows = load_jsonl(input_path_shape_jsonl)
    # Use value rows corresponding to this split for teacher predictions by mapping date/pos from path rows.
    split_value_like = [{"date": r.get("date"), "signal_pos": r.get("signal_pos"), "prompt": r.get("prompt"), "action": json.dumps({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}), "utility": 0.0} for r in path_rows]
    preds = teacher_predictions(split_value_like, fs, w)
    pred_by_key = {(str(p["date"]), int(p["signal_pos"])): p for p in preds}
    out = []
    counts: Counter[str] = Counter()
    for row in path_rows:
        target = row.get("analyzer_target", {}) if isinstance(row.get("analyzer_target"), dict) else {}
        pressure = str(target.get("direction_pressure", "NO_TRADE_FAVORED"))
        key = (str(row.get("date")), int(row.get("signal_pos", -1)))
        teacher = pred_by_key[key]
        compact = compact_summary_from_prompt(str(row.get("prompt", "")))
        counts[pressure] += 1
        out.append({"task": "teacher_compact_path_pressure_analyzer_sft", "date": row.get("date"), "signal_pos": row.get("signal_pos"), "prompt": teacher_prompt(compact, teacher), "target": json.dumps({"direction_pressure": pressure}, ensure_ascii=False, sort_keys=True), "pressure": pressure, "teacher": teacher, "leakage_guard": {"teacher_fit_on_train_only": True, "prompt_uses_future_path": False, "target_uses_future_path_pressure_for_training_only": True}})
    write_jsonl(output, out)
    lens = [len(r["prompt"]) for r in out]
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "output": output, "rows": len(out), "pressure_counts": dict(sorted(counts.items())), "majority_baseline_accuracy": max(counts.values()) / max(1, len(out)) if counts else 0.0, "prompt_chars": {"min": min(lens) if lens else 0, "max": max(lens) if lens else 0, "mean": sum(lens)/max(1,len(lens))}, "teacher_fit": {"train_value_jsonl": train_value_jsonl, "horizon_bars": horizon_bars, "target_pct": target_pct, "stop_pct": stop_pct}}
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-value-jsonl", required=True)
    p.add_argument("--input-path-shape-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--target-pct", type=float, default=0.5)
    p.add_argument("--stop-pct", type=float, default=0.6)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_teacher_pressure_sft(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
