"""Grid-search calibration for VLM likelihood action biases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.eval_vlm_policy import ACTION_LABELS, select_action_from_scores, summarize_action_metrics


def _frange_inclusive(vmin: float, vmax: float, step: float) -> list[float]:
    """Inclusive float range with stable decimal rounding."""
    lo = float(vmin)
    hi = float(vmax)
    inc = float(step)
    if inc <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if lo > hi:
        raise ValueError(f"min must be <= max, got {vmin} > {vmax}")

    out: list[float] = []
    cur = lo
    eps = inc * 1e-6
    while cur <= hi + eps:
        out.append(float(round(cur, 8)))
        cur += inc
    # Ensure exact vmax is present.
    if out and abs(out[-1] - hi) > eps:
        out.append(float(round(hi, 8)))
    return out


def load_action_scores(report_paths: list[str]) -> list[dict]:
    """Load and merge action score rows from eval-vlm report JSON files."""
    rows: list[dict] = []
    for p in report_paths:
        report = json.loads(Path(p).read_text())
        action_scores = report.get("action_scores")
        if not isinstance(action_scores, list) or not action_scores:
            raise ValueError(
                f"report has no action_scores (run eval-vlm with --store-action-scores true): {p}"
            )
        rows.extend(action_scores)
    if not rows:
        raise ValueError("No action score rows loaded.")
    return rows


def score_metrics(
    metrics: dict,
    weight_accuracy: float = 1.0,
    weight_balanced_recall: float = 0.15,
    weight_directional_mean: float = 0.05,
    weight_directional_gap: float = 0.10,
) -> float:
    """
    Objective for bias calibration.

    Higher is better:
    + accuracy
    + balanced recall (BUY/HOLD/SELL)
    + directional mean recall (BUY/SELL)
    - directional recall gap |BUY-SELL|
    """
    return float(
        weight_accuracy * float(metrics["accuracy"])
        + weight_balanced_recall * float(metrics.get("balanced_recall", 0.0))
        + weight_directional_mean * float(metrics.get("directional_recall_mean", 0.0))
        - weight_directional_gap * float(metrics.get("directional_recall_gap", 0.0))
    )


def _evaluate_bias(rows: list[dict], action_biases: dict[str, float]) -> dict:
    preds: list[str] = []
    targets: list[str] = []
    for row in rows:
        pred, _ = select_action_from_scores(
            scores=row["scores"],
            action_biases=action_biases,
            labels=ACTION_LABELS,
        )
        preds.append(pred)
        targets.append(str(row["target"]))
    return summarize_action_metrics(targets=targets, predictions=preds, labels=ACTION_LABELS)


def calibrate_action_biases(
    rows: list[dict],
    *,
    buy_min: float = -0.8,
    buy_max: float = 0.8,
    buy_step: float = 0.1,
    hold_min: float = -0.6,
    hold_max: float = 0.6,
    hold_step: float = 0.1,
    sell_min: float = -0.8,
    sell_max: float = 0.8,
    sell_step: float = 0.1,
    top_k: int = 10,
    weight_accuracy: float = 1.0,
    weight_balanced_recall: float = 0.15,
    weight_directional_mean: float = 0.05,
    weight_directional_gap: float = 0.10,
) -> dict:
    """Run bias grid-search and return ranked candidates."""
    buy_values = _frange_inclusive(buy_min, buy_max, buy_step)
    hold_values = _frange_inclusive(hold_min, hold_max, hold_step)
    sell_values = _frange_inclusive(sell_min, sell_max, sell_step)

    candidates: list[dict] = []
    for b_buy in buy_values:
        for b_hold in hold_values:
            for b_sell in sell_values:
                biases = {"BUY": float(b_buy), "HOLD": float(b_hold), "SELL": float(b_sell)}
                metrics = _evaluate_bias(rows=rows, action_biases=biases)
                objective = score_metrics(
                    metrics,
                    weight_accuracy=weight_accuracy,
                    weight_balanced_recall=weight_balanced_recall,
                    weight_directional_mean=weight_directional_mean,
                    weight_directional_gap=weight_directional_gap,
                )
                candidates.append(
                    {
                        "biases": biases,
                        "objective": float(objective),
                        "accuracy": float(metrics["accuracy"]),
                        "balanced_recall": float(metrics.get("balanced_recall", 0.0)),
                        "directional_recall_mean": float(metrics.get("directional_recall_mean", 0.0)),
                        "directional_recall_gap": float(metrics.get("directional_recall_gap", 0.0)),
                        "metrics": metrics,
                    }
                )

    candidates.sort(
        key=lambda x: (
            float(x["objective"]),
            float(x["accuracy"]),
            -float(x["directional_recall_gap"]),
        ),
        reverse=True,
    )
    keep = max(1, int(top_k))
    best = candidates[0]
    return {
        "num_rows": int(len(rows)),
        "grid": {
            "buy": {"min": float(buy_min), "max": float(buy_max), "step": float(buy_step), "count": len(buy_values)},
            "hold": {
                "min": float(hold_min),
                "max": float(hold_max),
                "step": float(hold_step),
                "count": len(hold_values),
            },
            "sell": {
                "min": float(sell_min),
                "max": float(sell_max),
                "step": float(sell_step),
                "count": len(sell_values),
            },
            "num_candidates": int(len(candidates)),
        },
        "objective": {
            "formula": "w_acc*acc + w_bal*balanced_recall + w_dir*directional_recall_mean - w_gap*directional_recall_gap",
            "weights": {
                "accuracy": float(weight_accuracy),
                "balanced_recall": float(weight_balanced_recall),
                "directional_recall_mean": float(weight_directional_mean),
                "directional_recall_gap": float(weight_directional_gap),
            },
        },
        "best": best,
        "top_candidates": candidates[:keep],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate VLM likelihood action biases.")
    parser.add_argument(
        "--input-report",
        type=str,
        action="append",
        required=True,
        help="eval-vlm JSON path containing action_scores (repeatable)",
    )
    parser.add_argument("--buy-min", type=float, default=-0.8)
    parser.add_argument("--buy-max", type=float, default=0.8)
    parser.add_argument("--buy-step", type=float, default=0.1)
    parser.add_argument("--hold-min", type=float, default=-0.6)
    parser.add_argument("--hold-max", type=float, default=0.6)
    parser.add_argument("--hold-step", type=float, default=0.1)
    parser.add_argument("--sell-min", type=float, default=-0.8)
    parser.add_argument("--sell-max", type=float, default=0.8)
    parser.add_argument("--sell-step", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--weight-accuracy", type=float, default=1.0)
    parser.add_argument("--weight-balanced-recall", type=float, default=0.15)
    parser.add_argument("--weight-directional-mean", type=float, default=0.05)
    parser.add_argument("--weight-directional-gap", type=float, default=0.10)
    parser.add_argument("--output", type=str, default="results/vlm_bias_calibration.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_action_scores(args.input_report)
    report = calibrate_action_biases(
        rows=rows,
        buy_min=args.buy_min,
        buy_max=args.buy_max,
        buy_step=args.buy_step,
        hold_min=args.hold_min,
        hold_max=args.hold_max,
        hold_step=args.hold_step,
        sell_min=args.sell_min,
        sell_max=args.sell_max,
        sell_step=args.sell_step,
        top_k=args.top_k,
        weight_accuracy=args.weight_accuracy,
        weight_balanced_recall=args.weight_balanced_recall,
        weight_directional_mean=args.weight_directional_mean,
        weight_directional_gap=args.weight_directional_gap,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[calibrate-vlm-bias] saved={out.resolve()}")
    print(json.dumps(report["best"], indent=2))


if __name__ == "__main__":
    main()
