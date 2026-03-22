"""Grid-search calibration for VLM likelihood action biases."""

from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path

from models.option_b_vlm import ACTION_SCHEMA_LABELS, get_action_labels
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


def _evaluate_bias(
    rows: list[dict],
    action_biases: dict[str, float],
    *,
    labels: tuple[str, ...] = ACTION_LABELS,
) -> dict:
    preds: list[str] = []
    targets: list[str] = []
    for row in rows:
        pred, _ = select_action_from_scores(
            scores=row["scores"],
            action_biases=action_biases,
            labels=labels,
        )
        preds.append(pred)
        targets.append(str(row["target"]))
    return summarize_action_metrics(targets=targets, predictions=preds, labels=labels)


def _default_bias_grid(labels: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    if labels == ACTION_LABELS:
        return {
            "BUY": (-0.8, 0.8, 0.1),
            "HOLD": (-0.6, 0.6, 0.1),
            "SELL": (-0.8, 0.8, 0.1),
        }
    return {label: (-1.5, 1.5, 0.1) for label in labels}


def _bias_grid_from_legacy_args(
    labels: tuple[str, ...],
    *,
    buy_min: float,
    buy_max: float,
    buy_step: float,
    hold_min: float,
    hold_max: float,
    hold_step: float,
    sell_min: float,
    sell_max: float,
    sell_step: float,
    trade_min: float,
    trade_max: float,
    trade_step: float,
    no_trade_min: float,
    no_trade_max: float,
    no_trade_step: float,
    long_min: float,
    long_max: float,
    long_step: float,
    short_min: float,
    short_max: float,
    short_step: float,
) -> dict[str, tuple[float, float, float]]:
    if labels == ("TRADE", "NO_TRADE"):
        return {
            "TRADE": (float(trade_min), float(trade_max), float(trade_step)),
            "NO_TRADE": (float(no_trade_min), float(no_trade_max), float(no_trade_step)),
        }
    if labels == ("LONG", "SHORT"):
        return {
            "LONG": (float(long_min), float(long_max), float(long_step)),
            "SHORT": (float(short_min), float(short_max), float(short_step)),
        }
    return {
        "BUY": (float(buy_min), float(buy_max), float(buy_step)),
        "HOLD": (float(hold_min), float(hold_max), float(hold_step)),
        "SELL": (float(sell_min), float(sell_max), float(sell_step)),
    }


def calibrate_action_biases(
    rows: list[dict],
    *,
    labels: tuple[str, ...] = ACTION_LABELS,
    bias_grid: dict[str, tuple[float, float, float]] | None = None,
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
    labels = tuple(str(label).upper() for label in labels)
    if not labels:
        raise ValueError("labels must not be empty")
    effective_grid = dict(bias_grid or _default_bias_grid(labels))
    grid_values = {
        label: _frange_inclusive(*effective_grid[label]) for label in labels
    }

    candidates: list[dict] = []
    for combo in itertools.product(*(grid_values[label] for label in labels)):
        biases = {label: float(value) for label, value in zip(labels, combo)}
        metrics = _evaluate_bias(rows=rows, action_biases=biases, labels=labels)
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
            "labels": list(labels),
            "ranges": {
                label: {
                    "min": float(effective_grid[label][0]),
                    "max": float(effective_grid[label][1]),
                    "step": float(effective_grid[label][2]),
                    "count": len(grid_values[label]),
                }
                for label in labels
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
    parser.add_argument(
        "--action-schema",
        type=str,
        default="buy_hold_sell",
        choices=sorted(ACTION_SCHEMA_LABELS),
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
    parser.add_argument("--trade-min", type=float, default=-1.5)
    parser.add_argument("--trade-max", type=float, default=1.5)
    parser.add_argument("--trade-step", type=float, default=0.1)
    parser.add_argument("--no-trade-min", type=float, default=-1.5)
    parser.add_argument("--no-trade-max", type=float, default=1.5)
    parser.add_argument("--no-trade-step", type=float, default=0.1)
    parser.add_argument("--long-min", type=float, default=-1.5)
    parser.add_argument("--long-max", type=float, default=1.5)
    parser.add_argument("--long-step", type=float, default=0.1)
    parser.add_argument("--short-min", type=float, default=-1.5)
    parser.add_argument("--short-max", type=float, default=1.5)
    parser.add_argument("--short-step", type=float, default=0.1)
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
    labels = get_action_labels(args.action_schema)
    bias_grid = _bias_grid_from_legacy_args(
        labels,
        buy_min=args.buy_min,
        buy_max=args.buy_max,
        buy_step=args.buy_step,
        hold_min=args.hold_min,
        hold_max=args.hold_max,
        hold_step=args.hold_step,
        sell_min=args.sell_min,
        sell_max=args.sell_max,
        sell_step=args.sell_step,
        trade_min=args.trade_min,
        trade_max=args.trade_max,
        trade_step=args.trade_step,
        no_trade_min=args.no_trade_min,
        no_trade_max=args.no_trade_max,
        no_trade_step=args.no_trade_step,
        long_min=args.long_min,
        long_max=args.long_max,
        long_step=args.long_step,
        short_min=args.short_min,
        short_max=args.short_max,
        short_step=args.short_step,
    )
    report = calibrate_action_biases(
        rows=rows,
        labels=labels,
        bias_grid=bias_grid,
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
