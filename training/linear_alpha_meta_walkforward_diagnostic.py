"""Walk-forward diagnostic for linear-alpha meta-controller prompt features.

This checks whether a continuously refit lightweight model can adapt across
regimes using only past rows.  It is a CPU preflight before attempting rolling
Gemma adapters or calibration heads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.linear_alpha_meta_feature_diagnostic import (
    _feature_space,
    _fit_logistic,
    _matrix,
    _metrics,
    _read_jsonl,
    _sigmoid,
    _standardize_apply,
    _standardize_fit,
    _target_decision,
)
from training.linear_alpha_meta_stability_diagnostic import _date, _period_key


@dataclass(frozen=True)
class WalkForwardConfig:
    inputs: str
    output: str
    period: str = "halfyear"
    max_features: int = 192
    min_train_rows: int = 1000
    min_eval_rows: int = 500
    lr: float = 0.05
    steps: int = 800
    l2: float = 0.02
    train_window_periods: int = 0
    trade_only: bool = True


def _load_inputs(inputs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in inputs.split(","):
        path = raw.strip()
        if path:
            rows.extend(_read_jsonl(path))
    return sorted(rows, key=_date)


def _periods(rows: list[dict[str, Any]], period: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(_period_key(_date(row), period), []).append(row)
    return dict(sorted(out.items()))


def _threshold_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    return _metrics(y, prob - float(threshold) + 0.5)


def _best_threshold(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    best = {"threshold": 0.5, "balanced_recall": -1.0, "accuracy": 0.0}
    for threshold in np.linspace(0.1, 0.9, 33):
        m = _threshold_metrics(y, prob, float(threshold))
        score = float(m["balanced_recall"])
        if score > float(best["balanced_recall"]):
            best = {"threshold": float(threshold), "balanced_recall": score, "accuracy": float(m["accuracy"])}
    return best


def run(cfg: WalkForwardConfig) -> dict[str, Any]:
    rows = _load_inputs(cfg.inputs)
    if cfg.trade_only:
        rows = [r for r in rows if str(r.get("metadata", {}).get("candidate_gate", "")).upper() == "TRADE"]
    by_period = _periods(rows, cfg.period)
    period_names = list(by_period)
    results: list[dict[str, Any]] = []
    all_eval_y: list[float] = []
    all_eval_prob: list[float] = []
    for idx, period in enumerate(period_names):
        eval_rows = by_period[period]
        if len(eval_rows) < int(cfg.min_eval_rows):
            continue
        train_periods = period_names[:idx]
        if cfg.train_window_periods and cfg.train_window_periods > 0:
            train_periods = train_periods[-int(cfg.train_window_periods):]
        train_rows = [r for p in train_periods for r in by_period[p]]
        if len(train_rows) < int(cfg.min_train_rows):
            continue
        features = _feature_space(train_rows, int(cfg.max_features))
        x_train, y_train = _matrix(train_rows, features)
        x_eval, y_eval = _matrix(eval_rows, features)
        mu, sigma = _standardize_fit(x_train)
        z_train = _standardize_apply(x_train, mu, sigma)
        w, b = _fit_logistic(z_train, y_train, lr=float(cfg.lr), steps=int(cfg.steps), l2=float(cfg.l2))
        train_prob = _sigmoid(z_train @ w + b)
        eval_prob = _sigmoid(_standardize_apply(x_eval, mu, sigma) @ w + b)
        train_best = _best_threshold(y_train, train_prob)
        fixed = _metrics(y_eval, eval_prob)
        calibrated = _threshold_metrics(y_eval, eval_prob, float(train_best["threshold"]))
        results.append(
            {
                "period": period,
                "train_periods": train_periods,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "target_counts": {
                    "SKIP": int(np.sum(y_eval == 0.0)),
                    "TAKE": int(np.sum(y_eval == 1.0)),
                },
                "train_best_threshold": train_best,
                "fixed_threshold_metrics": fixed,
                "train_calibrated_threshold_metrics": calibrated,
            }
        )
        all_eval_y.extend(y_eval.tolist())
        all_eval_prob.extend(eval_prob.tolist())
    aggregate: dict[str, Any] = {}
    if all_eval_y:
        yy = np.asarray(all_eval_y, dtype=float)
        pp = np.asarray(all_eval_prob, dtype=float)
        aggregate = {
            "fixed_threshold_metrics": _metrics(yy, pp),
            "global_eval_best_threshold": _best_threshold(yy, pp),
        }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(rows),
        "periods": period_names,
        "evaluated_periods": results,
        "aggregate": aggregate,
        "leakage_guard": {
            "each_period_fits_only_past_rows": True,
            "feature_space_selected_from_train_rows_only": True,
            "threshold_calibrated_on_train_rows_only": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward diagnostic for linear-alpha meta-controller features")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--period", choices=["year", "halfyear", "quarter"], default=WalkForwardConfig.period)
    p.add_argument("--max-features", type=int, default=WalkForwardConfig.max_features)
    p.add_argument("--min-train-rows", type=int, default=WalkForwardConfig.min_train_rows)
    p.add_argument("--min-eval-rows", type=int, default=WalkForwardConfig.min_eval_rows)
    p.add_argument("--lr", type=float, default=WalkForwardConfig.lr)
    p.add_argument("--steps", type=int, default=WalkForwardConfig.steps)
    p.add_argument("--l2", type=float, default=WalkForwardConfig.l2)
    p.add_argument("--train-window-periods", type=int, default=WalkForwardConfig.train_window_periods)
    p.add_argument("--include-no-trade", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = WalkForwardConfig(
        inputs=args.inputs,
        output=args.output,
        period=args.period,
        max_features=args.max_features,
        min_train_rows=args.min_train_rows,
        min_eval_rows=args.min_eval_rows,
        lr=args.lr,
        steps=args.steps,
        l2=args.l2,
        train_window_periods=args.train_window_periods,
        trade_only=not bool(args.include_no_trade),
    )
    report = run(cfg)
    print(json.dumps({"rows": report["rows"], "evaluated_periods": report["evaluated_periods"], "aggregate": report["aggregate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
