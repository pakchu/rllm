"""Leak-free monthly continual evaluation for Gemma action value rankers.

The base adapter is trained on a fixed historical window.  For each prediction
month, the evaluator scores only that month, then optionally fine-tunes the same
adapter on labels that became available after the month.  This tests whether an
LLM value-ranker can adapt to recent regimes without using future labels for the
current prediction month.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from training.economic_action_backtest import EconomicActionBacktestConfig, strict_backtest_actions
from training.score_action_value_candidates import score_value_rows
from training.strict_bar_backtest import load_market_bars
from training.train_text_sft import TextSFTConfig, train_text_sft


@dataclass(frozen=True)
class MonthFold:
    label: str
    start: str
    end: str


@dataclass(frozen=True)
class ContinualValueConfig:
    value_jsonl_paths: tuple[str, ...]
    market_csv: str
    model_name: str
    base_adapter_dir: str
    output: str
    work_dir: str = "results/continual_value_ranker"
    checkpoint_dir: str = "checkpoints/continual_value_ranker"
    start_month: str = "2024-01"
    end_month: str = "2024-12"
    label_embargo_bars: int = 432
    bar_minutes: int = 5
    batch_size: int = 8
    score_key: str = "mean"
    threshold: float = 0.0
    gate_mode: str = "fixed"
    rolling_quantile: float = 0.676
    rolling_warmup: int = 20
    rolling_history_predictions: tuple[str, ...] = ()
    update_steps: int = 16
    max_update_samples: int = 2048
    max_seq_length: int = 2048
    learning_rate: float = 1e-5
    sample_mode: str = "balanced"
    seed: int = 4100
    dry_run: bool = False
    skip_scoring: bool = False
    skip_training: bool = False


def _read_rows(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1), json.dumps(r.get("action", {}), sort_keys=True)))
    return rows


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def _month_folds(start_month: str, end_month: str) -> list[MonthFold]:
    y, m = [int(x) for x in start_month.split("-")]
    ey, em = [int(x) for x in end_month.split("-")]
    out: list[MonthFold] = []
    while (y, m) <= (ey, em):
        start = datetime(y, m, 1)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = datetime(ny, nm, 1)
        out.append(MonthFold(f"{y:04d}-{m:02d}", start.isoformat(sep=" "), end.isoformat(sep=" ")))
        y, m = ny, nm
    return out


def _filter_rows(rows: list[dict[str, Any]], start: datetime | None = None, end: datetime | None = None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        d = _dt(str(row["date"]))
        if start is not None and d < start:
            continue
        if end is not None and d >= end:
            continue
        out.append(row)
    return out


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _run_backtest(predictions_jsonl: str, market_csv: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(predictions_jsonl).read_text().splitlines() if line.strip()]
    return strict_backtest_actions(rows, load_market_bars(market_csv), EconomicActionBacktestConfig())


NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "NONE", "confidence": "HIGH"}


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("inf")
    xs = sorted(float(x) for x in values)
    idx = min(len(xs) - 1, max(0, int(float(q) * (len(xs) - 1))))
    return float(xs[idx])


def _load_margin_history(paths: tuple[str, ...]) -> list[float]:
    history: list[float] = []
    for path in paths:
        if not path:
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "value_margin" in row:
                history.append(float(row["value_margin"]))
    return history


def _apply_prediction_gate(
    *,
    predictions_jsonl: str | Path,
    output_jsonl: str | Path,
    mode: str,
    fixed_threshold: float,
    rolling_history: list[float],
    rolling_quantile: float,
    rolling_warmup: int,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(predictions_jsonl).read_text().splitlines() if line.strip()]
    out: list[dict[str, Any]] = []
    mode = str(mode).strip().lower()
    no_trade = 0
    trade = 0
    thresholds: list[float] = []
    for row in sorted(rows, key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1))):
        margin = float(row.get("value_margin", float("-inf")))
        if mode == "rolling_quantile":
            if len(rolling_history) < int(rolling_warmup):
                threshold = float("inf")
            else:
                threshold = _quantile(rolling_history, float(rolling_quantile))
            rolling_history.append(margin)
        elif mode == "fixed":
            threshold = float(fixed_threshold)
        else:
            raise ValueError("gate_mode must be one of {'fixed','rolling_quantile'}")
        pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
        if margin < threshold:
            pred = dict(NO_TRADE)
            no_trade += 1
        else:
            pred["gate"] = "TRADE"
            trade += 1
        thresholds.append(threshold)
        out.append({**row, "prediction": pred, "applied_gate_mode": mode, "applied_threshold": threshold})
    _write_jsonl(output_jsonl, out)
    finite_thresholds = [x for x in thresholds if x != float("inf")]
    return {
        "mode": mode,
        "output_jsonl": str(output_jsonl),
        "rows": len(out),
        "trade": trade,
        "no_trade": no_trade,
        "fixed_threshold": fixed_threshold,
        "rolling_quantile": rolling_quantile,
        "rolling_warmup": rolling_warmup,
        "threshold_min": min(finite_thresholds) if finite_thresholds else None,
        "threshold_max": max(finite_thresholds) if finite_thresholds else None,
        "history_size_after": len(rolling_history),
    }


def _aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    # Aggregate by compounding fold-level returns and taking max fold MDD as a conservative first-pass proxy.
    eq = 1.0
    start = end = ""
    trades = 0
    max_mdd = 0.0
    for fold in folds:
        sim = fold.get("backtest", {}).get("sim", {})
        period = fold.get("backtest", {}).get("period", {})
        if not start or str(period.get("start", "")) < start:
            start = str(period.get("start", ""))
        if not end or str(period.get("end", "")) > end:
            end = str(period.get("end", ""))
        eq *= 1.0 + float(sim.get("ret_pct", 0.0)) / 100.0
        trades += int(sim.get("trade_entries", 0) or 0)
        max_mdd = max(max_mdd, float(sim.get("strict_mdd_pct", 0.0) or 0.0))
    years = max(1.0 / 365.25, (_dt(end) - _dt(start)).days / 365.25) if start and end else 1.0
    cagr = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0 else -100.0
    return {"period": {"start": start, "end": end, "years": years}, "ret_pct": (eq - 1.0) * 100.0, "cagr_pct": cagr, "strict_mdd_pct_proxy": max_mdd, "cagr_to_strict_mdd_proxy": cagr / max_mdd if max_mdd > 1e-12 else 0.0, "trade_entries": trades}


def run_continual(cfg: ContinualValueConfig) -> dict[str, Any]:
    rows = _read_rows(cfg.value_jsonl_paths)
    folds = _month_folds(cfg.start_month, cfg.end_month)
    work = Path(cfg.work_dir)
    ckpts = Path(cfg.checkpoint_dir)
    work.mkdir(parents=True, exist_ok=True)
    ckpts.mkdir(parents=True, exist_ok=True)
    embargo = timedelta(minutes=int(cfg.label_embargo_bars) * int(cfg.bar_minutes))
    current_adapter = cfg.base_adapter_dir
    # The base adapter is treated as already trained through the first
    # prediction boundary.  Continual updates must only consume labels that
    # become available after the live evaluation starts.
    last_trained_until: datetime | None = _dt(folds[0].start) if folds else None
    report: dict[str, Any] = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "folds": []}
    rolling_history = _load_margin_history(cfg.rolling_history_predictions)

    for idx, fold in enumerate(folds, start=1):
        fold_start = _dt(fold.start)
        fold_end = _dt(fold.end)
        score_rows = _filter_rows(rows, fold_start, fold_end)
        pred_path = work / f"{fold.label}_predictions.jsonl"
        raw_pred_path = work / f"{fold.label}_raw_predictions.jsonl"
        scores_path = work / f"{fold.label}_scores.jsonl"
        fold_report: dict[str, Any] = {"fold": asdict(fold), "adapter_before_prediction": current_adapter, "score_rows": len(score_rows)}
        score_jsonl = work / f"{fold.label}_score_rows.jsonl"
        _write_jsonl(score_jsonl, score_rows)

        if not cfg.dry_run and not cfg.skip_scoring and score_rows:
            fold_report["score_summary"] = score_value_rows(
                value_jsonl=str(score_jsonl),
                predictions_output=str(raw_pred_path if cfg.gate_mode == "rolling_quantile" else pred_path),
                scores_output=str(scores_path),
                model_name=cfg.model_name,
                adapter_dir=current_adapter,
                batch_size=int(cfg.batch_size),
                max_signals=0,
                score_key=cfg.score_key,
                threshold=float(-1e9 if cfg.gate_mode == "rolling_quantile" else cfg.threshold),
            )
            if cfg.gate_mode == "rolling_quantile":
                fold_report["gate_summary"] = _apply_prediction_gate(
                    predictions_jsonl=raw_pred_path,
                    output_jsonl=pred_path,
                    mode=cfg.gate_mode,
                    fixed_threshold=float(cfg.threshold),
                    rolling_history=rolling_history,
                    rolling_quantile=float(cfg.rolling_quantile),
                    rolling_warmup=int(cfg.rolling_warmup),
                )
            fold_report["backtest"] = _run_backtest(str(pred_path), cfg.market_csv)

        train_end = fold_end - embargo
        train_start = last_trained_until
        update_rows = _filter_rows(rows, train_start, train_end)
        if last_trained_until is not None:
            update_rows = [r for r in update_rows if _dt(str(r["date"])) >= last_trained_until]
        update_jsonl = work / f"{fold.label}_update_rows.jsonl"
        _write_jsonl(update_jsonl, update_rows)
        next_adapter = ckpts / f"adapter_after_{fold.label}"
        fold_report["update"] = {"available_after_prediction_only": True, "train_start": train_start.isoformat(sep=" ") if train_start else None, "train_end": train_end.isoformat(sep=" "), "rows": len(update_rows), "output_adapter": str(next_adapter)}
        if not cfg.dry_run and not cfg.skip_training and update_rows:
            train_text_sft(
                TextSFTConfig(
                    model_name=cfg.model_name,
                    train_jsonl=str(update_jsonl),
                    output_dir=str(next_adapter),
                    max_samples=int(cfg.max_update_samples),
                    max_seq_length=int(cfg.max_seq_length),
                    sample_mode=cfg.sample_mode,
                    max_steps=int(cfg.update_steps),
                    learning_rate=float(cfg.learning_rate),
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=8,
                    seed=int(cfg.seed) + idx,
                    base_adapter_dir=current_adapter,
                )
            )
            current_adapter = str(next_adapter)
            last_trained_until = train_end
        elif update_rows:
            last_trained_until = train_end
        fold_report["adapter_after_update"] = current_adapter
        report["folds"].append(fold_report)
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    completed = [f for f in report["folds"] if "backtest" in f]
    if completed:
        report["summary"] = _aggregate(completed)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly leak-free continual Gemma value-ranker eval")
    p.add_argument("--value-jsonl-paths", required=True, help="Comma-separated value JSONL files")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--model-name", default="gemma4-e4b")
    p.add_argument("--base-adapter-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default="results/continual_value_ranker")
    p.add_argument("--checkpoint-dir", default="checkpoints/continual_value_ranker")
    p.add_argument("--start-month", default="2024-01")
    p.add_argument("--end-month", default="2024-12")
    p.add_argument("--label-embargo-bars", type=int, default=432)
    p.add_argument("--bar-minutes", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--score-key", choices=["mean", "sum"], default="mean")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--gate-mode", choices=["fixed", "rolling_quantile"], default="fixed")
    p.add_argument("--rolling-quantile", type=float, default=0.676)
    p.add_argument("--rolling-warmup", type=int, default=20)
    p.add_argument("--rolling-history-predictions", default="", help="Comma-separated prior prediction JSONL files used only as past margin history")
    p.add_argument("--update-steps", type=int, default=16)
    p.add_argument("--max-update-samples", type=int, default=2048)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="balanced")
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-scoring", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = ContinualValueConfig(
        value_jsonl_paths=tuple(x for x in a.value_jsonl_paths.split(",") if x),
        market_csv=a.market_csv,
        model_name=a.model_name,
        base_adapter_dir=a.base_adapter_dir,
        output=a.output,
        work_dir=a.work_dir,
        checkpoint_dir=a.checkpoint_dir,
        start_month=a.start_month,
        end_month=a.end_month,
        label_embargo_bars=a.label_embargo_bars,
        bar_minutes=a.bar_minutes,
        batch_size=a.batch_size,
        score_key=a.score_key,
        threshold=a.threshold,
        gate_mode=a.gate_mode,
        rolling_quantile=a.rolling_quantile,
        rolling_warmup=a.rolling_warmup,
        rolling_history_predictions=tuple(x for x in a.rolling_history_predictions.split(",") if x),
        update_steps=a.update_steps,
        max_update_samples=a.max_update_samples,
        max_seq_length=a.max_seq_length,
        learning_rate=a.learning_rate,
        sample_mode=a.sample_mode,
        seed=a.seed,
        dry_run=bool(a.dry_run),
        skip_scoring=bool(a.skip_scoring),
        skip_training=bool(a.skip_training),
    )
    out = run_continual(cfg)
    print(json.dumps({"summary": out.get("summary"), "folds": len(out.get("folds", [])), "output": cfg.output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
