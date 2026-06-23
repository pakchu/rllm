"""Rolling stable-context selector and strict diagnostic backtest.

For each evaluation month, select contexts using only rows before that month:
- train window: historical rows before validation window;
- validation window: recent rows before the month;
- target month: transformed with the frozen selected context map.

This prevents a static train/test context map from overfitting one holdout era.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.backtest_single_policy_predictions import simulate
from training.stable_context_policy_dataset import StableContextPolicyCfg, select_contexts, transform_rows
from training.strict_bar_backtest import BarExecutionConfig


@dataclass(frozen=True)
class RollingStableContextCfg:
    input_jsonl: str
    predictions_output: str
    summary_output: str
    market_csv: str = ""
    backtest_output: str = ""
    eval_start: str = "2024-07-01"
    eval_end: str = "2026-06-01"
    train_days: int = 730
    validation_days: int = 180
    context_keys: str = StableContextPolicyCfg.context_keys
    min_train_rows: int = 8
    min_test_rows: int = 3
    min_train_mean_pct: float = 0.05
    min_test_mean_pct: float = 0.0
    min_train_gap_pct: float = 0.05
    min_test_gap_pct: float = -0.05
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["_dt"] = pd.Timestamp(str(r["date"]))
                rows.append(r)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return sorted(rows, key=lambda r: r["_dt"])


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).normalize().replace(day=1), pd.Timestamp(end).normalize(), freq="MS"))


def _selector_cfg(cfg: RollingStableContextCfg) -> StableContextPolicyCfg:
    return StableContextPolicyCfg(
        input_jsonl=cfg.input_jsonl,
        output=cfg.predictions_output,
        context_keys=cfg.context_keys,
        min_train_rows=cfg.min_train_rows,
        min_test_rows=cfg.min_test_rows,
        min_train_mean_pct=cfg.min_train_mean_pct,
        min_test_mean_pct=cfg.min_test_mean_pct,
        min_train_gap_pct=cfg.min_train_gap_pct,
        min_test_gap_pct=cfg.min_test_gap_pct,
    )


def _relabel_for_selection(rows: list[dict[str, Any]], train_start: pd.Timestamp, val_start: pd.Timestamp, month_start: pd.Timestamp) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        dt = r["_dt"]
        if train_start <= dt < val_start:
            nr = dict(r); nr["split"] = "train"; out.append(nr)
        elif val_start <= dt < month_start:
            nr = dict(r); nr["split"] = "test"; out.append(nr)
    return out


def rolling_transform(cfg: RollingStableContextCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(cfg.input_jsonl)
    sel_cfg = _selector_cfg(cfg)
    eval_start = pd.Timestamp(cfg.eval_start)
    eval_end = pd.Timestamp(cfg.eval_end)
    predictions: list[dict[str, Any]] = []
    months: list[dict[str, Any]] = []
    for mstart in month_starts(cfg.eval_start, cfg.eval_end):
        if mstart >= eval_end:
            continue
        mend = min(mstart + pd.offsets.MonthBegin(1), eval_end)
        train_start = mstart - pd.Timedelta(days=int(cfg.validation_days) + int(cfg.train_days))
        val_start = mstart - pd.Timedelta(days=int(cfg.validation_days))
        selection_rows = _relabel_for_selection(rows, train_start, val_start, mstart)
        month_rows = [dict(r) for r in rows if max(eval_start, mstart) <= r["_dt"] < mend]
        if not month_rows:
            continue
        selected, diag = select_contexts(selection_rows, sel_cfg)
        transformed = transform_rows(month_rows, selected, sel_cfg)
        for r in transformed:
            r.pop("_dt", None)
            r["rolling_context_window"] = {
                "month": str(mstart.date())[:7],
                "train_start": str(train_start),
                "validation_start": str(val_start),
                "selection_cutoff_exclusive": str(mstart),
            }
        predictions.extend(transformed)
        action_counts: dict[str, int] = {}
        selected_count = 0
        for r in transformed:
            action = json.loads(str(r["target"])).get("action", "NO_TRADE")
            action_counts[action] = action_counts.get(action, 0) + 1
            selected_count += int(bool((r.get("context_selection") or {}).get("selected")))
        months.append({
            "month": str(mstart.date())[:7],
            "train_start": str(train_start),
            "validation_start": str(val_start),
            "rows": len(transformed),
            "selected_rows": selected_count,
            "action_counts": dict(sorted(action_counts.items())),
            "selection": diag,
        })
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(predictions),
        "months": months,
        "leakage_guard": {
            "each_month_selection_uses_rows_before_month_only": True,
            "target_month_not_used_for_selection": True,
            "eval_not_globally_tuned": True,
        },
    }
    return predictions, summary


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def run(cfg: RollingStableContextCfg) -> dict[str, Any]:
    predictions, summary = rolling_transform(cfg)
    write_jsonl(cfg.predictions_output, predictions)
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    report: dict[str, Any] = {"summary": summary}
    if cfg.market_csv and cfg.backtest_output:
        exec_cfg = BarExecutionConfig(
            leverage=cfg.leverage,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            drawdown_stop=1.0,
            pause_bars=0,
            monthly_loss_stop=1.0,
            entry_delay_bars=cfg.entry_delay_bars,
        )
        bt = simulate(predictions, cfg.market_csv, exec_cfg, cooldown_bars=cfg.cooldown_bars, allow_target_echo=True)
        bt_report = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "inputs": {"predictions_jsonl": cfg.predictions_output, "market_csv": cfg.market_csv},
            "result": bt,
            "leakage_guard": summary["leakage_guard"] | {"target_echo_is_frozen_rolling_context_map": True},
        }
        Path(cfg.backtest_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.backtest_output).write_text(json.dumps(bt_report, indent=2, ensure_ascii=False))
        report["backtest"] = bt_report
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling stable context policy transform/backtest")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--market-csv", default="")
    p.add_argument("--backtest-output", default="")
    p.add_argument("--eval-start", default=RollingStableContextCfg.eval_start)
    p.add_argument("--eval-end", default=RollingStableContextCfg.eval_end)
    p.add_argument("--train-days", type=int, default=RollingStableContextCfg.train_days)
    p.add_argument("--validation-days", type=int, default=RollingStableContextCfg.validation_days)
    p.add_argument("--context-keys", default=RollingStableContextCfg.context_keys)
    p.add_argument("--min-train-rows", type=int, default=RollingStableContextCfg.min_train_rows)
    p.add_argument("--min-test-rows", type=int, default=RollingStableContextCfg.min_test_rows)
    p.add_argument("--min-train-mean-pct", type=float, default=RollingStableContextCfg.min_train_mean_pct)
    p.add_argument("--min-test-mean-pct", type=float, default=RollingStableContextCfg.min_test_mean_pct)
    p.add_argument("--min-train-gap-pct", type=float, default=RollingStableContextCfg.min_train_gap_pct)
    p.add_argument("--min-test-gap-pct", type=float, default=RollingStableContextCfg.min_test_gap_pct)
    p.add_argument("--leverage", type=float, default=RollingStableContextCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=RollingStableContextCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=RollingStableContextCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=RollingStableContextCfg.entry_delay_bars)
    p.add_argument("--cooldown-bars", type=int, default=RollingStableContextCfg.cooldown_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RollingStableContextCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
