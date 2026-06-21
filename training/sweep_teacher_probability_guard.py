"""Sweep a wave-teacher probability guard over a base prediction stream.

The base model proposes direction/holding period.  The wave teacher supplies a
causal rolling probability from a separate feature family.  A trade is allowed
only when the teacher probability is direction-consistent and above/below the
chosen threshold.  Thresholds are selected on test and then replayed unchanged
on eval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bisect

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class GuardSweepConfig:
    base_test_predictions: str
    teacher_test_predictions: str
    base_eval_predictions: str
    teacher_eval_predictions: str
    market_csv: str
    work_dir: str
    output: str
    long_thresholds: str = "0.54,0.57,0.60,0.63,0.66"
    short_thresholds: str = "0.35,0.38,0.41,0.44,0.47"
    leverage: float = 0.3
    pause_after_losses: int = 4
    pause_bars: int = 288
    min_test_trades: int = 30


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1)))
    return rows


def _float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _teacher_series(path: str) -> tuple[list[int], dict[int, float]]:
    positions: list[int] = []
    probs: dict[int, float] = {}
    for row in _read_jsonl(path):
        if "teacher_probability_long" not in row:
            continue
        pos = int(row.get("signal_pos", -1) or -1)
        if pos < 0:
            continue
        positions.append(pos)
        probs[pos] = float(row["teacher_probability_long"])
    positions = sorted(set(positions))
    return positions, probs


def _asof_teacher_probability(positions: list[int], probs: dict[int, float], signal_pos: int, max_lag_bars: int = 2) -> float | None:
    idx = bisect.bisect_right(positions, signal_pos) - 1
    if idx < 0:
        return None
    pos = positions[idx]
    if signal_pos - pos > max_lag_bars:
        return None
    return probs.get(pos)


def _guard_predictions(base_path: str, teacher_path: str, output: Path, long_th: float, short_th: float) -> dict[str, Any]:
    teacher_positions, teacher_probs = _teacher_series(teacher_path)
    kept = blocked = missing = no_trade = 0
    side_counts = {"LONG": 0, "SHORT": 0}
    with output.open("w") as f:
        for row in _read_jsonl(base_path):
            pred = dict(row.get("prediction", {}))
            gate = str(pred.get("gate", "NO_TRADE"))
            side = str(pred.get("side", "NONE"))
            if gate == "TRADE" and side in {"LONG", "SHORT"}:
                signal_pos = int(row.get("signal_pos", -1) or -1)
                prob = _asof_teacher_probability(teacher_positions, teacher_probs, signal_pos)
                allowed = False
                if prob is None:
                    missing += 1
                elif side == "LONG":
                    allowed = prob >= long_th
                elif side == "SHORT":
                    allowed = prob <= short_th
                if allowed:
                    kept += 1
                    side_counts[side] += 1
                    row = dict(row)
                    row["teacher_probability_long"] = prob
                    row["teacher_guard"] = {"long_threshold": long_th, "short_threshold": short_th}
                else:
                    blocked += 1
                    row = dict(row)
                    row["prediction"] = {"confidence": "LOW", "family": pred.get("family", "base"), "gate": "NO_TRADE", "hold_bars": 0, "side": "NONE"}
                    row["teacher_probability_long"] = prob
                    row["teacher_guard"] = {"long_threshold": long_th, "short_threshold": short_th, "blocked_base_side": side}
            else:
                no_trade += 1
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": str(output), "kept_trade_rows": kept, "blocked_trade_rows": blocked, "missing_teacher_probability": missing, "base_no_trade_rows": no_trade, "side_counts": side_counts}


def _score(sim: dict[str, Any], min_trades: int) -> float:
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", -999.0) or -999.0)
    mdd = float(sim.get("strict_mdd_pct", 999.0) or 999.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
    if trades < min_trades or mdd > 15.0 or cagr <= 0.0:
        return -1e9 + trades + cagr - mdd
    return ratio * 1000.0 + trades


def run_sweep(cfg: GuardSweepConfig) -> dict[str, Any]:
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    for long_th in _float_list(cfg.long_thresholds):
        for short_th in _float_list(cfg.short_thresholds):
            tag = f"lt{long_th:.2f}_st{short_th:.2f}".replace(".", "p")
            test_pred = _guard_predictions(cfg.base_test_predictions, cfg.teacher_test_predictions, work / f"{tag}_test_predictions.jsonl", long_th, short_th)
            test_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=test_pred["path"], market_csv=cfg.market_csv, output=str(work / f"{tag}_test_backtest.json"), leverage=cfg.leverage, pause_after_losses=cfg.pause_after_losses, pause_bars=cfg.pause_bars))
            row = {"long_threshold": long_th, "short_threshold": short_th, "test_guard": test_pred, "test_sim": test_bt["sim"], "test_trade_stats": test_bt["trade_stats"], "score": _score(test_bt["sim"], cfg.min_test_trades)}
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
    assert best is not None
    eval_pred = _guard_predictions(cfg.base_eval_predictions, cfg.teacher_eval_predictions, work / "selected_eval_predictions.jsonl", float(best["long_threshold"]), float(best["short_threshold"]))
    eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred["path"], market_csv=cfg.market_csv, output=str(work / "selected_eval_backtest.json"), leverage=cfg.leverage, pause_after_losses=cfg.pause_after_losses, pause_bars=cfg.pause_bars))
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "selection_rule": "maximize test score with min trades, positive cagr, strict_mdd<=15; eval uses selected thresholds unchanged",
        "best_test": best,
        "selected_eval": {"eval_guard": eval_pred, "eval_sim": eval_bt["sim"], "eval_trade_stats": eval_bt["trade_stats"]},
        "top10": rows[:10],
        "leakage_guard": {"thresholds_selected_on_test_only": True, "eval_thresholds_frozen_before_eval": True, "teacher_predictions_are_rolling_train_before_test": True, "teacher_probability_asof_max_lag_5m_bars": 2},
    }
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep wave-teacher probability guard over base predictions")
    for name in ["base-test-predictions", "teacher-test-predictions", "base-eval-predictions", "teacher-eval-predictions", "market-csv", "work-dir", "output"]:
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--long-thresholds", default="0.54,0.57,0.60,0.63,0.66")
    p.add_argument("--short-thresholds", default="0.35,0.38,0.41,0.44,0.47")
    p.add_argument("--leverage", type=float, default=0.3)
    p.add_argument("--pause-after-losses", type=int, default=4)
    p.add_argument("--pause-bars", type=int, default=288)
    p.add_argument("--min-test-trades", type=int, default=30)
    return p.parse_args()


def main() -> None:
    report = run_sweep(GuardSweepConfig(**vars(parse_args())))
    print(json.dumps({"best_test": report["best_test"], "selected_eval": report["selected_eval"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
