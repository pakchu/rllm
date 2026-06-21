"""Sweep wave-teacher thresholds under rllm's strict 5m execution replay.

The native wave_trading sweep evaluates returns inside wave_trading.  This script
keeps the same rolling-train 15m teacher probabilities but replays candidate
signals through rllm's online evaluator with timeframe-aligned 5m execution:
entry_delay_bars=3 and optional ATR trailing stop.  It is intended to prevent the
5m/15m fill mismatch that can invert conclusions.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.validate_wave_trading_best import _build_best_features, _load_wave_module


@dataclass(frozen=True)
class RllmThresholdSweepConfig:
    wave_root: str
    market_5m_csv: str
    output: str
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-02"
    test_start: str = "2024-07-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    lr_c: float = 0.05
    lr_penalty: str = "l1"
    long_thresholds: str = "0.57,0.60,0.63,0.66,0.69"
    short_thresholds: str = "0.32,0.35,0.38,0.41,0.44"
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45
    min_test_trades: int = 50


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _read_5m_dates(path: str) -> np.ndarray:
    df = pd.read_csv(path, usecols=["date"], compression="gzip" if path.endswith(".gz") else None)
    return pd.to_datetime(df["date"], errors="raise").to_numpy(dtype="datetime64[ns]")


def _rolling_prob_rows(psr: Any, data: dict[str, Any], *, market_5m_csv: str, eval_start: str, eval_end: str, lr_c: float, lr_penalty: str) -> list[dict[str, Any]]:
    X = data["X"]
    dates15 = np.asarray(data["dates"], dtype="datetime64[ns]")
    y_label = data["y_label"]
    long_ret = data["long_ret"]
    short_ret = data["short_ret"]
    holding = int(data["params"]["holding_period"])
    bars_per_month = int(data["bars_per_month"])
    train_bars = int(data["params"]["train_months"] * bars_per_month)
    test_bars = int(data["params"]["test_months"] * bars_per_month)
    purge_gap = holding * 2
    start_dt = np.datetime64(eval_start)
    end_dt = np.datetime64(eval_end)
    valid = (~np.any(~np.isfinite(X), axis=1)) & np.isfinite(y_label) & np.isfinite(long_ret) & np.isfinite(short_ret) & (y_label != 0)
    y_bin = (y_label > 0).astype(np.int32)
    dates5 = _read_5m_dates(market_5m_csv)
    rows: list[dict[str, Any]] = []
    seen_pos: set[int] = set()
    pos = max(0, int(np.searchsorted(dates15, start_dt)) - purge_gap - train_bars)
    while pos + train_bars + purge_gap < len(X):
        train_start = pos
        train_end = pos + train_bars
        test_start = train_end + purge_gap
        test_end = min(test_start + test_bars, len(X))
        if dates15[test_start] > end_dt:
            break
        if dates15[test_end - 1] < start_dt:
            pos += test_bars
            continue
        tr = np.arange(train_start, train_end)
        te = np.arange(test_start, test_end)
        tr = tr[valid[tr]]
        te = te[valid[te] & (dates15[te] >= start_dt) & (dates15[te] <= end_dt)]
        if len(tr) < 1000 or len(te) < 100:
            pos += test_bars
            continue
        proba = psr._predict_proba_lr(X[tr], y_bin[tr], X[te], lr_c, lr_penalty)
        for idx, prob in zip(te, proba):
            dt = dates15[idx]
            p5 = int(np.searchsorted(dates5, dt, side="right") - 1)
            if p5 < 0 or p5 in seen_pos:
                continue
            seen_pos.add(p5)
            rows.append({"date": str(pd.Timestamp(dt)), "signal_pos": p5, "teacher_15m_index": int(idx), "teacher_probability_long": float(prob)})
        pos += test_bars
    rows.sort(key=lambda r: (int(r["signal_pos"]), str(r["date"])))
    return rows


def _write_predictions(rows: list[dict[str, Any]], path: Path, *, long_th: float, short_th: float, hold_bars: int) -> dict[str, Any]:
    trade_rows = long_count = short_count = 0
    with path.open("w") as f:
        for row in rows:
            prob = float(row["teacher_probability_long"])
            side = "NONE"
            if prob >= long_th:
                side = "LONG"
            elif prob <= short_th:
                side = "SHORT"
            pred = {"confidence": "HIGH", "family": "wave_teacher_threshold_sweep", "gate": "NO_TRADE", "hold_bars": 0, "side": "NONE"}
            if side != "NONE":
                pred = {"confidence": "HIGH", "family": "wave_teacher_threshold_sweep", "gate": "TRADE", "hold_bars": int(hold_bars), "side": side}
                trade_rows += 1
                long_count += int(side == "LONG")
                short_count += int(side == "SHORT")
            f.write(json.dumps({**row, "prediction": pred, "teacher_thresholds": {"long": long_th, "short": short_th}}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(rows), "trade_rows": trade_rows, "long": long_count, "short": short_count}


def _score(sim: dict[str, Any], min_trades: int) -> float:
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", -999.0) or -999.0)
    mdd = float(sim.get("strict_mdd_pct", 999.0) or 999.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
    if trades < min_trades or cagr <= 0.0 or mdd > 20.0:
        return -1000.0 + trades / 1000.0 + cagr / 1000.0 - max(0.0, mdd - 20.0) / 1000.0
    return ratio + min(1.0, trades / 200.0)


def run_sweep(cfg: RllmThresholdSweepConfig) -> dict[str, Any]:
    psr = _load_wave_module(cfg.wave_root)
    data = _build_best_features(psr, start_date=cfg.start_date, end_date=cfg.end_date, time_interval="15m")
    hold_bars = int(data["params"]["holding_period"]) * 3
    test_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.test_start, eval_end=cfg.test_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    eval_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.eval_start, eval_end=cfg.eval_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="rllm_wave_threshold_") as tmp_raw:
        tmp = Path(tmp_raw)
        for lt in _parse_floats(cfg.long_thresholds):
            for st in _parse_floats(cfg.short_thresholds):
                if st >= lt:
                    continue
                tag = f"lt{lt:.2f}_st{st:.2f}".replace(".", "p")
                pred_path = tmp / f"{tag}_test.jsonl"
                pred_summary = _write_predictions(test_rows, pred_path, long_th=lt, short_th=st, hold_bars=hold_bars)
                bt = run_overlay(OnlineRiskOverlayConfig(
                    predictions_jsonl=str(pred_path),
                    market_csv=cfg.market_5m_csv,
                    output=str(tmp / f"{tag}_test_bt.json"),
                    leverage=cfg.leverage,
                    entry_delay_bars=cfg.entry_delay_bars,
                    atr_trailing_stop_mult=cfg.atr_trailing_stop_mult,
                    atr_period=cfg.atr_period,
                ))
                row = {"long_th": lt, "short_th": st, "prediction_summary": pred_summary, "test_sim": bt["sim"], "test_trade_stats": bt["trade_stats"], "score": _score(bt["sim"], cfg.min_test_trades)}
                candidates.append(row)
                if best is None or float(row["score"]) > float(best["score"]):
                    best = row
        assert best is not None
        eval_pred_path = tmp / "selected_eval.jsonl"
        eval_pred_summary = _write_predictions(eval_rows, eval_pred_path, long_th=float(best["long_th"]), short_th=float(best["short_th"]), hold_bars=hold_bars)
        eval_bt = run_overlay(OnlineRiskOverlayConfig(
            predictions_jsonl=str(eval_pred_path),
            market_csv=cfg.market_5m_csv,
            output=str(tmp / "selected_eval_bt.json"),
            leverage=cfg.leverage,
            entry_delay_bars=cfg.entry_delay_bars,
            atr_trailing_stop_mult=cfg.atr_trailing_stop_mult,
            atr_period=cfg.atr_period,
        ))
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    report = {
        "config": asdict(cfg),
        "teacher_params": data["params"],
        "data": {"test_rows": len(test_rows), "eval_rows": len(eval_rows), "hold_bars_5m": hold_bars},
        "selection_rule": "select threshold on test only; eval replays selected threshold unchanged",
        "best_test": best,
        "selected_eval": {"prediction_summary": eval_pred_summary, "eval_sim": eval_bt["sim"], "eval_trade_stats": eval_bt["trade_stats"]},
        "top10": candidates[:10],
        "leakage_guard": {"rolling_train_before_test": True, "purge_gap_bars_15m": int(data["params"]["holding_period"] * 2), "mapped_to_5m_without_future": True, "entry_delay_bars_aligns_next_15m_bar": int(cfg.entry_delay_bars) == 3, "eval_not_used_for_selection": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep 15m wave teacher thresholds through rllm strict replay")
    p.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    p.add_argument("--market-5m-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2026-06-02")
    p.add_argument("--test-start", default="2024-07-01")
    p.add_argument("--test-end", default="2025-12-31 23:59:59")
    p.add_argument("--eval-start", default="2026-01-01")
    p.add_argument("--eval-end", default="2026-06-01 00:00:00")
    p.add_argument("--lr-c", type=float, default=0.05)
    p.add_argument("--lr-penalty", default="l1")
    p.add_argument("--long-thresholds", default="0.57,0.60,0.63,0.66,0.69")
    p.add_argument("--short-thresholds", default="0.32,0.35,0.38,0.41,0.44")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=3)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=3.75)
    p.add_argument("--atr-period", type=int, default=45)
    p.add_argument("--min-test-trades", type=int, default=50)
    return p.parse_args()


def main() -> None:
    report = run_sweep(RllmThresholdSweepConfig(**vars(parse_args())))
    print(json.dumps({"best_test": report["best_test"], "selected_eval": report["selected_eval"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
