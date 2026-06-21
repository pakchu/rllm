"""Evaluate past-only regime filters for 15m wave-teacher candidates.

This tests a structurally different path from global threshold tuning: keep the
wave teacher as a candidate generator, then allow long/short candidates only in
simple past-only market regimes.  Selection is performed on a pre-period and the
chosen regime policy is replayed unchanged on a later held-out period.
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
from training.sweep_wave_teacher_rllm_thresholds import _rolling_prob_rows
from training.validate_wave_trading_best import _build_best_features, _load_wave_module


@dataclass(frozen=True)
class WaveRegimeFilterConfig:
    wave_root: str
    market_5m_csv: str
    output: str
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-02"
    test_start: str = "2023-01-01"
    test_end: str = "2024-06-30 23:59:59"
    eval_start: str = "2024-07-01"
    eval_end: str = "2026-06-01 00:00:00"
    lr_c: float = 0.05
    lr_penalty: str = "l1"
    long_thresholds: str = "0.66,0.69"
    short_thresholds: str = "0.32,0.35"
    trend_windows: str = "288,864"
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45
    min_test_trades: int = 40


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _load_market(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)


def _trend_at(closes: np.ndarray, pos: int, window: int) -> float | None:
    ref = pos - int(window)
    if ref < 0 or pos >= len(closes):
        return None
    a = float(closes[ref])
    b = float(closes[pos])
    if a <= 0 or b <= 0:
        return None
    return b / a - 1.0


def _allow_by_regime(side: str, trend: float | None, mode: str) -> bool:
    if mode == "all":
        return True
    if trend is None:
        return False
    if mode == "up":
        return trend >= 0.0
    if mode == "down":
        return trend < 0.0
    if mode == "contrarian_long":
        return side == "LONG" and trend < 0.0
    if mode == "contrarian_short":
        return side == "SHORT" and trend >= 0.0
    raise ValueError(f"unknown regime mode: {mode}")


def _write_predictions(rows: list[dict[str, Any]], path: Path, *, closes: np.ndarray, long_th: float, short_th: float, trend_window: int, long_mode: str, short_mode: str, hold_bars: int) -> dict[str, Any]:
    trade_rows = blocked_regime = long_count = short_count = 0
    with path.open("w") as f:
        for row in rows:
            pos = int(row["signal_pos"])
            prob = float(row["teacher_probability_long"])
            side = "NONE"
            if prob >= long_th:
                side = "LONG"
            elif prob <= short_th:
                side = "SHORT"
            trend = _trend_at(closes, pos, trend_window)
            if side == "LONG" and not _allow_by_regime(side, trend, long_mode):
                blocked_regime += 1
                side = "NONE"
            if side == "SHORT" and not _allow_by_regime(side, trend, short_mode):
                blocked_regime += 1
                side = "NONE"
            pred = {"confidence": "HIGH", "family": "wave_teacher_regime_filter", "gate": "NO_TRADE", "hold_bars": 0, "side": "NONE"}
            if side != "NONE":
                pred = {"confidence": "HIGH", "family": "wave_teacher_regime_filter", "gate": "TRADE", "hold_bars": int(hold_bars), "side": side}
                trade_rows += 1
                long_count += int(side == "LONG")
                short_count += int(side == "SHORT")
            f.write(json.dumps({**row, "prediction": pred, "teacher_thresholds": {"long": long_th, "short": short_th}, "regime_filter": {"trend_window_5m": trend_window, "trend": trend, "long_mode": long_mode, "short_mode": short_mode}}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(rows), "trade_rows": trade_rows, "blocked_regime": blocked_regime, "long": long_count, "short": short_count}


def _score(sim: dict[str, Any], stats: dict[str, Any], min_trades: int) -> float:
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", -999.0) or -999.0)
    mdd = float(sim.get("strict_mdd_pct", 999.0) or 999.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
    pval = float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
    if trades < min_trades or cagr <= 0.0 or mdd > 18.0:
        return -1000.0 + trades / 1000.0 + cagr / 1000.0 - max(0.0, mdd - 18.0) / 1000.0
    return ratio + min(1.0, trades / 200.0) + max(0.0, 0.25 - pval)


def run_sweep(cfg: WaveRegimeFilterConfig) -> dict[str, Any]:
    psr = _load_wave_module(cfg.wave_root)
    data = _build_best_features(psr, start_date=cfg.start_date, end_date=cfg.end_date, time_interval="15m")
    hold_bars = int(data["params"]["holding_period"]) * 3
    market = _load_market(cfg.market_5m_csv)
    closes = market["close"].to_numpy(dtype=float)
    test_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.test_start, eval_end=cfg.test_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    eval_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.eval_start, eval_end=cfg.eval_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    modes = ["all", "up", "down"]
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="rllm_wave_regime_") as tmp_raw:
        tmp = Path(tmp_raw)
        for lt in _parse_floats(cfg.long_thresholds):
            for st in _parse_floats(cfg.short_thresholds):
                if st >= lt:
                    continue
                for tw in _parse_ints(cfg.trend_windows):
                    for lm in modes:
                        for sm in modes:
                            tag = f"lt{lt:.2f}_st{st:.2f}_tw{tw}_lm{lm}_sm{sm}".replace(".", "p")
                            pred_path = tmp / f"{tag}_test.jsonl"
                            pred_summary = _write_predictions(test_rows, pred_path, closes=closes, long_th=lt, short_th=st, trend_window=tw, long_mode=lm, short_mode=sm, hold_bars=hold_bars)
                            bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred_path), market_csv=cfg.market_5m_csv, output=str(tmp / f"{tag}_test_bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
                            row = {"long_th": lt, "short_th": st, "trend_window": tw, "long_mode": lm, "short_mode": sm, "prediction_summary": pred_summary, "test_sim": bt["sim"], "test_trade_stats": bt["trade_stats"], "score": _score(bt["sim"], bt["trade_stats"], cfg.min_test_trades)}
                            candidates.append(row)
                            if best is None or float(row["score"]) > float(best["score"]):
                                best = row
        assert best is not None
        eval_pred = tmp / "selected_eval.jsonl"
        eval_summary = _write_predictions(eval_rows, eval_pred, closes=closes, long_th=float(best["long_th"]), short_th=float(best["short_th"]), trend_window=int(best["trend_window"]), long_mode=str(best["long_mode"]), short_mode=str(best["short_mode"]), hold_bars=hold_bars)
        eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(eval_pred), market_csv=cfg.market_5m_csv, output=str(tmp / "selected_eval_bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    report = {
        "config": asdict(cfg),
        "teacher_params": data["params"],
        "data": {"test_rows": len(test_rows), "eval_rows": len(eval_rows), "hold_bars_5m": hold_bars},
        "best_test": best,
        "selected_eval": {"prediction_summary": eval_summary, "eval_sim": eval_bt["sim"], "eval_trade_stats": eval_bt["trade_stats"]},
        "top20": candidates[:20],
        "leakage_guard": {"regime_features_use_signal_or_prior_close_only": True, "rolling_train_before_test": True, "eval_not_used_for_selection": True, "entry_delay_bars_aligns_next_15m_bar": int(cfg.entry_delay_bars) == 3},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep simple past-only regime filters over wave teacher candidates")
    p.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    p.add_argument("--market-5m-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2026-06-02")
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--test-end", default="2024-06-30 23:59:59")
    p.add_argument("--eval-start", default="2024-07-01")
    p.add_argument("--eval-end", default="2026-06-01 00:00:00")
    p.add_argument("--lr-c", type=float, default=0.05)
    p.add_argument("--lr-penalty", default="l1")
    p.add_argument("--long-thresholds", default="0.66,0.69")
    p.add_argument("--short-thresholds", default="0.32,0.35")
    p.add_argument("--trend-windows", default="288,864")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=3)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=3.75)
    p.add_argument("--atr-period", type=int, default=45)
    p.add_argument("--min-test-trades", type=int, default=40)
    return p.parse_args()


def main() -> None:
    report = run_sweep(WaveRegimeFilterConfig(**vars(parse_args())))
    print(json.dumps({"best_test": report["best_test"], "selected_eval": report["selected_eval"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
