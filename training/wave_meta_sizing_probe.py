"""Probe train-derived meta position sizing for robust wave policy."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.sweep_wave_fold_consistency import _load_closes, _parse_folds, _policy_side
from training.sweep_wave_teacher_rllm_thresholds import _rolling_prob_rows
from training.validate_wave_trading_best import _build_best_features, _load_wave_module
from training.wave_meta_abstain_probe import _executed_training_rows, _features, _fit_ridge


@dataclass(frozen=True)
class MetaSizingProbeConfig:
    wave_root: str
    market_5m_csv: str
    fold_consistency_report: str
    output: str
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-02"
    selection_folds: str = "2021-01-01|2021-06-30 23:59:59,2021-07-01|2021-12-31 23:59:59,2022-01-01|2022-06-30 23:59:59,2022-07-01|2022-12-31 23:59:59,2023-01-01|2023-06-30 23:59:59,2023-07-01|2023-12-31 23:59:59,2024-01-01|2024-06-30 23:59:59"
    eval_start: str = "2024-07-01"
    eval_end: str = "2026-06-01 00:00:00"
    lr_c: float = 0.05
    lr_penalty: str = "l1"
    ridge_lambda: float = 1.0
    low_scale: float = 0.3
    mid_scale: float = 0.6
    high_scale: float = 1.0
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45


def _write_sized_eval(rows, path: Path, *, closes: np.ndarray, policy: dict, hold_bars: int, w: np.ndarray, q25: float, q75: float, cfg: MetaSizingProbeConfig) -> dict:
    counts = {"LONG": 0, "SHORT": 0}
    scales = []
    with path.open("w") as f:
        for row in rows:
            side = _policy_side(row, closes, policy)
            pred = {"confidence": "HIGH", "family": "wave_meta_sizing", "gate": "NO_TRADE", "hold_bars": 0, "side": "NONE"}
            scale = 0.0
            score = -999.0
            if side in counts:
                score = float(np.asarray(_features({**row, "side": side}, closes, policy), dtype=float) @ w)
                if score <= q25:
                    scale = float(cfg.low_scale)
                elif score >= q75:
                    scale = float(cfg.high_scale)
                else:
                    scale = float(cfg.mid_scale)
                pred = {"confidence": "HIGH", "family": "wave_meta_sizing", "gate": "TRADE", "hold_bars": int(hold_bars), "side": side}
                counts[side] += 1
                scales.append(scale)
            f.write(json.dumps({**row, "prediction": pred, "position_scale": scale, "meta_score": score, "scale_bins": {"q25": q25, "q75": q75, "low": cfg.low_scale, "mid": cfg.mid_scale, "high": cfg.high_scale}}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(rows), "trade_rows": counts["LONG"] + counts["SHORT"], "long": counts["LONG"], "short": counts["SHORT"], "mean_position_scale": float(np.mean(scales)) if scales else 0.0}


def run_probe(cfg: MetaSizingProbeConfig) -> dict:
    report = json.loads(Path(cfg.fold_consistency_report).read_text())
    policy = report["selected_policies"][0]
    psr = _load_wave_module(cfg.wave_root)
    data = _build_best_features(psr, start_date=cfg.start_date, end_date=cfg.end_date, time_interval="15m")
    hold_bars = int(data["params"]["holding_period"]) * 3
    closes = _load_closes(cfg.market_5m_csv)
    folds = _parse_folds(cfg.selection_folds)
    train_rows_by_fold = [_rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=a, eval_end=b, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty) for a, b in folds]
    eval_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.eval_start, eval_end=cfg.eval_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    with tempfile.TemporaryDirectory(prefix="rllm_wave_meta_sizing_") as tmp_raw:
        tmp = Path(tmp_raw)
        train_exec = []
        for i, rows in enumerate(train_rows_by_fold):
            train_exec.extend(_executed_training_rows(rows, cfg.market_5m_csv, closes, policy, tmp, f"train_f{i}", cfg, hold_bars))
        xs = np.asarray([r["x"] for r in train_exec], dtype=float)
        y = np.asarray([r["reward"] for r in train_exec], dtype=float)
        w = _fit_ridge(xs, y, float(cfg.ridge_lambda))
        train_scores = xs @ w
        q25 = float(np.quantile(train_scores, 0.25))
        q75 = float(np.quantile(train_scores, 0.75))
        pred = tmp / "sized_eval.jsonl"
        pred_summary = _write_sized_eval(eval_rows, pred, closes=closes, policy=policy, hold_bars=hold_bars, w=w, q25=q25, q75=q75, cfg=cfg)
        bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=cfg.market_5m_csv, output=str(tmp / "sized_eval.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
    out = {
        "config": asdict(cfg),
        "base_policy": policy,
        "train_executed_trades": len(train_exec),
        "train_score_quantiles": {"q25": q25, "q75": q75},
        "weights": [float(v) for v in w],
        "heldout_eval": {"prediction_summary": pred_summary, "eval_sim": bt["sim"], "eval_trade_stats": bt["trade_stats"]},
        "leakage_guard": {"weights_and_scale_bins_train_only": True, "eval_replay_frozen": True, "features_are_signal_time_past_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe train-derived meta position sizing")
    p.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    p.add_argument("--market-5m-csv", required=True)
    p.add_argument("--fold-consistency-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2026-06-02")
    p.add_argument("--selection-folds", default=MetaSizingProbeConfig.selection_folds)
    p.add_argument("--eval-start", default="2024-07-01")
    p.add_argument("--eval-end", default="2026-06-01 00:00:00")
    p.add_argument("--lr-c", type=float, default=0.05)
    p.add_argument("--lr-penalty", default="l1")
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--low-scale", type=float, default=0.3)
    p.add_argument("--mid-scale", type=float, default=0.6)
    p.add_argument("--high-scale", type=float, default=1.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=3)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=3.75)
    p.add_argument("--atr-period", type=int, default=45)
    return p.parse_args()


def main() -> None:
    out = run_probe(MetaSizingProbeConfig(**vars(parse_args())))
    print(json.dumps({"train_executed_trades": out["train_executed_trades"], "train_score_quantiles": out["train_score_quantiles"], "heldout_eval": out["heldout_eval"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
