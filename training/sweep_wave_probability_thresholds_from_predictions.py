"""Sweep wave-teacher probability thresholds from cached prediction rows."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class WaveProbThresholdSweepCfg:
    test_predictions_jsonl: str
    eval_predictions_jsonl: str
    market_csv: str
    output: str
    long_thresholds: str = "0.54,0.56,0.58,0.60,0.62,0.64,0.66"
    short_thresholds: str = "0.34,0.36,0.38,0.40,0.42,0.44,0.46"
    hold_bars: int = 12
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45
    min_test_trades: int = 30
    top_k: int = 30


def _read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _parse(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _write_policy(rows: list[dict[str, Any]], path: Path, *, long_th: float, short_th: float, hold_bars: int) -> None:
    out = []
    for row in rows:
        p = float(row.get("teacher_probability_long", 0.5) or 0.5)
        if p >= float(long_th):
            pred = {"gate": "TRADE", "side": "LONG", "hold_bars": int(hold_bars), "confidence": "HIGH", "family": "wave_prob_threshold"}
        elif p <= float(short_th):
            pred = {"gate": "TRADE", "side": "SHORT", "hold_bars": int(hold_bars), "confidence": "HIGH", "family": "wave_prob_threshold"}
        else:
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "wave_prob_threshold"}
        out.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "teacher_probability_long": p, "prediction": pred, "thresholds": {"long": long_th, "short": short_th}})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")


def _bt(path: Path, cfg: WaveProbThresholdSweepCfg) -> dict[str, Any]:
    return run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=str(path), market_csv=cfg.market_csv, output=str(path.with_suffix(".bt.json")),
        leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period,
    ))


def _score(bt: dict[str, Any], min_trades: int) -> float:
    sim = bt["sim"]; stats = bt["trade_stats"]
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -1e9 + trades
    return float(sim.get("cagr_to_strict_mdd", -999) or -999) + 0.02 * float(sim.get("cagr_pct", 0) or 0) - float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)


def _small(bt: dict[str, Any]) -> dict[str, Any]:
    return {"period": bt.get("period"), "sim": bt.get("sim"), "trade_stats": bt.get("trade_stats")}


def run(cfg: WaveProbThresholdSweepCfg) -> dict[str, Any]:
    test_rows = _read(cfg.test_predictions_jsonl)
    eval_rows = _read(cfg.eval_predictions_jsonl)
    rows = []
    with tempfile.TemporaryDirectory(prefix="wave_prob_sweep_") as td:
        tmp = Path(td)
        for lt in _parse(cfg.long_thresholds):
            for st in _parse(cfg.short_thresholds):
                if st >= lt:
                    continue
                test_path = tmp / f"test_l{lt:.2f}_s{st:.2f}.jsonl"
                eval_path = tmp / f"eval_l{lt:.2f}_s{st:.2f}.jsonl"
                _write_policy(test_rows, test_path, long_th=lt, short_th=st, hold_bars=int(cfg.hold_bars))
                _write_policy(eval_rows, eval_path, long_th=lt, short_th=st, hold_bars=int(cfg.hold_bars))
                test_bt = _bt(test_path, cfg)
                eval_bt = _bt(eval_path, cfg)
                rows.append({"long_th": lt, "short_th": st, "test": _small(test_bt), "eval": _small(eval_bt), "selection_score": _score(test_bt, int(cfg.min_test_trades))})
    rows.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    report = {"config": asdict(cfg), "ranked_by_test": rows[: int(cfg.top_k)], "all_count": len(rows), "leakage_guard": {"thresholds_ranked_on_test_only": True, "eval_not_used_for_selection": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-predictions-jsonl", required=True)
    p.add_argument("--eval-predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--long-thresholds", default=WaveProbThresholdSweepCfg.long_thresholds)
    p.add_argument("--short-thresholds", default=WaveProbThresholdSweepCfg.short_thresholds)
    p.add_argument("--hold-bars", type=int, default=WaveProbThresholdSweepCfg.hold_bars)
    p.add_argument("--leverage", type=float, default=WaveProbThresholdSweepCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=WaveProbThresholdSweepCfg.entry_delay_bars)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=WaveProbThresholdSweepCfg.atr_trailing_stop_mult)
    p.add_argument("--atr-period", type=int, default=WaveProbThresholdSweepCfg.atr_period)
    p.add_argument("--min-test-trades", type=int, default=WaveProbThresholdSweepCfg.min_test_trades)
    p.add_argument("--top-k", type=int, default=WaveProbThresholdSweepCfg.top_k)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(WaveProbThresholdSweepCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
