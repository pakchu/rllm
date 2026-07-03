"""Sweep online risk overlays on train and replay selected configs on test/eval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


def _score(sim: dict[str, Any], stats: dict[str, Any]) -> float:
    return float(sim["cagr_to_strict_mdd"]) + 0.03 * float(sim["cagr_pct"]) - 0.15 * max(0.0, float(sim["strict_mdd_pct"]) - 12.0) + 0.001 * float(sim["trade_entries"]) - 0.25 * max(0.0, float(stats.get("p_value_mean_ret_approx", 1.0)) - 0.1)


def _cfg(predictions: str, market: str, output: str, params: dict[str, Any]) -> OnlineRiskOverlayConfig:
    return OnlineRiskOverlayConfig(predictions_jsonl=predictions, market_csv=market, output=output, leverage=0.5, max_hold_bars=144, **params)


def sweep_overlay(*, train_predictions: str, test_predictions: str, eval_predictions: str, market_csv: str, output: str, work_dir: str) -> dict[str, Any]:
    grid = []
    for pause_after_losses in [0, 2, 3]:
        for pause_bars in [288, 864, 2016]:
            for rolling_window_trades in [0, 8, 16, 32]:
                for rolling_loss_stop_pct in ([0.0, 2.0, 4.0, 6.0] if rolling_window_trades else [0.0]):
                    for rolling_drawdown_stop_pct in ([0.0, 3.0, 5.0, 8.0] if rolling_window_trades else [0.0]):
                        for monthly_loss_stop_pct in [0.0, 3.0, 5.0, 8.0]:
                            grid.append({
                                "pause_after_losses": pause_after_losses,
                                "pause_bars": pause_bars,
                                "rolling_window_trades": rolling_window_trades,
                                "rolling_loss_stop_pct": rolling_loss_stop_pct,
                                "rolling_drawdown_stop_pct": rolling_drawdown_stop_pct,
                                "monthly_loss_stop_pct": monthly_loss_stop_pct,
                            })
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, params in enumerate(grid):
        train_out = str(Path(work_dir) / f"train_{i:04d}.json")
        train = run_overlay(_cfg(train_predictions, market_csv, train_out, params))
        sim = train["sim"]
        if sim["trade_entries"] < 80 or sim["cagr_pct"] <= 0.0:
            continue
        row = {"params": params, "train": {"sim": sim, "trade_stats": train["trade_stats"]}, "selection_score": _score(sim, train["trade_stats"])}
        rows.append(row)
    rows.sort(key=lambda r: r["selection_score"], reverse=True)
    top = rows[:20]
    for i, row in enumerate(top):
        params = row["params"]
        for split, pred in [("test", test_predictions), ("eval", eval_predictions)]:
            res = run_overlay(_cfg(pred, market_csv, str(Path(work_dir) / f"{split}_top{i:02d}.json"), params))
            row[split] = {"sim": res["sim"], "trade_stats": res["trade_stats"]}
    report = {
        "leakage_guard": {"configs_ranked_on_train_only": True, "test_eval_replayed_after_selection": True, "overlay_uses_only_completed_prior_trades": True},
        "grid_count": len(grid),
        "candidate_count": len(rows),
        "top": top,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep online risk overlays")
    p.add_argument("--train-predictions", required=True)
    p.add_argument("--test-predictions", required=True)
    p.add_argument("--eval-predictions", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(sweep_overlay(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
