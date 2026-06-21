"""Sweep thresholds over regime-conditioned policy score streams.

The regime policy writes one best scored action per timestamp regardless of the
final gate.  This helper replays that score stream with stricter/looser
thresholds without retraining, backtests each threshold set, and records the
selection result.  It is intended for leak-safe model selection on historical
validation before applying the selected thresholds to a later eval period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "REGIME_THRESHOLD", "confidence": "HIGH"}


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _float_grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def replay(rows: list[dict[str, Any]], *, threshold: float, min_gap: float, expert_margin: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        action = row.get("selected_action", {}) if isinstance(row.get("selected_action"), dict) else {}
        passes = (
            action
            and float(row.get("predicted_utility", 0.0) or 0.0) >= float(threshold)
            and float(row.get("runner_up_gap", 0.0) or 0.0) >= float(min_gap)
            and float(row.get("expert_second_gap", 0.0) or 0.0) >= float(expert_margin)
        )
        if passes:
            pred = {
                "gate": "TRADE",
                "family": action.get("family", "UNKNOWN"),
                "side": str(action.get("side", "NONE")).upper(),
                "hold_bars": int(action.get("hold_bars", 0) or 0),
                "confidence": "HIGH",
                "expert": row.get("expert"),
            }
        else:
            pred = dict(NO_TRADE)
        out.append({**row, "prediction": pred})
    return out


def run(*, predictions_jsonl: str, market_csv: str, output: str, work_dir: str, thresholds: str, gaps: str, expert_margins: str, leverage: float, trade_take_profit_pct: float, min_trades: int) -> dict[str, Any]:
    rows = _read(predictions_jsonl)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    i = 0
    for th in _float_grid(thresholds):
        for gap in _float_grid(gaps):
            for egap in _float_grid(expert_margins):
                pred_path = work / f"pred_{i}.jsonl"
                bt_path = work / f"bt_{i}.json"
                replayed = replay(rows, threshold=th, min_gap=gap, expert_margin=egap)
                pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in replayed) + "\n")
                bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred_path), market_csv=market_csv, output=str(bt_path), leverage=float(leverage), trade_take_profit_pct=float(trade_take_profit_pct)))
                sim = bt["sim"]
                ts = bt["trade_stats"]
                trades = int(sim["trade_entries"])
                selection_score = float(sim["cagr_to_strict_mdd"]) if trades >= int(min_trades) and float(sim["cagr_pct"]) > 0 else -999.0 + trades / 1000.0
                results.append({
                    "threshold": th,
                    "min_gap": gap,
                    "expert_margin": egap,
                    "selection_score": selection_score,
                    "sim": sim,
                    "trade_stats": ts,
                    "predictions": str(pred_path),
                    "backtest": str(bt_path),
                })
                i += 1
    ranked = sorted(results, key=lambda r: (float(r["selection_score"]), int(r["sim"]["trade_entries"]), float(r["sim"]["cagr_pct"])), reverse=True)
    report = {
        "predictions_jsonl": predictions_jsonl,
        "market_csv": market_csv,
        "work_dir": work_dir,
        "sweep_space": {"thresholds": _float_grid(thresholds), "gaps": _float_grid(gaps), "expert_margins": _float_grid(expert_margins), "configs": len(results), "min_trades": int(min_trades)},
        "selected": ranked[0] if ranked else None,
        "ranked": ranked[:50],
        "leakage_guard": {"replays_existing_score_stream_without_refit": True, "selection_period_defined_by_input_predictions": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep replay thresholds over regime policy score stream")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--thresholds", default="0.003,0.004,0.005,0.006,0.008,0.010,0.012")
    p.add_argument("--gaps", default="0,0.0005,0.001,0.002,0.004")
    p.add_argument("--expert-margins", default="0,0.0005,0.001,0.002,0.004")
    p.add_argument("--leverage", type=float, default=0.76)
    p.add_argument("--trade-take-profit-pct", type=float, default=4.0)
    p.add_argument("--min-trades", type=int, default=30)
    return p.parse_args()


def main() -> None:
    report = run(**vars(parse_args()))
    sel = report.get("selected") or {}
    print(json.dumps({"selected": {k: sel.get(k) for k in ["threshold", "min_gap", "expert_margin", "selection_score"]}, "sim": sel.get("sim"), "trade_stats": sel.get("trade_stats")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
