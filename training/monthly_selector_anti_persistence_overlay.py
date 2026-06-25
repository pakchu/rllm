"""Apply anti-persistence abstention to a monthly selector report.

If validation performance is too strong, abstain in the next eval month. This is a
hypothesis test for validation-spike inversion; thresholds must be selected on a
separate validation regime before deployment.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "ANTI_PERSISTENCE", "confidence": "HIGH"}


@dataclass(frozen=True)
class AntiPersistenceCfg:
    selector_report: str
    predictions_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/monthly_selector_anti_persistence"
    max_val_ratio: float = 10.0
    max_val_cagr_pct: float = 100.0
    max_val_t: float = 2.0
    leverage: float = 1.0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def _blocked_months(report: dict[str, Any], cfg: AntiPersistenceCfg) -> dict[str, dict[str, Any]]:
    out = {}
    for m in report.get("months", []):
        sel = m.get("selected") or {}
        val_obj = sel.get("backtest") if isinstance(sel.get("backtest"), dict) else sel.get("val", {})
        sim = (val_obj or {}).get("sim") or {}
        stats = (val_obj or {}).get("trade_stats") or {}
        ratio = float(sim.get("cagr_to_strict_mdd", 0.0) or 0.0)
        cagr = float(sim.get("cagr_pct", 0.0) or 0.0)
        t = float(stats.get("t_stat_like", 0.0) or 0.0)
        reasons = []
        if ratio > float(cfg.max_val_ratio): reasons.append("val_ratio_too_high")
        if cagr > float(cfg.max_val_cagr_pct): reasons.append("val_cagr_too_high")
        if t > float(cfg.max_val_t): reasons.append("val_t_too_high")
        if reasons:
            out[str(m.get("month"))] = {"reasons": reasons, "val_ratio": ratio, "val_cagr_pct": cagr, "val_t": t}
    return out


def run(cfg: AntiPersistenceCfg) -> dict[str, Any]:
    report = json.loads(Path(cfg.selector_report).read_text())
    rows = _read_jsonl(cfg.predictions_jsonl)
    blocked = _blocked_months(report, cfg)
    out = []
    blocked_rows = 0
    for r in rows:
        month = str(r.get("date", ""))[:7]
        if month in blocked:
            pred = r.get("prediction", {}) if isinstance(r.get("prediction"), dict) else {}
            if pred.get("gate") == "TRADE":
                blocked_rows += 1
            out.append({**r, "blocked_prediction": r.get("prediction"), "prediction": {**NO_TRADE, "reason": "validation_spike_anti_persistence"}, "anti_persistence": blocked[month]})
        else:
            out.append(r)
    work = Path(cfg.work_dir); work.mkdir(parents=True, exist_ok=True)
    pred_path = work / "anti_persistence_predictions.jsonl"
    _write_jsonl(pred_path, out)
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred_path), market_csv=cfg.market_csv, output=str(work / "anti_persistence_backtest.json"), leverage=float(cfg.leverage)))
    result = {"config": asdict(cfg), "blocked_months": blocked, "blocked_trade_rows": blocked_rows, "prediction_file": str(pred_path), "backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"], "period": bt["period"]}, "leakage_guard": {"overlay_uses_selector_validation_metrics_only": True, "eval_trade_returns_not_used_by_overlay": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument('--selector-report',required=True); p.add_argument('--predictions-jsonl',required=True); p.add_argument('--market-csv',required=True); p.add_argument('--output',required=True); p.add_argument('--work-dir',default=AntiPersistenceCfg.work_dir)
    p.add_argument('--max-val-ratio',type=float,default=10.0); p.add_argument('--max-val-cagr-pct',type=float,default=100.0); p.add_argument('--max-val-t',type=float,default=2.0); p.add_argument('--leverage',type=float,default=1.0)
    return p.parse_args()


def main():
    r=run(AntiPersistenceCfg(**vars(parse_args())))
    print(json.dumps({"blocked_months":r['blocked_months'],"sim":r['backtest']['sim'],"trade_stats":r['backtest']['trade_stats']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
