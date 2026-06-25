"""Audit validation-to-eval inversion in monthly selector reports."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MonthlySelectorInversionAuditCfg:
    reports: str
    output: str
    min_eval_trades: int = 1


def _safe_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def _month_row(report_path: str, month: dict[str, Any]) -> dict[str, Any] | None:
    selected = month.get("selected") or {}
    eval_obj = month.get("eval") or {}
    eval_bt = eval_obj.get("backtest") if isinstance(eval_obj.get("backtest"), dict) else eval_obj
    val_obj = selected.get("backtest") if isinstance(selected.get("backtest"), dict) else selected.get("val", {})
    val_sim = (val_obj or {}).get("sim") or {}
    val_stats = (val_obj or {}).get("trade_stats") or {}
    eval_sim = (eval_bt or {}).get("sim") or {}
    eval_stats = (eval_bt or {}).get("trade_stats") or {}
    if not eval_sim:
        return None
    val_month_stats = selected.get("validation_month_stats") or {}
    return {
        "report": report_path,
        "month": month.get("month"),
        "status": month.get("status"),
        "target": selected.get("target"),
        "threshold": selected.get("threshold"),
        "validation_passed": selected.get("validation_passed"),
        "val_cagr_pct": float(val_sim.get("cagr_pct", 0.0) or 0.0),
        "val_mdd_pct": float(val_sim.get("strict_mdd_pct", 0.0) or 0.0),
        "val_ratio": float(val_sim.get("cagr_to_strict_mdd", 0.0) or 0.0),
        "val_trades": int(val_sim.get("trade_entries", 0) or 0),
        "val_p": float(val_stats.get("p_value_mean_ret_approx", 1.0) or 1.0),
        "val_t": float(val_stats.get("t_stat_like", 0.0) or 0.0),
        "val_positive_months": int(val_month_stats.get("positive_months", 0) or 0),
        "val_worst_month_ret_pct": float(val_month_stats.get("worst_month_ret_pct", 0.0) or 0.0),
        "eval_cagr_pct": float(eval_sim.get("cagr_pct", 0.0) or 0.0),
        "eval_mdd_pct": float(eval_sim.get("strict_mdd_pct", 0.0) or 0.0),
        "eval_ratio": float(eval_sim.get("cagr_to_strict_mdd", 0.0) or 0.0),
        "eval_trades": int(eval_sim.get("trade_entries", 0) or 0),
        "eval_p": float(eval_stats.get("p_value_mean_ret_approx", 1.0) or 1.0),
    }


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run(cfg: MonthlySelectorInversionAuditCfg) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in [x.strip() for x in cfg.reports.split(",") if x.strip()]:
        report = json.loads(Path(path).read_text())
        for month in report.get("months", []):
            row = _month_row(path, month)
            if row is not None and int(row["eval_trades"]) >= int(cfg.min_eval_trades):
                rows.append(row)
    inversions = [r for r in rows if float(r["val_cagr_pct"]) > 0.0 and float(r["eval_cagr_pct"]) < 0.0]
    both_positive = [r for r in rows if float(r["val_cagr_pct"]) > 0.0 and float(r["eval_cagr_pct"]) > 0.0]
    val_cagr = [float(r["val_cagr_pct"]) for r in rows]
    eval_cagr = [float(r["eval_cagr_pct"]) for r in rows]
    val_ratio = [float(r["val_ratio"]) for r in rows]
    eval_ratio = [float(r["eval_ratio"]) for r in rows]
    report = {
        "config": asdict(cfg),
        "rows": rows,
        "summary": {
            "n_eval_months": len(rows),
            "n_val_positive_eval_negative": len(inversions),
            "n_val_positive_eval_positive": len(both_positive),
            "val_to_eval_cagr_corr": _corr(val_cagr, eval_cagr),
            "val_to_eval_ratio_corr": _corr(val_ratio, eval_ratio),
            "mean_eval_cagr_when_val_positive": float(np.mean([r["eval_cagr_pct"] for r in rows if r["val_cagr_pct"] > 0.0])) if any(r["val_cagr_pct"] > 0.0 for r in rows) else 0.0,
            "mean_eval_cagr_when_val_nonpositive": float(np.mean([r["eval_cagr_pct"] for r in rows if r["val_cagr_pct"] <= 0.0])) if any(r["val_cagr_pct"] <= 0.0 for r in rows) else 0.0,
        },
        "worst_inversions": sorted(inversions, key=lambda r: float(r["eval_cagr_pct"]))[:20],
        "leakage_guard": {"audit_only": True, "eval_results_used_for_diagnosis_not_selection": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reports", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-eval-trades", type=int, default=1)
    return p.parse_args()


def main() -> None:
    r = run(MonthlySelectorInversionAuditCfg(**vars(parse_args())))
    print(json.dumps({"summary": r["summary"], "worst_inversions": r["worst_inversions"][:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
