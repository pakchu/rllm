"""Audit monthly side-map reliability labels from existing predictions.

For each month, replay the generated predictions as-is (`pass`), with sides
flipped (`invert`), and as no-trade (`block`).  The output is a label table for
researching whether side-map reliability is learnable from prior state.  It is
not a live selector by itself because month labels use same-month outcomes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.nested_score_geometry_transform_selection import _invert_prediction
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "MONTH_SIDE_MAP_AUDIT", "confidence": "HIGH"}


@dataclass(frozen=True)
class MonthlySideMapReliabilityAuditCfg:
    predictions_jsonl: str
    market_csv: str
    output: str
    work_dir: str
    start_month: str = "2024-01"
    end_month: str = "2026-05"
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 3.0
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    keep_artifacts: bool = False


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _month(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))[:7]


def _apply(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        nr = dict(row)
        pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
        if mode == "pass":
            nr["prediction"] = pred
        elif mode == "invert":
            nr["prediction"] = _invert_prediction(pred)
        elif mode == "block":
            nr["prediction"] = dict(NO_TRADE)
        else:
            raise ValueError(f"unknown mode {mode}")
        nr["side_map_audit_mode"] = mode
        out.append(nr)
    return out


def _score(bt: dict[str, Any]) -> float:
    sim = bt.get("sim", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", 0.0) or 0.0)
    mdd = float(sim.get("strict_mdd_pct", 0.0) or 0.0)
    ratio = float(sim.get("cagr_to_strict_mdd", 0.0) or 0.0)
    if trades == 0:
        return 0.0
    if cagr <= 0:
        return -100.0 + cagr / 10.0 - mdd / 10.0 + trades / 1000.0
    return ratio + cagr / 100.0 - max(0.0, mdd - 15.0) / 10.0 + min(1.0, trades / 20.0)


def _run_bt(rows: list[dict[str, Any]], cfg: MonthlySideMapReliabilityAuditCfg, path: Path) -> dict[str, Any]:
    pred = path.with_suffix(".jsonl")
    out = path.with_suffix(".bt.json")
    _write_jsonl(pred, rows)
    result = run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=str(pred),
        market_csv=cfg.market_csv,
        output=str(out),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        entry_delay_bars=int(cfg.entry_delay_bars),
        trade_stop_loss_pct=float(cfg.trade_stop_loss_pct),
        trade_take_profit_pct=float(cfg.trade_take_profit_pct),
    ))
    if not bool(cfg.keep_artifacts):
        pred.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
    return result


def run(cfg: MonthlySideMapReliabilityAuditCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.predictions_jsonl)
    months = sorted({_month(r) for r in rows if cfg.start_month <= _month(r) <= cfg.end_month})
    work = Path(cfg.work_dir); work.mkdir(parents=True, exist_ok=True)
    out_rows = []
    label_counts: dict[str, int] = {}
    for month in months:
        mrows = [r for r in rows if _month(r) == month]
        variants = {}
        for mode in ("pass", "invert", "block"):
            bt = _run_bt(_apply(mrows, mode), cfg, work / f"{month}_{mode}")
            variants[mode] = {"score": _score(bt), "sim": bt["sim"], "trade_stats": bt["trade_stats"]}
        best = max(variants, key=lambda k: (float(variants[k]["score"]), float(variants[k]["sim"].get("cagr_pct", 0.0))))
        label = {"pass": "normal", "invert": "inverse", "block": "unreliable"}[best]
        label_counts[label] = label_counts.get(label, 0) + 1
        out_rows.append({"month": month, "label": label, "best_mode": best, "variants": variants})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "months": out_rows,
        "label_counts": label_counts,
        "leakage_guard": {
            "audit_uses_same_month_outcomes_for_labels": True,
            "not_a_live_selector": True,
            "intended_for_training_label_design_only": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit monthly normal/inverse/unreliable side-map labels")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--start-month", default=MonthlySideMapReliabilityAuditCfg.start_month)
    p.add_argument("--end-month", default=MonthlySideMapReliabilityAuditCfg.end_month)
    p.add_argument("--trade-stop-loss-pct", type=float, default=MonthlySideMapReliabilityAuditCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=MonthlySideMapReliabilityAuditCfg.trade_take_profit_pct)
    p.add_argument("--leverage", type=float, default=MonthlySideMapReliabilityAuditCfg.leverage)
    p.add_argument("--keep-artifacts", action="store_true")
    return p.parse_args()


def main() -> None:
    report = run(MonthlySideMapReliabilityAuditCfg(**vars(parse_args())))
    compact = [{"month": r["month"], "label": r["label"], "pass": r["variants"]["pass"]["sim"]["cagr_pct"], "invert": r["variants"]["invert"]["sim"]["cagr_pct"], "block": r["variants"]["block"]["sim"]["cagr_pct"]} for r in report["months"]]
    print(json.dumps({"output": report["config"]["output"], "label_counts": report["label_counts"], "months": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
