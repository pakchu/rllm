"""Apply a causal month-level validation health gate to prediction rows.

The gate uses only the rolling summary generated before each target month:
if the selected prior-validation score for that target month is below a fixed
threshold, all predictions in that target month are converted to NO_TRADE.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "MONTH_VALIDATION_GATE", "confidence": "HIGH"}


@dataclass(frozen=True)
class MonthValidationGateCfg:
    predictions_jsonl: str
    rolling_summary_json: str
    output_jsonl: str
    threshold: float = 0.5
    market_csv: str = ""
    backtest_output: str = ""
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0
    rolling_window_trades: int = 0
    rolling_loss_stop_pct: float = 0.0
    pause_bars: int = 864


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _month_scores(summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in summary.get("months", []):
        month = str(m.get("month"))
        try:
            out[month] = float((m.get("selected") or {}).get("score", float("-inf")))
        except Exception:
            out[month] = float("-inf")
    return out


def apply_gate(cfg: MonthValidationGateCfg) -> dict[str, Any]:
    rows = _load_jsonl(cfg.predictions_jsonl)
    summary = json.loads(Path(cfg.rolling_summary_json).read_text())
    scores = _month_scores(summary)
    out: list[dict[str, Any]] = []
    blocked = 0
    passed = 0
    trade_before = 0
    trade_after = 0
    month_counts: dict[str, dict[str, int | float]] = {}
    for row in rows:
        month = str(row.get("date", ""))[:7]
        score = float(scores.get(month, float("-inf")))
        pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
        was_trade = pred.get("gate") == "TRADE"
        trade_before += int(was_trade)
        nr = dict(row)
        m = month_counts.setdefault(month, {"rows": 0, "blocked_rows": 0, "trade_before": 0, "trade_after": 0, "score": score})
        m["rows"] = int(m["rows"]) + 1
        m["trade_before"] = int(m["trade_before"]) + int(was_trade)
        if score < float(cfg.threshold):
            nr["prediction"] = dict(NO_TRADE)
            nr["month_validation_gate_blocked"] = True
            nr["month_validation_score"] = score
            blocked += 1
            m["blocked_rows"] = int(m["blocked_rows"]) + 1
        else:
            nr["month_validation_gate_blocked"] = False
            nr["month_validation_score"] = score
            passed += 1
        if isinstance(nr.get("prediction"), dict) and nr["prediction"].get("gate") == "TRADE":
            trade_after += 1
            m["trade_after"] = int(m["trade_after"]) + 1
        out.append(nr)
    _write_jsonl(cfg.output_jsonl, out)
    report: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(out),
        "blocked_rows": blocked,
        "passed_rows": passed,
        "trade_signals_before": trade_before,
        "trade_signals_after": trade_after,
        "months": month_counts,
        "leakage_guard": {
            "uses_target_month_outcomes": False,
            "uses_prior_validation_score_from_rolling_summary": True,
            "threshold_is_fixed_input_not_fit_here": True,
        },
    }
    if cfg.market_csv and cfg.backtest_output:
        bt = run_overlay(OnlineRiskOverlayConfig(
            predictions_jsonl=cfg.output_jsonl,
            market_csv=cfg.market_csv,
            output=cfg.backtest_output,
            leverage=float(cfg.leverage),
            fee_rate=float(cfg.fee_rate),
            slippage_rate=float(cfg.slippage_rate),
            entry_delay_bars=int(cfg.entry_delay_bars),
            cooldown_bars=int(cfg.cooldown_bars),
            trade_stop_loss_pct=float(cfg.trade_stop_loss_pct),
            trade_take_profit_pct=float(cfg.trade_take_profit_pct),
            rolling_window_trades=int(cfg.rolling_window_trades),
            rolling_loss_stop_pct=float(cfg.rolling_loss_stop_pct),
            pause_bars=int(cfg.pause_bars),
        ))
        report["backtest"] = {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply month validation health gate to predictions")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--rolling-summary-json", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--threshold", type=float, default=MonthValidationGateCfg.threshold)
    p.add_argument("--market-csv", default="")
    p.add_argument("--backtest-output", default="")
    p.add_argument("--leverage", type=float, default=MonthValidationGateCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=MonthValidationGateCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=MonthValidationGateCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=MonthValidationGateCfg.entry_delay_bars)
    p.add_argument("--cooldown-bars", type=int, default=MonthValidationGateCfg.cooldown_bars)
    p.add_argument("--trade-stop-loss-pct", type=float, default=MonthValidationGateCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=MonthValidationGateCfg.trade_take_profit_pct)
    p.add_argument("--rolling-window-trades", type=int, default=MonthValidationGateCfg.rolling_window_trades)
    p.add_argument("--rolling-loss-stop-pct", type=float, default=MonthValidationGateCfg.rolling_loss_stop_pct)
    p.add_argument("--pause-bars", type=int, default=MonthValidationGateCfg.pause_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(apply_gate(MonthValidationGateCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
