"""Nested gate/overlay selection for rolling prediction rows.

Select month-validation gate and execution overlay parameters on one period, then
apply the chosen configs to a later untouched evaluation period.  This prevents
claiming diagnostic overlay sweeps as live/eval performance.
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
class NestedGateOverlaySelectionCfg:
    predictions_jsonl: str
    rolling_summary_json: str
    market_csv: str
    output: str
    work_dir: str
    selection_start: str = "2024-01"
    selection_end: str = "2025-12"
    eval_start: str = "2026-01"
    eval_end: str = "2026-05"
    thresholds: str = "-1000,-500,-100,0,0.5,1,2,3,5"
    stop_losses: str = "0,2,3,4"
    take_profits: str = "0,2,3,4,6"
    rolling_loss_specs: str = "0:0,10:5,20:8"
    min_selection_trades: int = 30
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    pause_bars: int = 864
    top_k: int = 20


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _loss_specs(raw: str) -> list[tuple[int, float]]:
    out = []
    for part in str(raw).split(","):
        if not part.strip():
            continue
        n, loss = part.split(":", 1)
        out.append((int(n), float(loss)))
    return out


def _month_scores(summary: dict[str, Any]) -> dict[str, float]:
    out = {}
    for m in summary.get("months", []):
        try:
            out[str(m.get("month"))] = float((m.get("selected") or {}).get("score", float("-inf")))
        except Exception:
            out[str(m.get("month"))] = float("-inf")
    return out


def _period_rows(rows: list[dict[str, Any]], start_month: str, end_month: str) -> list[dict[str, Any]]:
    return [r for r in rows if start_month <= str(r.get("date", ""))[:7] <= end_month]


def _apply_month_gate(rows: list[dict[str, Any]], scores: dict[str, float], threshold: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        month = str(row.get("date", ""))[:7]
        score = float(scores.get(month, float("-inf")))
        nr = dict(row)
        if score < float(threshold):
            nr["prediction"] = dict(NO_TRADE)
            nr["month_validation_gate_blocked"] = True
        else:
            nr["month_validation_gate_blocked"] = False
        nr["month_validation_score"] = score
        out.append(nr)
    return out


def _score_selection(bt: dict[str, Any], min_trades: int) -> float:
    sim = bt.get("sim", {})
    stats = bt.get("trade_stats", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -999.0 + trades / 1000.0
    cagr = float(sim.get("cagr_pct", -100.0))
    mdd = float(sim.get("strict_mdd_pct", 999.0))
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0))
    p = float(stats.get("p_value_mean_ret_approx", 1.0))
    if cagr <= 0:
        return -500.0 + trades / 1000.0 + cagr / 100.0
    return ratio + 0.01 * cagr - 0.05 * max(0.0, mdd - 15.0) - p + min(1.0, trades / 100.0)


def _run_bt(rows: list[dict[str, Any]], cfg: NestedGateOverlaySelectionCfg, path: Path, *, stop: float, take: float, rolling_n: int, rolling_loss: float) -> dict[str, Any]:
    pred = path.with_suffix(".jsonl")
    out = path.with_suffix(".bt.json")
    _write_jsonl(pred, rows)
    return run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=str(pred),
        market_csv=cfg.market_csv,
        output=str(out),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        entry_delay_bars=int(cfg.entry_delay_bars),
        trade_stop_loss_pct=float(stop),
        trade_take_profit_pct=float(take),
        rolling_window_trades=int(rolling_n),
        rolling_loss_stop_pct=float(rolling_loss),
        pause_bars=int(cfg.pause_bars),
    ))


def run(cfg: NestedGateOverlaySelectionCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.predictions_jsonl)
    summary = json.loads(Path(cfg.rolling_summary_json).read_text())
    scores = _month_scores(summary)
    select_base = _period_rows(rows, cfg.selection_start, cfg.selection_end)
    eval_base = _period_rows(rows, cfg.eval_start, cfg.eval_end)
    work = Path(cfg.work_dir); work.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    i = 0
    for threshold in _floats(cfg.thresholds):
        gated_select = _apply_month_gate(select_base, scores, threshold)
        for stop in _floats(cfg.stop_losses):
            for take in _floats(cfg.take_profits):
                for rolling_n, rolling_loss in _loss_specs(cfg.rolling_loss_specs):
                    bt = _run_bt(gated_select, cfg, work / f"select_{i}", stop=stop, take=take, rolling_n=rolling_n, rolling_loss=rolling_loss)
                    conf = {"threshold": threshold, "stop": stop, "take": take, "rolling_window_trades": rolling_n, "rolling_loss_stop_pct": rolling_loss}
                    candidates.append({"config": conf, "selection_score": _score_selection(bt, cfg.min_selection_trades), "selection": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}})
                    i += 1
    ranked = sorted(candidates, key=lambda r: (float(r["selection_score"]), float(r["selection"]["sim"].get("cagr_to_strict_mdd", -999)), float(r["selection"]["sim"].get("cagr_pct", -999))), reverse=True)
    for j, row in enumerate(ranked[: int(cfg.top_k)]):
        c = row["config"]
        gated_eval = _apply_month_gate(eval_base, scores, float(c["threshold"]))
        bt = _run_bt(gated_eval, cfg, work / f"eval_top{j}", stop=float(c["stop"]), take=float(c["take"]), rolling_n=int(c["rolling_window_trades"]), rolling_loss=float(c["rolling_loss_stop_pct"]))
        row["eval"] = {"sim": bt["sim"], "trade_stats": bt["trade_stats"], "predictions_jsonl": str((work / f"eval_top{j}").with_suffix(".jsonl")), "backtest_json": str((work / f"eval_top{j}").with_suffix(".bt.json"))}
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "selection_rows": len(select_base),
        "eval_rows": len(eval_base),
        "top": ranked[: int(cfg.top_k)],
        "all_count": len(ranked),
        "leakage_guard": {"configs_ranked_on_selection_period_only": True, "eval_period_not_used_for_selection": True, "month_gate_scores_are_prior_validation_scores": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested gate/overlay selection with untouched eval")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--rolling-summary-json", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--selection-start", default=NestedGateOverlaySelectionCfg.selection_start)
    p.add_argument("--selection-end", default=NestedGateOverlaySelectionCfg.selection_end)
    p.add_argument("--eval-start", default=NestedGateOverlaySelectionCfg.eval_start)
    p.add_argument("--eval-end", default=NestedGateOverlaySelectionCfg.eval_end)
    p.add_argument("--thresholds", default=NestedGateOverlaySelectionCfg.thresholds)
    p.add_argument("--stop-losses", default=NestedGateOverlaySelectionCfg.stop_losses)
    p.add_argument("--take-profits", default=NestedGateOverlaySelectionCfg.take_profits)
    p.add_argument("--rolling-loss-specs", default=NestedGateOverlaySelectionCfg.rolling_loss_specs)
    p.add_argument("--min-selection-trades", type=int, default=NestedGateOverlaySelectionCfg.min_selection_trades)
    p.add_argument("--leverage", type=float, default=NestedGateOverlaySelectionCfg.leverage)
    p.add_argument("--top-k", type=int, default=NestedGateOverlaySelectionCfg.top_k)
    return p.parse_args()


def main() -> None:
    report = run(NestedGateOverlaySelectionCfg(**vars(parse_args())))
    compact = []
    for row in report["top"][:10]:
        compact.append({"config": row["config"], "selection": row["selection"]["sim"], "eval": row.get("eval", {}).get("sim", {})})
    print(json.dumps({"output": report["config"]["output"], "selection_rows": report["selection_rows"], "eval_rows": report["eval_rows"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
