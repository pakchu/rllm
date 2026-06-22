"""Rolling selector over pre-declared micro-path rules.

Candidate rules come from a prior validation-only sweep. For each target month,
select the best rule using only rows before that month within a lookback window,
then apply it to that month. This approximates a live continuous rule selector.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from training.event_micro_rule_sweep import Cfg as SweepCfg, eval_rule, group, load, write_predictions
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    rule_report: str
    output: str
    predictions_output: str
    work_dir: str = "results/event_micro_rule_rolling_selector"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    start_date: str = "2025-01-01"
    end_date: str = "2026-06-01"
    lookback_days: int = 730
    top_k_rules: int = 20
    min_select_trades: int = 40


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s)[:19])


def month_starts(start: datetime, end: datetime) -> list[datetime]:
    cur = datetime(start.year, start.month, 1)
    out=[]
    while cur < end:
        out.append(cur)
        if cur.month == 12:
            cur = datetime(cur.year+1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month+1, 1)
    return out


def groups_between(groups: list[list[dict[str, Any]]], start: datetime, end: datetime) -> list[list[dict[str, Any]]]:
    return [g for g in groups if start <= parse_dt(g[0]["date"]) < end]


def score_selection(res: dict[str, Any], min_trades: int) -> float:
    score = float(res["sim"]["cagr_to_strict_mdd"])
    if int(res["sim"]["trade_entries"]) < min_trades:
        score -= 1000.0
    p = float(res["trade_stats"].get("p_value_mean_ret_approx", 1.0) or 1.0)
    # Small preference for statistically cleaner training-window rules without making p a hard gate.
    return score + max(0.0, 0.10 - p)


def run(cfg: Cfg) -> dict[str, Any]:
    rule_data = json.load(open(cfg.rule_report))
    params = [r["params"] for r in rule_data["top_val"][:cfg.top_k_rules]]
    all_groups = group(load(cfg.train_candidates) + load(cfg.eval_candidates))
    start = parse_dt(cfg.start_date)
    end = parse_dt(cfg.end_date)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    pred_rows=[]
    decisions=[]
    sweep_cfg = SweepCfg(train_candidates=cfg.train_candidates, eval_candidates=cfg.eval_candidates, output=cfg.output, work_dir=cfg.work_dir, market_csv=cfg.market_csv)
    for mi, m0 in enumerate(month_starts(start, end)):
        m1 = month_starts(m0 + timedelta(days=32), m0 + timedelta(days=70))[0] if False else None
        if m0.month == 12:
            m1 = datetime(m0.year+1, 1, 1)
        else:
            m1 = datetime(m0.year, m0.month+1, 1)
        m1 = min(m1, end)
        hist_start = m0 - timedelta(days=cfg.lookback_days)
        hist_groups = groups_between(all_groups, hist_start, m0)
        month_groups = groups_between(all_groups, m0, m1)
        if not month_groups:
            continue
        scored=[]
        for ri, p in enumerate(params):
            res = eval_rule(hist_groups, sweep_cfg, f"select_m{mi:02d}_r{ri:02d}", p)
            scored.append({"rule_index": ri, "score": score_selection(res, cfg.min_select_trades), "params": p, "selection_sim": res["sim"], "selection_trade_stats": res["trade_stats"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        chosen = scored[0]
        month_pred_path = str(Path(cfg.work_dir)/f"month_{m0:%Y%m}_predictions.jsonl")
        ps = write_predictions(month_groups, month_pred_path, **chosen["params"])
        for line in open(month_pred_path):
            if line.strip():
                pred_rows.append(json.loads(line))
        decisions.append({"month": f"{m0:%Y-%m}", "rows": len(month_groups), "selected": chosen, "month_prediction_summary": ps})
    Path(cfg.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.predictions_output).write_text("\n".join(json.dumps(r, sort_keys=True) for r in pred_rows)+("\n" if pred_rows else ""))
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=cfg.predictions_output, market_csv=cfg.market_csv, output=str(Path(cfg.work_dir)/"rolling_eval_backtest.json"), leverage=1.0, entry_delay_bars=1))
    report = {"config": cfg.__dict__, "candidate_rule_count": len(params), "months": decisions, "prediction_rows": len(pred_rows), "eval_backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}, "leakage_guard": "Each month selects from predeclared validation rules using only rows before month start."}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument("--train-candidates", required=True)
    p.add_argument("--eval-candidates", required=True)
    p.add_argument("--rule-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--work-dir", default=Cfg.work_dir)
    p.add_argument("--market-csv", default=Cfg.market_csv)
    p.add_argument("--start-date", default=Cfg.start_date)
    p.add_argument("--end-date", default=Cfg.end_date)
    p.add_argument("--lookback-days", type=int, default=Cfg.lookback_days)
    p.add_argument("--top-k-rules", type=int, default=Cfg.top_k_rules)
    p.add_argument("--min-select-trades", type=int, default=Cfg.min_select_trades)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
