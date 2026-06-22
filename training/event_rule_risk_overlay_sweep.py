"""Apply stop/take-profit risk overlay sweep to a selected event micro rule.

The rule itself comes from a prior validation-selected report. This script tests
whether execution risk controls can make that rule robust on eval. Overlay params
are selected on validation only; eval is final holdout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.event_micro_rule_sweep import load, group, write_predictions
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    rule_report: str
    output: str
    work_dir: str = "results/event_rule_risk_overlay_sweep"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    stop_losses: str = "0,2,4,6,8,10,12"
    take_profits: str = "0,2,4,6,8,10,12"
    max_hold_bars: str = "72,144,288"
    min_val_trades: int = 50


def date(g: list[dict[str, Any]]) -> str:
    return str(g[0].get("date", ""))


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(',') if x.strip()]


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(',') if x.strip()]


def score(sim: dict[str, Any], min_trades: int) -> float:
    sc=float(sim["cagr_to_strict_mdd"])
    if int(sim["trade_entries"]) < min_trades:
        sc -= 1000.0
    return sc


def run(c: Cfg) -> dict[str, Any]:
    d=json.load(open(c.rule_report))
    rule=d["selected"]["params"] if "params" in d.get("selected",{}) else d["selected"]["rule"]["params"]
    allg=group(load(c.train_candidates)); evg=group(load(c.eval_candidates))
    val=[g for g in allg if c.validation_start <= date(g) <= c.validation_end]
    Path(c.work_dir).mkdir(parents=True, exist_ok=True)
    val_pred=str(Path(c.work_dir)/"val_base_predictions.jsonl")
    eval_pred=str(Path(c.work_dir)/"eval_base_predictions.jsonl")
    val_ps=write_predictions(val, val_pred, **rule)
    eval_ps=write_predictions(evg, eval_pred, **rule)
    rows=[]
    for sl in parse_floats(c.stop_losses):
        for tp in parse_floats(c.take_profits):
            for mh in parse_ints(c.max_hold_bars):
                tag=f"sl{sl:g}_tp{tp:g}_mh{mh}"
                bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=val_pred,market_csv=c.market_csv,output=str(Path(c.work_dir)/f"val_{tag}.bt.json"),leverage=1.0,entry_delay_bars=1,max_hold_bars=mh,trade_stop_loss_pct=sl,trade_take_profit_pct=tp))
                rows.append({"overlay":{"stop_loss":sl,"take_profit":tp,"max_hold_bars":mh},"score":score(bt["sim"],c.min_val_trades),"val_sim":bt["sim"],"val_trade_stats":bt["trade_stats"]})
    rows.sort(key=lambda r:r["score"], reverse=True)
    sel=rows[0]
    ov=sel["overlay"]
    ebt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred,market_csv=c.market_csv,output=str(Path(c.work_dir)/"selected_eval.bt.json"),leverage=1.0,entry_delay_bars=1,max_hold_bars=int(ov["max_hold_bars"]),trade_stop_loss_pct=float(ov["stop_loss"]),trade_take_profit_pct=float(ov["take_profit"])))
    report={"config":c.__dict__,"rule":rule,"prediction_summary":{"val":val_ps,"eval":eval_ps},"top_val":rows[:30],"selected":sel,"eval":{"sim":ebt["sim"],"trade_stats":ebt["trade_stats"]},"leakage_guard":"base rule and overlay parameters selected on validation; eval final holdout"}
    Path(c.output).parent.mkdir(parents=True, exist_ok=True)
    Path(c.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument('--train-candidates', required=True); p.add_argument('--eval-candidates', required=True); p.add_argument('--rule-report', required=True); p.add_argument('--output', required=True)
    p.add_argument('--work-dir', default=Cfg.work_dir); p.add_argument('--validation-start', default=Cfg.validation_start); p.add_argument('--validation-end', default=Cfg.validation_end); p.add_argument('--market-csv', default=Cfg.market_csv)
    p.add_argument('--stop-losses', default=Cfg.stop_losses); p.add_argument('--take-profits', default=Cfg.take_profits); p.add_argument('--max-hold-bars', default=Cfg.max_hold_bars); p.add_argument('--min-val-trades', type=int, default=Cfg.min_val_trades)
    return Cfg(**vars(p.parse_args()))

def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
if __name__ == '__main__': main()
