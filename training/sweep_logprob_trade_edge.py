"""Sweep trade_edge thresholds on logprob policy predictions and replay selected threshold."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class SweepCfg:
    val_predictions: str
    eval_predictions: str
    market_csv: str
    output: str
    work_dir: str = "results/logprob_trade_edge_sweep"
    quantiles: str = "0.50,0.60,0.70,0.80,0.85,0.90,0.95"
    min_val_trades: int = 20
    leverage: float = 1.0
    entry_delay_bars: int = 1


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _write_threshold(rows: list[dict[str, Any]], path: str, threshold: float) -> dict[str, Any]:
    out=[]; counts={"TRADE":0,"NO_TRADE":0,"LONG":0,"SHORT":0,"FULL":0,"SMALL":0}
    for r in rows:
        row=dict(r)
        if float(row.get("trade_edge", -999.0)) < threshold:
            row["position_scale"] = 0.0
            row["prediction"] = {"gate":"NO_TRADE","side":"NONE","hold_bars":0,"confidence":"LOW","family":"event_rule_rationale_logprob_threshold"}
            counts["NO_TRADE"] += 1
        else:
            pred=row.get("prediction",{})
            side=str(pred.get("side","NONE"))
            scale=float(row.get("position_scale",0.0) or 0.0)
            if pred.get("gate") == "TRADE" and side in {"LONG","SHORT"} and scale>0:
                counts["TRADE"] += 1; counts[side] += 1; counts["FULL" if scale>=1 else "SMALL"] += 1
            else:
                counts["NO_TRADE"] += 1
        out.append(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in out)+"\n")
    return {"rows":len(out),"counts":counts,"threshold":threshold,"output":path}


def run(cfg:SweepCfg)->dict[str,Any]:
    val=_load(cfg.val_predictions); ev=_load(cfg.eval_predictions)
    qs=[float(x) for x in cfg.quantiles.split(',') if x.strip()]
    edges=np.asarray([float(r.get('trade_edge',-999.0)) for r in val],dtype=float)
    Path(cfg.work_dir).mkdir(parents=True,exist_ok=True)
    cands=[]
    for q in qs:
        thr=float(np.quantile(edges,q)) if len(edges) else 999.0
        pred=str(Path(cfg.work_dir)/f'val_q{q}.jsonl')
        ps=_write_threshold(val,pred,thr)
        bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pred,market_csv=cfg.market_csv,output=str(Path(cfg.work_dir)/f'val_q{q}.bt.json'),leverage=cfg.leverage,entry_delay_bars=cfg.entry_delay_bars))
        sim=bt['sim']; score=float(sim.get('cagr_to_strict_mdd',-999) or -999)
        if int(sim.get('trade_entries',0) or 0) < cfg.min_val_trades: score-=1000
        cands.append({'q':q,'threshold':thr,'prediction_summary':ps,'val_sim':sim,'val_trade_stats':bt['trade_stats'],'score':score})
    cands.sort(key=lambda r:r['score'],reverse=True)
    sel=cands[0]
    eval_pred=str(Path(cfg.work_dir)/'selected_eval_predictions.jsonl')
    eps=_write_threshold(ev,eval_pred,float(sel['threshold']))
    ebt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred,market_csv=cfg.market_csv,output=str(Path(cfg.work_dir)/'selected_eval_backtest.json'),leverage=cfg.leverage,entry_delay_bars=cfg.entry_delay_bars))
    report={'config':cfg.__dict__,'top_val':cands,'selected':sel,'eval_prediction_summary':eps,'eval_backtest':{'sim':ebt['sim'],'trade_stats':ebt['trade_stats']},'leakage_guard':{'threshold_selected_on_val_only':True}}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--val-predictions',required=True); p.add_argument('--eval-predictions',required=True); p.add_argument('--market-csv',required=True); p.add_argument('--output',required=True)
    p.add_argument('--work-dir',default=SweepCfg.work_dir); p.add_argument('--quantiles',default=SweepCfg.quantiles); p.add_argument('--min-val-trades',type=int,default=SweepCfg.min_val_trades); p.add_argument('--leverage',type=float,default=1.0); p.add_argument('--entry-delay-bars',type=int,default=1)
    return p.parse_args()


def main(): print(json.dumps(run(SweepCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
