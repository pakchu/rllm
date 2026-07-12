"""Train-only prior calibration for REX choice-label scores."""
from __future__ import annotations
import argparse,json
from collections import Counter
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any
from training.build_rex_event_choice_label_data import ACTIONS
from training.event_candidate_pool_probe import EventPoolConfig,_load_market,_simulate_rows

CANDS=["CHOICE_A_LONG","CHOICE_B_SHORT","CHOICE_C_SKIP"]

@dataclass(frozen=True)
class Cfg:
    scores_json: str
    output_json: str
    market_csv: str
    calibration_split: str = "train"
    hold_bars: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def backtest(rows, preds, cfg):
    m=_load_market(cfg.market_csv)
    trs=[]
    for r,p in zip(rows,preds):
        a=ACTIONS.get(p,"NO_TRADE")
        if a in {"LONG","SHORT"}:
            trs.append({"date":r["date"],"signal_date":r["date"],"side":a,"family":"rex_choice_calibrated","strength":1.0,"score_mean":1.0})
    ecfg=EventPoolConfig(input_csv=cfg.market_csv, output="", hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate)
    res=_simulate_rows(trs,m,ecfg)
    return {"predicted_trade_rows":len(trs),"sim":res.get("sim",{}),"trade_stats":res.get("trade_stats",{})}


def eval_rows(rows,preds):
    corr=0; conf=Counter(); pc=Counter(); tc=Counter()
    for r,p in zip(rows,preds):
        t=str(r["target"]); corr+=int(t==p); conf[f"target={t}|pred={p}"]+=1; pc[p]+=1; tc[t]+=1
    return {"rows":len(rows),"accuracy":corr/max(1,len(rows)),"target_counts":dict(tc),"prediction_counts":dict(pc),"confusion":dict(conf)}


def run(cfg):
    obj=json.load(open(cfg.scores_json)); rows=obj["score_rows"]
    fit=[r for r in rows if r.get("target") and r.get("date") and _split(r)==cfg.calibration_split]
    means={c:sum(float(r["scores"][c]) for r in fit)/max(1,len(fit)) for c in CANDS}
    # Also fit target-conditioned means for audit only; not used for calibration.
    target_means={}
    for c in CANDS:
        part=[r for r in fit if r.get("target")==c]
        target_means[c]={k:sum(float(r["scores"][k]) for r in part)/max(1,len(part)) for k in CANDS}
    report={"config":asdict(cfg),"calibration":{"split":cfg.calibration_split,"rows":len(fit),"mean_score_by_candidate":means,"target_conditioned_score_means_audit":target_means},"splits":{},"leakage_guard":{"calibration_uses_train_scores_only":True,"test_eval_targets_metrics_only":True}}
    for sp in ["train","test","eval"]:
        part=[r for r in rows if _split(r)==sp]
        raw=[]; cal=[]
        for r in part:
            raw.append(max(CANDS,key=lambda c:float(r["scores"][c])))
            cal.append(max(CANDS,key=lambda c:float(r["scores"][c])-float(means[c])))
        report["splits"][sp]={"raw":eval_rows(part,raw),"calibrated":eval_rows(part,cal),"raw_backtest":backtest(part,raw,cfg),"calibrated_backtest":backtest(part,cal,cfg)}
    Path(cfg.output_json).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def _split(r: dict[str,Any]) -> str:
    # score_rows omit split; infer from dates fixed for this dataset.
    import pandas as pd
    ts=pd.Timestamp(str(r["date"]))
    if ts < pd.Timestamp("2025-01-01"): return "train"
    if ts < pd.Timestamp("2026-01-01"): return "test"
    return "eval"

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--scores-json",required=True); p.add_argument("--output-json",required=True); p.add_argument("--market-csv",required=True)
    print(json.dumps(run(Cfg(**vars(p.parse_args()))),indent=2,ensure_ascii=False))
