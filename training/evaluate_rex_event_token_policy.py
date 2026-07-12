"""Evaluate a simple token Naive Bayes policy on REX event reasoning rows."""
from __future__ import annotations

import argparse, json, math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from training.event_candidate_pool_probe import EventPoolConfig, _load_market, _simulate_rows

ACTIONS = ("LONG", "SHORT", "NO_TRADE")

@dataclass(frozen=True)
class Cfg:
    input_jsonl: str
    market_csv: str
    output: str
    hold_bars: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    alpha: float = 1.0


def read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tokens(row: dict[str, Any]) -> list[str]:
    st = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
    return [f"{k}={v}" for k, v in sorted(st.items())]


def target(row: dict[str, Any]) -> str:
    return str(json.loads(row["target"])["action"])


def train_nb(rows: list[dict[str, Any]], alpha: float):
    class_counts = Counter(target(r) for r in rows)
    token_counts = {a: Counter() for a in ACTIONS}
    vocab=set()
    for r in rows:
        y=target(r)
        for t in tokens(r):
            token_counts[y][t]+=1; vocab.add(t)
    totals={a:sum(token_counts[a].values()) for a in ACTIONS}
    n=sum(class_counts.values())
    v=max(1,len(vocab))
    def predict(r):
        toks=tokens(r)
        scores={}
        for a in ACTIONS:
            scores[a]=math.log((class_counts[a]+alpha)/(n+alpha*len(ACTIONS)))
            denom=totals[a]+alpha*v
            for t in toks:
                scores[a]+=math.log((token_counts[a][t]+alpha)/denom)
        return max(ACTIONS, key=lambda a:scores[a]), scores
    return predict, {"class_counts":dict(class_counts),"vocab_size":len(vocab),"token_totals":totals}


def sim_predictions(rows: list[dict[str, Any]], preds: list[str], market, cfg: Cfg):
    trade_rows=[]
    for r,p in zip(rows,preds):
        if p in {"LONG","SHORT"}:
            trade_rows.append({"date":r["date"],"signal_date":r["date"],"side":p,"family":"rex_event_token_nb","strength":1.0,"score_mean":1.0})
    ecfg=EventPoolConfig(input_csv=cfg.market_csv, output="", hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate)
    return {"predicted_trade_rows":len(trade_rows), **_simulate_rows(trade_rows, market, ecfg)}


def run(cfg: Cfg) -> dict[str, Any]:
    rows=read_jsonl(cfg.input_jsonl)
    train=[r for r in rows if r.get("split")=="train"]
    predict, model_info=train_nb(train, cfg.alpha)
    market=_load_market(cfg.market_csv)
    report={"config":asdict(cfg),"model_info":model_info,"splits":{},"leakage_guard":{"trained_on_train_rows_only":True,"test_eval_targets_used_for_metrics_only":True,"predictions_use_state_tokens_only":True}}
    for sp in ("train","test","eval"):
        part=[r for r in rows if r.get("split")==sp]
        preds=[]; correct=0; confusion=Counter(); target_counts=Counter(); pred_counts=Counter()
        for r in part:
            p,_=predict(r); y=target(r); preds.append(p)
            correct+=int(p==y); confusion[f"target={y}|pred={p}"]+=1; target_counts[y]+=1; pred_counts[p]+=1
        sim=sim_predictions(part,preds,market,cfg)
        report["splits"][sp]={"rows":len(part),"accuracy":correct/max(1,len(part)),"target_counts":dict(target_counts),"prediction_counts":dict(pred_counts),"confusion":dict(confusion),"backtest":{"sim":sim.get("sim",{}),"trade_stats":sim.get("trade_stats",{}),"predicted_trade_rows":sim.get("predicted_trade_rows")}}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-jsonl",required=True); p.add_argument("--market-csv",required=True); p.add_argument("--output",required=True)
    p.add_argument("--alpha",type=float,default=1.0)
    args=p.parse_args()
    print(json.dumps(run(Cfg(**vars(args))), indent=2, ensure_ascii=False))
if __name__=="__main__": main()
