"""Sweep one-feature gates over fixed event candidate rows.

This is a cheap rankability/regime-filter check: candidate rows already contain
past-only feature snapshots and action metadata. Gate candidates are selected
using train+test only; eval metrics are emitted as holdout evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from training.economic_action_backtest import strict_backtest_actions, EconomicActionBacktestConfig
from training.strict_bar_backtest import load_market_bars

def load(p: str) -> list[dict[str, Any]]:
 out=[]
 for l in Path(p).read_text().splitlines():
  if not l.strip(): continue
  r=json.loads(l); a=r['action']; fs=r.get('feature_snapshot',{})
  out.append({**r,'prediction':{'gate':'TRADE','family':a['family'],'side':a['side'],'hold_bars':a['hold_bars']},'_fs':fs})
 return out
def to_preds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]: return [{'date':r['date'],'signal_pos':r['signal_pos'],'prediction':r['prediction']} for r in rows]
def bt(rows: list[dict[str, Any]], market: Any, lev: float = 0.5) -> dict[str, Any] | None:
 if not rows: return None
 return strict_backtest_actions(to_preds(rows),market,EconomicActionBacktestConfig(leverage=lev,fee_rate=0.0004,slippage_rate=0.0001,entry_delay_bars=1,max_hold_bars=144))
def score(sim: dict[str, Any]) -> float:
 s=sim['sim']; st=sim['trade_stats']; return s['cagr_to_strict_mdd']+0.01*s['cagr_pct']+0.001*s['trade_entries']-0.5*max(0,st.get('p_value_mean_ret_approx',1)-0.1)
def run(*, train_jsonl: str, test_jsonl: str, eval_jsonl: str, market_csv: str, output: str) -> dict[str, Any]:
    market=load_market_bars(market_csv)
    sets={"train":load(train_jsonl),"test":load(test_jsonl),"eval":load(eval_jsonl)}
    keys=sorted({k for r in sets["train"] for k,v in r["_fs"].items() if isinstance(v,(int,float)) and np.isfinite(float(v))})
    rows=[]
    for key in keys:
     vals=np.array([float(r["_fs"].get(key,0) or 0) for r in sets["train"]],float)
     for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
      thr=float(np.quantile(vals,q))
      for op in [">=","<="]:
       filt={}
       ok=True
       for split,rs in sets.items():
        rr=[r for r in rs if (float(r["_fs"].get(key,0) or 0)>=thr if op==">=" else float(r["_fs"].get(key,0) or 0)<=thr)]
        res=bt(rr,market,0.5)
        if res is None: ok=False; break
        filt[split]={"n_rows":len(rr),"sim":res["sim"],"trade_stats":res["trade_stats"]}
       if not ok: continue
       if filt["train"]["sim"]["trade_entries"]<120 or filt["test"]["sim"]["trade_entries"]<25: continue
       if filt["train"]["sim"]["cagr_pct"]<=0 or filt["test"]["sim"]["cagr_pct"]<=0: continue
       if filt["train"]["sim"]["strict_mdd_pct"]>25 or filt["test"]["sim"]["strict_mdd_pct"]>15: continue
       sel_score=score(filt["train"])+2*score(filt["test"])
       rows.append({"feature":key,"op":op,"threshold":thr,"train":filt["train"],"test":filt["test"],"eval":filt["eval"],"selection_score":sel_score})
    rows.sort(key=lambda x:x["selection_score"],reverse=True)
    top=rows[:20]
    for r in top[:5]:
     levs=[]
     for lev in [0.5,1.0,1.25,1.5,2.0]:
      rs=[x for x in sets["eval"] if (float(x["_fs"].get(r["feature"],0) or 0)>=r["threshold"] if r["op"]==">=" else float(x["_fs"].get(r["feature"],0) or 0)<=r["threshold"])]
      res=bt(rs,market,lev); levs.append({"leverage":lev,"eval":{"n_rows":len(rs),"sim":res["sim"],"trade_stats":res["trade_stats"]}})
     r["eval_leverage_grid"]=levs
    out={"leakage_guard":{"gate_candidates_scored_on_train_and_test_only":True,"eval_not_used_for_selection":True},"candidate_count":len(rows),"top":top}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,ensure_ascii=False))
    return out

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Sweep one-feature gates over fixed event candidate rows")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()

def main() -> None:
    a=parse_args(); rep=run(train_jsonl=a.train_jsonl,test_jsonl=a.test_jsonl,eval_jsonl=a.eval_jsonl,market_csv=a.market_csv,output=a.output)
    print(json.dumps({"candidate_count":rep["candidate_count"],"top":rep["top"][:10]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
