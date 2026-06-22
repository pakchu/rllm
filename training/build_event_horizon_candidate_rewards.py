"""Recompute event candidate rewards for alternate holding horizons.

Inputs are existing side-specific candidate rows; features remain signal-time only.
Only reward/target/candidate hold_bars change for offline label experiments.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    market_csv: str
    output_dir: str
    horizons: str = "36,72,144,288,576"
    fee_pct: float = 0.1
    mae_penalty: float = 0.35
    full_net_return_pct: float = 1.2
    small_net_return_pct: float = 0.25
    max_full_mae_pct: float = 5.0
    max_small_mae_pct: float = 7.5
    min_full_utility_pct: float = 0.5
    min_small_utility_pct: float = 0.0


def load(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(path) if l.strip()]


def write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)+"\n")


def target(rew: dict[str, float], c: Cfg) -> dict[str, str]:
    net=rew["net_return_pct"]; mae=rew["mae_pct"]; util=rew["utility"]
    if net>=c.full_net_return_pct and mae<=c.max_full_mae_pct and util>=c.min_full_utility_pct:
        return {"decision":"TAKE_FULL","risk_reason":"reward_strong_after_path_risk"}
    if net>=c.small_net_return_pct and mae<=c.max_small_mae_pct and util>=c.min_small_utility_pct:
        return {"decision":"TAKE_SMALL","risk_reason":"reward_positive_but_thin"}
    return {"decision":"ABSTAIN","risk_reason":"reward_not_worth_path_risk"}


def reward_for(pos: int, side: str, h: int, m: pd.DataFrame, c: Cfg) -> dict[str, float] | None:
    entry_i=pos+1; exit_i=pos+h
    if entry_i>=len(m) or exit_i>=len(m): return None
    entry=float(m.iloc[entry_i]["open"]); exitp=float(m.iloc[exit_i]["close"])
    path=m.iloc[entry_i:exit_i+1]
    if entry<=0 or len(path)==0: return None
    high=float(path["high"].max()); low=float(path["low"].min())
    if side=="LONG":
        gross=(exitp/entry-1.0)*100.0
        mae=max(0.0,(entry-low)/entry*100.0)
        mfe=max(0.0,(high-entry)/entry*100.0)
    else:
        gross=(entry-exitp)/entry*100.0
        mae=max(0.0,(high-entry)/entry*100.0)
        mfe=max(0.0,(entry-low)/entry*100.0)
    net=gross-c.fee_pct
    return {"net_return_pct":net,"mae_pct":mae,"mfe_pct":mfe,"utility":net-c.mae_penalty*mae}


def convert(rows: list[dict[str, Any]], h: int, m: pd.DataFrame, c: Cfg) -> list[dict[str, Any]]:
    out=[]
    for row in rows:
        side=str(row.get("side")); pos=int(row.get("signal_pos"))
        rew=reward_for(pos,side,h,m,c)
        if rew is None: continue
        r=dict(row)
        cand=dict(r.get("candidate",{}) if isinstance(r.get("candidate"),dict) else {})
        cand["hold_bars"]=h; cand["side"]=side
        r["candidate"]=cand; r["reward"]=rew; r["target"]=target(rew,c)
        lg=dict(r.get("leakage_guard",{}) if isinstance(r.get("leakage_guard"),dict) else {})
        lg["alternate_horizon_reward_training_only"]=True
        r["leakage_guard"]=lg
        out.append(r)
    return out


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dec=defaultdict(int); side=defaultdict(int); net=[]; util=[]
    for r in rows:
        dec[str((r.get("target") or {}).get("decision"))]+=1; side[str(r.get("side"))]+=1
        net.append(float(r["reward"]["net_return_pct"])); util.append(float(r["reward"]["utility"]))
    a=np.asarray(net,float); u=np.asarray(util,float)
    return {"rows":len(rows),"decisions":dict(dec),"sides":dict(side),"net_mean":float(a.mean()) if len(a) else 0.0,"net_positive_rate":float((a>0).mean()) if len(a) else 0.0,"utility_mean":float(u.mean()) if len(u) else 0.0}


def run(c: Cfg) -> dict[str, Any]:
    m=pd.read_csv(c.market_csv)
    train=load(c.train_candidates); ev=load(c.eval_candidates)
    report={"config":c.__dict__,"horizons":{}}
    for h in [int(x) for x in c.horizons.split(',') if x.strip()]:
        tr=convert(train,h,m,c); er=convert(ev,h,m,c)
        tp=str(Path(c.output_dir)/f"event_candidate_ext_micro_h{h}_train.jsonl")
        ep=str(Path(c.output_dir)/f"event_candidate_ext_micro_h{h}_eval.jsonl")
        write(tp,tr); write(ep,er)
        report["horizons"][str(h)]={"train":{**summary(tr),"output":tp},"eval":{**summary(er),"output":ep}}
    sp=str(Path(c.output_dir)/"event_horizon_candidate_rewards_summary.json")
    Path(sp).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser(); p.add_argument('--train-candidates',required=True); p.add_argument('--eval-candidates',required=True); p.add_argument('--market-csv',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--horizons',default=Cfg.horizons); p.add_argument('--fee-pct',type=float,default=Cfg.fee_pct); p.add_argument('--mae-penalty',type=float,default=Cfg.mae_penalty)
    return Cfg(**vars(p.parse_args()))

def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
