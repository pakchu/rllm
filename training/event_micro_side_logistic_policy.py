"""Opportunity-filtered binary side logistic for event signals.

Fits LONG-vs-SHORT side from signal-time features, then sweeps only opportunity
and confidence thresholds on validation before final eval.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_micro_rule_sweep import OPPORTUNITY_FEATURES
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

SIDES = ["LONG", "SHORT"]
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]
CONFS = [0.5, 0.55, 0.6, 0.65, 0.7]

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    output: str
    work_dir: str = "results/event_micro_side_logistic"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    min_val_trades: int = 50
    epochs: int = 600
    lr: float = 0.08
    l2: float = 0.001
    use_state_tokens: int = 1


def load(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(path) if l.strip()]


def group(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[int(r["signal_pos"])].append(r)
    return [d[k] for k in sorted(d)]


def date(g): return str(g[0].get("date", ""))

def reward(g, side: str) -> float:
    for r in g:
        if str(r.get("side")) == side:
            rw = r.get("reward", {}) if isinstance(r.get("reward"), dict) else {}
            return float(rw.get("utility", rw.get("net_return_pct", 0.0)) or 0.0)
    return -999.0


def side_label(g) -> int:
    return 0 if reward(g,"LONG") >= reward(g,"SHORT") else 1


def feat(g, name: str) -> float:
    snap = g[0].get("feature_snapshot", {}) if isinstance(g[0].get("feature_snapshot"), dict) else {}
    try: return float(snap.get(name, 0.0) or 0.0)
    except Exception: return 0.0


def names(groups, use_tokens: bool):
    nums=set(); cats=set()
    for g in groups:
        nums.update((g[0].get("feature_snapshot") or {}).keys())
        if use_tokens:
            for k,v in (g[0].get("state_tokens") or {}).items(): cats.add(f"{k}={v}")
    return sorted(nums), sorted(cats)


def xy(groups, nums, cats):
    ci={c:i for i,c in enumerate(cats)}; X=np.zeros((len(groups),len(nums)+len(cats))); y=np.zeros(len(groups),int)
    for i,g in enumerate(groups):
        snap=g[0].get("feature_snapshot") or {}; toks=g[0].get("state_tokens") or {}
        X[i,:len(nums)] = [float(snap.get(n,0.0) or 0.0) for n in nums]
        for k,v in toks.items():
            j=ci.get(f"{k}={v}")
            if j is not None: X[i,len(nums)+j]=1.0
        y[i]=side_label(g)
    return X,y


def standardize(fit, other):
    mu=fit.mean(0); sd=fit.std(0); sd=np.where(sd<1e-9,1,sd); return (fit-mu)/sd, (other-mu)/sd


def train(X,y,lr,epochs,l2):
    Xb=np.c_[np.ones(len(X)),X]; W=np.zeros((Xb.shape[1],2)); Y=np.eye(2)[y]
    for _ in range(epochs):
        z=Xb@W; z-=z.max(1,keepdims=True); P=np.exp(z); P/=P.sum(1,keepdims=True)
        grad=Xb.T@(P-Y)/len(X)+l2*W; grad[0]=Xb.T[0]@(P-Y)/len(X); W-=lr*grad
    return W


def pred(X,W):
    Xb=np.c_[np.ones(len(X)),X]; z=Xb@W; z-=z.max(1,keepdims=True); P=np.exp(z); P/=P.sum(1,keepdims=True); return P


def write(groups, P, path, opp_feature, threshold, min_conf):
    out=[]; counts={"TRADE":0,"NO_TRADE":0,"LONG":0,"SHORT":0}
    for g,p in zip(groups,P):
        side_idx=int(np.argmax(p)); conf=float(p[side_idx]); side=SIDES[side_idx]
        if feat(g,opp_feature) >= threshold and conf >= min_conf:
            pr={"gate":"TRADE","side":side,"hold_bars":288,"confidence":"HIGH","family":"event_micro_side_logistic"}; scale=0.5; counts["TRADE"]+=1; counts[side]+=1
        else:
            pr={"gate":"NO_TRADE","side":"NONE","hold_bars":0,"confidence":"LOW","family":"event_micro_side_logistic"}; scale=0.0; counts["NO_TRADE"]+=1
        out.append({"date":g[0]["date"],"signal_pos":g[0]["signal_pos"],"prediction":pr,"position_scale":scale,"side_probs":dict(zip(SIDES,map(float,p)))})
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text("\n".join(json.dumps(r,sort_keys=True) for r in out)+"\n")
    return {"rows":len(out),"counts":counts,"output":path,"opp_feature":opp_feature,"threshold":threshold,"min_conf":min_conf}


def thresholds(groups, feature):
    vals=np.asarray([feat(g,feature) for g in groups],float); vals=vals[np.isfinite(vals)]
    if len(vals)<50 or np.std(vals)<1e-12: return []
    return [(q,float(np.quantile(vals,q))) for q in QUANTILES]


def side_metrics(y,P):
    pr=P.argmax(1)
    return {"accuracy":float((pr==y).mean()),"pred_counts":dict(zip(SIDES,np.bincount(pr,minlength=2).astype(int).tolist())),"label_counts":dict(zip(SIDES,np.bincount(y,minlength=2).astype(int).tolist()))}


def run(c: Cfg):
    allg=group(load(c.train_candidates)); evg=group(load(c.eval_candidates))
    fit=[g for g in allg if date(g)<c.validation_start]; val=[g for g in allg if c.validation_start<=date(g)<=c.validation_end]
    nums,cats=names(fit,bool(int(c.use_state_tokens)))
    Xf,yf=xy(fit,nums,cats); Xv,yv=xy(val,nums,cats); Xe,ye=xy(evg,nums,cats)
    Xfz,Xvz=standardize(Xf,Xv); _,Xez=standardize(Xf,Xe)
    W=train(Xfz,yf,c.lr,c.epochs,c.l2); Pv=pred(Xvz,W); Pe=pred(Xez,W)
    Path(c.work_dir).mkdir(parents=True,exist_ok=True)
    vals=[]
    for opp in OPPORTUNITY_FEATURES:
        for q,thr in thresholds(fit,opp):
            for cf in CONFS:
                pp=str(Path(c.work_dir)/f"val_{len(vals):04d}.jsonl"); ps=write(val,Pv,pp,opp,thr,cf)
                bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pp,market_csv=c.market_csv,output=str(Path(c.work_dir)/f"val_{len(vals):04d}.bt.json"),leverage=1.0,entry_delay_bars=1))
                score=float(bt['sim']['cagr_to_strict_mdd'])
                if int(bt['sim']['trade_entries'])<c.min_val_trades: score-=1000
                vals.append({'quantile':q,'score':score,'prediction_summary':ps,'val_sim':bt['sim'],'val_trade_stats':bt['trade_stats']})
    vals.sort(key=lambda r:r['score'],reverse=True); sel=vals[0]
    ep=str(Path(c.work_dir)/'selected_eval_predictions.jsonl'); eps=write(evg,Pe,ep,sel['prediction_summary']['opp_feature'],sel['prediction_summary']['threshold'],sel['prediction_summary']['min_conf'])
    ebt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=ep,market_csv=c.market_csv,output=str(Path(c.work_dir)/'selected_eval_backtest.json'),leverage=1.0,entry_delay_bars=1))
    rep={'config':c.__dict__,'rows':{'fit':len(fit),'val':len(val),'eval':len(evg)},'features':{'numeric':len(nums),'categorical':len(cats)},'side_metrics':{'fit':side_metrics(yf,pred(Xfz,W)),'val':side_metrics(yv,Pv),'eval':side_metrics(ye,Pe)},'top_val':vals[:20],'selected':sel,'eval_prediction_summary':eps,'eval_backtest':{'sim':ebt['sim'],'trade_stats':ebt['trade_stats']},'leakage_guard':'side model fit on <validation_start; validation selects opportunity/conf; eval final'}
    Path(c.output).parent.mkdir(parents=True,exist_ok=True); Path(c.output).write_text(json.dumps(rep,indent=2,ensure_ascii=False)); return rep


def parse_args():
    p=argparse.ArgumentParser();
    p.add_argument('--train-candidates',required=True); p.add_argument('--eval-candidates',required=True); p.add_argument('--output',required=True)
    p.add_argument('--work-dir',default=Cfg.work_dir); p.add_argument('--validation-start',default=Cfg.validation_start); p.add_argument('--validation-end',default=Cfg.validation_end); p.add_argument('--market-csv',default=Cfg.market_csv)
    p.add_argument('--min-val-trades',type=int,default=Cfg.min_val_trades); p.add_argument('--epochs',type=int,default=Cfg.epochs); p.add_argument('--lr',type=float,default=Cfg.lr); p.add_argument('--l2',type=float,default=Cfg.l2); p.add_argument('--use-state-tokens',type=int,default=Cfg.use_state_tokens)
    return Cfg(**vars(p.parse_args()))

def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
