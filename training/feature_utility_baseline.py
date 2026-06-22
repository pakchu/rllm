"""Fit no-leak feature utility baselines from portfolio decision rows.

Uses only feature_snapshot/state_tokens available at decision time.  This is a
fast feasibility check before spending more Gemma/RL cycles.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BaselineCfg:
    input_jsonl: str
    output: str
    predictions_output: str
    fit_split: str = "train"
    eval_split: str = "eval"
    ridge_alpha: float = 10.0
    trade_threshold: float = 0.0
    min_abs_edge: float = 0.0


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names=set()
    for r in rows:
        snap=r.get('feature_snapshot',{}) if isinstance(r.get('feature_snapshot'),dict) else {}
        names.update(str(k) for k in snap.keys())
    return sorted(names)


def _xy(rows: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    x=[]; yl=[]; ys=[]
    for r in rows:
        snap=r.get('feature_snapshot',{}) if isinstance(r.get('feature_snapshot'),dict) else {}
        x.append([float(snap.get(n,0.0) or 0.0) for n in names])
        audit=r.get('reward_audit',{})
        yl.append(float(audit.get('LONG',{}).get('utility',0.0)))
        ys.append(float(audit.get('SHORT',{}).get('utility',0.0)))
    return np.asarray(x,float), np.asarray(yl,float), np.asarray(ys,float)


def _standardize(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray,np.ndarray,dict[str, Any]]:
    mu=x_train.mean(axis=0)
    sd=x_train.std(axis=0)
    sd=np.where(sd<1e-9,1.0,sd)
    return (x_train-mu)/sd, (x_eval-mu)/sd, {'mean':mu.tolist(),'std':sd.tolist()}


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb=np.c_[np.ones(len(x)),x]
    reg=np.eye(xb.shape[1])*float(alpha)
    reg[0,0]=0.0
    return np.linalg.solve(xb.T@xb+reg, xb.T@y)


def _pred(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)),x]@w


def _make_predictions(rows: list[dict[str, Any]], long_hat: np.ndarray, short_hat: np.ndarray, cfg: BaselineCfg) -> list[dict[str, Any]]:
    out=[]
    for r,lh,sh in zip(rows,long_hat,short_hat):
        best=max(float(lh),float(sh))
        gap=abs(float(lh)-float(sh))
        if best < float(cfg.trade_threshold) or gap < float(cfg.min_abs_edge):
            action='NO_TRADE'
        else:
            action='LONG' if lh>=sh else 'SHORT'
        out.append({'date':r.get('date'),'signal_pos':r.get('signal_pos'),'prediction':action,'target':r.get('target'),'candidate':r.get('candidate',{}),'predicted_utility':{'LONG':float(lh),'SHORT':float(sh),'best':best,'gap':gap}})
    return out


def _corr(a: np.ndarray,b: np.ndarray)->float:
    if len(a)<2 or float(np.std(a))<1e-9 or float(np.std(b))<1e-9: return 0.0
    return float(np.corrcoef(a,b)[0,1])


def run(cfg: BaselineCfg)->dict[str,Any]:
    rows=_load(cfg.input_jsonl)
    fit=[r for r in rows if str(r.get('split'))==cfg.fit_split]
    ev=[r for r in rows if str(r.get('split'))==cfg.eval_split]
    names=_feature_names(fit)
    xtr,yltr,ystr=_xy(fit,names); xev,ylev,ysev=_xy(ev,names)
    xtrz,xevz,scaler=_standardize(xtr,xev)
    wl=_ridge(xtrz,yltr,cfg.ridge_alpha); ws=_ridge(xtrz,ystr,cfg.ridge_alpha)
    pl=_pred(xevz,wl); ps=_pred(xevz,ws)
    preds=_make_predictions(ev,pl,ps,cfg)
    counts={}
    for p in preds: counts[p['prediction']]=counts.get(p['prediction'],0)+1
    report={'config':cfg.__dict__,'fit_rows':len(fit),'eval_rows':len(ev),'features':names,'metrics':{'long_utility_corr':_corr(pl,ylev),'short_utility_corr':_corr(ps,ysev),'side_gap_corr':_corr(pl-ps,ylev-ysev)},'prediction_counts':dict(sorted(counts.items())),'scaler_summary':{'n_features':len(names)}}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if cfg.predictions_output:
        Path(cfg.predictions_output).parent.mkdir(parents=True,exist_ok=True)
        Path(cfg.predictions_output).write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in preds)+'\n')
    return report


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description='Feature utility ridge baseline')
    p.add_argument('--input-jsonl',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--predictions-output',required=True)
    p.add_argument('--fit-split',default='train')
    p.add_argument('--eval-split',default='eval')
    p.add_argument('--ridge-alpha',type=float,default=10.0)
    p.add_argument('--trade-threshold',type=float,default=0.0)
    p.add_argument('--min-abs-edge',type=float,default=0.0)
    return p.parse_args()


def main()->None:
    print(json.dumps(run(BaselineCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))

if __name__=='__main__': main()
