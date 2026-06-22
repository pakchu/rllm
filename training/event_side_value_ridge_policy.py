"""Causal side expected-value ridge policy for event signals.

Fits separate LONG and SHORT utility regressors on pre-validation data. Validation
selects EV/margin thresholds. Eval is final holdout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_micro_side_logistic_policy import group, load, date, names, xy
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    output: str
    work_dir: str = "results/event_side_value_ridge"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    use_state_tokens: int = 1
    l2: float = 10.0
    min_val_trades: int = 50
    ev_thresholds: str = "-0.5,-0.25,0,0.1,0.2,0.3,0.5,0.75,1.0"
    margin_thresholds: str = "0,0.05,0.1,0.2,0.3,0.5,0.75"



def side_reward(g: list[dict[str, Any]], side: str) -> dict[str, float]:
    for row in g:
        if str(row.get("side")) == side:
            rw = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
            net = float(rw.get("net_return_pct", 0.0) or 0.0)
            util = float(rw.get("utility", net) or 0.0)
            return {"net": net, "utility": util}
    return {"net": 0.0, "utility": -999.0}

def standardize(fit: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mu=fit.mean(0); sd=fit.std(0); sd=np.where(sd<1e-9,1,sd)
    return (fit-mu)/sd, (other-mu)/sd, {"mean":mu.tolist(),"std":sd.tolist()}


def targets(groups: list[list[dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray]:
    yl=np.asarray([side_reward(g,"LONG")["utility"] for g in groups],float)
    ys=np.asarray([side_reward(g,"SHORT")["utility"] for g in groups],float)
    return yl,ys


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    Xb=np.c_[np.ones(len(X)),X]
    reg=np.eye(Xb.shape[1])*l2; reg[0,0]=0.0
    return np.linalg.solve(Xb.T@Xb+reg, Xb.T@y)


def pred(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)),X]@w


def corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<10 or a[m].std()<1e-12 or b[m].std()<1e-12: return 0.0
    return float(np.corrcoef(a[m],b[m])[0,1])


def metrics(pl, ps, yl, ys):
    pred_side=np.where(pl>=ps,0,1); true_side=np.where(yl>=ys,0,1)
    return {
        "long_corr": corr(pl,yl), "short_corr": corr(ps,ys), "diff_corr": corr(pl-ps,yl-ys),
        "side_accuracy": float((pred_side==true_side).mean()),
        "pred_long_rate": float((pred_side==0).mean()), "true_long_rate": float((true_side==0).mean()),
    }


def write_predictions(groups, pl, ps, path, ev_thr, margin_thr):
    rows=[]; counts={"TRADE":0,"NO_TRADE":0,"LONG":0,"SHORT":0}
    for g,l,s in zip(groups,pl,ps):
        best=max(float(l),float(s)); margin=abs(float(l)-float(s)); side="LONG" if l>=s else "SHORT"
        if best>=ev_thr and margin>=margin_thr:
            pr={"gate":"TRADE","side":side,"hold_bars":288,"confidence":"HIGH","family":"event_side_value_ridge"}; scale=0.5; counts["TRADE"]+=1; counts[side]+=1
        else:
            pr={"gate":"NO_TRADE","side":"NONE","hold_bars":0,"confidence":"LOW","family":"event_side_value_ridge"}; scale=0.0; counts["NO_TRADE"]+=1
        rows.append({"date":g[0]["date"],"signal_pos":g[0]["signal_pos"],"prediction":pr,"position_scale":scale,"pred_long_utility":float(l),"pred_short_utility":float(s)})
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text("\n".join(json.dumps(r,sort_keys=True) for r in rows)+"\n")
    return {"rows":len(rows),"counts":counts,"ev_threshold":ev_thr,"margin_threshold":margin_thr,"output":path}


def run(c: Cfg):
    allg=group(load(c.train_candidates)); evg=group(load(c.eval_candidates))
    fit=[g for g in allg if date(g)<c.validation_start]
    val=[g for g in allg if c.validation_start<=date(g)<=c.validation_end]
    nums,cats=names(fit,bool(int(c.use_state_tokens)))
    Xf,_=xy(fit,nums,cats); Xv,_=xy(val,nums,cats); Xe,_=xy(evg,nums,cats)
    Xfz,Xvz,_=standardize(Xf,Xv); _,Xez,_=standardize(Xf,Xe)
    yfl,yfs=targets(fit); yvl,yvs=targets(val); yel,yes=targets(evg)
    wl=fit_ridge(Xfz,yfl,c.l2); ws=fit_ridge(Xfz,yfs,c.l2)
    pfl,pfs=pred(Xfz,wl),pred(Xfz,ws); pvl,pvs=pred(Xvz,wl),pred(Xvz,ws); pel,pes=pred(Xez,wl),pred(Xez,ws)
    Path(c.work_dir).mkdir(parents=True,exist_ok=True)
    vals=[]
    for ev_thr in [float(x) for x in c.ev_thresholds.split(',') if x.strip()]:
        for mg in [float(x) for x in c.margin_thresholds.split(',') if x.strip()]:
            pp=str(Path(c.work_dir)/f"val_{len(vals):03d}.jsonl"); psum=write_predictions(val,pvl,pvs,pp,ev_thr,mg)
            bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pp,market_csv=c.market_csv,output=str(Path(c.work_dir)/f"val_{len(vals):03d}.bt.json"),leverage=1.0,entry_delay_bars=1))
            score=float(bt['sim']['cagr_to_strict_mdd'])
            if int(bt['sim']['trade_entries'])<c.min_val_trades: score-=1000
            vals.append({'score':score,'prediction_summary':psum,'val_sim':bt['sim'],'val_trade_stats':bt['trade_stats']})
    vals.sort(key=lambda r:r['score'],reverse=True); sel=vals[0]
    ep=str(Path(c.work_dir)/'selected_eval_predictions.jsonl'); eps=write_predictions(evg,pel,pes,ep,sel['prediction_summary']['ev_threshold'],sel['prediction_summary']['margin_threshold'])
    ebt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=ep,market_csv=c.market_csv,output=str(Path(c.work_dir)/'selected_eval_backtest.json'),leverage=1.0,entry_delay_bars=1))
    rep={'config':c.__dict__,'rows':{'fit':len(fit),'val':len(val),'eval':len(evg)},'features':{'numeric':len(nums),'categorical':len(cats)},'metrics':{'fit':metrics(pfl,pfs,yfl,yfs),'val':metrics(pvl,pvs,yvl,yvs),'eval':metrics(pel,pes,yel,yes)},'top_val':vals[:20],'selected':sel,'eval_prediction_summary':eps,'eval_backtest':{'sim':ebt['sim'],'trade_stats':ebt['trade_stats']},'leakage_guard':'ridge fit on pre-validation rows; validation selects thresholds; eval final'}
    Path(c.output).parent.mkdir(parents=True,exist_ok=True); Path(c.output).write_text(json.dumps(rep,indent=2,ensure_ascii=False)); return rep


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--train-candidates',required=True); p.add_argument('--eval-candidates',required=True); p.add_argument('--output',required=True)
    p.add_argument('--work-dir',default=Cfg.work_dir); p.add_argument('--validation-start',default=Cfg.validation_start); p.add_argument('--validation-end',default=Cfg.validation_end); p.add_argument('--market-csv',default=Cfg.market_csv)
    p.add_argument('--use-state-tokens',type=int,default=Cfg.use_state_tokens); p.add_argument('--l2',type=float,default=Cfg.l2); p.add_argument('--min-val-trades',type=int,default=Cfg.min_val_trades); p.add_argument('--ev-thresholds',default=Cfg.ev_thresholds); p.add_argument('--margin-thresholds',default=Cfg.margin_thresholds)
    return Cfg(**vars(p.parse_args()))

def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
