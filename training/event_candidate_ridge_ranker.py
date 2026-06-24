"""No-leak ridge ranker for event candidate TAKE/ABSTAIN policies.

Protocol:
- Fit ridge expected net return on candidate rows before validation year.
- Select score quantile and full-size margin on validation year only.
- Refit on all train rows and replay selected quantile rule on eval rows.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class EventCandidateRidgeCfg:
    train_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_ridge"
    validation_start: str = "2024-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    ridge_alpha: float = 100.0
    quantiles: str = "0.50,0.60,0.70,0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.25,0.5,1.0"
    min_val_trades: int = 30
    leverage: float = 1.0
    entry_delay_bars: int = 1


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _date(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))


def _feature_names(rows: list[dict[str, Any]], drop_prefixes: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    nums=set(); cats=set()
    for r in rows:
        snap=r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        nums.update(str(k) for k in snap.keys() if not str(k).startswith(drop_prefixes))
        toks=r.get("state_tokens", {}) if isinstance(r.get("state_tokens"), dict) else {}
        for k,v in toks.items():
            cats.add(f"tok:{k}={v}")
    return sorted(nums), sorted(cats)


def _xy(rows: list[dict[str, Any]], num_names: list[str], cat_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cat_index={c:i for i,c in enumerate(cat_names)}
    x=np.zeros((len(rows), len(num_names)*2 + len(cat_names) + 3), dtype=float)
    y=np.zeros(len(rows), dtype=float)
    for i,r in enumerate(rows):
        side=str(r.get("side"))
        sign=1.0 if side=="LONG" else -1.0
        snap=r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        vals=np.asarray([float(snap.get(n,0.0) or 0.0) for n in num_names], dtype=float)
        x[i, :len(num_names)] = vals
        x[i, len(num_names):len(num_names)*2] = vals * sign
        base=len(num_names)*2
        toks=r.get("state_tokens", {}) if isinstance(r.get("state_tokens"), dict) else {}
        for k,v in toks.items():
            j=cat_index.get(f"tok:{k}={v}")
            if j is not None:
                x[i, base+j]=1.0
        base += len(cat_names)
        x[i, base:base+3] = [1.0 if side=="LONG" else 0.0, 1.0 if side=="SHORT" else 0.0, sign]
        y[i]=float(r.get("reward",{}).get("net_return_pct",0.0))
    return x,y


def _standardize(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    mu=xtr.mean(axis=0); sd=xtr.std(axis=0); sd=np.where(sd<1e-9,1.0,sd)
    return (xtr-mu)/sd, (xte-mu)/sd, {"mean":mu.tolist(),"std":sd.tolist()}


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb=np.c_[np.ones(len(x)), x]
    reg=np.eye(xb.shape[1])*float(alpha); reg[0,0]=0.0
    return np.linalg.solve(xb.T@xb+reg, xb.T@y)


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)),x]@w


def _best_by_signal(rows: list[dict[str,Any]], scores: np.ndarray) -> list[dict[str,Any]]:
    best: dict[int, dict[str,Any]]={}
    for r,sc in zip(rows,scores):
        pos=int(r.get("signal_pos"))
        cur=best.get(pos)
        if cur is None or float(sc)>float(cur["score"]):
            best[pos]={"row":r,"score":float(sc)}
    return [best[k] for k in sorted(best)]


def _write_policy(
    best_rows: list[dict[str,Any]],
    output: str,
    threshold: float,
    full_margin: float,
    small_scale: float=0.5,
    allowed_sides: set[str] | None = None,
    side_scale_by_side: dict[str, float] | None = None,
    max_feature_name: str = "",
    max_feature_value: float | None = None,
    max_feature_filters: dict[str, float] | None = None,
    min_feature_filters: dict[str, float] | None = None,
) -> dict[str,Any]:
    max_filters = dict(max_feature_filters or {})
    if max_feature_name and max_feature_value is not None:
        max_filters[max_feature_name] = float(max_feature_value)
    min_filters = dict(min_feature_filters or {})
    out=[]; counts={"TRADE":0,"NO_TRADE":0,"LONG":0,"SHORT":0,"FULL":0,"SMALL":0}
    for item in best_rows:
        r=item["row"]; score=float(item["score"]); side=str(r.get("side")); hold=int(r.get("candidate",{}).get("hold_bars",288) or 288)
        snap = r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        feature_ok = all(float(snap.get(name, 0.0) or 0.0) <= float(value) for name, value in max_filters.items()) and all(
            float(snap.get(name, 0.0) or 0.0) >= float(value) for name, value in min_filters.items()
        )
        if score >= threshold and side in {"LONG","SHORT"} and (allowed_sides is None or side in allowed_sides) and feature_ok:
            scale=1.0 if score >= threshold + float(full_margin) else float(small_scale)
            if side_scale_by_side is not None:
                scale *= min(1.0, max(0.0, float(side_scale_by_side.get(side, 0.0))))
            pred={"gate":"TRADE","side":side,"hold_bars":hold,"confidence":"HIGH","family":"event_candidate_ridge"}
            counts["TRADE"]+=1; counts[side]+=1; counts["FULL" if scale>=1.0 else "SMALL"]+=1
        else:
            scale=0.0; pred={"gate":"NO_TRADE","side":"NONE","hold_bars":0,"confidence":"LOW","family":"event_candidate_ridge"}; counts["NO_TRADE"]+=1
        out.append({"date":r.get("date"),"signal_pos":r.get("signal_pos"),"prediction":pred,"position_scale":scale,"score":score,"side_candidate":side})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in out)+"\n")
    return {"rows":len(out),"counts":counts,"threshold":threshold,"full_margin":full_margin,"allowed_sides": sorted(allowed_sides) if allowed_sides is not None else None,"side_scale_by_side": side_scale_by_side,"max_feature_name": max_feature_name or None,"max_feature_value": max_feature_value,"max_feature_filters": max_filters or None,"min_feature_filters": min_filters or None,"output":output}


def _fit_score(fit_rows: list[dict[str,Any]], score_rows: list[dict[str,Any]], alpha: float, names: tuple[list[str],list[str]]|None=None) -> tuple[np.ndarray, np.ndarray, tuple[list[str],list[str]]]:
    if names is None:
        names=_feature_names(fit_rows)
    num,cat=names
    xtr,ytr=_xy(fit_rows,num,cat); xte,_=_xy(score_rows,num,cat)
    xtrz,xtez,_=_standardize(xtr,xte)
    w=_ridge(xtrz,ytr,alpha)
    return _predict(xtrz,w), _predict(xtez,w), names


def run(cfg: EventCandidateRidgeCfg) -> dict[str,Any]:
    train_all=_load(cfg.train_jsonl); eval_rows=_load(cfg.eval_jsonl)
    fit=[r for r in train_all if _date(r) < cfg.validation_start]
    val=[r for r in train_all if cfg.validation_start <= _date(r) <= cfg.validation_end]
    qs=[float(x) for x in cfg.quantiles.split(',') if x.strip()]
    margins=[float(x) for x in cfg.full_margins.split(',') if x.strip()]
    fit_scores,val_scores,names=_fit_score(fit,val,cfg.ridge_alpha)
    fit_best_scores=np.asarray([x["score"] for x in _best_by_signal(fit,fit_scores)], dtype=float)
    val_best=_best_by_signal(val,val_scores)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    candidates=[]
    with tempfile.TemporaryDirectory(prefix="rllm_event_ridge_val_") as tmp_raw:
        tmp=Path(tmp_raw)
        for q in qs:
            thr=float(np.quantile(fit_best_scores,q)) if len(fit_best_scores) else 999.0
            for margin in margins:
                pred=tmp/f"val_q{q}_m{margin}.jsonl"
                ps=_write_policy(val_best,str(pred),thr,margin)
                bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred),market_csv=cfg.market_csv,output=str(tmp/f"val_q{q}_m{margin}.bt.json"),leverage=cfg.leverage,entry_delay_bars=cfg.entry_delay_bars))
                sim=bt["sim"]; stats=bt["trade_stats"]
                score=float(sim.get("cagr_to_strict_mdd",-999.0) or -999.0)
                if int(sim.get("trade_entries",0) or 0) < cfg.min_val_trades:
                    score -= 1000.0
                candidates.append({"q":q,"full_margin":margin,"threshold":thr,"prediction_summary":ps,"val_sim":sim,"val_trade_stats":stats,"score":score})
    candidates.sort(key=lambda r:float(r["score"]), reverse=True)
    selected=candidates[0]
    all_train_scores,eval_scores,names2=_fit_score(train_all,eval_rows,cfg.ridge_alpha,names)
    train_best_scores=np.asarray([x["score"] for x in _best_by_signal(train_all,all_train_scores)], dtype=float)
    eval_thr=float(np.quantile(train_best_scores,float(selected["q"]))) if len(train_best_scores) else 999.0
    eval_best=_best_by_signal(eval_rows,eval_scores)
    eval_pred=str(Path(cfg.work_dir)/"selected_eval_predictions.jsonl")
    eval_ps=_write_policy(eval_best,eval_pred,eval_thr,float(selected["full_margin"]))
    eval_bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred,market_csv=cfg.market_csv,output=str(Path(cfg.work_dir)/"selected_eval_backtest.json"),leverage=cfg.leverage,entry_delay_bars=cfg.entry_delay_bars))
    report={"config":cfg.__dict__,"rows":{"fit":len(fit),"val":len(val),"train_all":len(train_all),"eval":len(eval_rows)},"features":{"numeric":len(names[0]),"categorical":len(names[1])},"selection_rule":"select q/full_margin on 2024 validation only; eval refits on all train and uses same q/full_margin","top10_val":candidates[:10],"selected":selected,"eval_threshold":eval_thr,"eval_prediction_summary":eval_ps,"eval_backtest":{"sim":eval_bt["sim"],"trade_stats":eval_bt["trade_stats"]},"leakage_guard":{"validation_only_selects_policy":True,"eval_not_used_for_fit_or_threshold_selection":True,"eval_threshold_uses_train_score_quantile":True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description="Event candidate ridge ranker")
    p.add_argument("--train-jsonl",required=True); p.add_argument("--eval-jsonl",required=True); p.add_argument("--market-csv",required=True); p.add_argument("--output",required=True)
    p.add_argument("--work-dir",default=EventCandidateRidgeCfg.work_dir); p.add_argument("--validation-start",default=EventCandidateRidgeCfg.validation_start); p.add_argument("--validation-end",default=EventCandidateRidgeCfg.validation_end)
    p.add_argument("--ridge-alpha",type=float,default=EventCandidateRidgeCfg.ridge_alpha); p.add_argument("--quantiles",default=EventCandidateRidgeCfg.quantiles); p.add_argument("--full-margins",default=EventCandidateRidgeCfg.full_margins); p.add_argument("--min-val-trades",type=int,default=EventCandidateRidgeCfg.min_val_trades); p.add_argument("--leverage",type=float,default=1.0); p.add_argument("--entry-delay-bars",type=int,default=1)
    return p.parse_args()


def main()->None:
    print(json.dumps(run(EventCandidateRidgeCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
