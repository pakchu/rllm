"""Audit feature/label drift for event signal action ranking.

The prior oracle-vs-logistic test showed a large future-label upper bound but a
poor no-leak feature model. This audit measures whether signal-time features have
stable information coefficient against LONG-vs-SHORT and TRADE-vs-NO_TRADE labels
across train/validation/eval periods.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DriftCfg:
    train_candidates: str
    eval_candidates: str
    output: str
    train_end: str = "2022-12-31 23:59:59"
    val_start: str = "2023-01-01"
    val_end: str = "2024-12-31 23:59:59"


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _date(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))


def _groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[int(r.get("signal_pos"))].append(r)
    return [d[k] for k in sorted(d)]


def _split(groups: list[list[dict[str, Any]]], cfg: DriftCfg) -> dict[str, list[list[dict[str, Any]]]]:
    return {
        "fit_2020_2022": [g for g in groups if _date(g[0]) <= cfg.train_end],
        "val_2023_2024": [g for g in groups if cfg.val_start <= _date(g[0]) <= cfg.val_end],
        "eval_2025_2026": [g for g in groups if _date(g[0]) > cfg.val_end],
    }


def _reward(group: list[dict[str, Any]], side: str) -> dict[str, float]:
    for r in group:
        if str(r.get("side")) == side:
            rw = r.get("reward", {}) if isinstance(r.get("reward"), dict) else {}
            net = float(rw.get("net_return_pct", 0.0) or 0.0)
            util = float(rw.get("utility", net) or 0.0)
            return {"net": net, "utility": util}
    return {"net": 0.0, "utility": -999.0}


def _labels(group: list[dict[str, Any]]) -> dict[str, float | str]:
    lr = _reward(group, "LONG")
    sr = _reward(group, "SHORT")
    best_side = "LONG" if lr["utility"] >= sr["utility"] else "SHORT"
    best = max(lr["utility"], sr["utility"], 0.0)
    best_action = best_side if max(lr["utility"], sr["utility"]) > 0.0 else "NO_TRADE"
    return {
        "long_utility": lr["utility"],
        "short_utility": sr["utility"],
        "long_net": lr["net"],
        "short_net": sr["net"],
        "side_utility_diff": lr["utility"] - sr["utility"],
        "best_trade_utility": max(lr["utility"], sr["utility"]),
        "best_vs_no_trade": best,
        "best_action": best_action,
    }


def _feature_names(groups: list[list[dict[str, Any]]]) -> list[str]:
    keys=set()
    for g in groups:
        snap = g[0].get("feature_snapshot", {}) if isinstance(g[0].get("feature_snapshot"), dict) else {}
        keys.update(snap.keys())
    return sorted(str(k) for k in keys)


def _feature(group: list[dict[str, Any]], name: str) -> float:
    snap = group[0].get("feature_snapshot", {}) if isinstance(group[0].get("feature_snapshot"), dict) else {}
    try:
        return float(snap.get(name, 0.0) or 0.0)
    except Exception:
        return 0.0


def _rank_corr(x: list[float], y: list[float]) -> float | None:
    xa=np.asarray(x,dtype=float); ya=np.asarray(y,dtype=float)
    mask=np.isfinite(xa)&np.isfinite(ya)
    xa=xa[mask]; ya=ya[mask]
    if len(xa)<50 or float(np.std(xa))<1e-12 or float(np.std(ya))<1e-12:
        return None
    rx=np.argsort(np.argsort(xa)).astype(float)
    ry=np.argsort(np.argsort(ya)).astype(float)
    return float(np.corrcoef(rx,ry)[0,1])


def _period_feature_ic(groups: list[list[dict[str, Any]]], features: list[str]) -> dict[str, Any]:
    labels=[_labels(g) for g in groups]
    targets={
        "side_utility_diff": [float(l["side_utility_diff"]) for l in labels],
        "best_trade_utility": [float(l["best_trade_utility"]) for l in labels],
        "long_utility": [float(l["long_utility"]) for l in labels],
        "short_utility": [float(l["short_utility"]) for l in labels],
    }
    out=[]
    for f in features:
        xs=[_feature(g,f) for g in groups]
        row={"feature":f}
        for t,ys in targets.items():
            c=_rank_corr(xs,ys)
            if c is not None: row[t]=c
        if len(row)>1: out.append(row)
    vals, counts = np.unique([str(l["best_action"]) for l in labels], return_counts=True)
    label_counts = {str(v): int(c) for v, c in zip(vals, counts)}
    return {"rows": int(len(groups)), "label_counts": label_counts, "ics": out}


def _compare(periods: dict[str, Any]) -> list[dict[str, Any]]:
    fit={r['feature']:r for r in periods['fit_2020_2022']['ics']}
    val={r['feature']:r for r in periods['val_2023_2024']['ics']}
    ev={r['feature']:r for r in periods['eval_2025_2026']['ics']}
    rows=[]
    for f in sorted(set(fit)|set(val)|set(ev)):
        for target in ['side_utility_diff','best_trade_utility','long_utility','short_utility']:
            a=fit.get(f,{}).get(target); b=val.get(f,{}).get(target); c=ev.get(f,{}).get(target)
            if a is None or b is None or c is None: continue
            rows.append({'feature':f,'target':target,'fit_ic':a,'val_ic':b,'eval_ic':c,'fit_val_same_sign':a*b>0,'val_eval_same_sign':b*c>0,'min_abs_ic':min(abs(a),abs(b),abs(c)),'drift_abs':abs(b-c)})
    rows.sort(key=lambda r:(r['val_eval_same_sign'], r['min_abs_ic']), reverse=True)
    return rows


def run(cfg: DriftCfg) -> dict[str, Any]:
    groups=_groups(_load(cfg.train_candidates)+_load(cfg.eval_candidates))
    periods={k:v for k,v in _split(groups,cfg).items()}
    features=_feature_names(groups)
    per={k:_period_feature_ic(v,features) for k,v in periods.items()}
    cmp=_compare(per)
    report={"config":cfg.__dict__,"features":len(features),"periods":per,"stable_top30":cmp[:30],"unstable_top30":sorted(cmp,key=lambda r:r['drift_abs'],reverse=True)[:30],"interpretation":"ICs are signal-level; all targets are future-label audit values, features are signal-time snapshots"}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--train-candidates',required=True); p.add_argument('--eval-candidates',required=True); p.add_argument('--output',required=True)
    p.add_argument('--train-end',default=DriftCfg.train_end); p.add_argument('--val-start',default=DriftCfg.val_start); p.add_argument('--val-end',default=DriftCfg.val_end)
    return p.parse_args()


def main(): print(json.dumps(run(DriftCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
