"""Audit candidate-row feature/reward stability by calendar year."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.event_candidate_ridge_ranker import _date, _load


@dataclass(frozen=True)
class CandidateFeatureStabilityCfg:
    input_jsonl: str
    output: str
    feature_prefixes: str = "mreg_"
    min_rows: int = 1000
    quantile: float = 0.2


def _utility(row: dict[str, Any]) -> float:
    reward = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("rank_utility", reward.get("net_return_pct", 0.0)) or 0.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    if x.size < 3:
        return 0.0
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    sx = float(np.std(xr)); sy = float(np.std(yr))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def _mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else 0.0


def _metrics(rows: list[dict[str, Any]], feature: str, cfg: CandidateFeatureStabilityCfg) -> dict[str, Any]:
    xs=[]; ys=[]
    for r in rows:
        snap = r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        if feature not in snap:
            continue
        x = float(snap.get(feature, 0.0) or 0.0); y = _utility(r)
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x); ys.append(y)
    x=np.asarray(xs,dtype=float); y=np.asarray(ys,dtype=float)
    if x.size < cfg.min_rows:
        return {"n": int(x.size)}
    q=float(np.clip(cfg.quantile,0.01,0.49)); lo=float(np.quantile(x,q)); hi=float(np.quantile(x,1-q))
    low=y[x<=lo]; high=y[x>=hi]
    return {"n": int(x.size), "spearman_ic": _spearman(x,y), "q_high_minus_low": _mean(high)-_mean(low), "q_low": lo, "q_high": hi}


def _sign(v: float) -> int:
    return 1 if v > 1e-12 else -1 if v < -1e-12 else 0


def run(cfg: CandidateFeatureStabilityCfg) -> dict[str, Any]:
    rows=_load(cfg.input_jsonl)
    prefixes=tuple(x.strip() for x in cfg.feature_prefixes.split(',') if x.strip())
    feats=sorted({str(k) for r in rows for k,v in (r.get('feature_snapshot',{}) if isinstance(r.get('feature_snapshot'),dict) else {}).items() if isinstance(v,(int,float)) and str(k).startswith(prefixes)})
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_year[_date(r)[:4]].append(r)
    out_rows=[]
    for feat in feats:
        per={y:_metrics(rs,feat,cfg) for y,rs in sorted(by_year.items())}
        valid=[m for m in per.values() if int(m.get('n',0))>=cfg.min_rows]
        spreads=[float(m.get('q_high_minus_low',0.0)) for m in valid]
        ics=[float(m.get('spearman_ic',0.0)) for m in valid]
        spread_signs=[_sign(v) for v in spreads if _sign(v)]
        ic_signs=[_sign(v) for v in ics if _sign(v)]
        spread_consistency=abs(sum(spread_signs))/max(1,len(spread_signs)) if spread_signs else 0.0
        ic_consistency=abs(sum(ic_signs))/max(1,len(ic_signs)) if ic_signs else 0.0
        min_abs_spread=min([abs(v) for v in spreads], default=0.0)
        min_abs_ic=min([abs(v) for v in ics], default=0.0)
        score=min_abs_spread*1000.0 + min_abs_ic*100.0 + spread_consistency + ic_consistency
        out_rows.append({"feature":feat,"score":score,"spread_consistency":spread_consistency,"ic_consistency":ic_consistency,"min_abs_spread":min_abs_spread,"min_abs_ic":min_abs_ic,"years":per})
    out_rows.sort(key=lambda r: float(r['score']), reverse=True)
    report={"config":asdict(cfg),"feature_count":len(feats),"top":out_rows[:50],"all":out_rows,"leakage_note":"uses reward labels for offline feature selection; selected subsets need walk-forward validation"}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Candidate feature stability audit')
    p.add_argument('--input-jsonl',required=True); p.add_argument('--output',required=True)
    p.add_argument('--feature-prefixes',default=CandidateFeatureStabilityCfg.feature_prefixes)
    p.add_argument('--min-rows',type=int,default=CandidateFeatureStabilityCfg.min_rows)
    p.add_argument('--quantile',type=float,default=CandidateFeatureStabilityCfg.quantile)
    return p.parse_args()


def main()->None:
    print(json.dumps(run(CandidateFeatureStabilityCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))

if __name__=='__main__': main()
