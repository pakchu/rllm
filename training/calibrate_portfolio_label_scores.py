"""Calibrate additive biases for portfolio label log-prob scores."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("LONG", "SHORT", "NO_TRADE")


@dataclass(frozen=True)
class CalibrateCfg:
    calibration_predictions: str
    apply_predictions: str
    output_predictions: str
    report_output: str
    grid: str = "-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0,2.5,3.0"
    objective: str = "macro_f1"


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(',') if x.strip()]


def _load(path: str) -> list[dict[str, Any]]:
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _predict(row: dict[str, Any], bias: dict[str, float]) -> str:
    scores=row.get('scores') or {}
    vals={lab: float(scores.get(lab, -1e9)) + float(bias.get(lab,0.0)) for lab in LABELS}
    return max(LABELS, key=lambda lab: vals[lab])


def _metrics(rows: list[dict[str, Any]], bias: dict[str, float]) -> dict[str, Any]:
    cm={(t,p):0 for t in LABELS for p in LABELS}
    for r in rows:
        t=str(r.get('target'))
        p=_predict(r,bias)
        if t not in LABELS:
            t='NO_TRADE'
        cm[(t,p)]=cm.get((t,p),0)+1
    total=sum(cm.values())
    acc=sum(cm[(l,l)] for l in LABELS)/max(1,total)
    f1s=[]
    recalls=[]
    precisions=[]
    for lab in LABELS:
        tp=cm[(lab,lab)]
        fp=sum(cm[(t,lab)] for t in LABELS if t!=lab)
        fn=sum(cm[(lab,p)] for p in LABELS if p!=lab)
        prec=tp/max(1,tp+fp)
        rec=tp/max(1,tp+fn)
        f1=2*prec*rec/max(1e-12,prec+rec)
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    pred_counts={lab:sum(cm[(t,lab)] for t in LABELS) for lab in LABELS}
    target_counts={lab:sum(cm[(lab,p)] for p in LABELS) for lab in LABELS}
    return {'accuracy':acc,'macro_f1':sum(f1s)/len(f1s),'macro_recall':sum(recalls)/len(recalls),'macro_precision':sum(precisions)/len(precisions),'pred_counts':pred_counts,'target_counts':target_counts,'confusion':{f'target={t}|pred={p}':cm[(t,p)] for t in LABELS for p in LABELS if cm[(t,p)]}}


def run(cfg: CalibrateCfg) -> dict[str, Any]:
    cal=_load(cfg.calibration_predictions)
    grid=_parse_floats(cfg.grid)
    best=None
    rows=[]
    # Keep LONG as reference 0. Sweep SHORT and NO_TRADE additive offsets.
    for short_b in grid:
        for nt_b in grid:
            bias={'LONG':0.0,'SHORT':short_b,'NO_TRADE':nt_b}
            m=_metrics(cal,bias)
            score=float(m.get(cfg.objective, m['macro_f1']))
            row={'bias':bias,'metrics':m,'score':score}
            rows.append(row)
            if best is None or score>best['score']:
                best=row
    assert best is not None
    apply_rows=_load(cfg.apply_predictions)
    out=[]
    for r in apply_rows:
        pred=_predict(r,best['bias'])
        nr=dict(r)
        nr['raw_prediction']=r.get('prediction')
        nr['prediction']=pred
        nr['calibration_bias']=best['bias']
        out.append(nr)
    Path(cfg.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_predictions).write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in out)+'\n')
    report={'config':cfg.__dict__,'best':best,'apply_metrics':_metrics(out, {'LONG':0.0,'SHORT':0.0,'NO_TRADE':0.0}), 'top10':sorted(rows,key=lambda r:r['score'],reverse=True)[:10]}
    Path(cfg.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.report_output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Calibrate portfolio label scores')
    p.add_argument('--calibration-predictions',required=True)
    p.add_argument('--apply-predictions',required=True)
    p.add_argument('--output-predictions',required=True)
    p.add_argument('--report-output',required=True)
    p.add_argument('--grid',default=CalibrateCfg.grid)
    p.add_argument('--objective',choices=['accuracy','macro_f1','macro_recall','macro_precision'],default='macro_f1')
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(CalibrateCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
