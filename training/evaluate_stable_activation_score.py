"""Evaluate transparent stable-feature activation score thresholds."""
from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

NUM_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 /()_%.-]*?):\s*(-?\d+(?:\.\d+)?)\s*$")


def keyify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def parse_features(prompt: str) -> dict[str, float]:
    out={}
    for line in str(prompt).splitlines():
        m=NUM_RE.match(line.strip())
        if m:
            out[keyify(m.group(1))]=float(m.group(2))
    return out


def parse_target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row['target'])


def fit_scaler(rows: list[dict[str, Any]], features: list[str]) -> dict[str, dict[str, float]]:
    vals=[parse_features(r['prompt']) for r in rows]
    out={}
    for f in features:
        xs=[v[f] for v in vals if f in v]
        out[f]={"mean":mean(xs),"std":pstdev(xs) or 1.0}
    return out


def score_row(row: dict[str, Any], features: list[str], scaler: dict[str, dict[str,float]], weights: dict[str,float]) -> float:
    vals=parse_features(row['prompt'])
    s=0.0; n=0.0
    for f in features:
        if f not in vals: continue
        z=(vals[f]-scaler[f]['mean'])/scaler[f]['std']
        w=weights.get(f,1.0)
        s += w*z; n += abs(w)
    return s/max(1e-9,n)


def metrics(rows: list[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, Any]:
    pred=[s>=threshold for s in scores]
    labels=[parse_target(r)['decision']=='ACTIVATE' for r in rows]
    rets=[float(r.get('trade_ret_pct',0.0)) for r in rows]
    pred_ret=sum(r for r,p in zip(rets,pred) if p)
    oracle=sum(r for r,y in zip(rets,labels) if y)
    all_ret=sum(rets)
    tp=sum(p and y for p,y in zip(pred,labels)); fp=sum(p and not y for p,y in zip(pred,labels)); fn=sum((not p) and y for p,y in zip(pred,labels)); tn=sum((not p) and (not y) for p,y in zip(pred,labels))
    return {"threshold":threshold,"pred_sum_ret_pct":pred_ret,"oracle_sum_ret_pct":oracle,"all_activate_ret_pct":all_ret,"pred_activations":sum(pred),"oracle_activations":sum(labels),"tp":tp,"fp":fp,"fn":fn,"tn":tn}


def run(train_jsonl: str, val_jsonl: str, test_jsonl: str, selection_report: str, output: str, min_val_trades: int) -> dict[str, Any]:
    train=load_jsonl(train_jsonl); val=load_jsonl(val_jsonl); test=load_jsonl(test_jsonl)
    rep=json.load(open(selection_report))
    features=list(rep['selected'])
    scaler=fit_scaler(train,features)
    # Weight by the weaker train/val AUC edge; all selected directions are positive by construction.
    weights={f:min(rep['selected_stats'][f]['train']['edge'],rep['selected_stats'][f]['val']['edge']) for f in features}
    scores={
        'train':[score_row(r,features,scaler,weights) for r in train],
        'val':[score_row(r,features,scaler,weights) for r in val],
        'test':[score_row(r,features,scaler,weights) for r in test],
    }
    candidates=sorted(set(scores['train']+scores['val']))
    thresholds=[min(candidates)-1e-9]+[(a+b)/2 for a,b in zip(candidates,candidates[1:])]+[max(candidates)+1e-9]
    val_trials=[]
    for th in thresholds:
        m=metrics(val,scores['val'],th)
        if m['pred_activations'] >= min_val_trades:
            val_trials.append(m)
    best=max(val_trials, key=lambda m:(m['pred_sum_ret_pct'], -m['pred_activations'])) if val_trials else metrics(val,scores['val'],0.0)
    th=best['threshold']
    report={"features":features,"weights":weights,"scaler":scaler,"selected_threshold_from_val":th,"min_val_trades":min_val_trades,"train":metrics(train,scores['train'],th),"val":metrics(val,scores['val'],th),"test":metrics(test,scores['test'],th),"top_val_trials":sorted(val_trials,key=lambda m:m['pred_sum_ret_pct'],reverse=True)[:20]}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-jsonl',required=True); ap.add_argument('--val-jsonl',required=True); ap.add_argument('--test-jsonl',required=True)
    ap.add_argument('--selection-report',required=True); ap.add_argument('--output',required=True); ap.add_argument('--min-val-trades',type=int,default=8)
    args=ap.parse_args()
    print(json.dumps(run(**vars(args)),indent=2,ensure_ascii=False))

if __name__=='__main__': main()
