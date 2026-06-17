"""Build compact Kimchi-flow activation SFT data from stable train/val features.

This transforms existing activation rows into shorter prompts using only numeric
features whose activation AUC has the same direction on train and validation.
Test rows are transformed with the selected train/val feature set but are never
used to choose the feature list.
"""
from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
from statistics import mean
from typing import Any

NUM_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 /()_%.-]*?):\s*(-?\d+(?:\.\d+)?)\s*$")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def keyify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()


def parse_features(prompt: str) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for line in str(prompt).splitlines():
        m = NUM_RE.match(line.strip())
        if not m:
            continue
        label = m.group(1).strip()
        out[keyify(label)] = (label, float(m.group(2)))
    return out


def parse_target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["target"])


def auc_score(values: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(values, labels), key=lambda x: x[0])
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0; rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg * sum(lbl for _, lbl in pairs[i:j])
        rank += j - i; i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def split_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    feats = [parse_features(r["prompt"]) for r in rows]
    labels = [1 if parse_target(r)["decision"] == "ACTIVATE" else 0 for r in rows]
    keys = sorted(set().union(*(f.keys() for f in feats)))
    stats: dict[str, dict[str, Any]] = {}
    for k in keys:
        vals=[]; y=[]; display=k
        for f,lbl in zip(feats, labels):
            if k in f:
                display=f[k][0]; vals.append(f[k][1]); y.append(lbl)
        if len(vals) < max(10, len(rows)//2):
            continue
        auc=auc_score(vals,y)
        pos=[v for v,l in zip(vals,y) if l]
        neg=[v for v,l in zip(vals,y) if not l]
        stats[k]={"display":display,"auc":auc,"direction":1 if auc>=0.5 else -1,"edge":abs(auc-0.5),"mean_activate":mean(pos) if pos else None,"mean_abstain":mean(neg) if neg else None,"coverage":len(vals)}
    return stats


def select_features(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], *, min_edge: float, max_features: int) -> tuple[list[str], dict[str, Any]]:
    tr=split_stats(train_rows); va=split_stats(val_rows)
    candidates=[]
    for k in sorted(set(tr) & set(va)):
        if tr[k]["direction"] != va[k]["direction"]:
            continue
        if tr[k]["edge"] < min_edge or va[k]["edge"] < min_edge:
            continue
        score=min(tr[k]["edge"], va[k]["edge"]) + 0.25*(tr[k]["edge"]+va[k]["edge"])
        candidates.append((score,k))
    candidates.sort(reverse=True)
    selected=[k for _,k in candidates[:max_features]]
    report={"min_edge":min_edge,"max_features":max_features,"selected":selected,"selected_stats":{k:{"train":tr[k],"val":va[k]} for k in selected},"rejected_count":len(set(tr)&set(va))-len(candidates)}
    return selected, report


def stable_prompt(row: dict[str, Any], selected: list[str], labels: dict[str, str]) -> str:
    feats=parse_features(row["prompt"])
    lines=[]
    for k in selected:
        if k in feats:
            lines.append(f"{labels.get(k, feats[k][0])}: {feats[k][1]:.6g}")
    return "\n".join([
        "You are a BTCUSDT Kimchi-flow activation classifier.",
        "Use only these selected past-only features. They were selected on train/validation stability, not on test.",
        "Return exactly compact JSON with keys: regime, decision, side, quality, confidence.",
        "Allowed decisions: ACTIVATE or ABSTAIN. If ABSTAIN, side must be NONE.",
        "Prefer ABSTAIN unless the selected evidence supports a high-quality Kimchi-flow activation.",
        "",
        "Selected past-only features:",
        *lines,
    ])


def transform(rows: list[dict[str, Any]], selected: list[str], labels: dict[str, str], tag: str) -> list[dict[str, Any]]:
    out=[]
    for r in rows:
        nr=dict(r)
        nr["task"]="kimchi_flow_activation_stable_sft"
        nr["prompt"] = stable_prompt(r, selected, labels)
        nr["source_task"] = r.get("task")
        nr["feature_selection"] = {"method":"train_val_same_direction_auc", "tag": tag, "selected_features": selected}
        out.append(nr)
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-jsonl', required=True)
    ap.add_argument('--val-jsonl', required=True)
    ap.add_argument('--test-jsonl', required=True)
    ap.add_argument('--output-prefix', required=True)
    ap.add_argument('--report-output', required=True)
    ap.add_argument('--min-edge', type=float, default=0.07)
    ap.add_argument('--max-features', type=int, default=10)
    args=ap.parse_args()
    train=load_jsonl(args.train_jsonl); val=load_jsonl(args.val_jsonl); test=load_jsonl(args.test_jsonl)
    selected, report=select_features(train,val,min_edge=args.min_edge,max_features=args.max_features)
    if not selected:
        raise SystemExit('no stable features selected; lower --min-edge')
    labels={k: report['selected_stats'][k]['train']['display'] for k in selected}
    prefix=args.output_prefix
    write_jsonl(prefix+'_train.jsonl', transform(train,selected,labels,'train_val'))
    write_jsonl(prefix+'_val.jsonl', transform(val,selected,labels,'train_val'))
    write_jsonl(prefix+'_test.jsonl', transform(test,selected,labels,'train_val'))
    full={**report,"inputs":{"train":args.train_jsonl,"val":args.val_jsonl,"test":args.test_jsonl},"outputs":{"train":prefix+'_train.jsonl',"val":prefix+'_val.jsonl',"test":prefix+'_test.jsonl'}}
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(full, indent=2, ensure_ascii=False))
    print(json.dumps(full, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
