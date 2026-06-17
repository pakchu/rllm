"""Diagnose whether activation prompts contain leak-safe separable signal."""
from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
from statistics import mean
from typing import Any

NUM_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /()_%.-]*?):\s*(-?\d+(?:\.\d+)?)\s*$")


def load_rows(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def parse_target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["target"])


def parse_features(prompt: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in str(prompt).splitlines():
        m = NUM_RE.match(line.strip())
        if not m:
            continue
        key = re.sub(r"[^A-Za-z0-9]+", "_", m.group(1).strip()).strip("_").lower()
        out[key] = float(m.group(2))
    return out


def auc_score(values: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(values, labels), key=lambda x: x[0])
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(lbl for _, lbl in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize(path: str) -> dict[str, Any]:
    rows = load_rows(path)
    feats = [parse_features(r["prompt"]) for r in rows]
    labels = [1 if parse_target(r)["decision"] == "ACTIVATE" else 0 for r in rows]
    rets = [float(r.get("trade_ret_pct", 0.0)) for r in rows]
    keys = sorted(set().union(*(f.keys() for f in feats)))
    records = []
    for k in keys:
        vals = [f.get(k, float("nan")) for f in feats]
        ok = [i for i, v in enumerate(vals) if not math.isnan(v)]
        if len(ok) < max(10, len(rows)//2):
            continue
        v = [vals[i] for i in ok]
        y = [labels[i] for i in ok]
        r = [rets[i] for i in ok]
        auc = auc_score(v, y)
        pos_vals = [vals[i] for i in ok if labels[i]]
        neg_vals = [vals[i] for i in ok if not labels[i]]
        # signed return correlation proxy.
        mv = mean(v); mr = mean(r)
        cov = sum((a-mv)*(b-mr) for a,b in zip(v,r))
        vv = sum((a-mv)**2 for a in v); rr = sum((b-mr)**2 for b in r)
        corr = cov / math.sqrt(vv*rr) if vv and rr else 0.0
        records.append({
            "feature": k,
            "auc_activate": auc,
            "auc_edge": abs(auc-0.5) if not math.isnan(auc) else 0.0,
            "mean_activate": mean(pos_vals) if pos_vals else None,
            "mean_abstain": mean(neg_vals) if neg_vals else None,
            "return_corr": corr,
            "n": len(ok),
        })
    records.sort(key=lambda x: (x["auc_edge"], abs(x["return_corr"])), reverse=True)
    return {
        "path": path,
        "rows": len(rows),
        "activate": sum(labels),
        "abstain": len(labels)-sum(labels),
        "sum_ret_pct": sum(rets),
        "oracle_ret_pct": sum(r for r,y in zip(rets,labels) if y),
        "top_auc_features": records[:20],
        "top_return_corr_features": sorted(records, key=lambda x: abs(x["return_corr"]), reverse=True)[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report = {"splits": [summarize(p) for p in args.splits]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
