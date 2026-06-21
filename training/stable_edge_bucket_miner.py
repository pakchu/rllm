"""Mine stable symbolic edge buckets without fitting on eval.

This scans action-candidate JSONL files and computes realized label statistics for
symbolic buckets (action family/side/horizon plus regime/book tokens).  It is a
research tool for finding positive edge sources before training another model.

Leakage note: it uses future action_audit labels for offline research on the
specified selection windows. Do not select buckets on the final eval window.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def read_jsonl(path: str | Path, *, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = str(r.get("date", ""))
        if start and d < start:
            continue
        if end and d >= end:
            continue
        rows.append(r)
    return rows


def prompt_tokens(prompt: str) -> set[str]:
    toks: set[str] = set()
    for line in str(prompt).splitlines():
        if line.startswith("Regime tokens:"):
            toks.update(p.strip() for p in line.split(":", 1)[1].split(";") if p.strip())
        elif line.startswith("Candidate book tokens:"):
            for p in [x.strip() for x in line.split(":", 1)[1].split(";") if x.strip()]:
                fam, *rest = p.split(":")
                toks.add("book_" + fam)
                if rest:
                    toks.add("book_" + fam + ":" + rest[0])
    return toks


def action_tokens(row: dict[str, Any]) -> set[str]:
    a = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    side = str(a.get("side", "NONE")).upper()
    fam = str(a.get("family", "UNKNOWN"))
    hold = int(a.get("hold_bars", 0) or 0)
    horizon = "h72" if hold == 72 else "h144" if hold == 144 else "h288" if hold == 288 else "h432" if hold == 432 else f"h{hold}"
    return {f"side={side}", f"family={fam}", f"hold={hold}", f"horizon={horizon}", f"family={fam}|side={side}", f"family={fam}|hold={hold}", f"side={side}|hold={hold}", f"family={fam}|side={side}|hold={hold}"}


def bucket_tokens(row: dict[str, Any], *, pair_tokens: bool) -> set[str]:
    a = action_tokens(row)
    r = prompt_tokens(str(row.get("prompt", "")))
    toks = set(a)
    toks.update("regime:" + t for t in r if not t.startswith("book_"))
    toks.update("book:" + t.removeprefix("book_") for t in r if t.startswith("book_"))
    # Action x regime/book interactions are usually the actual edge surface.
    core_actions = [t for t in a if t.startswith("family=") or t.startswith("side=") or t.startswith("hold=")]
    for at in core_actions:
        for rt in r:
            toks.add(f"{at}&{rt}")
    if pair_tokens:
        rs = sorted(r)
        for i, x in enumerate(rs):
            for y in rs[i + 1 :]:
                toks.add(f"pair:{x}&{y}")
    return toks


def row_value(row: dict[str, Any]) -> dict[str, float]:
    au = row.get("action_audit", {}) if isinstance(row.get("action_audit"), dict) else {}
    net = float(au.get("net_return", 0.0) or 0.0)
    mae = max(0.0, float(au.get("mae", 0.0) or 0.0))
    mfe = max(0.0, float(au.get("mfe", 0.0) or 0.0))
    return {"net": net, "mae": mae, "mfe": mfe, "safe": net - mae, "tail": net - 1.5 * mae}


def summarize(vals: list[dict[str, float]]) -> dict[str, float]:
    nets = [v["net"] for v in vals]
    maes = [v["mae"] for v in vals]
    tails = [v["tail"] for v in vals]
    n = len(vals)
    mu = mean(nets) if n else 0.0
    sd = pstdev(nets) if n > 1 else 0.0
    t = mu / (sd / math.sqrt(n)) if n > 1 and sd > 1e-12 else 0.0
    return {
        "n": n,
        "mean_net": mu,
        "sum_net": sum(nets),
        "win_rate": sum(1 for x in nets if x > 0) / n if n else 0.0,
        "mean_mae": mean(maes) if n else 0.0,
        "mean_tail": mean(tails) if n else 0.0,
        "t_stat": t,
    }


def scan(rows: list[dict[str, Any]], *, min_count: int, pair_tokens: bool) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        v = row_value(r)
        for tok in bucket_tokens(r, pair_tokens=pair_tokens):
            buckets[tok].append(v)
    return {k: summarize(vs) for k, vs in buckets.items() if len(vs) >= int(min_count)}


def stable_select(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], *, min_mean_net: float, min_t: float, min_tail: float) -> list[dict[str, Any]]:
    out = []
    for tok in sorted(set(a) & set(b)):
        sa, sb = a[tok], b[tok]
        if sa["mean_net"] >= min_mean_net and sb["mean_net"] >= min_mean_net and sa["mean_tail"] >= min_tail and sb["mean_tail"] >= min_tail and sa["t_stat"] >= min_t and sb["t_stat"] >= min_t:
            score = min(sa["mean_net"], sb["mean_net"]) * math.sqrt(min(sa["n"], sb["n"]))
            out.append({"bucket": tok, "score": score, "select_a": sa, "select_b": sb})
    out.sort(key=lambda r: (r["score"], min(r["select_a"]["n"], r["select_b"]["n"])), reverse=True)
    return out


def eval_buckets(rows: list[dict[str, Any]], buckets: list[str], *, pair_tokens: bool) -> dict[str, dict[str, float]]:
    wanted = set(buckets)
    hits: dict[str, list[dict[str, float]]] = defaultdict(list)
    union_hits: list[dict[str, float]] = []
    for r in rows:
        toks = bucket_tokens(r, pair_tokens=pair_tokens)
        matched = wanted & toks
        if matched:
            v = row_value(r)
            union_hits.append(v)
            for m in matched:
                hits[m].append(v)
    out = {k: summarize(vs) for k, vs in hits.items()}
    out["__UNION__"] = summarize(union_hits)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine stable symbolic edge buckets")
    p.add_argument("--select-a-jsonl", required=True)
    p.add_argument("--select-b-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--select-a-start", default="")
    p.add_argument("--select-a-end", default="")
    p.add_argument("--select-b-start", default="")
    p.add_argument("--select-b-end", default="")
    p.add_argument("--eval-start", default="")
    p.add_argument("--eval-end", default="")
    p.add_argument("--min-count", type=int, default=40)
    p.add_argument("--min-mean-net", type=float, default=0.001)
    p.add_argument("--min-t", type=float, default=1.0)
    p.add_argument("--min-tail", type=float, default=-0.004)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--pair-tokens", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows_a = read_jsonl(a.select_a_jsonl, start=a.select_a_start or None, end=a.select_a_end or None)
    rows_b = read_jsonl(a.select_b_jsonl, start=a.select_b_start or None, end=a.select_b_end or None)
    rows_e = read_jsonl(a.eval_jsonl, start=a.eval_start or None, end=a.eval_end or None)
    scan_a = scan(rows_a, min_count=a.min_count, pair_tokens=a.pair_tokens)
    scan_b = scan(rows_b, min_count=a.min_count, pair_tokens=a.pair_tokens)
    selected = stable_select(scan_a, scan_b, min_mean_net=a.min_mean_net, min_t=a.min_t, min_tail=a.min_tail)[: int(a.top_k)]
    eval_stats = eval_buckets(rows_e, [r["bucket"] for r in selected], pair_tokens=a.pair_tokens)
    report = {
        "config": vars(a),
        "rows": {"select_a": len(rows_a), "select_b": len(rows_b), "eval": len(rows_e)},
        "selected": selected,
        "eval": eval_stats,
        "leakage_guard": {"eval_not_used_for_bucket_selection": True, "uses_future_labels_for_offline_research_only": True},
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"selected": len(selected), "eval_union": eval_stats.get("__UNION__", {}), "top": selected[:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
