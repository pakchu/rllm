"""Lightweight baseline for text-state action-value rows.

This does not replace LLM training.  It checks whether categorical state tokens
contain any learnable signal before spending GPU time on SFT/RL.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"^- ([^:]+): (.+)$")


@dataclass(frozen=True)
class BaselineCfg:
    input_jsonl: str
    output: str
    positive_label: str = "TAKE"
    min_count: int = 8
    smoothing: float = 2.0
    top_k_tokens: int = 8


def _load(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokens(row: dict[str, Any]) -> list[str]:
    toks = []
    cand = row.get("candidate", {})
    toks.append(f"candidate.side={cand.get('side')}")
    toks.append(f"candidate.hold={cand.get('hold_bars')}")
    st = row.get("state_tokens")
    if isinstance(st, dict):
        toks.extend(f"state.{k}={v}" for k, v in sorted(st.items()))
        side = str(cand.get("side"))
        hold = str(cand.get("hold_bars"))
        toks.extend(f"cross.{k}={v}|side={side}" for k, v in sorted(st.items()) if k in {"candidate_alignment_short", "candidate_alignment_daily", "daily_context", "weekly_context", "volatility", "range_location"})
        toks.extend(f"cross.{k}={v}|hold={hold}" for k, v in sorted(st.items()) if k in {"daily_context", "weekly_context", "volatility"})
        return toks
    # Fallback parser for older rows.
    for line in str(row.get("prompt", "")).splitlines():
        m = TOKEN_RE.match(line)
        if m:
            toks.append(f"state.{m.group(1)}={m.group(2)}")
    return toks


def _fit(train: list[dict[str, Any]], cfg: BaselineCfg) -> dict[str, Any]:
    total_pos = sum(1 for r in train if r.get("target") == cfg.positive_label)
    prior = total_pos / max(1, len(train))
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in train:
        y = 1 if r.get("target") == cfg.positive_label else 0
        for t in set(_tokens(r)):
            counts[t][0] += y
            counts[t][1] += 1
    weights = {}
    for t, (pos, n) in counts.items():
        if n < int(cfg.min_count):
            continue
        p = (pos + float(cfg.smoothing) * prior) / (n + float(cfg.smoothing))
        weights[t] = math.log(max(1e-6, p) / max(1e-6, prior))
    return {"prior": prior, "weights": weights, "counts": counts}


def _score(row: dict[str, Any], model: dict[str, Any], cfg: BaselineCfg) -> float:
    toks = sorted(set(_tokens(row)), key=lambda t: abs(float(model["weights"].get(t, 0.0))), reverse=True)
    logit = math.log(max(1e-6, model["prior"]) / max(1e-6, 1.0 - model["prior"]))
    for t in toks[: int(cfg.top_k_tokens)]:
        logit += float(model["weights"].get(t, 0.0))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((np.sum(ranks[y == 1]) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _eval(rows: list[dict[str, Any]], model: dict[str, Any], cfg: BaselineCfg) -> dict[str, Any]:
    y = np.asarray([1 if r.get("target") == cfg.positive_label else 0 for r in rows], dtype=int)
    s = np.asarray([_score(r, model, cfg) for r in rows], dtype=float)
    out: dict[str, Any] = {"rows": len(rows), "positive_rate": float(np.mean(y)) if len(y) else 0.0, "auc": _auc(y, s) if len(y) else 0.5}
    for q in (0.80, 0.90, 0.95):
        if len(s) == 0:
            continue
        th = float(np.quantile(s, q))
        pred = s >= th
        tp = int(np.sum((y == 1) & pred))
        fp = int(np.sum((y == 0) & pred))
        out[f"top_{int((1-q)*100)}pct"] = {"threshold": th, "selected": int(np.sum(pred)), "precision": tp / max(1, tp + fp), "recall": tp / max(1, int(np.sum(y)))}
    return out


def run(cfg: BaselineCfg) -> dict[str, Any]:
    rows = _load(cfg.input_jsonl)
    train = [r for r in rows if r.get("split") == "train"]
    eval_rows = [r for r in rows if r.get("split") == "eval"]
    model = _fit(train, cfg)
    top_weights = sorted(model["weights"].items(), key=lambda kv: kv[1], reverse=True)[:30]
    bottom_weights = sorted(model["weights"].items(), key=lambda kv: kv[1])[:30]
    report = {
        "config": cfg.__dict__,
        "train": _eval(train, model, cfg),
        "eval": _eval(eval_rows, model, cfg),
        "learned_token_count": len(model["weights"]),
        "top_positive_tokens": top_weights,
        "top_negative_tokens": bottom_weights,
        "target_counts": dict(Counter(str(r.get("target")) for r in rows)),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate categorical text-state value baseline")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--positive-label", default="TAKE")
    p.add_argument("--min-count", type=int, default=8)
    p.add_argument("--smoothing", type=float, default=2.0)
    p.add_argument("--top-k-tokens", type=int, default=8)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(BaselineCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
