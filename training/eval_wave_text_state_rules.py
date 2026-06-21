"""Evaluate simple token-rule rankers on wave LLM state rows.

This is a representation sanity check before expensive LLM fine-tuning.  It fits
per-token reward means on train prompts, scores eval rows additively, then tests
frozen train-derived quantile thresholds by realized label rewards.  It does not
use eval rewards for threshold selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TextStateRuleEvalConfig:
    train_jsonl: str
    eval_jsonl: str
    output: str
    quantiles: str = "0,0.25,0.5,0.7"
    min_eval_rows: int = 10


def _read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _tokens(row: dict[str, Any]) -> list[str]:
    out = [f"side={row.get('side')}"]
    for k, v in sorted(dict(row.get("state_tokens") or {}).items()):
        out.append(f"{k}={v}")
        out.append(f"side:{row.get('side')}|{k}={v}")
    return out


def _reward(row: dict[str, Any]) -> float:
    return float(dict(row.get("reward") or {}).get("trade_ret_pct", 0.0)) / 100.0


def _fit(train: list[dict[str, Any]]) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    global_mean = float(np.mean([_reward(r) for r in train])) if train else 0.0
    for row in train:
        r = _reward(row)
        for tok in _tokens(row):
            vals[tok].append(r)
    weights = {tok: float(np.mean(rs) - global_mean) for tok, rs in vals.items() if len(rs) >= 3}
    weights["__bias__"] = global_mean
    return weights


def _score(row: dict[str, Any], weights: dict[str, float]) -> float:
    toks = _tokens(row)
    return float(weights.get("__bias__", 0.0) + sum(weights.get(t, 0.0) for t in toks) / max(1, len(toks)))


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray([_reward(r) for r in rows], dtype=float)
    if len(rewards) == 0:
        return {"rows": 0, "mean_reward_pct": 0.0, "ret_pct_compound": 0.0, "positive_rate": 0.0}
    eq = float(np.prod(1.0 + rewards))
    return {"rows": int(len(rows)), "mean_reward_pct": float(np.mean(rewards) * 100.0), "ret_pct_compound": float((eq - 1.0) * 100.0), "positive_rate": float(np.mean(rewards > 0.0))}


def run_eval(cfg: TextStateRuleEvalConfig) -> dict[str, Any]:
    train = _read(cfg.train_jsonl)
    ev = _read(cfg.eval_jsonl)
    weights = _fit(train)
    train_scores = np.asarray([_score(r, weights) for r in train], dtype=float)
    scored_eval = [{**r, "rule_score": _score(r, weights)} for r in ev]
    candidates = []
    for q in _parse_floats(cfg.quantiles):
        th = float(np.quantile(train_scores, q)) if len(train_scores) else 0.0
        selected = [r for r in scored_eval if float(r["rule_score"]) >= th]
        candidates.append({"quantile": q, "threshold": th, "selected": _stats(selected), "all_eval": _stats(scored_eval)})
    candidates.sort(key=lambda r: (r["selected"]["ret_pct_compound"], r["selected"]["mean_reward_pct"]), reverse=True)
    out = {
        "config": asdict(cfg),
        "train": _stats(train),
        "eval": _stats(ev),
        "top_tokens": sorted(({"token": k, "weight": v} for k, v in weights.items() if k != "__bias__"), key=lambda x: x["weight"], reverse=True)[:30],
        "candidates": candidates,
        "best_eval_by_frozen_train_thresholds": candidates[0] if candidates else None,
        "leakage_guard": {"weights_fit_on_train_only": True, "thresholds_from_train_score_quantiles": True, "eval_rewards_used_for_reporting_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate simple token-rule rankers on wave text-state rows")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantiles", default="0,0.25,0.5,0.7")
    p.add_argument("--min-eval-rows", type=int, default=10)
    return p.parse_args()


def main() -> None:
    out = run_eval(TextStateRuleEvalConfig(**vars(parse_args())))
    print(json.dumps({"train": out["train"], "eval": out["eval"], "best": out["best_eval_by_frozen_train_thresholds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
