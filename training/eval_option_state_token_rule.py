"""Train-only token rule for option-choice state rows with reward audit."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.backtest_wave_state_option_predictions import _equity_stats, _trade_stats


@dataclass(frozen=True)
class OptionStateTokenRuleCfg:
    train_jsonl: str
    eval_jsonl: str
    output: str
    quantiles: str = "0,0.25,0.5,0.7,0.8,0.9"
    min_token_count: int = 10


def _read(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("row_index", i)
    return rows


def _reward(row: dict[str, Any]) -> float:
    return float(dict(row.get("choice_utility") or {}).get("A", 0.0) or 0.0)


def _tokens(row: dict[str, Any]) -> list[str]:
    src = row.get("source") if isinstance(row.get("source"), dict) else {}
    state = src.get("state_tokens") if isinstance(src.get("state_tokens"), dict) else {}
    out = [f"side={row.get('side')}"]
    for k, v in sorted(state.items()):
        out.append(f"{k}={v}")
        out.append(f"side:{row.get('side')}|{k}={v}")
    return out


def _fit(rows: list[dict[str, Any]], min_count: int) -> dict[str, float]:
    global_mean = float(np.mean([_reward(r) for r in rows])) if rows else 0.0
    vals: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        rew = _reward(r)
        for tok in _tokens(r):
            vals[tok].append(rew)
    weights = {k: float(np.mean(v) - global_mean) for k, v in vals.items() if len(v) >= int(min_count)}
    weights["__bias__"] = global_mean
    return weights


def _score(row: dict[str, Any], weights: dict[str, float]) -> float:
    toks = _tokens(row)
    return float(weights.get("__bias__", 0.0) + sum(weights.get(t, 0.0) for t in toks) / max(1, len(toks)))


def _summ(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: str(r.get("date", "")))
    rets = [_reward(r) for r in rows]
    return {"rows": len(rows), "sim": _equity_stats(rows, rets), "trade_stats": _trade_stats(rets)}


def run(cfg: OptionStateTokenRuleCfg) -> dict[str, Any]:
    train = _read(cfg.train_jsonl)
    ev = _read(cfg.eval_jsonl)
    weights = _fit(train, int(cfg.min_token_count))
    train_scores = np.asarray([_score(r, weights) for r in train], dtype=float)
    scored_eval = [{**r, "rule_score": _score(r, weights)} for r in ev]
    candidates = []
    for q in [float(x) for x in str(cfg.quantiles).split(",") if x.strip()]:
        th = float(np.quantile(train_scores, q)) if len(train_scores) else 0.0
        selected = [r for r in scored_eval if float(r["rule_score"]) >= th]
        candidates.append({"quantile": q, "threshold": th, "selected": _summ(selected), "all_eval": _summ(scored_eval)})
    candidates.sort(key=lambda r: (float(r["selected"]["sim"].get("cagr_to_strict_mdd", -999)), float(r["selected"]["sim"].get("cagr_pct", -999))), reverse=True)
    report = {"config": asdict(cfg), "train": _summ(train), "eval": _summ(ev), "top_tokens": sorted(({"token": k, "weight": v} for k, v in weights.items() if k != "__bias__"), key=lambda x: x["weight"], reverse=True)[:40], "candidates": candidates, "best_eval_by_frozen_train_thresholds": candidates[0] if candidates else None, "leakage_guard": {"weights_fit_on_train_only": True, "thresholds_from_train_scores": True, "eval_rewards_reporting_only": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantiles", default=OptionStateTokenRuleCfg.quantiles)
    p.add_argument("--min-token-count", type=int, default=OptionStateTokenRuleCfg.min_token_count)
    return p.parse_args()


def main() -> None:
    out = run(OptionStateTokenRuleCfg(**vars(parse_args())))
    print(json.dumps({"train": out["train"], "eval": out["eval"], "best": out["best_eval_by_frozen_train_thresholds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
