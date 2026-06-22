"""Audit whether categorical text-state token combinations contain stable train-to-eval edge.

This is intentionally model-free: it checks if past-only token combinations can select
LONG/SHORT using train reward_audit and remain profitable on eval. If this fails, SFT
on the same text surface is learning label noise rather than a stable alpha.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AuditCfg:
    input_jsonl: str
    output: str
    predictions_output: str = ""
    max_combo_size: int = 2
    min_train_count: int = 20
    min_mean_utility: float = 0.02
    min_side_gap: float = 0.01
    max_rules_per_row: int = 1


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _side_utils(row: dict[str, Any]) -> tuple[float, float]:
    audit = row.get("reward_audit", {}) if isinstance(row.get("reward_audit"), dict) else {}
    return (
        float(audit.get("LONG", {}).get("utility", 0.0)),
        float(audit.get("SHORT", {}).get("utility", 0.0)),
    )


def _features(row: dict[str, Any]) -> list[tuple[str, str]]:
    toks = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
    return sorted((str(k), str(v)) for k, v in toks.items())


def _combos(feats: list[tuple[str, str]], max_combo_size: int) -> Iterable[tuple[tuple[str, str], ...]]:
    for k in range(1, int(max_combo_size) + 1):
        yield from itertools.combinations(feats, k)


def _rule_key(combo: tuple[tuple[str, str], ...]) -> str:
    return " & ".join(f"{k}={v}" for k, v in combo)


def _build_rules(train: list[dict[str, Any]], cfg: AuditCfg) -> dict[tuple[tuple[str, str], ...], dict[str, Any]]:
    stats: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in train:
        long_u, short_u = _side_utils(row)
        for combo in _combos(_features(row), cfg.max_combo_size):
            s = stats[combo]
            s[0] += 1.0
            s[1] += long_u
            s[2] += short_u
    rules: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for combo, (n, long_sum, short_sum) in stats.items():
        n_i = int(n)
        if n_i < cfg.min_train_count:
            continue
        long_mean = long_sum / n
        short_mean = short_sum / n
        side = "LONG" if long_mean >= short_mean else "SHORT"
        best = max(long_mean, short_mean)
        gap = abs(long_mean - short_mean)
        if best < cfg.min_mean_utility or gap < cfg.min_side_gap:
            continue
        rules[combo] = {"rule": _rule_key(combo), "n_train": n_i, "side": side, "mean_utility": best, "side_gap": gap, "long_mean": long_mean, "short_mean": short_mean}
    return rules


def _predict(row: dict[str, Any], rules: dict[tuple[tuple[str, str], ...], dict[str, Any]], cfg: AuditCfg) -> tuple[str, list[dict[str, Any]]]:
    matches = [rules[c] for c in _combos(_features(row), cfg.max_combo_size) if c in rules]
    matches.sort(key=lambda r: (float(r["mean_utility"]), float(r["side_gap"]), int(r["n_train"])), reverse=True)
    matches = matches[: max(1, int(cfg.max_rules_per_row))]
    if not matches:
        return "NO_TRADE", []
    side_votes: dict[str, float] = defaultdict(float)
    for r in matches:
        side_votes[str(r["side"])] += float(r["mean_utility"]) * math.sqrt(float(r["n_train"]))
    pred = max(side_votes, key=side_votes.get)
    return pred, matches


def _classification(rows: list[dict[str, Any]], preds: list[str]) -> dict[str, Any]:
    conf: dict[str, int] = {}
    correct = 0
    for row, pred in zip(rows, preds):
        tgt = str(row.get("target"))
        correct += int(tgt == pred)
        conf[f"target={tgt}|pred={pred}"] = conf.get(f"target={tgt}|pred={pred}", 0) + 1
    return {"accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(conf.items()))}


def run(cfg: AuditCfg) -> dict[str, Any]:
    rows = _load(cfg.input_jsonl)
    train = [r for r in rows if r.get("split") == "train"]
    eval_rows = [r for r in rows if r.get("split") == "eval"]
    rules = _build_rules(train, cfg)
    pred_rows: list[dict[str, Any]] = []
    preds: list[str] = []
    eval_util = []
    for row in eval_rows:
        pred, matches = _predict(row, rules, cfg)
        preds.append(pred)
        long_u, short_u = _side_utils(row)
        util = 0.0 if pred == "NO_TRADE" else (long_u if pred == "LONG" else short_u)
        eval_util.append(util)
        pred_rows.append({
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prediction": pred,
            "target": row.get("target"),
            "candidate": row.get("candidate", {}),
            "matched_rules": matches,
            "eval_utility": util,
        })
    counts: dict[str, int] = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    nonflat = [u for p, u in zip(preds, eval_util) if p != "NO_TRADE"]
    report = {
        "config": cfg.__dict__,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "rules": len(rules),
        "prediction_counts": dict(sorted(counts.items())),
        "classification": _classification(eval_rows, preds),
        "eval_utility": {
            "mean_all": sum(eval_util) / max(1, len(eval_util)),
            "mean_traded": sum(nonflat) / max(1, len(nonflat)),
            "traded_rows": len(nonflat),
        },
        "top_rules": sorted(rules.values(), key=lambda r: (float(r["mean_utility"]), int(r["n_train"])), reverse=True)[:30],
        "leakage_guard": "Rules are fit only on split=train reward_audit; eval predictions use eval state_tokens only.",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if cfg.predictions_output:
        Path(cfg.predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.predictions_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pred_rows) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit train-to-eval token combo edge")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", default="")
    p.add_argument("--max-combo-size", type=int, default=2)
    p.add_argument("--min-train-count", type=int, default=20)
    p.add_argument("--min-mean-utility", type=float, default=0.02)
    p.add_argument("--min-side-gap", type=float, default=0.01)
    p.add_argument("--max-rules-per-row", type=int, default=1)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(AuditCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
