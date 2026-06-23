"""Evaluate simple memory baselines for side-map reliability SFT rows.

This tests whether the month-level reliability target is learnable before doing
an LLM fine-tune.  It predicts eval rows from prompt/source fields using only
train+val rows before eval.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SideMapReliabilityMemoryEvalCfg:
    input_jsonl: str
    output: str
    train_splits: str = "train,val"
    eval_split: str = "eval"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _target_label(row: dict[str, Any]) -> str:
    try:
        return str(json.loads(row.get("target", "{}")) .get("side_map", "unreliable"))
    except Exception:
        return "unreliable"


def _history_labels(row: dict[str, Any]) -> list[str]:
    labels = []
    for line in str(row.get("prompt", "")).splitlines():
        if " label=" in line:
            part = line.split(" label=", 1)[1].split()[0]
            labels.append(part)
    return labels


def _score_bucket(row: dict[str, Any]) -> str:
    src = row.get("source") if isinstance(row.get("source"), dict) else {}
    return str(src.get("prior_validation_score_bucket", "unknown"))



def _aux_signature(row: dict[str, Any]) -> str:
    src = row.get("source") if isinstance(row.get("source"), dict) else {}
    toks = src.get("prior_binance_aux_tokens") if isinstance(src.get("prior_binance_aux_tokens"), dict) else {}
    keys = ("prior_btc_premium_mean", "prior_btc_funding_mean", "prior_btc_funding_abs")
    return "|".join(f"{k}={toks.get(k, 'unknown')}" for k in keys)


def _combined_signature(row: dict[str, Any]) -> str:
    return f"score={_score_bucket(row)}|{_aux_signature(row)}"

def _majority(labels: list[str], default: str = "unreliable") -> str:
    if not labels:
        return default
    c = Counter(labels)
    return sorted(c.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]


def _predictors(train: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, str]:
    global_majority = _majority([_target_label(r) for r in train], "unreliable")
    hist = _history_labels(row)
    last = hist[-1] if hist else global_majority
    hist_majority = _majority(hist, global_majority)
    bucket = _score_bucket(row)
    by_bucket: dict[str, list[str]] = defaultdict(list)
    by_combined: dict[str, list[str]] = defaultdict(list)
    by_aux: dict[str, list[str]] = defaultdict(list)
    for r in train:
        by_bucket[_score_bucket(r)].append(_target_label(r))
        by_aux[_aux_signature(r)].append(_target_label(r))
        by_combined[_combined_signature(r)].append(_target_label(r))
    bucket_majority = _majority(by_bucket.get(bucket, []), global_majority)
    aux_majority = _majority(by_aux.get(_aux_signature(row), []), bucket_majority)
    combined_majority = _majority(by_combined.get(_combined_signature(row), []), aux_majority)
    return {
        "global_majority": global_majority,
        "last_history": last,
        "history_majority": hist_majority,
        "bucket_majority": bucket_majority,
        "aux_majority": aux_majority,
        "combined_aux_bucket_majority": combined_majority,
    }


def run(cfg: SideMapReliabilityMemoryEvalCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    train_splits = {s.strip() for s in cfg.train_splits.split(",") if s.strip()}
    train = [r for r in rows if str(r.get("split")) in train_splits]
    eval_rows = [r for r in rows if str(r.get("split")) == cfg.eval_split]
    pred_rows = []
    correct = Counter()
    total = Counter()
    for row in eval_rows:
        truth = _target_label(row)
        preds = _predictors(train, row)
        pred_rows.append({"month": row.get("month"), "truth": truth, "predictions": preds, "score_bucket": _score_bucket(row), "aux_signature": _aux_signature(row), "history": _history_labels(row)})
        for name, pred in preds.items():
            total[name] += 1
            correct[name] += int(pred == truth)
    metrics = {name: {"correct": int(correct[name]), "total": int(total[name]), "accuracy": float(correct[name] / total[name]) if total[name] else 0.0} for name in sorted(total)}
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "train_rows": len(train), "eval_rows": len(eval_rows), "metrics": metrics, "predictions": pred_rows, "leakage_guard": {"uses_eval_labels_for_training": False, "memory_built_from_train_splits_only": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate side-map reliability memory baselines")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-splits", default=SideMapReliabilityMemoryEvalCfg.train_splits)
    p.add_argument("--eval-split", default=SideMapReliabilityMemoryEvalCfg.eval_split)
    return p.parse_args()


def main() -> None:
    report = run(SideMapReliabilityMemoryEvalCfg(**vars(parse_args())))
    print(json.dumps({"train_rows": report["train_rows"], "eval_rows": report["eval_rows"], "metrics": report["metrics"], "predictions": report["predictions"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
