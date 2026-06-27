"""Numpy pairwise baseline for episode survival preference rows."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairwiseBaselineCfg:
    train_jsonl: str
    eval_jsonl: str
    output: str
    epochs: int = 250
    lr: float = 0.1
    l2: float = 0.001
    max_features: int = 30000
    seed: int = 42


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def _load(path: str) -> list[dict[str, Any]]:
    with _open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_json(prompt: str, key: str) -> dict[str, Any]:
    m = re.search(rf"^{re.escape(key)}: (\{{.*\}})$", prompt, re.M)
    return json.loads(m.group(1)) if m else {}


def _flatten(prefix: str, obj: Any, num: dict[str, float], cat: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, num, cat)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        num[prefix] = float(obj)
    else:
        cat[prefix] = str(obj)


def _side_features(prompt: str, side: str) -> tuple[dict[str, float], dict[str, str]]:
    obj = _extract_json(prompt, f"candidate_{side}")
    hist = _extract_json(prompt, "causal_history")
    num: dict[str, float] = {}
    cat: dict[str, str] = {}
    _flatten("cand", obj, num, cat)
    _flatten("history", hist, num, cat)
    return num, cat


def _target(row: dict[str, Any]) -> int:
    return 1 if json.loads(row["target"]).get("choice") == "A" else 0


def _fit_vocab(rows: list[dict[str, Any]], max_features: int) -> tuple[list[str], dict[str, int], dict[str, tuple[float, float]]]:
    nums = []
    cat_counts: dict[str, int] = {}
    num_keys = set()
    for r in rows:
        na, ca = _side_features(r["prompt"], "A")
        nb, cb = _side_features(r["prompt"], "B")
        diff = {k: na.get(k, 0.0) - nb.get(k, 0.0) for k in set(na) | set(nb)}
        nums.append(diff)
        num_keys.update(diff)
        for k, v in ca.items():
            cat_counts[f"A:{k}={v}"] = cat_counts.get(f"A:{k}={v}", 0) + 1
        for k, v in cb.items():
            cat_counts[f"B:{k}={v}"] = cat_counts.get(f"B:{k}={v}", 0) + 1
        # identity comparisons are useful for horizon/event preference.
        for k in set(ca) | set(cb):
            cat_counts[f"EQ:{k}={ca.get(k)==cb.get(k)}"] = cat_counts.get(f"EQ:{k}={ca.get(k)==cb.get(k)}", 0) + 1
    num_keys_l = sorted(num_keys)
    stats = {}
    for k in num_keys_l:
        vals = np.asarray([d.get(k, 0.0) for d in nums], dtype=float)
        stats[k] = (float(vals.mean()), float(vals.std()) or 1.0)
    vocab: dict[str, int] = {}
    for token, _ in sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(vocab) >= max_features:
            break
        vocab[token] = len(vocab)
    return num_keys_l, vocab, stats


def _matrix(rows: list[dict[str, Any]], num_keys: list[str], vocab: dict[str, int], stats: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), 1 + len(num_keys) + len(vocab)), dtype=np.float32)
    x[:, 0] = 1.0
    y = np.asarray([_target(r) for r in rows], dtype=float)
    for i, r in enumerate(rows):
        na, ca = _side_features(r["prompt"], "A")
        nb, cb = _side_features(r["prompt"], "B")
        for j, k in enumerate(num_keys):
            mu, sd = stats[k]
            x[i, 1 + j] = ((na.get(k, 0.0) - nb.get(k, 0.0)) - mu) / sd
        base = 1 + len(num_keys)
        for token in [f"A:{k}={v}" for k, v in ca.items()] + [f"B:{k}={v}" for k, v in cb.items()] + [f"EQ:{k}={ca.get(k)==cb.get(k)}" for k in set(ca) | set(cb)]:
            jj = vocab.get(token)
            if jj is not None:
                x[i, base + jj] = 1.0
    return x, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _train(x: np.ndarray, y: np.ndarray, cfg: PairwiseBaselineCfg) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    w = rng.normal(0, 0.01, x.shape[1])
    for _ in range(cfg.epochs):
        p = _sigmoid(x @ w)
        grad = (x.T @ (p - y)) / max(1, len(y))
        grad[1:] += cfg.l2 * w[1:]
        w -= cfg.lr * grad
    return w


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = p >= 0.5
    return {"rows": int(len(y)), "accuracy": float((pred == y).mean()) if len(y) else 0.0, "pred_A_rate": float(pred.mean()) if len(y) else 0.0, "target_A_rate": float(y.mean()) if len(y) else 0.0}


def run(cfg: PairwiseBaselineCfg) -> dict[str, Any]:
    train = _load(cfg.train_jsonl)
    ev = _load(cfg.eval_jsonl)
    num_keys, vocab, stats = _fit_vocab(train, cfg.max_features)
    xtr, ytr = _matrix(train, num_keys, vocab, stats)
    xev, yev = _matrix(ev, num_keys, vocab, stats)
    w = _train(xtr, ytr, cfg)
    ptr = _sigmoid(xtr @ w)
    pev = _sigmoid(xev @ w)
    report = {"config": asdict(cfg), "features": {"numeric": len(num_keys), "categorical": len(vocab), "expanded": int(xtr.shape[1])}, "train": _metrics(ytr, ptr), "eval": _metrics(yev, pev), "leakage_guard": {"features_parse_prompt_only": True, "targets_use_pairwise_future_utility_only_for_training": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=PairwiseBaselineCfg.epochs)
    p.add_argument("--lr", type=float, default=PairwiseBaselineCfg.lr)
    p.add_argument("--l2", type=float, default=PairwiseBaselineCfg.l2)
    p.add_argument("--max-features", type=int, default=PairwiseBaselineCfg.max_features)
    p.add_argument("--seed", type=int, default=PairwiseBaselineCfg.seed)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PairwiseBaselineCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
