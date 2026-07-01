"""Diagnose TAKE/SKIP separability for linear-alpha meta-controller SFT rows.

This is a cheap CPU baseline.  It parses the same signal-time prompt seen by the
LLM, fits a small numpy logistic regression on train rows only, and reports
chronological test/eval metrics plus univariate feature effects.  If this simple
baseline cannot separate TAKE from SKIP, larger Gemma fine-tuning is unlikely to
solve the current state surface without new features or labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetaFeatureDiagnosticConfig:
    train_jsonl: str
    test_jsonl: str
    output: str
    eval_jsonl: str = ""
    lr: float = 0.05
    steps: int = 600
    l2: float = 0.01
    max_features: int = 256


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _target_decision(row: dict[str, Any]) -> str:
    target = row.get("target", "")
    try:
        parsed = json.loads(target) if isinstance(target, str) else target
    except Exception:
        parsed = {}
    decision = str(parsed.get("decision", row.get("metadata", {}).get("target_decision", "SKIP"))).upper()
    return "TAKE" if decision == "TAKE" else "SKIP"


def _prompt_sections(prompt: str) -> tuple[dict[str, str], dict[str, float]]:
    tokens: dict[str, str] = {}
    numeric: dict[str, float] = {}
    mode = None
    for raw in str(prompt).splitlines():
        line = raw.strip()
        if line == "state_tokens:":
            mode = "tokens"
            continue
        if line == "numeric_state:":
            mode = "numeric"
            continue
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        key = key.strip()
        value = value.strip()
        if mode == "tokens":
            tokens[key] = value
        elif mode == "numeric":
            try:
                numeric[key] = float(value)
            except Exception:
                numeric[key] = 0.0
    # Preserve core candidate fields that are outside sections.
    for key in ("candidate_side", "candidate_hold_bars"):
        m = re.search(rf"^{key}:\s*(.+)$", str(prompt), flags=re.MULTILINE)
        if m:
            tokens[key] = m.group(1).strip()
    return tokens, numeric


def _num_bucket(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    v = float(value)
    av = abs(v)
    sign = "neg" if v < 0 else "pos"
    if av < 1e-9:
        return "zero"
    if av < 0.25:
        mag = "xs"
    elif av < 0.75:
        mag = "s"
    elif av < 1.5:
        mag = "m"
    elif av < 3.0:
        mag = "l"
    else:
        mag = "xl"
    return f"{sign}_{mag}"


def _row_features(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    tokens, numeric = _prompt_sections(str(row.get("prompt", "")))
    dense = dict(numeric)
    # Encode key categorical variables as deterministic indicator tokens.
    cats: dict[str, str] = {}
    for key, value in tokens.items():
        cats[f"tok:{key}={value}"] = "1"
    # Add robust numeric bucket tokens for nonlinear separability checks.
    for key, value in numeric.items():
        cats[f"bucket:{key}={_num_bucket(value)}"] = "1"
    return dense, cats


def _feature_space(rows: list[dict[str, Any]], max_features: int) -> list[str]:
    counts: Counter[str] = Counter()
    numeric_names: set[str] = set()
    for row in rows:
        dense, cats = _row_features(row)
        numeric_names.update(f"num:{k}" for k in dense)
        counts.update(cats.keys())
    cat_names = [name for name, _ in counts.most_common(max(0, int(max_features) - len(numeric_names)))]
    return sorted(numeric_names) + cat_names


def _matrix(rows: list[dict[str, Any]], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feat_to_idx = {f: i for i, f in enumerate(features)}
    x = np.zeros((len(rows), len(features)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    for r_i, row in enumerate(rows):
        dense, cats = _row_features(row)
        for key, value in dense.items():
            idx = feat_to_idx.get(f"num:{key}")
            if idx is not None:
                x[r_i, idx] = float(value) if math.isfinite(float(value)) else 0.0
        for key in cats:
            idx = feat_to_idx.get(key)
            if idx is not None:
                x[r_i, idx] = 1.0
        y[r_i] = 1.0 if _target_decision(row) == "TAKE" else 0.0
    return x, y


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1.0)
    return mu, sigma


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, lr: float, steps: int, l2: float) -> tuple[np.ndarray, float]:
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    n = max(1, len(y))
    for _ in range(max(1, int(steps))):
        p = _sigmoid(x @ w + b)
        err = p - y
        w -= float(lr) * ((x.T @ err) / n + float(l2) * w)
        b -= float(lr) * float(np.mean(err))
    return w, b


def _metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    pred = (prob >= 0.5).astype(np.float64)
    n = len(y)
    target_take = int(np.sum(y == 1.0))
    target_skip = int(np.sum(y == 0.0))
    pred_take = int(np.sum(pred == 1.0))
    pred_skip = int(np.sum(pred == 0.0))
    tp = int(np.sum((pred == 1.0) & (y == 1.0)))
    tn = int(np.sum((pred == 0.0) & (y == 0.0)))
    fp = int(np.sum((pred == 1.0) & (y == 0.0)))
    fn = int(np.sum((pred == 0.0) & (y == 1.0)))
    acc = float((tp + tn) / max(1, n))
    majority = max(target_take, target_skip) / max(1, n)
    take_recall = tp / max(1, target_take)
    skip_recall = tn / max(1, target_skip)
    return {
        "samples": int(n),
        "accuracy": acc,
        "majority_baseline_accuracy": float(majority),
        "beats_majority": bool(acc > majority),
        "balanced_recall": float(0.5 * (take_recall + skip_recall)),
        "take_recall": float(take_recall),
        "skip_recall": float(skip_recall),
        "target_counts": {"SKIP": target_skip, "TAKE": target_take},
        "pred_counts": {"SKIP": pred_skip, "TAKE": pred_take},
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "mean_prob_take": float(np.mean(prob)) if n else 0.0,
        "mean_abs_margin": float(np.mean(np.abs(prob - 0.5))) if n else 0.0,
    }


def _univariate(rows: list[dict[str, Any]], top_k: int = 30) -> list[dict[str, Any]]:
    values: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        dense, cats = _row_features(row)
        y = 1.0 if _target_decision(row) == "TAKE" else 0.0
        for key, value in dense.items():
            values.setdefault(f"num:{key}", []).append((float(value), y))
        for key in cats:
            values.setdefault(key, []).append((1.0, y))
    out = []
    for key, pairs in values.items():
        if len(pairs) < 20:
            continue
        xs = np.asarray([p[0] for p in pairs], dtype=float)
        ys = np.asarray([p[1] for p in pairs], dtype=float)
        if np.nanstd(xs) <= 1e-12:
            continue
        take = xs[ys == 1.0]
        skip = xs[ys == 0.0]
        if len(take) == 0 or len(skip) == 0:
            continue
        effect = (float(np.nanmean(take)) - float(np.nanmean(skip))) / max(1e-9, float(np.nanstd(xs)))
        corr = float(np.corrcoef(np.nan_to_num(xs), ys)[0, 1]) if len(xs) > 2 else 0.0
        out.append({"feature": key, "effect_take_minus_skip_std": effect, "corr": corr, "support": int(len(xs))})
    out.sort(key=lambda r: abs(float(r["corr"])), reverse=True)
    return out[:top_k]


def run(cfg: MetaFeatureDiagnosticConfig) -> dict[str, Any]:
    train_rows = _read_jsonl(cfg.train_jsonl)
    test_rows = _read_jsonl(cfg.test_jsonl)
    eval_rows = _read_jsonl(cfg.eval_jsonl) if cfg.eval_jsonl else []
    features = _feature_space(train_rows, int(cfg.max_features))
    x_train, y_train = _matrix(train_rows, features)
    x_test, y_test = _matrix(test_rows, features)
    x_eval, y_eval = _matrix(eval_rows, features) if eval_rows else (np.empty((0, len(features))), np.empty((0,)))
    mu, sigma = _standardize_fit(x_train)
    z_train = _standardize_apply(x_train, mu, sigma)
    w, b = _fit_logistic(z_train, y_train, lr=float(cfg.lr), steps=int(cfg.steps), l2=float(cfg.l2))
    split_metrics: dict[str, Any] = {}
    for name, x, y in (("train", x_train, y_train), ("test", x_test, y_test), ("eval", x_eval, y_eval)):
        if len(y) == 0:
            continue
        prob = _sigmoid(_standardize_apply(x, mu, sigma) @ w + b)
        split_metrics[name] = _metrics(y, prob)
    top_idx = np.argsort(np.abs(w))[::-1][: min(30, len(w))]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_count": len(features),
        "splits": split_metrics,
        "top_abs_weights": [{"feature": features[i], "weight": float(w[i])} for i in top_idx],
        "top_univariate_train": _univariate(train_rows, 30),
        "leakage_guard": {
            "features_parsed_from_llm_prompt_only": True,
            "train_only_standardization_and_fit": True,
            "test_eval_not_used_for_fit": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose linear-alpha meta-controller feature separability")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--lr", type=float, default=MetaFeatureDiagnosticConfig.lr)
    p.add_argument("--steps", type=int, default=MetaFeatureDiagnosticConfig.steps)
    p.add_argument("--l2", type=float, default=MetaFeatureDiagnosticConfig.l2)
    p.add_argument("--max-features", type=int, default=MetaFeatureDiagnosticConfig.max_features)
    return p.parse_args()


def main() -> None:
    report = run(MetaFeatureDiagnosticConfig(**vars(parse_args())))
    print(json.dumps({"feature_count": report["feature_count"], "splits": report["splits"], "top_univariate_train": report["top_univariate_train"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
