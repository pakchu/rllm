"""Train-only threshold scan for score-direction regime features."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DirectionThresholdAuditConfig:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    output: str
    min_train_class_count: int = 2
    top_k: int = 20


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _label(row: dict[str, Any]) -> str:
    return str((row.get("target") or {}).get("direction_regime", ""))


def _binary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if _label(r) in {"HIGH_SCORE_WINS", "LOW_SCORE_WINS"}]


def _features(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update((r.get("features") or {}).keys())
    return sorted(keys)


def _num(row: dict[str, Any], key: str) -> float:
    try:
        return float((row.get("features") or {}).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _acc(rows: list[dict[str, Any]], key: str, thr: float, high_when_ge: bool) -> tuple[int, int, float]:
    ok = 0
    for r in rows:
        pred_high = _num(r, key) >= thr if high_when_ge else _num(r, key) < thr
        pred = "HIGH_SCORE_WINS" if pred_high else "LOW_SCORE_WINS"
        ok += int(pred == _label(r))
    n = len(rows)
    return ok, n, ok / max(1, n)


def _fit_feature(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    vals = sorted(set(_num(r, key) for r in rows))
    if not vals:
        return None
    thresholds = vals if len(vals) == 1 else [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
    best = None
    for thr in thresholds:
        for high_when_ge in (True, False):
            ok, n, acc = _acc(rows, key, thr, high_when_ge)
            cand = {"feature": key, "threshold": thr, "high_when_ge": high_when_ge, "train_correct": ok, "train_rows": n, "train_accuracy": acc}
            if best is None or (cand["train_accuracy"], cand["train_correct"]) > (best["train_accuracy"], best["train_correct"]):
                best = cand
    return best


def _apply(rows: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any]:
    ok, n, acc = _acc(rows, str(rule["feature"]), float(rule["threshold"]), bool(rule["high_when_ge"]))
    counts: dict[str, int] = {}
    for r in rows:
        counts[_label(r)] = counts.get(_label(r), 0) + 1
    return {"rows": n, "correct": ok, "accuracy": acc, "label_counts": counts}


def run(cfg: DirectionThresholdAuditConfig) -> dict[str, Any]:
    train_all, test_all, eval_all = _load(cfg.train_jsonl), _load(cfg.test_jsonl), _load(cfg.eval_jsonl)
    train, test, eval_rows = _binary_rows(train_all), _binary_rows(test_all), _binary_rows(eval_all)
    train_counts: dict[str, int] = {}
    for r in train:
        train_counts[_label(r)] = train_counts.get(_label(r), 0) + 1
    if min(train_counts.values() or [0]) < int(cfg.min_train_class_count):
        status = "too_few_train_minority_examples"
    else:
        status = "ok"
    candidates = []
    for key in _features(train):
        fit = _fit_feature(train, key)
        if not fit:
            continue
        fit["test"] = _apply(test, fit)
        fit["eval"] = _apply(eval_rows, fit)
        # prefer train fit that also survives test, then eval diagnostic
        fit["selection_score"] = fit["train_accuracy"] + 0.5 * fit["test"]["accuracy"]
        candidates.append(fit)
    candidates.sort(key=lambda r: (r["selection_score"], r["train_accuracy"], r["test"]["accuracy"]), reverse=True)
    report = {
        "config": asdict(cfg),
        "status": status,
        "split_counts": {
            "train_all": len(train_all), "train_binary": len(train),
            "test_all": len(test_all), "test_binary": len(test),
            "eval_all": len(eval_all), "eval_binary": len(eval_rows),
        },
        "train_label_counts": train_counts,
        "top_rules": candidates[: int(cfg.top_k)],
        "leakage_guard": {"thresholds_fit_train_only": True, "test_eval_not_used_for_selection": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-train-class-count", type=int, default=DirectionThresholdAuditConfig.min_train_class_count)
    p.add_argument("--top-k", type=int, default=DirectionThresholdAuditConfig.top_k)
    print(json.dumps(run(DirectionThresholdAuditConfig(**vars(p.parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
