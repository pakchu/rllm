"""Train a causal bag-of-clause NB model for focused reward labels and export policy predictions."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.single_policy_sft_data import exit_profile_for_hold

LABELS = {
    "path_shape": [
        "CLEAN_WIN_PATH",
        "HIGH_ADVERSE_PATH",
        "FAILED_FOLLOW_THROUGH",
        "LOW_EDGE_PATH",
        "MIXED_PATH",
    ],
    "utility_bucket": ["UTILITY_LOW", "UTILITY_MID", "UTILITY_HIGH"],
}


@dataclass(frozen=True)
class FocusNbCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    output_dir: str
    min_token_count: int = 3
    alpha: float = 1.0
    min_clean_prob: float = 0.0
    min_high_prob: float = 0.0


def _load(path: str) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _target(row: dict[str, Any]) -> dict[str, str]:
    obj = json.loads(str(row["target"]))
    return {
        "path_shape": str(obj.get("path_shape")),
        "utility_bucket": str(obj.get("utility_bucket")),
    }


def _tokens(row: dict[str, Any]) -> list[str]:
    prompt = str(row.get("prompt", ""))
    # Keep clause-level words and key=value fragments. Drop date to avoid memorization.
    prompt = re.sub(r"date:.*", "", prompt)
    toks = re.findall(r"[A-Za-z_]+=?[A-Za-z_0-9.:-]*", prompt)
    return [t.lower() for t in toks if len(t) >= 2]


def _fit_one(rows: list[dict[str, Any]], key: str, min_token_count: int, alpha: float) -> dict[str, Any]:
    global_counts = Counter()
    docs = []
    for row in rows:
        toks = _tokens(row)
        global_counts.update(set(toks))
        docs.append((row, toks))

    vocab = {tok: i for i, (tok, c) in enumerate(global_counts.items()) if c >= int(min_token_count)}
    class_doc = Counter()
    class_tok = {label: Counter() for label in LABELS[key]}
    class_total = Counter()
    for row, toks in docs:
        label = _target(row)[key]
        if label not in LABELS[key]:
            continue
        class_doc[label] += 1
        seen = [t for t in toks if t in vocab]
        class_tok[label].update(seen)
        class_total[label] += len(seen)

    n_docs = sum(class_doc.values())
    v = max(1, len(vocab))
    priors = {
        label: math.log((class_doc[label] + alpha) / (n_docs + alpha * len(LABELS[key])))
        for label in LABELS[key]
    }
    cond = {}
    for label in LABELS[key]:
        denom = class_total[label] + alpha * v
        cond[label] = {tok: math.log((class_tok[label][tok] + alpha) / denom) for tok in vocab}
        cond[label]["__UNK__"] = math.log(alpha / denom)

    return {
        "key": key,
        "vocab": vocab,
        "priors": priors,
        "cond": cond,
        "class_doc": dict(class_doc),
    }


def _predict_one(model: dict[str, Any], row: dict[str, Any]) -> tuple[str, dict[str, float]]:
    toks = [t for t in _tokens(row) if t in model["vocab"]]
    scores = {}
    for label, prior in model["priors"].items():
        score = float(prior)
        cond = model["cond"][label]
        unk = cond["__UNK__"]
        for tok in toks:
            score += cond.get(tok, unk)
        scores[label] = score

    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    denom = sum(exp_scores.values()) or 1.0
    probs = {k: v / denom for k, v in exp_scores.items()}
    return max(probs.items(), key=lambda kv: kv[1])[0], probs


def _policy(
    row: dict[str, Any],
    pred: dict[str, str],
    probs: dict[str, dict[str, float]],
    cfg: FocusNbCfg,
) -> dict[str, str]:
    cand = dict(row.get("candidate") or {})
    side = str(cand.get("side", "")).upper()
    horizon = int(cand.get("horizon", 288) or 288)
    clean_p = float(probs["path_shape"].get("CLEAN_WIN_PATH", 0.0))
    high_p = float(probs["utility_bucket"].get("UTILITY_HIGH", 0.0))
    trade = (
        pred.get("path_shape") == "CLEAN_WIN_PATH"
        and pred.get("utility_bucket") == "UTILITY_HIGH"
        and clean_p >= float(cfg.min_clean_prob)
        and high_p >= float(cfg.min_high_prob)
        and side in {"LONG", "SHORT"}
    )
    if not trade:
        return {
            "regime": "RANGE",
            "edge_quality": "NONE",
            "risk": "LOW",
            "action": "NO_TRADE",
            "exit_profile": "AVOID",
            "confidence": "LOW",
        }
    return {
        "regime": "TREND_UP" if side == "LONG" else "TREND_DOWN",
        "edge_quality": "STRONG",
        "risk": "LOW",
        "action": side,
        "exit_profile": exit_profile_for_hold(horizon),
        "confidence": "HIGH",
    }


def _predict_rows(
    rows: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    cfg: FocusNbCfg,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        pred = {}
        probs = {}
        for key, model in models.items():
            label, label_probs = _predict_one(model, row)
            pred[key] = label
            probs[key] = label_probs
        out.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "candidate": row.get("candidate") or {},
                "policy_prediction": _policy(row, pred, probs, cfg),
                "focus_prediction": pred,
                "focus_probabilities": probs,
                "focus_target": _target(row),
                "target_audit": row.get("target_audit") or {},
            }
        )
    return out


def _metrics(rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in LABELS:
        correct = sum(pr["focus_prediction"][key] == _target(row)[key] for row, pr in zip(rows, pred_rows))
        out[key] = {"accuracy": correct / max(1, len(rows)), "correct": correct}
    actions = Counter(pr["policy_prediction"]["action"] for pr in pred_rows)
    return {"rows": len(rows), "per_key": out, "actions": dict(actions)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def run(cfg: FocusNbCfg) -> dict[str, Any]:
    train = _load(cfg.train_jsonl)
    test = _load(cfg.test_jsonl)
    eval_rows = _load(cfg.eval_jsonl)
    models = {key: _fit_one(train, key, int(cfg.min_token_count), float(cfg.alpha)) for key in LABELS}

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": asdict(cfg),
        "model_summary": {k: {"vocab": len(m["vocab"]), "class_doc": m["class_doc"]} for k, m in models.items()},
        "splits": {},
    }
    for split, rows in (("train", train), ("test", test), ("eval", eval_rows)):
        preds = _predict_rows(rows, models, cfg)
        path = out_dir / f"focus_nb_{split}_policy_predictions.jsonl"
        _write_jsonl(path, preds)
        report["splits"][split] = {**_metrics(rows, preds), "output": str(path)}

    (out_dir / "focus_nb_policy_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-token-count", type=int, default=FocusNbCfg.min_token_count)
    p.add_argument("--alpha", type=float, default=FocusNbCfg.alpha)
    p.add_argument("--min-clean-prob", type=float, default=FocusNbCfg.min_clean_prob)
    p.add_argument("--min-high-prob", type=float, default=FocusNbCfg.min_high_prob)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FocusNbCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
