"""Fast evaluator for candidate-wise ordinal utility labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_label import _load_text_model
from training.fast_score_action_value_candidates import _batched_label_scores

LABELS = ("AVOID", "LOW", "MID", "HIGH")
LABEL_RANK = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class FastEvalOrdinalUtilityConfig:
    input_jsonl: str
    output: str
    predictions_output: str = ""
    scores_output: str = ""
    model_name: str = RECOMMENDED_VLM_MODEL
    adapter_dir: str = ""
    prediction_mode: str = "target_echo"
    max_samples: int = 0
    batch_size: int = 8
    score_key: str = "mean"
    progress_every: int = 500


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _target(row: dict[str, Any]) -> str:
    val = str(row.get("target", "LOW")).strip().upper()
    return val if val in LABELS else "LOW"


def _metrics(rows: list[dict[str, Any]], preds: list[str]) -> dict[str, Any]:
    correct = 0
    abs_rank_error = 0
    target_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    for row, pred in zip(rows, preds):
        target = _target(row)
        pred = str(pred).strip().upper()
        if pred not in LABEL_RANK:
            pred = "LOW"
        correct += int(pred == target)
        abs_rank_error += abs(LABEL_RANK[pred] - LABEL_RANK[target])
        target_counts[target] += 1
        pred_counts[pred] += 1
        confusion[f"target={target}|pred={pred}"] += 1
    return {
        "num_samples": len(rows),
        "accuracy": correct / max(1, len(rows)),
        "mean_abs_rank_error": abs_rank_error / max(1, len(rows)),
        "target_counts": dict(sorted(target_counts.items())),
        "prediction_counts": dict(sorted(pred_counts.items())),
        "confusion": dict(sorted(confusion.items())),
    }


def _prediction_rows(rows: list[dict[str, Any]], preds: list[str], margins: list[float] | None = None) -> list[dict[str, Any]]:
    margins = margins or [0.0] * len(rows)
    out = []
    for row, pred, margin in zip(rows, preds, margins):
        out.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "target": _target(row), "prediction": pred, "high_minus_low_margin": float(margin), "action": row.get("action")})
    return out


def _score_model(rows: list[dict[str, Any]], cfg: FastEvalOrdinalUtilityConfig) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    if not cfg.adapter_dir:
        raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
    normalize = str(cfg.score_key).lower().strip()
    if normalize not in {"mean", "sum"}:
        raise ValueError("score_key must be mean or sum")
    tokenizer, model = _load_text_model(cfg.model_name, cfg.adapter_dir)
    label_ids: dict[str, list[int]] = {}
    for label in LABELS:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        label_ids[label] = ids
    preds: list[str] = []
    margins: list[float] = []
    score_rows: list[dict[str, Any]] = []
    t0 = time.time()
    bs = max(1, int(cfg.batch_size))
    progress_every = max(0, int(cfg.progress_every))
    next_progress = progress_every if progress_every > 0 else 0
    for offset in range(0, len(rows), bs):
        batch = rows[offset : offset + bs]
        batch_scores = _batched_label_scores(model, tokenizer, [str(row.get("prompt", "")) for row in batch], label_ids, normalize)
        for row, scores in zip(batch, batch_scores):
            label_scores = {label: float(scores[label][normalize]) for label in LABELS}
            pred = max(LABELS, key=lambda label: label_scores[label])
            margin = label_scores["HIGH"] - label_scores["LOW"]
            preds.append(pred)
            margins.append(margin)
            score_rows.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "target": _target(row), "score": scores, "prediction": pred, "high_minus_low_margin": margin, "action": row.get("action")})
        done = len(preds)
        if progress_every > 0 and done >= next_progress:
            print(json.dumps({"scored_rows": done, "rows": len(rows), "elapsed_sec": round(time.time() - t0, 2)}, ensure_ascii=False), flush=True)
            while next_progress <= done:
                next_progress += progress_every
    return preds, margins, score_rows


def run(cfg: FastEvalOrdinalUtilityConfig) -> dict[str, Any]:
    rows = read_jsonl(cfg.input_jsonl)
    if int(cfg.max_samples) > 0:
        rows = rows[: int(cfg.max_samples)]
    mode = str(cfg.prediction_mode).strip().lower()
    if mode == "target_echo":
        preds = [_target(row) for row in rows]
        margins = [0.0] * len(rows)
        score_rows: list[dict[str, Any]] = []
    elif mode == "candidate_logprob":
        preds, margins, score_rows = _score_model(rows, cfg)
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','candidate_logprob'}")
    if cfg.predictions_output:
        Path(cfg.predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.predictions_output).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in _prediction_rows(rows, preds, margins)))
    if cfg.scores_output:
        Path(cfg.scores_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.scores_output).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in score_rows))
    report = {
        "config": asdict(cfg),
        "model_name": resolve_vlm_model_alias(cfg.model_name, prefer_latest=True) if mode == "candidate_logprob" else None,
        "prediction_mode": mode,
        "labels": list(LABELS),
        "metrics": _metrics(rows, preds),
        "leakage_guard": {"target_echo_is_oracle_only": mode == "target_echo", "candidate_logprob_uses_prompt_only": mode == "candidate_logprob"},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ordinal utility label rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", default="")
    p.add_argument("--scores-output", default="")
    p.add_argument("--model-name", default=FastEvalOrdinalUtilityConfig.model_name)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--prediction-mode", choices=["target_echo", "candidate_logprob"], default="target_echo")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--score-key", choices=["mean", "sum"], default="mean")
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FastEvalOrdinalUtilityConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
