"""Score Gemma text safety labels for pre-entry regime gating.

The safety classifier is evaluated by candidate label log-probability instead of
free-form generation so the live gate has deterministic, auditable behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_label import _chat_prompt_text, _load_text_model
from training.train_text_sft import load_jsonl

SAFETY_LABELS = ("SAFE_TRADE", "UNSAFE_NO_EDGE", "BREAK_RISK")


def parse_safety_label(text: str) -> str:
    raw = str(text).strip().upper()
    for label in SAFETY_LABELS:
        if raw.startswith(label) or label in raw.split():
            return label
    return "UNSAFE_NO_EDGE"


def _score_batch(model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int, int]]) -> tuple[list[float], list[float]]:
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sums: list[float] = []
    means: list[float] = []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected_logits = logits[i, positions, :].float()
        label_logits = selected_logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
        token_scores = label_logits - torch.logsumexp(selected_logits, dim=-1)
        sums.append(float(token_scores.sum().detach().cpu()))
        means.append(float(token_scores.mean().detach().cpu()))
    return sums, means


def score_safety_rows(*, safety_jsonl: str, predictions_output: str, report_output: str, model_name: str = RECOMMENDED_VLM_MODEL, adapter_dir: str, batch_size: int = 8, max_samples: int = 0, sample_mode: str = "sequential", seed: int = 42, score_key: str = "mean") -> dict[str, Any]:
    rows = load_jsonl(safety_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    label_ids: dict[str, list[int]] = {}
    for label in SAFETY_LABELS:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        label_ids[label] = ids

    pred_rows: list[dict[str, Any]] = []
    correct = 0
    confusion: dict[str, int] = {}
    bs = max(1, int(batch_size))
    for offset in range(0, len(rows), bs):
        batch = rows[offset : offset + bs]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for row in batch:
            prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
            start = len(prompt_ids)
            for label in SAFETY_LABELS:
                ids = label_ids[label]
                sequences.append(prompt_ids + ids)
                spans.append((start, start + len(ids)))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        sums, means = _score_batch(model, encoded["input_ids"].to(model.device), encoded["attention_mask"].to(model.device), spans)
        p = 0
        for row in batch:
            scores = {label: {"sum": sums[p + i], "mean": means[p + i]} for i, label in enumerate(SAFETY_LABELS)}
            pred = max(SAFETY_LABELS, key=lambda label: scores[label][score_key])
            target = parse_safety_label(str(row.get("target", "")))
            correct += int(pred == target)
            confusion[f"target={target}|pred={pred}"] = confusion.get(f"target={target}|pred={pred}", 0) + 1
            safe_margin = float(scores["SAFE_TRADE"][score_key] - max(scores["UNSAFE_NO_EDGE"][score_key], scores["BREAK_RISK"][score_key]))
            pred_rows.append({
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prediction": pred,
                "target": target,
                "safe_margin": safe_margin,
                "scores": scores,
            })
            p += len(SAFETY_LABELS)

    Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pred_rows) + "\n")
    report = {
        "safety_jsonl": str(Path(safety_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "labels": list(SAFETY_LABELS),
        "score_key": score_key,
        "predictions_output": predictions_output,
        "metrics": {"num_samples": len(rows), "accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(confusion.items()))},
    }
    Path(report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(report_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score regime safety classifier labels")
    p.add_argument("--safety-jsonl", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--report-output", required=True)
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-key", choices=["mean", "sum"], default="mean")
    return p.parse_args()


def main() -> None:
    print(json.dumps(score_safety_rows(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
