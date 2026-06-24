"""Audit logprob priors for candidate text labels on a prompt set.

This is a guardrail for LLM classification experiments: before treating label
logprobs as model skill, measure whether the label strings themselves dominate
predictions.  It supports arbitrary labels and optional adapters.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup
from training.fast_score_action_value_candidates import _batched_label_scores


@dataclass(frozen=True)
class LabelPriorAuditConfig:
    input_jsonl: str
    output: str
    labels: tuple[str, ...]
    model_name: str = RECOMMENDED_VLM_MODEL
    adapter_dir: str = ""
    max_samples: int = 64
    batch_size: int = 1
    score_key: str = "mean"
    progress_every: int = 64


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(x.strip() for x in str(raw).split(",") if x.strip())
    if len(labels) < 2:
        raise ValueError("at least two labels are required")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    return labels


def _load_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    model.eval()
    return tokenizer, model


def _token_info(tokenizer: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        out[label] = {"token_ids": [int(x) for x in ids], "token_count": len(ids)}
    return out


def _summarize_scores(score_rows: list[dict[str, Any]], labels: tuple[str, ...], score_key: str) -> dict[str, Any]:
    means: dict[str, float] = {}
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for label in labels:
        xs = [float(row["score"][label][score_key]) for row in score_rows]
        means[label] = sum(xs) / max(1, len(xs))
        mins[label] = min(xs) if xs else 0.0
        maxs[label] = max(xs) if xs else 0.0
    pred_counts = Counter(str(row["prediction"]) for row in score_rows)
    spread = max(means.values()) - min(means.values()) if means else 0.0
    return {
        "mean_score_by_label": dict(sorted(means.items())),
        "min_score_by_label": dict(sorted(mins.items())),
        "max_score_by_label": dict(sorted(maxs.items())),
        "prediction_counts": dict(sorted(pred_counts.items())),
        "mean_score_spread": spread,
        "dominant_label": max(labels, key=lambda label: means[label]) if labels else None,
    }


def run(cfg: LabelPriorAuditConfig) -> dict[str, Any]:
    labels = tuple(cfg.labels)
    normalize = str(cfg.score_key).strip().lower()
    if normalize not in {"mean", "sum"}:
        raise ValueError("score_key must be mean or sum")
    rows = read_jsonl(cfg.input_jsonl)
    if int(cfg.max_samples) > 0:
        rows = rows[: int(cfg.max_samples)]
    tokenizer, model = _load_model(cfg.model_name, cfg.adapter_dir)
    label_ids: dict[str, list[int]] = {}
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        label_ids[label] = ids
    score_rows: list[dict[str, Any]] = []
    t0 = time.time()
    bs = max(1, int(cfg.batch_size))
    progress_every = max(0, int(cfg.progress_every))
    next_progress = progress_every if progress_every > 0 else 0
    for offset in range(0, len(rows), bs):
        batch = rows[offset : offset + bs]
        batch_scores = _batched_label_scores(model, tokenizer, [str(row.get("prompt", "")) for row in batch], label_ids, normalize)
        for row, scores in zip(batch, batch_scores):
            scalar = {label: float(scores[label][normalize]) for label in labels}
            pred = max(labels, key=lambda label: scalar[label])
            score_rows.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "target": row.get("target"), "prediction": pred, "score": scores})
        done = len(score_rows)
        if progress_every > 0 and done >= next_progress:
            print(json.dumps({"scored_rows": done, "rows": len(rows), "elapsed_sec": round(time.time() - t0, 2)}, ensure_ascii=False), flush=True)
            while next_progress <= done:
                next_progress += progress_every
    report = {
        "config": asdict(cfg) | {"labels": list(labels)},
        "model_name": resolve_vlm_model_alias(cfg.model_name, prefer_latest=True),
        "adapter_dir": cfg.adapter_dir,
        "rows_scored": len(score_rows),
        "token_info": _token_info(tokenizer, labels),
        "summary": _summarize_scores(score_rows, labels, normalize),
        "score_rows": score_rows,
        "leakage_guard": {"uses_prompt_only": True, "does_not_train_or_choose_strategy": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit label logprob priors on a prompt set")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--labels", required=True, help="Comma-separated labels")
    p.add_argument("--model-name", default=LabelPriorAuditConfig.model_name)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--score-key", choices=["mean", "sum"], default="mean")
    p.add_argument("--progress-every", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(
                LabelPriorAuditConfig(
                    input_jsonl=args.input_jsonl,
                    output=args.output,
                    labels=parse_labels(args.labels),
                    model_name=args.model_name,
                    adapter_dir=args.adapter_dir,
                    max_samples=args.max_samples,
                    batch_size=args.batch_size,
                    score_key=args.score_key,
                    progress_every=args.progress_every,
                )
            )
            | {"score_rows": "omitted_in_stdout"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
