"""Score TRADE vs NO_TRADE label candidates for gate calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_label import _chat_prompt_text, _load_text_model, parse_label
from training.train_text_sft import load_jsonl

LABELS = ["NO_TRADE", "TRADE"]


def _score_batch(model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int, int]]) -> tuple[list[float], list[float]]:
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sum_scores: list[float] = []
    mean_scores: list[float] = []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected_logits = logits[i, positions, :].float()
        label_logits = selected_logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
        token_scores = label_logits - torch.logsumexp(selected_logits, dim=-1)
        sum_scores.append(float(token_scores.sum().detach().cpu()))
        mean_scores.append(float(token_scores.mean().detach().cpu()))
    return sum_scores, mean_scores


def score_gate_candidates(*, eval_jsonl: str, output: str, model_name: str = RECOMMENDED_VLM_MODEL, adapter_dir: str, batch_size: int = 16, max_samples: int = 0, sample_mode: str = "sequential", seed: int = 42) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    label_ids: dict[str, list[int]] = {}
    for label in LABELS:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        label_ids[label] = ids
    out: list[dict[str, Any]] = []
    for offset in range(0, len(rows), max(1, int(batch_size))):
        batch = rows[offset : offset + max(1, int(batch_size))]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for row in batch:
            prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
            start = len(prompt_ids)
            for label in LABELS:
                ids = label_ids[label]
                sequences.append(prompt_ids + ids)
                spans.append((start, start + len(ids)))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        sums, means = _score_batch(model, encoded["input_ids"].to(model.device), encoded["attention_mask"].to(model.device), spans)
        p = 0
        for row in batch:
            score = {
                "NO_TRADE": {"sum": sums[p], "mean": means[p]},
                "TRADE": {"sum": sums[p + 1], "mean": means[p + 1]},
            }
            out.append(
                {
                    "date": row.get("date"),
                    "signal_pos": row.get("signal_pos"),
                    "target": parse_label(str(row["target"]), key="gate"),
                    "score": score,
                    "margin_sum_trade_minus_no_trade": score["TRADE"]["sum"] - score["NO_TRADE"]["sum"],
                    "margin_mean_trade_minus_no_trade": score["TRADE"]["mean"] - score["NO_TRADE"]["mean"],
                }
            )
            p += 2
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"eval_jsonl": str(Path(eval_jsonl).resolve()), "output": output, "rows": len(out), "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True), "adapter_dir": adapter_dir}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score gate label candidates")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    print(json.dumps(score_gate_candidates(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
