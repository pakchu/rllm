"""Evaluate LONG/SHORT/NO_TRADE portfolio text labels.

Supports target_echo for pipeline validation, free generation, and candidate
log-prob scoring.  Candidate log-prob is preferred for trading labels because it
forces exactly one allowed action and avoids parser drift.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.train_text_sft import load_jsonl, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

LABELS = ("LONG", "SHORT", "NO_TRADE")


def parse_portfolio_label(text: str) -> str:
    raw = str(text).strip().upper()
    raw = re.sub(r"[^A-Z_]+", " ", raw)
    toks = raw.split()
    for lab in LABELS:
        if lab in toks or raw.startswith(lab):
            return lab
    return "NO_TRADE"


def _metrics(rows: list[dict[str, Any]], preds: list[str]) -> dict[str, Any]:
    confusion: dict[str, int] = {}
    correct = 0
    target_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for row, pred in zip(rows, preds):
        target = parse_portfolio_label(str(row["target"]))
        correct += int(pred == target)
        target_counts[target] = target_counts.get(target, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        key = f"target={target}|pred={pred}"
        confusion[key] = confusion.get(key, 0) + 1
    return {
        "num_samples": len(rows),
        "accuracy": correct / max(1, len(rows)),
        "target_counts": dict(sorted(target_counts.items())),
        "pred_counts": dict(sorted(pred_counts.items())),
        "confusion": dict(sorted(confusion.items())),
    }


def _load_text_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def _chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _generate_predictions(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, max_new_tokens: int) -> list[str]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    preds: list[str] = []
    for row in rows:
        text = _chat_prompt_text(tokenizer, str(row["prompt"]))
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        preds.append(parse_portfolio_label(generated))
    return preds


def _candidate_logprob_predictions(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, score_normalization: str) -> tuple[list[str], list[dict[str, float]]]:
    import torch

    tokenizer, model = _load_text_model(model_name, adapter_dir)
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean", "first_token"}:
        raise ValueError("score_normalization must be one of {'sum','mean','first_token'}")
    preds: list[str] = []
    score_rows: list[dict[str, float]] = []
    for row in rows:
        prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
        if normalize == "first_token":
            encoded = tokenizer.pad({"input_ids": [prompt_ids]}, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            first_ids = [tokenizer(lab, add_special_tokens=False)["input_ids"][0] for lab in LABELS]
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits[0, len(prompt_ids) - 1, :].float(), dim=-1)
            scores = [float(log_probs[int(tid)].detach().cpu()) for tid in first_ids]
        else:
            # Score each candidate independently.  Batched padded completion scoring
            # produced identical LONG/SHORT scores on Gemma4 for unequal label lengths.
            scores = []
            for lab in LABELS:
                label_ids = tokenizer(lab, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    label_ids = label_ids + [int(tokenizer.eos_token_id)]
                start = len(prompt_ids)
                end = start + len(label_ids)
                seq = prompt_ids + label_ids
                encoded = tokenizer.pad({"input_ids": [seq]}, return_tensors="pt")
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                positions = torch.arange(start - 1, end - 1, device=log_probs.device)
                label_tensor = input_ids[0, start:end]
                token_scores = log_probs[0, positions, label_tensor]
                score = token_scores.sum() if normalize == "sum" else token_scores.mean()
                scores.append(float(score.detach().cpu()))
        best = max(range(len(scores)), key=lambda i: scores[i])
        preds.append(LABELS[best])
        score_rows.append({lab: scores[i] for i, lab in enumerate(LABELS)})
    return preds, score_rows


def _prediction_rows(rows: list[dict[str, Any]], preds: list[str], score_rows: list[dict[str, float]] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (row, pred) in enumerate(zip(rows, preds)):
        out.append({
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prediction": pred,
            "target": parse_portfolio_label(str(row["target"])),
            "candidate": row.get("candidate", {}),
            "scores": (score_rows or [{}])[i] if score_rows is not None else {},
        })
    return out


def evaluate(*, eval_jsonl: str, output: str, model_name: str = "gemma4-e4b", adapter_dir: str = "", split: str = "eval", max_samples: int = 0, sample_mode: str = "sequential", seed: int = 42, prediction_mode: str = "target_echo", max_new_tokens: int = 8, score_normalization: str = "mean", predictions_output: str = "") -> dict[str, Any]:
    raw_rows = load_jsonl(eval_jsonl, max_samples=0, sample_mode=sample_mode, seed=seed)
    rows = [r for r in raw_rows if not split or str(r.get("split", "")) == str(split)]
    if max_samples:
        from training.train_text_sft import _select_rows
        rows = _select_rows(rows, max_samples=int(max_samples), sample_mode=sample_mode, seed=int(seed))
    score_rows = None
    if prediction_mode == "target_echo":
        preds = [parse_portfolio_label(str(r["target"])) for r in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = _generate_predictions(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    elif prediction_mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        preds, score_rows = _candidate_logprob_predictions(rows, model_name=model_name, adapter_dir=adapter_dir, score_normalization=score_normalization)
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model','candidate_logprob'}")
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "split": split,
        "score_normalization": score_normalization if prediction_mode == "candidate_logprob" else None,
        "predictions_output": predictions_output,
        "labels": LABELS,
        "metrics": _metrics(rows, preds),
    }
    if predictions_output:
        Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(predictions_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in _prediction_rows(rows, preds, score_rows)) + "\n")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate portfolio text labels")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default="gemma4-e4b")
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--split", default="eval")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--score-normalization", choices=["sum", "mean", "first_token"], default="mean")
    p.add_argument("--predictions-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
