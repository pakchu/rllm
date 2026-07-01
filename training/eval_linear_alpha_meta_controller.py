"""Evaluate Gemma linear-alpha meta-controller JSON decisions.

The evaluator supports target echo for dataset validation and model generation for
adapter smoke tests.  It also writes live-style prediction JSONL so generated
TAKE/SKIP decisions can be replayed by ``online_risk_overlay_backtest``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.train_text_sft import load_jsonl, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

SIZE_SCALE = {"FULL": 1.0, "SMALL": 0.5, "NONE": 0.0}
CANDIDATE_LABELS = [
    {"decision": "SKIP", "size_bucket": "NONE"},
    {"decision": "TAKE", "size_bucket": "SMALL"},
    {"decision": "TAKE", "size_bucket": "FULL"},
]


def parse_meta_json(text: str) -> dict[str, str]:
    raw = str(text).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        obj = json.loads(raw)
    except Exception:
        up = raw.upper()
        decision = "TAKE" if "TAKE" in up and "SKIP" not in up[: max(0, up.find("TAKE"))] else "SKIP"
        if "FULL" in up:
            size = "FULL"
        elif "SMALL" in up:
            size = "SMALL"
        else:
            size = "NONE" if decision == "SKIP" else "SMALL"
        return {"decision": decision, "size_bucket": size, "risk_reason": "parser_fallback"}
    if not isinstance(obj, dict):
        return {"decision": "SKIP", "size_bucket": "NONE", "risk_reason": "non_dict_output"}
    decision = str(obj.get("decision", "SKIP")).upper()
    size = str(obj.get("size_bucket", "NONE")).upper()
    if decision not in {"TAKE", "SKIP"}:
        decision = "SKIP"
    if size not in {"FULL", "SMALL", "NONE"}:
        size = "NONE" if decision == "SKIP" else "SMALL"
    if decision == "SKIP":
        size = "NONE"
    if decision == "TAKE" and size == "NONE":
        size = "SMALL"
    return {"decision": decision, "size_bucket": size, "risk_reason": str(obj.get("risk_reason", ""))[:120]}


def _target(row: dict[str, Any]) -> dict[str, str]:
    return parse_meta_json(str(row.get("target", "{}")))


def _candidate_side_from_prompt(prompt: str) -> str:
    for line in str(prompt).splitlines():
        if line.startswith("candidate_side:"):
            side = line.split(":", 1)[1].strip().upper()
            return side if side in {"LONG", "SHORT"} else "NONE"
    return "NONE"


def _candidate_hold_from_prompt(prompt: str) -> int:
    for line in str(prompt).splitlines():
        if line.startswith("candidate_hold_bars:"):
            try:
                return int(float(line.split(":", 1)[1].strip()))
            except Exception:
                return 0
    return 0


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


def _generate(rows: list[dict[str, Any]], model_name: str, adapter_dir: str, max_new_tokens: int) -> tuple[list[dict[str, str]], list[str]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    outputs: list[dict[str, str]] = []
    raw_outputs: list[str] = []
    for row in rows:
        text = _chat_prompt_text(tokenizer, str(row["prompt"]))
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        raw_outputs.append(generated)
        outputs.append(parse_meta_json(generated))
    return outputs, raw_outputs



def _candidate_text(label: dict[str, str]) -> str:
    return json.dumps(label, ensure_ascii=False, sort_keys=True)


def _candidate_logprob(rows: list[dict[str, Any]], model_name: str, adapter_dir: str, score_normalization: str) -> tuple[list[dict[str, str]], list[str]]:
    import torch

    tokenizer, model = _load_text_model(model_name, adapter_dir)
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean", "first_token"}:
        raise ValueError("score_normalization must be one of {'sum','mean','first_token'}")
    preds: list[dict[str, str]] = []
    raw_outputs: list[str] = []
    for row in rows:
        prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
        scores = []
        for label in CANDIDATE_LABELS:
            label_text = _candidate_text(label)
            label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
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
                log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            positions = torch.arange(start - 1, end - 1, device=log_probs.device)
            label_tensor = input_ids[0, start:end]
            token_scores = log_probs[0, positions, label_tensor]
            if normalize == "sum":
                score = token_scores.sum()
            elif normalize == "first_token":
                score = token_scores[0]
            else:
                score = token_scores.mean()
            scores.append(float(score.detach().cpu()))
        best = max(range(len(scores)), key=lambda i: scores[i])
        pred = {**CANDIDATE_LABELS[best], "risk_reason": "candidate_logprob"}
        preds.append(pred)
        raw_outputs.append(json.dumps({"scores": {_candidate_text(CANDIDATE_LABELS[i]): scores[i] for i in range(len(scores))}, "prediction": pred}, ensure_ascii=False, sort_keys=True))
    return preds, raw_outputs

def _metrics(rows: list[dict[str, Any]], preds: list[dict[str, str]]) -> dict[str, Any]:
    target_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    confusion: dict[str, int] = {}
    decision_correct = 0
    size_correct = 0
    for row, pred in zip(rows, preds):
        target = _target(row)
        td, ts = target["decision"], target["size_bucket"]
        pd, ps = pred["decision"], pred["size_bucket"]
        target_counts[f"{td}/{ts}"] = target_counts.get(f"{td}/{ts}", 0) + 1
        pred_counts[f"{pd}/{ps}"] = pred_counts.get(f"{pd}/{ps}", 0) + 1
        confusion[f"target={td}/{ts}|pred={pd}/{ps}"] = confusion.get(f"target={td}/{ts}|pred={pd}/{ps}", 0) + 1
        decision_correct += int(td == pd)
        size_correct += int(td == pd and ts == ps)
    n = max(1, len(rows))
    return {
        "num_samples": len(rows),
        "decision_accuracy": decision_correct / n,
        "decision_size_accuracy": size_correct / n,
        "target_counts": dict(sorted(target_counts.items())),
        "pred_counts": dict(sorted(pred_counts.items())),
        "confusion": dict(sorted(confusion.items())),
    }


def _prediction_rows(rows: list[dict[str, Any]], preds: list[dict[str, str]], raw_outputs: list[str]) -> list[dict[str, Any]]:
    out = []
    for row, pred, raw in zip(rows, preds, raw_outputs):
        side = _candidate_side_from_prompt(str(row.get("prompt", "")))
        hold_bars = _candidate_hold_from_prompt(str(row.get("prompt", "")))
        take = pred["decision"] == "TAKE" and side in {"LONG", "SHORT"}
        action = {
            "gate": "TRADE" if take else "NO_TRADE",
            "side": side if take else "NONE",
            "hold_bars": hold_bars if take else 0,
            "family": "linear_alpha_meta_controller",
            "confidence": "MEDIUM",
        }
        out.append({
            "date": row.get("metadata", {}).get("date"),
            "signal_pos": row.get("metadata", {}).get("signal_pos"),
            "prediction": action,
            "position_scale": SIZE_SCALE[pred["size_bucket"]],
            "meta_prediction": pred,
            "target": _target(row),
            "raw_output": raw,
        })
    return out


def evaluate(
    *,
    eval_jsonl: str,
    output: str,
    model_name: str = "gemma4-e4b",
    adapter_dir: str = "",
    prediction_mode: str = "target_echo",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    max_new_tokens: int = 96,
    score_normalization: str = "mean",
    predictions_output: str = "",
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=int(max_samples), sample_mode=sample_mode, seed=int(seed))
    raw_outputs = [str(row.get("target", "")) for row in rows]
    if prediction_mode == "target_echo":
        preds = [_target(row) for row in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds, raw_outputs = _generate(rows, model_name, adapter_dir, int(max_new_tokens))
    elif prediction_mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        preds, raw_outputs = _candidate_logprob(rows, model_name, adapter_dir, score_normalization)
    else:
        raise ValueError("prediction_mode must be target_echo|model|candidate_logprob")
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "max_samples": int(max_samples),
        "sample_mode": sample_mode,
        "score_normalization": score_normalization if prediction_mode == "candidate_logprob" else None,
        "metrics": _metrics(rows, preds),
        "predictions_output": predictions_output,
    }
    if predictions_output:
        Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(predictions_output).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in _prediction_rows(rows, preds, raw_outputs)) + "\n")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate linear-alpha meta-controller JSON decisions")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="gemma4-e4b")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob"], default="target_echo")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--score-normalization", choices=["sum", "mean", "first_token"], default="mean")
    parser.add_argument("--predictions-output", default="")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
