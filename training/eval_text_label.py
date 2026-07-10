"""Evaluate plain single-label text SFT outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.train_text_sft import RECOMMENDED_TEXT_CAUSAL_LM_MODEL, load_jsonl, resolve_text_causal_lm_alias
from utils import disable_transformers_allocator_warmup

VALID_VALUES = {"gate": ("NO_TRADE", "TRADE"), "side": ("LONG", "SHORT"), "decision": ("ABSTAIN", "TRADE")}


def parse_label(text: str, *, key: str) -> str:
    key = str(key).strip().lower()
    vals = VALID_VALUES[key]
    raw = str(text).strip().upper()
    raw = re.sub(r"[^A-Z_]+", " ", raw)
    tokens = raw.split()
    for val in vals:
        if val in tokens or raw.startswith(val):
            return val
    return vals[0]


def _metrics(rows: list[dict[str, Any]], predictions: list[str], *, key: str) -> dict[str, Any]:
    correct = 0
    confusion: dict[str, int] = {}
    for row, pred in zip(rows, predictions):
        target = parse_label(str(row["target"]), key=key)
        correct += int(pred == target)
        ckey = f"target={target}|pred={pred}"
        confusion[ckey] = confusion.get(ckey, 0) + 1
    return {"num_samples": len(rows), "accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(confusion.items()))}


def _generate_predictions(rows: list[dict[str, Any]], *, key: str, model_name: str, adapter_dir: str, max_new_tokens: int, load_in_4bit: bool = False) -> list[str]:
    resolved = _assert_adapter_matches_model(model_name, adapter_dir)
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto", quantization_config=quantization_config)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    preds: list[str] = []
    for row in rows:
        prompt = str(row["prompt"])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if getattr(tokenizer, "chat_template", None) else f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        preds.append(parse_label(generated, key=key))
    return preds


def _adapter_base_model(adapter_dir: str) -> str | None:
    cfg = Path(adapter_dir) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        return None
    base = data.get("base_model_name_or_path")
    return str(base) if base else None


def _assert_adapter_matches_model(model_name: str, adapter_dir: str) -> str:
    resolved = resolve_text_causal_lm_alias(model_name, prefer_latest=True)
    adapter_base = _adapter_base_model(adapter_dir)
    if adapter_base and adapter_base != resolved:
        raise ValueError(
            f"adapter was trained on base_model_name_or_path={adapter_base!r}, "
            f"but selector model resolved to {resolved!r}; retrain/export the adapter on the text-only base"
        )
    return resolved


def _load_text_model(model_name: str, adapter_dir: str, *, load_in_4bit: bool = False):
    resolved = _assert_adapter_matches_model(model_name, adapter_dir)
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto", quantization_config=quantization_config)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def _chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if getattr(tokenizer, "chat_template", None)
        else f"<|user|>\n{prompt}\n<|assistant|>\n"
    )


def _candidate_logprob_predictions(rows: list[dict[str, Any]], *, key: str, model_name: str, adapter_dir: str, score_normalization: str = "mean", load_in_4bit: bool = False) -> list[str]:
    import torch

    tokenizer, model = _load_text_model(model_name, adapter_dir, load_in_4bit=load_in_4bit)
    labels = list(VALID_VALUES[key])
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean", "first_token"}:
        raise ValueError("score_normalization must be one of {'sum','mean','first_token'}")
    preds: list[str] = []
    for row in rows:
        prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for label in labels:
            label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                label_ids = label_ids + [int(tokenizer.eos_token_id)]
            start = len(prompt_ids)
            end = start + len(label_ids)
            sequences.append(prompt_ids + label_ids)
            spans.append((start, end))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        scores: list[float] = []
        for i, (start, end) in enumerate(spans):
            positions = torch.arange(start - 1, end - 1, device=log_probs.device)
            label_tensor = input_ids[i, start:end]
            token_scores = log_probs[i, positions, label_tensor]
            if normalize == "first_token":
                score = token_scores[0]
            else:
                score = token_scores.sum() if normalize == "sum" else token_scores.mean()
            scores.append(float(score.detach().cpu()))
        preds.append(labels[max(range(len(scores)), key=lambda i: scores[i])])
    return preds


def _prediction_rows(rows: list[dict[str, Any]], preds: list[str], *, key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, pred in zip(rows, preds):
        out.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prediction": pred,
                "target": parse_label(str(row["target"]), key=key),
            }
        )
    return out


def evaluate_text_label(*, eval_jsonl: str, output: str, key: str, model_name: str = RECOMMENDED_TEXT_CAUSAL_LM_MODEL, adapter_dir: str = "", max_samples: int = 0, sample_mode: str = "sequential", seed: int = 42, prediction_mode: str = "target_echo", max_new_tokens: int = 8, score_normalization: str = "mean", predictions_output: str = "", load_in_4bit: bool = False) -> dict[str, Any]:
    key = str(key).strip().lower()
    if key not in VALID_VALUES:
        raise ValueError("key must be one of {'gate','side','decision'}")
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    if prediction_mode == "target_echo":
        preds = [parse_label(str(r["target"]), key=key) for r in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = _generate_predictions(rows, key=key, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens, load_in_4bit=load_in_4bit)
    elif prediction_mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        preds = _candidate_logprob_predictions(rows, key=key, model_name=model_name, adapter_dir=adapter_dir, score_normalization=score_normalization, load_in_4bit=load_in_4bit)
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model','candidate_logprob'}")
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "key": key,
        "model_name": resolve_text_causal_lm_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "predictions_output": predictions_output,
        "score_normalization": score_normalization if prediction_mode == "candidate_logprob" else None,
        "load_in_4bit": bool(load_in_4bit),
        "metrics": _metrics(rows, preds, key=key),
    }
    if predictions_output:
        outp = Path(predictions_output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in _prediction_rows(rows, preds, key=key)) + "\n")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate plain-label text adapter")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--key", choices=["gate", "side", "decision"], required=True)
    p.add_argument("--model-name", default=RECOMMENDED_TEXT_CAUSAL_LM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--score-normalization", choices=["sum", "mean", "first_token"], default="mean")
    p.add_argument("--predictions-output", default="")
    p.add_argument("--load-in-4bit", action="store_true")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_text_label(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
