"""Evaluate single-key JSON text SFT outputs, e.g. gate-only or side-only adapters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.train_text_sft import load_jsonl
from utils import disable_transformers_allocator_warmup


VALID_VALUES = {
    "gate": {"TRADE", "NO_TRADE"},
    "side": {"LONG", "SHORT"},
}
DEFAULT_VALUES = {"gate": "NO_TRADE", "side": "LONG"}


def parse_key_json(text: str, *, key: str) -> str:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
    val = str(obj.get(key, DEFAULT_VALUES[key])).upper()
    return val if val in VALID_VALUES[key] else DEFAULT_VALUES[key]


def _metrics(rows: list[dict[str, Any]], predictions: list[str], *, key: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    correct = 0
    for row, pred in zip(rows, predictions):
        target = parse_key_json(str(row["target"]), key=key)
        correct += int(pred == target)
        ckey = f"target={target}|pred={pred}"
        counts[ckey] = counts.get(ckey, 0) + 1
    return {"num_samples": len(rows), "accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(counts.items()))}


def _generate_predictions(rows: list[dict[str, Any]], *, key: str, model_name: str, adapter_dir: str, max_new_tokens: int) -> list[str]:
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
    preds: list[str] = []
    for row in rows:
        prompt = str(row["prompt"])
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        preds.append(parse_key_json(generated, key=key))
    return preds


def evaluate_text_json_key(
    *,
    eval_jsonl: str,
    output: str,
    key: str,
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str = "",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    prediction_mode: str = "target_echo",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    key = str(key).strip().lower()
    if key not in VALID_VALUES:
        raise ValueError("key must be one of {'gate','side'}")
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    if prediction_mode == "target_echo":
        preds = [parse_key_json(str(r["target"]), key=key) for r in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = _generate_predictions(rows, key=key, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model'}")
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "key": key,
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "metrics": _metrics(rows, preds, key=key),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate single-key JSON text adapter")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--key", choices=["gate", "side"], required=True)
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=16)
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_text_json_key(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
