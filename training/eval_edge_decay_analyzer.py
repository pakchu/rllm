"""Evaluate/export edge-decay analyzer JSON predictions.

Supports target-echo for pipeline checks and model generation for actual Gemma
analyzer adapters.  Output JSONL preserves original rows and adds a
``prediction`` field, making it directly consumable by
``edge_decay_router_backtest``.
"""

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
    "trend_side": {"LONG", "SHORT", "NONE"},
    "edge_decay_label": {"EDGE_PERSIST", "WEAK_PERSIST", "EDGE_DECAY", "WEAK_DECAY", "REVERSAL_RISK", "ADVERSE_STRESS", "NO_EDGE", "NO_CLEAR_TREND"},
    "transition_label": {"TREND_CONTINUATION", "TREND_REVERSAL", "CHOP_OR_DECAY", "RANGE_UNKNOWN"},
    "risk_label": {"LOW_ADVERSE_EXCURSION", "MANAGEABLE_ADVERSE_EXCURSION", "HIGH_ADVERSE_EXCURSION", "UNKNOWN"},
    "recommended_router_hint": {"ALLOW_TREND_SPECIALIST", "REDUCE_OR_SKIP_TREND_SPECIALIST", "CONSIDER_REVERSAL_SPECIALIST", "RANGE_ROUTER_ONLY", "LOW_CONFIDENCE_ROUTER"},
}
DEFAULTS = {
    "trend_side": "NONE",
    "edge_decay_label": "NO_CLEAR_TREND",
    "transition_label": "RANGE_UNKNOWN",
    "risk_label": "UNKNOWN",
    "recommended_router_hint": "LOW_CONFIDENCE_ROUTER",
}


def parse_edge_decay_json(text: str) -> dict[str, str]:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
    out: dict[str, str] = {}
    for key, valid in VALID_VALUES.items():
        value = str(obj.get(key, DEFAULTS[key])).upper()
        out[key] = value if value in valid else DEFAULTS[key]
    return out


def _metrics(rows: list[dict[str, Any]], predictions: list[dict[str, str]]) -> dict[str, Any]:
    key_metrics: dict[str, Any] = {}
    for key in VALID_VALUES:
        correct = 0
        confusion: dict[str, int] = {}
        for row, pred in zip(rows, predictions):
            target = parse_edge_decay_json(str(row["target"]))[key]
            pval = pred[key]
            correct += int(target == pval)
            ckey = f"target={target}|pred={pval}"
            confusion[ckey] = confusion.get(ckey, 0) + 1
        key_metrics[key] = {
            "accuracy": correct / max(1, len(rows)),
            "confusion": dict(sorted(confusion.items())),
        }
    exact = sum(int(parse_edge_decay_json(str(row["target"])) == pred) for row, pred in zip(rows, predictions))
    return {"num_samples": len(rows), "exact_all_keys_accuracy": exact / max(1, len(rows)), "keys": key_metrics}


def _generate_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_dir: str,
    max_new_tokens: int,
    batch_size: int,
    progress_every: int,
    prediction_output: str = "",
) -> list[dict[str, str]]:
    disable_transformers_allocator_warmup()
    import torch

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    device = next(model.parameters()).device
    preds: list[dict[str, str]] = []
    stream = None
    if prediction_output:
        out_path = Path(prediction_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stream = out_path.open("w")
    try:
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            texts = []
            for row in batch:
                messages = [{"role": "user", "content": str(row["prompt"])}]
                if getattr(tokenizer, "chat_template", None):
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    text = f"<|user|>\n{row['prompt']}\n<|assistant|>\n"
                texts.append(text)
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_width = inputs["input_ids"].shape[-1]
            for idx, generated_ids in enumerate(out[:, prompt_width:]):
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
                pred = parse_edge_decay_json(generated)
                preds.append(pred)
                if stream is not None:
                    merged = dict(batch[idx])
                    merged["prediction"] = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    stream.write(json.dumps(merged, ensure_ascii=False) + "\n")
            if stream is not None:
                stream.flush()
            done = min(start + len(batch), len(rows))
            if progress_every > 0 and (done == len(rows) or done % progress_every == 0):
                print(f"generated {done}/{len(rows)} edge-decay predictions", flush=True)
    finally:
        if stream is not None:
            stream.close()
    return preds


def write_prediction_jsonl(path: str | Path, rows: list[dict[str, Any]], predictions: list[dict[str, str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row, pred in zip(rows, predictions):
            merged = dict(row)
            merged["prediction"] = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")


def evaluate_edge_decay_analyzer(
    *,
    eval_jsonl: str,
    output: str,
    prediction_output: str = "",
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str = "",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    prediction_mode: str = "target_echo",
    max_new_tokens: int = 128,
    batch_size: int = 4,
    progress_every: int = 64,
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    if prediction_mode == "target_echo":
        preds = [parse_edge_decay_json(str(r["target"])) for r in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = _generate_predictions(
            rows,
            model_name=model_name,
            adapter_dir=adapter_dir,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            progress_every=progress_every,
            prediction_output=prediction_output,
        )
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model'}")
    if prediction_output and prediction_mode != "model":
        write_prediction_jsonl(prediction_output, rows, preds)
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "prediction_output": prediction_output,
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "metrics": _metrics(rows, preds),
        "leakage_guard": {
            "target_echo_is_oracle_only": prediction_mode == "target_echo",
            "model_mode_uses_prompt_only": prediction_mode == "model",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate/export edge-decay analyzer predictions")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prediction-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=64)
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_edge_decay_analyzer(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
