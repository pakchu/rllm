"""Evaluate compact path-shape router-state analyzer JSON predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.train_text_sft import load_jsonl
from utils import disable_transformers_allocator_warmup

VALID = {
    "trend_side": {"LONG", "SHORT", "NONE"},
    "action_path": {"TREND", "FADE", "NONE"},
    "horizon_bars": {"0", "36", "72", "144", "288", "432"},
    "horizon_policy": {"SHORT_STEP", "MID_STEP", "LONG_STEP", "SKIP_STEP"},
    "edge_quality": {"STRONG", "MODERATE", "WEAK", "NO_EDGE"},
    "risk_budget": {"AGGRESSIVE_OK", "NORMAL", "SMALL", "AVOID_OR_TINY"},
    "score_bucket": {"HIGH", "MEDIUM", "LOW", "NEGATIVE_OR_TOO_WEAK"},
    "direction_stability": {"TREND_STABLE", "FADE_STABLE", "HORIZON_CONFLICT", "MIXED_WEAK", "NO_STABLE_EDGE"},
    "reversal_pressure": {"HIGH", "MEDIUM", "LOW"},
}
DEFAULTS = {
    "trend_side": "NONE",
    "action_path": "NONE",
    "horizon_bars": "0",
    "horizon_policy": "SKIP_STEP",
    "edge_quality": "NO_EDGE",
    "risk_budget": "AVOID_OR_TINY",
    "score_bucket": "NEGATIVE_OR_TOO_WEAK",
    "direction_stability": "NO_STABLE_EDGE",
    "reversal_pressure": "LOW",
}
PRIMARY_KEYS = ("action_path", "horizon_policy", "risk_budget")


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def parse_compact_path_shape_json(text: str) -> dict[str, str]:
    obj = _extract_json_object(text)
    out: dict[str, str] = {}
    for key, valid in VALID.items():
        raw = obj.get(key, DEFAULTS[key])
        val = str(int(raw)) if key == "horizon_bars" and isinstance(raw, (int, float)) else str(raw).upper()
        out[key] = val if val in valid else DEFAULTS[key]
    return out


def _confusion_add(confusion: dict[str, int], target: str, pred: str) -> None:
    key = f"target={target}|pred={pred}"
    confusion[key] = confusion.get(key, 0) + 1


def _metrics(rows: list[dict[str, Any]], preds: list[dict[str, str]]) -> dict[str, Any]:
    per_key: dict[str, Any] = {}
    exact_all = 0
    exact_primary = 0
    for key in VALID:
        correct = 0
        confusion: dict[str, int] = {}
        for row, pred in zip(rows, preds):
            target = parse_compact_path_shape_json(str(row["target"]))[key]
            pval = pred[key]
            correct += int(target == pval)
            _confusion_add(confusion, target, pval)
        per_key[key] = {"accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(confusion.items()))}
    for row, pred in zip(rows, preds):
        target = parse_compact_path_shape_json(str(row["target"]))
        exact_all += int(all(target[k] == pred[k] for k in VALID))
        exact_primary += int(all(target[k] == pred[k] for k in PRIMARY_KEYS))
    return {
        "num_samples": len(rows),
        "exact_all_keys_accuracy": exact_all / max(1, len(rows)),
        "exact_primary_keys_accuracy": exact_primary / max(1, len(rows)),
        "primary_keys": list(PRIMARY_KEYS),
        "per_key": per_key,
    }


def _format_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    messages = [{"role": "user", "content": str(row["prompt"])}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{row['prompt']}\n<|assistant|>\n"


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
            texts = [_format_prompt(row, tokenizer) for row in batch]
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
                pred = parse_compact_path_shape_json(generated)
                preds.append(pred)
                if stream is not None:
                    merged = dict(batch[idx])
                    merged["prediction"] = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    stream.write(json.dumps(merged, ensure_ascii=False) + "\n")
            if stream is not None:
                stream.flush()
            done = min(start + len(batch), len(rows))
            if progress_every > 0 and (done == len(rows) or done % progress_every == 0):
                print(f"generated {done}/{len(rows)} compact path-shape predictions", flush=True)
    finally:
        if stream is not None:
            stream.close()
    return preds


def write_prediction_jsonl(path: str | Path, rows: list[dict[str, Any]], preds: list[dict[str, str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row, pred in zip(rows, preds):
            merged = dict(row)
            merged["prediction"] = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")


def evaluate_compact_path_shape_analyzer(
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
    max_new_tokens: int = 384,
    batch_size: int = 2,
    progress_every: int = 64,
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    if prediction_mode == "target_echo":
        preds = [parse_compact_path_shape_json(str(r["target"])) for r in rows]
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
            "compact_targets_are_compressed_from_future_path_shape_labels": True,
            "target_is_router_state_not_final_order": True,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate compact path-shape router-state analyzer predictions")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prediction-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--progress-every", type=int, default=64)
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_compact_path_shape_analyzer(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
