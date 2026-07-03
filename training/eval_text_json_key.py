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
    "decision": {"ABSTAIN", "TAKE_FULL", "TAKE_SMALL"},
    "action": {"NO_TRADE", "LONG", "SHORT"},
    "side_map": {"NORMAL", "INVERSE", "UNRELIABLE"},
    "side_pair": {"NORMAL", "INVERSE"},
    "direction_regime": {"HIGH_SCORE_WINS", "LOW_SCORE_WINS", "ABSTAIN"},
    "trust_score_rank": {"HIGH", "LOW"},
}
DEFAULT_VALUES = {"gate": "NO_TRADE", "side": "LONG", "decision": "ABSTAIN", "action": "NO_TRADE", "side_map": "UNRELIABLE", "side_pair": "NORMAL", "direction_regime": "ABSTAIN", "trust_score_rank": "HIGH"}


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
    tokenizer.padding_side = "right"
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


def _candidate_values(key: str) -> list[str]:
    return sorted(VALID_VALUES[key])


def _candidate_json(key: str, value: str) -> str:
    # side_map/side_pair SFT targets are intentionally lowercase JSON values while
    # parse_key_json normalizes them to uppercase labels for metrics.
    raw_value = str(value).lower() if str(key).lower() in {"side_map", "side_pair"} else value
    return json.dumps({key: raw_value}, sort_keys=True, ensure_ascii=False)


def _score_candidate_batch(
    *,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    spans: list[tuple[int, int]],
    score_normalization: str,
) -> list[float]:
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    scores: list[float] = []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected_logits = logits[i, positions, :].float()
        label_logits = selected_logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
        token_scores = label_logits - torch.logsumexp(selected_logits, dim=-1)
        score = token_scores.sum() if score_normalization == "sum" else token_scores.mean()
        scores.append(float(score.detach().cpu()))
    return scores


def _candidate_logprob_predictions(
    rows: list[dict[str, Any]],
    *,
    key: str,
    model_name: str,
    adapter_dir: str,
    score_normalization: str = "mean",
    batch_size: int = 8,
) -> list[str]:
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    values = _candidate_values(key)
    candidate_token_ids: list[list[int]] = []
    for value in values:
        cand_ids = tokenizer(_candidate_json(key, value), add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            cand_ids = cand_ids + [int(tokenizer.eos_token_id)]
        candidate_token_ids.append(cand_ids)
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean"}:
        raise ValueError("score_normalization must be one of {'sum','mean'}")
    batch_size = max(1, int(batch_size))
    preds: list[str] = []
    for offset in range(0, len(rows), batch_size):
        row_batch = rows[offset : offset + batch_size]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        candidate_count_by_row: list[int] = []
        for row in row_batch:
            prompt = str(row["prompt"])
            messages = [{"role": "user", "content": prompt}]
            if getattr(tokenizer, "chat_template", None):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            start = len(prompt_ids)
            candidate_count_by_row.append(len(candidate_token_ids))
            for cand_ids in candidate_token_ids:
                end = start + len(cand_ids)
                sequences.append(prompt_ids + cand_ids)
                spans.append((start, end))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        flat_scores = _score_candidate_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            spans=spans,
            score_normalization=normalize,
        )
        score_offset = 0
        for candidate_count in candidate_count_by_row:
            scores = flat_scores[score_offset : score_offset + candidate_count]
            preds.append(values[max(range(len(scores)), key=lambda i: scores[i])])
            score_offset += candidate_count
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
    batch_size: int = 8,
    score_normalization: str = "mean",
) -> dict[str, Any]:
    key = str(key).strip().lower()
    if key not in VALID_VALUES:
        raise ValueError(f"key must be one of {sorted(VALID_VALUES)}")
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    if prediction_mode == "target_echo":
        preds = [parse_key_json(str(r["target"]), key=key) for r in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = _generate_predictions(rows, key=key, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    elif prediction_mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        preds = _candidate_logprob_predictions(
            rows,
            key=key,
            model_name=model_name,
            adapter_dir=adapter_dir,
            score_normalization=score_normalization,
            batch_size=batch_size,
        )
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model','candidate_logprob'}")
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "key": key,
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "batch_size": batch_size if prediction_mode == "candidate_logprob" else None,
        "score_normalization": score_normalization if prediction_mode == "candidate_logprob" else None,
        "metrics": _metrics(rows, preds, key=key),
        "predictions": [
            {"index": i, "prediction": pred, "target": parse_key_json(str(row["target"]), key=key)}
            for i, (row, pred) in enumerate(zip(rows, preds))
        ],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate single-key JSON text adapter")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--key", choices=sorted(VALID_VALUES), required=True)
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8, help="Rows per candidate-logprob scoring batch")
    p.add_argument("--score-normalization", choices=["sum", "mean"], default="mean")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_text_json_key(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
