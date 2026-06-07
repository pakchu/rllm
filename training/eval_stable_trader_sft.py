"""Evaluate stable trader SFT outputs with action/risk JSON targets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.train_text_sft import load_jsonl
from utils import disable_transformers_allocator_warmup

VALID_ACTIONS = {"LONG", "SHORT", "NO_TRADE"}
VALID_RISKS = {"LOW", "MEDIUM", "HIGH"}
ACTION_TO_PRESSURE = {"LONG": "LONG_FAVORED", "SHORT": "SHORT_FAVORED", "NO_TRADE": "NO_TRADE_FAVORED"}


def parse_stable_trader_json(text: str) -> dict[str, str]:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
    action = str(obj.get("action", "NO_TRADE")).upper()
    risk = str(obj.get("risk", "HIGH")).upper()
    if action in {"NONE", "HOLD"}:
        action = "NO_TRADE"
    if action not in VALID_ACTIONS:
        action = "NO_TRADE"
    if risk not in VALID_RISKS:
        risk = "HIGH"
    return {"action": action, "risk": risk}


def metrics(rows: list[dict[str, Any]], predictions: list[dict[str, str]]) -> dict[str, Any]:
    action_ok = 0
    risk_ok = 0
    exact = 0
    trade_targets = 0
    trade_pred = 0
    side_ok_when_target_trade = 0
    confusion: dict[str, int] = {}
    for row, pred in zip(rows, predictions):
        target = parse_stable_trader_json(str(row["target"]))
        if target["action"] != "NO_TRADE":
            trade_targets += 1
            if pred["action"] == target["action"]:
                side_ok_when_target_trade += 1
        if pred["action"] != "NO_TRADE":
            trade_pred += 1
        if pred["action"] == target["action"]:
            action_ok += 1
        if pred["risk"] == target["risk"]:
            risk_ok += 1
        if pred == target:
            exact += 1
        key = f"target={target['action']}/{target['risk']}|pred={pred['action']}/{pred['risk']}"
        confusion[key] = confusion.get(key, 0) + 1
    n = len(rows)
    return {
        "num_samples": n,
        "action_accuracy": action_ok / max(1, n),
        "risk_accuracy": risk_ok / max(1, n),
        "exact_accuracy": exact / max(1, n),
        "trade_targets": trade_targets,
        "trade_predictions": trade_pred,
        "side_accuracy_when_target_trade": side_ok_when_target_trade / max(1, trade_targets),
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
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def _chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def generate_predictions(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, max_new_tokens: int) -> list[dict[str, str]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    preds = []
    for row in rows:
        prompt = _chat_prompt_text(tokenizer, str(row["prompt"]))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        preds.append(parse_stable_trader_json(text))
    return preds


def _candidate_json(action: str, risk: str) -> str:
    return json.dumps({"action": action, "risk": risk}, sort_keys=True, ensure_ascii=False)


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
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    scores: list[float] = []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=log_probs.device)
        labels = input_ids[i, start:end]
        token_scores = log_probs[i, positions, labels]
        score = token_scores.sum() if score_normalization == "sum" else token_scores.mean()
        scores.append(float(score.detach().cpu()))
    return scores


def candidate_logprob_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_dir: str,
    score_normalization: str = "mean",
    batch_size: int = 4,
) -> list[dict[str, str]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    candidates = [{"action": a, "risk": r} for a in ["NO_TRADE", "LONG", "SHORT"] for r in ["HIGH", "MEDIUM", "LOW"]]
    candidate_token_ids: list[list[int]] = []
    for candidate in candidates:
        cand_ids = tokenizer(_candidate_json(candidate["action"], candidate["risk"]), add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            cand_ids = cand_ids + [int(tokenizer.eos_token_id)]
        candidate_token_ids.append(cand_ids)
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean"}:
        raise ValueError("score_normalization must be one of {'sum','mean'}")
    batch_size = max(1, int(batch_size))
    preds: list[dict[str, str]] = []
    for offset in range(0, len(rows), batch_size):
        row_batch = rows[offset : offset + batch_size]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        candidate_count_by_row: list[int] = []
        for row in row_batch:
            prompt_text = _chat_prompt_text(tokenizer, str(row["prompt"]))
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
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            preds.append(dict(candidates[best_idx]))
            score_offset += candidate_count
    return preds


def write_prediction_jsonl(path: str | Path, rows: list[dict[str, Any]], predictions: list[dict[str, str]]) -> None:
    out = []
    for row, pred in zip(rows, predictions):
        r = dict(row)
        r["prediction"] = {"direction_pressure": ACTION_TO_PRESSURE[pred["action"]], "risk": pred["risk"]}
        r["raw_action_prediction"] = pred
        out.append(r)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")


def evaluate_stable_trader(
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
    max_new_tokens: int = 48,
    batch_size: int = 4,
    score_normalization: str = "mean",
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    mode = str(prediction_mode).lower().strip()
    if mode == "target_echo":
        predictions = [parse_stable_trader_json(str(r["target"])) for r in rows]
    elif mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        predictions = generate_predictions(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    elif mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        predictions = candidate_logprob_predictions(
            rows,
            model_name=model_name,
            adapter_dir=adapter_dir,
            score_normalization=score_normalization,
            batch_size=batch_size,
        )
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model','candidate_logprob'}")
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": mode,
        "batch_size": batch_size if mode == "candidate_logprob" else None,
        "score_normalization": score_normalization if mode == "candidate_logprob" else None,
        "metrics": metrics(rows, predictions),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if prediction_output:
        write_prediction_jsonl(prediction_output, rows, predictions)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prediction-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=4, help="Rows per candidate-logprob scoring batch")
    p.add_argument("--score-normalization", choices=["sum", "mean"], default="mean")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_stable_trader(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
