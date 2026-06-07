"""Evaluate text-trader JSON outputs against leakage-safe trader JSONL labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.train_text_sft import load_jsonl
from utils import disable_transformers_allocator_warmup


VALID_GATES = {"TRADE", "NO_TRADE"}
VALID_SIDES = {"LONG", "SHORT", "NONE"}


def _action_json(action: dict[str, Any]) -> str:
    payload = {
        "gate": str(action["gate"]),
        "hold_bars": int(action.get("hold_bars", 0) or 0),
        "side": str(action["side"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _candidate_actions(hold_candidates: list[int]) -> list[dict[str, Any]]:
    holds = [int(h) for h in hold_candidates if int(h) > 0]
    actions: list[dict[str, Any]] = [{"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}]
    for side in ("LONG", "SHORT"):
        for hold in holds:
            actions.append({"gate": "TRADE", "side": side, "hold_bars": hold})
    return actions


def parse_trader_json(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {}
        for match in re.finditer(r"\{[^{}]*\}", raw, flags=re.DOTALL):
            try:
                candidate = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(candidate, dict):
                obj = candidate
                break
    gate = str(obj.get("gate", "NO_TRADE")).upper()
    side = str(obj.get("side", "NONE")).upper()
    if gate not in VALID_GATES:
        gate = "NO_TRADE"
    if side not in VALID_SIDES:
        side = "NONE"
    try:
        hold_bars = int(obj.get("hold_bars", 0) or 0)
    except Exception:
        hold_bars = 0
    if gate == "NO_TRADE":
        side = "NONE"
        hold_bars = 0
    elif hold_bars <= 0:
        hold_bars = 0
    return {"gate": gate, "side": side, "hold_bars": hold_bars}


def _metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gate_ok = 0
    side_ok_when_trade = 0
    trade_targets = 0
    hold_ok_when_trade = 0
    exact = 0
    confusion: dict[str, int] = {}
    for row, pred in zip(rows, predictions):
        target = parse_trader_json(str(row["target"]))
        if pred["gate"] == target["gate"]:
            gate_ok += 1
        if target["gate"] == "TRADE":
            trade_targets += 1
            if pred["side"] == target["side"]:
                side_ok_when_trade += 1
            if int(pred.get("hold_bars", 0) or 0) == int(target.get("hold_bars", 0) or 0):
                hold_ok_when_trade += 1
        if pred == target:
            exact += 1
        key = (
            f"target={target['gate']}/{target['side']}/{target.get('hold_bars', 0)}|"
            f"pred={pred['gate']}/{pred['side']}/{pred.get('hold_bars', 0)}"
        )
        confusion[key] = confusion.get(key, 0) + 1
    return {
        "num_samples": n,
        "gate_accuracy": gate_ok / max(1, n),
        "side_accuracy_when_target_trade": side_ok_when_trade / max(1, trade_targets),
        "hold_accuracy_when_target_trade": hold_ok_when_trade / max(1, trade_targets),
        "exact_action_accuracy": exact / max(1, n),
        "trade_targets": trade_targets,
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


def _generate_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_dir: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    preds: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["prompt"])
        text = _chat_prompt_text(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        preds.append(parse_trader_json(generated))
    return preds


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
        token_positions = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected_logits = logits[i, token_positions, :].float()
        label_logits = selected_logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
        token_scores = label_logits - torch.logsumexp(selected_logits, dim=-1)
        score = token_scores.sum() if score_normalization == "sum" else token_scores.mean()
        scores.append(float(score.detach().cpu()))
    return scores


def _candidate_logprob_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_dir: str,
    hold_candidates: list[int],
    score_normalization: str = "mean",
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    actions = _candidate_actions(hold_candidates)
    action_token_ids: list[list[int]] = []
    for action in actions:
        action_ids = tokenizer(_action_json(action), add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            action_ids = action_ids + [int(tokenizer.eos_token_id)]
        action_token_ids.append(action_ids)
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean"}:
        raise ValueError("score_normalization must be one of {'sum','mean'}")
    batch_size = max(1, int(batch_size))

    preds: list[dict[str, Any]] = []
    for offset in range(0, len(rows), batch_size):
        row_batch = rows[offset : offset + batch_size]
        sequences: list[list[int]] = []
        candidate_spans: list[tuple[int, int]] = []
        candidate_count_by_row: list[int] = []
        for row in row_batch:
            prompt_text = _chat_prompt_text(tokenizer, str(row["prompt"]))
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            start = len(prompt_ids)
            candidate_count_by_row.append(len(action_token_ids))
            for action_ids in action_token_ids:
                end = start + len(action_ids)
                sequences.append(prompt_ids + action_ids)
                candidate_spans.append((start, end))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        flat_scores = _score_candidate_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            spans=candidate_spans,
            score_normalization=normalize,
        )
        score_offset = 0
        for candidate_count in candidate_count_by_row:
            scores = flat_scores[score_offset : score_offset + candidate_count]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            preds.append(dict(actions[best_idx]))
            score_offset += candidate_count
    return preds


def evaluate_text_trader(
    *,
    eval_jsonl: str,
    output: str,
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str = "",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    prediction_mode: str = "target_echo",
    max_new_tokens: int = 32,
    hold_candidates: str = "48,96,144,288",
    score_normalization: str = "mean",
    batch_size: int = 4,
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    mode = str(prediction_mode).strip().lower()
    if mode == "target_echo":
        predictions = [parse_trader_json(str(r["target"])) for r in rows]
    elif mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        predictions = _generate_predictions(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    elif mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        holds = [int(x) for x in str(hold_candidates).split(",") if str(x).strip()]
        predictions = _candidate_logprob_predictions(
            rows,
            model_name=model_name,
            adapter_dir=adapter_dir,
            hold_candidates=holds,
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
        "metrics": _metrics(rows, predictions),
        "candidate_logprob": {
            "hold_candidates": hold_candidates,
            "score_normalization": score_normalization,
            "batch_size": batch_size,
        } if mode == "candidate_logprob" else None,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate text trader outputs against trader JSONL labels")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--score-normalization", choices=["sum", "mean"], default="mean")
    p.add_argument("--batch-size", type=int, default=4, help="Rows per candidate-logprob scoring batch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate_text_trader(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
