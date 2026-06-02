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


def parse_trader_json(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
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


def _generate_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_dir: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
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
    preds: list[dict[str, Any]] = []
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
        preds.append(parse_trader_json(generated))
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
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    mode = str(prediction_mode).strip().lower()
    if mode == "target_echo":
        predictions = [parse_trader_json(str(r["target"])) for r in rows]
    elif mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        predictions = _generate_predictions(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model'}")
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": mode,
        "metrics": _metrics(rows, predictions),
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
    p.add_argument("--prediction-mode", choices=["target_echo", "model"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate_text_trader(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
