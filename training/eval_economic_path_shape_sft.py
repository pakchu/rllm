"""Evaluate path-shape analyzer/trader SFT outputs.

Analyzer metric focuses on direction_pressure and path grades. Trader metric
focuses on gate/side/template action. Supports target_echo smoke checks and
actual LoRA generation.
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

VALID_PRESSURE = {"LONG_FAVORED", "SHORT_FAVORED", "NO_TRADE_FAVORED", "BOTH_SIDES_VOLATILE"}
VALID_GRADE = {"CLEAN_TARGET", "NOISY_TARGET", "STOP_FIRST", "DRIFT_POSITIVE", "NO_EDGE"}
VALID_GATES = {"TRADE", "NO_TRADE"}
VALID_SIDES = {"LONG", "SHORT", "NONE"}


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


def parse_analyzer(text: str) -> dict[str, str]:
    obj = _extract_json_object(text)
    pressure = str(obj.get("direction_pressure", "NO_TRADE_FAVORED")).upper()
    long_path = obj.get("long_path", {}) if isinstance(obj.get("long_path"), dict) else {}
    short_path = obj.get("short_path", {}) if isinstance(obj.get("short_path"), dict) else {}
    long_grade = str(long_path.get("grade", "NO_EDGE")).upper()
    short_grade = str(short_path.get("grade", "NO_EDGE")).upper()
    return {
        "direction_pressure": pressure if pressure in VALID_PRESSURE else "NO_TRADE_FAVORED",
        "long_grade": long_grade if long_grade in VALID_GRADE else "NO_EDGE",
        "short_grade": short_grade if short_grade in VALID_GRADE else "NO_EDGE",
    }


def parse_trader(text: str) -> dict[str, Any]:
    obj = _extract_json_object(text)
    gate = str(obj.get("gate", "NO_TRADE")).upper()
    side = str(obj.get("side", "NONE")).upper()
    # Common small-model schema slip: put LONG/SHORT in gate even though the
    # action schema reserves gate for TRADE/NO_TRADE.  Repair it for metrics and
    # downstream action extraction instead of counting it as NO_TRADE collapse.
    if gate in {"LONG", "SHORT"} and side in {"LONG", "SHORT", "NONE"}:
        side = gate
        gate = "TRADE"
    if gate not in VALID_GATES:
        gate = "NO_TRADE"
    if side not in VALID_SIDES:
        side = "NONE"
    if gate == "NO_TRADE":
        side = "NONE"
    return {
        "gate": gate,
        "side": side,
        "target_pct": float(obj.get("target_pct", 0.0) or 0.0),
        "stop_pct": float(obj.get("stop_pct", 0.0) or 0.0),
        "max_hold_bars": int(obj.get("max_hold_bars", 0) or 0),
    }


def _confusion_add(confusion: dict[str, int], target: str, pred: str) -> None:
    key = f"target={target}|pred={pred}"
    confusion[key] = confusion.get(key, 0) + 1


def analyzer_metrics(rows: list[dict[str, Any]], preds: list[dict[str, str]]) -> dict[str, Any]:
    keys = ("direction_pressure", "long_grade", "short_grade")
    per_key = {}
    exact = 0
    for key in keys:
        correct = 0
        confusion: dict[str, int] = {}
        for row, pred in zip(rows, preds):
            target = parse_analyzer(str(row["target"]))[key]
            pval = pred[key]
            correct += int(target == pval)
            _confusion_add(confusion, target, pval)
        per_key[key] = {"accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(confusion.items()))}
    for row, pred in zip(rows, preds):
        target = parse_analyzer(str(row["target"]))
        exact += int(all(target[k] == pred[k] for k in keys))
    return {"num_samples": len(rows), "exact_pressure_and_grades_accuracy": exact / max(1, len(rows)), "per_key": per_key}


def trader_metrics(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gate_ok = side_ok = exact_action = 0
    confusion: dict[str, int] = {}
    for row, pred in zip(rows, preds):
        target = parse_trader(str(row["target"]))
        gate_ok += int(pred["gate"] == target["gate"])
        side_ok += int(pred["side"] == target["side"])
        exact_action += int(pred["gate"] == target["gate"] and pred["side"] == target["side"] and int(pred["max_hold_bars"]) == int(target["max_hold_bars"]))
        _confusion_add(confusion, f"{target['gate']}/{target['side']}", f"{pred['gate']}/{pred['side']}")
    return {"num_samples": n, "gate_accuracy": gate_ok / max(1, n), "side_accuracy": side_ok / max(1, n), "exact_template_accuracy": exact_action / max(1, n), "confusion": dict(sorted(confusion.items()))}


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _load_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
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
    return tokenizer, model


def generate(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, max_new_tokens: int, batch_size: int, task: str, prediction_output: str = "") -> list[dict[str, Any]]:
    tokenizer, model = _load_model(model_name, adapter_dir)
    device = next(model.parameters()).device
    preds: list[dict[str, Any]] = []
    stream = None
    if prediction_output:
        out = Path(prediction_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        stream = out.open("w")
    try:
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            texts = [_format_prompt(tokenizer, str(r["prompt"])) for r in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            out_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
            prompt_width = inputs["input_ids"].shape[-1]
            for row, gen_ids in zip(batch, out_ids[:, prompt_width:]):
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred = parse_analyzer(text) if task == "analyzer" else parse_trader(text)
                preds.append(pred)
                if stream is not None:
                    merged = dict(row)
                    merged["raw_prediction"] = text
                    merged["prediction"] = pred
                    stream.write(json.dumps(merged, ensure_ascii=False) + "\n")
            if stream is not None:
                stream.flush()
    finally:
        if stream is not None:
            stream.close()
    return preds


def evaluate_path_shape_sft(
    *,
    eval_jsonl: str,
    output: str,
    task: str,
    prediction_output: str = "",
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str = "",
    prediction_mode: str = "target_echo",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    max_new_tokens: int = 384,
    batch_size: int = 2,
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    task = str(task).lower().strip()
    if task not in {"analyzer", "trader"}:
        raise ValueError("task must be analyzer or trader")
    mode = str(prediction_mode).lower().strip()
    if mode == "target_echo":
        preds = [parse_analyzer(str(r["target"])) if task == "analyzer" else parse_trader(str(r["target"])) for r in rows]
    elif mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = generate(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens, batch_size=batch_size, task=task, prediction_output=prediction_output)
    else:
        raise ValueError("prediction_mode must be target_echo or model")
    metrics = analyzer_metrics(rows, preds) if task == "analyzer" else trader_metrics(rows, preds)
    report = {"eval_jsonl": str(Path(eval_jsonl).resolve()), "task": task, "prediction_mode": mode, "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True), "adapter_dir": adapter_dir, "prediction_output": prediction_output, "metrics": metrics}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", choices=["analyzer", "trader"], required=True)
    p.add_argument("--prediction-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--prediction-mode", choices=["target_echo", "model"], default="target_echo")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=2)
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_path_shape_sft(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
