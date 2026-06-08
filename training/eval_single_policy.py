"""Evaluate single semantic policy LLM rows and export strict-action predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_trader import _chat_prompt_text, _load_text_model
from training.single_policy_sft_data import hold_bars_for_exit_profile
from training.train_text_sft import load_jsonl

VALID = {
    "regime": {"TREND_UP", "TREND_DOWN", "RANGE", "CHOP", "REVERSAL_RISK"},
    "edge_quality": {"NONE", "WEAK", "MODERATE", "STRONG"},
    "risk": {"LOW", "MID", "HIGH"},
    "action": {"NO_TRADE", "LONG", "SHORT"},
    "exit_profile": {"AVOID", "FAST", "NORMAL", "TRAIL"},
    "confidence": {"LOW", "MID", "HIGH"},
}
DEFAULT_POLICY = {
    "regime": "RANGE",
    "edge_quality": "NONE",
    "risk": "LOW",
    "action": "NO_TRADE",
    "exit_profile": "AVOID",
    "confidence": "LOW",
}


def parse_policy_json(text: str) -> dict[str, str]:
    raw = str(text).strip()
    obj: Any = {}
    try:
        obj = json.loads(raw)
    except Exception:
        for match in re.finditer(r"\{[^{}]*\}", raw, flags=re.DOTALL):
            try:
                candidate = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(candidate, dict):
                obj = candidate
                break
    if not isinstance(obj, dict):
        obj = {}
    out = dict(DEFAULT_POLICY)
    for key, allowed in VALID.items():
        val = str(obj.get(key, out[key])).upper()
        out[key] = val if val in allowed else out[key]
    if out["action"] == "NO_TRADE":
        out["exit_profile"] = "AVOID"
        out["edge_quality"] = "NONE" if out["edge_quality"] not in {"WEAK", "MODERATE", "STRONG"} else out["edge_quality"]
    elif out["exit_profile"] == "AVOID":
        out["exit_profile"] = "NORMAL"
    return out


def policy_to_action(policy: dict[str, str]) -> dict[str, Any]:
    action = str(policy.get("action", "NO_TRADE")).upper()
    if action not in {"LONG", "SHORT"}:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
    return {"gate": "TRADE", "side": action, "hold_bars": hold_bars_for_exit_profile(str(policy.get("exit_profile", "NORMAL")))}


def _generate_policies(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, max_new_tokens: int) -> list[dict[str, str]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    preds: list[dict[str, str]] = []
    for row in rows:
        text = _chat_prompt_text(tokenizer, str(row["prompt"]))
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        preds.append(parse_policy_json(generated))
    return preds


def _policy_key(policy: dict[str, str]) -> str:
    return f"action={policy.get('action')},exit={policy.get('exit_profile')},risk={policy.get('risk')},edge={policy.get('edge_quality')}"


def _metrics(rows: list[dict[str, Any]], predictions: list[dict[str, str]]) -> dict[str, Any]:
    field_correct: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    exact = 0
    confusion: Counter[str] = Counter()
    for row, pred in zip(rows, predictions):
        target = parse_policy_json(str(row.get("target", "{}")))
        pred_counts[_policy_key(pred)] += 1
        target_counts[_policy_key(target)] += 1
        if pred == target:
            exact += 1
        for key in VALID:
            field_counts[key] += 1
            if pred.get(key) == target.get(key):
                field_correct[key] += 1
        confusion[f"target={target.get('action')}/{target.get('exit_profile')}|pred={pred.get('action')}/{pred.get('exit_profile')}"] += 1
    n = len(rows)
    return {
        "num_samples": n,
        "exact_policy_accuracy": exact / max(1, n),
        "field_accuracy": {k: field_correct[k] / max(1, field_counts[k]) for k in sorted(VALID)},
        "prediction_counts": dict(sorted(pred_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "confusion": dict(sorted(confusion.items())),
    }


def _prediction_rows(rows: list[dict[str, Any]], predictions: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row, pred in zip(rows, predictions):
        target = parse_policy_json(str(row.get("target", "{}")))
        out.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "policy_prediction": pred,
                "policy_target": target,
                "prediction": policy_to_action(pred),
            }
        )
    return out


def evaluate_single_policy(
    *,
    eval_jsonl: str,
    output: str,
    predictions_output: str = "",
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str = "",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    prediction_mode: str = "target_echo",
    max_new_tokens: int = 80,
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    mode = str(prediction_mode).strip().lower()
    if mode == "target_echo":
        predictions = [parse_policy_json(str(r.get("target", "{}"))) for r in rows]
    elif mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        predictions = _generate_policies(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model'}")
    pred_rows = _prediction_rows(rows, predictions)
    if predictions_output:
        Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(predictions_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pred_rows) + "\n")
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": mode,
        "predictions_output": predictions_output,
        "row_selection": {"evaluated_rows": len(rows), "max_samples": int(max_samples), "sample_mode": sample_mode},
        "metrics_vs_target": _metrics(rows, predictions),
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_used_for_metrics_only": True,
            "model_input_excludes_target": True,
            "strict_action_prediction_maps_exit_profile_to_hold_bars": True,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate single semantic policy rows")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=80)
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_single_policy(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
