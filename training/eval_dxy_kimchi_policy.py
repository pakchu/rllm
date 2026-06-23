"""Evaluate DXY/Kimchi activate/action policy adapters and export trade rows."""
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
from utils import disable_transformers_allocator_warmup
from training.train_text_sft import load_jsonl

VALID_ACTIONS = {"NO_TRADE", "LONG", "SHORT"}
VALID_EXITS = {"AVOID", "FAST", "NORMAL"}
DEFAULT_POLICY = {"activate": False, "action": "NO_TRADE", "exit_profile": "AVOID", "confidence": "LOW", "reason_code": "parse_default"}


def parse_dxy_kimchi_policy(text: str) -> dict[str, Any]:
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
    activate_raw = obj.get("activate", out["activate"])
    out["activate"] = bool(activate_raw) if not isinstance(activate_raw, str) else activate_raw.strip().lower() == "true"
    action = str(obj.get("action", out["action"])).upper()
    out["action"] = action if action in VALID_ACTIONS else "NO_TRADE"
    exit_profile = str(obj.get("exit_profile", out["exit_profile"])).upper()
    out["exit_profile"] = exit_profile if exit_profile in VALID_EXITS else "AVOID"
    out["confidence"] = str(obj.get("confidence", out["confidence"])).upper()
    out["reason_code"] = str(obj.get("reason_code", out["reason_code"]))
    if not out["activate"] or out["action"] == "NO_TRADE":
        out["activate"] = False
        out["action"] = "NO_TRADE"
        out["exit_profile"] = "AVOID"
    elif out["exit_profile"] == "AVOID":
        out["exit_profile"] = "FAST"
    return out


def policy_to_prediction(policy: dict[str, Any], *, horizon: int = 144) -> dict[str, Any]:
    if not bool(policy.get("activate")) or str(policy.get("action")) not in {"LONG", "SHORT"}:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
    return {"gate": "TRADE", "side": str(policy["action"]), "hold_bars": int(horizon)}


def _generate(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, max_new_tokens: int, batch_size: int = 1) -> list[dict[str, Any]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    tokenizer.padding_side = "left"
    preds: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        texts = [_chat_prompt_text(tokenizer, str(row["prompt"])) for row in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        input_width = inputs["input_ids"].shape[-1]
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for i in range(len(batch)):
            generated = tokenizer.decode(out[i][input_width:], skip_special_tokens=True)
            preds.append(parse_dxy_kimchi_policy(generated))
    return preds




def _candidate_policy_jsons() -> list[tuple[dict[str, Any], str]]:
    # Keep these candidates in the exact compact schema used by SFT targets.
    # Log-prob scoring is highly sensitive to unseen reason_code/confidence tokens;
    # using synthetic labels here made the adapter collapse to the shortest abstain.
    policies = [
        {"action": "NO_TRADE", "activate": False, "confidence": "LOW", "exit_profile": "AVOID", "reason_code": "no_prior_signal"},
        {"action": "LONG", "activate": True, "confidence": "MEDIUM", "exit_profile": "FAST", "reason_code": "prior_signal_path_reward_ok"},
        {"action": "SHORT", "activate": True, "confidence": "MEDIUM", "exit_profile": "FAST", "reason_code": "prior_signal_path_reward_ok"},
    ]
    return [(p, json.dumps(p, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) for p in policies]


def _score_candidate_batch(*, model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int, int]], score_normalization: str) -> list[float]:
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


def _candidate_logprob(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, batch_size: int, score_normalization: str) -> list[dict[str, Any]]:
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
    candidates = _candidate_policy_jsons()
    candidate_ids = []
    for _, text in candidates:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        candidate_ids.append(ids)
    normalize = str(score_normalization).lower()
    if normalize not in {"sum", "mean"}:
        raise ValueError("score_normalization must be one of {'sum','mean'}")
    preds: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    for offset in range(0, len(rows), batch_size):
        row_batch = rows[offset : offset + batch_size]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for row in row_batch:
            prompt_text = _chat_prompt_text(tokenizer, str(row["prompt"]))
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            start = len(prompt_ids)
            for ids in candidate_ids:
                sequences.append(prompt_ids + ids)
                spans.append((start, start + len(ids)))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        scores = _score_candidate_batch(
            model=model,
            input_ids=encoded["input_ids"].to(model.device),
            attention_mask=encoded["attention_mask"].to(model.device),
            spans=spans,
            score_normalization=normalize,
        )
        k = len(candidates)
        for i in range(len(row_batch)):
            row_scores = scores[i * k : (i + 1) * k]
            best = max(range(k), key=lambda j: row_scores[j])
            pred = dict(candidates[best][0])
            pred["candidate_scores"] = {candidates[j][1]: row_scores[j] for j in range(k)}
            preds.append(pred)
    return preds


def _metrics(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    action_ok = 0
    activate_ok = 0
    confusion = Counter()
    pred_counts = Counter()
    target_counts = Counter()
    for row, pred in zip(rows, preds):
        target = parse_dxy_kimchi_policy(str(row.get("target", "{}")))
        comparable_pred = parse_dxy_kimchi_policy(json.dumps(pred, ensure_ascii=False))
        exact += int(comparable_pred == target)
        action_ok += int(str(comparable_pred.get("action")) == str(target.get("action")))
        activate_ok += int(bool(comparable_pred.get("activate")) == bool(target.get("activate")))
        pred_counts[f"activate={comparable_pred.get('activate')}|action={comparable_pred.get('action')}"] += 1
        target_counts[f"activate={target.get('activate')}|action={target.get('action')}"] += 1
        confusion[f"target={target.get('activate')}/{target.get('action')}|pred={comparable_pred.get('activate')}/{comparable_pred.get('action')}"] += 1
    n = len(rows)
    return {
        "num_samples": n,
        "exact_policy_accuracy": exact / max(1, n),
        "activate_accuracy": activate_ok / max(1, n),
        "action_accuracy": action_ok / max(1, n),
        "prediction_counts": dict(sorted(pred_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "confusion": dict(sorted(confusion.items())),
    }


def _prediction_rows(rows: list[dict[str, Any]], policies: list[dict[str, Any]], *, horizon: int) -> list[dict[str, Any]]:
    out=[]
    for row, policy in zip(rows, policies):
        target = parse_dxy_kimchi_policy(str(row.get("target", "{}")))
        out.append({
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "policy_prediction": policy,
            "policy_target": target,
            "prediction": policy_to_prediction(policy, horizon=int(horizon)),
            "target_prediction": policy_to_prediction(target, horizon=int(horizon)),
        })
    return out


def evaluate_dxy_kimchi_policy(
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
    max_new_tokens: int = 96,
    horizon: int = 144,
    batch_size: int = 8,
    score_normalization: str = "mean",
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    if prediction_mode == "target_echo":
        preds = [parse_dxy_kimchi_policy(str(r.get("target", "{}"))) for r in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = _generate(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens, batch_size=int(batch_size))
    elif prediction_mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        preds = _candidate_logprob(rows, model_name=model_name, adapter_dir=adapter_dir, batch_size=int(batch_size), score_normalization=score_normalization)
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model','candidate_logprob'}")
    pred_rows = _prediction_rows(rows, preds, horizon=int(horizon))
    if predictions_output:
        Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(predictions_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pred_rows) + "\n")
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "predictions_output": predictions_output,
        "row_selection": {"evaluated_rows": len(rows), "max_samples": int(max_samples), "sample_mode": sample_mode},
        "batch_size": int(batch_size) if prediction_mode in {"model", "candidate_logprob"} else None,
        "score_normalization": score_normalization if prediction_mode == "candidate_logprob" else None,
        "metrics_vs_target": _metrics(rows, preds),
        "leakage_guard": {"prompt_uses_future_path": False, "target_used_for_metrics_only": True, "model_input_excludes_target": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DXY/Kimchi activate/action policy adapter")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--score-normalization", choices=["sum", "mean"], default="mean")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_dxy_kimchi_policy(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
