"""Evaluate side-map rationale candidates by logprob scoring."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import resolve_vlm_model_alias
from training.build_event_side_rationale_preference import build_prompt, candidate_response, read_jsonl, target_side_pair
from utils import disable_transformers_allocator_warmup

CANDIDATES = ("normal", "inverse")


@dataclass(frozen=True)
class EvalEventSideRationalePreferenceCfg:
    eval_jsonl: str
    output_json: str
    model_name: str = "gemma4-e4b-it"
    adapter_dir: str = ""
    batch_size: int = 4
    score_normalization: str = "mean"
    prior_json: str = ""
    prior_weight: float = 1.0


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
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
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    model.eval()
    return tokenizer, model, resolved


def _score_batch(model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int, int]], normalize: str) -> list[float]:
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    scores: list[float] = []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected = logits[i, positions, :].float()
        label_logits = selected.gather(1, labels.reshape(-1, 1)).squeeze(1)
        token_scores = label_logits - torch.logsumexp(selected, dim=-1)
        score = token_scores.mean() if normalize == "mean" else token_scores.sum()
        scores.append(float(score.detach().cpu()))
    return scores



def _load_prior_scores(path: str) -> dict[int, dict[str, float]]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text())
    out: dict[int, dict[str, float]] = {}
    for pred in obj.get("predictions", []):
        idx = int(pred.get("index"))
        raw_scores = pred.get("raw_scores", pred.get("scores", {}))
        out[idx] = {str(k).lower(): float(v) for k, v in dict(raw_scores).items()}
    return out


def _adjust_scores(scores: dict[str, float], prior_scores: dict[str, float] | None, prior_weight: float) -> dict[str, float]:
    if not prior_scores:
        return dict(scores)
    return {label: float(score) - float(prior_weight) * float(prior_scores.get(label, 0.0)) for label, score in scores.items()}

def evaluate(cfg: EvalEventSideRationalePreferenceCfg) -> dict[str, Any]:
    normalize = str(cfg.score_normalization).strip().lower()
    if normalize not in {"mean", "sum"}:
        raise ValueError("score_normalization must be mean or sum")
    rows = [r for r in read_jsonl(cfg.eval_jsonl) if target_side_pair(r)]
    prior_by_index = _load_prior_scores(cfg.prior_json)
    tokenizer, model, resolved = _load_model(cfg.model_name, cfg.adapter_dir)
    flat: list[tuple[int, str, list[int], int]] = []
    for i, row in enumerate(rows):
        prompt_ids = tokenizer(_chat_prompt(tokenizer, build_prompt(row)), add_special_tokens=False)["input_ids"]
        for label in CANDIDATES:
            cand_ids = tokenizer(candidate_response(row, label), add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                cand_ids = cand_ids + [int(tokenizer.eos_token_id)]
            flat.append((i, label, prompt_ids + cand_ids, len(prompt_ids)))
    batch_size = max(1, int(cfg.batch_size))
    scored: list[tuple[int, str, float]] = []
    for offset in range(0, len(flat), batch_size):
        chunk = flat[offset : offset + batch_size]
        enc = tokenizer.pad({"input_ids": [x[2] for x in chunk]}, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        scores = _score_batch(model, input_ids, attention_mask, [(x[3], len(x[2])) for x in chunk], normalize)
        scored.extend((idx, label, score) for (idx, label, _, _), score in zip(chunk, scores))
    by_row: dict[int, dict[str, float]] = {}
    for idx, label, score in scored:
        by_row.setdefault(idx, {})[label] = score
    predictions: list[dict[str, Any]] = []
    correct = 0
    counts: dict[str, int] = {}
    confusion: dict[str, int] = {}
    for i, row in enumerate(rows):
        raw_scores = by_row[i]
        scores = _adjust_scores(raw_scores, prior_by_index.get(i), float(cfg.prior_weight))
        pred = max(CANDIDATES, key=lambda k: scores[k])
        target = target_side_pair(row)
        correct += int(pred == target)
        counts[pred] = counts.get(pred, 0) + 1
        confusion[f"target={target.upper()}|pred={pred.upper()}"] = confusion.get(f"target={target.upper()}|pred={pred.upper()}", 0) + 1
        predictions.append({
            "index": i,
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prediction": pred.upper(),
            "target": target.upper(),
            "scores": scores,
            "raw_scores": raw_scores,
            "prior_scores": prior_by_index.get(i, {}),
        })
    report = {
        "config": asdict(cfg),
        "model_name_resolved": resolved,
        "metrics": {
            "num_samples": len(rows),
            "accuracy": correct / max(1, len(rows)),
            "prediction_counts": dict(sorted(counts.items())),
            "confusion": dict(sorted(confusion.items())),
        },
        "predictions": predictions,
        "leakage_guard": {
            "candidate_rationales_recomputed_from_signal_time_tokens": True,
            "target_used_for_metrics_only": True,
            "prior_scores_use_same_signal_time_candidates_when_provided": bool(cfg.prior_json),
        },
    }
    Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate event side rationale preference scoring")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--model-name", default=EvalEventSideRationalePreferenceCfg.model_name)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    p.add_argument("--prior-json", default="", help="Optional base/prior eval JSON whose candidate scores are subtracted by index")
    p.add_argument("--prior-weight", type=float, default=1.0, help="Multiplier for prior score subtraction")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate(EvalEventSideRationalePreferenceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
