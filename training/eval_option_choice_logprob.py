"""Evaluate A/B/C option-choice rows by single-token option logprob."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

OPTIONS = ["A", "B", "C"]


@dataclass(frozen=True)
class OptionChoiceEvalCfg:
    eval_jsonl: str
    output: str
    predictions_jsonl: str = ""
    model_name: str = RECOMMENDED_VLM_MODEL
    adapter_dir: str = ""
    max_samples: int = 512
    sample_mode: str = "random"
    seed: int = 42
    batch_size: int = 16
    max_length: int = 2048


def _load(path: str, max_samples: int, sample_mode: str, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if max_samples and int(max_samples) < len(rows):
        rng = random.Random(int(seed))
        mode = str(sample_mode).lower()
        if mode == "sequential":
            return rows[: int(max_samples)]
        if mode == "random":
            idx = sorted(rng.sample(range(len(rows)), int(max_samples)))
            return [rows[i] for i in idx]
        if mode == "balanced":
            buckets: dict[str, list[int]] = {}
            for i, row in enumerate(rows):
                buckets.setdefault(str(row.get("target", "")), []).append(i)
            per = max(1, int(max_samples) // max(1, len(buckets)))
            selected: list[int] = []
            for key in sorted(buckets):
                vals = list(buckets[key])
                rng.shuffle(vals)
                selected.extend(vals[: min(per, len(vals))])
            if len(selected) < int(max_samples):
                used = set(selected)
                rest = [i for vals in buckets.values() for i in vals if i not in used]
                rng.shuffle(rest)
                selected.extend(rest[: int(max_samples) - len(selected)])
            return [rows[i] for i in sorted(selected[: int(max_samples)])]
        raise ValueError("sample_mode must be sequential, random, or balanced")
    return rows


def _chat(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _load_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model, resolved


def _score_options(tokenizer: Any, model: Any, prompts: list[str], batch_size: int, max_length: int) -> list[dict[str, float]]:
    import torch

    prefixes = [_chat(tokenizer, p) for p in prompts]
    texts: list[str] = []
    prefix_lens: list[int] = []
    opt_labels: list[str] = []
    for prefix in prefixes:
        prefix_len = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[-1]
        for opt in OPTIONS:
            texts.append(prefix + opt)
            prefix_lens.append(prefix_len)
            opt_labels.append(opt)
    scores_flat: list[float] = []
    for start in range(0, len(texts), int(batch_size)):
        enc = tokenizer(texts[start : start + int(batch_size)], return_tensors="pt", padding=True, truncation=True, max_length=int(max_length)).to(model.device)
        pls = prefix_lens[start : start + int(batch_size)]
        lengths = enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1, :]
        target_ids = enc["input_ids"][:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        for i, (prefix_len, length) in enumerate(zip(pls, lengths)):
            a = max(0, int(prefix_len) - 1)
            b = max(a, int(length) - 1)
            scores_flat.append(float(token_logp[i, a:b].sum().detach().cpu()))
    grouped: list[dict[str, float]] = []
    for i in range(0, len(scores_flat), len(OPTIONS)):
        grouped.append({opt: scores_flat[i + j] for j, opt in enumerate(OPTIONS)})
    return grouped


def run(cfg: OptionChoiceEvalCfg) -> dict[str, Any]:
    rows = _load(cfg.eval_jsonl, int(cfg.max_samples), cfg.sample_mode, int(cfg.seed))
    tokenizer, model, resolved = _load_model(cfg.model_name, cfg.adapter_dir)
    scores = _score_options(tokenizer, model, [str(r["prompt"]) for r in rows], int(cfg.batch_size), int(cfg.max_length))
    pred_rows = []
    counts = Counter(str(r.get("target", "")) for r in rows)
    pred_counts: Counter[str] = Counter()
    correct_by_target: Counter[str] = Counter()
    for row, score in zip(rows, scores):
        pred = max(score.items(), key=lambda kv: kv[1])[0]
        target = str(row.get("target", ""))
        pred_counts[pred] += 1
        correct_by_target[target] += int(pred == target)
        pred_rows.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "target": target, "prediction": pred, "scores": score, "correct": pred == target, "choice_utility": row.get("choice_utility")})
    total_correct = sum(1 for r in pred_rows if r["correct"])
    n = max(1, len(rows))
    report = {
        "config": asdict(cfg),
        "model_name_resolved": resolved,
        "rows": len(rows),
        "accuracy": total_correct / n,
        "correct": total_correct,
        "target_counts": dict(sorted(counts.items())),
        "prediction_counts": dict(sorted(pred_counts.items())),
        "accuracy_by_target": {k: correct_by_target[k] / counts[k] for k in sorted(counts)},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if cfg.predictions_jsonl:
        Path(cfg.predictions_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.predictions_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pred_rows) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-jsonl", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=OptionChoiceEvalCfg.max_samples)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default=OptionChoiceEvalCfg.sample_mode)
    p.add_argument("--seed", type=int, default=OptionChoiceEvalCfg.seed)
    p.add_argument("--batch-size", type=int, default=OptionChoiceEvalCfg.batch_size)
    p.add_argument("--max-length", type=int, default=OptionChoiceEvalCfg.max_length)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(OptionChoiceEvalCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
