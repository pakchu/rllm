"""Evaluate preference adapters by chosen-vs-rejected response logprob margins."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup


@dataclass(frozen=True)
class PreferenceLogprobEvalCfg:
    eval_jsonl: str
    output: str
    predictions_jsonl: str = ""
    model_name: str = RECOMMENDED_VLM_MODEL
    adapter_dir: str = ""
    max_samples: int = 256
    sample_mode: str = "random"
    seed: int = 42
    batch_size: int = 4
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
        if mode == "gate_balanced":
            buckets: dict[str, list[int]] = {}
            for i, row in enumerate(rows):
                try:
                    obj = json.loads(str(row.get("chosen", "{}")))
                    key = f"{obj.get('gate')}:{obj.get('side')}" if isinstance(obj, dict) else "UNKNOWN"
                except Exception:
                    key = "UNKNOWN"
                buckets.setdefault(key, []).append(i)
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
        raise ValueError("sample_mode must be sequential, random, or gate_balanced")
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


def _score_texts(tokenizer: Any, model: Any, prompts: list[str], responses: list[str], batch_size: int, max_length: int) -> list[float]:
    import torch

    texts: list[str] = []
    prefix_lens: list[int] = []
    for prompt, response in zip(prompts, responses):
        prefix = _chat(tokenizer, str(prompt))
        text = prefix + str(response)
        prefix_lens.append(tokenizer(prefix, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[-1])
        texts.append(text)

    scores: list[float] = []
    for start in range(0, len(texts), int(batch_size)):
        enc = tokenizer(
            texts[start : start + int(batch_size)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        ).to(model.device)
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
            scores.append(float(token_logp[i, a:b].sum().detach().cpu()))
    return scores


def _bucket(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return "UNKNOWN"
    if not isinstance(obj, dict):
        return "UNKNOWN"
    return f"{obj.get('gate')}:{obj.get('side')}"


def run(cfg: PreferenceLogprobEvalCfg) -> dict[str, Any]:
    rows = _load(cfg.eval_jsonl, int(cfg.max_samples), cfg.sample_mode, int(cfg.seed))
    tokenizer, model, resolved = _load_model(cfg.model_name, cfg.adapter_dir)
    prompts = [str(r["prompt"]) for r in rows]
    chosen = [str(r["chosen"]) for r in rows]
    rejected = [str(r["rejected"]) for r in rows]
    chosen_scores = _score_texts(tokenizer, model, prompts, chosen, int(cfg.batch_size), int(cfg.max_length))
    rejected_scores = _score_texts(tokenizer, model, prompts, rejected, int(cfg.batch_size), int(cfg.max_length))
    pred_rows = []
    margins = []
    correct = 0
    pair_counts: Counter[str] = Counter()
    pair_correct: Counter[str] = Counter()
    for row, cs, rs in zip(rows, chosen_scores, rejected_scores):
        margin = float(cs - rs)
        margins.append(margin)
        ok = margin > 0.0
        correct += int(ok)
        key = f"{_bucket(str(row['chosen']))}>{_bucket(str(row['rejected']))}"
        pair_counts[key] += 1
        pair_correct[key] += int(ok)
        pred_rows.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "chosen": row.get("chosen"),
                "rejected": row.get("rejected"),
                "chosen_score": cs,
                "rejected_score": rs,
                "margin": margin,
                "correct": ok,
                "utility_gap": row.get("utility_gap"),
            }
        )
    n = max(1, len(rows))
    mean = sum(margins) / n
    var = sum((x - mean) ** 2 for x in margins) / n
    std = math.sqrt(var)
    report = {
        "config": asdict(cfg),
        "model_name_resolved": resolved,
        "rows": len(rows),
        "accuracy": correct / n,
        "correct": correct,
        "margin": {"mean": mean, "std": std, "min": min(margins) if margins else 0.0, "max": max(margins) if margins else 0.0},
        "pair_counts": dict(sorted(pair_counts.items())),
        "pair_accuracy": {k: pair_correct[k] / pair_counts[k] for k in sorted(pair_counts)},
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
    p.add_argument("--max-samples", type=int, default=PreferenceLogprobEvalCfg.max_samples)
    p.add_argument("--sample-mode", choices=["sequential", "random", "gate_balanced"], default=PreferenceLogprobEvalCfg.sample_mode)
    p.add_argument("--seed", type=int, default=PreferenceLogprobEvalCfg.seed)
    p.add_argument("--batch-size", type=int, default=PreferenceLogprobEvalCfg.batch_size)
    p.add_argument("--max-length", type=int, default=PreferenceLogprobEvalCfg.max_length)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PreferenceLogprobEvalCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
