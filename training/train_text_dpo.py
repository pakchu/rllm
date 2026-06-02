"""LoRA DPO training entrypoint for text trader preference pairs."""

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


@dataclass(frozen=True)
class TextDPOConfig:
    model_name: str = RECOMMENDED_VLM_MODEL
    train_jsonl: str = "data/text_step_pref_sft.jsonl"
    output_dir: str = "checkpoints/text_dpo"
    max_samples: int = 0
    max_length: int = 2048
    sample_mode: str = "sequential"
    max_steps: int = 50
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-7
    beta: float = 0.1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42


def _action_bucket(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return str(text)[:80]
    if isinstance(obj, dict):
        return f"gate={obj.get('gate')},side={obj.get('side')},hold={obj.get('hold_bars')}"
    return str(obj)[:80]


def _select_rows(rows: list[dict[str, Any]], *, max_samples: int, sample_mode: str, seed: int) -> list[dict[str, Any]]:
    if not max_samples or int(max_samples) >= len(rows):
        return rows
    mode = str(sample_mode).strip().lower()
    if mode not in {"sequential", "random", "balanced"}:
        raise ValueError("sample_mode must be one of {'sequential','random','balanced'}")
    rng = random.Random(int(seed))
    max_n = int(max_samples)
    if mode == "sequential":
        return rows[:max_n]
    if mode == "random":
        idx = sorted(rng.sample(range(len(rows)), max_n))
        return [rows[i] for i in idx]
    buckets: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        buckets.setdefault(_action_bucket(str(row.get("chosen", ""))), []).append(i)
    per_bucket = max(1, max_n // max(1, len(buckets)))
    selected: list[int] = []
    for key in sorted(buckets):
        idxs = list(buckets[key])
        rng.shuffle(idxs)
        selected.extend(idxs[: min(per_bucket, len(idxs))])
    if len(selected) < max_n:
        used = set(selected)
        rest = [i for i in range(len(rows)) if i not in used]
        rng.shuffle(rest)
        selected.extend(rest[: max_n - len(selected)])
    return [rows[i] for i in sorted(selected[:max_n])]


def load_preference_jsonl(path: str | Path, *, max_samples: int = 0, sample_mode: str = "sequential", seed: int = 42) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    for row in rows[:5]:
        for key in ("prompt", "chosen", "rejected"):
            if key not in row:
                raise ValueError(f"preference row lacks {key}")
    return _select_rows(rows, max_samples=max_samples, sample_mode=sample_mode, seed=seed)


def summarize_rows(rows: list[dict[str, Any]], cfg: TextDPOConfig, resolved_model: str) -> dict[str, Any]:
    chosen_counts = Counter(_action_bucket(str(r["chosen"])) for r in rows)
    rejected_counts = Counter(_action_bucket(str(r["rejected"])) for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    return {
        "model_name": resolved_model,
        "train_jsonl": str(Path(cfg.train_jsonl).resolve()),
        "output_dir": cfg.output_dir,
        "rows": len(rows),
        "chosen_counts": dict(sorted(chosen_counts.items())),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "prompt_chars": {"min": min(prompt_lens), "max": max(prompt_lens), "mean": sum(prompt_lens) / len(prompt_lens)},
        "config": asdict(cfg),
    }


def _dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [{"role": "user", "content": str(r["prompt"])}],
            "chosen": [{"role": "assistant", "content": str(r["chosen"])}],
            "rejected": [{"role": "assistant", "content": str(r["rejected"])}],
        }
        for r in rows
    ]


def train_text_dpo(cfg: TextDPOConfig, *, dry_run: bool = False) -> dict[str, Any]:
    resolved_model = resolve_vlm_model_alias(cfg.model_name, prefer_latest=True)
    rows = load_preference_jsonl(cfg.train_jsonl, max_samples=cfg.max_samples, sample_mode=cfg.sample_mode, seed=cfg.seed)
    summary = summarize_rows(rows, cfg, resolved_model)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(cfg.output_dir) / "dpo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if dry_run:
        return {**summary, "dry_run": True, "summary_path": str(summary_path)}

    disable_transformers_allocator_warmup()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(resolved_model, trust_remote_code=True, device_map="auto")
    peft_config = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    args = DPOConfig(
        output_dir=cfg.output_dir,
        max_steps=int(cfg.max_steps),
        num_train_epochs=float(cfg.num_train_epochs),
        learning_rate=float(cfg.learning_rate),
        beta=float(cfg.beta),
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        logging_steps=1,
        save_steps=max(1, int(cfg.max_steps)),
        seed=int(cfg.seed),
        bf16=True,
        report_to=[],
        max_length=int(cfg.max_length),
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list(_dataset_rows(rows)),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return {**summary, "dry_run": False, "summary_path": str(summary_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune text trader with LoRA DPO on preference pairs")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TextDPOConfig(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        sample_mode=args.sample_mode,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )
    print(json.dumps(train_text_dpo(cfg, dry_run=bool(args.dry_run)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
