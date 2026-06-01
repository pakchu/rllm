"""Text-only LoRA SFT entrypoint for analyzer/trader LLM stages."""

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
class TextSFTConfig:
    model_name: str = RECOMMENDED_VLM_MODEL
    train_jsonl: str = "data/text_trader_sft.jsonl"
    output_dir: str = "checkpoints/text_sft"
    max_samples: int = 0
    max_seq_length: int = 2048
    sample_mode: str = "sequential"
    max_steps: int = 50
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = False
    seed: int = 42


def _row_bucket(row: dict[str, Any]) -> str:
    target = str(row.get("target", ""))
    try:
        parsed = json.loads(target)
    except Exception:
        return target[:80]
    if isinstance(parsed, dict) and "gate" in parsed:
        return f"gate={parsed.get('gate')},side={parsed.get('side')}"
    if isinstance(parsed, dict):
        return str(parsed.get("regime", row.get("task", "unknown")))
    return str(parsed)[:80]


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
        buckets.setdefault(_row_bucket(row), []).append(i)
    per_bucket = max(1, max_n // max(1, len(buckets)))
    selected: list[int] = []
    for bucket in sorted(buckets):
        idxs = list(buckets[bucket])
        rng.shuffle(idxs)
        selected.extend(idxs[: min(per_bucket, len(idxs))])
    if len(selected) < max_n:
        remaining = [i for i in range(len(rows)) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_n - len(selected)])
    selected = sorted(selected[:max_n])
    return [rows[i] for i in selected]


def load_jsonl(path: str | Path, *, max_samples: int = 0, sample_mode: str = "sequential", seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return _select_rows(rows, max_samples=int(max_samples), sample_mode=sample_mode, seed=int(seed))


def _target_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        target = str(row.get("target", ""))
        try:
            parsed = json.loads(target)
            if isinstance(parsed, dict) and "gate" in parsed:
                counts[f"gate={parsed.get('gate')},side={parsed.get('side')}"] += 1
            elif isinstance(parsed, dict):
                for key in ("regime", "risk_state", "trend_alignment", "location"):
                    if key in parsed:
                        counts[f"{key}={parsed[key]}"] += 1
            else:
                counts[target[:80]] += 1
        except Exception:
            counts[target[:80]] += 1
    return dict(counts)


def build_training_text(row: dict[str, Any], tokenizer: Any | None = None) -> str:
    prompt = str(row["prompt"])
    target = str(row["target"])
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return f"<|user|>\n{prompt}\n<|assistant|>\n{target}"


def summarize_rows(rows: list[dict[str, Any]], cfg: TextSFTConfig, resolved_model: str) -> dict[str, Any]:
    tasks: Counter[str] = Counter(str(r.get("task", "unknown")) for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    target_lens = [len(str(r.get("target", ""))) for r in rows]
    return {
        "model_name": resolved_model,
        "train_jsonl": str(Path(cfg.train_jsonl).resolve()),
        "output_dir": cfg.output_dir,
        "rows": len(rows),
        "tasks": dict(tasks),
        "target_counts": _target_counter(rows),
        "prompt_chars": {"min": min(prompt_lens), "max": max(prompt_lens), "mean": sum(prompt_lens) / len(prompt_lens)},
        "target_chars": {"min": min(target_lens), "max": max(target_lens), "mean": sum(target_lens) / len(target_lens)},
        "config": asdict(cfg),
    }


def train_text_sft(cfg: TextSFTConfig, *, dry_run: bool = False) -> dict[str, Any]:
    resolved_model = resolve_vlm_model_alias(cfg.model_name, prefer_latest=True)
    rows = load_jsonl(cfg.train_jsonl, max_samples=cfg.max_samples, sample_mode=cfg.sample_mode, seed=cfg.seed)
    summary = summarize_rows(rows, cfg, resolved_model)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(cfg.output_dir) / "sft_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if dry_run:
        return {**summary, "dry_run": True, "summary_path": str(summary_path)}

    disable_transformers_allocator_warmup()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list([{"text": build_training_text(row, tokenizer)} for row in rows])
    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    peft_config = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    args = SFTConfig(
        output_dir=cfg.output_dir,
        max_steps=int(cfg.max_steps),
        num_train_epochs=float(cfg.num_train_epochs),
        learning_rate=float(cfg.learning_rate),
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        logging_steps=1,
        save_steps=max(1, int(cfg.max_steps)),
        seed=int(cfg.seed),
        bf16=True,
        report_to=[],
        max_length=int(cfg.max_seq_length),
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return {**summary, "dry_run": False, "summary_path": str(summary_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Gemma/Qwen text analyzer or trader with LoRA SFT")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TextSFTConfig(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        sample_mode=args.sample_mode,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
    )
    print(json.dumps(train_text_sft(cfg, dry_run=bool(args.dry_run)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
