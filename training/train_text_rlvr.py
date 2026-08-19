"""Generic text-only SFT-to-RLVR training with TRL GRPO.

The input is deliberately small and closed-world: each JSONL row must contain a
text ``prompt`` and a ``target`` from the selected label schema.  There is no
out-of-sample/evaluation input in this entrypoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


LABEL_SCHEMAS: dict[str, tuple[str, ...]] = {
    "gate": ("NO_TRADE", "TRADE"),
    "gate_utility": ("NO_TRADE", "TRADE"),
    "pair": ("A", "B"),
    "ordinal": ("Q0", "Q1", "Q2", "Q3", "Q4"),
    "pposm_state": ("SKIP", "TP4", "TP12"),
    "pposm_action_utility": ("SKIP", "TP4", "TP12"),
}


@dataclass(frozen=True)
class TextRLVRConfig:
    base_model: str
    sft_adapter_dir: str
    train_jsonl: str
    output_dir: str
    label_schema: str
    max_samples: int = 0
    sample_mode: str = "sequential"
    max_steps: int = 50
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    max_prompt_length: int = 2048
    max_completion_length: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    seed: int = 42
    reward_variance_guard: str = "auto"
    scale_rewards: str = "group"
    require_nonzero_reward_std: bool = False
    require_nonzero_gradient: bool = False
    min_reward_std: float = 0.0
    min_gradient_norm: float = 0.0
    local_files_only: bool = False
    trust_remote_code: bool = False
    bf16: bool = False
    utility_scale: float = 0.005


def allowed_labels(label_schema: str) -> tuple[str, ...]:
    key = str(label_schema).strip().lower()
    if key not in LABEL_SCHEMAS:
        raise ValueError(
            "label_schema must be one of "
            "{'gate','gate_utility','pair','ordinal','pposm_state','pposm_action_utility'}"
        )
    return LABEL_SCHEMAS[key]


def load_jsonl(
    path: str | Path,
    *,
    label_schema: str,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    """Load and strictly validate prompt/target JSONL rows."""
    source = Path(path)
    labels = set(allowed_labels(label_schema))
    rows: list[dict[str, str]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {source}: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number} of {source} must be a JSON object")
            if "prompt" not in raw or "target" not in raw:
                raise ValueError(f"line {line_number} of {source} must contain prompt and target")
            prompt, target = raw["prompt"], raw["target"]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"line {line_number} of {source} has a non-string or empty prompt")
            if not isinstance(target, str) or target not in labels:
                raise ValueError(
                    f"line {line_number} of {source} target must be exactly one of {sorted(labels)}"
                )
            record: dict[str, Any] = {"prompt": prompt, "target": target}
            if str(label_schema).strip().lower() == "gate_utility":
                metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
                utility = metadata.get("net_return", raw.get("utility"))
                try:
                    utility = float(utility)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"line {line_number} of {source} lacks finite metadata.net_return"
                    ) from exc
                if not math.isfinite(utility):
                    raise ValueError(f"line {line_number} of {source} has non-finite utility")
                record["utility"] = utility
            if str(label_schema).strip().lower() == "pposm_action_utility":
                metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
                utilities = metadata.get("action_utilities")
                if not isinstance(utilities, dict) or set(utilities) != labels:
                    raise ValueError(
                        f"line {line_number} of {source} action_utilities must match labels"
                    )
                parsed = {label: float(utilities[label]) for label in labels}
                if not all(math.isfinite(value) for value in parsed.values()):
                    raise ValueError(f"line {line_number} of {source} has non-finite action utility")
                record["action_utilities"] = parsed
            rows.append(record)
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
    if not rows:
        raise ValueError(f"no training rows loaded from {source}")
    return rows


def completion_text(completion: Any) -> str:
    """Return text from TRL plain or conversational completion values."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        value = completion.get("content", "")
        return value if isinstance(value, str) else str(value)
    if isinstance(completion, (list, tuple)) and completion:
        return completion_text(completion[-1])
    return ""


def make_format_reward(label_schema: str) -> Callable[..., list[float]]:
    labels = frozenset(allowed_labels(label_schema))

    def format_reward(completions: Sequence[Any], **_: Any) -> list[float]:
        """Reward only an exact, bare, allowed label (whitespace is invalid)."""
        return [0.2 if completion_text(item) in labels else -0.5 for item in completions]

    format_reward.__name__ = "format_reward"
    return format_reward


def exact_target_reward(
    completions: Sequence[Any], target: Sequence[str], **_: Any
) -> list[float]:
    if len(completions) != len(target):
        raise ValueError("completions and target must have equal lengths")
    return [
        1.0 if completion_text(completion) == expected else 0.0
        for completion, expected in zip(completions, target)
    ]


def ordinal_distance_reward(
    completions: Sequence[Any], target: Sequence[str], **_: Any
) -> list[float]:
    """Reward valid ordinal labels by linearly decreasing distance from target."""
    if len(completions) != len(target):
        raise ValueError("completions and target must have equal lengths")
    positions = {label: index for index, label in enumerate(LABEL_SCHEMAS["ordinal"])}
    rewards: list[float] = []
    for completion, expected in zip(completions, target):
        predicted = completion_text(completion)
        if predicted not in positions or expected not in positions:
            rewards.append(-1.0)
        else:
            distance = abs(positions[predicted] - positions[expected])
            rewards.append(max(-1.0, 1.0 - 0.5 * distance))
    return rewards


def make_economic_utility_reward(utility_scale: float) -> Callable[..., list[float]]:
    scale = float(utility_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("utility_scale must be positive and finite")

    def economic_utility_reward(
        completions: Sequence[Any], utility: Sequence[float], **_: Any
    ) -> list[float]:
        if len(completions) != len(utility):
            raise ValueError("completions and utility must have equal lengths")
        rewards: list[float] = []
        for completion, raw_utility in zip(completions, utility):
            label = completion_text(completion)
            value = float(raw_utility)
            if label == "TRADE":
                rewards.append(float(max(-1.0, min(1.0, value / scale))))
            elif label == "NO_TRADE":
                rewards.append(0.0)
            else:
                rewards.append(-1.0)
        return rewards

    economic_utility_reward.__name__ = "economic_utility_reward"
    return economic_utility_reward


def make_action_utility_reward(utility_scale: float) -> Callable[..., list[float]]:
    scale = float(utility_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("utility_scale must be positive and finite")

    def action_utility_reward(
        completions: Sequence[Any], action_utilities: Sequence[dict[str, float]], **_: Any
    ) -> list[float]:
        if len(completions) != len(action_utilities):
            raise ValueError("completions and action_utilities must have equal lengths")
        rewards: list[float] = []
        for completion, utilities in zip(completions, action_utilities):
            label = completion_text(completion)
            if label not in utilities:
                rewards.append(-1.0)
                continue
            rewards.append(float(max(-1.0, min(1.0, float(utilities[label]) / scale))))
        return rewards

    action_utility_reward.__name__ = "action_utility_reward"
    return action_utility_reward


def build_reward_functions(
    label_schema: str, *, utility_scale: float = 0.005
) -> list[Callable[..., list[float]]]:
    schema = str(label_schema).strip().lower()
    rewards: list[Callable[..., list[float]]] = [make_format_reward(schema)]
    if schema == "gate_utility":
        rewards.append(make_economic_utility_reward(utility_scale))
        return rewards
    if schema == "pposm_action_utility":
        rewards.append(make_action_utility_reward(utility_scale))
        return rewards
    rewards.append(exact_target_reward)
    if schema == "ordinal":
        rewards.append(ordinal_distance_reward)
    else:
        allowed_labels(schema)  # validate before returning
    return rewards


def apply_reward_variance_guard(
    *, num_generations: int, dataset_size: int, mode: str
) -> tuple[int, list[str]]:
    """Ensure GRPO has at least two alternatives in each reward group."""
    guard = str(mode).strip().lower()
    if guard not in {"auto", "off", "error"}:
        raise ValueError("reward_variance_guard must be one of {'auto','off','error'}")
    generations = int(num_generations)
    if generations < 1:
        raise ValueError("num_generations must be positive")
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    notes: list[str] = []
    if generations < 2:
        if guard == "error":
            raise ValueError("GRPO reward variance guard requires num_generations >= 2")
        if guard == "auto":
            generations = 2
            notes.append("Raised num_generations from 1 to 2 to permit within-group reward variance.")
    return generations, notes


def sample_rows(
    rows: list[dict[str, Any]], *, mode: str, max_samples: int, seed: int
) -> list[dict[str, Any]]:
    key = str(mode).strip().lower()
    if key == "sequential":
        return rows[: int(max_samples)] if int(max_samples) > 0 else rows
    if key != "balanced_oversample":
        raise ValueError("sample_mode must be sequential or balanced_oversample")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["target"]), []).append(row)
    if len(groups) < 2:
        raise ValueError("balanced_oversample requires at least two target classes")
    total = int(max_samples) if int(max_samples) > 0 else max(len(v) for v in groups.values()) * len(groups)
    rng = random.Random(int(seed))
    labels = sorted(groups)
    quotas = {label: total // len(labels) for label in labels}
    for label in labels[: total % len(labels)]:
        quotas[label] += 1
    sampled: list[dict[str, Any]] = []
    for label in labels:
        source = groups[label]
        sampled.extend(source[rng.randrange(len(source))] for _ in range(quotas[label]))
    rng.shuffle(sampled)
    return sampled


def _reward_diagnostics(label_schema: str, *, utility_scale: float = 0.005) -> dict[str, Any]:
    labels = allowed_labels(label_schema)
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    reward_funcs = build_reward_functions(label_schema, utility_scale=utility_scale)
    for target in labels:
        matrix[target] = {}
        for completion in labels:
            values: dict[str, float] = {}
            for reward in reward_funcs:
                if reward.__name__ == "economic_utility_reward":
                    kwargs = {"utility": [0.01]}
                elif reward.__name__ == "action_utility_reward":
                    kwargs = {"action_utilities": [{label: (0.01 if label == target else 0.0) for label in labels}]}
                else:
                    kwargs = {"target": [target]}
                values[reward.__name__] = reward([completion], **kwargs)[0]
            matrix[target][completion] = values
    return {
        "schema": str(label_schema).lower(),
        "allowed_labels": list(labels),
        "reward_functions": [reward.__name__ for reward in reward_funcs],
        "deterministic_reward_matrix": matrix,
        "observed_reward_std": [],
        "max_observed_reward_std": None,
        "status": "not_run",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _metric_values(log_history: Sequence[dict[str, Any]], fragment: str) -> list[float]:
    values: list[float] = []
    for entry in log_history:
        for key, raw in entry.items():
            if fragment not in str(key).lower():
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return values


def train_text_rlvr(cfg: TextRLVRConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Train the existing SFT LoRA adapter with deterministic RLVR rewards."""
    schema = str(cfg.label_schema).strip().lower()
    labels = allowed_labels(schema)
    rows = load_jsonl(cfg.train_jsonl, label_schema=schema, max_samples=0)
    rows = sample_rows(
        rows, mode=cfg.sample_mode, max_samples=cfg.max_samples, seed=cfg.seed
    )
    effective_generations, guard_notes = apply_reward_variance_guard(
        num_generations=cfg.num_generations,
        dataset_size=len(rows),
        mode=cfg.reward_variance_guard,
    )
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_counts = dict(sorted(Counter(row["target"] for row in rows).items()))

    config_diagnostics = {
        "trainer": "trl.GRPOTrainer",
        "trl_required_version": "0.29",
        "text_only": True,
        "train_rows": len(rows),
        "target_counts": target_counts,
        "allowed_labels": list(labels),
        "effective_num_generations": effective_generations,
        "reward_variance_guard_notes": guard_notes,
        "dry_run": bool(dry_run),
        "config": asdict(cfg),
    }
    reward_diagnostics = _reward_diagnostics(schema, utility_scale=cfg.utility_scale)
    gradient_diagnostics: dict[str, Any] = {
        "observed_gradient_norms": [],
        "max_observed_gradient_norm": None,
        "trainable_parameters": None,
        "status": "not_run",
    }
    config_path = output / "config_diagnostics.json"
    reward_path = output / "reward_diagnostics.json"
    gradient_path = output / "gradient_diagnostics.json"
    _write_json(config_path, config_diagnostics)
    _write_json(reward_path, reward_diagnostics)
    _write_json(gradient_path, gradient_diagnostics)

    result = {
        **config_diagnostics,
        "config_diagnostics_path": str(config_path),
        "reward_diagnostics_path": str(reward_path),
        "gradient_diagnostics_path": str(gradient_path),
    }
    if dry_run:
        return result

    # Heavy dependencies remain lazy so validation and dry-runs work offline.
    import torch
    import trl
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if not str(getattr(trl, "__version__", "")).startswith("0.29"):
        raise RuntimeError(f"TRL 0.29 is required, found {getattr(trl, '__version__', 'unknown')}")
    random.seed(int(cfg.seed))
    set_seed(int(cfg.seed), deterministic=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        local_files_only=bool(cfg.local_files_only),
        trust_remote_code=bool(cfg.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if cfg.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        local_files_only=bool(cfg.local_files_only),
        trust_remote_code=bool(cfg.trust_remote_code),
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(
        model,
        cfg.sft_adapter_dir,
        is_trainable=True,
        local_files_only=bool(cfg.local_files_only),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0:
        raise RuntimeError("the SFT LoRA adapter has no trainable parameters")
    gradient_diagnostics["trainable_parameters"] = int(trainable)
    _write_json(gradient_path, gradient_diagnostics)

    generation_batch_size = (
        (int(cfg.per_device_train_batch_size) + effective_generations - 1)
        // effective_generations
    ) * effective_generations
    args = GRPOConfig(
        output_dir=str(output),
        max_steps=int(cfg.max_steps),
        num_train_epochs=float(cfg.num_train_epochs),
        learning_rate=float(cfg.learning_rate),
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        num_generations=effective_generations,
        generation_batch_size=generation_batch_size,
        max_completion_length=int(cfg.max_completion_length),
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        top_k=int(cfg.top_k),
        beta=0.0,
        loss_type="dapo",
        scale_rewards=str(cfg.scale_rewards),
        logging_steps=1,
        save_steps=max(1, int(cfg.max_steps)),
        seed=int(cfg.seed),
        data_seed=int(cfg.seed),
        full_determinism=True,
        bf16=bool(cfg.bf16),
        report_to=[],
    )
    # The SFT adapter was trained with chat-templated user messages.  Passing
    # raw strings here makes GRPO continue the prompt prose instead of entering
    # the assistant turn, collapsing every verifier reward to the invalid
    # completion penalty.
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "target": row["target"],
                **({"utility": row["utility"]} if "utility" in row else {}),
                **({"action_utilities": row["action_utilities"]} if "action_utilities" in row else {}),
            }
            for row in rows
        ]
    ).shuffle(seed=int(cfg.seed))
    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=build_reward_functions(schema, utility_scale=cfg.utility_scale),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    history = list(trainer.state.log_history)
    gradient_norms = _metric_values(history, "grad_norm")
    reward_stds = _metric_values(history, "reward_std")
    max_gradient = max(gradient_norms, default=0.0)
    max_reward_std = max(reward_stds, default=0.0)
    gradient_diagnostics.update(
        observed_gradient_norms=gradient_norms,
        max_observed_gradient_norm=max_gradient,
        status="passed" if max_gradient > float(cfg.min_gradient_norm) else "zero_or_unobserved",
    )
    reward_diagnostics.update(
        observed_reward_std=reward_stds,
        max_observed_reward_std=max_reward_std,
        status="passed" if max_reward_std > float(cfg.min_reward_std) else "zero_or_unobserved",
    )
    _write_json(gradient_path, gradient_diagnostics)
    _write_json(reward_path, reward_diagnostics)

    failures: list[str] = []
    if cfg.require_nonzero_reward_std and max_reward_std <= float(cfg.min_reward_std):
        failures.append(f"reward std was zero or unobserved (max={max_reward_std:.6g})")
    if cfg.require_nonzero_gradient and max_gradient <= float(cfg.min_gradient_norm):
        failures.append(f"gradient norm was zero or unobserved (max={max_gradient:.6g})")
    if failures:
        raise RuntimeError("RLVR verification failed: " + "; ".join(failures))

    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    return {
        **result,
        "dry_run": False,
        "max_observed_reward_std": max_reward_std,
        "max_observed_gradient_norm": max_gradient,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SFT LoRA adapter with text-only GRPO/RLVR")
    parser.add_argument("--base-model", required=True, help="Base causal LM name or local path")
    parser.add_argument("--sft-adapter-dir", required=True, help="Existing SFT LoRA adapter")
    parser.add_argument("--train-jsonl", required=True, help="Training JSONL with prompt/target rows")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-schema", required=True, choices=sorted(LABEL_SCHEMAS))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--sample-mode",
        choices=["sequential", "balanced_oversample"],
        default="sequential",
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-variance-guard", choices=["auto", "off", "error"], default="auto")
    parser.add_argument("--scale-rewards", choices=["group", "batch", "none"], default="group")
    parser.add_argument("--require-nonzero-reward-std", action="store_true")
    parser.add_argument("--require-nonzero-gradient", "--require-nonzero-grad", action="store_true")
    parser.add_argument("--min-reward-std", type=float, default=0.0)
    parser.add_argument("--min-gradient-norm", type=float, default=0.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--utility-scale", type=float, default=0.005)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = TextRLVRConfig(
        **{
            key: value
            for key, value in vars(args).items()
            if key != "dry_run"
        }
    )
    print(json.dumps(train_text_rlvr(cfg, dry_run=bool(args.dry_run)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
