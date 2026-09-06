"""Score closed-world text labels by conditional mean token log probability.

This entrypoint is deliberately offline: both the base model and LoRA adapter
are loaded with ``local_files_only=True``.  It emits one identity-preserving
record per input row and never generates text or samples tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LABELS = ("NO_TRADE", "TRADE")


@dataclass(frozen=True)
class Config:
    base_model: str
    adapter_dir: Path
    input_jsonl: Path
    output_jsonl: Path
    labels: tuple[str, ...] = DEFAULT_LABELS
    max_seq_length: int = 2048
    trust_remote_code: bool = False


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            if not isinstance(row.get("prompt"), str) or not row["prompt"]:
                raise ValueError(f"line {line_number} of {path} has no non-empty prompt")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(label) for label in labels)
    if not normalized or any(not label for label in normalized):
        raise ValueError("candidate labels must be non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("candidate labels must be unique")
    return normalized


def chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def candidate_mean_logprobs(
    *,
    prompt: str,
    labels: Sequence[str],
    tokenizer: Any,
    model: Any,
    max_seq_length: int,
) -> dict[str, float]:
    """Return deterministic conditional mean log probabilities for labels.

    Prompt and label IDs are tokenized separately and concatenated.  This
    gives every candidate the exact same prompt token prefix and makes the
    scored target span independent of tokenizer boundary merging.
    """

    import torch

    candidates = _validate_labels(labels)
    if int(max_seq_length) < 2:
        raise ValueError("max_seq_length must be at least 2")
    prompt_ids = list(
        tokenizer(chat_prompt_text(tokenizer, prompt), add_special_tokens=False)[
            "input_ids"
        ]
    )
    if not prompt_ids:
        raise ValueError("tokenized prompt is empty")

    scores: dict[str, float] = {}
    device = _model_device(model)
    for label in candidates:
        label_ids = list(tokenizer(label, add_special_tokens=False)["input_ids"])
        if not label_ids:
            raise ValueError(f"candidate label tokenized to no tokens: {label!r}")
        prompt_budget = int(max_seq_length) - len(label_ids)
        if prompt_budget < 1:
            raise ValueError(
                f"candidate label {label!r} leaves no prompt token at max_seq_length"
            )
        kept_prompt = prompt_ids[-prompt_budget:]
        start = len(kept_prompt)
        input_ids = torch.tensor([kept_prompt + label_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        positions = torch.arange(start - 1, start + len(label_ids) - 1, device=device)
        targets = input_ids[0, start:]
        score = float(log_probs[0, positions, targets].mean().detach().cpu())
        if not math.isfinite(score):
            raise RuntimeError(f"non-finite mean log probability for label {label!r}")
        scores[label] = score
    return scores


def score_rows(
    rows: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    tokenizer: Any,
    model: Any,
    max_seq_length: int = 2048,
) -> list[dict[str, Any]]:
    candidates = _validate_labels(labels)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        target = row.get("target")
        if target is not None and target not in candidates:
            raise ValueError(f"row {index} target is not a candidate label: {target!r}")
        scores = candidate_mean_logprobs(
            prompt=str(row["prompt"]),
            labels=candidates,
            tokenizer=tokenizer,
            model=model,
            max_seq_length=max_seq_length,
        )
        best = max(range(len(candidates)), key=lambda i: (scores[candidates[i]], -i))
        item: dict[str, Any] = {
            "index": index,
            "target": target,
            "labels": list(candidates),
            "label_mean_logprobs": scores,
            "scores": [
                {"label": label, "mean_logprob": scores[label]} for label in candidates
            ],
            "prediction": candidates[best],
        }
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            item["metadata"] = metadata
            if isinstance(metadata.get("identity"), str):
                item["identity"] = metadata["identity"]
        if "TRADE" in scores and "NO_TRADE" in scores:
            item["margin"] = scores["TRADE"] - scores["NO_TRADE"]
        output.append(item)
    return output


def load_model(cfg: Config) -> tuple[Any, Any]:
    """Load a local base causal LM plus a local LoRA adapter."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        local_files_only=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        local_files_only=True,
        trust_remote_code=cfg.trust_remote_code,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        base,
        str(cfg.adapter_dir),
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()
    return tokenizer, model


def run(cfg: Config) -> dict[str, Any]:
    rows = load_jsonl(cfg.input_jsonl)
    tokenizer, model = load_model(cfg)
    scored = score_rows(
        rows,
        labels=cfg.labels,
        tokenizer=tokenizer,
        model=model,
        max_seq_length=cfg.max_seq_length,
    )
    content = b"".join((canonical_json(row) + "\n").encode() for row in scored)
    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_jsonl.write_bytes(content)
    return {
        "protocol": "generic_candidate_label_mean_logprob_v1",
        "config": {
            **asdict(cfg),
            "adapter_dir": str(cfg.adapter_dir),
            "input_jsonl": str(cfg.input_jsonl),
            "output_jsonl": str(cfg.output_jsonl),
            "labels": list(cfg.labels),
            "local_files_only": True,
        },
        "rows": len(scored),
        "output_sha256": sha256_bytes(content),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    args.labels = _validate_labels(part.strip() for part in args.labels.split(","))
    return args


def main() -> None:
    print(json.dumps(run(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
