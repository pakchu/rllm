"""Run the frozen EBOC-72 Gemma 4 E2B synthetic LoRA gate exactly once."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training.preregister_sec_edgar_bitcoin_operational_capacity import (
    CLASSES,
    MODEL_ID,
    MODEL_REVISION,
    Config as PreregistrationConfig,
    canonical_hash,
    guarded_output,
    parse_model_output,
    sha256_file,
    validate_local_model,
)
from utils import disable_transformers_allocator_warmup


PROTOCOL_VERSION = "sec_edgar_bitcoin_operational_capacity_synthetic_gate_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/sec_edgar_bitcoin_operational_capacity_"
    "preregistration_2026-07-24.json"
)
PREREGISTRATION_SHA256 = (
    "5ce80db5875e282a1d173489bb910026b53f073f9c783c5140ee99fe3d72605d"
)
PREREGISTRATION_CONTRACT_HASH = (
    "da227af07d68626974dd69c8934fa5258d3ea14f697b5a4e7960a5a460dda391"
)
PREREGISTRATION_MANIFEST_HASH = (
    "76fe7387078dfd0c8783c889cdf81685ac2b3de8aa16c8854fb96b8c7a82901e"
)


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/sec_edgar_bitcoin_operational_capacity_"
        "synthetic_gate_2026-07-24.json"
    )
    checkpoint_root: str = (
        "checkpoints/sec_edgar_bitcoin_operational_capacity_2026-07-24"
    )


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            rows.append(value)
    return rows


def _validate_config(cfg: Config) -> None:
    if cfg != Config():
        raise ValueError("EBOC-72 synthetic runner paths are frozen")


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("EBOC-72 preregistration artifact hash mismatch")
    payload = _read_json(PREREGISTRATION)
    if payload.get("contract_hash") != PREREGISTRATION_CONTRACT_HASH:
        raise ValueError("EBOC-72 preregistration contract hash mismatch")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise ValueError("EBOC-72 preregistration manifest hash mismatch")
    manifest = dict(payload)
    recorded_manifest = manifest.pop("manifest_hash")
    if recorded_manifest != canonical_hash(manifest):
        raise ValueError("EBOC-72 preregistration is not self-consistent")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("EBOC-72 contract is missing")
    if canonical_hash(contract) != PREREGISTRATION_CONTRACT_HASH:
        raise ValueError("EBOC-72 live contract drifted")
    source = payload.get("anchors", {}).get("preregistration_source", {})
    if sha256_file(str(source.get("path"))) != source.get("sha256"):
        raise ValueError("EBOC-72 preregistration source drifted")
    datasets = contract.get("synthetic", {}).get("datasets", {})
    if set(datasets) != {"train", "calibration", "adversarial", "swaps"}:
        raise ValueError("EBOC-72 synthetic dataset manifest is incomplete")
    for name, dataset in datasets.items():
        if sha256_file(str(dataset.get("path"))) != dataset.get("sha256"):
            raise ValueError(f"EBOC-72 {name} dataset hash mismatch")
    decision = payload.get("decision", {})
    if not decision.get("synthetic_training_authorized"):
        raise ValueError("EBOC-72 synthetic training is not authorized")
    if not decision.get("synthetic_final_gate_authorized"):
        raise ValueError("EBOC-72 synthetic final gate is not authorized")
    if decision.get("filing_body_transport_authorized"):
        raise ValueError("EBOC-72 preregistration unexpectedly opens SEC bodies")
    if decision.get("economic_evaluation_authorized"):
        raise ValueError("EBOC-72 preregistration unexpectedly opens outcomes")
    boundary = payload.get("outcome_boundary", {})
    forbidden_nonzero = (
        "filing_bodies_opened",
        "historical_windows_created",
        "historical_semantic_labels_created",
        "historical_semantic_model_calls",
        "btc_market_rows_read",
        "funding_rows_read",
        "future_return_rows_read",
        "return_or_pnl_fields_read",
        "comparator_rows_parsed",
        "comparator_clock_fields_read",
        "2024_or_later_source_rows_read",
    )
    if any(boundary.get(key) != 0 for key in forbidden_nonzero):
        raise ValueError("EBOC-72 preregistration outcome boundary is open")
    return payload


def validate_split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_rows: int,
    expected_per_class: int,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"EBOC-72 {split} row count drifted")
    if Counter(str(row.get("class")) for row in rows) != Counter(
        {label: expected_per_class for label in CLASSES}
    ):
        raise ValueError(f"EBOC-72 {split} class balance drifted")
    row_ids = [str(row.get("row_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"EBOC-72 {split} row IDs are not unique")
    for row in rows:
        output = str(row.get("expected_output"))
        window = str(row.get("window"))
        if parse_model_output(output, window) is None:
            raise ValueError(f"EBOC-72 invalid expected output: {row.get('row_id')}")
        guard = guarded_output(window)
        if bool(guard) != bool(row.get("guarded")):
            raise ValueError(f"EBOC-72 guard drift: {row.get('row_id')}")
        if guard is not None and guard != output:
            raise ValueError(f"EBOC-72 guarded output drift: {row.get('row_id')}")


def schedule_multiplier(
    optimizer_step: int,
    *,
    total_steps: int = 64,
    warmup_steps: int = 4,
) -> float:
    if optimizer_step < 1 or optimizer_step > total_steps:
        raise ValueError("optimizer step is outside the frozen schedule")
    if optimizer_step <= warmup_steps:
        return optimizer_step / warmup_steps
    progress = (
        optimizer_step - warmup_steps - 1
    ) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_rank(summary: Mapping[str, Any], step: int) -> tuple[Any, ...]:
    per_class = summary.get("per_class", {})
    minimum_share = min(
        float(per_class[label]["exact_share"]) for label in CLASSES
    )
    return (
        int(summary["exact_count"]),
        minimum_share,
        -int(summary["malformed_count"]),
        -int(step),
    )


def select_checkpoint(
    summaries: Mapping[int, Mapping[str, Any]]
) -> tuple[int, dict[str, Any]]:
    expected = {16, 32, 48, 64}
    if set(summaries) != expected:
        raise ValueError("EBOC-72 calibration checkpoint set drifted")
    selected = max(
        expected,
        key=lambda step: checkpoint_rank(summaries[step], step),
    )
    return selected, {
        str(step): list(checkpoint_rank(summaries[step], step))
        for step in sorted(expected)
    }


def prediction_summary(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("prediction cardinality mismatch")
    per_class: dict[str, dict[str, Any]] = {}
    exact_count = 0
    malformed_count = 0
    evidence_exists_count = 0
    model_calls = 0
    for label in CLASSES:
        indexes = [
            index for index, row in enumerate(rows) if str(row["class"]) == label
        ]
        exact = sum(bool(predictions[index]["exact"]) for index in indexes)
        malformed = sum(
            predictions[index]["parsed"] is None for index in indexes
        )
        per_class[label] = {
            "rows": len(indexes),
            "exact_count": exact,
            "exact_share": exact / len(indexes),
            "malformed_count": malformed,
        }
        exact_count += exact
        malformed_count += malformed
    for prediction in predictions:
        parsed = prediction.get("parsed")
        if parsed is not None:
            evidence_exists_count += 1
        model_calls += bool(prediction.get("model_called"))
    return {
        "rows": len(rows),
        "model_calls": model_calls,
        "exact_count": exact_count,
        "exact_share": exact_count / len(rows),
        "malformed_count": malformed_count,
        "strict_parse_share": (len(rows) - malformed_count) / len(rows),
        "evidence_existence_share": evidence_exists_count / len(rows),
        "per_class": per_class,
    }


def tagged_unsupported_share(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    tag: str,
) -> tuple[int, float]:
    indexes = [
        index for index, row in enumerate(rows) if tag in row.get("tags", [])
    ]
    if not indexes:
        raise ValueError(f"missing frozen tagged rows: {tag}")
    correct = sum(
        prediction_class(predictions[index]) == "UNSUPPORTED" for index in indexes
    )
    return len(indexes), correct / len(indexes)


def prediction_class(prediction: Mapping[str, Any]) -> str | None:
    parsed = prediction.get("parsed")
    return str(parsed.get("class")) if isinstance(parsed, dict) else None


def swap_invariance(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("swap prediction cardinality mismatch")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["pair_id"])].append(index)
    if len(groups) != 64 or any(len(indexes) != 2 for indexes in groups.values()):
        raise ValueError("frozen swap-pair cardinality drifted")
    invariant = 0
    exact = 0
    for indexes in groups.values():
        left, right = indexes
        left_parsed = predictions[left].get("parsed")
        right_parsed = predictions[right].get("parsed")
        if left_parsed is not None and left_parsed == right_parsed:
            invariant += 1
        if predictions[left].get("exact") and predictions[right].get("exact"):
            exact += 1
    return {
        "pairs": len(groups),
        "invariant_pairs": invariant,
        "invariance_share": invariant / len(groups),
        "both_exact_pairs": exact,
        "both_exact_share": exact / len(groups),
    }


def final_gate(
    *,
    adversarial_rows: Sequence[Mapping[str, Any]],
    adversarial_predictions: Sequence[Mapping[str, Any]],
    swap_rows: Sequence[Mapping[str, Any]],
    swap_predictions: Sequence[Mapping[str, Any]],
    training_memory: Mapping[str, int],
    inference_memory: Mapping[str, int],
    cfg: PreregistrationConfig = PreregistrationConfig(),
) -> dict[str, Any]:
    adversarial = prediction_summary(adversarial_rows, adversarial_predictions)
    swaps = prediction_summary(swap_rows, swap_predictions)
    combined_rows = len(adversarial_rows) + len(swap_rows)
    combined_malformed = (
        int(adversarial["malformed_count"]) + int(swaps["malformed_count"])
    )
    guarded_indexes = [
        index for index, row in enumerate(adversarial_rows) if row.get("guarded")
    ]
    guarded_exact = sum(
        bool(adversarial_predictions[index]["exact"]) for index in guarded_indexes
    )
    guarded_calls = sum(
        bool(adversarial_predictions[index]["model_called"])
        for index in guarded_indexes
    )
    ebct_rows, ebct_share = tagged_unsupported_share(
        adversarial_rows, adversarial_predictions, "ebct_negative"
    )
    bpax_rows, bpax_share = tagged_unsupported_share(
        adversarial_rows, adversarial_predictions, "bpax_negative"
    )
    swaps_gate = swap_invariance(swap_rows, swap_predictions)
    checks = {
        "combined_strict_parse": combined_malformed == 0,
        "combined_evidence_existence": combined_malformed == 0,
        "adversarial_overall_exact": adversarial["exact_share"] >= 0.95,
        "adversarial_online_exact": (
            adversarial["per_class"]["CAPACITY_ONLINE"]["exact_share"] >= 0.95
        ),
        "adversarial_offline_exact": (
            adversarial["per_class"]["CAPACITY_OFFLINE"]["exact_share"] >= 0.95
        ),
        "adversarial_unsupported_exact": (
            adversarial["per_class"]["UNSUPPORTED"]["exact_share"] >= 0.97
        ),
        "adversarial_mixed_exact": (
            adversarial["per_class"]["MIXED"]["exact_share"] == 1.0
        ),
        "guarded_exact": guarded_exact == len(guarded_indexes) == 8,
        "guarded_zero_model_calls": guarded_calls == 0,
        "ebct_negative_unsupported": ebct_rows == 12 and ebct_share == 1.0,
        "bpax_negative_unsupported": bpax_rows == 12 and bpax_share == 1.0,
        "swap_invariance": swaps_gate["invariance_share"] == 1.0,
        "swap_exact": swaps_gate["both_exact_share"] == 1.0,
        "training_peak_allocated": (
            int(training_memory["peak_allocated_bytes"])
            <= cfg.maximum_training_peak_bytes
        ),
        "training_peak_reserved": (
            int(training_memory["peak_reserved_bytes"])
            <= cfg.maximum_training_peak_bytes
        ),
        "inference_peak_allocated": (
            int(inference_memory["peak_allocated_bytes"])
            <= cfg.maximum_inference_peak_allocated_bytes
        ),
        "inference_peak_reserved": (
            int(inference_memory["peak_reserved_bytes"])
            <= cfg.maximum_inference_peak_reserved_bytes
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "combined": {
            "rows": combined_rows,
            "malformed_count": combined_malformed,
            "strict_parse_share": (
                (combined_rows - combined_malformed) / combined_rows
            ),
        },
        "adversarial": adversarial,
        "swaps": swaps,
        "swap_gate": swaps_gate,
        "guarded": {
            "rows": len(guarded_indexes),
            "exact_count": guarded_exact,
            "model_calls": guarded_calls,
        },
        "ebct_negative": {"rows": ebct_rows, "unsupported_share": ebct_share},
        "bpax_negative": {"rows": bpax_rows, "unsupported_share": bpax_share},
        "training_memory": dict(training_memory),
        "inference_memory": dict(inference_memory),
    }


def _directory_manifest(path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            files.append(
                {
                    "path": str(candidate.relative_to(path)),
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "files": files,
        "bytes": sum(int(row["bytes"]) for row in files),
        "manifest_hash": canonical_hash(files),
    }


class Gemma4E2BLoRARunner:
    def __init__(
        self,
        preregistration: Mapping[str, Any],
        cfg: Config,
    ) -> None:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        self.torch = torch
        self.cfg = cfg
        self.frozen = PreregistrationConfig()
        self.contract = preregistration["contract"]
        if not torch.cuda.is_available():
            raise RuntimeError("EBOC-72 frozen run requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "EBOC-72 frozen run requires exactly one visible CUDA device"
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("EBOC-72 frozen run requires CUDA BF16 support")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        snapshot = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--google--gemma-4-E2B-it"
            / "snapshots"
            / MODEL_REVISION
        )
        if "HF_HUB_CACHE" in os.environ:
            snapshot = (
                Path(os.environ["HF_HUB_CACHE"])
                / "models--google--gemma-4-E2B-it"
                / "snapshots"
                / MODEL_REVISION
            )
        elif "HF_HOME" in os.environ:
            snapshot = (
                Path(os.environ["HF_HOME"])
                / "hub"
                / "models--google--gemma-4-E2B-it"
                / "snapshots"
                / MODEL_REVISION
            )
        self.snapshot = snapshot
        self.processor = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_started = time.perf_counter()
        with disable_transformers_allocator_warmup():
            base = AutoModelForMultimodalLM.from_pretrained(
                snapshot,
                local_files_only=True,
                quantization_config=quantization,
                device_map={"": 0},
                dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=False,
            )
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=False,
        )
        lora = LoraConfig(
            r=self.frozen.lora_rank,
            lora_alpha=self.frozen.lora_alpha,
            lora_dropout=self.frozen.lora_dropout,
            bias="none",
            target_modules=self.frozen.lora_target_regex,
        )
        self.model = get_peft_model(base, lora)
        self.model.config.use_cache = False
        self.load_seconds = time.perf_counter() - load_started
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if trainable != self.frozen.trainable_parameters:
            raise RuntimeError(
                f"EBOC-72 trainable parameter drift: {trainable:,}"
            )
        matched = sorted(
            name
            for name, module in self.model.named_modules()
            if "lora_A" in name and hasattr(module, "weight")
        )
        if not matched or any("language_model" not in name for name in matched):
            raise RuntimeError("EBOC-72 LoRA escaped the text language model")
        self.trainable_parameters = trainable
        self.lora_a_modules = matched

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def _chat_inputs(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> Mapping[str, Any]:
        return self.processor.apply_chat_template(
            list(messages),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

    def encode_training_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        user = {"role": "user", "content": str(row["prompt"])}
        assistant = {"role": "assistant", "content": str(row["expected_output"])}
        prompt_inputs = self._chat_inputs([user], add_generation_prompt=True)
        full_inputs = self._chat_inputs(
            [user, assistant],
            add_generation_prompt=False,
        )
        prompt_length = int(prompt_inputs["input_ids"].shape[-1])
        full_length = int(full_inputs["input_ids"].shape[-1])
        if full_length > self.frozen.maximum_input_tokens:
            raise RuntimeError(
                f"EBOC-72 training row exceeds token cap: {row['row_id']}"
            )
        if not self.torch.equal(
            full_inputs["input_ids"][0, :prompt_length],
            prompt_inputs["input_ids"][0],
        ):
            raise RuntimeError("EBOC-72 completion boundary is not prefix exact")
        encoded = {
            key: value.squeeze(0).cpu()
            for key, value in full_inputs.items()
            if self.torch.is_tensor(value)
        }
        labels = encoded["input_ids"].clone()
        labels[:prompt_length] = -100
        if bool((labels[prompt_length:] == -100).all()):
            raise RuntimeError("EBOC-72 completion labels are empty")
        encoded["labels"] = labels
        encoded["prompt_tokens"] = prompt_length
        encoded["total_tokens"] = full_length
        return encoded

    def _to_device(self, encoded: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.unsqueeze(0).to(self.device)
            for key, value in encoded.items()
            if self.torch.is_tensor(value)
        }

    def train(
        self,
        ordered_rows: Sequence[Mapping[str, Any]],
        checkpoint_root: Path,
    ) -> dict[str, Any]:
        frozen = self.frozen
        if len(ordered_rows) != (
            frozen.optimizer_steps * frozen.gradient_accumulation_steps
        ):
            raise RuntimeError("EBOC-72 train rows do not fill exactly 64 steps")
        encoded_rows = [self.encode_training_row(row) for row in ordered_rows]
        token_counts = [int(row["total_tokens"]) for row in encoded_rows]
        if max(token_counts) > frozen.maximum_input_tokens:
            raise RuntimeError("EBOC-72 training token cap drifted")
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        optimizer = self.torch.optim.AdamW(
            parameters,
            lr=frozen.learning_rate,
            weight_decay=frozen.weight_decay,
        )
        checkpoint_root.mkdir(parents=True, exist_ok=False)
        (checkpoint_root / "RUN_STARTED.json").write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "preregistration_sha256": PREREGISTRATION_SHA256,
                    "runner_sha256": sha256_file(Path(__file__)),
                    "seed": frozen.seed,
                    "started_at": _utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.model.train()
        self.model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        self.torch.cuda.empty_cache()
        self.torch.cuda.reset_peak_memory_stats()
        losses: list[float] = []
        step_metrics: list[dict[str, Any]] = []
        started = time.perf_counter()
        for micro_index, encoded in enumerate(encoded_rows, start=1):
            inputs = self._to_device(encoded)
            with self.torch.autocast(
                device_type="cuda",
                dtype=self.torch.bfloat16,
            ):
                output = self.model(**inputs)
                raw_loss = output.loss
                loss = raw_loss / frozen.gradient_accumulation_steps
            loss.backward()
            losses.append(float(raw_loss.detach().cpu()))
            del output, raw_loss, loss, inputs
            if micro_index % frozen.gradient_accumulation_steps:
                continue
            optimizer_step = micro_index // frozen.gradient_accumulation_steps
            gradient_norm = self.torch.nn.utils.clip_grad_norm_(
                parameters,
                frozen.maximum_gradient_norm,
            )
            multiplier = schedule_multiplier(
                optimizer_step,
                total_steps=frozen.optimizer_steps,
                warmup_steps=frozen.warmup_steps,
            )
            learning_rate = frozen.learning_rate * multiplier
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_metric = {
                "optimizer_step": optimizer_step,
                "micro_examples_seen": micro_index,
                "mean_raw_loss": sum(losses[-8:]) / 8,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - started,
            }
            step_metrics.append(step_metric)
            if optimizer_step in frozen.checkpoint_steps:
                checkpoint = checkpoint_root / f"step-{optimizer_step:03d}"
                self.model.save_pretrained(
                    checkpoint,
                    safe_serialization=True,
                )
        self.torch.cuda.synchronize()
        if len(step_metrics) != frozen.optimizer_steps:
            raise RuntimeError("EBOC-72 optimizer-step count drifted")
        training_memory = {
            "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
        }
        return {
            "rows": len(ordered_rows),
            "optimizer_steps": len(step_metrics),
            "mean_raw_loss": sum(losses) / len(losses),
            "first_step_mean_loss": step_metrics[0]["mean_raw_loss"],
            "last_step_mean_loss": step_metrics[-1]["mean_raw_loss"],
            "maximum_total_tokens": max(token_counts),
            "minimum_total_tokens": min(token_counts),
            "elapsed_seconds": time.perf_counter() - started,
            "memory": training_memory,
            "steps": step_metrics,
        }

    def load_calibration_adapters(
        self, checkpoint_root: Path
    ) -> dict[int, str]:
        adapters = {64: "default"}
        for step in (16, 32, 48):
            name = f"step_{step:03d}"
            self.model.load_adapter(
                checkpoint_root / f"step-{step:03d}",
                adapter_name=name,
                is_trainable=False,
            )
            adapters[step] = name
        return adapters

    def classify(self, row: Mapping[str, Any]) -> dict[str, Any]:
        guard = guarded_output(str(row["window"]))
        if guard is not None:
            parsed = parse_model_output(guard, str(row["window"]))
            return {
                "row_id": row["row_id"],
                "expected_output": row["expected_output"],
                "final_content": guard,
                "decoded_suffix": None,
                "parsed": parsed,
                "exact": guard == row["expected_output"],
                "model_called": False,
                "input_tokens": 0,
                "generated_tokens": 0,
                "inference_seconds": 0.0,
            }
        user = {"role": "user", "content": str(row["prompt"])}
        inputs = self._chat_inputs([user], add_generation_prompt=True)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > self.frozen.maximum_input_tokens:
            raise RuntimeError(f"EBOC-72 inference row exceeds cap: {row['row_id']}")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if self.torch.is_tensor(value)
        }
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.frozen.maximum_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.model.generation_config.eos_token_id,
            )
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        prefix = int(inputs["input_ids"].shape[-1])
        suffix = generated[0, prefix:]
        decoded = self.processor.decode(
            suffix,
            skip_special_tokens=False,
        ).strip()
        response = self.processor.parse_response(decoded)
        content = response.get("content") if isinstance(response, dict) else None
        final_content = content if isinstance(content, str) else ""
        parsed = parse_model_output(final_content, str(row["window"]))
        return {
            "row_id": row["row_id"],
            "expected_output": row["expected_output"],
            "final_content": final_content,
            "decoded_suffix": decoded,
            "parsed": parsed,
            "exact": final_content == row["expected_output"] and parsed is not None,
            "model_called": True,
            "input_tokens": input_tokens,
            "generated_tokens": int(suffix.shape[-1]),
            "inference_seconds": elapsed,
        }

    def evaluate(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        adapter_name: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.model.set_adapter(adapter_name)
        self.model.eval()
        self.model.config.use_cache = True
        predictions = [self.classify(row) for row in rows]
        return prediction_summary(rows, predictions), predictions

    def reset_inference_memory(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.reset_peak_memory_stats()

    def inference_memory(self) -> dict[str, int]:
        return {
            "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
        }


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _ordered_train_rows(
    rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    order = preregistration["contract"]["synthetic"]["train_permutation_row_ids"]
    by_id = {str(row["row_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(order) != set(by_id):
        raise ValueError("EBOC-72 frozen train permutation drifted")
    return [by_id[row_id] for row_id in order]


def run(cfg: Config = Config()) -> dict[str, Any]:
    _validate_config(cfg)
    output = _path(cfg.output)
    checkpoint_root = _path(cfg.checkpoint_root)
    if output.exists():
        raise FileExistsError("EBOC-72 synthetic result is write-once")
    if checkpoint_root.exists():
        raise FileExistsError("EBOC-72 checkpoint root is write-once")
    preregistration = validate_preregistration()
    local_model = validate_local_model()
    datasets = preregistration["contract"]["synthetic"]["datasets"]

    # Only train and calibration are parsed before checkpoint selection.
    train_rows = _read_jsonl(datasets["train"]["path"])
    calibration_rows = _read_jsonl(datasets["calibration"]["path"])
    validate_split_rows(
        train_rows,
        split="train",
        expected_rows=512,
        expected_per_class=128,
    )
    validate_split_rows(
        calibration_rows,
        split="calibration",
        expected_rows=128,
        expected_per_class=32,
    )
    ordered_train = _ordered_train_rows(train_rows, preregistration)
    frozen = PreregistrationConfig()
    _seed_everything(frozen.seed)
    started_at = _utc_now()
    runner = Gemma4E2BLoRARunner(preregistration, cfg)
    training = runner.train(ordered_train, checkpoint_root)
    checkpoint_manifests = {
        str(step): _directory_manifest(checkpoint_root / f"step-{step:03d}")
        for step in frozen.checkpoint_steps
    }
    retained_bytes = sum(
        int(manifest["bytes"]) for manifest in checkpoint_manifests.values()
    )
    if retained_bytes >= 1024**3:
        raise RuntimeError("EBOC-72 retained checkpoints exceed 1 GiB")

    adapter_names = runner.load_calibration_adapters(checkpoint_root)
    calibration_summaries: dict[int, dict[str, Any]] = {}
    calibration_predictions: dict[str, list[dict[str, Any]]] = {}
    for step in frozen.checkpoint_steps:
        summary, predictions = runner.evaluate(
            calibration_rows,
            adapter_name=adapter_names[step],
        )
        calibration_summaries[step] = summary
        calibration_predictions[str(step)] = predictions
    selected_step, ranking = select_checkpoint(calibration_summaries)
    selected_adapter_name = adapter_names[selected_step]

    # The untouched final splits become parseable only after selection.
    adversarial_rows = _read_jsonl(datasets["adversarial"]["path"])
    swap_rows = _read_jsonl(datasets["swaps"]["path"])
    validate_split_rows(
        adversarial_rows,
        split="adversarial",
        expected_rows=192,
        expected_per_class=48,
    )
    validate_split_rows(
        swap_rows,
        split="swaps",
        expected_rows=128,
        expected_per_class=32,
    )
    runner.reset_inference_memory()
    _, adversarial_predictions = runner.evaluate(
        adversarial_rows,
        adapter_name=selected_adapter_name,
    )
    _, swap_predictions = runner.evaluate(
        swap_rows,
        adapter_name=selected_adapter_name,
    )
    inference_memory = runner.inference_memory()
    gate = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=adversarial_predictions,
        swap_rows=swap_rows,
        swap_predictions=swap_predictions,
        training_memory=training["memory"],
        inference_memory=inference_memory,
        cfg=frozen,
    )

    selected_path: Path | None = None
    selected_manifest: dict[str, Any] | None = None
    if gate["passed"]:
        selected_path = checkpoint_root / "selected"
        shutil.copytree(
            checkpoint_root / f"step-{selected_step:03d}",
            selected_path,
        )
        selected_manifest = _directory_manifest(selected_path)
        if retained_bytes + int(selected_manifest["bytes"]) >= 1024**3:
            raise RuntimeError("EBOC-72 selected retention exceeds 1 GiB")

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "contract_hash": PREREGISTRATION_CONTRACT_HASH,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(Path(__file__)),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_validation": local_model,
            "load_seconds": runner.load_seconds,
            "trainable_parameters": runner.trainable_parameters,
            "lora_a_modules": runner.lora_a_modules,
        },
        "outcome_boundary": {
            "filing_bodies_opened": 0,
            "historical_windows_created": 0,
            "historical_semantic_model_calls": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "comparator_rows_parsed": 0,
            "comparator_clock_fields_read": 0,
            "2024_or_later_source_rows_read": 0,
        },
        "training": training,
        "checkpoints": {
            "manifests": checkpoint_manifests,
            "retained_bytes_before_result_commit": retained_bytes,
        },
        "calibration": {
            "summaries": {
                str(step): calibration_summaries[step]
                for step in sorted(calibration_summaries)
            },
            "predictions": calibration_predictions,
            "ranking": ranking,
            "selected_step": selected_step,
        },
        "final": {
            "gate": gate,
            "adversarial_predictions": adversarial_predictions,
            "swap_predictions": swap_predictions,
            "selected_adapter": selected_manifest,
        },
        "decision": {
            "status": "passed" if gate["passed"] else "retired_synthetic_failure",
            "synthetic_gate_passed": bool(gate["passed"]),
            "historical_body_transport_authorized": bool(gate["passed"]),
            "historical_semantic_execution_authorized": bool(gate["passed"]),
            "novelty_evaluation_authorized": False,
            "economic_evaluation_authorized": False,
            "2024_or_later_authorized": False,
            "repair_authorized": False,
            "next_step": (
                "commit result and selected adapter bindings, then freeze the "
                "historical SEC body/support builder"
                if gate["passed"]
                else "retire EBOC-72 unchanged before all historical bodies and outcomes"
            ),
        },
    }
    result["manifest_hash"] = canonical_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate frozen hashes without parsing any synthetic split",
    )
    args = parser.parse_args()
    if args.validate_only:
        payload = validate_preregistration()
        print(
            json.dumps(
                {
                    "contract_hash": payload["contract_hash"],
                    "decision": payload["decision"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run()
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
