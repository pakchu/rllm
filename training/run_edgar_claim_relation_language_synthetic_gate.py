"""Run the frozen ECRL-1 Gemma 4 E2B synthetic QLoRA gate once."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.preregister_edgar_claim_relation_language import (
    AS_OF_DATE,
    MECHANISM_DOCUMENT,
    MECHANISM_DOCUMENT_SHA256,
    MODEL_FILES,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_TEMPLATE,
    REPOSITORY_ROOT,
    ROW_KEYS,
    SCENARIO_TARGETS,
    SEED,
    SPLIT_ROWS_PER_SCENARIO,
    _training_order,
    canonical_hash,
    parse_model_output,
    prefilter_reason,
    relation_contrast_groups,
    render_prompt,
    required_count,
    sha256_file,
)
from utils import disable_transformers_allocator_warmup


PROTOCOL_VERSION = "edgar_claim_relation_language_synthetic_gate_v1"
M0_COMMIT = "db2671b4945a629d0bd8440fee47b298c544ccf5"
PREREGISTRATION = Path(
    "results/edgar_claim_relation_language_m0_preregistration_2026-07-25.json"
)
PREREGISTRATION_SHA256 = (
    "c34b805bb7db79a7011eabf25be071d88410974024fbaf9ab731105233830136"
)
PREREGISTRATION_SELF_HASH = (
    "dfce3b18f68533b412fba5c9924cd320c4c7266730c445a1f666da0d9b227823"
)
PREREGISTRATION_CONTRACT_HASH = (
    "d2ed4f60415f297a0719b2111f2a00bfb1450ebdaad0261d39c550798620cc88"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "a7cac72c728aa646fe26622f90df63da881848feb4073f0e21b88a68da48c50b"
)
GENERATION_CONFIG_SHA256 = (
    "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de"
)

RUNTIME_VERSIONS: Mapping[str, str] = {
    "torch": "2.9.0",
    "transformers": "5.7.0.dev0",
    "peft": "0.18.1",
    "bitsandbytes": "0.49.2",
    "accelerate": "1.12.0",
}
TORCH_BUILD_VERSION = "2.9.0+cu128"
TORCH_CUDA_VERSION = "12.8"

CHECKPOINT_STEPS = (64, 128, 192, 256)
OPTIMIZER_STEPS = 256
WARMUP_STEPS = 16
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 0.000075
WEIGHT_DECAY = 0.01
MAXIMUM_GRADIENT_NORM = 1.0
MAXIMUM_INPUT_TOKENS = 2048
MAXIMUM_NEW_TOKENS = 24
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
TRAINABLE_PARAMETERS = 5_357_568
MAXIMUM_TRAINING_PEAK_BYTES = 24 * 1024**3
MAXIMUM_SELECTED_ADAPTER_BYTES = 512 * 1024**2
MAXIMUM_FILESYSTEM_USED_BYTES = 300 * 1024**3


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/edgar_claim_relation_language_synthetic_gate_2026-07-25.json"
    )
    checkpoint_root: str = (
        "checkpoints/edgar_claim_relation_language_2026-07-25"
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


def _git_blob(commit: str, path: str | Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _filesystem_used_bytes() -> int:
    return int(shutil.disk_usage(REPOSITORY_ROOT).used)


def _validate_disk_cap(stage: str) -> int:
    used = _filesystem_used_bytes()
    if used >= MAXIMUM_FILESYSTEM_USED_BYTES:
        raise RuntimeError(
            f"ECRL-1 filesystem cap exceeded at {stage}: {used} bytes"
        )
    return used


def _validate_runtime_versions() -> dict[str, str]:
    observed: dict[str, str] = {}
    for package, expected in RUNTIME_VERSIONS.items():
        value = importlib.metadata.version(package)
        observed[package] = value
        normalized = value.split("+", 1)[0]
        if normalized != expected:
            raise RuntimeError(
                f"ECRL-1 runtime drift: {package}={value}, expected={expected}"
            )
    import torch

    observed["torch_build"] = str(torch.__version__)
    observed["torch_cuda"] = str(torch.version.cuda)
    if observed["torch_build"] != TORCH_BUILD_VERSION:
        raise RuntimeError(
            "ECRL-1 torch build drift: "
            f"{observed['torch_build']}, expected={TORCH_BUILD_VERSION}"
        )
    if observed["torch_cuda"] != TORCH_CUDA_VERSION:
        raise RuntimeError(
            "ECRL-1 torch CUDA drift: "
            f"{observed['torch_cuda']}, expected={TORCH_CUDA_VERSION}"
        )
    return observed


def _local_snapshot() -> Path:
    if "HF_HUB_CACHE" in os.environ:
        root = Path(os.environ["HF_HUB_CACHE"])
    elif "HF_HOME" in os.environ:
        root = Path(os.environ["HF_HOME"]) / "hub"
    else:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    return (
        root
        / "models--google--gemma-4-E2B-it"
        / "snapshots"
        / MODEL_REVISION
    )


def validate_local_model() -> dict[str, Any]:
    snapshot = _local_snapshot()
    expected = dict(MODEL_FILES)
    expected["generation_config.json"] = GENERATION_CONFIG_SHA256
    files: dict[str, Any] = {}
    for name, digest in sorted(expected.items()):
        candidate = snapshot / name
        if not candidate.is_file():
            raise FileNotFoundError(f"ECRL-1 local model file missing: {candidate}")
        observed = sha256_file(candidate)
        if observed != digest:
            raise ValueError(f"ECRL-1 local model hash drifted: {name}")
        files[name] = {
            "path": str(candidate),
            "bytes": candidate.stat().st_size,
            "sha256": observed,
        }
    return {
        "snapshot": str(snapshot),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": files,
    }


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("ECRL-1 preregistration file hash drifted")
    if hashlib.sha256(_git_blob(M0_COMMIT, PREREGISTRATION)).hexdigest() != (
        PREREGISTRATION_SHA256
    ):
        raise ValueError("ECRL-1 M0 commit does not bind the preregistration")
    payload = _read_json(PREREGISTRATION)
    recorded_self_hash = payload.get("self_hash")
    unhashed = dict(payload)
    unhashed.pop("self_hash", None)
    if recorded_self_hash != PREREGISTRATION_SELF_HASH:
        raise ValueError("ECRL-1 preregistration self-hash drifted")
    if canonical_hash(unhashed) != recorded_self_hash:
        raise ValueError("ECRL-1 preregistration self-hash is inconsistent")
    if payload.get("contract_hash") != PREREGISTRATION_CONTRACT_HASH:
        raise ValueError("ECRL-1 preregistration contract hash drifted")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("ECRL-1 preregistration contract is missing")
    if canonical_hash(contract) != PREREGISTRATION_CONTRACT_HASH:
        raise ValueError("ECRL-1 preregistration contract is inconsistent")
    source = payload.get("anchors", {}).get("preregistration_source", {})
    if source.get("sha256") != PREREGISTRATION_SOURCE_SHA256:
        raise ValueError("ECRL-1 preregistration source binding drifted")
    if sha256_file(str(source.get("path"))) != PREREGISTRATION_SOURCE_SHA256:
        raise ValueError("ECRL-1 preregistration source file drifted")
    if sha256_file(MECHANISM_DOCUMENT) != MECHANISM_DOCUMENT_SHA256:
        raise ValueError("ECRL-1 mechanism document drifted")
    if payload.get("m0_counters") != {
        "2024_or_later_rows_read": 0,
        "SEC_body_requests": 0,
        "SEC_header_requests": 0,
        "funding_rows_read": 0,
        "historical_pairs_created": 0,
        "market_rows_read": 0,
        "model_calls": 0,
        "model_loads": 0,
        "premium_rows_read": 0,
        "reward_rows_read": 0,
        "tokenizer_loads": 0,
    }:
        raise ValueError("ECRL-1 M0 boundary is not closed")
    decision = payload.get("decision", {})
    if decision.get("status") != "PASS":
        raise ValueError("ECRL-1 M0 did not pass")
    if not decision.get("synthetic_training_authorized"):
        raise ValueError("ECRL-1 synthetic training is not authorized")
    if decision.get("historical_SEC_transport_authorized"):
        raise ValueError("ECRL-1 M0 unexpectedly authorizes historical SEC")
    if decision.get("market_or_reward_access_authorized"):
        raise ValueError("ECRL-1 M0 unexpectedly authorizes economics")

    prompt = contract.get("prompt", {})
    if sha256_file(str(prompt.get("path"))) != prompt.get("sha256"):
        raise ValueError("ECRL-1 prompt artifact drifted")
    if _path(str(prompt.get("path"))).read_bytes() != PROMPT_TEMPLATE.encode(
        "utf-8"
    ):
        raise ValueError("ECRL-1 prompt bytes drifted")
    inventory = contract.get("template_inventory", {})
    if sha256_file(str(inventory.get("path"))) != inventory.get("sha256"):
        raise ValueError("ECRL-1 template inventory drifted")
    datasets = contract.get("datasets", {})
    if set(datasets) != {"train", "calibration", "adversarial", "swap"}:
        raise ValueError("ECRL-1 dataset manifest is incomplete")
    for split, metadata in datasets.items():
        if sha256_file(str(metadata.get("path"))) != metadata.get("sha256"):
            raise ValueError(f"ECRL-1 {split} dataset hash drifted")
    if contract.get("guard_gate") != {
        "rows": 32,
        "rejections_required": 32,
        "model_calls_required": 0,
        "adversarial_model_denominator": 736,
        "per_scenario_model_rows": 46,
    }:
        raise ValueError("ECRL-1 guard contract drifted")
    relation = contract.get("relation_contrast", {})
    if relation.get("group_count") != 16 or relation.get("row_count") != 64:
        raise ValueError("ECRL-1 relation-contrast contract drifted")
    return payload


def validate_split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> None:
    expected_rows = SPLIT_ROWS_PER_SCENARIO[split] * len(SCENARIO_TARGETS)
    if len(rows) != expected_rows:
        raise ValueError(f"ECRL-1 {split} row count drifted")
    if Counter(str(row.get("scenario_id")) for row in rows) != Counter(
        {
            scenario: SPLIT_ROWS_PER_SCENARIO[split]
            for scenario in SCENARIO_TARGETS
        }
    ):
        raise ValueError(f"ECRL-1 {split} scenario balance drifted")
    if any(tuple(row) != ROW_KEYS for row in rows):
        raise ValueError(f"ECRL-1 {split} row key order drifted")
    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"ECRL-1 {split} row IDs are not unique")
    if row_ids != sorted(row_ids, key=lambda value: tuple(value.split(":"))):
        raise ValueError(f"ECRL-1 {split} row order drifted")
    guarded = 0
    for row in rows:
        parsed = parse_model_output(
            str(row["target"]),
            prior=str(row["prior"]),
            current=str(row["current"]),
        )
        if not parsed["valid"]:
            raise ValueError(
                f"ECRL-1 invalid target {row['row_id']}: {parsed['error']}"
            )
        reason = prefilter_reason(str(row["prior"]), str(row["current"]))
        if reason is not None:
            guarded += 1
            if split != "adversarial":
                raise ValueError(f"ECRL-1 guard escaped into {split}")
        elif render_prompt(row) is None:
            raise ValueError(f"ECRL-1 prompt unexpectedly missing: {row['row_id']}")
    expected_guarded = 32 if split == "adversarial" else 0
    if guarded != expected_guarded:
        raise ValueError(f"ECRL-1 {split} guard count drifted")
    if split == "swap":
        groups: dict[str, int] = Counter(str(row["pair_id"]) for row in rows)
        if len(groups) != 256 or set(groups.values()) != {2}:
            raise ValueError("ECRL-1 swap pair cardinality drifted")
    if split == "adversarial":
        relation_contrast_groups(rows)


def schedule_multiplier(
    optimizer_step: int,
    *,
    total_steps: int = OPTIMIZER_STEPS,
    warmup_steps: int = WARMUP_STEPS,
) -> float:
    if optimizer_step < 1 or optimizer_step > total_steps:
        raise ValueError("optimizer step is outside the frozen schedule")
    if optimizer_step <= warmup_steps:
        return optimizer_step / warmup_steps
    progress = (
        optimizer_step - warmup_steps - 1
    ) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_rank(
    summary: Mapping[str, Any],
    step: int,
) -> tuple[Any, ...]:
    status = summary["fields"]["status"]
    relation = summary["fields"]["relation"]
    return (
        int(summary["exact_count"]),
        min(float(value["accuracy"]) for value in status.values()),
        min(float(value["accuracy"]) for value in relation.values()),
        -int(summary["malformed_count"]),
        -int(step),
    )


def select_checkpoint(
    summaries: Mapping[int, Mapping[str, Any]],
) -> tuple[int, dict[str, list[Any]]]:
    if set(summaries) != set(CHECKPOINT_STEPS):
        raise ValueError("ECRL-1 calibration checkpoint set drifted")
    selected = max(
        CHECKPOINT_STEPS,
        key=lambda step: checkpoint_rank(summaries[step], step),
    )
    return selected, {
        str(step): list(checkpoint_rank(summaries[step], step))
        for step in CHECKPOINT_STEPS
    }


def _prediction_fields(prediction: Mapping[str, Any]) -> dict[str, str] | None:
    parsed = prediction.get("parsed")
    if not isinstance(parsed, dict) or not parsed.get("valid"):
        return None
    return {
        "status": str(parsed["status"]),
        "delta": str(parsed["delta"]),
        "relation": str(parsed["relation"]),
        "current_evidence": str(parsed["current_evidence"]),
        "prior_evidence": str(parsed["prior_evidence"]),
    }


def prediction_summary(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("ECRL-1 prediction cardinality mismatch")
    model_indexes = [
        index
        for index, prediction in enumerate(predictions)
        if bool(prediction.get("model_called"))
    ]
    guard_indexes = [
        index
        for index, prediction in enumerate(predictions)
        if not bool(prediction.get("model_called"))
    ]
    exact_count = sum(
        bool(predictions[index].get("exact")) for index in model_indexes
    )
    parsed_count = sum(
        _prediction_fields(predictions[index]) is not None
        for index in model_indexes
    )
    per_scenario: dict[str, Any] = {}
    for scenario in SCENARIO_TARGETS:
        indexes = [
            index
            for index in model_indexes
            if str(rows[index]["scenario_id"]) == scenario
        ]
        exact = sum(bool(predictions[index].get("exact")) for index in indexes)
        per_scenario[scenario] = {
            "rows": len(indexes),
            "exact_count": exact,
            "exact_share": exact / len(indexes) if indexes else None,
        }
    fields: dict[str, dict[str, Any]] = {
        "status": {},
        "delta": {},
        "relation": {},
    }
    field_positions = {"status": 0, "delta": 1, "relation": 2}
    for field, position in field_positions.items():
        values = sorted(
            {str(row["target"]).split("|")[position] for row in rows}
        )
        for value in values:
            indexes = [
                index
                for index in model_indexes
                if str(rows[index]["target"]).split("|")[position] == value
            ]
            correct = 0
            for index in indexes:
                parsed = _prediction_fields(predictions[index])
                if parsed is not None and parsed[field] == value:
                    correct += 1
            fields[field][value] = {
                "rows": len(indexes),
                "correct_count": correct,
                "accuracy": correct / len(indexes) if indexes else None,
            }
    return {
        "rows": len(rows),
        "model_rows": len(model_indexes),
        "guard_rows": len(guard_indexes),
        "model_calls": sum(
            bool(prediction.get("model_called")) for prediction in predictions
        ),
        "exact_count": exact_count,
        "exact_share": exact_count / len(model_indexes) if model_indexes else None,
        "parsed_count": parsed_count,
        "parse_share": parsed_count / len(model_indexes) if model_indexes else None,
        "malformed_count": len(model_indexes) - parsed_count,
        "per_scenario": per_scenario,
        "fields": fields,
    }


def swap_summary(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("ECRL-1 swap prediction cardinality mismatch")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["pair_id"])].append(index)
    if len(groups) != 256 or any(len(indexes) != 2 for indexes in groups.values()):
        raise ValueError("ECRL-1 swap groups drifted")
    invariant = 0
    both_exact = 0
    for indexes in groups.values():
        left, right = indexes
        left_fields = _prediction_fields(predictions[left])
        right_fields = _prediction_fields(predictions[right])
        if (
            left_fields is not None
            and right_fields is not None
            and predictions[left].get("final_content")
            == predictions[right].get("final_content")
        ):
            invariant += 1
        if predictions[left].get("exact") and predictions[right].get("exact"):
            both_exact += 1
    return {
        "pairs": len(groups),
        "invariant_pairs": invariant,
        "invariance_share": invariant / len(groups),
        "both_exact_pairs": both_exact,
        "both_exact_share": both_exact / len(groups),
    }


def relation_contrast_summary(
    preregistration: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = preregistration["contract"]["relation_contrast"]["groups"]
    by_id = {
        str(prediction["row_id"]): prediction for prediction in predictions
    }
    exact_groups = 0
    rows = 0
    exact_rows = 0
    for row_ids in groups.values():
        group_predictions = [by_id[str(row_id)] for row_id in row_ids]
        group_exact = all(
            bool(prediction.get("exact")) for prediction in group_predictions
        )
        exact_groups += group_exact
        exact_rows += sum(
            bool(prediction.get("exact")) for prediction in group_predictions
        )
        rows += len(group_predictions)
    return {
        "groups": len(groups),
        "exact_groups": exact_groups,
        "group_exact_share": exact_groups / len(groups),
        "rows": rows,
        "exact_rows": exact_rows,
        "row_exact_share": exact_rows / rows,
    }


def _combine_field_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for field in ("status", "delta", "relation"):
        values = sorted(
            {
                value
                for summary in summaries
                for value in summary["fields"][field]
            }
        )
        output[field] = {}
        for value in values:
            rows = sum(
                int(summary["fields"][field].get(value, {}).get("rows", 0))
                for summary in summaries
            )
            correct = sum(
                int(
                    summary["fields"][field]
                    .get(value, {})
                    .get("correct_count", 0)
                )
                for summary in summaries
            )
            output[field][value] = {
                "rows": rows,
                "correct_count": correct,
                "accuracy": correct / rows,
                "required_count": required_count(0.95, rows),
            }
    return output


def final_gate(
    *,
    preregistration: Mapping[str, Any],
    adversarial_rows: Sequence[Mapping[str, Any]],
    adversarial_predictions: Sequence[Mapping[str, Any]],
    swap_rows: Sequence[Mapping[str, Any]],
    swap_predictions: Sequence[Mapping[str, Any]],
    training_memory: Mapping[str, int],
    selected_adapter_bytes: int,
    disk_used_after: int,
) -> dict[str, Any]:
    adversarial = prediction_summary(
        adversarial_rows,
        adversarial_predictions,
    )
    swaps = prediction_summary(swap_rows, swap_predictions)
    summaries = (adversarial, swaps)
    model_rows = sum(int(summary["model_rows"]) for summary in summaries)
    exact_count = sum(int(summary["exact_count"]) for summary in summaries)
    parsed_count = sum(int(summary["parsed_count"]) for summary in summaries)
    fields = _combine_field_summaries(summaries)
    swap_gate = swap_summary(swap_rows, swap_predictions)
    relation_gate = relation_contrast_summary(
        preregistration,
        adversarial_predictions,
    )
    guard_predictions = [
        prediction
        for prediction in adversarial_predictions
        if not bool(prediction.get("model_called"))
    ]
    guard_rejected = sum(
        bool(prediction.get("guard_rejected"))
        for prediction in guard_predictions
    )
    scenario_totals: dict[str, dict[str, int]] = {}
    for scenario in SCENARIO_TARGETS:
        rows = sum(
            int(summary["per_scenario"][scenario]["rows"])
            for summary in summaries
        )
        exact = sum(
            int(summary["per_scenario"][scenario]["exact_count"])
            for summary in summaries
        )
        scenario_totals[scenario] = {
            "rows": rows,
            "exact_count": exact,
            "required_98_count": required_count(0.98, rows),
        }
    field_checks = {
        f"{field}_{value}": int(values["correct_count"])
        >= int(values["required_count"])
        for field, by_value in fields.items()
        for value, values in by_value.items()
    }
    family_scenarios = {
        "mixed": "MIXED",
        "risk_only": "RISK_ONLY",
        "third_party": "THIRD_PARTY",
        "negation": "NO_CLAIM",
        "attribution": "THIRD_PARTY",
    }
    family_checks = {
        name: scenario_totals[scenario]["exact_count"]
        >= scenario_totals[scenario]["required_98_count"]
        for name, scenario in family_scenarios.items()
    }
    checks = {
        "model_row_count": model_rows == 1_248,
        "overall_exact_98": exact_count >= required_count(0.98, model_rows),
        "parse_rate_100": parsed_count == model_rows,
        "evidence_validity_100": parsed_count == model_rows,
        **field_checks,
        **family_checks,
        "guard_rows_32": len(guard_predictions) == 32,
        "guard_rejections_32": guard_rejected == 32,
        "guard_model_calls_0": sum(
            bool(prediction.get("model_called"))
            for prediction in guard_predictions
        )
        == 0,
        "swap_invariance_100": swap_gate["invariant_pairs"] == 256,
        "swap_pair_exact_99": swap_gate["both_exact_pairs"]
        >= required_count(0.99, 256),
        "relation_contrast_100": relation_gate["exact_groups"] == 16,
        "training_peak_allocated_below_24_gib": int(
            training_memory["peak_allocated_bytes"]
        )
        < MAXIMUM_TRAINING_PEAK_BYTES,
        "training_peak_reserved_below_24_gib": int(
            training_memory["peak_reserved_bytes"]
        )
        < MAXIMUM_TRAINING_PEAK_BYTES,
        "selected_adapter_below_512_mib": selected_adapter_bytes
        < MAXIMUM_SELECTED_ADAPTER_BYTES,
        "filesystem_below_300_gib": disk_used_after
        < MAXIMUM_FILESYSTEM_USED_BYTES,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "combined": {
            "model_rows": model_rows,
            "exact_count": exact_count,
            "required_exact_count": required_count(0.98, model_rows),
            "exact_share": exact_count / model_rows,
            "parsed_count": parsed_count,
            "parse_share": parsed_count / model_rows,
            "fields": fields,
            "scenario_totals": scenario_totals,
        },
        "adversarial": adversarial,
        "swap": swaps,
        "swap_gate": swap_gate,
        "relation_contrast_gate": relation_gate,
        "guard": {
            "rows": len(guard_predictions),
            "rejections": guard_rejected,
            "model_calls": 0,
        },
        "training_memory": dict(training_memory),
        "selected_adapter_bytes": selected_adapter_bytes,
        "disk_used_after": disk_used_after,
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


def _ordered_train_rows(
    rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    order = _training_order(rows)
    expected_hash = preregistration["contract"]["training_order"][
        "ordered_row_ids_sha256"
    ]
    if canonical_hash(order) != expected_hash:
        raise ValueError("ECRL-1 training order hash drifted")
    by_id = {str(row["row_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(order) != set(by_id):
        raise ValueError("ECRL-1 training row identity drifted")
    return [by_id[row_id] for row_id in order]


def _prepared_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prompt = render_prompt(row)
    return {
        "row_id": str(row["row_id"]),
        "scenario_id": str(row["scenario_id"]),
        "pair_id": row.get("pair_id"),
        "prior": str(row["prior"]),
        "current": str(row["current"]),
        "target": str(row["target"]),
        "prompt": prompt,
        "guard_reason": prefilter_reason(
            str(row["prior"]),
            str(row["current"]),
        ),
    }


class Gemma4E2BRelationRunner:
    def __init__(self) -> None:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("ECRL-1 requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("ECRL-1 requires exactly one visible CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("ECRL-1 requires BF16 support")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        self.snapshot = _local_snapshot()
        load_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            self.snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        with disable_transformers_allocator_warmup():
            base = AutoModelForMultimodalLM.from_pretrained(
                self.snapshot,
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
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=LORA_TARGET_REGEX,
        )
        self.model = get_peft_model(base, lora)
        self.model.config.use_cache = False
        self.load_seconds = time.perf_counter() - load_started
        self.trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if self.trainable_parameters != TRAINABLE_PARAMETERS:
            raise RuntimeError(
                "ECRL-1 trainable parameter drift: "
                f"{self.trainable_parameters:,}"
            )
        self.lora_a_modules = sorted(
            name
            for name, module in self.model.named_modules()
            if "lora_A" in name and hasattr(module, "weight")
        )
        if not self.lora_a_modules or any(
            "language_model" not in name for name in self.lora_a_modules
        ):
            raise RuntimeError("ECRL-1 LoRA escaped the language model")

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
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("ECRL-1 guarded row escaped into training")
        user = {"role": "user", "content": prompt}
        assistant = {"role": "assistant", "content": str(row["target"])}
        prompt_inputs = self._chat_inputs([user], add_generation_prompt=True)
        full_inputs = self._chat_inputs(
            [user, assistant],
            add_generation_prompt=False,
        )
        prompt_length = int(prompt_inputs["input_ids"].shape[-1])
        full_length = int(full_inputs["input_ids"].shape[-1])
        if full_length > MAXIMUM_INPUT_TOKENS:
            raise RuntimeError(
                f"ECRL-1 training row exceeds token cap: {row['row_id']}"
            )
        if not self.torch.equal(
            full_inputs["input_ids"][0, :prompt_length],
            prompt_inputs["input_ids"][0],
        ):
            raise RuntimeError("ECRL-1 completion boundary is not prefix exact")
        encoded = {
            key: value.squeeze(0).cpu()
            for key, value in full_inputs.items()
            if self.torch.is_tensor(value)
        }
        labels = encoded["input_ids"].clone()
        labels[:prompt_length] = -100
        if bool((labels[prompt_length:] == -100).all()):
            raise RuntimeError("ECRL-1 completion labels are empty")
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
        if len(ordered_rows) != (
            OPTIMIZER_STEPS * GRADIENT_ACCUMULATION_STEPS
        ):
            raise RuntimeError("ECRL-1 train rows do not fill 256 steps")
        encoded_rows = [self.encode_training_row(row) for row in ordered_rows]
        token_counts = [int(row["total_tokens"]) for row in encoded_rows]
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        optimizer = self.torch.optim.AdamW(
            parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        checkpoint_root.mkdir(parents=True, exist_ok=False)
        (checkpoint_root / "RUN_STARTED.json").write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "m0_commit": M0_COMMIT,
                    "preregistration_sha256": PREREGISTRATION_SHA256,
                    "runner_sha256": sha256_file(Path(__file__)),
                    "seed": SEED,
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
        raw_losses: list[float] = []
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
                loss = raw_loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            raw_losses.append(float(raw_loss.detach().cpu()))
            del output, raw_loss, loss, inputs
            if micro_index % GRADIENT_ACCUMULATION_STEPS:
                continue
            optimizer_step = micro_index // GRADIENT_ACCUMULATION_STEPS
            gradient_norm = self.torch.nn.utils.clip_grad_norm_(
                parameters,
                MAXIMUM_GRADIENT_NORM,
            )
            learning_rate = LEARNING_RATE * schedule_multiplier(optimizer_step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metric = {
                "optimizer_step": optimizer_step,
                "micro_examples_seen": micro_index,
                "mean_raw_loss": (
                    sum(raw_losses[-GRADIENT_ACCUMULATION_STEPS:])
                    / GRADIENT_ACCUMULATION_STEPS
                ),
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().cpu()
                ),
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - started,
            }
            step_metrics.append(metric)
            if optimizer_step % 8 == 0 or optimizer_step == 1:
                print(json.dumps(metric, sort_keys=True), flush=True)
            if optimizer_step in CHECKPOINT_STEPS:
                checkpoint = checkpoint_root / f"step-{optimizer_step:03d}"
                self.model.save_pretrained(
                    checkpoint,
                    safe_serialization=True,
                )
                _validate_disk_cap(f"checkpoint-{optimizer_step}")
        self.torch.cuda.synchronize()
        if len(step_metrics) != OPTIMIZER_STEPS:
            raise RuntimeError("ECRL-1 optimizer-step count drifted")
        return {
            "rows": len(ordered_rows),
            "optimizer_steps": len(step_metrics),
            "mean_raw_loss": sum(raw_losses) / len(raw_losses),
            "first_step_mean_loss": step_metrics[0]["mean_raw_loss"],
            "last_step_mean_loss": step_metrics[-1]["mean_raw_loss"],
            "minimum_total_tokens": min(token_counts),
            "maximum_total_tokens": max(token_counts),
            "elapsed_seconds": time.perf_counter() - started,
            "memory": {
                "peak_allocated_bytes": int(
                    self.torch.cuda.max_memory_allocated()
                ),
                "peak_reserved_bytes": int(
                    self.torch.cuda.max_memory_reserved()
                ),
            },
            "steps": step_metrics,
        }

    def load_calibration_adapters(
        self,
        checkpoint_root: Path,
    ) -> dict[int, str]:
        adapters = {256: "default"}
        for step in (64, 128, 192):
            name = f"step_{step:03d}"
            self.model.load_adapter(
                checkpoint_root / f"step-{step:03d}",
                adapter_name=name,
                is_trainable=False,
            )
            adapters[step] = name
        return adapters

    def classify(self, row: Mapping[str, Any]) -> dict[str, Any]:
        prompt = row.get("prompt")
        if prompt is None:
            reason = row.get("guard_reason")
            if not isinstance(reason, str):
                raise RuntimeError("ECRL-1 prompt missing without guard")
            return {
                "row_id": row["row_id"],
                "scenario_id": row["scenario_id"],
                "pair_id": row.get("pair_id"),
                "expected_output": row["target"],
                "final_content": None,
                "decoded_suffix": None,
                "parsed": None,
                "exact": False,
                "model_called": False,
                "guard_rejected": True,
                "guard_reason": reason,
                "input_tokens": 0,
                "generated_tokens": 0,
                "inference_seconds": 0.0,
            }
        user = {"role": "user", "content": str(prompt)}
        inputs = self._chat_inputs([user], add_generation_prompt=True)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > MAXIMUM_INPUT_TOKENS:
            raise RuntimeError(
                f"ECRL-1 inference row exceeds token cap: {row['row_id']}"
            )
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
                max_new_tokens=MAXIMUM_NEW_TOKENS,
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
        )
        response = self.processor.parse_response(decoded)
        content = response.get("content") if isinstance(response, dict) else None
        final_content = content if isinstance(content, str) else ""
        parsed = parse_model_output(
            final_content,
            prior=str(row["prior"]),
            current=str(row["current"]),
        )
        return {
            "row_id": row["row_id"],
            "scenario_id": row["scenario_id"],
            "pair_id": row.get("pair_id"),
            "expected_output": row["target"],
            "final_content": final_content,
            "decoded_suffix": decoded,
            "parsed": parsed,
            "exact": bool(parsed["valid"]) and final_content == row["target"],
            "model_called": True,
            "guard_rejected": False,
            "guard_reason": None,
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
        predictions: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            predictions.append(self.classify(row))
            if index % 128 == 0:
                print(
                    json.dumps(
                        {
                            "adapter": adapter_name,
                            "evaluated": index,
                            "rows": len(rows),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
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


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def run(cfg: Config = Config()) -> dict[str, Any]:
    if cfg != Config():
        raise ValueError("ECRL-1 production runner paths are frozen")
    output = _path(cfg.output)
    checkpoint_root = _path(cfg.checkpoint_root)
    if output.exists():
        raise FileExistsError("ECRL-1 synthetic result is write-once")
    if checkpoint_root.exists():
        raise FileExistsError("ECRL-1 checkpoint root is write-once")
    disk_used_before = _validate_disk_cap("before-model-run")
    preregistration = validate_preregistration()
    local_model = validate_local_model()
    runtime_versions = _validate_runtime_versions()
    datasets = preregistration["contract"]["datasets"]

    # Only train and calibration rows are parsed before checkpoint selection.
    train_rows = _read_jsonl(datasets["train"]["path"])
    calibration_rows = _read_jsonl(datasets["calibration"]["path"])
    validate_split_rows(train_rows, split="train")
    validate_split_rows(calibration_rows, split="calibration")
    prepared_train = [_prepared_row(row) for row in train_rows]
    prepared_calibration = [_prepared_row(row) for row in calibration_rows]
    ordered_train = _ordered_train_rows(prepared_train, preregistration)

    _seed_everything(SEED)
    started_at = _utc_now()
    runner = Gemma4E2BRelationRunner()
    training = runner.train(ordered_train, checkpoint_root)
    disk_used_after_training = _validate_disk_cap("after-training")
    checkpoint_manifests = {
        str(step): _directory_manifest(
            checkpoint_root / f"step-{step:03d}"
        )
        for step in CHECKPOINT_STEPS
    }
    adapter_names = runner.load_calibration_adapters(checkpoint_root)
    calibration_summaries: dict[int, dict[str, Any]] = {}
    calibration_predictions: dict[str, list[dict[str, Any]]] = {}
    for step in CHECKPOINT_STEPS:
        summary, predictions = runner.evaluate(
            prepared_calibration,
            adapter_name=adapter_names[step],
        )
        calibration_summaries[step] = summary
        calibration_predictions[str(step)] = predictions
    selected_step, ranking = select_checkpoint(calibration_summaries)
    selected_adapter_name = adapter_names[selected_step]

    # Final split content becomes parseable only after checkpoint selection.
    adversarial_rows = _read_jsonl(datasets["adversarial"]["path"])
    swap_rows = _read_jsonl(datasets["swap"]["path"])
    validate_split_rows(adversarial_rows, split="adversarial")
    validate_split_rows(swap_rows, split="swap")
    prepared_adversarial = [_prepared_row(row) for row in adversarial_rows]
    prepared_swap = [_prepared_row(row) for row in swap_rows]
    runner.reset_inference_memory()
    _, adversarial_predictions = runner.evaluate(
        prepared_adversarial,
        adapter_name=selected_adapter_name,
    )
    _, swap_predictions = runner.evaluate(
        prepared_swap,
        adapter_name=selected_adapter_name,
    )
    inference_memory = runner.inference_memory()
    selected_checkpoint_manifest = checkpoint_manifests[str(selected_step)]
    disk_used_after = _validate_disk_cap("after-model-run")
    gate = final_gate(
        preregistration=preregistration,
        adversarial_rows=prepared_adversarial,
        adversarial_predictions=adversarial_predictions,
        swap_rows=prepared_swap,
        swap_predictions=swap_predictions,
        training_memory=training["memory"],
        selected_adapter_bytes=int(selected_checkpoint_manifest["bytes"]),
        disk_used_after=disk_used_after,
    )

    selected_manifest: dict[str, Any] | None = None
    if gate["passed"]:
        selected_path = checkpoint_root / "selected"
        shutil.copytree(
            checkpoint_root / f"step-{selected_step:03d}",
            selected_path,
        )
        selected_manifest = _directory_manifest(selected_path)
        if int(selected_manifest["bytes"]) >= MAXIMUM_SELECTED_ADAPTER_BYTES:
            raise RuntimeError("ECRL-1 copied selected adapter exceeds cap")
        _validate_disk_cap("after-selected-copy")

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "m0": {
            "commit": M0_COMMIT,
            "path": str(PREREGISTRATION),
            "file_sha256": PREREGISTRATION_SHA256,
            "self_hash": PREREGISTRATION_SELF_HASH,
            "contract_hash": PREREGISTRATION_CONTRACT_HASH,
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(Path(__file__)),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_validation": local_model,
            "runtime_versions": runtime_versions,
            "load_seconds": runner.load_seconds,
            "trainable_parameters": runner.trainable_parameters,
            "lora_a_modules": runner.lora_a_modules,
        },
        "outcome_boundary": {
            "SEC_body_requests": 0,
            "SEC_header_requests": 0,
            "historical_pairs_created": 0,
            "historical_semantic_model_calls": 0,
            "market_rows_read": 0,
            "funding_rows_read": 0,
            "premium_rows_read": 0,
            "reward_rows_read": 0,
            "2024_or_later_rows_read": 0,
        },
        "disk": {
            "used_before": disk_used_before,
            "used_after_training": disk_used_after_training,
            "used_after_final": disk_used_after,
            "cap_bytes": MAXIMUM_FILESYSTEM_USED_BYTES,
        },
        "training": training,
        "checkpoints": {
            "manifests": checkpoint_manifests,
            "selected_step": selected_step,
            "selected_manifest": selected_manifest,
        },
        "calibration": {
            "summaries": {
                str(step): calibration_summaries[step]
                for step in CHECKPOINT_STEPS
            },
            "predictions": calibration_predictions,
            "ranking": ranking,
            "selected_step": selected_step,
        },
        "final": {
            "gate": gate,
            "inference_memory": inference_memory,
            "adversarial_predictions": adversarial_predictions,
            "swap_predictions": swap_predictions,
        },
        "decision": {
            "status": (
                "passed" if gate["passed"] else "retired_synthetic_failure"
            ),
            "synthetic_gate_passed": bool(gate["passed"]),
            "historical_body_transport_authorized": bool(gate["passed"]),
            "historical_semantic_execution_authorized": bool(gate["passed"]),
            "economic_evaluation_authorized": False,
            "2024_or_later_authorized": False,
            "repair_authorized": False,
            "next_step": (
                "commit result and adapter binding, delete unselected "
                "checkpoints, then freeze the historical SEC support builder"
                if gate["passed"]
                else "retire ECRL-1 unchanged before historical bodies and outcomes"
            ),
        },
    }
    result["self_hash"] = canonical_hash(result)
    _write_result(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate M0/model hashes without parsing a synthetic split",
    )
    args = parser.parse_args()
    if args.validate_only:
        preregistration = validate_preregistration()
        model = validate_local_model()
        print(
            json.dumps(
                {
                    "m0_commit": M0_COMMIT,
                    "contract_hash": preregistration["contract_hash"],
                    "model_revision": model["revision"],
                    "runtime_versions": _validate_runtime_versions(),
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
