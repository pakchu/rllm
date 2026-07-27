#!/usr/bin/env python3
"""Run the one-shot PSIM-D8-RLLM2 base memorization gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import preregister_psim_d8_rllm2_operational_successor as prereg
from training import run_psim_d8_rllm1_base_memorization_gate as base
from utils import disable_transformers_allocator_warmup


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = "psim_d8_rllm2_base_memorization_gate_v1"
DEFAULT_OUTPUT = prereg.RLLM2_RESULT
DEFAULT_ATTEMPT = prereg.RLLM2_ATTEMPT
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "85ede8a56393b11f4f1ced7e304adb3c2639132c1f0b008ed973aae92af9ef54"
)
PREREGISTRATION_MANIFEST_HASH = (
    "c9b8a7527d90e8de3b1aeadac834c4b9d7a97bc3358c08256f79fa24fc18266c"
)
SCIENTIFIC_CONTRACT_HASH = (
    "59a7c1dd03155d8552614e4886087ca1dd08db4cc8c8953257c2f6f68d28af23"
)
CASE_ROSTER_HASH = prereg.CASE_ROSTER_HASH
FAILURE_ACTION = (
    "REJECT_PSIM_D8_RLLM2_BEFORE_NEXT_MARKET_STAGE_NO_REPAIR_"
    "RESAMPLE_MODEL_SWAP_OR_RERUN"
)
PASS_ACTION = (
    "ACCEPT_PSIM_D8_RLLM2_BASE_MEMORIZATION_GATE_SOURCE_FEATURES_ONLY"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return base.canonical_json_bytes(payload, pretty=pretty)


def canonical_hash(payload: Any) -> str:
    return base.canonical_hash(payload)


def sha256_file(path: str | Path) -> str:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"unsafe PSIM-D8-RLLM2 file: {path}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    return base._display_path(path)


def _clean_committed_head() -> str:
    return base._clean_committed_head()


def validate_preregistration() -> dict[str, Any]:
    target = repository_path(PREREGISTRATION)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("unsafe PSIM-D8-RLLM2 preregistration artifact")
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREREGISTRATION_SHA256:
        raise RuntimeError("PSIM-D8-RLLM2 preregistration SHA changed")
    payload = json.loads(raw.decode("utf-8"))
    if (
        payload != prereg.build_preregistration()
        or payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload.get("candidate", {}).get("id") != prereg.POLICY_ID
        or payload.get("inherited_scientific_contract", {}).get(
            "contract_hash"
        )
        != SCIENTIFIC_CONTRACT_HASH
        or payload.get("inherited_scientific_contract", {}).get(
            "case_roster_hash"
        )
        != CASE_ROSTER_HASH
        or payload.get("access_boundary", {}).get(
            "market_or_funding_payload_bytes_hashed"
        )
        is not False
    ):
        raise RuntimeError("PSIM-D8-RLLM2 preregistration contract changed")
    return payload


def _normalized_device_target(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value).strip().lower()
    if text == "0":
        return "cuda:0"
    if text == "cuda":
        return "cuda:0"
    return text


def validate_loaded_model_placement(
    model: Any,
    torch_module: Any,
) -> dict[str, Any]:
    model_class = type(model).__name__
    if model_class != "Gemma4ForConditionalGeneration":
        raise RuntimeError(f"frozen model class changed: {model_class}")
    if not bool(getattr(model, "is_quantized", False)):
        raise RuntimeError("frozen model did not load quantized")
    expected = torch_module.device("cuda:0")
    model_device = torch_module.device(model.device)
    try:
        first_parameter = next(model.parameters())
    except StopIteration as exc:
        raise RuntimeError("frozen model has no parameters") from exc
    parameter_device = torch_module.device(first_parameter.device)
    if model_device != expected or parameter_device != expected:
        raise RuntimeError(
            "frozen model is not resident on CUDA device zero: "
            f"model={model_device}, parameter={parameter_device}"
        )
    advisory = dict(getattr(model, "hf_device_map", {}) or {})
    serializable_advisory = {
        str(key): (
            value
            if isinstance(value, (int, str))
            else str(value)
        )
        for key, value in advisory.items()
    }
    normalized = {
        str(key): _normalized_device_target(value)
        for key, value in advisory.items()
    }
    if normalized and set(normalized.values()) != {"cuda:0"}:
        raise RuntimeError(
            f"frozen model advisory device map is unsafe: {normalized!r}"
        )
    allocated = int(torch_module.cuda.memory_allocated(0))
    if allocated <= 0:
        raise RuntimeError("frozen model allocated no CUDA memory")
    return {
        "model_class": model_class,
        "is_quantized": True,
        "model_device": str(model_device),
        "first_parameter_device": str(parameter_device),
        "hf_device_map_advisory": serializable_advisory,
        "hf_device_map_normalized": normalized,
        "empty_hf_device_map_accepted": not bool(advisory),
        "cuda_memory_allocated_after_load_bytes": allocated,
        "validated": True,
    }


class Gemma4CodeScorer:
    def __init__(self, processor: Any) -> None:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("PSIM-D8-RLLM2 base gate requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "PSIM-D8-RLLM2 base gate requires one visible CUDA device"
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("PSIM-D8-RLLM2 base gate requires CUDA BF16")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.torch = torch
        self.processor = processor
        self.forward_calls_started = 0
        self.snapshot = base._model_snapshot()
        tokenizer_contract = base.validate_processor_tokenizer(processor)
        self.code_ids = dict(
            tokenizer_contract["challenge_code_token_ids"]
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        started = time.perf_counter()
        with disable_transformers_allocator_warmup():
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.snapshot,
                local_files_only=True,
                quantization_config=quantization,
                device_map={"": 0},
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=False,
            ).eval()
        self.load_seconds = time.perf_counter() - started
        self.placement = validate_loaded_model_placement(
            self.model,
            torch,
        )

    def score(self, prompt: str) -> dict[str, Any]:
        inputs = base._processor_inputs(self.processor, prompt)
        tokens = int(inputs["input_ids"].shape[-1])
        if tokens > base.MAXIMUM_INPUT_TOKENS:
            raise RuntimeError("challenge prompt exceeds frozen token cap")
        inputs = inputs.to(self.model.device)
        started = time.perf_counter()
        self.forward_calls_started += 1
        with self.torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        elapsed = time.perf_counter() - started
        logits = outputs.logits[0, -1].float()
        scores = {
            code: float(logits[token_id].item())
            for code, token_id in self.code_ids.items()
        }
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError("memorization code logits are nonfinite")
        best = max(scores.values())
        predicted = next(
            code
            for code in base.prereg.MEMORIZATION_CHALLENGE_CODES
            if scores[code] == best
        )
        return {
            "predicted_code": predicted,
            "code_logits": scores,
            "input_tokens": tokens,
            "inference_seconds": elapsed,
        }

    def metrics(self) -> dict[str, Any]:
        properties = self.torch.cuda.get_device_properties(0)
        peak_allocated = int(self.torch.cuda.max_memory_allocated())
        peak_reserved = int(self.torch.cuda.max_memory_reserved())
        if peak_allocated > base.MAXIMUM_PEAK_ALLOCATED_BYTES:
            raise RuntimeError("PSIM-D8-RLLM2 peak VRAM cap exceeded")
        return {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "cuda_device_count": self.torch.cuda.device_count(),
            "bf16_supported": self.torch.cuda.is_bf16_supported(),
            "model_load_seconds": self.load_seconds,
            "placement": self.placement,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "maximum_peak_allocated_bytes": (
                base.MAXIMUM_PEAK_ALLOCATED_BYTES
            ),
        }


def prepare_source_only_gate() -> dict[str, Any]:
    successor = validate_preregistration()
    inherited = base.prepare_source_only_gate()
    cases = inherited["cases"]
    roster_hash = canonical_hash(
        [case["case_hash"] for case in cases]
    )
    if roster_hash != CASE_ROSTER_HASH:
        raise RuntimeError("PSIM-D8-RLLM2 case roster changed")
    return {
        "successor_preregistration": successor,
        **inherited,
    }


def _attempt_payload(
    *,
    execution_commit: str,
    output_path: Path,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "execution_commit": execution_commit,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": _display_path(output_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_id": base.prereg.MODEL_ID,
        "model_revision": base.prereg.MODEL_REVISION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_manifest_hash": (
            PREREGISTRATION_MANIFEST_HASH
        ),
        "scientific_contract_hash": SCIENTIFIC_CONTRACT_HASH,
        "predecessor_failure_result_hash": (
            prereg.RLLM1_FAILURE_RESULT_HASH
        ),
        "case_roster_hash": canonical_hash(
            [case["case_hash"] for case in cases]
        ),
        "model_inference_authorized": True,
        "market_access_authorized": False,
    }
    return {**core, "attempt_hash": canonical_hash(core)}


def _failure_payload(
    *,
    execution_commit: str,
    attempt_path: Path,
    attempt_payload: Mapping[str, Any],
    stage: str,
    error: Exception,
    forwards_started: int,
    predictions_created: int,
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "execution_commit": execution_commit,
        "attempt": {
            "path": _display_path(attempt_path),
            "sha256": sha256_file(attempt_path),
            "attempt_hash": attempt_payload["attempt_hash"],
        },
        "decision": "reject",
        "terminal_action": FAILURE_ACTION,
        "failure": {
            "stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "observations": {
            "model_forwards_started": forwards_started,
            "challenge_predictions_created": predictions_created,
            "challenge_statistics_computed": False,
        },
        "market_access_authorized": False,
        "source_feature_construction_authorized": False,
        "rerun_authorized": False,
        "access_boundary": {
            "market_or_funding_paths_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
    }
    return {**core, "result_hash": canonical_hash(core)}


def _success_payload(
    *,
    execution_commit: str,
    attempt_path: Path,
    attempt_payload: Mapping[str, Any],
    prepared: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    scorer: Any,
) -> dict[str, Any]:
    cases = prepared["cases"]
    model_metrics = scorer.metrics()
    evaluation = dict(base.evaluate_predictions(cases, predictions))
    evaluation["terminal_action"] = (
        FAILURE_ACTION
        if evaluation["decision"] == "reject"
        else PASS_ACTION
    )
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "execution_commit": execution_commit,
        "attempt": {
            "path": _display_path(attempt_path),
            "sha256": sha256_file(attempt_path),
            "attempt_hash": attempt_payload["attempt_hash"],
        },
        "preregistration": {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            "scientific_contract_hash": SCIENTIFIC_CONTRACT_HASH,
        },
        "predecessor_terminal_evidence": prepared[
            "successor_preregistration"
        ]["predecessor_terminal_evidence"],
        "source_authority": prepared["preregistration"]["source_authority"],
        "runtime": prepared["runtime"],
        "prompt_capacity": prepared["prompt_capacity"],
        "challenge": {
            "version": base.CHALLENGE_VERSION,
            "case_count": len(cases),
            "case_roster_hash": canonical_hash(
                [case["case_hash"] for case in cases]
            ),
            "choice_codes": list(
                base.prereg.MEMORIZATION_CHALLENGE_CODES
            ),
            "predictions": predictions,
            **evaluation,
        },
        "model_metrics": model_metrics,
        "access_boundary": {
            "source_paths_read": sorted(
                {
                    PREREGISTRATION.as_posix(),
                    *prepared["successor_preregistration"][
                        "access_boundary"
                    ]["files_read"],
                    *prepared["preregistration"]["access_boundary"][
                        "source_files_read"
                    ],
                }
            ),
            "model_snapshot_read": str(base._model_snapshot()),
            "market_or_funding_paths_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
    }
    return {**core, "result_hash": canonical_hash(core)}


def run_gate(
    *,
    scorer_factory: Callable[[Any], Any] = Gemma4CodeScorer,
) -> dict[str, Any]:
    output_path = repository_path(DEFAULT_OUTPUT)
    attempt_path = repository_path(DEFAULT_ATTEMPT)
    if (
        output_path.exists()
        or output_path.is_symlink()
        or attempt_path.exists()
        or attempt_path.is_symlink()
    ):
        raise RuntimeError(
            "PSIM-D8-RLLM2 base memorization gate already attempted"
        )
    execution_commit = _clean_committed_head()
    prepared = prepare_source_only_gate()
    cases = prepared["cases"]
    attempt_payload = _attempt_payload(
        execution_commit=execution_commit,
        output_path=output_path,
        cases=cases,
    )
    write_once(attempt_path, attempt_payload)
    stage = "MODEL_CONSTRUCTION"
    scorer: Any | None = None
    predictions: list[dict[str, Any]] = []
    try:
        scorer = scorer_factory(prepared["processor"])
        stage = "CHALLENGE_FORWARD"
        for case in cases:
            scored = scorer.score(base.render_challenge_prompt(case))
            predictions.append(
                {
                    "case_hash": case["case_hash"],
                    "protocol": case["protocol"],
                    "effective_year": case["effective_year"],
                    "true_code": case["true_code"],
                    **scored,
                    "correct": (
                        scored["predicted_code"] == case["true_code"]
                    ),
                }
            )
        stage = "RESULT_ASSEMBLY"
        payload = _success_payload(
            execution_commit=execution_commit,
            attempt_path=attempt_path,
            attempt_payload=attempt_payload,
            prepared=prepared,
            predictions=predictions,
            scorer=scorer,
        )
        write_once(output_path, payload)
        return payload
    except Exception as error:
        forwards_started = int(
            getattr(scorer, "forward_calls_started", len(predictions))
        )
        failure = _failure_payload(
            execution_commit=execution_commit,
            attempt_path=attempt_path,
            attempt_payload=attempt_payload,
            stage=stage,
            error=error,
            forwards_started=forwards_started,
            predictions_created=len(predictions),
        )
        if not output_path.exists() and not output_path.is_symlink():
            write_once(output_path, failure)
        raise


def write_once(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = repository_path(path)
    encoded = canonical_json_bytes(dict(payload), pretty=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RuntimeError(
            "PSIM-D8-RLLM2 base memorization gate already attempted"
        )
    with target.open("xb") as handle:
        handle.write(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate successor preregistration, inherited source, exact "
            "runtime, tokenizer, roster, and prompt caps without loading "
            "model weights or consuming the official attempt."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        prepared = prepare_source_only_gate()
        cases = prepared["cases"]
        print(
            json.dumps(
                {
                    "candidate": prereg.POLICY_ID,
                    "runtime": prepared["runtime"],
                    "prompt_capacity": prepared["prompt_capacity"],
                    "case_roster_hash": canonical_hash(
                        [case["case_hash"] for case in cases]
                    ),
                    "scientific_contract_hash": (
                        SCIENTIFIC_CONTRACT_HASH
                    ),
                    "model_weights_loaded": False,
                    "official_attempt_consumed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    payload = run_gate()
    print(
        json.dumps(
            {
                "decision": payload["challenge"]["decision"],
                "terminal_action": payload["challenge"][
                    "terminal_action"
                ],
                "statistics": payload["challenge"]["statistics"],
                "result_hash": payload["result_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
