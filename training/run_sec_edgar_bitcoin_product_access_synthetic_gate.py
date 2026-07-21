"""Run the frozen BPAX-120 Gemma 4 E2B synthetic semantic and memory gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.preregister_sec_edgar_bitcoin_product_access import (
    AS_OF_DATE,
    META_INSTRUCTION_PATTERN,
    MODEL_ID,
    MODEL_REVISION,
    POLICY_ID,
    PROMPT,
    SYNTHETIC_CASES,
    Config as PreregistrationConfig,
    canonical_hash,
    parse_model_output,
    redact_excerpt,
    sha256_file,
    validate_local_model,
)
from utils import disable_transformers_allocator_warmup


PROTOCOL_VERSION = "sec_edgar_bitcoin_product_access_synthetic_gate_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/sec_edgar_bitcoin_product_access_preregistration_2026-07-22.json"
)
PREREGISTRATION_SHA256 = (
    "ab975eea454fbe1a784adaee979c5ad6162be9b18363c7fe3aa47959e075b883"
)
PREREGISTRATION_CONTRACT_HASH = (
    "124d37318a0e2bd60f81e81f76484fd3f3e356168bee71b137603b65f0ddb7ff"
)
SYNTHETIC_CASES_SHA256 = (
    "d258922f785479758c52827b8837053ac259e881ff13f134294e65742aeaa6a0"
)


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/sec_edgar_bitcoin_product_access_synthetic_gate_2026-07-22.json"
    )
    maximum_input_tokens: int = 1_536
    maximum_new_tokens: int = 96
    maximum_peak_allocated_bytes: int = 7 * 1024**3
    maximum_peak_reserved_bytes: int = int(7.25 * 1024**3)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _validate_config(cfg: Config) -> None:
    if cfg != Config(output=cfg.output):
        raise ValueError("BPAX-120 synthetic configuration is frozen")


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("BPAX-120 preregistration artifact hash mismatch")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    if payload.get("contract_hash") != PREREGISTRATION_CONTRACT_HASH:
        raise ValueError("BPAX-120 preregistration contract hash mismatch")
    contract = payload.get("contract", {})
    if (
        contract.get("synthetic_controls", {}).get("cases_sha256")
        != SYNTHETIC_CASES_SHA256
    ):
        raise ValueError("BPAX-120 frozen synthetic case hash mismatch")
    prereg_cfg = PreregistrationConfig()
    gate = contract.get("synthetic_gate", {})
    expected_gate = {
        "maximum_peak_allocated_bytes": prereg_cfg.maximum_peak_allocated_bytes,
        "maximum_peak_reserved_bytes": prereg_cfg.maximum_peak_reserved_bytes,
    }
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise ValueError("BPAX-120 frozen synthetic memory gate mismatch")
    source_anchor = payload.get("anchors", {}).get("preregistration_source", {})
    source_path = source_anchor.get("path")
    source_sha256 = source_anchor.get("sha256")
    if not isinstance(source_path, str) or not isinstance(source_sha256, str):
        raise ValueError("BPAX-120 preregistration source anchor is missing")
    if sha256_file(source_path) != source_sha256:
        raise ValueError("BPAX-120 live preregistration source drifted")
    if canonical_hash(SYNTHETIC_CASES) != SYNTHETIC_CASES_SHA256:
        raise ValueError("BPAX-120 live synthetic constants drifted")
    prompt_sha256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    if contract.get("prompt_sha256") != prompt_sha256:
        raise ValueError("BPAX-120 live prompt drifted")
    model = contract.get("model", {})
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("BPAX-120 live model constants drifted")
    redaction = contract.get("preprocessing", {}).get("redaction_implementation", {})
    redaction_path = redaction.get("path")
    redaction_sha256 = redaction.get("sha256")
    if not isinstance(redaction_path, str) or not isinstance(redaction_sha256, str):
        raise ValueError("BPAX-120 redaction implementation anchor is missing")
    if sha256_file(redaction_path) != redaction_sha256:
        raise ValueError("BPAX-120 redaction implementation drifted")
    decision = payload.get("decision", {})
    if not decision.get("synthetic_model_gate_authorized"):
        raise ValueError("BPAX-120 synthetic model gate is not authorized")
    if decision.get("filing_body_transport_authorized"):
        raise ValueError("BPAX-120 preregistration unexpectedly opens filing bodies")
    if decision.get("economic_evaluation_authorized"):
        raise ValueError("BPAX-120 preregistration unexpectedly opens outcomes")
    return payload


def frozen_cases(
    preregistration: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payload = (
        dict(preregistration)
        if preregistration is not None
        else validate_preregistration()
    )
    artifact_cases = (
        payload.get("contract", {}).get("synthetic_controls", {}).get("cases")
    )
    if not isinstance(artifact_cases, list) or len(artifact_cases) != len(
        SYNTHETIC_CASES
    ):
        raise ValueError("BPAX-120 artifact synthetic cases are incomplete")
    cases: list[dict[str, Any]] = []
    for source in artifact_cases:
        if not isinstance(source, dict):
            raise ValueError("BPAX-120 artifact synthetic case is malformed")
        row = dict(source)
        observed_redaction = redact_excerpt(
            str(row["source"]),
            issuer_aliases=tuple(row["issuer_aliases"]),
            issuer_tickers=tuple(row["issuer_tickers"]),
        )
        if row.get("redacted_excerpt") != observed_redaction:
            raise ValueError(f"BPAX-120 redaction drift for case {row.get('name')}")
        row["guard_match"] = bool(
            META_INSTRUCTION_PATTERN.search(str(row["redacted_excerpt"]))
        )
        row["case_hash"] = canonical_hash(row)
        cases.append(row)
    return cases


class Gemma4E2BExtractor:
    def __init__(self, cfg: Config) -> None:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        self.cfg = cfg
        self.torch = torch
        if "HF_HUB_CACHE" in os.environ:
            hub = Path(os.environ["HF_HUB_CACHE"])
        elif "HF_HOME" in os.environ:
            hub = Path(os.environ["HF_HOME"]) / "hub"
        else:
            hub = Path.home() / ".cache" / "huggingface" / "hub"
        self.snapshot = (
            hub / "models--google--gemma-4-E2B-it" / "snapshots" / MODEL_REVISION
        )
        if not torch.cuda.is_available():
            raise RuntimeError("BPAX-120 frozen Gemma 4 E2B gate requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "BPAX-120 frozen runtime requires exactly one visible CUDA device"
            )
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.processor = AutoProcessor.from_pretrained(
            self.snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        started = time.perf_counter()
        with disable_transformers_allocator_warmup():
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.snapshot,
                local_files_only=True,
                quantization_config=quantization,
                device_map={"": 0},
                dtype=torch.float16,
                attn_implementation="eager",
                trust_remote_code=False,
            ).eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    def classify(self, excerpt: str) -> dict[str, Any]:
        content = PROMPT.format(excerpt=excerpt)
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_count = int(inputs["input_ids"].shape[-1])
        if token_count > self.cfg.maximum_input_tokens:
            raise RuntimeError("frozen synthetic prompt exceeds the token cap")
        inputs = inputs.to(self.model.device)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.maximum_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.model.generation_config.eos_token_id,
            )
        self.torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        prefix = int(inputs["input_ids"].shape[-1])
        generated_suffix = generated[0, prefix:]
        decoded = self.processor.decode(
            generated_suffix, skip_special_tokens=False
        ).strip()
        response = self.processor.parse_response(decoded)
        content_value = response.get("content") if isinstance(response, dict) else None
        final_content = content_value if isinstance(content_value, str) else ""
        parsed = parse_model_output(final_content, excerpt)
        return {
            "input_tokens": token_count,
            "generated_tokens": int(generated_suffix.shape[-1]),
            "inference_seconds": inference_seconds,
            "decoded_suffix": decoded,
            "final_content": final_content,
            "parsed": parsed,
        }

    def runtime_metrics(self) -> dict[str, Any]:
        properties = self.torch.cuda.get_device_properties(0)
        return {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "model_load_seconds": self.load_seconds,
            "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
            "cuda_device_count": self.torch.cuda.device_count(),
        }


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    cfg: Config = Config(),
    *,
    expected_cases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = list(expected_cases) if expected_cases is not None else frozen_cases()
    expected_manifest = [
        (
            str(row["name"]),
            str(row["case_hash"]),
            bool(row["guarded"]),
            row.get("equivalence_group"),
        )
        for row in expected
    ]
    observed_manifest = [
        (
            str(row["name"]),
            str(row["case_hash"]),
            bool(row["guarded"]),
            row.get("equivalence_group"),
        )
        for row in records
    ]
    if observed_manifest != expected_manifest:
        raise ValueError("synthetic record manifest differs from frozen cases")
    expected_total = len(expected_manifest)
    model_records = [row for row in records if not bool(row["guarded"])]
    guarded_records = [row for row in records if bool(row["guarded"])]
    parsed = sum(bool(row["parsed_ok"]) for row in model_records)
    exact = sum(bool(row["expected_match"]) for row in records)
    quote_valid = sum(bool(row["quote_valid"]) for row in model_records)
    guard_correct = sum(bool(row["guard_correct"]) for row in guarded_records)
    equivalence: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        group = row.get("equivalence_group")
        if group:
            equivalence[str(group)].append(row)
    equivalence_checks: dict[str, bool] = {}
    for group, rows in equivalence.items():
        prompts = {str(row["redacted_excerpt"]) for row in rows}
        outputs = {str(row["actual_class"]) for row in rows}
        equivalence_checks[group] = len(rows) >= 2 and len(prompts) == len(outputs) == 1
    allocated = int(runtime.get("peak_allocated_bytes", -1))
    reserved = int(runtime.get("peak_reserved_bytes", -1))
    checks = {
        "all_expected_classes": exact == expected_total,
        "all_model_outputs_parse": parsed == len(model_records),
        "all_model_quotes_validate": quote_valid == len(model_records),
        "all_guarded_cases_skip_model_and_are_unsupported": guard_correct
        == len(guarded_records),
        "all_equivalence_groups_invariant": bool(equivalence_checks)
        and all(equivalence_checks.values()),
        "peak_allocated_within_cap": 0 <= allocated <= cfg.maximum_peak_allocated_bytes,
        "peak_reserved_within_cap": 0 <= reserved <= cfg.maximum_peak_reserved_bytes,
    }
    return {
        "counts": {
            "cases": expected_total,
            "model_calls": len(model_records),
            "guarded_cases": len(guarded_records),
            "parsed_model_outputs": parsed,
            "exact_expected_cases": exact,
            "quote_valid_model_outputs": quote_valid,
            "actual_classes": dict(
                sorted(Counter(str(row["actual_class"]) for row in records).items())
            ),
        },
        "memory_caps": {
            "maximum_peak_allocated_bytes": cfg.maximum_peak_allocated_bytes,
            "maximum_peak_reserved_bytes": cfg.maximum_peak_reserved_bytes,
            "observed_peak_allocated_bytes": allocated,
            "observed_peak_reserved_bytes": reserved,
        },
        "equivalence_checks": equivalence_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    preregistration = validate_preregistration()
    model_validation = validate_local_model()
    cases = frozen_cases(preregistration)
    extractor = Gemma4E2BExtractor(cfg)
    records: list[dict[str, Any]] = []
    for case in cases:
        guarded = bool(case["guarded"])
        guard_correct = guarded and bool(case["guard_match"])
        if guarded:
            parsed = {"class": "UNSUPPORTED", "quote": ""}
            generated: dict[str, Any] = {
                "input_tokens": 0,
                "generated_tokens": 0,
                "inference_seconds": 0.0,
                "decoded_suffix": "",
                "final_content": "",
                "parsed": parsed,
            }
        else:
            if case["guard_match"]:
                raise RuntimeError(
                    f"non-guard synthetic case unexpectedly matched guard: {case['name']}"
                )
            generated = extractor.classify(str(case["redacted_excerpt"]))
            parsed = generated["parsed"]
        actual_class = parsed["class"] if isinstance(parsed, dict) else "PARSE_FAILURE"
        quote = parsed["quote"] if isinstance(parsed, dict) else ""
        quote_valid = bool(
            isinstance(parsed, dict)
            and (
                (actual_class == "UNSUPPORTED" and quote == "")
                or (
                    actual_class != "UNSUPPORTED"
                    and quote != ""
                    and quote in case["redacted_excerpt"]
                )
            )
        )
        records.append(
            {
                "name": case["name"],
                "case_hash": case["case_hash"],
                "redacted_excerpt": case["redacted_excerpt"],
                "expected_class": case["expected_class"],
                "actual_class": actual_class,
                "evidence_quote": quote,
                "parsed_ok": isinstance(parsed, dict),
                "quote_valid": quote_valid,
                "expected_match": actual_class == case["expected_class"],
                "guarded": guarded,
                "guard_match": case["guard_match"],
                "guard_correct": guard_correct if guarded else True,
                "equivalence_group": case.get("equivalence_group"),
                "input_tokens": generated["input_tokens"],
                "generated_tokens": generated["generated_tokens"],
                "inference_seconds": generated["inference_seconds"],
                "decoded_suffix": generated["decoded_suffix"],
                "final_content": generated["final_content"],
            }
        )
    runtime = extractor.runtime_metrics()
    evaluation = evaluate_records(records, runtime, cfg, expected_cases=cases)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "policy_id": POLICY_ID,
        "anchors": {
            "preregistration": {
                "path": str(PREREGISTRATION),
                "sha256": PREREGISTRATION_SHA256,
                "contract_hash": preregistration["contract_hash"],
            },
            "runner": {
                "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(Path(__file__)),
            },
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "validation": model_validation,
            },
            "synthetic_cases_sha256": SYNTHETIC_CASES_SHA256,
        },
        "runtime": runtime,
        "evaluation": evaluation,
        "records": records,
        "outcome_boundary": {
            "filing_bodies_opened": 0,
            "historical_semantic_rows_opened": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "2024_or_later_source_rows_read": 0,
            "clean_room_claimed": False,
        },
        "decision": {
            "synthetic_gate_passed": bool(evaluation["passed"]),
            "filing_body_transport_authorized": bool(evaluation["passed"]),
            "historical_semantic_execution_authorized": False,
            "novelty_evaluation_authorized": False,
            "economic_evaluation_authorized": False,
            "2024_or_later_authorized": False,
            "target_3060ti_live_deployment_authorized": False,
            "next_step": (
                "build and audit frozen pre-2024 SEC body transport; keep 3060 Ti live deployment sealed"
                if evaluation["passed"]
                else "retire BPAX-120 without opening SEC bodies or market outcomes"
            ),
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    output = _path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Config().output)
    args = parser.parse_args()
    payload = run(Config(output=args.output))
    print(json.dumps(payload["evaluation"], indent=2, sort_keys=True))
    if not payload["evaluation"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
