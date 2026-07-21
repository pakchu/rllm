"""Run the frozen EBCT-72 Gemma 4 synthetic semantic gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.preregister_sec_edgar_bitcoin_constraint_transition_breadth import (
    AS_OF_DATE,
    META_INSTRUCTION_PATTERN,
    MODEL_ID,
    MODEL_REVISION,
    POLICY_ID,
    PROMPT,
    SYNTHETIC_CASES,
    canonical_hash,
    parse_model_output,
    redact_excerpt,
    sha256_file,
    validate_local_model,
)
from utils import disable_transformers_allocator_warmup


PROTOCOL_VERSION = "sec_edgar_bitcoin_constraint_synthetic_gate_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/sec_edgar_bitcoin_constraint_transition_breadth_"
    "preregistration_2026-07-21.json"
)
PREREGISTRATION_SHA256 = (
    "a9c55b98202b341ffb51bede731e5d2a2281d3851fbee604670868ea47470405"
)
PREREGISTRATION_CONTRACT_HASH = (
    "7c52d3c0b6c5b2869ab90723d864e73f0f06097f657a67167f8557b420465e48"
)
SYNTHETIC_CASES_SHA256 = (
    "43eee006a625c16463f11ca0421be5fe0608cf7b91946d92e18fc4e2eb72fa00"
)


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/sec_edgar_bitcoin_constraint_synthetic_gate_2026-07-21.json"
    )
    maximum_input_tokens: int = 1_536
    maximum_new_tokens: int = 160


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _validate_config(cfg: Config) -> None:
    if cfg != Config(output=cfg.output):
        raise ValueError("EBCT-72 synthetic configuration is frozen")


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("EBCT-72 preregistration artifact hash mismatch")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    if payload.get("contract_hash") != PREREGISTRATION_CONTRACT_HASH:
        raise ValueError("EBCT-72 preregistration contract hash mismatch")
    if payload.get("contract", {}).get("synthetic_controls", {}).get(
        "cases_sha256"
    ) != SYNTHETIC_CASES_SHA256:
        raise ValueError("EBCT-72 frozen synthetic case hash mismatch")
    decision = payload.get("decision", {})
    if not decision.get("synthetic_model_gate_authorized"):
        raise ValueError("EBCT-72 synthetic model gate is not authorized")
    if decision.get("filing_body_transport_authorized"):
        raise ValueError("EBCT-72 preregistration unexpectedly opens filing bodies")
    if decision.get("economic_evaluation_authorized"):
        raise ValueError("EBCT-72 preregistration unexpectedly opens outcomes")
    return payload


def frozen_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for source in SYNTHETIC_CASES:
        row = dict(source)
        row["redacted_excerpt"] = redact_excerpt(
            str(row["source"]),
            issuer_aliases=tuple(row["issuer_aliases"]),
            issuer_tickers=tuple(row["issuer_tickers"]),
        )
        row["guard_match"] = bool(
            META_INSTRUCTION_PATTERN.search(str(row["redacted_excerpt"]))
        )
        row["case_hash"] = canonical_hash(row)
        cases.append(row)
    return cases


class Gemma4Extractor:
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
            hub
            / "models--google--gemma-4-E4B-it"
            / "snapshots"
            / MODEL_REVISION
        )
        if not torch.cuda.is_available():
            raise RuntimeError("EBCT-72 frozen Gemma 4 gate requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "EBCT-72 frozen runtime requires exactly one visible CUDA device"
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


def evaluate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_total = len(SYNTHETIC_CASES)
    names = [str(row["name"]) for row in records]
    if len(records) != expected_total or len(set(names)) != expected_total:
        raise ValueError("synthetic record set is incomplete or duplicated")
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
        outputs = {
            (str(row["actual_label"]), str(row["actual_role"])) for row in rows
        }
        equivalence_checks[group] = len(rows) >= 2 and len(prompts) == len(outputs) == 1
    checks = {
        "all_expected_labels_and_roles": exact == expected_total,
        "all_model_outputs_parse": parsed == len(model_records),
        "all_supported_quotes_validate": quote_valid == len(model_records),
        "all_guarded_cases_skip_model_and_are_unsupported": guard_correct
        == len(guarded_records),
        "all_equivalence_groups_invariant": bool(equivalence_checks)
        and all(equivalence_checks.values()),
    }
    return {
        "counts": {
            "cases": expected_total,
            "model_calls": len(model_records),
            "guarded_cases": len(guarded_records),
            "parsed_model_outputs": parsed,
            "exact_expected_cases": exact,
            "quote_valid_model_outputs": quote_valid,
            "actual_labels": dict(
                sorted(Counter(str(row["actual_label"]) for row in records).items())
            ),
        },
        "equivalence_checks": equivalence_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    preregistration = validate_preregistration()
    model_validation = validate_local_model()
    cases = frozen_cases()
    extractor = Gemma4Extractor(cfg)
    records: list[dict[str, Any]] = []
    for case in cases:
        guarded = bool(case["guarded"])
        guard_correct = guarded and bool(case["guard_match"])
        if guarded:
            parsed = {"label": "UNSUPPORTED", "role": "NONE", "quote": ""}
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
        actual_label = parsed["label"] if isinstance(parsed, dict) else "PARSE_FAILURE"
        actual_role = parsed["role"] if isinstance(parsed, dict) else "PARSE_FAILURE"
        quote = parsed["quote"] if isinstance(parsed, dict) else ""
        quote_valid = bool(
            isinstance(parsed, dict)
            and (
                (actual_label == "UNSUPPORTED" and quote == "")
                or (actual_label != "UNSUPPORTED" and quote in case["redacted_excerpt"])
            )
        )
        records.append(
            {
                "name": case["name"],
                "case_hash": case["case_hash"],
                "redacted_excerpt": case["redacted_excerpt"],
                "expected_label": case["expected_label"],
                "expected_role": case["expected_role"],
                "actual_label": actual_label,
                "actual_role": actual_role,
                "evidence_quote": quote,
                "parsed_ok": isinstance(parsed, dict),
                "quote_valid": quote_valid,
                "expected_match": actual_label == case["expected_label"]
                and actual_role == case["expected_role"],
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
    evaluation = evaluate_records(records)
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
        "runtime": extractor.runtime_metrics(),
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
            "next_step": (
                "build and audit frozen pre-2024 SEC body transport"
                if evaluation["passed"]
                else "retire EBCT-72 without opening SEC bodies or market outcomes"
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
