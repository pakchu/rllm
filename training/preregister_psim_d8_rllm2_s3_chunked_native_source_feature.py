#!/usr/bin/env python3
"""Preregister the PSIM-D8-RLLM2-S3 chunked-native source operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import (
    preregister_psim_d8_rllm2_s2_chunked_source_feature as s2,
)

REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = (
    "psim_d8_rllm2_s3_chunked_native_source_feature_preregistration_v1"
)
STAGE_ID = "PSIM-D8-RLLM2-S3"
AS_OF_DATE = "2026-07-27"

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s3_chunked_native_source_feature_"
    "preregistration_2026-07-27.json"
)

S2_PREREGISTRATION = s2.DEFAULT_OUTPUT
S2_PREREGISTRATION_SHA256 = (
    "4b779704368e3f6ba824d8d34045ba0c3899807b91ff46268ab6b86bc1a8c394"
)
S2_PREREGISTRATION_MANIFEST_HASH = (
    "e700698d175489697843fa5269159dd909b371fa2001b970cb20112d82a4fc22"
)
S2_ATTEMPT = s2.ATTEMPT_PATH
S2_ATTEMPT_SHA256 = (
    "d5f69f4bbdfb99d6a6f04e0cf30d5c9a212a80edc50c875a3d4b8dba6076529d"
)
S2_ATTEMPT_HASH = (
    "956c542f6cc3fa7190aa88933d9f4e62e4d759eed9484cb38d260cfb7757fa7c"
)
S2_GATE = s2.EQUIVALENCE_RESULT_PATH
S2_GATE_SHA256 = (
    "4d145777f91b6d2777412e280f967a9d21c7349dc7993bd18d9b08e42efb0b80"
)
S2_GATE_RESULT_HASH = (
    "86cb443a75ce46347f5e32f7ecb7dc5da37c9969a7e3e82c1a16c0164b243f13"
)
S2_FAILURE = s2.RESULT_PATH
S2_FAILURE_SHA256 = (
    "85cac32a947fb417686183e0447fdd248bfa54754fe4a1de98fafc1cd80f5613"
)
S2_FAILURE_RESULT_HASH = (
    "305e53e231edc942adf7f3de73aff20f816cb6eae38dd4530bab263d6cc082a9"
)
S2_FAILURE_LOG = Path(
    "results/psim_d8_rllm2_s2_chunked_source_feature_"
    "seal_failure_2026-07-27.log"
)
S2_FAILURE_LOG_SHA256 = (
    "1e13e833438a360c6acf25ea4408b1d29662576b8a86d839d422831996011631"
)
S2_TERMINAL_REJECTION_DOC = Path(
    "docs/psim-d8-rllm2-s2-chunked-source-feature-"
    "terminal-rejection-2026-07-27.md"
)
S2_TERMINAL_REJECTION_DOC_SHA256 = (
    "987f9aa38f3ddee4e9e90fddfd102f7e1118a3a5815a2536c2dd69f6be26bde3"
)
S2_EXECUTION_COMMIT = "2c4c89ec41675144a67fe1e0254737dfb9953dcf"
S2_TERMINAL_RECORD_COMMIT = (
    "4778c2e9ae135532e5b24979cd036b0ea1f4ad29"
)
S2_RUNNER_SHA256 = (
    "fe798475ae32e7c2ab42a1a14a1f85b350cc2f7b928f7d3450fb1b46408b5849"
)

SOURCE_ROW_ROSTER_HASH = s2.SOURCE_ROW_ROSTER_HASH
CHUNK_SIZE = s2.CHUNK_SIZE
MAXIMUM_INPUT_TOKENS = s2.MAXIMUM_INPUT_TOKENS
CHECKPOINT_SHARD_SIZE = 1
MAXIMUM_PEAK_ALLOCATED_BYTES = s2.MAXIMUM_PEAK_ALLOCATED_BYTES
REPEAT_COUNT = 2
REPEAT_MODEL_LOAD_COUNT = 2
REPEATABILITY_ROSTER = s2.EQUIVALENCE_CASES
CAPACITY_CASE = s2.CAPACITY_CASE

ATTEMPT_PATH = Path(
    "results/psim_d8_rllm2_s3_chunked_native_source_feature_"
    "attempt_2026-07-27.json"
)
REPEATABILITY_RESULT_PATH = Path(
    "results/psim_d8_rllm2_s3_chunked_native_repeatability_"
    "gate_2026-07-27.json"
)
RESULT_PATH = Path(
    "results/psim_d8_rllm2_s3_chunked_native_source_feature_"
    "seal_2026-07-27.json"
)
SOURCE_ROWS_PATH = Path(
    "data/psim_d8_rllm2_s3_chunked_native_source_feature_"
    "rows_2020_2023.jsonl.gz"
)
EMBEDDINGS_PATH = Path(
    "data/psim_d8_rllm2_s3_chunked_native_source_"
    "embeddings_2020_2023.npz"
)
RELATION_LOGITS_PATH = Path(
    "data/psim_d8_rllm2_s3_chunked_native_relation_teacher_"
    "logits_2020_2023.npz"
)
RELATION_ROWS_PATH = Path(
    "data/psim_d8_rllm2_s3_chunked_native_relation_teacher_"
    "rows_2020_2023.jsonl.gz"
)
CHECKPOINT_DIRECTORY = Path(
    "checkpoints/psim_d8_rllm2_s3_chunked_native_"
    "source_feature_2026-07-27"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return s2.canonical_json_bytes(payload, pretty=pretty)


def canonical_hash(payload: Any) -> str:
    return s2.canonical_hash(payload)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_exact_json(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"unsafe RLLM2-S3 authority artifact: {path}")
    raw = target.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"RLLM2-S3 authority hash changed: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004
            f"malformed RLLM2-S3 authority artifact: {path}"
        )
    return payload


def _validate_absent_predecessor_outputs() -> None:
    for predecessor in (s2.s1, s2):
        for path in (
            predecessor.CHECKPOINT_DIRECTORY,
            predecessor.SOURCE_ROWS_PATH,
            predecessor.EMBEDDINGS_PATH,
            predecessor.RELATION_LOGITS_PATH,
            predecessor.RELATION_ROWS_PATH,
        ):
            target = repository_path(path)
            if target.exists() or target.is_symlink():
                raise RuntimeError(
                    "RLLM2 predecessor partial/final output unexpectedly exists"
                )


def validate_predecessor() -> dict[str, Any]:
    preregistration = _read_exact_json(
        S2_PREREGISTRATION,
        expected_sha256=S2_PREREGISTRATION_SHA256,
    )
    attempt = _read_exact_json(
        S2_ATTEMPT,
        expected_sha256=S2_ATTEMPT_SHA256,
    )
    gate = _read_exact_json(
        S2_GATE,
        expected_sha256=S2_GATE_SHA256,
    )
    failure = _read_exact_json(
        S2_FAILURE,
        expected_sha256=S2_FAILURE_SHA256,
    )
    for path, expected in (
        (S2_FAILURE_LOG, S2_FAILURE_LOG_SHA256),
        (
            S2_TERMINAL_REJECTION_DOC,
            S2_TERMINAL_REJECTION_DOC_SHA256,
        ),
    ):
        target = repository_path(path)
        if (
            target.is_symlink()
            or not target.is_file()
            or sha256_bytes(target.read_bytes()) != expected
        ):
            raise RuntimeError("RLLM2-S2 terminal evidence changed")
    for payload, hash_field, expected in (
        (preregistration, "manifest_hash", S2_PREREGISTRATION_MANIFEST_HASH),
        (attempt, "attempt_hash", S2_ATTEMPT_HASH),
        (gate, "result_hash", S2_GATE_RESULT_HASH),
        (failure, "result_hash", S2_FAILURE_RESULT_HASH),
    ):
        core = {
            key: value
            for key, value in payload.items()
            if key != hash_field
        }
        if payload.get(hash_field) != canonical_hash(core):
            raise RuntimeError("RLLM2-S2 terminal self-hash changed")
        if payload.get(hash_field) != expected:
            raise RuntimeError("RLLM2-S2 terminal canonical hash changed")
    cases = gate.get("case_results")
    boundary = failure.get("access_boundary", {})
    if (
        preregistration.get("candidate", {}).get("id") != s2.STAGE_ID
        or attempt.get("execution_commit") != S2_EXECUTION_COMMIT
        or attempt.get("runner_sha256") != S2_RUNNER_SHA256
        or gate.get("decision") != "reject"
        or not isinstance(cases, list)
        or len(cases) != 10
        or sum(case.get("pass") is True for case in cases) != 3
        or gate.get("capacity_result") is not None
        or gate.get("full_source_extraction_authorized") is not False
        or failure.get("decision") != "reject"
        or failure.get("failure", {}).get("stage")
        != "PRE_MARKET_EQUIVALENCE_AND_CAPACITY"
        or failure.get("observations", {}).get("completed_source_rows") != 0
        or failure.get("source_feature_seal_authorized") is not False
        or failure.get("open_2020_train_outcomes_authorized") is not False
        or failure.get("market_access_authorized") is not False
        or failure.get("resume_authorized") is not False
        or failure.get("rerun_authorized") is not False
        or boundary.get("market_or_funding_paths_read") != []
        or boundary.get("market_rows_parsed") != 0
        or boundary.get("funding_rows_parsed") != 0
        or boundary.get("market_or_funding_payload_bytes_hashed") is not False
        or boundary.get("rewards_created") != 0
        or boundary.get("economic_metrics_computed") != 0
        or boundary.get("train_2020_outcomes_opened") is not False
        or boundary.get("test_outcomes_opened") is not False
        or boundary.get("eval_outcomes_opened") is not False
    ):
        raise RuntimeError("RLLM2-S2 terminal failure evidence changed")
    _validate_absent_predecessor_outputs()
    return {
        "preregistration": preregistration,
        "attempt": attempt,
        "gate": gate,
        "failure": failure,
    }


def build_source_rows() -> list[dict[str, Any]]:
    rows = s2.build_source_rows()
    roster = s2.s1.source_roster_contract(rows)
    if roster["source_row_roster_hash"] != SOURCE_ROW_ROSTER_HASH:
        raise RuntimeError("RLLM2-S3 source roster changed")
    return rows


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_predecessor()
    rows = build_source_rows()
    roster = s2.s1.source_roster_contract(rows)
    frozen = s2.validate_frozen_cases(rows)
    inherited = predecessor["preregistration"][
        "unchanged_scientific_contract"
    ]
    model_contract = dict(inherited["model"])
    if model_contract.get("single_forward_per_logical_decision") is not True:
        raise RuntimeError("RLLM2 predecessor forward invariant changed")
    model_contract["single_forward_per_logical_decision"] = False
    model_contract["logical_prompt_forward_schedule"] = (
        "fixed_512_chunked_causal_cache_scan"
    )
    repeatability_cases = [
        dict(case) for case in REPEATABILITY_ROSTER
    ]
    repeatability_case_hash = canonical_hash(
        [canonical_hash(case) for case in repeatability_cases]
    )
    if (
        repeatability_cases != frozen["equivalence_cases"]
        or repeatability_case_hash
        != frozen["equivalence_case_roster_hash"]
    ):
        raise RuntimeError("RLLM2-S3 repeatability roster changed")
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": STAGE_ID,
            "predecessor": s2.STAGE_ID,
            "stage": "source_only_chunked_native_representation_successor",
            "profitability_claim": False,
            "market_access_authorized": False,
        },
        "predecessor_terminal_evidence": {
            "terminal_record_commit": S2_TERMINAL_RECORD_COMMIT,
            "execution_commit": S2_EXECUTION_COMMIT,
            "runner_sha256": S2_RUNNER_SHA256,
            "preregistration": {
                "path": S2_PREREGISTRATION.as_posix(),
                "sha256": S2_PREREGISTRATION_SHA256,
                "manifest_hash": S2_PREREGISTRATION_MANIFEST_HASH,
            },
            "attempt": {
                "path": S2_ATTEMPT.as_posix(),
                "sha256": S2_ATTEMPT_SHA256,
                "attempt_hash": S2_ATTEMPT_HASH,
            },
            "equivalence_gate": {
                "path": S2_GATE.as_posix(),
                "sha256": S2_GATE_SHA256,
                "result_hash": S2_GATE_RESULT_HASH,
                "passed_cases": 3,
                "total_cases": 10,
                "capacity_gate_executed": False,
            },
            "failure": {
                "path": S2_FAILURE.as_posix(),
                "sha256": S2_FAILURE_SHA256,
                "result_hash": S2_FAILURE_RESULT_HASH,
                "log_path": S2_FAILURE_LOG.as_posix(),
                "log_sha256": S2_FAILURE_LOG_SHA256,
                "completed_source_rows": 0,
            },
            "terminal_rejection_document": {
                "path": S2_TERMINAL_REJECTION_DOC.as_posix(),
                "sha256": S2_TERMINAL_REJECTION_DOC_SHA256,
            },
            "s1_or_s2_checkpoint_or_model_output_reuse_authorized": False,
            "source_feature_seal_authorized": False,
            "open_2020_train_outcomes_authorized": False,
            "market_access_authorized": False,
            "market_or_outcome_information_used": False,
        },
        "scientific_contract": {
            "model": model_contract,
            "selector": inherited["selector"],
            "model_visible": inherited["model_visible"],
            "semantic_encoder_gate": inherited["semantic_encoder_gate"],
            "conditional_rllm": inherited["conditional_rllm"],
            "source_row_schema": inherited["source_row_schema"],
            "source_row_roster": roster,
            "source_row_roster_hash": SOURCE_ROW_ROSTER_HASH,
            "policy_prompt_roster_hash": roster[
                "policy_prompt_roster_hash"
            ],
            "relation_teacher_prompt_roster_hash": roster[
                "relation_teacher_prompt_roster_hash"
            ],
            "model_or_quantization_change": False,
            "model_forward_schedule_change": True,
            "source_resample_or_partial_reuse": False,
        },
        "scientific_operator_definition": {
            "representation_family": "chunked_native_not_one_pass_equivalent",
            "one_pass_equivalence_claim": False,
            "tokenization": "exact prompt tokenized once without truncation",
            "chunk_size_tokens": CHUNK_SIZE,
            "chunk_size_basis": "frozen Gemma4 sliding_window",
            "scan": {
                "use_cache": True,
                "past_key_values": "exact prior returned cache",
                "position_ids": "absolute original token positions",
                "attention_mask": "complete prefix through chunk end",
                "mm_token_type_ids": None,
            },
            "embedding": "final chunk final hidden token float32",
            "relation_logits": (
                "exact lm_head plus final_logit_softcapping on final "
                "chunk final hidden token, selected A-F IDs only"
            ),
            "cache_reset_between_logical_prompts": True,
            "cuda_empty_cache_after_each_logical_prompt": True,
            "generation_or_decoded_text": False,
        },
        "pre_market_repeatability_and_capacity_gate": {
            "repeat_count": REPEAT_COUNT,
            "repeat_model_load_count": REPEAT_MODEL_LOAD_COUNT,
            "repeat_execution": (
                "one full roster and capacity pass per independently "
                "loaded model instance; destroy model and clear CUDA "
                "before the second load"
            ),
            "repeatability_cases": repeatability_cases,
            "repeatability_case_roster_hash": repeatability_case_hash,
            "per_prompt_requirements": {
                "exact_token_count": True,
                "exact_token_reconstruction": True,
                "independent_cache_reset_between_repeats": True,
                "independent_model_reload_between_repeats": True,
                "embedding_shape": [s2.s1.EMBEDDING_WIDTH],
                "embedding_all_values_finite": True,
                "embedding_float32_bytes_sha256_identical": True,
                "relation_logits_finite_or_canonical_nan": True,
                "relation_logits_canonical_float32_bytes_sha256_identical": (
                    True
                ),
                "predicted_relation_code_identical": True,
            },
            "capacity_case": dict(CAPACITY_CASE),
            "capacity_case_hash": canonical_hash(CAPACITY_CASE),
            "capacity_repeat_count": REPEAT_COUNT,
            "maximum_peak_allocated_bytes": (
                MAXIMUM_PEAK_ALLOCATED_BYTES
            ),
            "gate_outputs_reused_by_full_extraction": False,
            "full_extraction_uses_fresh_third_model_load": True,
            "failure_action": (
                "TERMINAL_REJECT_S3_WITHOUT_FULL_EXTRACTION_OR_MARKET_OPEN"
            ),
            "pass_action": (
                "AUTHORIZE_S3_FULL_CHUNKED_NATIVE_SOURCE_EXTRACTION_ONLY"
            ),
        },
        "execution_contract": {
            "attempt_path": ATTEMPT_PATH.as_posix(),
            "repeatability_result_path": (
                REPEATABILITY_RESULT_PATH.as_posix()
            ),
            "result_path": RESULT_PATH.as_posix(),
            "checkpoint_directory": CHECKPOINT_DIRECTORY.as_posix(),
            "checkpoint_shard_size": CHECKPOINT_SHARD_SIZE,
            "clean_head_equals_origin_main_required": True,
            "fixed_paths_no_output_override": True,
            "attempt_before_model_weight_load": True,
            "repeatability_and_capacity_before_full_extraction": True,
            "resume_before_gate_pass": False,
            "resume_after_gate_pass": (
                "only contiguous hash-verified completed S3 rows"
            ),
            "inflight_without_committed_row": (
                "terminal_reject_without_repeating_ambiguous_forward"
            ),
            "fresh_s3_outputs_only": True,
            "s1_or_s2_checkpoint_or_model_outputs_read": False,
            "final_artifacts_staged_and_hash_verified": True,
            "authorizing_result_published_last": True,
        },
        "artifact_contract": {
            "source_rows": SOURCE_ROWS_PATH.as_posix(),
            "embeddings": EMBEDDINGS_PATH.as_posix(),
            "relation_logits": RELATION_LOGITS_PATH.as_posix(),
            "relation_rows": RELATION_ROWS_PATH.as_posix(),
            "embedding_shape": [1_461, s2.s1.EMBEDDING_WIDTH],
            "embedding_dtype": s2.s1.EMBEDDING_DTYPE,
            "relation_logits_shape": [
                1_461,
                len(s2.s1.RELATION_CODE_ORDER),
            ],
            "relation_logits_dtype": s2.s1.RELATION_LOGIT_DTYPE,
            "deterministic_npz_and_gzip": True,
            "all_artifact_hashes_bound_in_terminal_result": True,
        },
        "terminal_actions": {
            "failure": (
                "REJECT_PSIM_D8_RLLM2_S3_NO_REPAIR_RERUN_MODEL_SWAP_"
                "OR_MARKET_ACCESS"
            ),
            "success": (
                "ACCEPT_PSIM_D8_RLLM2_S3_CHUNKED_NATIVE_SOURCE_"
                "FEATURE_SEAL_OPEN_2020_TRAIN_OUTCOMES_ONLY"
            ),
        },
        "access_boundary": {
            "source_files_read": sorted(
                {
                    S2_PREREGISTRATION.as_posix(),
                    S2_ATTEMPT.as_posix(),
                    S2_GATE.as_posix(),
                    S2_FAILURE.as_posix(),
                    S2_FAILURE_LOG.as_posix(),
                    S2_TERMINAL_REJECTION_DOC.as_posix(),
                    s2.s1.rllm1.D8_CARDS.as_posix(),
                }
            ),
            "s1_or_s2_checkpoint_or_model_outputs_read": False,
            "market_or_funding_paths_read": [],
            "forbidden_bound_paths": sorted(
                s2.s1.rllm1.FORBIDDEN_BOUND_PATHS
            ),
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "model_loaded": False,
            "model_outputs_created": 0,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "train_2020_outcomes_opened": False,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
        "next_authorized_step": (
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_PSIM_D8_RLLM2_S3_"
            "CHUNKED_NATIVE_REPEATABILITY_CAPACITY_AND_SOURCE_SEAL_RUNNER"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(
    path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = build_preregistration()
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload, pretty=True)
    if target.exists() and target.read_bytes() != encoded:
        raise RuntimeError(f"PSIM-D8-RLLM2-S3 preregistration drift: {target}")
    target.write_bytes(encoded)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"]["id"],
                "manifest_hash": payload["manifest_hash"],
                "repeatability_case_count": len(
                    payload[
                        "pre_market_repeatability_and_capacity_gate"
                    ]["repeatability_cases"]
                ),
                "capacity_case": payload[
                    "pre_market_repeatability_and_capacity_gate"
                ]["capacity_case"],
                "next_authorized_step": payload["next_authorized_step"],
                "access_boundary": payload["access_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
