#!/usr/bin/env python3
"""Preregister the PSIM-D8-RLLM2-S2 chunked source-feature successor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import preregister_psim_d8_rllm2_source_feature_seal as s1


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = (
    "psim_d8_rllm2_s2_chunked_source_feature_preregistration_v1"
)
STAGE_ID = "PSIM-D8-RLLM2-S2"
AS_OF_DATE = "2026-07-27"

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s2_chunked_source_feature_"
    "preregistration_2026-07-27.json"
)

S1_PREREGISTRATION = s1.DEFAULT_OUTPUT
S1_PREREGISTRATION_SHA256 = (
    "de50bdd2615cbe7b5232d31fc2388646a50997a3800af465f507bad82c94691c"
)
S1_PREREGISTRATION_MANIFEST_HASH = (
    "4178275879f98cb3ef1b068c8060aa8ad761e3dcd72acd67940c8150bb98a32a"
)
S1_ATTEMPT = s1.ATTEMPT_PATH
S1_ATTEMPT_SHA256 = (
    "d9ac5603db929c489e9a77a70aa2c2115e47d25ce318b9a4c35c7d56d434a4f8"
)
S1_ATTEMPT_HASH = (
    "cf2f9af9ff4589d18b13a845351eda399fcf4ea34f87fe8912101e9174dcb8f6"
)
S1_FAILURE = s1.RESULT_PATH
S1_FAILURE_SHA256 = (
    "6a4a2c2bc783d9aff4f189b530f46678278b406fb7a042c6dd9e3f6cc6161146"
)
S1_FAILURE_RESULT_HASH = (
    "d9311f04a0a44c133993a6eb8c0d023c9b2704d364608e26b8f63196d0d4ca65"
)
S1_FAILURE_LOG = Path(
    "results/psim_d8_rllm2_source_feature_seal_"
    "failure_2026-07-27.log"
)
S1_FAILURE_LOG_SHA256 = (
    "b76be448e78f883688091dd6414974168e047239b8d5e155b4db69ac4046342c"
)
S1_TERMINAL_REJECTION_DOC = Path(
    "docs/psim-d8-rllm2-source-feature-seal-"
    "terminal-rejection-2026-07-27.md"
)
S1_TERMINAL_REJECTION_DOC_SHA256 = (
    "d0784480679450a44dd4136a76b1d28d86e04679174d00fe10e3c1c933d2d72f"
)
S1_EXECUTION_COMMIT = "ff74a29de88acb04c9807586bd59b17dc4b3fc44"
S1_TERMINAL_RECORD_COMMIT = (
    "0ef9f8644c9937a839924ea92d5908dd1003d1f5"
)
S1_RUNNER_SHA256 = (
    "521b9e194a63a09d27f16765747938308c6aa7c8c74e8d5871d36264f881aee4"
)
SOURCE_ROW_ROSTER_HASH = (
    "033df68d9067a88cb14eb83f92b7638f0addc2372ef08184ef75e9fe3f7ba47c"
)

ATTEMPT_PATH = Path(
    "results/psim_d8_rllm2_s2_chunked_source_feature_"
    "attempt_2026-07-27.json"
)
EQUIVALENCE_RESULT_PATH = Path(
    "results/psim_d8_rllm2_s2_chunked_equivalence_gate_2026-07-27.json"
)
RESULT_PATH = Path(
    "results/psim_d8_rllm2_s2_chunked_source_feature_"
    "seal_2026-07-27.json"
)
SOURCE_ROWS_PATH = Path(
    "data/psim_d8_rllm2_s2_chunked_source_feature_"
    "rows_2020_2023.jsonl.gz"
)
EMBEDDINGS_PATH = Path(
    "data/psim_d8_rllm2_s2_chunked_source_"
    "embeddings_2020_2023.npz"
)
RELATION_LOGITS_PATH = Path(
    "data/psim_d8_rllm2_s2_chunked_relation_teacher_"
    "logits_2020_2023.npz"
)
RELATION_ROWS_PATH = Path(
    "data/psim_d8_rllm2_s2_chunked_relation_teacher_"
    "rows_2020_2023.jsonl.gz"
)
CHECKPOINT_DIRECTORY = Path(
    "checkpoints/psim_d8_rllm2_s2_chunked_source_feature_2026-07-27"
)

CHUNK_SIZE = 512
ONE_PASS_SAFE_MAXIMUM_TOKENS = 8_192
MAXIMUM_INPUT_TOKENS = s1.MAXIMUM_INPUT_TOKENS
CHECKPOINT_SHARD_SIZE = 1
MAXIMUM_PEAK_ALLOCATED_BYTES = 30 * 1024**3
EQUIVALENCE_SELECTION_VERSION = (
    "PSIM_D8_RLLM2_S2_CHUNK_EQUIVALENCE_V1"
)
EQUIVALENCE_RANKS = (
    ("min", 0.00),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("max", 1.00),
)
EQUIVALENCE_BOUNDARIES = (512, 1_024, 2_048, 4_096, 8_192)
EQUIVALENCE_ELIGIBLE_COUNT = 1_286

EQUIVALENCE_CASES: tuple[Mapping[str, Any], ...] = (
    {
        "row_index": 0,
        "row_hash": (
            "a9c8d7dc2d5a18812d6af5b419413b0a3722047cb750e94b16dca3b0fb527b5c"
        ),
        "policy_tokens": 327,
        "relation_tokens": 328,
        "selection_reasons": ["min", "nearest_512"],
    },
    {
        "row_index": 108,
        "row_hash": (
            "206c93d107bf584f9ffe5a04a2ea7710cad5ee808e220564ba0887af7b36ff4b"
        ),
        "policy_tokens": 1_034,
        "relation_tokens": 1_035,
        "selection_reasons": ["p50"],
    },
    {
        "row_index": 151,
        "row_hash": (
            "b12c2d34561576d6f259812d78fc822ee25bfe0f0380f1a8b1f9c1c7ee1f0897"
        ),
        "policy_tokens": 327,
        "relation_tokens": 328,
        "selection_reasons": ["p10"],
    },
    {
        "row_index": 253,
        "row_hash": (
            "bb604f7dccc0f6439b69625b247190f69e7ab858531abd0ce610507775a1b836"
        ),
        "policy_tokens": 4_467,
        "relation_tokens": 4_468,
        "selection_reasons": ["p90"],
    },
    {
        "row_index": 658,
        "row_hash": (
            "31266b8382ffed770d910402a095955e3af3cf26e145869cad01ed416e37dab5"
        ),
        "policy_tokens": 2_047,
        "relation_tokens": 2_048,
        "selection_reasons": ["nearest_2048"],
    },
    {
        "row_index": 672,
        "row_hash": (
            "2eabc9ac533377705124629314cd345b6afbd9c23d507fe12eea46f145e87643"
        ),
        "policy_tokens": 327,
        "relation_tokens": 328,
        "selection_reasons": ["p25"],
    },
    {
        "row_index": 1_144,
        "row_hash": (
            "1367045a54bcba29e9209b343f5c23718d9040f23e7eededdd5b83980686052f"
        ),
        "policy_tokens": 1_028,
        "relation_tokens": 1_029,
        "selection_reasons": ["nearest_1024"],
    },
    {
        "row_index": 1_209,
        "row_hash": (
            "b822519beb9733ef854ab2cf7b12205697d4d2710710621d0a180b2d9ccfedcc"
        ),
        "policy_tokens": 4_093,
        "relation_tokens": 4_094,
        "selection_reasons": ["nearest_4096"],
    },
    {
        "row_index": 1_265,
        "row_hash": (
            "458a7eb0cd5610290f53be872aaed32bab5eccdf64b5c9e65a5941db99e34dae"
        ),
        "policy_tokens": 8_152,
        "relation_tokens": 8_153,
        "selection_reasons": ["max", "nearest_8192"],
    },
    {
        "row_index": 1_329,
        "row_hash": (
            "61be786f68894946583d42ea658a0b1232da45a0701ef80b87230cbb9acf0bde"
        ),
        "policy_tokens": 2_488,
        "relation_tokens": 2_489,
        "selection_reasons": ["p75"],
    },
)

CAPACITY_CASE: Mapping[str, Any] = {
    "row_index": 341,
    "row_hash": (
        "539ff2a7ac56f1559cac390c9c31a3635df79f880e66cf0e3cfb6adfe292a48b"
    ),
    "decision_at": "2020-12-07T12:05:00Z",
    "policy_tokens": 29_727,
    "relation_tokens": 29_728,
    "eligible_relation_unit_count": 57,
    "policy_prompt_utf8_bytes": 122_113,
}


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return s1.canonical_json_bytes(payload, pretty=pretty)


def canonical_hash(payload: Any) -> str:
    return s1.canonical_hash(payload)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_exact_json(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"unsafe RLLM2-S2 authority artifact: {path}")
    raw = target.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"RLLM2-S2 authority hash changed: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"malformed RLLM2-S2 authority artifact: {path}")
    return payload


def validate_predecessor() -> dict[str, Any]:
    preregistration = _read_exact_json(
        S1_PREREGISTRATION,
        expected_sha256=S1_PREREGISTRATION_SHA256,
    )
    attempt = _read_exact_json(
        S1_ATTEMPT,
        expected_sha256=S1_ATTEMPT_SHA256,
    )
    failure = _read_exact_json(
        S1_FAILURE,
        expected_sha256=S1_FAILURE_SHA256,
    )
    log_path = repository_path(S1_FAILURE_LOG)
    if (
        log_path.is_symlink()
        or not log_path.is_file()
        or sha256_bytes(log_path.read_bytes()) != S1_FAILURE_LOG_SHA256
    ):
        raise RuntimeError("RLLM2-S1 failure log changed")
    terminal_doc = repository_path(S1_TERMINAL_REJECTION_DOC)
    if (
        terminal_doc.is_symlink()
        or not terminal_doc.is_file()
        or sha256_bytes(terminal_doc.read_bytes())
        != S1_TERMINAL_REJECTION_DOC_SHA256
    ):
        raise RuntimeError("RLLM2-S1 terminal rejection document changed")
    preregistration_core = {
        key: value
        for key, value in preregistration.items()
        if key != "manifest_hash"
    }
    attempt_core = {
        key: value
        for key, value in attempt.items()
        if key != "attempt_hash"
    }
    failure_core = {
        key: value
        for key, value in failure.items()
        if key != "result_hash"
    }
    observations = failure.get("observations", {})
    access = failure.get("access_boundary", {})
    if (
        preregistration.get("manifest_hash")
        != canonical_hash(preregistration_core)
        or preregistration.get("manifest_hash")
        != S1_PREREGISTRATION_MANIFEST_HASH
        or preregistration.get("source_row_contract", {})
        .get("roster", {})
        .get("source_row_roster_hash")
        != SOURCE_ROW_ROSTER_HASH
    ):
        raise RuntimeError("RLLM2-S1 preregistration changed")
    if (
        attempt.get("attempt_hash") != canonical_hash(attempt_core)
        or attempt.get("attempt_hash") != S1_ATTEMPT_HASH
        or attempt.get("execution_commit") != S1_EXECUTION_COMMIT
        or attempt.get("runner_sha256") != S1_RUNNER_SHA256
    ):
        raise RuntimeError("RLLM2-S1 attempt evidence changed")
    if (
        failure.get("result_hash") != canonical_hash(failure_core)
        or failure.get("result_hash") != S1_FAILURE_RESULT_HASH
        or failure.get("decision") != "reject"
        or failure.get("terminal_action")
        != (
            "REJECT_PSIM_D8_RLLM2_S1_NO_REPAIR_RERUN_MODEL_SWAP_"
            "OR_MARKET_ACCESS"
        )
        or failure.get("failure", {}).get("exception_type")
        != "OutOfMemoryError"
        or failure.get("failure", {}).get("stage")
        != "SOURCE_FEATURE_FORWARDS"
        or observations
        != {
            "completed_source_rows": 341,
            "embedding_forwards_started": 342,
            "model_forwards_started": 680,
            "relation_forwards_started": 338,
        }
        or failure.get("resume_authorized") is not False
        or failure.get("rerun_authorized") is not False
        or failure.get("market_access_authorized") is not False
        or failure.get("open_2020_train_outcomes_authorized") is not False
        or failure.get("source_feature_seal_authorized") is not False
        or access.get("market_or_funding_paths_read") != []
        or access.get("market_rows_parsed") != 0
        or access.get("funding_rows_parsed") != 0
        or access.get("market_or_funding_payload_bytes_hashed") is not False
        or access.get("rewards_created") != 0
        or access.get("economic_metrics_computed") != 0
        or access.get("test_outcomes_opened") is not False
        or access.get("eval_outcomes_opened") is not False
    ):
        raise RuntimeError("RLLM2-S1 terminal failure evidence changed")
    return {
        "preregistration": preregistration,
        "attempt": attempt,
        "failure": failure,
    }


def build_source_rows() -> list[dict[str, Any]]:
    rows = s1.build_source_rows()
    roster = s1.source_roster_contract(rows)
    if roster["source_row_roster_hash"] != SOURCE_ROW_ROSTER_HASH:
        raise RuntimeError("RLLM2-S2 source roster changed")
    return rows


def validate_frozen_cases(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_index = {int(row["row_index"]): row for row in rows}
    if len(by_index) != 1_461:
        raise RuntimeError("RLLM2-S2 source rows are incomplete")
    cases = []
    for expected in EQUIVALENCE_CASES:
        row = by_index[int(expected["row_index"])]
        if (
            row["row_hash"] != expected["row_hash"]
            or not bool(row["relation_teacher_forward_required"])
        ):
            raise RuntimeError("RLLM2-S2 equivalence case changed")
        cases.append(dict(expected))
    capacity_row = by_index[int(CAPACITY_CASE["row_index"])]
    if (
        capacity_row["row_hash"] != CAPACITY_CASE["row_hash"]
        or capacity_row["decision_at"] != CAPACITY_CASE["decision_at"]
        or capacity_row["eligible_relation_unit_count"]
        != CAPACITY_CASE["eligible_relation_unit_count"]
        or len(capacity_row["policy_prompt"].encode("utf-8"))
        != CAPACITY_CASE["policy_prompt_utf8_bytes"]
    ):
        raise RuntimeError("RLLM2-S2 capacity case changed")
    case_hashes = [
        canonical_hash(case)
        for case in cases
    ]
    return {
        "equivalence_case_count": len(cases),
        "equivalence_cases": cases,
        "equivalence_case_roster_hash": canonical_hash(case_hashes),
        "capacity_case": dict(CAPACITY_CASE),
        "capacity_case_hash": canonical_hash(CAPACITY_CASE),
    }


def select_equivalence_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_token_counts: Sequence[int],
    relation_token_counts: Sequence[int],
) -> list[dict[str, Any]]:
    if (
        len(rows) != 1_461
        or len(policy_token_counts) != len(rows)
        or len(relation_token_counts) != len(rows)
    ):
        raise RuntimeError("RLLM2-S2 token roster is incomplete")
    eligible = sorted(
        (
            int(policy_token_counts[index]),
            index,
            row,
        )
        for index, row in enumerate(rows)
        if bool(row["relation_teacher_forward_required"])
        and int(policy_token_counts[index])
        <= ONE_PASS_SAFE_MAXIMUM_TOKENS
        and int(relation_token_counts[index])
        <= ONE_PASS_SAFE_MAXIMUM_TOKENS
    )
    if len(eligible) != EQUIVALENCE_ELIGIBLE_COUNT:
        raise RuntimeError("RLLM2-S2 equivalence eligibility changed")
    reasons: dict[int, list[str]] = {}
    for name, fraction in EQUIVALENCE_RANKS:
        item = eligible[round((len(eligible) - 1) * fraction)]
        reasons.setdefault(item[1], []).append(name)
    for boundary in EQUIVALENCE_BOUNDARIES:
        item = min(
            eligible,
            key=lambda value: (
                abs(value[0] - boundary),
                value[1],
            ),
        )
        reasons.setdefault(item[1], []).append(f"nearest_{boundary}")
    selected = [
        {
            "row_index": index,
            "row_hash": rows[index]["row_hash"],
            "policy_tokens": int(policy_token_counts[index]),
            "relation_tokens": int(relation_token_counts[index]),
            "selection_reasons": selected_reasons,
        }
        for index, selected_reasons in sorted(reasons.items())
    ]
    if selected != [dict(case) for case in EQUIVALENCE_CASES]:
        raise RuntimeError("RLLM2-S2 equivalence case selection changed")
    return selected


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_predecessor()
    rows = build_source_rows()
    roster = s1.source_roster_contract(rows)
    challenge = validate_frozen_cases(rows)
    inherited = predecessor["preregistration"]
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": STAGE_ID,
            "predecessor": s1.STAGE_ID,
            "stage": (
                "source_only_fixed_chunk_causal_cache_operational_successor"
            ),
            "profitability_claim": False,
            "market_access_authorized": False,
        },
        "predecessor_terminal_evidence": {
            "terminal_record_commit": S1_TERMINAL_RECORD_COMMIT,
            "execution_commit": S1_EXECUTION_COMMIT,
            "runner_sha256": S1_RUNNER_SHA256,
            "preregistration": {
                "path": S1_PREREGISTRATION.as_posix(),
                "sha256": S1_PREREGISTRATION_SHA256,
                "manifest_hash": S1_PREREGISTRATION_MANIFEST_HASH,
            },
            "attempt": {
                "path": S1_ATTEMPT.as_posix(),
                "sha256": S1_ATTEMPT_SHA256,
                "attempt_hash": S1_ATTEMPT_HASH,
            },
            "failure": {
                "path": S1_FAILURE.as_posix(),
                "sha256": S1_FAILURE_SHA256,
                "result_hash": S1_FAILURE_RESULT_HASH,
                "log_path": S1_FAILURE_LOG.as_posix(),
                "log_sha256": S1_FAILURE_LOG_SHA256,
                "stage": "SOURCE_FEATURE_FORWARDS",
                "exception_type": "OutOfMemoryError",
            },
            "terminal_rejection_document": {
                "path": S1_TERMINAL_REJECTION_DOC.as_posix(),
                "sha256": S1_TERMINAL_REJECTION_DOC_SHA256,
                "fresh_successor_authorized_only_under_new_preregistration": (
                    True
                ),
            },
            "completed_source_rows": 341,
            "source_feature_seal_authorized": False,
            "open_2020_train_outcomes_authorized": False,
            "market_access_authorized": False,
            "s1_resume_or_rerun_authorized": False,
            "s1_checkpoint_or_model_output_reuse_authorized": False,
            "market_or_outcome_information_used": False,
        },
        "unchanged_scientific_contract": {
            "model": inherited["inherited_contract"]["model"],
            "selector": inherited["inherited_contract"]["selector"],
            "model_visible": inherited["inherited_contract"][
                "model_visible"
            ],
            "semantic_encoder_gate": inherited["inherited_contract"][
                "semantic_encoder_gate"
            ],
            "conditional_rllm": inherited["inherited_contract"][
                "conditional_rllm"
            ],
            "source_row_schema": inherited["source_row_contract"][
                "schema_version"
            ],
            "source_row_roster": roster,
            "source_row_roster_hash": SOURCE_ROW_ROSTER_HASH,
            "policy_prompt_roster_hash": roster[
                "policy_prompt_roster_hash"
            ],
            "relation_teacher_prompt_roster_hash": roster[
                "relation_teacher_prompt_roster_hash"
            ],
            "relation_mapping_or_prompt_change": False,
            "model_or_quantization_change": False,
            "source_resample_or_partial_reuse": False,
        },
        "sole_operational_delta": {
            "s1_operator": (
                "single full-prompt SDPA forward with use_cache=false"
            ),
            "s2_operator": (
                "tokenize the exact prompt once and scan contiguous fixed "
                "token chunks with use_cache=true and returned "
                "past_key_values"
            ),
            "chunk_size_tokens": CHUNK_SIZE,
            "chunk_size_basis": (
                "exact frozen Gemma4 text sliding_window value"
            ),
            "attention_implementation": "sdpa",
            "first_chunk": {
                "past_key_values": None,
                "use_cache": True,
                "position_ids": "range_0_to_chunk_end_exclusive",
                "attention_mask": "full_prefix_through_chunk_end",
                "mm_token_type_ids": None,
            },
            "later_chunks": {
                "past_key_values": "exact_prior_returned_cache",
                "use_cache": True,
                "position_ids": (
                    "range_chunk_start_to_chunk_end_exclusive"
                ),
                "attention_mask": "full_prefix_through_chunk_end",
                "mm_token_type_ids": None,
            },
            "embedding": (
                "last token of final chunk model.model last_hidden_state"
            ),
            "relation_logits": (
                "exact lm_head and frozen final_logit_softcapping applied "
                "only to the final chunk last hidden token"
            ),
            "cache_reset_between_logical_prompts": True,
            "cuda_empty_cache_after_each_logical_prompt": True,
            "generation_or_decoded_text": False,
            "dynamic_chunk_selection_or_tuning": False,
        },
        "pre_market_equivalence_gate": {
            "selection_version": EQUIVALENCE_SELECTION_VERSION,
            "one_pass_safe_maximum_tokens": (
                ONE_PASS_SAFE_MAXIMUM_TOKENS
            ),
            "selection": {
                "eligible_rows": (
                    "relation_teacher_forward_required=true and both exact "
                    "prompt token counts <= 8192"
                ),
                "sort": "policy_tokens_then_row_index",
                "ranks": [
                    {"name": name, "fraction": fraction}
                    for name, fraction in EQUIVALENCE_RANKS
                ],
                "rank_index": "round((eligible_count-1)*fraction)",
                "eligible_count": EQUIVALENCE_ELIGIBLE_COUNT,
                "nearest_boundaries": list(EQUIVALENCE_BOUNDARIES),
                "nearest_tie": "lower_row_index",
                "dedupe": "row_index",
            },
            **challenge,
            "prompt_kinds_per_case": ["policy", "relation_teacher"],
            "reference_operator": "s1_one_pass_use_cache_false",
            "candidate_operator": "s2_fixed_512_causal_cache",
            "embedding_thresholds": {
                "minimum_cosine_similarity": 0.99999,
                "maximum_rms_absolute_delta": 0.01,
                "maximum_absolute_delta": 0.05,
                "all_values_finite": True,
            },
            "relation_thresholds": {
                "predicted_code_identical": True,
                "maximum_mean_absolute_delta": 0.01,
                "maximum_absolute_delta": 0.03,
                "finite_or_same_canonical_nan_semantics": True,
            },
            "token_reconstruction_exact": True,
            "failure_action": (
                "TERMINAL_REJECT_S2_WITHOUT_FULL_EXTRACTION_OR_MARKET_OPEN"
            ),
            "pass_action": (
                "AUTHORIZE_S2_FULL_SOURCE_EXTRACTION_ONLY"
            ),
        },
        "long_context_capacity_gate": {
            "case": dict(CAPACITY_CASE),
            "one_pass_reference_forbidden": True,
            "chunked_policy_embedding_finite": True,
            "chunked_relation_logits_finite_or_canonical_nan": True,
            "maximum_peak_allocated_bytes": (
                MAXIMUM_PEAK_ALLOCATED_BYTES
            ),
            "cpu_disk_or_meta_offload": False,
            "outputs_reused_by_full_extraction": False,
            "must_pass_before_full_extraction": True,
        },
        "execution_contract": {
            "attempt_path": ATTEMPT_PATH.as_posix(),
            "equivalence_result_path": (
                EQUIVALENCE_RESULT_PATH.as_posix()
            ),
            "result_path": RESULT_PATH.as_posix(),
            "checkpoint_directory": CHECKPOINT_DIRECTORY.as_posix(),
            "checkpoint_shard_size": CHECKPOINT_SHARD_SIZE,
            "fresh_s2_outputs_only": True,
            "s1_checkpoint_directory_read": False,
            "clean_head_equals_origin_main_required": True,
            "fixed_paths_no_output_override": True,
            "attempt_before_model_weight_load": True,
            "equivalence_and_capacity_before_full_extraction": True,
            "resume_before_equivalence_pass": False,
            "resume_after_equivalence_pass": (
                "only contiguous hash-verified completed S2 rows"
            ),
            "inflight_without_committed_row": (
                "terminal_reject_without_repeating_ambiguous_forward"
            ),
            "final_artifacts_staged_and_hash_verified": True,
            "authorizing_result_published_last": True,
        },
        "artifact_contract": {
            "source_rows": SOURCE_ROWS_PATH.as_posix(),
            "embeddings": EMBEDDINGS_PATH.as_posix(),
            "relation_logits": RELATION_LOGITS_PATH.as_posix(),
            "relation_rows": RELATION_ROWS_PATH.as_posix(),
            "embedding_shape": [1_461, s1.EMBEDDING_WIDTH],
            "embedding_dtype": s1.EMBEDDING_DTYPE,
            "relation_logits_shape": [
                1_461,
                len(s1.RELATION_CODE_ORDER),
            ],
            "relation_logits_dtype": s1.RELATION_LOGIT_DTYPE,
            "deterministic_npz_and_gzip": True,
            "all_artifact_hashes_bound_in_terminal_result": True,
        },
        "terminal_actions": {
            "failure": (
                "REJECT_PSIM_D8_RLLM2_S2_NO_REPAIR_RERUN_MODEL_SWAP_"
                "OR_MARKET_ACCESS"
            ),
            "success": (
                "ACCEPT_PSIM_D8_RLLM2_S2_CHUNKED_SOURCE_FEATURE_SEAL_"
                "OPEN_2020_TRAIN_OUTCOMES_ONLY"
            ),
        },
        "access_boundary": {
            "source_files_read": sorted(
                {
                    S1_PREREGISTRATION.as_posix(),
                    S1_ATTEMPT.as_posix(),
                    S1_FAILURE.as_posix(),
                    S1_FAILURE_LOG.as_posix(),
                    S1_TERMINAL_REJECTION_DOC.as_posix(),
                    s1.rllm1.D8_CARDS.as_posix(),
                }
            ),
            "s1_checkpoint_or_partial_model_outputs_read": False,
            "market_or_funding_paths_read": [],
            "forbidden_bound_paths": sorted(s1.rllm1.FORBIDDEN_BOUND_PATHS),
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
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_PSIM_D8_RLLM2_S2_"
            "CHUNKED_EQUIVALENCE_CAPACITY_AND_SOURCE_SEAL_RUNNER"
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
        raise RuntimeError(f"PSIM-D8-RLLM2-S2 preregistration drift: {target}")
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
                "equivalence_case_count": payload[
                    "pre_market_equivalence_gate"
                ]["equivalence_case_count"],
                "capacity_case": payload["long_context_capacity_gate"][
                    "case"
                ],
                "next_authorized_step": payload["next_authorized_step"],
                "access_boundary": payload["access_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
