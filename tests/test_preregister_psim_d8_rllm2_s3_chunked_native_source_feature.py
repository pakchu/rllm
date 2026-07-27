from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    preregister_psim_d8_rllm2_s3_chunked_native_source_feature as prereg,
)


def test_preregistration_is_exact_and_deterministic() -> None:
    path = prereg.repository_path(prereg.DEFAULT_OUTPUT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == prereg.build_preregistration()
    assert payload["manifest_hash"] == (
        "263f0476ce887cad7b5e9c0174e02b6f714a8ce54029ef5788a6179ad1c396f3"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "7edb7eeef115e0579099f9aa990648f1db1cde28053867c0ae1edb6c60deb196"
    )


def test_predecessor_is_terminal_and_no_outputs_are_reused() -> None:
    predecessor = prereg.validate_predecessor()
    assert predecessor["gate"]["decision"] == "reject"
    assert predecessor["gate"]["capacity_result"] is None
    assert predecessor["failure"]["decision"] == "reject"
    assert predecessor["failure"]["resume_authorized"] is False
    assert predecessor["failure"]["rerun_authorized"] is False
    payload = prereg.build_preregistration()
    evidence = payload["predecessor_terminal_evidence"]
    assert (
        evidence[
            "s1_or_s2_checkpoint_or_model_output_reuse_authorized"
        ]
        is False
    )
    assert evidence["open_2020_train_outcomes_authorized"] is False
    assert evidence["market_access_authorized"] is False
    assert payload["access_boundary"][
        "s1_or_s2_checkpoint_or_model_outputs_read"
    ] is False


def test_chunked_native_operator_drops_one_pass_equivalence_claim() -> None:
    payload = prereg.build_preregistration()
    operator = payload["scientific_operator_definition"]
    assert operator["representation_family"] == (
        "chunked_native_not_one_pass_equivalent"
    )
    assert operator["one_pass_equivalence_claim"] is False
    assert operator["chunk_size_tokens"] == 512
    assert operator["generation_or_decoded_text"] is False
    model = payload["scientific_contract"]["model"]
    assert model["single_forward_per_logical_decision"] is False
    assert model["logical_prompt_forward_schedule"] == (
        "fixed_512_chunked_causal_cache_scan"
    )
    gate = payload["pre_market_repeatability_and_capacity_gate"]
    assert gate["repeat_count"] == 2
    assert gate["repeat_model_load_count"] == 2
    assert gate["capacity_repeat_count"] == 2
    assert gate["gate_outputs_reused_by_full_extraction"] is False
    assert gate["full_extraction_uses_fresh_third_model_load"] is True
    requirements = gate["per_prompt_requirements"]
    assert requirements[
        "embedding_float32_bytes_sha256_identical"
    ] is True
    assert requirements[
        "relation_logits_canonical_float32_bytes_sha256_identical"
    ] is True
    assert requirements[
        "independent_model_reload_between_repeats"
    ] is True


def test_repeatability_roster_and_capacity_case_are_frozen() -> None:
    payload = prereg.build_preregistration()
    gate = payload["pre_market_repeatability_and_capacity_gate"]
    assert gate["repeatability_cases"] == [
        dict(case) for case in prereg.s2.EQUIVALENCE_CASES
    ]
    assert gate["repeatability_case_roster_hash"] == (
        "d960f57da0d5e8b37004643a20b39caa5a569278bd0385238e2845ce41585bcf"
    )
    assert gate["capacity_case"] == dict(prereg.s2.CAPACITY_CASE)
    assert gate["capacity_case_hash"] == (
        "5875ce6259a15bfc226d6b12ab50c7d6da7d4866011a7b87222a4e5e491c0ab0"
    )
    rows = prereg.build_source_rows()
    assert len(rows) == 1461
    assert prereg.s2.s1.source_roster_contract(rows)[
        "source_row_roster_hash"
    ] == prereg.SOURCE_ROW_ROSTER_HASH


def test_s3_paths_are_fresh_and_market_boundary_is_closed() -> None:
    payload = prereg.build_preregistration()
    execution = payload["execution_contract"]
    assert execution["fresh_s3_outputs_only"] is True
    assert execution[
        "s1_or_s2_checkpoint_or_model_outputs_read"
    ] is False
    s3_paths = {
        prereg.ATTEMPT_PATH,
        prereg.REPEATABILITY_RESULT_PATH,
        prereg.RESULT_PATH,
        prereg.SOURCE_ROWS_PATH,
        prereg.EMBEDDINGS_PATH,
        prereg.RELATION_LOGITS_PATH,
        prereg.RELATION_ROWS_PATH,
        prereg.CHECKPOINT_DIRECTORY,
    }
    s2_paths = {
        prereg.s2.ATTEMPT_PATH,
        prereg.s2.EQUIVALENCE_RESULT_PATH,
        prereg.s2.RESULT_PATH,
        prereg.s2.SOURCE_ROWS_PATH,
        prereg.s2.EMBEDDINGS_PATH,
        prereg.s2.RELATION_LOGITS_PATH,
        prereg.s2.RELATION_ROWS_PATH,
        prereg.s2.CHECKPOINT_DIRECTORY,
    }
    s1_paths = {
        prereg.s2.s1.SOURCE_ROWS_PATH,
        prereg.s2.s1.EMBEDDINGS_PATH,
        prereg.s2.s1.RELATION_LOGITS_PATH,
        prereg.s2.s1.RELATION_ROWS_PATH,
        prereg.s2.s1.CHECKPOINT_DIRECTORY,
    }
    assert s3_paths.isdisjoint(s2_paths)
    assert s3_paths.isdisjoint(s1_paths)
    boundary = payload["access_boundary"]
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["train_2020_outcomes_opened"] is False
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_outputs_created"] == 0
    for path in s3_paths:
        assert not prereg.repository_path(path).exists()
    s2_output_paths = {
        prereg.s2.SOURCE_ROWS_PATH,
        prereg.s2.EMBEDDINGS_PATH,
        prereg.s2.RELATION_LOGITS_PATH,
        prereg.s2.RELATION_ROWS_PATH,
        prereg.s2.CHECKPOINT_DIRECTORY,
    }
    for path in s1_paths | s2_output_paths:
        assert not prereg.repository_path(path).exists()


def test_result_file_records_no_market_payload_access() -> None:
    path = Path(prereg.repository_path(prereg.DEFAULT_OUTPUT))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidate"]["profitability_claim"] is False
    boundary = payload["access_boundary"]
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["rewards_created"] == 0
    assert boundary["economic_metrics_computed"] == 0
