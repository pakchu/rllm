from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from training import (
    run_psim_d8_rllm2_s2_chunked_source_feature_seal as runner,
)

ATTEMPT = runner.repository_path(runner.DEFAULT_ATTEMPT)
EQUIVALENCE = runner.repository_path(runner.DEFAULT_EQUIVALENCE_RESULT)
FAILURE = runner.repository_path(runner.DEFAULT_OUTPUT)
LOG = (
    runner.REPO_ROOT
    / "results/psim_d8_rllm2_s2_chunked_source_feature_"
    "seal_failure_2026-07-27.log"
)
EXECUTION_COMMIT = "2c4c89ec41675144a67fe1e0254737dfb9953dcf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_self_hash(
    path: Path,
    *,
    field: str,
    expected: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != field
    }
    assert payload[field] == _canonical_hash(core)
    assert payload[field] == expected
    return payload


def test_attempt_gate_and_failure_are_exact_and_canonical() -> None:
    assert _sha256(ATTEMPT) == (
        "d5f69f4bbdfb99d6a6f04e0cf30d5c9a212a80edc50c875a3d4b8dba6076529d"
    )
    assert _sha256(EQUIVALENCE) == (
        "4d145777f91b6d2777412e280f967a9d21c7349dc7993bd18d9b08e42efb0b80"
    )
    assert _sha256(FAILURE) == (
        "85cac32a947fb417686183e0447fdd248bfa54754fe4a1de98fafc1cd80f5613"
    )
    assert _sha256(LOG) == (
        "1e13e833438a360c6acf25ea4408b1d29662576b8a86d839d422831996011631"
    )
    _assert_self_hash(
        ATTEMPT,
        field="attempt_hash",
        expected=(
            "956c542f6cc3fa7190aa88933d9f4e62e4d759eed9484cb38d260cfb7757fa7c"
        ),
    )
    _assert_self_hash(
        EQUIVALENCE,
        field="result_hash",
        expected=(
            "86cb443a75ce46347f5e32f7ecb7dc5da37c9969a7e3e82c1a16c0164b243f13"
        ),
    )
    _assert_self_hash(
        FAILURE,
        field="result_hash",
        expected=(
            "305e53e231edc942adf7f3de73aff20f816cb6eae38dd4530bab263d6cc082a9"
        ),
    )


def test_executed_runner_and_equivalence_failure_are_bound() -> None:
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    gate = json.loads(EQUIVALENCE.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    executed_runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{EXECUTION_COMMIT}:training/"
                "run_psim_d8_rllm2_s2_chunked_source_feature_seal.py"
            ),
        ],
        cwd=runner.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(executed_runner).hexdigest() == (
        attempt["runner_sha256"]
    )
    assert attempt["runner_sha256"] == (
        "fe798475ae32e7c2ab42a1a14a1f85b350cc2f7b928f7d3450fb1b46408b5849"
    )
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert gate["execution_commit"] == EXECUTION_COMMIT
    assert failure["execution_commit"] == EXECUTION_COMMIT
    assert gate["decision"] == "reject"
    assert gate["failure"] == {
        "reason": "frozen equivalence threshold failed",
        "stage": "PRE_MARKET_EQUIVALENCE",
    }
    assert failure["failure"]["stage"] == (
        "PRE_MARKET_EQUIVALENCE_AND_CAPACITY"
    )


def test_multi_chunk_operator_failed_without_capacity_or_extraction() -> None:
    gate = json.loads(EQUIVALENCE.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    cases = gate["case_results"]
    assert len(cases) == 10
    assert sum(case["pass"] for case in cases) == 3
    assert all(
        case["pass"] is True
        for case in cases
        if case["policy_tokens"] <= runner.prereg.CHUNK_SIZE
    )
    multi_chunk = [
        case
        for case in cases
        if case["policy_tokens"] > runner.prereg.CHUNK_SIZE
    ]
    assert len(multi_chunk) == 7
    assert all(case["pass"] is False for case in multi_chunk)
    assert min(
        case["embedding_comparison"]["cosine_similarity"]
        for case in multi_chunk
    ) == 0.9995441801113741
    assert max(
        case["embedding_comparison"]["rms_absolute_delta"]
        for case in multi_chunk
    ) == 0.16501405624525725
    assert max(
        case["embedding_comparison"]["maximum_absolute_delta"]
        for case in multi_chunk
    ) == 1.125
    assert sum(
        case["relation_comparison"]["reference_code"]
        == case["relation_comparison"]["candidate_code"]
        for case in multi_chunk
    ) == 6
    assert gate["capacity_result"] is None
    assert failure["observations"] == {
        "chunk_forwards_started": 102,
        "completed_source_rows": 0,
        "embedding_forwards_started": 10,
        "model_forwards_started": 122,
        "reference_forwards_started": 20,
        "relation_forwards_started": 10,
    }


def test_failure_is_terminal_and_opened_no_market_or_outcomes() -> None:
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    assert failure["decision"] == "reject"
    assert failure["terminal_action"] == runner.FAILURE_ACTION
    assert failure["source_feature_seal_authorized"] is False
    assert failure["open_2020_train_outcomes_authorized"] is False
    assert failure["market_access_authorized"] is False
    assert failure["resume_authorized"] is False
    assert failure["rerun_authorized"] is False
    boundary = failure["access_boundary"]
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["rewards_created"] == 0
    assert boundary["economic_metrics_computed"] == 0
    assert boundary["train_2020_outcomes_opened"] is False
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False
    for path in (
        runner.DEFAULT_CHECKPOINT_DIRECTORY,
        runner.DEFAULT_SOURCE_ROWS,
        runner.DEFAULT_EMBEDDINGS,
        runner.DEFAULT_RELATION_LOGITS,
        runner.DEFAULT_RELATION_ROWS,
    ):
        assert not runner.repository_path(path).exists()
