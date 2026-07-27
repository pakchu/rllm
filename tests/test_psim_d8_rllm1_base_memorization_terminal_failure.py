from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from training import run_psim_d8_rllm1_base_memorization_gate as gate


ATTEMPT = (
    gate.REPO_ROOT
    / "results/psim_d8_rllm1_base_memorization_gate_attempt_2026-07-27.json"
)
FAILURE = (
    gate.REPO_ROOT
    / "results/psim_d8_rllm1_base_memorization_gate_failure_2026-07-27.json"
)
LOG = (
    gate.REPO_ROOT
    / "results/psim_d8_rllm1_base_memorization_gate_failure_2026-07-27.log"
)
EXECUTION_COMMIT = "ce9ba77782ff0cc34411d60dc1ba7def5bea707f"


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


def test_attempt_and_failure_evidence_are_exact_and_canonical() -> None:
    assert _sha256(ATTEMPT) == (
        "a325fb09286cf921e5b9e1d65e4655a03bde11058aa09a6fe0cd5d1fc79c3179"
    )
    assert _sha256(LOG) == (
        "2a3ef2ce55b8b668d41e5e7097168a1a619986e39c23e2087c59c2ef8fdc71ae"
    )
    assert _sha256(FAILURE) == (
        "02728096681f058144c12090cfa5876a973fb1cbd5146e35d59e2aa260dca812"
    )
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    attempt_core = {
        key: value
        for key, value in attempt.items()
        if key != "attempt_hash"
    }
    assert attempt["attempt_hash"] == _canonical_hash(attempt_core)
    assert attempt["attempt_hash"] == (
        "db2e1d7c5ce0bc7dbb061ff6f3e1d4a674d018db16dbf04a7509e04566d3a609"
    )
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    failure_core = {
        key: value
        for key, value in failure.items()
        if key != "result_hash"
    }
    assert failure["result_hash"] == _canonical_hash(failure_core)
    assert failure["result_hash"] == (
        "b0a40fa9904dd9b7877b3b64c9f382999d0b24a75a4edbcc687ccfe8b424fe69"
    )


def test_executed_runner_and_failure_stage_are_bound() -> None:
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{EXECUTION_COMMIT}:training/"
                "run_psim_d8_rllm1_base_memorization_gate.py"
            ),
        ],
        cwd=gate.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(runner).hexdigest() == attempt["runner_sha256"]
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert failure["execution_commit"] == EXECUTION_COMMIT
    assert failure["failure"]["stage"] == (
        "POST_WEIGHT_LOAD_PRE_FIRST_FORWARD_RUNTIME_ASSERTION"
    )
    assert failure["failure"]["exception_message"] == (
        "frozen model device map changed: {}"
    )
    observations = failure["observations"]
    assert observations["weight_tensors_loaded"] == 2_076
    assert observations["weight_tensors_expected"] == 2_076
    assert observations["model_weights_loaded"] is True
    assert observations["scorer_construction_completed"] is False
    assert observations["model_forwards_started"] == 0
    assert observations["challenge_predictions_created"] == 0
    assert observations["challenge_statistics_computed"] is False
    assert observations["official_result_artifact_created"] is False
    assert not (gate.REPO_ROOT / gate.DEFAULT_OUTPUT).exists()


def test_failure_is_terminal_and_opened_no_market_or_economics() -> None:
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    assert failure["decision"] == "reject"
    assert failure["terminal_action"] == gate.MEMORIZATION_FAILURE_ACTION
    assert failure["rerun_authorized"] is False
    boundary = failure["access_boundary"]
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["rewards_created"] == 0
    assert boundary["economic_metrics_computed"] == 0
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False
