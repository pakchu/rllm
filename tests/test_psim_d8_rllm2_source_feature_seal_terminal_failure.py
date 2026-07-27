from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from training import run_psim_d8_rllm2_source_feature_seal as runner


ATTEMPT = (
    runner.REPO_ROOT
    / "results/psim_d8_rllm2_source_feature_seal_attempt_2026-07-27.json"
)
FAILURE = (
    runner.REPO_ROOT
    / "results/psim_d8_rllm2_source_feature_seal_2026-07-27.json"
)
LOG = (
    runner.REPO_ROOT
    / "results/psim_d8_rllm2_source_feature_seal_failure_2026-07-27.log"
)
EXECUTION_COMMIT = "ff74a29de88acb04c9807586bd59b17dc4b3fc44"


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
        "d9ac5603db929c489e9a77a70aa2c2115e47d25ce318b9a4c35c7d56d434a4f8"
    )
    assert _sha256(FAILURE) == (
        "6a4a2c2bc783d9aff4f189b530f46678278b406fb7a042c6dd9e3f6cc6161146"
    )
    assert _sha256(LOG) == (
        "b76be448e78f883688091dd6414974168e047239b8d5e155b4db69ac4046342c"
    )
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    attempt_core = {
        key: value
        for key, value in attempt.items()
        if key != "attempt_hash"
    }
    assert attempt["attempt_hash"] == _canonical_hash(attempt_core)
    assert attempt["attempt_hash"] == (
        "cf2f9af9ff4589d18b13a845351eda399fcf4ea34f87fe8912101e9174dcb8f6"
    )
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    failure_core = {
        key: value
        for key, value in failure.items()
        if key != "result_hash"
    }
    assert failure["result_hash"] == _canonical_hash(failure_core)
    assert failure["result_hash"] == (
        "d9311f04a0a44c133993a6eb8c0d023c9b2704d364608e26b8f63196d0d4ca65"
    )


def test_executed_runner_and_oom_stage_are_bound() -> None:
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    executed_runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{EXECUTION_COMMIT}:training/"
                "run_psim_d8_rllm2_source_feature_seal.py"
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
        "521b9e194a63a09d27f16765747938308c6aa7c8c74e8d5871d36264f881aee4"
    )
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert failure["execution_commit"] == EXECUTION_COMMIT
    assert failure["failure"]["stage"] == "SOURCE_FEATURE_FORWARDS"
    assert failure["failure"]["exception_type"] == "OutOfMemoryError"
    assert "Tried to allocate 6.59 GiB" in failure["failure"][
        "exception_message"
    ]
    observations = failure["observations"]
    assert observations == {
        "completed_source_rows": 341,
        "embedding_forwards_started": 342,
        "model_forwards_started": 680,
        "relation_forwards_started": 338,
    }


def test_failure_is_terminal_and_opened_no_market_or_economics() -> None:
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
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False
    for path in (
        runner.DEFAULT_SOURCE_ROWS,
        runner.DEFAULT_EMBEDDINGS,
        runner.DEFAULT_RELATION_LOGITS,
        runner.DEFAULT_RELATION_ROWS,
    ):
        assert not runner.repository_path(path).exists()
