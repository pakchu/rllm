from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = REPO_ROOT / (
    "results/psim_d8_rllm2_s6_train2020_"
    "action_mean_residual_ridge_fqi_attempt_2026-07-27.json"
)
FAILURE = REPO_ROOT / (
    "results/psim_d8_rllm2_s6_runner_implementation_failure_2026-07-27.json"
)
EXECUTION_COMMIT = "31cd9ba330a7f3c53b7a5a642d365e729d1e7cca"
RUNNER_PATH = (
    "training/run_psim_d8_rllm2_s6_train2020_"
    "action_mean_residual_ridge_fqi.py"
)
RUNNER_SHA256 = (
    "0a50b7abbdd1dea454f080f05afb36320235ce53afcbfda07e721cc02b35dadd"
)
ATTEMPT_SHA256 = (
    "23328b8c3ec233356700dea4618f66c1765d81b1ee5a3136aa9dfc5f9a54157e"
)
ATTEMPT_HASH = (
    "0b686a89dc796800422b218888fd904a24ddbfa6c7ca2e350662621085e7c45d"
)
FAILURE_SHA256 = (
    "b0c45438d29126cfac65b7d7d8ed9318d595b218ffd2062bcf64d13c521237ad"
)
FAILURE_HASH = (
    "9c95f0d80c1b44e4116edc43d9973b4cb240b9b3a1670226eca71043b47a3249"
)


def _canonical_hash(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt_failure_and_executed_runner_are_immutable() -> None:
    assert _sha256(ATTEMPT) == ATTEMPT_SHA256
    assert _sha256(FAILURE) == FAILURE_SHA256
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    attempt_core = {
        key: value
        for key, value in attempt.items()
        if key != "attempt_hash"
    }
    failure_core = {
        key: value
        for key, value in failure.items()
        if key != "failure_hash"
    }
    assert attempt["attempt_hash"] == _canonical_hash(attempt_core)
    assert attempt["attempt_hash"] == ATTEMPT_HASH
    assert failure["failure_hash"] == _canonical_hash(failure_core)
    assert failure["failure_hash"] == FAILURE_HASH
    executed_runner = subprocess.run(
        ["git", "show", f"{EXECUTION_COMMIT}:{RUNNER_PATH}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(executed_runner).hexdigest() == RUNNER_SHA256
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert attempt["runner_sha256"] == RUNNER_SHA256
    assert failure["execution_commit"] == EXECUTION_COMMIT
    assert failure["runner"]["sha256"] == RUNNER_SHA256


def test_failure_opens_no_2021_outcome_and_authorizes_only_code_repair() -> None:
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))

    assert failure["decision"] == (
        "implementation_error_before_schedule_construction"
    )
    assert failure["failure"]["exception_type"] == "AttributeError"
    assert failure["failure"]["failed_expression"] == (
        "residual.reconstruct_reward_tensor"
    )
    boundary = failure["access_boundary_after_failure"]
    assert boundary["2020_transition_ledger_rows_parsed"] == 3_288
    assert boundary["2020_original_reward_tensor_reconstructed"] is False
    assert boundary["2020_residual_reward_values_created"] == 0
    assert boundary["2020_economic_metrics_computed"] == 0
    assert boundary["raw_market_or_funding_paths_read"] == []
    assert boundary["2021_market_or_funding_paths_read"] == []
    assert boundary["2021_reward_rows_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0
    assert boundary["2021_policy_specific_outcomes_opened"] is False
    assert boundary["2022_or_later_outcomes_opened"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_forwards_started"] == 0
    assert failure["output_state"] == {
        "attempt_written": True,
        "base_schedules_written": False,
        "delayed_schedule_written": False,
        "residual_ledger_written": False,
        "result_written": False,
        "schedule_manifest_written": False,
    }
    repair = failure["repair_contract"]
    assert repair["delete_or_overwrite_failed_attempt"] is False
    assert repair["reuse_original_result_or_artifact_paths"] is False
    assert repair["same_scientific_hypothesis_required"] is True
    assert repair["new_preregistration_and_distinct_write_once_paths_required"]
    assert repair["2021_outcome_access_before_repaired_schedule_gate"] is False
