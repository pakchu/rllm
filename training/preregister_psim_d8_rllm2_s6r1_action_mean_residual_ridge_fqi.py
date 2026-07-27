#!/usr/bin/env python3
"""Preregister the code-only repair of the PSIM S6 runner.

The original S6 attempt stopped on a symbol-resolution error after writing its
immutable attempt and before reconstructing rewards or constructing schedules.
S6R1 changes no scientific contract.  It binds a new runner and distinct
write-once paths, while preserving the exact S6 source features, reward
transform, FQI parameters, controls, and outcome-blind readiness gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import (
    preregister_psim_d8_rllm2_s6_action_mean_residual_ridge_fqi as s6,
)

REPO_ROOT = _BOOTSTRAP_REPO_ROOT
s4 = s6.s4
s5 = s6.s5

PROTOCOL_VERSION = (
    "psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi_"
    "code_repair_preregistration_v1"
)
STAGE_ID = "PSIM-D8-RLLM2-S6R1-ACTION-MEAN-RESIDUAL-RIDGE-FQI"
AS_OF_DATE = "2026-07-27"

PRIMARY_POLICY_ID = s6.PRIMARY_POLICY_ID
CONTROL_POLICY_IDS = s6.CONTROL_POLICY_IDS
POLICY_FAMILY_IDS = s6.POLICY_FAMILY_IDS
COMBINED_2021_FAMILY_IDS = s6.COMBINED_2021_FAMILY_IDS
DEGENERATE_CONTROL_IDS = s6.DEGENERATE_CONTROL_IDS

RIDGE_ALPHA = s6.RIDGE_ALPHA
BELLMAN_ITERATIONS = s6.BELLMAN_ITERATIONS
DISCOUNT = s6.DISCOUNT
FIXED_SHUFFLE_SEED = s6.FIXED_SHUFFLE_SEED
FIXED_CIRCULAR_REWARD_SHIFT = s6.FIXED_CIRCULAR_REWARD_SHIFT
MIN_NONFLAT_TARGET_ROWS = s6.MIN_NONFLAT_TARGET_ROWS
MIN_DIRECTION_SHARE = s6.MIN_DIRECTION_SHARE
FAMILYWISE_P_MAX = s6.FAMILYWISE_P_MAX
S5_BASE_SCHEDULES_SHA256 = s6.S5_BASE_SCHEDULES_SHA256

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi_"
    "preregistration_2026-07-27.json"
)
RUNNER_PATH = Path(
    "training/run_psim_d8_rllm2_s6r1_train2020_"
    "action_mean_residual_ridge_fqi.py"
)
ATTEMPT_PATH = Path(
    "results/psim_d8_rllm2_s6r1_train2020_"
    "action_mean_residual_ridge_fqi_attempt_2026-07-27.json"
)
RESULT_PATH = Path(
    "results/psim_d8_rllm2_s6r1_train2020_action_mean_residual_ridge_fqi_"
    "pre2021_gate_2026-07-27.json"
)
RESIDUAL_LEDGER_PATH = Path(
    "data/psim_d8_rllm2_s6r1_action_mean_residual_reward_ledger_2020.csv.gz"
)
SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi_"
    "schedules_2021.csv.gz"
)
DELAYED_SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi_"
    "delayed_schedule_2021.csv.gz"
)
SCHEDULE_MANIFEST_PATH = Path(
    "results/psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi_"
    "schedule_gate_2021_2026-07-27.json"
)

S6_PREREGISTRATION_SHA256 = (
    "3f80a51a001d422e4aea43072b1be157b5a311f14c7f0feb50fe9b36f413f219"
)
S6_PREREGISTRATION_MANIFEST_HASH = (
    "e35975dc79e6dd0dd694f0998cb1fee7100c8e0d8d98585c74cf025fb501abdb"
)
S6_EXECUTION_COMMIT = "31cd9ba330a7f3c53b7a5a642d365e729d1e7cca"
S6_RUNNER_SHA256 = (
    "0a50b7abbdd1dea454f080f05afb36320235ce53afcbfda07e721cc02b35dadd"
)
S6_CORE_SHA256 = (
    "c9b0bfe33ddbacaf3f8fbf9fd520ed3423e457341a45ce87aaaa4b39b2656927"
)
S6_ATTEMPT_SHA256 = (
    "23328b8c3ec233356700dea4618f66c1765d81b1ee5a3136aa9dfc5f9a54157e"
)
S6_ATTEMPT_HASH = (
    "0b686a89dc796800422b218888fd904a24ddbfa6c7ca2e350662621085e7c45d"
)
S6_FAILURE_PATH = Path(
    "results/psim_d8_rllm2_s6_runner_implementation_failure_2026-07-27.json"
)
S6_FAILURE_SHA256 = (
    "b0c45438d29126cfac65b7d7d8ed9318d595b218ffd2062bcf64d13c521237ad"
)
S6_FAILURE_HASH = (
    "9c95f0d80c1b44e4116edc43d9973b4cb240b9b3a1670226eca71043b47a3249"
)
S6_FAILURE_DOCUMENT = Path(
    "docs/psim-d8-rllm2-s6-runner-implementation-failure-2026-07-27.md"
)
S6_FAILURE_DOCUMENT_SHA256 = (
    "261c445766fc5781e02b7dd886c7511100cc0df2e658ea81044f86cb08dcea80"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + (b"\n" if pretty else b"")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact_json(
    path: str | Path,
    *,
    expected_sha256: str,
    self_hash_field: str,
    expected_self_hash: str,
) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM S6R1 evidence missing: {path}")
    if sha256_file(target) != expected_sha256:
        raise RuntimeError(f"PSIM S6R1 evidence changed: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != self_hash_field
    }
    if (
        payload.get(self_hash_field) != expected_self_hash
        or payload.get(self_hash_field) != canonical_hash(core)
    ):
        raise RuntimeError(f"PSIM S6R1 evidence self hash changed: {path}")
    return payload


def validate_s6_implementation_failure() -> dict[str, Any]:
    registration = _read_exact_json(
        s6.DEFAULT_OUTPUT,
        expected_sha256=S6_PREREGISTRATION_SHA256,
        self_hash_field="manifest_hash",
        expected_self_hash=S6_PREREGISTRATION_MANIFEST_HASH,
    )
    attempt = _read_exact_json(
        s6.ATTEMPT_PATH,
        expected_sha256=S6_ATTEMPT_SHA256,
        self_hash_field="attempt_hash",
        expected_self_hash=S6_ATTEMPT_HASH,
    )
    failure = _read_exact_json(
        S6_FAILURE_PATH,
        expected_sha256=S6_FAILURE_SHA256,
        self_hash_field="failure_hash",
        expected_self_hash=S6_FAILURE_HASH,
    )
    for path, expected in (
        (s6.RUNNER_PATH, S6_RUNNER_SHA256),
        (Path("training/psim_action_mean_residual_fqi.py"), S6_CORE_SHA256),
        (S6_FAILURE_DOCUMENT, S6_FAILURE_DOCUMENT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"PSIM S6R1 predecessor file changed: {path}")
    boundary = failure.get("access_boundary_after_failure", {})
    outputs = failure.get("output_state", {})
    repair = failure.get("repair_contract", {})
    if (
        registration != s6.build_preregistration()
        or registration.get("candidate", {}).get("id") != s6.STAGE_ID
        or attempt.get("execution_commit") != S6_EXECUTION_COMMIT
        or attempt.get("runner_sha256") != S6_RUNNER_SHA256
        or failure.get("decision")
        != "implementation_error_before_schedule_construction"
        or failure.get("failure", {}).get("exception_type")
        != "AttributeError"
        or failure.get("failure", {}).get("failed_expression")
        != "residual.reconstruct_reward_tensor"
        or boundary.get("2020_transition_ledger_rows_parsed") != 3_288
        or boundary.get("2020_original_reward_tensor_reconstructed")
        is not False
        or boundary.get("2020_residual_reward_values_created") != 0
        or boundary.get("2020_economic_metrics_computed") != 0
        or boundary.get("raw_market_or_funding_paths_read") != []
        or boundary.get("2021_market_or_funding_paths_read") != []
        or boundary.get("2021_reward_rows_created") != 0
        or boundary.get("2021_economic_metrics_computed") != 0
        or boundary.get("2021_policy_specific_outcomes_opened") is not False
        or boundary.get("2022_or_later_outcomes_opened") is not False
        or outputs.get("attempt_written") is not True
        or any(
            outputs.get(name) is not False
            for name in (
                "result_written",
                "residual_ledger_written",
                "base_schedules_written",
                "delayed_schedule_written",
                "schedule_manifest_written",
            )
        )
        or repair.get("delete_or_overwrite_failed_attempt") is not False
        or repair.get("same_scientific_hypothesis_required") is not True
        or repair.get(
            "new_preregistration_and_distinct_write_once_paths_required"
        )
        is not True
        or repair.get("2021_outcome_access_before_repaired_schedule_gate")
        is not False
    ):
        raise RuntimeError("PSIM S6R1 predecessor failure boundary changed")
    return {
        "registration": registration,
        "attempt": attempt,
        "failure": failure,
    }


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_s6_implementation_failure()
    original = deepcopy(predecessor["registration"])
    unchanged_contract = {
        key: original[key]
        for key in (
            "frozen_source_and_outcome_artifacts",
            "action_mean_residual_reward_contract",
            "fitted_q_contract",
            "control_family",
            "pre2021_schedule_readiness_gate",
            "future_2021_transfer_gate",
        )
    }
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            **original["candidate"],
            "id": STAGE_ID,
            "predecessor": s6.STAGE_ID,
            "stage": "code_only_repair_and_repeat_identical_s6_gate",
            "scientific_hypothesis_changed": False,
            "outcome_information_used_for_repair": False,
        },
        "failed_execution_evidence": {
            "s6_preregistration": {
                "path": s6.DEFAULT_OUTPUT.as_posix(),
                "sha256": S6_PREREGISTRATION_SHA256,
                "manifest_hash": S6_PREREGISTRATION_MANIFEST_HASH,
            },
            "s6_execution_commit": S6_EXECUTION_COMMIT,
            "s6_runner": {
                "path": s6.RUNNER_PATH.as_posix(),
                "sha256": S6_RUNNER_SHA256,
            },
            "s6_core_sha256": S6_CORE_SHA256,
            "s6_attempt": {
                "path": s6.ATTEMPT_PATH.as_posix(),
                "sha256": S6_ATTEMPT_SHA256,
                "attempt_hash": S6_ATTEMPT_HASH,
            },
            "s6_failure": {
                "path": S6_FAILURE_PATH.as_posix(),
                "sha256": S6_FAILURE_SHA256,
                "failure_hash": S6_FAILURE_HASH,
            },
            "s6_failure_document": {
                "path": S6_FAILURE_DOCUMENT.as_posix(),
                "sha256": S6_FAILURE_DOCUMENT_SHA256,
            },
        },
        "repair_contract": {
            "authorized_change": (
                "replace residual.reconstruct_reward_tensor with "
                "residual.s5_core.reconstruct_reward_tensor"
            ),
            "authorized_change_count": 1,
            "model_changed": False,
            "source_features_changed": False,
            "reward_formula_changed": False,
            "fit_rows_changed": False,
            "algorithm_or_hyperparameters_changed": False,
            "policy_or_control_family_changed": False,
            "readiness_gate_changed": False,
            "future_transfer_gate_changed": False,
            "failed_attempt_deleted_or_overwritten": False,
            "distinct_write_once_paths": True,
        },
        "scientific_contract_identity": {
            "source_stage": s6.STAGE_ID,
            "unchanged_sections": list(unchanged_contract),
            "unchanged_contract_hash": canonical_hash(unchanged_contract),
        },
        **unchanged_contract,
        "global_outcome_contamination_disclosure": {
            **original["global_outcome_contamination_disclosure"],
            "s6r1_policy_specific_2021_metrics_already_exist": False,
            "s6_failure_revealed_2021_outcomes": False,
        },
        "chronology": [
            "verify the exact sealed S6 implementation failure",
            "commit and hash-bind the new S6R1 runner",
            "write a distinct S6R1 attempt before parsing the 2020 ledger",
            "read no raw market or funding payload",
            "call the existing S4 reward reconstruction through s5_core",
            "execute the otherwise byte-equivalent S6 scientific pipeline",
            "seal 2021 schedules before any 2021 outcome read",
            "run the identical S6 activity/direction/invariance gate",
            (
                "terminally reject on failure; on pass authorize only a "
                "separate report-only 2021 transfer preregistration"
            ),
        ],
        "artifact_contract": {
            "runner": RUNNER_PATH.as_posix(),
            "attempt": ATTEMPT_PATH.as_posix(),
            "result": RESULT_PATH.as_posix(),
            "residual_reward_ledger_2020": (
                RESIDUAL_LEDGER_PATH.as_posix()
            ),
            "base_schedules_2021": SCHEDULE_PATH.as_posix(),
            "delayed_primary_schedule_2021": (
                DELAYED_SCHEDULE_PATH.as_posix()
            ),
            "schedule_gate_manifest_2021": (
                SCHEDULE_MANIFEST_PATH.as_posix()
            ),
            "fixed_paths_no_output_override": True,
            "deterministic_csv_gzip_and_json": True,
            "write_once": True,
            "result_published_last": True,
            "all_paths_distinct_from_failed_s6": True,
        },
        "execution_contract": {
            **original["execution_contract"],
            "failed_s6_attempt_preserved": True,
            "exact_s6_failure_validation_required": True,
            "full_execute_smoke_test_before_commit_required": True,
        },
        "access_boundary": {
            "source_files_read": sorted(
                {
                    *original["access_boundary"]["source_files_read"],
                    s6.DEFAULT_OUTPUT.as_posix(),
                    s6.ATTEMPT_PATH.as_posix(),
                    s6.RUNNER_PATH.as_posix(),
                    S6_FAILURE_PATH.as_posix(),
                    S6_FAILURE_DOCUMENT.as_posix(),
                    "training/psim_action_mean_residual_fqi.py",
                }
            ),
            "raw_market_or_funding_paths_read": [],
            "original_2020_outcome_artifact_bytes_hashed": True,
            "original_2020_transition_reward_rows_parsed": 0,
            "2021_market_or_funding_paths_read": [],
            "2021_market_rows_parsed": 0,
            "2021_funding_rows_parsed": 0,
            "2021_rewards_created": 0,
            "2021_economic_metrics_computed": 0,
            "2021_policy_specific_outcomes_opened": False,
            "2022_or_later_outcomes_opened": False,
            "model_loaded": False,
            "model_forwards_started": 0,
        },
        "next_authorized_step": (
            "IMPLEMENT_FULL_EXECUTE_SMOKE_REVIEW_COMMIT_AND_PUSH_S6R1_"
            "RUNNER_THEN_EXECUTE_OUTCOME_BLIND_PRE2021_SCHEDULE_GATE_ONLY"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(
    path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    target = repository_path(path)
    payload = build_preregistration()
    raw = canonical_bytes(payload, pretty=True)
    if target.exists():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise RuntimeError("PSIM S6R1 preregistration drift")
        return payload
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"]["id"],
                "manifest_hash": payload["manifest_hash"],
                "next_authorized_step": payload["next_authorized_step"],
                "output": repository_path(args.output).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
