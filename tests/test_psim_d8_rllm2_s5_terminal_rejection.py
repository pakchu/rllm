from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from training import (
    preregister_psim_d8_rllm2_s5_direction_residual_ridge_fqi as prereg,
)
from training import psim_direction_residual_fqi as residual
from training import (
    run_psim_d8_rllm2_s5_train2020_direction_residual_ridge_fqi as runner,
)


ATTEMPT = runner.repository_path(prereg.ATTEMPT_PATH)
RESULT = runner.repository_path(prereg.RESULT_PATH)
MANIFEST = runner.repository_path(prereg.SCHEDULE_MANIFEST_PATH)
EXECUTION_COMMIT = "c814807c7a25b0a9ec28c2bd7db4bbd5ac6ac520"
RUNNER_SHA256 = (
    "320382ccea359704a5fbe3cc2599c981dee456f26fd66c4f5a98c67274e179f2"
)
EXPECTED_FILE_SHA256 = {
    "attempt": (
        "ab59c4fe359fe4a8a3b05c7a128999489ba2679bf453e79c1bbb9428baaa0d42"
    ),
    "result": (
        "aabc479e030a3a08ddd07047ccb25ce872425655ca10d877b8e78c4687bf73a7"
    ),
    "manifest": (
        "76f66a4353d0d8e3bb8f2aa1671f6a5416dd7f344c22bdeed63a485f984a1d60"
    ),
    "residual_ledger": (
        "b395aa4cb0b0288eb7287eea297bc3b1b416e1326b379a48952a6fb8489fc19b"
    ),
    "base_schedules": (
        "ed5ed208104242f571f4a07a53a815a1db2742957f43e9fcbecd9c18221a3833"
    ),
    "delayed_schedule": (
        "e0580cf4bb1a0a3a05710945abfe20edb9c05bb47a30c34aafdb43556d85e8b5"
    ),
}
ATTEMPT_HASH = (
    "f0c4dd9ac216ba9458a4541cc01af2290ff51d1bc4caa54d2b873e67d84a7efc"
)
RESULT_HASH = (
    "2331b46e3847d80200c88b3f3522ef35f3f710dead5ea6dca85575f249b0541e"
)
MANIFEST_HASH = (
    "ca0ef82f5e9886288942c2ff8b115fa9e64eeee143b358ebd90ceb32b7fb5022"
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _self_hashed(
    path: Path,
    *,
    field: str,
    expected: str,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != field
    }
    assert payload[field] == _canonical_hash(core)
    assert payload[field] == expected
    return payload


def test_attempt_result_manifest_and_executed_runner_are_exact() -> None:
    assert runner._sha256_file(ATTEMPT) == EXPECTED_FILE_SHA256["attempt"]
    assert runner._sha256_file(RESULT) == EXPECTED_FILE_SHA256["result"]
    assert runner._sha256_file(MANIFEST) == EXPECTED_FILE_SHA256["manifest"]
    attempt = _self_hashed(
        ATTEMPT,
        field="attempt_hash",
        expected=ATTEMPT_HASH,
    )
    result = _self_hashed(
        RESULT,
        field="result_hash",
        expected=RESULT_HASH,
    )
    manifest = _self_hashed(
        MANIFEST,
        field="manifest_hash",
        expected=MANIFEST_HASH,
    )
    executed_runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{EXECUTION_COMMIT}:training/"
                "run_psim_d8_rllm2_s5_train2020_"
                "direction_residual_ridge_fqi.py"
            ),
        ],
        cwd=runner.prereg.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(executed_runner).hexdigest() == RUNNER_SHA256
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert attempt["runner_sha256"] == RUNNER_SHA256
    assert result["execution_commit"] == EXECUTION_COMMIT
    assert result["runner_sha256"] == RUNNER_SHA256
    assert result["attempt_hash"] == ATTEMPT_HASH
    assert manifest["attempt_hash"] == ATTEMPT_HASH


def test_residual_ledger_is_exact_and_equalizes_long_short_means() -> None:
    path = runner.repository_path(prereg.RESIDUAL_LEDGER_PATH)
    assert runner._sha256_file(path) == EXPECTED_FILE_SHA256[
        "residual_ledger"
    ]
    ledger = pd.read_csv(path)

    assert tuple(ledger.columns) == residual.RESIDUAL_LEDGER_COLUMNS
    assert len(ledger) == 3_288
    assert np.all(np.isfinite(ledger["reward"]))
    assert np.all(np.isfinite(ledger["direction_residual_reward"]))
    for position in (
        "POSITION_SHORT",
        "POSITION_FLAT",
        "POSITION_LONG",
    ):
        selected = ledger.loc[ledger["current_position"].eq(position)]
        long_mean = selected.loc[
            selected["action_name"].eq("TARGET_LONG"),
            "direction_residual_reward",
        ].mean()
        short_mean = selected.loc[
            selected["action_name"].eq("TARGET_SHORT"),
            "direction_residual_reward",
        ].mean()
        assert abs(float(long_mean - short_mean)) <= 1e-15
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["fit"]["direction_residual_delta_by_current_position"] == {
        "POSITION_FLAT": 0.002227566963135719,
        "POSITION_LONG": 0.0031416503931195293,
        "POSITION_SHORT": 0.0013514627359444101,
    }
    assert result["fit"]["fitted_q_count"] == 7
    assert result["fit"]["ridge_fitted_q_count"] == 7
    assert result["fit"]["extra_trees_fitted_q_count"] == 0


def test_sealed_schedules_are_complete_invariant_and_fail_only_activity() -> None:
    base_path = runner.repository_path(prereg.SCHEDULE_PATH)
    delayed_path = runner.repository_path(prereg.DELAYED_SCHEDULE_PATH)
    assert runner._sha256_file(base_path) == EXPECTED_FILE_SHA256[
        "base_schedules"
    ]
    assert runner._sha256_file(delayed_path) == EXPECTED_FILE_SHA256[
        "delayed_schedule"
    ]
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    artifacts = result["artifacts"]
    assert hashlib.sha256(gzip.decompress(base_path.read_bytes())).hexdigest() == (
        artifacts["base_schedules_2021"]["decoded_sha256"]
    )
    assert hashlib.sha256(
        gzip.decompress(delayed_path.read_bytes())
    ).hexdigest() == (
        artifacts["delayed_primary_schedule_2021"]["decoded_sha256"]
    )
    base = pd.read_csv(base_path)
    delayed = pd.read_csv(delayed_path)
    assert base.shape == (2_920, 4)
    assert delayed.shape == (365, 4)
    assert tuple(base["policy_id"].drop_duplicates()) == (
        prereg.POLICY_FAMILY_IDS
    )
    assert base.groupby("policy_id", sort=False).size().to_dict() == {
        policy_id: 365 for policy_id in prereg.POLICY_FAMILY_IDS
    }
    assert base.duplicated(["policy_id", "sequence_id"]).sum() == 0
    primary = base.loc[
        base["policy_id"].eq(prereg.PRIMARY_POLICY_ID)
    ].reset_index(drop=True)
    permutation = base.loc[
        base["policy_id"].eq(
            f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation"
        )
    ].reset_index(drop=True)
    assert primary["sequence_id"].equals(permutation["sequence_id"])
    assert primary["entry_time"].equals(permutation["entry_time"])
    assert primary["target"].equals(permutation["target"])
    readiness = result["schedule_readiness"]
    assert readiness["target_counts"] == {
        "TARGET_FLAT": 291,
        "TARGET_LONG": 44,
        "TARGET_SHORT": 30,
    }
    assert readiness["delayed_target_counts"] == readiness["target_counts"]
    assert readiness["nonflat_target_rows"] == 74
    assert readiness["long_share_of_nonflat_targets"] == 44 / 74
    assert readiness["short_share_of_nonflat_targets"] == 30 / 74
    assert readiness["action_code_permutation_mismatch_count"] == 0
    assert readiness["gates"] == {
        "action_code_permutation_exact_target_identity": True,
        "all_degenerate_control_hamming_positive": True,
        "base_primary_schedule_rows": True,
        "delayed_primary_schedule_rows": True,
        "delayed_target_counts_equal_base": True,
        "minimum_long_share": True,
        "minimum_nonflat_target_rows": False,
        "minimum_short_share": True,
    }
    assert readiness["passed"] is False


def test_terminal_rejection_opens_no_2021_or_economic_metric() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["decision"] == "reject"
    assert result["terminal_action"] == (
        "TERMINAL_REJECT_S5_WITHOUT_2021_MARKET_FUNDING_REWARD_"
        "OR_METRIC_ACCESS"
    )
    assert (
        result["authorize_separate_2021_transfer_preregistration"] is False
    )
    assert result["authorize_2022_or_later_outcomes"] is False
    assert result["qlora_authorized"] is False
    boundary = result["access_boundary"]
    assert boundary["raw_market_or_funding_paths_read"] == []
    assert boundary["2020_transition_ledger_rows_parsed"] == 3_288
    assert boundary["2020_economic_metrics_computed"] == 0
    assert boundary["2021_market_or_funding_paths_read"] == []
    assert boundary["2021_market_rows_parsed"] == 0
    assert boundary["2021_funding_rows_parsed"] == 0
    assert boundary["2021_reward_rows_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0
    assert boundary["2021_policy_specific_outcomes_opened"] is False
    assert boundary["2022_or_later_outcomes_opened"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_forwards_started"] == 0
    encoded = RESULT.read_text(encoding="utf-8").lower()
    assert '"cagr"' not in encoded
    assert '"strict_mdd"' not in encoded
    disclosure = result["global_2021_outcome_disclosure"]
    assert disclosure["globally_pristine_2021_claim"] is False
    assert disclosure["prior_unrelated_2021_transfer_result_exists"] is True
    assert disclosure["s5_policy_specific_2021_metrics_exist"] is False


def test_existing_result_resume_revalidates_exact_terminal_record() -> None:
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    existing = runner._verify_existing_result(attempt)

    assert existing is not None
    assert existing["result_hash"] == RESULT_HASH
    assert existing["decision"] == "reject"
