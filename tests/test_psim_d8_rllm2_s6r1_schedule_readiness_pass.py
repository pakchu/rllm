from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from training import (
    preregister_psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi as prereg,
)
from training import psim_action_mean_residual_fqi as residual
from training import (
    run_psim_d8_rllm2_s6r1_train2020_action_mean_residual_ridge_fqi as runner,
)

ATTEMPT = runner.repository_path(prereg.ATTEMPT_PATH)
RESULT = runner.repository_path(prereg.RESULT_PATH)
MANIFEST = runner.repository_path(prereg.SCHEDULE_MANIFEST_PATH)
EXECUTION_COMMIT = "aee87ef7a543715b08a581f7c70a86547a39d135"
RUNNER_SHA256 = (
    "c791054d32b5f4e1891916bf4bce9beb070d9fdddeb759318393374068595016"
)
EXPECTED_FILE_SHA256 = {
    "attempt": (
        "f4669c0a37878351bd89ad0d0554f02b6509a15b3ab80e4dbf7f2def9616cbaa"
    ),
    "result": (
        "020c86002df8348c497407b70e24d6e85d583c3734242c6bddb0b5764b260d2f"
    ),
    "manifest": (
        "816afbc7bca16df0313636194e4b0780bbe760cb7cd10f7944736c6968352644"
    ),
    "residual_ledger": (
        "073565b87c67deb9754a4e6e99f1ba59e855ef8fe3937ed9a25cb3a9367a0273"
    ),
    "base_schedules": (
        "f850b3f9e18e9d942b8279065512fa33e77a9493abbd4d927672dc025ab41971"
    ),
    "delayed_schedule": (
        "b324347e72e08de8266dc39f7f800fae2a73a98129ab7092cb02b92893a02e28"
    ),
}
ATTEMPT_HASH = (
    "b50b2c4314603931e03c3ae37b92a3cc4d1ed426784440761dad9c17c731cdd6"
)
RESULT_HASH = (
    "63bec557d8e95a21becdad69c648f339d6f12510a28a679668a7bef3f4edd862"
)
MANIFEST_HASH = (
    "314298356bbf3bc94e394ae362ee4f1894fd8f07e6e1a0b47137dd785f78970e"
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
        ["git", "show", f"{EXECUTION_COMMIT}:{prereg.RUNNER_PATH}"],
        cwd=prereg.REPO_ROOT,
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


def test_residual_ledger_equalizes_every_action_mean() -> None:
    path = runner.repository_path(prereg.RESIDUAL_LEDGER_PATH)
    assert runner._sha256_file(path) == EXPECTED_FILE_SHA256[
        "residual_ledger"
    ]
    ledger = pd.read_csv(path)

    assert tuple(ledger.columns) == residual.RESIDUAL_LEDGER_COLUMNS
    assert len(ledger) == 3_288
    assert np.all(np.isfinite(ledger["reward"]))
    assert np.all(np.isfinite(ledger["action_mean_residual_reward"]))
    for position in residual.fqi.POSITION_NAMES:
        selected = ledger.loc[ledger["current_position"].eq(position)]
        means = [
            selected.loc[
                selected["action_name"].eq(action),
                "action_mean_residual_reward",
            ].mean()
            for action in residual.fqi.ACTION_NAMES
        ]
        assert float(np.ptp(means)) <= 1e-15
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["fit"]["fitted_q_count"] == 7
    assert result["fit"]["ridge_fitted_q_count"] == 7
    assert result["fit"]["extra_trees_fitted_q_count"] == 0
    assert result["fit"]["primary_selected_from_2020_metrics"] is False


def test_sealed_schedules_are_complete_balanced_and_exactly_delayed() -> None:
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
    primary = (
        base.loc[base["policy_id"].eq(prereg.PRIMARY_POLICY_ID)]
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    delayed = delayed.sort_values("entry_time").reset_index(drop=True)
    assert delayed["sequence_id"].tolist() == [
        f"{value}:delay_5m" for value in primary["sequence_id"]
    ]
    assert delayed["target"].equals(primary["target"])
    assert (
        pd.to_datetime(delayed["entry_time"], utc=True)
        == pd.to_datetime(primary["entry_time"], utc=True)
        + pd.Timedelta(minutes=5)
    ).all()
    readiness = result["schedule_readiness"]
    assert readiness["target_counts"] == {
        "TARGET_FLAT": 9,
        "TARGET_LONG": 141,
        "TARGET_SHORT": 215,
    }
    assert readiness["nonflat_target_rows"] == 356
    assert readiness["long_share_of_nonflat_targets"] == 141 / 356
    assert readiness["short_share_of_nonflat_targets"] == 215 / 356
    assert readiness["action_code_permutation_mismatch_count"] == 0
    assert readiness["s5_primary_target_hamming"] == 282
    assert all(readiness["gates"].values())
    assert readiness["passed"] is True


def test_schedule_pass_opens_no_2021_economic_outcome() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["decision"] == "pass"
    assert result["terminal_action"] == (
        "SEAL_S6_2021_SCHEDULE_AUTHORIZE_SEPARATE_REPORT_ONLY_"
        "2021_TRANSFER_PREREGISTRATION"
    )
    assert result["authorize_separate_2021_transfer_preregistration"] is True
    assert result["authorize_2022_or_later_outcomes"] is False
    assert result["qlora_authorized"] is False
    boundary = result["access_boundary"]
    assert boundary == runner._expected_access_boundary(
        prereg.build_preregistration()
    )
    encoded = RESULT.read_text(encoding="utf-8").lower()
    assert '"cagr"' not in encoded
    assert '"strict_mdd"' not in encoded
    disclosure = result["global_2021_outcome_disclosure"]
    assert disclosure["globally_pristine_2021_claim"] is False
    assert disclosure["s6r1_policy_specific_2021_metrics_exist"] is False


def test_existing_result_resume_revalidates_exact_pass_record() -> None:
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    existing = runner._verify_existing_result(attempt)

    assert existing is not None
    assert existing["result_hash"] == RESULT_HASH
    assert existing["decision"] == "pass"
