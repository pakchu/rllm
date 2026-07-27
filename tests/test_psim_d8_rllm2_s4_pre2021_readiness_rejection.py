from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from training import (
    audit_psim_d8_rllm2_s4_pre2021_readiness as audit,
)
from training import (
    preregister_psim_d8_rllm2_s4_semantic_fqi as prereg,
)


RESULT = audit.repository_path(audit.DEFAULT_OUTPUT)
EXPECTED_AUDIT_FILE_SHA256 = (
    "e670c0965ba9fccccc06f472cff696e1392a2853d4d8016350191849d4548560"
)
EXPECTED_AUDIT_RESULT_HASH = (
    "36e81a7a42aa3a06b95e01c5203ec8f363a11b6771bafd28614ce4f055611ba9"
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


def test_s4_execution_and_every_terminal_artifact_are_exact() -> None:
    assert audit.sha256_file(RESULT) == EXPECTED_AUDIT_FILE_SHA256
    for name, path in {
        "preregistration": audit.S4_PREREGISTRATION,
        "attempt": audit.S4_ATTEMPT,
        "result": audit.S4_RESULT,
        "schedule_manifest": audit.S4_SCHEDULE_MANIFEST,
        "base_schedules": audit.S4_BASE_SCHEDULES,
        "delayed_schedules": audit.S4_DELAYED_SCHEDULES,
        "transition_ledger": audit.S4_TRANSITION_LEDGER,
        "pca": audit.S4_PCA,
    }.items():
        assert audit.sha256_file(path) == audit.EXPECTED_FILE_SHA256[name]
    attempt = json.loads(
        audit.repository_path(audit.S4_ATTEMPT).read_text(encoding="utf-8")
    )
    attempt_core = {
        key: value
        for key, value in attempt.items()
        if key != "attempt_hash"
    }
    assert attempt["attempt_hash"] == audit.EXPECTED_ATTEMPT_HASH
    assert attempt["attempt_hash"] == _canonical_hash(attempt_core)
    s4_result = json.loads(
        audit.repository_path(audit.S4_RESULT).read_text(encoding="utf-8")
    )
    s4_core = {
        key: value
        for key, value in s4_result.items()
        if key != "result_hash"
    }
    assert s4_result["result_hash"] == audit.EXPECTED_RESULT_HASH
    assert s4_result["result_hash"] == _canonical_hash(s4_core)
    manifest = json.loads(
        audit.repository_path(audit.S4_SCHEDULE_MANIFEST).read_text(
            encoding="utf-8"
        )
    )
    manifest_core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    assert (
        manifest["manifest_hash"]
        == audit.EXPECTED_SCHEDULE_MANIFEST_HASH
    )
    assert manifest["manifest_hash"] == _canonical_hash(manifest_core)
    executed_runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{audit.EXPECTED_EXECUTION_COMMIT}:training/"
                "run_psim_d8_rllm2_s4_train2020_semantic_fqi.py"
            ),
        ],
        cwd=audit.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(executed_runner).hexdigest() == (
        audit.EXPECTED_RUNNER_SHA256
    )


def test_schedule_roster_clock_domain_and_numeric_artifacts_are_exact() -> None:
    base_raw = gzip.decompress(
        audit.repository_path(audit.S4_BASE_SCHEDULES).read_bytes()
    )
    delayed_raw = gzip.decompress(
        audit.repository_path(audit.S4_DELAYED_SCHEDULES).read_bytes()
    )
    assert hashlib.sha256(base_raw).hexdigest() == (
        audit.EXPECTED_DECODED_SHA256["base_schedules"]
    )
    assert hashlib.sha256(delayed_raw).hexdigest() == (
        audit.EXPECTED_DECODED_SHA256["delayed_schedules"]
    )
    base = pd.read_csv(audit.repository_path(audit.S4_BASE_SCHEDULES))
    delayed = pd.read_csv(audit.repository_path(audit.S4_DELAYED_SCHEDULES))
    assert tuple(base.columns) == audit.EXPECTED_COLUMNS
    assert tuple(delayed.columns) == audit.EXPECTED_COLUMNS
    assert base.shape == (9_125, 4)
    assert delayed.shape == (730, 4)
    assert tuple(base["policy_id"].drop_duplicates()) == (
        prereg.POLICY_FAMILY_IDS
    )
    assert base.groupby("policy_id", sort=False).size().to_dict() == {
        policy_id: 365 for policy_id in prereg.POLICY_FAMILY_IDS
    }
    assert delayed.groupby("policy_id", sort=False).size().to_dict() == {
        primary_id: 365 for primary_id in prereg.PRIMARY_POLICY_IDS
    }
    assert base.duplicated(["policy_id", "sequence_id"]).sum() == 0
    assert delayed.duplicated(["policy_id", "sequence_id"]).sum() == 0
    assert set(base["target"]) == set(audit.EXPECTED_TARGETS)
    assert set(delayed["target"]) == set(audit.EXPECTED_TARGETS)
    assert base["entry_time"].min() == "2021-01-01T12:05:00Z"
    assert base["entry_time"].max() == "2021-12-31T12:05:00Z"
    assert delayed["entry_time"].min() == "2021-01-01T12:10:00Z"
    assert delayed["entry_time"].max() == "2021-12-31T12:10:00Z"
    ledger = pd.read_csv(
        audit.repository_path(audit.S4_TRANSITION_LEDGER)
    )
    assert ledger.shape == (3_288, 16)
    assert int(ledger["reachable"].sum()) == 3_288
    assert np.all(np.isfinite(ledger["reward"]))
    with np.load(
        audit.repository_path(audit.S4_PCA),
        allow_pickle=False,
    ) as pca:
        assert pca["components"].shape == (32, 2_560)
        assert pca["fit_row_index"].shape == (366,)
        assert pca["mean"].shape == (2_560,)
        assert np.all(np.isfinite(pca["components"]))
        assert np.all(np.isfinite(pca["mean"]))


def test_pre2021_readiness_rejects_both_primaries_without_outcomes() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    assert payload["result_hash"] == EXPECTED_AUDIT_RESULT_HASH
    assert payload["result_hash"] == _canonical_hash(core)
    assert payload["decision"] == "reject"
    assert payload["terminal_action"] == audit.REJECT_ACTION
    assert payload["eligible_primary_ids"] == []
    assert payload["eligible_primary_count"] == 0
    assert payload["audit_authorizes_open_2021_outcomes"] is False
    ridge = payload["primary_schedule_diagnostics"][
        "semantic_ridge_fqi"
    ]
    assert ridge["target_counts"] == {
        "TARGET_FLAT": 252,
        "TARGET_LONG": 99,
        "TARGET_SHORT": 14,
    }
    assert ridge["nonflat_target_rows"] == 113
    assert ridge["long_share_of_nonflat_targets"] == 99 / 113
    assert ridge["short_share_of_nonflat_targets"] == 14 / 113
    assert ridge["activity_pass"] is True
    assert ridge["direction_balance_pass"] is False
    assert (
        ridge["action_code_permutation_exact_target_identity"] is True
    )
    extra = payload["primary_schedule_diagnostics"][
        "semantic_extra_trees_fqi"
    ]
    assert extra["target_counts"] == {
        "TARGET_FLAT": 325,
        "TARGET_LONG": 40,
        "TARGET_SHORT": 0,
    }
    assert extra["nonflat_target_rows"] == 40
    assert extra["long_share_of_nonflat_targets"] == 1.0
    assert extra["short_share_of_nonflat_targets"] == 0.0
    assert extra["activity_pass"] is False
    assert extra["direction_balance_pass"] is False
    assert (
        extra["action_code_permutation_exact_target_identity"] is False
    )
    assert extra["action_code_permutation_mismatch_count"] == 2
    assert [
        row["entry_time"]
        for row in extra["action_code_permutation_mismatches"]
    ] == [
        "2021-11-30T12:05:00Z",
        "2021-12-01T12:05:00Z",
    ]
    assert all(
        row["primary_target"] == "TARGET_FLAT"
        and row["permuted_target"] == "TARGET_LONG"
        for row in extra["action_code_permutation_mismatches"]
    )


def test_rejection_preserves_every_future_outcome_boundary() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    boundary = payload["access_boundary"]
    assert boundary == {
        "audit_economic_metrics_computed": 0,
        "audit_funding_rows_parsed": 0,
        "audit_market_or_funding_paths_read": [],
        "audit_market_rows_parsed": 0,
        "audit_rewards_created": 0,
        "globally_pristine_2021_outcome_claim_allowed": False,
        "s4_protocol_2021_outcomes_remain_sealed": True,
        "s4_protocol_2022_or_later_outcomes_remain_sealed": True,
    }
    assert payload["predecessor_2021_authorization_exercised"] is False
    disposition = payload["scientific_disposition"]
    assert disposition["in_sample_2020_metrics_are_alpha_evidence"] is False
    assert disposition["2021_oos_metrics_exist"] is False
    assert disposition["candidate_promoted"] is False
    assert disposition["s4_rerun_authorized"] is False
    assert disposition["separate_successor_preregistration_authorized"] is True
    assert disposition["successor_must_not_use_known_2021_metrics_for_design"]
    assert disposition["globally_pristine_evidence_requires_forward_or_live_data"]
    history = payload["global_2021_outcome_history"]
    assert history["globally_pristine_2021_claim_allowed"] is False
    assert history["prior_2021_transfer_result_exists"] is True
    assert history["prior_2021_stage_market_rows"] == 105_120
    assert history["prior_2021_stage_funding_rows"] == 1_095
    assert audit.sha256_file(
        history["prior_2021_transfer_result_document"]
    ) == audit.EXPECTED_PRIOR_2021_TRANSFER_DOCUMENT_SHA256
    assert audit.sha256_file(
        history["prior_2021_stage_source_manifest"]
    ) == audit.EXPECTED_PRIOR_2021_STAGE_SOURCE_MANIFEST_SHA256


def test_audit_replays_exactly_and_write_once_fails_closed(
    tmp_path: Path,
) -> None:
    replay = audit.build_audit()
    stored = json.loads(RESULT.read_text(encoding="utf-8"))
    assert replay == stored
    output = tmp_path / "audit.json"
    assert audit.write_once(output, replay) == "created"
    assert audit.write_once(output, replay) == "verified_existing"
    changed = json.loads(json.dumps(replay))
    changed["eligible_primary_count"] = 1
    try:
        audit.write_once(output, changed)
    except RuntimeError as exc:
        assert "write-once" in str(exc)
    else:
        raise AssertionError("PSIM S4 readiness accepted drift")
