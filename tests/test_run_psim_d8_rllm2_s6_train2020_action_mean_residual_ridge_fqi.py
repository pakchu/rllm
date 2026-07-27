from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training import (
    preregister_psim_d8_rllm2_s6_action_mean_residual_ridge_fqi as prereg,
)
from training import (
    run_psim_d8_rllm2_s6_train2020_action_mean_residual_ridge_fqi as runner,
)


def test_validate_preregistration_is_exact_and_outcome_closed() -> None:
    payload = runner.validate_preregistration()

    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert runner._sha256_file(prereg.DEFAULT_OUTPUT) == (
        runner.PREREGISTRATION_SHA256
    )
    assert payload["candidate"]["id"] == prereg.STAGE_ID
    assert payload["fitted_q_contract"]["primary_policy_id"] == (
        prereg.PRIMARY_POLICY_ID
    )
    assert payload["control_family"]["new_policy_family_ids"] == list(
        prereg.POLICY_FAMILY_IDS
    )
    boundary = payload["access_boundary"]
    assert boundary["raw_market_or_funding_paths_read"] == []
    assert boundary["2021_market_or_funding_paths_read"] == []
    assert boundary["2021_rewards_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0


def test_attempt_is_self_hashed_and_precedes_every_parse() -> None:
    payload = runner._attempt_payload("a" * 40, "b" * 64)
    core = {
        key: value
        for key, value in payload.items()
        if key != "attempt_hash"
    }

    assert payload["attempt_hash"] == runner._canonical_hash(core)
    assert payload["authorization"] == {
        "parse_frozen_2020_transition_ledger": True,
        "read_raw_market_or_funding_payload": False,
        "open_2021_policy_specific_outcomes": False,
        "open_2022_or_later_outcomes": False,
    }
    assert payload["access_boundary_at_attempt"] == {
        "raw_market_or_funding_paths_read": [],
        "2020_transition_ledger_rows_parsed": 0,
        "2020_residual_reward_values_created": 0,
        "2020_economic_metrics_computed": 0,
        "2021_market_or_funding_paths_read": [],
        "2021_reward_rows_created": 0,
        "2021_economic_metrics_computed": 0,
        "2021_policy_specific_outcomes_opened": False,
        "2022_or_later_outcomes_opened": False,
        "model_loaded": False,
        "model_forwards_started": 0,
    }


def test_frozen_pca_and_source_hashes_validate_without_market_access() -> None:
    runner._verify_source_artifacts()
    bundle = runner.features.load_source_bundle(
        runner.repository_path(prereg.s4.s3.SOURCE_ROWS_PATH),
        runner.repository_path(prereg.s4.s3.EMBEDDINGS_PATH),
        runner.repository_path(prereg.s4.s3.RELATION_LOGITS_PATH),
        runner.repository_path(prereg.s4.s3.RELATION_ROWS_PATH),
    )
    train_indices = runner.features.year_indices(bundle.rows, 2020)
    pca = runner._load_frozen_pca(train_indices)

    assert len(train_indices) == 366
    assert pca.fit_row_count == 366
    assert pca.components.shape == (32, 2_560)
    assert pca.mean.shape == (2_560,)
    assert np.all(np.isfinite(pca.components))
    assert np.all(np.isfinite(pca.mean))


def test_deterministic_gzip_and_schedule_manifest_are_self_bound() -> None:
    frame = pd.DataFrame(
        [
            {
                "policy_id": prereg.PRIMARY_POLICY_ID,
                "sequence_id": "x",
                "entry_time": "2021-01-01T12:05:00Z",
                "target": "TARGET_FLAT",
            }
        ]
    )
    first, decoded = runner._deterministic_csv_gzip_bytes(frame)
    second, second_decoded = runner._deterministic_csv_gzip_bytes(frame)
    assert first == second
    assert decoded == second_decoded
    assert hashlib.sha256(gzip.decompress(first)).hexdigest() == decoded
    artifact = runner._artifact_record(
        "data/example.csv.gz",
        first,
        extra={"decoded_sha256": decoded, "rows": 1},
    )
    readiness = {
        "passed": False,
        "gates": {"minimum_short_share": False},
    }
    manifest = runner._schedule_manifest(
        attempt_hash="a" * 64,
        residual_ledger_artifact=artifact,
        base_artifact=artifact,
        delayed_artifact=artifact,
        readiness=readiness,
    )
    core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    assert manifest["manifest_hash"] == runner._canonical_hash(core)
    assert manifest["decision"] == "reject"
    assert manifest["outcome_boundary"] == {
        "raw_market_or_funding_payload_opened": False,
        "2021_market_or_funding_payload_opened": False,
        "2021_reward_rows_created": 0,
        "2021_economic_metrics_computed": 0,
        "2021_policy_specific_outcomes_opened": False,
        "2022_or_later_outcomes_opened": False,
    }
    assert manifest["global_2021_pristine_claim"] is False


def test_write_once_accepts_identity_and_rejects_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    runner._write_once(output, b"first")
    runner._write_once(output, b"first")

    assert output.read_bytes() == b"first"
    with pytest.raises(RuntimeError, match="write-once artifact drift"):
        runner._write_once(output, b"second")


def test_validate_preregistration_rejects_tampered_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tampered = json.loads(
        prereg.repository_path(prereg.DEFAULT_OUTPUT).read_text(
            encoding="utf-8"
        )
    )
    tampered["candidate"]["profitability_claim"] = True
    core = {
        key: value
        for key, value in tampered.items()
        if key != "manifest_hash"
    }
    tampered["manifest_hash"] = runner._canonical_hash(core)
    output = tmp_path / "prereg.json"
    output.write_bytes(runner._canonical_bytes(tampered, pretty=True))
    monkeypatch.setattr(prereg, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(
        runner,
        "PREREGISTRATION_SHA256",
        hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "PREREGISTRATION_MANIFEST_HASH",
        tampered["manifest_hash"],
    )

    with pytest.raises(RuntimeError, match="contract changed"):
        runner.validate_preregistration()


def test_frozen_input_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = tmp_path / "pca.npz"
    fake.write_bytes(b"tampered")
    monkeypatch.setattr(prereg.s4, "PCA_PATH", fake)

    with pytest.raises(RuntimeError, match="frozen input changed"):
        runner._verify_source_artifacts()



def test_execute_records_attempt_before_transition_ledger_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    attempt = {
        "execution_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "attempt_hash": "c" * 64,
    }
    rows = [
        {
            "decision_at": "2020-01-01T12:05:00Z",
            "row_hash": f"{index:064x}",
        }
        for index in range(731)
    ]
    monkeypatch.setattr(runner, "validate_preregistration", dict)
    monkeypatch.setattr(
        runner,
        "assert_committed_clean_runner",
        lambda: ("a" * 40, "b" * 64),
    )

    def write_attempt(*_args: str) -> dict[str, str]:
        events.append("attempt")
        return attempt

    monkeypatch.setattr(runner, "_write_or_validate_attempt", write_attempt)
    monkeypatch.setattr(runner, "_verify_existing_result", lambda _arg: None)
    monkeypatch.setattr(
        runner,
        "_verify_source_artifacts",
        lambda: events.append("source_hashes"),
    )
    monkeypatch.setattr(
        runner.features,
        "load_source_bundle",
        lambda *_args: SimpleNamespace(rows=rows),
    )
    monkeypatch.setattr(
        runner.features,
        "year_indices",
        lambda _rows, year: (
            np.arange(366)
            if year == 2020
            else np.arange(366, 731)
        ),
    )

    def reject_ledger_parse(*_args: object, **_kwargs: object) -> None:
        events.append("transition_ledger_parse")
        raise RuntimeError("sentinel transition parse")

    monkeypatch.setattr(runner.pd, "read_csv", reject_ledger_parse)

    with pytest.raises(RuntimeError, match="sentinel transition parse"):
        runner.execute()

    assert events == [
        "attempt",
        "source_hashes",
        "transition_ledger_parse",
    ]


def test_existing_result_cannot_authorize_failed_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    residual_path = tmp_path / "residual.csv.gz"
    base_path = tmp_path / "base.csv.gz"
    delayed_path = tmp_path / "delayed.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    for path, raw in (
        (residual_path, b"residual"),
        (base_path, b"base"),
        (delayed_path, b"delayed"),
    ):
        path.write_bytes(raw)
    monkeypatch.setattr(prereg, "RESIDUAL_LEDGER_PATH", residual_path)
    monkeypatch.setattr(prereg, "SCHEDULE_PATH", base_path)
    monkeypatch.setattr(prereg, "DELAYED_SCHEDULE_PATH", delayed_path)
    monkeypatch.setattr(prereg, "SCHEDULE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(prereg, "RESULT_PATH", result_path)
    monkeypatch.setattr(
        prereg,
        "build_preregistration",
        lambda: {
            "pre2021_schedule_readiness_gate": {
                "pass_action": "PASS",
                "failure_action": "REJECT",
            }
        },
    )
    readiness = {"passed": False, "gates": {"required_gate": False}}
    residual_artifact = runner._artifact_record(
        residual_path,
        residual_path.read_bytes(),
    )
    base_artifact = runner._artifact_record(
        base_path,
        base_path.read_bytes(),
    )
    delayed_artifact = runner._artifact_record(
        delayed_path,
        delayed_path.read_bytes(),
    )
    manifest_core = {
        "protocol_version": runner.SCHEDULE_PROTOCOL_VERSION,
        "attempt_hash": "c" * 64,
        "decision": "pass",
        "schedule_readiness": readiness,
        "residual_reward_ledger_2020": residual_artifact,
        "base_schedules_2021": base_artifact,
        "delayed_primary_schedule_2021": delayed_artifact,
    }
    manifest = {
        **manifest_core,
        "manifest_hash": runner._canonical_hash(manifest_core),
    }
    manifest_path.write_bytes(runner._canonical_bytes(manifest, pretty=True))
    manifest_artifact = runner._artifact_record(
        manifest_path,
        manifest_path.read_bytes(),
    )
    attempt = {
        "execution_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "attempt_hash": "c" * 64,
    }
    result_core = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "decision": "pass",
        "execution_commit": attempt["execution_commit"],
        "runner_sha256": attempt["runner_sha256"],
        "attempt_hash": attempt["attempt_hash"],
        "preregistration_manifest_hash": (
            runner.PREREGISTRATION_MANIFEST_HASH
        ),
        "terminal_action": "PASS",
        "authorize_separate_2021_transfer_preregistration": True,
        "authorize_2022_or_later_outcomes": False,
        "qlora_authorized": False,
        "schedule_readiness": readiness,
        "artifacts": {
            "residual_reward_ledger_2020": residual_artifact,
            "base_schedules_2021": base_artifact,
            "delayed_primary_schedule_2021": delayed_artifact,
            "schedule_gate_manifest_2021": manifest_artifact,
        },
        "access_boundary": {
            "raw_market_or_funding_paths_read": [],
            "2021_market_or_funding_paths_read": [],
            "2021_reward_rows_created": 0,
            "2021_economic_metrics_computed": 0,
            "2021_policy_specific_outcomes_opened": False,
            "2022_or_later_outcomes_opened": False,
        },
    }
    result = {
        **result_core,
        "result_hash": runner._canonical_hash(result_core),
    }
    result_path.write_bytes(runner._canonical_bytes(result, pretty=True))

    with pytest.raises(RuntimeError, match="existing result changed"):
        runner._verify_existing_result(attempt)


def test_expected_artifact_paths_are_fixed_to_preregistration() -> None:
    assert runner._expected_artifact_paths() == {
        "residual_reward_ledger_2020": prereg.RESIDUAL_LEDGER_PATH,
        "base_schedules_2021": prereg.SCHEDULE_PATH,
        "delayed_primary_schedule_2021": prereg.DELAYED_SCHEDULE_PATH,
        "schedule_gate_manifest_2021": prereg.SCHEDULE_MANIFEST_PATH,
    }
