from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import (
    preregister_psim_d8_rllm2_s5_direction_residual_ridge_fqi as prereg,
)
from training import (
    run_psim_d8_rllm2_s5_train2020_direction_residual_ridge_fqi as runner,
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


def test_official_terminal_outputs_are_now_complete() -> None:
    for path in (
        prereg.ATTEMPT_PATH,
        prereg.RESULT_PATH,
        prereg.RESIDUAL_LEDGER_PATH,
        prereg.SCHEDULE_PATH,
        prereg.DELAYED_SCHEDULE_PATH,
        prereg.SCHEDULE_MANIFEST_PATH,
    ):
        assert runner.repository_path(path).is_file()
