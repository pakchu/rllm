from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_psim_d8_rllm2_s4_semantic_fqi as prereg
from training import (
    preregister_psim_d8_rllm2_s3_chunked_native_source_feature as s3,
)


def test_build_preregistration_binds_successful_s3_and_keeps_oos_closed() -> None:
    payload = prereg.build_preregistration()

    assert payload["candidate"] == {
        "id": prereg.STAGE_ID,
        "predecessor": s3.STAGE_ID,
        "stage": "train_2020_semantic_fqi_and_seal_2021",
        "profitability_claim": False,
        "selection_from_2020_training_metrics": False,
    }
    evidence = payload["predecessor_terminal_evidence"]
    assert evidence["terminal_result"]["result_hash"] == prereg.S3_RESULT_HASH
    assert (
        evidence["terminal_result"]["terminal_action"]
        == "ACCEPT_PSIM_D8_RLLM2_S3_CHUNKED_NATIVE_SOURCE_FEATURE_"
        "SEAL_OPEN_2020_TRAIN_OUTCOMES_ONLY"
    )
    boundary = payload["access_boundary"]
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["train_2020_outcomes_opened"] is False
    assert boundary["2021_outcomes_opened"] is False
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_forwards_started"] == 0
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == s3.canonical_hash(core)


def test_semantic_family_and_training_degrees_are_frozen() -> None:
    payload = prereg.build_preregistration()

    assert payload["fitted_q_contract"]["primary_policy_ids"] == [
        "semantic_ridge_fqi",
        "semantic_extra_trees_fqi",
    ]
    assert payload["fitted_q_contract"]["bellman_iterations"] == 25
    assert payload["fitted_q_contract"]["discount"] == 0.99
    assert (
        payload["fitted_q_contract"]["algorithms"][
            "semantic_ridge_fqi"
        ]["alpha"]
        == 100.0
    )
    extra = payload["fitted_q_contract"]["algorithms"][
        "semantic_extra_trees_fqi"
    ]
    assert extra == {
        "kind": "extra_trees",
        **prereg.EXTRA_TREES_CONTRACT,
    }
    assert payload["semantic_state_contract"]["pca"] == {
        "fit_rows": "2020 only",
        "components": 32,
        "solver": "full_svd",
        "whiten": False,
        "deterministic_sign_rule": (
            "for each component, its largest-absolute loading is forced "
            "positive; lowest index breaks ties"
        ),
        "refit_on_2021_source_distribution": False,
    }
    assert tuple(
        payload["control_family"]["policy_family_ids"]
    ) == prereg.POLICY_FAMILY_IDS
    assert len(prereg.POLICY_FAMILY_IDS) == len(
        set(prereg.POLICY_FAMILY_IDS)
    )
    assert payload["chronology_and_gates"]["qlora_authorized_in_s4"] is False
    assert (
        payload["chronology_and_gates"][
            "2021_or_later_outcomes_authorized_in_s4"
        ]
        is False
    )
    authorized = payload["authorized_2020_outcome_sources"]
    assert authorized["stage"] == "2020"
    assert authorized["start_inclusive"] == "2020-01-01T00:00:00Z"
    assert authorized["end_exclusive"] == "2021-01-01T00:00:00Z"
    assert authorized["market"]["expected_rows"] == 105_408
    assert authorized["funding"]["expected_rows"] == 1_098
    assert authorized["full_parent_payload_hash_during_s4"] is False
    assert authorized["2021_or_later_numeric_rows_authorized"] is False
    assert (
        authorized[
            "existing_2021_or_later_stage_local_payload_open_authorized"
        ]
        is False
    )


def test_source_feature_hashes_and_dimensions_are_exact() -> None:
    payload = prereg.build_preregistration()
    source = payload["source_feature_binding"]

    assert source["source_rows"]["sha256"] == prereg.S3_SOURCE_ROWS_SHA256
    assert (
        source["source_rows"]["source_row_roster_hash"]
        == prereg.S3_SOURCE_ROW_ROSTER_HASH
    )
    assert source["embeddings"]["shape"] == [1_461, 2_560]
    assert source["embeddings"]["dtype"] == "float32"
    assert source["relation_logits"]["shape"] == [1_461, 6]
    assert source["relation_logits"]["dtype"] == "float32"
    assert source["relation_rows"]["rows"] == 1_461


def test_write_is_deterministic_and_drift_closed(tmp_path: Path) -> None:
    output = tmp_path / "preregistration.json"
    first = prereg.write_preregistration(output)
    raw = output.read_bytes()
    second = prereg.write_preregistration(output)

    assert first == second
    assert output.read_bytes() == raw
    parsed = json.loads(raw)
    assert parsed == first

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="preregistration drift"):
        prereg.write_preregistration(output)


def test_committed_generated_artifact_matches_builder() -> None:
    payload = prereg.build_preregistration()
    target = prereg.repository_path(prereg.DEFAULT_OUTPUT)

    assert target.read_bytes() == s3.canonical_json_bytes(
        payload,
        pretty=True,
    )
    assert json.loads(target.read_bytes())["manifest_hash"] == payload[
        "manifest_hash"
    ]


def test_s3_outcome_payload_hashing_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reader = prereg._read_exact_json
    result = json.loads(
        prereg.repository_path(s3.RESULT_PATH).read_text(encoding="utf-8")
    )
    result["access_boundary"][
        "market_or_funding_payload_bytes_hashed"
    ] = True
    core = {
        key: value for key, value in result.items() if key != "result_hash"
    }
    result["result_hash"] = s3.canonical_hash(core)

    def fake_reader(
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> dict[str, object]:
        if Path(path) == s3.RESULT_PATH:
            return result
        return original_reader(path, expected_sha256=expected_sha256)

    monkeypatch.setattr(prereg, "_read_exact_json", fake_reader)
    monkeypatch.setattr(prereg, "S3_RESULT_HASH", result["result_hash"])

    with pytest.raises(RuntimeError, match="success evidence changed"):
        prereg.validate_s3_success()


def test_s3_artifact_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = tmp_path / "embeddings.npz"
    fake.write_bytes(b"tampered")
    monkeypatch.setattr(s3, "EMBEDDINGS_PATH", fake)

    with pytest.raises(RuntimeError, match="artifact changed"):
        prereg.build_preregistration()
