from __future__ import annotations

from training import preregister_blockspace_load_settlement_relay as prereg


EXPECTED_ARTIFACT_FILE_SHA256 = (
    "d8a79e22a5670e88baa2a0108d79ce515c5f6f9e093615cba9290ec4b1369f5b"
)
EXPECTED_MANIFEST_HASH = (
    "87d4486be4c37d6f6078409c719f35ff196583c80fa834cf0fdf47d5f437a29a"
)
EXPECTED_POLICY_HASH = (
    "f9194cb23d01e910861f573a735293faa087f315c422000ef4da049409f6f62f"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "f26015b4b18af4c6e3ca12330abffb389b71300158c206409eb383215d4f0810"
)


def test_frozen_blsr288_preregistration_artifact() -> None:
    assert prereg.sha256_file(prereg.DEFAULT_OUTPUT) == EXPECTED_ARTIFACT_FILE_SHA256
    artifact = prereg.load_preregistration()

    assert artifact["protocol_version"] == prereg.PROTOCOL_VERSION
    assert artifact["policy_id"] == prereg.POLICY_ID
    assert artifact["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert artifact["policy_hash"] == EXPECTED_POLICY_HASH
    assert artifact["preregistration_source"] == {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    assert artifact["mechanism_decision"] == {
        "path": str(prereg.MECHANISM_DECISION),
        "sha256": prereg.MECHANISM_DECISION_SHA256,
    }
    assert artifact["source_validator"] == {
        "path": str(prereg.SOURCE_VALIDATOR),
        "sha256": prereg.SOURCE_VALIDATOR_SHA256,
    }
    assert artifact["source_manifest"]["source_output"]["sha256"] == (
        prereg.source_contract.EXPECTED_SOURCE_OUTPUT_SHA256
    )
    assert artifact["comparator_bindings"] == prereg.COMPARATOR_BINDINGS


def test_frozen_blsr288_preregistration_keeps_outcomes_closed() -> None:
    artifact = prereg.load_preregistration()

    assert artifact["outcomes_opened"] is False
    assert artifact["outcome_boundary"] == prereg.OUTCOME_BOUNDARY
    assert artifact["outcome_boundary"]["source_manifest_json_read"] is True
    assert artifact["outcome_boundary"]["source_artifact_bytes_hashed"] is True
    assert artifact["outcome_boundary"]["comparator_artifact_bytes_hashed"] is True
    for field in (
        "source_csv_values_read",
        "comparator_rows_read",
        "blsr_feature_rows_derived",
        "signal_incidence_rows_derived",
        "market_rows_loaded",
        "funding_rows_loaded",
        "return_or_pnl_fields",
        "post_2023_rows_loaded",
        "network_calls",
        "subprocess_calls",
    ):
        assert artifact["outcome_boundary"][field] == 0


def test_frozen_blsr288_policy_is_singleton_and_has_no_grid() -> None:
    artifact = prereg.load_preregistration()
    policy = artifact["policy"]

    assert policy == prereg.policy()
    assert policy["singleton"] is True
    assert policy["normalization"]["parameter_grid"] == []
    assert policy["relay"]["deadline_packets_after_onset"] == 3
    assert policy["relay"]["no_retry"] is True
    assert policy["execution"]["hold_bars"] == 288
    assert policy["calendar"]["sealed"] == "2024+"
    assert policy["support_gates"]["support_failure_action"] == (
        "reject BLSR-288 before outcomes; no repair"
    )
    assert "post_outcome_repair" in policy["performance_gates"]
    assert policy["performance_gates"]["post_outcome_repair"] == "forbidden"
