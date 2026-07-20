from __future__ import annotations

from training import preregister_fee_endpoint_topology_disagreement as prereg


EXPECTED_ARTIFACT_FILE_SHA256 = (
    "2de820b6f78d0cd566f2750f91bfca8c092795ab93b81121326bdb067247e285"
)
EXPECTED_MANIFEST_HASH = (
    "8e2042c5439ca6a7b73cbc05eb2c8f0600edc622833bc33afaabb12e178e0fbe"
)
EXPECTED_POLICY_HASH = (
    "7bd47d38c93c6507ce75bab455f3b3e7efee69a803f56bc1ab78ebcb85d9434e"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "ae1329f0cd124787d822096e56dc3bc3ed05ccd2f2f6f0cb86f47e5cd766c413"
)


def test_frozen_fetd_preregistration_artifact() -> None:
    assert prereg.sha256_file(prereg.DEFAULT_OUTPUT) == EXPECTED_ARTIFACT_FILE_SHA256
    artifact = prereg.load_preregistration()
    assert artifact["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert artifact["policy_hash"] == EXPECTED_POLICY_HASH
    assert artifact["preregistration_source"] == {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    assert artifact["source_manifest"]["source_output"]["sha256"] == (
        prereg.EXPECTED_SOURCE_OUTPUT_SHA256
    )
    assert artifact["outcomes_opened"] is False
    assert artifact["outcome_boundary"]["source_csv_values_read"] == 0
    assert artifact["outcome_boundary"]["fetd_feature_rows_derived"] == 0
    assert artifact["outcome_boundary"]["market_rows_loaded"] == 0
    assert artifact["outcome_boundary"]["funding_rows_loaded"] == 0
    assert artifact["outcome_boundary"]["return_or_pnl_fields"] == 0
