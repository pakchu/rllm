from __future__ import annotations

from training import preregister_fee_endpoint_topology_disagreement as prereg


EXPECTED_ARTIFACT_FILE_SHA256 = (
    "d27fdf851b328cd3735b58992d877c74986da9aa169b30ee4daf7993204e394f"
)
EXPECTED_MANIFEST_HASH = (
    "d45ef8806dff23bb457b857749b806ef1ecda22fea90aa15bf1940ae147ea45b"
)
EXPECTED_POLICY_HASH = (
    "ac7b4bfdf9da7d79dfcb8627557d691d50656474c55c4c20cdaedb2cb695896c"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "887b74fa5ae1ba123c4a51440273f52d5ea4c52df4bf5029001c357c11c49950"
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

