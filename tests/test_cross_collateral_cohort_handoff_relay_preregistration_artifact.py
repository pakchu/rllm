from __future__ import annotations

from training import preregister_cross_collateral_cohort_handoff_relay as prereg


EXPECTED_ARTIFACT_FILE_SHA256 = (
    "b3792b0f24c4e0e022903c359db143337b0c998034600480edbc29095d039056"
)
EXPECTED_MANIFEST_HASH = (
    "eff9fbc086f5238a05e748570902835a950433a9eb3b51fbd132969bbcb1269d"
)
EXPECTED_POLICY_HASH = (
    "d71619a973315c55266cb2c7708cd5a922e10114ef3c2f923c3b1698d03cb901"
)
EXPECTED_CANDIDATE_MAP_HASH = (
    "f46537d7883c87ffe3542d1e12de675b4fac72f0d1ec385ee3d26a57f8c11af8"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "797dffd052fd9b5f922ad06780c21c94815ee3403212b38d12405076382c800f"
)


def test_frozen_cchr288_preregistration_artifact() -> None:
    assert prereg.sha256_file(prereg.DEFAULT_OUTPUT) == EXPECTED_ARTIFACT_FILE_SHA256
    artifact = prereg.load_preregistration()

    assert artifact["protocol_version"] == prereg.PROTOCOL_VERSION
    assert artifact["policy_id"] == prereg.POLICY_ID
    assert artifact["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert artifact["policy_hash"] == EXPECTED_POLICY_HASH
    assert artifact["comparator_candidate_map_hash"] == EXPECTED_CANDIDATE_MAP_HASH
    assert artifact["preregistration_source"] == {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    assert artifact["mechanism_decision"] == {
        "path": str(prereg.MECHANISM_DECISION),
        "sha256": prereg.MECHANISM_DECISION_SHA256,
    }
    assert artifact["source_binding"]["sha256"] == prereg.SOURCE_SHA256
    assert artifact["source_binding"]["manifest_sha256"] == (
        prereg.SOURCE_MANIFEST_SHA256
    )


def test_frozen_cchr288_preregistration_keeps_incidence_and_outcomes_closed() -> None:
    artifact = prereg.load_preregistration()

    assert artifact["outcomes_opened"] is False
    assert artifact["outcome_boundary"] == prereg.OUTCOME_BOUNDARY
    assert artifact["authorization"] == {
        "outcome_evaluator_authorized": False,
        "real_incidence_authorized_by_this_artifact": False,
        "reason": "mandatory pure-clock comparator freeze is a separate prerequisite",
    }
    for field in (
        "source_csv_values_read",
        "comparator_rows_read",
        "outcome_bearing_provenance_json_parsed",
        "cchr_feature_rows_derived",
        "signal_incidence_rows_derived",
        "market_rows_loaded",
        "funding_rows_loaded",
        "return_or_pnl_fields",
        "post_2023_rows_loaded",
        "network_calls",
        "subprocess_calls",
    ):
        assert artifact["outcome_boundary"][field] == 0


def test_frozen_cchr288_policy_and_comparator_prerequisites_are_exact() -> None:
    artifact = prereg.load_preregistration()
    policy = artifact["policy"]

    assert policy == prereg.policy()
    assert policy["singleton"] is True
    assert policy["features"]["parameter_grid"] == []
    assert policy["features"]["rank_lookback_hourly_anchors"] == 168
    assert policy["execution"]["hold_bars"] == 288
    assert policy["source"]["existing_source_complete_excluded"] is True
    assert policy["support_gates"]["failure_action"] == (
        "retire CCHR-288 before outcomes; no repair"
    )
    assert len(artifact["comparator_candidate_map"]) == 62
    assert set(artifact["pure_clock_requirements"]) == {
        "pdlh",
        "dtv",
        "far",
        "live",
    }
    assert (
        artifact["comparator_freeze_requirement"]["required_before_real_incidence"]
        is True
    )
