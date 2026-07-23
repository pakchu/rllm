from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_ofr_repo_collateral_routing_efficiency as rcre


ARTIFACT = Path(
    "results/ofr_repo_collateral_routing_efficiency_preregistration_2026-07-23.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_preregistration_artifact_is_hash_bound_and_replayable() -> None:
    report = payload()
    assert sha256(ARTIFACT) == (
        "1cd5c773e22101ae19d8d0753ca53e3248b10de3ecc947fc2e4e353961ee69e2"
    )
    assert report["manifest_hash"] == (
        "a8a7831c773666b42b98d673165b3e7844111c5d5ed280468307306c5843bf4b"
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == rcre.canonical_hash(core)
    assert report["policy_hash"] == (
        "5ef37a4e6b1a8132fa51a4e7d6db460459dcdb4e69cb6a7bc2aa0a00a3dadfcd"
    )
    assert report["policy_hash"] == rcre.canonical_hash(report["policy"])
    rcre.validate_preregistration(report, verify_sources=True)


def test_artifact_binds_local_source_comparators_history_and_protocol() -> None:
    report = payload()
    assert report["verification_mode"] == "verified_hashes"
    assert report["artifact_eligible"] is True
    assert report["source_binding"]["observations_sha256"] == (
        "6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a"
    )
    assert report["source_binding"]["manifest_observation_rows"] == 77_369
    assert report["source_binding"]["manifest_series"] == 82
    assert len(report["comparator_bindings"]) == 13
    assert report["comparator_bindings"][-1]["name"] == (
        "ofr_repo_mix_shock_resolution_race_primary"
    )
    assert report["comparator_bindings"][-1]["sha256"] == (
        "bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6"
    )
    assert len(report["history_bindings"]) == 10
    assert report["common_window_policy"]["sha256"] == (
        "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
    )
    assert report["preregistration_source"]["sha256"] == (
        "c082f38f0556e2c7ecbb1796b3a64128d200e008b133a5bb2a4263f8788220db"
    )


def test_artifact_discloses_reuse_and_keeps_new_information_closed() -> None:
    report = payload()
    disclosure = report["prior_research_disclosure"]
    assert disclosure["source_row_values_previously_opened"] is True
    assert disclosure["rvfc_and_rmsr_absolute_component_incidence_previously_opened"]
    assert disclosure["signed_quantity_gap_opened"] is False
    assert disclosure["signed_rate_gap_opened"] is False
    assert disclosure["routing_pressure_or_rcre_incidence_opened"] is False
    assert disclosure["prior_comparator_timing_rows_partially_opened_for_validation"]
    assert report["signed_features_or_rcre_incidence_opened"] is False
    assert report["comparator_rows_opened_during_preregistration"] is False
    assert report["outcomes_opened"] is False
    assert report["performance_values_opened"] is False

    boundary = report["outcome_boundary"]
    assert boundary["source_observation_value_rows_read_during_preregistration"] == 0
    assert boundary["source_metadata_definition_rows_read_during_preregistration"] == 0
    assert boundary["history_value_rows_read_during_preregistration"] == 0
    assert boundary["comparator_value_rows_read_during_preregistration"] == 0
    assert boundary["signed_features_computed"] == 0
    assert boundary["rcre_states_or_events_derived"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
