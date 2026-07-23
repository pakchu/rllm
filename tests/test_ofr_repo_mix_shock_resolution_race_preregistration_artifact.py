from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_ofr_repo_mix_shock_resolution_race as rmsr


ARTIFACT = Path(
    "results/ofr_repo_mix_shock_resolution_race_preregistration_2026-07-23.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_preregistration_artifact_is_hash_bound_and_replayable() -> None:
    report = payload()
    assert sha256(ARTIFACT) == (
        "3a85ac8fa41aa4caacb6ff4875ec4e19d01ec0450f8959f84cb16787a5cc3f45"
    )
    assert report["manifest_hash"] == (
        "a9ad2e858e9b037e459a1f5fe2cfba1af5a921a35ab100a4d1eca027807baa53"
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == rmsr.canonical_hash(core)
    assert report["policy_hash"] == (
        "30ef81e6ba53e8923521c9573bb60e234086d92308568f4b25ea19619d52aac9"
    )
    assert report["policy_hash"] == rmsr.canonical_hash(report["policy"])
    rmsr.validate_preregistration(report, verify_sources=True)


def test_artifact_binds_verified_source_comparators_and_protocol() -> None:
    report = payload()
    assert report["verification_mode"] == "verified_hashes"
    assert report["artifact_eligible"] is True
    assert report["source_binding"]["observations_sha256"] == (
        "6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a"
    )
    assert report["source_binding"]["manifest_observation_rows"] == 77_369
    assert report["source_binding"]["manifest_series"] == 82
    assert len(report["comparator_bindings"]) == 12
    assert report["comparator_bindings"][-1]["name"] == (
        "ofr_repo_venue_fragmentation_consensus_primary"
    )
    assert report["comparator_bindings"][-1]["sha256"] == (
        "b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e"
    )
    assert report["preregistration_source"]["sha256"] == (
        "8819bdef5b9ca7f8bf0a0b548fe2e8e0b615859679710173597f9a297a1f1ab8"
    )


def test_artifact_discloses_prior_source_exposure_and_keeps_new_boundary_closed() -> None:
    report = payload()
    disclosure = report["prior_research_disclosure"]
    assert disclosure["source_row_values_previously_opened"] is True
    assert disclosure["rvfc_component_values_and_incidence_previously_opened"]
    assert disclosure["exact_rmsr_precursor_terminal_pairing_opened"] is False
    assert disclosure["exact_rmsr_race_incidence_or_side_mix_opened"] is False
    assert report["exact_race_incidence_opened"] is False
    assert report["comparator_rows_opened"] is False
    assert report["outcomes_opened"] is False
    assert report["performance_values_opened"] is False

    boundary = report["outcome_boundary"]
    assert boundary["source_observation_value_rows_read_during_preregistration"] == 0
    assert boundary["source_metadata_definition_rows_read_during_preregistration"] == 0
    assert boundary["prior_source_support_value_rows_read_during_preregistration"] == 0
    assert boundary["comparator_value_rows_read_during_preregistration"] == 0
    assert boundary["rmsr_features_computed"] == 0
    assert boundary["rmsr_races_or_events_derived"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
