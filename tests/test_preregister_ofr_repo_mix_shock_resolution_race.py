from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_ofr_repo_mix_shock_resolution_race as rmsr


def test_policy_freezes_exact_race_source_and_gates() -> None:
    policy = rmsr.policy_payload()
    assert policy["candidate"] == "RMSR-72-SOURCE-REUSE"
    assert tuple(policy["source"]["required_series"]) == rmsr.REQUIRED_SERIES
    assert policy["source"]["TRI_including_fed_forbidden"] is True
    assert policy["source"]["DVP_and_venue_total_rows_forbidden"] is True
    assert policy["source"]["required_row_availability"] == (
        "max(observation_date+8 elapsed calendar days,2020-09-10T00:00:00Z)"
    )
    assert policy["materiality"]["each_ag_and_t_share_minimum"] == "1/20"
    assert set(policy["components"]) == set(rmsr.COMPONENTS)
    assert policy["normalization"]["history_complete_dates"] == 252
    assert policy["race"]["terminal_window_complete_decision_dates"] == 20
    assert policy["race"]["terminal_must_be_strictly_later"] is True
    assert policy["race"]["terminal_date_cannot_rearm"] is True
    assert policy["execution"]["hold_elapsed_hours"] == 72
    assert policy["windows"]["sealed_from"] == "2024-01-01T00:00:00Z"
    assert policy["source_support_gates"]["train_total_minimum"] == 35
    assert policy["source_support_gates"]["selection_total_minimum"] == 14
    assert policy["source_support_gates"]["strict_terminal_after_precursor_required"]
    assert policy["source_support_gates"][
        "strictly_earlier_eligible_precursor_required"
    ]
    assert policy["source_support_gates"]["accepted_ambiguity_count_required"] == 0
    assert policy["source_support_gates"]["global_nonoverlap_required"] is True
    assert policy["source_support_gates"]["post_2023_source_rows_read_required"] == 0
    assert len(policy["novelty"]["comparators"]) == 12
    assert policy["mutable_parameters"] == []


def test_preregistration_discloses_prior_source_but_opens_no_new_values() -> None:
    payload = rmsr.build_preregistration(verify_sources=False)
    rmsr.validate_preregistration(payload, verify_sources=False)
    assert payload["source_family_values_previously_opened"] is True
    assert payload["source_component_incidence_previously_opened"] is True
    assert payload["exact_race_incidence_opened"] is False
    assert payload["comparator_rows_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["verification_mode"] == "static_test_fixture"
    assert payload["artifact_eligible"] is False
    assert payload["source_binding"][
        "observation_value_rows_read_during_preregistration"
    ] == 0
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )
    assert all(
        row["values_read_during_rmsr_preregistration"] == 0
        for row in payload["history_bindings"]
    )
    assert payload["outcome_boundary"] == rmsr.STATIC_TEST_OUTCOME_BOUNDARY
    assert payload["outcome_boundary"][
        "source_file_bytes_hashed_during_preregistration"
    ] is False
    assert payload["source_binding"]["manifest_metadata_parsed"] is False
    assert all(
        row["read_mode"] == "declared static fixture binding; no file read or hash"
        for row in (*payload["comparator_bindings"], *payload["history_bindings"])
    )


def test_real_source_comparator_and_history_hashes_are_bound() -> None:
    payload = rmsr.build_preregistration(verify_sources=True)
    rmsr.validate_preregistration(payload, verify_sources=True)
    assert payload["source_binding"]["manifest_observation_rows"] == 77_369
    assert payload["source_binding"]["manifest_series"] == 82
    assert len(payload["comparator_bindings"]) == len(rmsr.COMPARATOR_SPECS)
    assert len(payload["history_bindings"]) == len(rmsr.HISTORY_BINDINGS)
    assert payload["comparator_bindings"][-1]["name"] == (
        "ofr_repo_venue_fragmentation_consensus_primary"
    )
    assert payload["mechanism_decision"]["sha256"] == (
        rmsr.MECHANISM_DECISION_SHA256
    )
    assert payload["verification_mode"] == "verified_hashes"
    assert payload["artifact_eligible"] is True
    assert payload["outcome_boundary"] == rmsr.EXPECTED_OUTCOME_BOUNDARY


def test_policy_or_boundary_tampering_fails_closed() -> None:
    payload = rmsr.build_preregistration(verify_sources=False)
    payload["policy"]["race"]["terminal_window_complete_decision_dates"] = 19
    with pytest.raises(RuntimeError, match="policy drift"):
        rmsr.validate_preregistration(payload, verify_sources=False)

    payload = rmsr.build_preregistration(verify_sources=False)
    payload["exact_race_incidence_opened"] = True
    with pytest.raises(RuntimeError, match="boundary opened"):
        rmsr.validate_preregistration(payload, verify_sources=False)


def test_repository_paths_reject_escape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rmsr, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="repository-relative"):
        rmsr._repository_path("../escape.json")
    with pytest.raises(RuntimeError, match="repository-relative"):
        rmsr._repository_path(tmp_path / "absolute.json")


def test_write_is_deterministic_and_refuses_different_existing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(rmsr, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(rmsr, "MECHANISM_DECISION", Path("mechanism.md"))
    monkeypatch.setattr(rmsr, "SCRIPT_PATH", Path("preregister.py"))
    (tmp_path / "mechanism.md").write_text("fixture\n")
    (tmp_path / "preregister.py").write_text("# fixture\n")
    monkeypatch.setattr(
        rmsr,
        "MECHANISM_DECISION_SHA256",
        rmsr.sha256_file("mechanism.md"),
    )
    monkeypatch.setattr(rmsr, "_source_binding", rmsr._static_source_binding)
    monkeypatch.setattr(
        rmsr,
        "_hash_bindings",
        lambda specs, *, history: rmsr._static_bindings(specs, history=history),
    )

    cfg = rmsr.Config(output="out/prereg.json")
    first, status = rmsr.write_preregistration(cfg)
    assert status == "created"
    second, status = rmsr.write_preregistration(cfg)
    assert status == "verified_existing"
    assert first == second

    path = tmp_path / cfg.output
    changed = json.loads(path.read_text())
    changed["manifest_hash"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        rmsr.write_preregistration(cfg)


def test_atomic_write_never_clobbers_existing_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(rmsr, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "out.json"
    path.write_text("sentinel\n")
    with pytest.raises(FileExistsError):
        rmsr._atomic_write(path, {"candidate": "replacement"})
    assert path.read_text() == "sentinel\n"
