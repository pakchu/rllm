from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_ofr_repo_venue_fragmentation_consensus as rvfc


def test_policy_freezes_exact_source_mechanism_and_gates() -> None:
    policy = rvfc.policy_payload()
    assert policy["candidate"] == "RVFC-72-NEW-SOURCE"
    assert tuple(policy["source"]["required_series"]) == rvfc.REQUIRED_SERIES
    assert policy["source"]["TRI_including_fed_forbidden"] is True
    assert policy["materiality"]["each_ag_and_t_share_minimum"] == "1/20"
    assert set(policy["components"]) == set(rvfc.COMPONENTS)
    assert policy["normalization"]["history_complete_dates"] == 252
    assert policy["state"]["positive_score_minimum"] == 0.50
    assert policy["execution"]["hold_elapsed_hours"] == 72
    assert policy["windows"]["sealed_from"] == "2024-01-01T00:00:00Z"
    assert policy["source_support_gates"]["train_total_minimum"] == 60
    assert policy["source_support_gates"]["selection_total_minimum"] == 20
    assert len(policy["novelty"]["comparators"]) == 11
    assert policy["novelty"]["maximum_rvfc_one_day_containment"] == 0.35
    assert policy["mutable_parameters"] == []


def test_preregistration_is_incidence_comparator_and_outcome_blind() -> None:
    payload = rvfc.build_preregistration(verify_sources=False)
    rvfc.validate_preregistration(payload, verify_sources=False)
    assert payload["exact_source_incidence_opened"] is False
    assert payload["comparator_rows_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["source_binding"][
        "observation_value_rows_read_during_preregistration"
    ] == 0
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )
    assert payload["outcome_boundary"] == rvfc.EXPECTED_OUTCOME_BOUNDARY


def test_real_source_and_comparator_hashes_are_bound() -> None:
    payload = rvfc.build_preregistration(verify_sources=True)
    rvfc.validate_preregistration(payload, verify_sources=True)
    assert payload["source_binding"]["manifest_observation_rows"] == 77_369
    assert payload["source_binding"]["manifest_series"] == 82
    assert len(payload["comparator_bindings"]) == len(rvfc.COMPARATOR_SPECS)
    assert len(payload["history_bindings"]) == len(rvfc.HISTORY_BINDINGS)
    assert payload["mechanism_decision"]["sha256"] == (
        rvfc.MECHANISM_DECISION_SHA256
    )


def test_policy_or_boundary_tampering_fails_closed() -> None:
    payload = rvfc.build_preregistration(verify_sources=False)
    payload["policy"]["execution"]["hold_elapsed_hours"] = 24
    with pytest.raises(RuntimeError, match="policy drift"):
        rvfc.validate_preregistration(payload, verify_sources=False)

    payload = rvfc.build_preregistration(verify_sources=False)
    payload["comparator_rows_opened"] = True
    with pytest.raises(RuntimeError, match="boundary opened"):
        rvfc.validate_preregistration(payload, verify_sources=False)


def test_write_is_deterministic_and_refuses_different_existing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(rvfc, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(rvfc, "MECHANISM_DECISION", Path("mechanism.md"))
    monkeypatch.setattr(rvfc, "SCRIPT_PATH", Path("preregister.py"))
    (tmp_path / "mechanism.md").write_text("fixture\n")
    (tmp_path / "preregister.py").write_text("# fixture\n")
    monkeypatch.setattr(
        rvfc,
        "MECHANISM_DECISION_SHA256",
        rvfc.sha256_file("mechanism.md"),
    )
    monkeypatch.setattr(rvfc, "_source_binding", rvfc._static_source_binding)
    monkeypatch.setattr(
        rvfc,
        "_hash_bindings",
        lambda specs, *, history: [
            {
                "name": spec["name"],
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
                "read_mode": "raw bytes for SHA-256 only",
                **(
                    {
                        "historical_values_previously_opened": True,
                        "values_read_during_rvfc_preregistration": 0,
                    }
                    if history
                    else {
                        "parser": spec["parser"],
                        "comparison": [
                            "2021-01-01T00:00:00Z",
                            "2024-01-01T00:00:00Z",
                        ],
                        "value_rows_read_during_preregistration": 0,
                    }
                ),
            }
            for spec in specs
        ],
    )

    cfg = rvfc.Config(output="out/prereg.json")
    first, status = rvfc.write_preregistration(cfg)
    assert status == "created"
    second, status = rvfc.write_preregistration(cfg)
    assert status == "verified_existing"
    assert first == second

    path = tmp_path / cfg.output
    changed = json.loads(path.read_text())
    changed["manifest_hash"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        rvfc.write_preregistration(cfg)
