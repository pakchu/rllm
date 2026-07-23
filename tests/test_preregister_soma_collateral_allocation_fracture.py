from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_soma_collateral_allocation_fracture as p


def test_manifest_is_deterministic_and_self_hashing() -> None:
    first = p.build_manifest()
    second = p.build_manifest()
    assert first == second
    core = {
        key: value for key, value in first.items() if key != "manifest_hash"
    }
    assert first["manifest_hash"] == p.canonical_hash(core)
    p.validate_manifest(first)


def test_frozen_dependencies_and_exact_headers_validate() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.OPERATIONS) == p.OPERATIONS_HEADER_SHA256
    assert p.sha256_csv_header(p.DETAILS) == p.DETAILS_HEADER_SHA256
    assert p.sha256_csv_header(p.SLCS_CLOCK) == p.SLCS_CLOCK_HEADER_SHA256
    assert set(p.OPERATIONS_ALLOWLIST).issubset(p.csv_header(p.OPERATIONS))
    assert set(p.DETAILS_ALLOWLIST).issubset(p.csv_header(p.DETAILS))
    assert set(p.SLCS_USECOLS).issubset(p.csv_header(p.SLCS_CLOCK))


def test_preregistration_opens_no_incidence_comparator_or_outcome() -> None:
    payload = p.build_manifest()
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert all(
        value in (0, False)
        for value in payload["evidence_boundary"].values()
    )
    assert payload["economic_rllm_sequence"][
        "economic_evaluator_authorized"
    ] is False


def test_source_contract_uses_only_frozen_allowlists() -> None:
    payload = p.build_manifest()
    operations = payload["source_contracts"]["operations"]["read_csv"]
    details = payload["source_contracts"]["details"]["read_csv"]
    assert operations == {
        "usecols": list(p.OPERATIONS_ALLOWLIST),
        "dtype": "string",
        "keep_default_na": False,
        "na_filter": False,
    }
    assert details == {
        "usecols": list(p.DETAILS_ALLOWLIST),
        "dtype": "string",
        "keep_default_na": False,
        "na_filter": False,
    }
    source = p.SCRIPT_PATH.read_text(encoding="utf-8")
    assert "pandas" not in source
    assert ".read_csv(" not in source


def test_batch_component_and_execution_contract_are_frozen() -> None:
    payload = p.build_manifest()
    assert payload["batch_contract"]["key"] == "exact available_at_utc"
    assert payload["batch_contract"]["atom_identity"] == [
        "operation_id",
        "cusip",
    ]
    assert payload["batch_contract"][
        "invalid_batch_breaks_continuity"
    ] is True
    assert payload["components"]["order"] == list(p.COMPONENT_ORDER)
    assert payload["transition"]["fracture"] == "count(UP) >= 3"
    assert payload["transition"]["relief"] == "count(DOWN) >= 3"
    assert payload["execution"]["hold_bars"] == 576
    assert payload["execution"]["hold_minutes"] == 2_880
    assert payload["execution"]["global_nonoverlap_before_split"] is True


def test_controls_and_support_gates_match_mechanism() -> None:
    payload = p.build_manifest()
    assert payload["controls"]["order"] == list(p.CONTROL_ORDER)
    assert payload["controls"]["random_side"]["input"] == (
        "SCAF-48|<primary_signal_id>|RANDOM_SIDE"
    )
    assert payload["source_support_gate"]["coverage"] == {
        "train_complete_batches_min": 700,
        "selection_complete_batches_min": 220,
        "train_valid_transitions_min": 690,
        "selection_valid_transitions_min": 215,
        "each_split_raw_consensus_share_min": 0.10,
        "each_split_raw_consensus_share_max": 0.65,
    }
    assert payload["source_support_gate"]["train"]["events_min"] == 120
    assert payload["source_support_gate"]["selection"]["events_min"] == 35
    assert payload["composition_gate"]["flat_component_counts_as_disagreement"]


def test_slcs_novelty_contract_is_exact_and_source_gated() -> None:
    novelty = p.build_manifest()["novelty_contract"]
    assert novelty["opens_only_after_source_and_composition_pass"] is True
    comparator = novelty["comparator"]
    assert comparator["groups"] == list(p.SLCS_GROUPS)
    assert comparator["read_csv"]["usecols"] == list(p.SLCS_USECOLS)
    assert comparator["minimum_contained_rows_each"] == 20
    assert novelty["thresholds_each_group"] == {
        "exact_entry_jaccard_max": 0.25,
        "one_new_york_calendar_day_jaccard_max": 0.50,
        "same_entry_same_side_reproduction_max": 0.30,
        "absolute_signed_occupancy_pearson_max": 0.35,
    }


def test_manifest_rejects_any_opened_evidence() -> None:
    payload = p.build_manifest()
    payload["evidence_boundary"]["operation_value_rows_read"] = 1
    with pytest.raises(RuntimeError, match="differs from code"):
        p.validate_manifest(payload)


def test_dependency_hash_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = p.sha256_file

    def drift(path: str | Path) -> str:
        if str(path) == str(p.MECHANISM):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(p, "sha256_file", drift)
    with pytest.raises(RuntimeError, match="frozen dependency changed"):
        p.validate_frozen_dependencies()


def test_write_once_is_repository_confined_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    (tmp_path / "results").mkdir()
    output = Path("results/preregistration.json")
    payload = p.build_manifest()
    assert p.write_once(output, payload) == "created"
    assert p.write_once(output, payload) == "verified_existing"
    written = tmp_path / output
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed == payload
    written.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="noncanonical"):
        p.write_once(output, payload)
    with pytest.raises(RuntimeError, match="repository-relative"):
        p.write_once("../escape.json", payload)
