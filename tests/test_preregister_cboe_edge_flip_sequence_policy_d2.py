from __future__ import annotations

import json

import pytest

from training import preregister_cboe_edge_flip_sequence_policy as d1
from training import preregister_cboe_edge_flip_sequence_policy_d2 as d2


FAKE_PRODUCER = {
    "path": d2.PRODUCER_SCRIPT,
    "commit": "a" * 40,
    "sha256": "b" * 64,
}


def synthetic_manifest() -> dict:
    core = d2.manifest_core(FAKE_PRODUCER)
    return {**core, "manifest_hash": d2.canonical_hash(core)}


def test_runtime_authority_uses_absolute_hash_bound_git() -> None:
    authority = d2.validate_runtime_authority()
    assert authority["path"] == "/usr/bin/git"
    assert authority["sha256"] == d2.GIT_EXECUTABLE_SHA256
    assert authority["path_component"] == "/usr/bin"
    assert authority["version"].startswith("git version ")
    assert d2._git_output("--version").startswith("git version ")


def test_runtime_rejects_missing_usr_bin_even_with_absolute_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/bin")
    with pytest.raises(RuntimeError, match="lacks exact /usr/bin"):
        d2.validate_runtime_authority()
    assert d2._git_output("--version").startswith("git version ")


def test_frozen_dependencies_bind_d1_terminal_and_pure_engine() -> None:
    authority = d2.validate_frozen_dependencies()
    assert authority["d1_engine"] == {
        "path": d2.D1_ENGINE,
        "commit": d2.D1_ENGINE_COMMIT,
        "sha256": d2.D1_ENGINE_SHA256,
    }
    assert authority["d1_rejection"]["result_hash"] == (
        d2.D1_REJECTION_RESULT_HASH
    )
    assert authority["d1_rejection"]["source_values_opened"] is False
    assert authority["d1_rejection"]["outcomes_opened"] is False


def test_scientific_mechanism_is_exact_d1_inheritance() -> None:
    mechanism = d2.scientific_mechanism()
    d1_payload = json.loads(
        d1.repository_path(d1.DEFAULT_OUTPUT).read_text()
    )
    expected = {
        key: d1_payload[key] for key in d2.D1_SCIENTIFIC_KEYS
    }
    assert mechanism["d1_scientific_contract"] == expected
    assert mechanism["d1_scientific_contract_hash"] == d2.canonical_hash(
        expected
    )
    assert expected["relation_language"]["ordered_edges"] == list(
        d1.EDGE_NAMES
    )
    assert expected["relation_language"]["edge_formulas"] == dict(
        d1.EDGE_FORMULAS
    )
    assert expected["sequence_language"]["action_space"] == [
        "TARGET_LONG",
        "TARGET_FLAT",
        "TARGET_SHORT",
    ]
    assert expected["clock"]["scheduled_exit"] == (
        "entry plus 288 five-minute bars"
    )
    assert expected["controls"]["ordered_ids"] == list(d1.CONTROL_IDS)
    assert expected["support_gates"]["minimum_total_intervals"] == 920


def test_synthetic_manifest_is_source_and_outcome_blind() -> None:
    payload = synthetic_manifest()
    d2.validate_manifest(payload)
    assert payload["policy_id"] == "CEFS-D2"
    assert payload["source_rows_parsed"] == 0
    assert payload["source_values_opened"] is False
    assert payload["outcomes_opened"] is False
    assert all(value == 0 for value in payload["forbidden_access"].values())
    source_contract = payload["scientific_mechanism"][
        "d1_scientific_contract"
    ]["sources"]
    assert all(
        panel["path"].startswith(("data/", "results/"))
        for panel in source_contract.values()
    )
    assert payload["terminal_actions"] == d2.TERMINAL_ACTIONS


def test_manifest_tamper_is_rejected() -> None:
    payload = synthetic_manifest()
    payload["source_values_opened"] = True
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = d2.canonical_hash(core)
    with pytest.raises(RuntimeError, match="invariant mismatch"):
        d2.validate_manifest(payload)


def test_preregistration_output_path_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="path is frozen"):
        d2.write_once("results/not-cefs-d2.json", synthetic_manifest())
