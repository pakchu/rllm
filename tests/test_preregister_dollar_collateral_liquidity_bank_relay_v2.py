from __future__ import annotations

import json

import pytest

from training import preregister_dollar_collateral_liquidity_bank_relay as v1
from training import preregister_dollar_collateral_liquidity_bank_relay_v2 as p


def test_v2_is_outcome_blind_and_preserves_v1_base() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["protocol_version"].endswith("_v2")
    assert payload["policy"]["policy_id"] == "DCLB-864"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert all(value == 0 for value in payload["evidence_boundary"].values())
    base = payload["supersedes_without_mutating"]["base_preregistration"]
    assert base["sha256"] == p.BASE_PREREGISTRATION_SHA256
    assert base["manifest_hash"] == p.BASE_MANIFEST_HASH


def test_only_effective_mechanism_change_is_control_balance_token() -> None:
    base = v1.build_manifest()
    payload = p.build_manifest()
    relation = payload["source_algebra"]["macro"][
        "control_only_balanced_relation"
    ]
    assert relation["token"] == "MACRO_BALANCED_OPPOSITION"
    assert relation["allowed_controls"] == [
        "h41_only",
        "rrp_interval_only",
        "h8_only",
    ]
    assert relation["primary_eligible"] is False
    assert relation["primary_rllm_token_allowed"] is False
    assert payload["policy"] == base["policy"]
    assert payload["source_support_gate"] == base["source_support_gate"]
    assert payload["novelty_contract"] == base["novelty_contract"]
    assert payload["rllm_boundary"] == base["rllm_boundary"]


def test_amendment_and_executable_v1_builder_are_hash_bound() -> None:
    p.validate_frozen_dependencies()
    dependencies = p.frozen_dependencies()
    assert dependencies[p.AMENDMENT_DOCUMENT] == p.AMENDMENT_DOCUMENT_SHA256
    assert dependencies[p.BASE_BUILDER] == p.BASE_BUILDER_SHA256
    assert (
        dependencies[p.BASE_PREREGISTRATION]
        == p.BASE_PREREGISTRATION_SHA256
    )
    assert len(dependencies) == len(v1.frozen_dependencies()) + 3


def test_manifest_is_deterministic_and_self_rehash_drift_fails() -> None:
    assert p.build_manifest() == p.build_manifest()
    payload = p.build_manifest()
    payload["source_algebra"]["macro"]["control_only_balanced_relation"][
        "primary_eligible"
    ] = True
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)


def test_write_once_is_canonical(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    monkeypatch.setattr(v1, "REPOSITORY_ROOT", tmp_path)
    output = "freeze.json"
    assert p.write_once(output, p.build_manifest()) == "created"
    assert (tmp_path / output).read_text(encoding="utf-8") == (
        p._canonical_manifest_text()
    )
    assert p.write_once(output, p.build_manifest()) == "verified_existing"
    drift = json.loads((tmp_path / output).read_text(encoding="utf-8"))
    drift["policy"]["hold_bars"] = 1
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(drift)
