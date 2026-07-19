from __future__ import annotations

import copy

import pytest

from training import preregister_cross_collateral_inventory_pressure_absorption as p


def test_manifest_is_source_only_and_hash_stable() -> None:
    manifest = p.build_manifest()
    p.validate_manifest(manifest)
    assert manifest["outcomes_opened"] is False
    assert manifest["causal_feature_contract"]["price_signal_columns"] == []
    assert manifest["policy"]["policy_id"] == "CIPA-48"
    assert manifest["policy"]["hold_bars"] == 48


def test_rule_is_disjoint_from_ccpr_concordance() -> None:
    manifest = p.build_manifest()
    assert "sign(R[t])=-sign(T[t])" in manifest["causal_feature_contract"]["setup"]
    assert "disjoint" in manifest["novelty_boundary"]["not_a_ccpr_repair"]


def test_policy_mutation_is_rejected() -> None:
    manifest = p.build_manifest()
    changed = copy.deepcopy(manifest)
    changed["policy"]["oi_rotation_rank_min"] = 0.85
    changed["manifest_hash"] = p.canonical_hash(
        {key: value for key, value in changed.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="policy changed"):
        p.validate_manifest(changed, verify_sources=False)


def test_outcome_opening_is_rejected() -> None:
    manifest = p.build_manifest()
    changed = copy.deepcopy(manifest)
    changed["outcomes_opened"] = True
    changed["manifest_hash"] = p.canonical_hash(
        {key: value for key, value in changed.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="opened outcomes"):
        p.validate_manifest(changed, verify_sources=False)
