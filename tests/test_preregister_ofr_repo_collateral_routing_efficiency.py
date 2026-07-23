from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_ofr_repo_collateral_routing_efficiency as rcre


def test_policy_freezes_invariant_signed_interaction_and_gates() -> None:
    policy = rcre.policy_payload()
    assert policy["candidate"] == "RCRE-72-SOURCE-REUSE"
    assert tuple(policy["source"]["required_series"]) == rcre.REQUIRED_SERIES
    assert policy["features"]["routing_pressure"] == "quantity_gap*rate_gap"
    assert "preserves routing_pressure" in policy["features"]["venue_swap_identity"]
    assert policy["normalization"]["history_complete_dates"] == 252
    assert policy["execution"]["hold_elapsed_hours"] == 72
    assert policy["source_support_gates"]["train_total_minimum"] == 45
    assert policy["source_support_gates"]["selection_total_minimum"] == 20
    assert policy["source_support_gates"]["train_each_quadrant_minimum_share"] == 0.10
    assert policy["source_support_gates"]["selection_maximum_quadrant_share"] == 0.60
    assert policy["source_controls"]["label_pair_controls_can_falsify_economics"] is False
    assert len(policy["novelty"]["comparators"]) == 13
    assert policy["common_window_policy"]["sha256"] == rcre.COMMON_WINDOW_POLICY_SHA256
    assert policy["mutable_parameters"] == []


def test_comparator_cohort_is_locally_frozen_without_transitive_registry() -> None:
    assert [row["name"] for row in rcre.COMPARATOR_SPECS] == [
        "overnight_rrp_flow_release_all_controls",
        "overnight_rrp_participant_breadth_all_controls",
        "federal_liquidity_component_concordance_all_groups",
        "daily_treasury_fiscal_flow_breadth_primary",
        "daily_treasury_fiscal_flow_breadth_controls",
        "sofr_rate_dislocation_primary",
        "bank_deposit_secured_repo_concordance_all_clocks",
        "fed_h8_deposit_migration_primary",
        "soma_lending_collateral_scarcity_primary",
        "cross_domain_liquidity_transmission_all_clocks",
        "live_portfolio_pure_clocks",
        "ofr_repo_venue_fragmentation_consensus_primary",
        "ofr_repo_mix_shock_resolution_race_primary",
    ]
    assert all(isinstance(row["path"], Path) for row in rcre.COMPARATOR_SPECS)
    assert all(len(row["sha256"]) == 64 for row in rcre.COMPARATOR_SPECS)


def test_static_preregistration_is_explicitly_noneligible_and_value_blind() -> None:
    payload = rcre.build_preregistration(verify_sources=False)
    rcre.validate_preregistration(payload, verify_sources=False)
    assert payload["verification_mode"] == "static_test_fixture"
    assert payload["artifact_eligible"] is False
    assert payload["source_family_values_previously_opened"] is True
    assert payload["signed_features_or_rcre_incidence_opened"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False
    assert payload["outcomes_opened"] is False
    assert payload["source_binding"]["manifest_metadata_parsed"] is False
    assert payload["outcome_boundary"] == rcre.STATIC_TEST_OUTCOME_BOUNDARY
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )


def test_real_source_policy_comparator_and_history_hashes_are_bound() -> None:
    payload = rcre.build_preregistration(verify_sources=True)
    rcre.validate_preregistration(payload, verify_sources=True)
    assert payload["artifact_eligible"] is True
    assert payload["source_binding"]["manifest_observation_rows"] == 77_369
    assert len(payload["comparator_bindings"]) == 13
    assert payload["comparator_bindings"][-1]["name"] == (
        "ofr_repo_mix_shock_resolution_race_primary"
    )
    assert all(
        row["common_window_policy_sha256"] == rcre.COMMON_WINDOW_POLICY_SHA256
        for row in payload["comparator_bindings"]
    )
    assert payload["common_window_policy"]["sha256"] == (
        rcre.COMMON_WINDOW_POLICY_SHA256
    )
    assert payload["mechanism_decision"]["sha256"] == (
        rcre.MECHANISM_DECISION_SHA256
    )


def test_policy_or_boundary_tampering_fails_closed() -> None:
    payload = rcre.build_preregistration(verify_sources=False)
    payload["policy"]["state"]["friction"] = "changed"
    with pytest.raises(RuntimeError, match="policy drift"):
        rcre.validate_preregistration(payload, verify_sources=False)

    payload = rcre.build_preregistration(verify_sources=False)
    payload["signed_features_or_rcre_incidence_opened"] = True
    with pytest.raises(RuntimeError, match="boundary opened"):
        rcre.validate_preregistration(payload, verify_sources=False)


@pytest.mark.parametrize(
    ("field", "opened"),
    [
        ("btc_market_rows_read", 1),
        ("funding_rows_read", 1),
        ("return_rows_read", 1),
        ("pnl_cagr_mdd_opened", True),
        ("candidate_incidence_opened", True),
        ("candidate_features_computed", ["routing_pressure"]),
        ("final_source_rows_read", 1),
    ],
)
def test_source_manifest_outcome_boundary_fails_closed(
    tmp_path: Path, monkeypatch, field: str, opened: object
) -> None:
    observations = Path("observations.csv.gz")
    metadata = Path("metadata.json.gz")
    manifest_path = Path("manifest.json")
    audit = Path("audit.md")
    for path in (observations, metadata, audit):
        (tmp_path / path).write_bytes(f"{path}\n".encode())

    manifest = {
        "manifest_hash": "canonical-manifest",
        "observations": {"sha256": "", "rows": 77_369},
        "metadata": {"sha256": "", "series": 82},
        "source_checks": {"complete": True},
        "research_boundary": {
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "candidate_incidence_opened": False,
            "candidate_features_computed": [],
            "final_source_rows_read": 0,
        },
    }
    manifest["research_boundary"][field] = opened

    monkeypatch.setattr(rcre, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(rcre, "OBSERVATIONS", observations)
    monkeypatch.setattr(rcre, "METADATA", metadata)
    monkeypatch.setattr(rcre, "SOURCE_MANIFEST", manifest_path)
    monkeypatch.setattr(rcre, "SOURCE_AUDIT", audit)
    monkeypatch.setattr(
        rcre, "OBSERVATIONS_SHA256", rcre.sha256_file(observations)
    )
    monkeypatch.setattr(rcre, "METADATA_SHA256", rcre.sha256_file(metadata))
    monkeypatch.setattr(rcre, "SOURCE_AUDIT_SHA256", rcre.sha256_file(audit))
    manifest["observations"]["sha256"] = rcre.OBSERVATIONS_SHA256
    manifest["metadata"]["sha256"] = rcre.METADATA_SHA256
    (tmp_path / manifest_path).write_text(json.dumps(manifest))
    monkeypatch.setattr(
        rcre, "SOURCE_MANIFEST_SHA256", rcre.sha256_file(manifest_path)
    )
    monkeypatch.setattr(
        rcre, "SOURCE_CANONICAL_MANIFEST_HASH", "canonical-manifest"
    )

    with pytest.raises(RuntimeError, match=f"boundary opened: {field}"):
        rcre._source_binding()


def test_write_is_immutable_and_no_clobber(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rcre, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(rcre, "MECHANISM_DECISION", Path("mechanism.md"))
    monkeypatch.setattr(rcre, "COMMON_WINDOW_POLICY", Path("window.md"))
    monkeypatch.setattr(rcre, "SCRIPT_PATH", Path("preregister.py"))
    for name in ("mechanism.md", "window.md", "preregister.py"):
        (tmp_path / name).write_text(f"{name}\n")
    monkeypatch.setattr(
        rcre, "MECHANISM_DECISION_SHA256", rcre.sha256_file("mechanism.md")
    )
    monkeypatch.setattr(
        rcre, "COMMON_WINDOW_POLICY_SHA256", rcre.sha256_file("window.md")
    )
    monkeypatch.setattr(rcre, "_source_binding", rcre._static_source_binding)
    original_bindings = rcre._bindings
    monkeypatch.setattr(
        rcre,
        "_bindings",
        lambda specs, *, history, verify: original_bindings(
            specs, history=history, verify=False
        ),
    )

    cfg = rcre.Config(output="out/prereg.json")
    first, status = rcre.write_preregistration(cfg)
    assert status == "created"
    second, status = rcre.write_preregistration(cfg)
    assert status == "verified_existing"
    assert first == second

    path = tmp_path / cfg.output
    changed = json.loads(path.read_text())
    changed["manifest_hash"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        rcre.write_preregistration(cfg)

    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("sentinel\n")
    with pytest.raises(FileExistsError):
        rcre._atomic_write(sentinel, {"replacement": True})
    assert sentinel.read_text() == "sentinel\n"
