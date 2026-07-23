from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_ofr_dvp_maturity_stock_flow_handoff as dmsh


def test_policy_freezes_handoff_causality_controls_and_economics() -> None:
    policy = dmsh.policy_payload()
    assert policy["candidate"] == "DMSH-168-SOURCE-REUSE"
    assert tuple(policy["source"]["required_series"]) == dmsh.REQUIRED_SERIES
    assert policy["features"]["maturity_flow_gap"].startswith(
        "flow_overnight_share-stock_overnight_share"
    )
    assert policy["normalization"]["history_complete_dates"] == 252
    assert policy["state_machine"]["confirmation_window_complete_rows"] == 10
    assert "cancels before confirmation" in policy["state_machine"][
        "contradiction_priority"
    ]
    assert policy["execution"]["hold_elapsed_hours"] == 168
    assert policy["source_support_gates"]["train_total_minimum"] == 40
    assert policy["source_support_gates"]["selection_total_minimum"] == 18
    assert policy["controls"]["placebo_economic_evaluation_forbidden"] is True
    assert len(policy["controls"]["scheduled_causal"]) == 8
    assert len(policy["novelty"]["comparators"]) == 14
    assert policy["economics"]["base_cagr_to_strict_mdd_minimum"] == 3.0
    assert policy["economics"]["weekly_signflip_draws"] == 100_000
    assert policy["mutable_parameters"] == []


def test_exact_comparator_cohort_and_hashes_are_frozen() -> None:
    assert len(dmsh.COMPARATOR_SPECS) == 14
    assert dmsh.COMPARATOR_SPECS[-1]["name"] == (
        "ofr_repo_collateral_routing_efficiency_primary"
    )
    assert dmsh.COMPARATOR_SPECS[-1]["sha256"] == (
        "cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826"
    )
    assert all(isinstance(row["path"], Path) for row in dmsh.COMPARATOR_SPECS)
    assert all(len(row["sha256"]) == 64 for row in dmsh.COMPARATOR_SPECS)


def test_static_preregistration_is_ineligible_and_opens_nothing() -> None:
    payload = dmsh.build_preregistration(verify_sources=False)
    dmsh.validate_preregistration(payload, verify_sources=False)
    assert payload["verification_mode"] == "static_test_fixture"
    assert payload["artifact_eligible"] is False
    assert payload["candidate_features_or_incidence_opened"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False
    assert payload["outcomes_opened"] is False
    assert payload["source_binding"]["manifest_metadata_parsed"] is False
    assert payload["outcome_boundary"] == dmsh.STATIC_TEST_OUTCOME_BOUNDARY
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )


def test_real_preregistration_binds_source_comparators_history_and_decision() -> None:
    payload = dmsh.build_preregistration(verify_sources=True)
    dmsh.validate_preregistration(payload, verify_sources=True)
    assert payload["artifact_eligible"] is True
    assert payload["source_binding"]["manifest_observation_rows"] == 77_369
    assert payload["source_binding"]["manifest_series"] == 82
    assert len(payload["comparator_bindings"]) == 14
    assert len(payload["history_bindings"]) == 6
    assert payload["mechanism_decision"]["sha256"] == (
        dmsh.MECHANISM_DECISION_SHA256
    )
    assert payload["common_window_policy"]["sha256"] == (
        dmsh.COMMON_WINDOW_POLICY_SHA256
    )


def test_policy_and_boundary_tampering_fail_closed() -> None:
    payload = dmsh.build_preregistration(verify_sources=False)
    payload["policy"]["state_machine"]["confirmation_window_complete_rows"] = 11
    with pytest.raises(RuntimeError, match="policy drift"):
        dmsh.validate_preregistration(payload, verify_sources=False)

    payload = dmsh.build_preregistration(verify_sources=False)
    payload["candidate_features_or_incidence_opened"] = True
    with pytest.raises(RuntimeError, match="boundary opened"):
        dmsh.validate_preregistration(payload, verify_sources=False)


@pytest.mark.parametrize(
    ("field", "opened"),
    [
        ("btc_market_rows_read", 1),
        ("funding_rows_read", 1),
        ("return_rows_read", 1),
        ("pnl_cagr_mdd_opened", True),
        ("candidate_incidence_opened", True),
        ("candidate_features_computed", ["maturity_flow_gap"]),
        ("final_source_rows_read", 1),
    ],
)
def test_source_manifest_boundary_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, opened: object
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

    monkeypatch.setattr(dmsh, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(dmsh, "OBSERVATIONS", observations)
    monkeypatch.setattr(dmsh, "METADATA", metadata)
    monkeypatch.setattr(dmsh, "SOURCE_MANIFEST", manifest_path)
    monkeypatch.setattr(dmsh, "SOURCE_AUDIT", audit)
    monkeypatch.setattr(dmsh, "OBSERVATIONS_SHA256", dmsh.sha256_file(observations))
    monkeypatch.setattr(dmsh, "METADATA_SHA256", dmsh.sha256_file(metadata))
    monkeypatch.setattr(dmsh, "SOURCE_AUDIT_SHA256", dmsh.sha256_file(audit))
    manifest["observations"]["sha256"] = dmsh.OBSERVATIONS_SHA256
    manifest["metadata"]["sha256"] = dmsh.METADATA_SHA256
    (tmp_path / manifest_path).write_text(json.dumps(manifest))
    monkeypatch.setattr(
        dmsh, "SOURCE_MANIFEST_SHA256", dmsh.sha256_file(manifest_path)
    )
    monkeypatch.setattr(
        dmsh, "SOURCE_CANONICAL_MANIFEST_HASH", "canonical-manifest"
    )

    with pytest.raises(RuntimeError, match=f"boundary opened: {field}"):
        dmsh._source_binding()


def test_write_is_immutable_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dmsh, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(dmsh, "MECHANISM_DECISION", Path("mechanism.md"))
    monkeypatch.setattr(dmsh, "COMMON_WINDOW_POLICY", Path("window.md"))
    monkeypatch.setattr(dmsh, "SCRIPT_PATH", Path("preregister.py"))
    for name in ("mechanism.md", "window.md", "preregister.py"):
        (tmp_path / name).write_text(f"{name}\n")
    monkeypatch.setattr(
        dmsh, "MECHANISM_DECISION_SHA256", dmsh.sha256_file("mechanism.md")
    )
    monkeypatch.setattr(
        dmsh, "COMMON_WINDOW_POLICY_SHA256", dmsh.sha256_file("window.md")
    )
    monkeypatch.setattr(dmsh, "_source_binding", dmsh._static_source_binding)
    original_bindings = dmsh._bindings
    monkeypatch.setattr(
        dmsh,
        "_bindings",
        lambda specs, *, history, verify: original_bindings(
            specs, history=history, verify=False
        ),
    )

    cfg = dmsh.Config(output="out/prereg.json")
    first, status = dmsh.write_preregistration(cfg)
    assert status == "created"
    second, status = dmsh.write_preregistration(cfg)
    assert status == "verified_existing"
    assert first == second

    path = tmp_path / cfg.output
    changed = json.loads(path.read_text())
    changed["manifest_hash"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="existing DMSH preregistration differs"):
        dmsh.write_preregistration(cfg)

    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("sentinel\n")
    with pytest.raises(FileExistsError):
        dmsh._atomic_write(sentinel, {"replacement": True})
    assert sentinel.read_text() == "sentinel\n"


def test_existing_mismatched_output_is_never_json_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dmsh, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(dmsh, "MECHANISM_DECISION", Path("mechanism.md"))
    monkeypatch.setattr(dmsh, "COMMON_WINDOW_POLICY", Path("window.md"))
    monkeypatch.setattr(dmsh, "SCRIPT_PATH", Path("preregister.py"))
    for name in ("mechanism.md", "window.md", "preregister.py"):
        (tmp_path / name).write_text(f"{name}\n")
    monkeypatch.setattr(
        dmsh, "MECHANISM_DECISION_SHA256", dmsh.sha256_file("mechanism.md")
    )
    monkeypatch.setattr(
        dmsh, "COMMON_WINDOW_POLICY_SHA256", dmsh.sha256_file("window.md")
    )
    monkeypatch.setattr(dmsh, "_source_binding", dmsh._static_source_binding)
    original_bindings = dmsh._bindings
    monkeypatch.setattr(
        dmsh,
        "_bindings",
        lambda specs, *, history, verify: original_bindings(
            specs, history=history, verify=False
        ),
    )
    output = tmp_path / "out/existing.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"forbidden_outcome": 123}\n')
    monkeypatch.setattr(
        dmsh.json,
        "loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing output was parsed")
        ),
    )

    with pytest.raises(RuntimeError, match="existing DMSH preregistration differs"):
        dmsh.write_preregistration(dmsh.Config(output="out/existing.json"))
    assert output.read_text() == '{"forbidden_outcome": 123}\n'
