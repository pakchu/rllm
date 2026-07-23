from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_ofr_dvp_maturity_stock_flow_handoff as dmsh


ARTIFACT = Path(
    "results/ofr_dvp_maturity_stock_flow_handoff_preregistration_2026-07-23.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_is_hash_bound_replayable_and_immutable() -> None:
    report = payload()
    assert sha256(ARTIFACT) == (
        "3351845c3a18e6e9af30cdc58b7ce6f71dc9d55674b3fa6e330e521ee1e4c475"
    )
    assert report["manifest_hash"] == (
        "8c958e00649db244aff147c82c9ed1b9631ca4d015bdf8ce8085383e0619c678"
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == dmsh.canonical_hash(core)
    assert report["policy_hash"] == (
        "c911842ada7d100f1b98e28f9ca3c3eb6e03ba5b7055d172bd7f38596bef24c0"
    )
    assert report["policy_hash"] == dmsh.canonical_hash(report["policy"])
    dmsh.validate_preregistration(report, verify_sources=True)


def test_artifact_binds_exact_protocol_source_history_and_comparators() -> None:
    report = payload()
    assert report["verification_mode"] == "verified_hashes"
    assert report["artifact_eligible"] is True
    assert report["preregistration_source"]["sha256"] == (
        "f665d352934ffecc0475d8d6de41f7edec00397d09c83a5d99828dbd76f27c59"
    )
    assert report["mechanism_decision"]["sha256"] == (
        "82533ea9981015f57fcff5f5cf777668ad60329adec4acbc88441416b1234b92"
    )
    assert report["source_binding"]["observations_sha256"] == (
        "6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a"
    )
    assert report["source_binding"]["manifest_observation_rows"] == 77_369
    assert report["source_binding"]["manifest_series"] == 82
    assert len(report["comparator_bindings"]) == 14
    assert report["comparator_bindings"][-1]["name"] == (
        "ofr_repo_collateral_routing_efficiency_primary"
    )
    assert len(report["history_bindings"]) == 6


def test_artifact_keeps_candidate_comparators_and_outcomes_closed() -> None:
    report = payload()
    assert report["candidate_features_or_incidence_opened"] is False
    assert report["comparator_rows_opened_during_preregistration"] is False
    assert report["outcomes_opened"] is False
    assert report["performance_values_opened"] is False
    boundary = report["outcome_boundary"]
    assert boundary["source_observation_value_rows_read"] == 0
    assert boundary["source_metadata_definition_rows_read"] == 0
    assert boundary["comparator_value_rows_read"] == 0
    assert boundary["candidate_features_computed"] == 0
    assert boundary["candidate_incidence_derived"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
    assert boundary["network_calls"] == 0
    assert boundary["subprocess_calls"] == 0
