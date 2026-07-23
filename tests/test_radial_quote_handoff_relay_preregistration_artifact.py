from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_radial_quote_handoff_relay as rqhr


ARTIFACT = Path(
    "results/radial_quote_handoff_relay_preregistration_2026-07-23.json"
)
ARTIFACT_SHA256 = (
    "cecc1e7d77913e5cb0c0e2f6d16775289b18e9f31e6dbfddcb24a605c522da77"
)
MANIFEST_HASH = (
    "d31a726bcbd1721e26a14ee65140b1e32f2a335236ab85a23c8109f6f2449398"
)
POLICY_HASH = (
    "c20c4b9c78ae51b6d3ff4d1fb83c970f2a57b7ba1e0a9d477154cf0ace65d671"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "78e2cdf3044a837dca7a7dbcab74aecd6edc273e14357785d505cea4b990ca69"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_preregistration_artifact_is_hash_bound_and_replayable() -> None:
    report = payload()
    assert sha256(ARTIFACT) == ARTIFACT_SHA256
    assert report["manifest_hash"] == MANIFEST_HASH
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == rqhr.canonical_hash(core)
    assert report["policy_hash"] == POLICY_HASH
    assert report["policy_hash"] == rqhr.canonical_hash(report["policy"])
    assert report["preregistration_source"]["sha256"] == (
        PREREGISTRATION_SOURCE_SHA256
    )
    rqhr.validate_preregistration(report, verify_sources=True)


def test_artifact_binds_exact_source_history_and_comparator_cohort() -> None:
    report = payload()
    assert report["artifact_eligible"] is True
    assert report["verification_mode"] == "verified_hashes_without_value_parsing"
    assert report["mechanism_decision"]["sha256"] == (
        rqhr.MECHANISM_DECISION_SHA256
    )
    assert report["source_binding"]["panel_sha256"] == rqhr.SOURCE_PANEL_SHA256
    assert report["source_binding"]["manifest_sha256"] == (
        rqhr.SOURCE_MANIFEST_SHA256
    )
    assert report["source_binding"]["builder_sha256"] == (
        rqhr.SOURCE_BUILDER_SHA256
    )
    assert report["source_binding"]["manifest_metadata_parsed"] is False
    assert len(report["history_bindings"]) == 5
    assert [row["group"] for row in report["comparator_bindings"]] == [
        "ccbvfr:primary",
        "pdf10:primary",
        "crrc:primary",
    ]
    assert [row["expected_raw_rows"] for row in report["comparator_bindings"]] == [
        144,
        591,
        156,
    ]
    assert all(
        row["common_window_policy_sha256"] == rqhr.COMMON_WINDOW_POLICY_SHA256
        for row in report["comparator_bindings"]
    )


def test_artifact_keeps_candidate_comparators_and_outcomes_closed() -> None:
    report = payload()
    assert report["source_family_values_previously_opened"] is True
    assert report["rqhr_net_path_efficiency_values_opened"] is False
    assert report["rqhr_features_arms_confirmations_or_events_opened"] is False
    assert report["synthetic_nulls_run"] is False
    assert report["comparator_rows_opened_during_preregistration"] is False
    assert report["outcomes_opened"] is False
    assert report["performance_values_opened"] is False

    boundary = report["outcome_boundary"]
    assert boundary["source_manifest_metadata_parsed"] is False
    assert boundary["source_value_rows_read"] == 0
    assert boundary["rqhr_columns_read"] == 0
    assert boundary["rqhr_features_computed"] == 0
    assert boundary["synthetic_nulls_run"] == 0
    assert boundary["rqhr_arms_or_terminals_derived"] == 0
    assert boundary["rqhr_events_derived"] == 0
    assert boundary["comparator_value_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
