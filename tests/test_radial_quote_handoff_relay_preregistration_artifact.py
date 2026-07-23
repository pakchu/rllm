from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_radial_quote_handoff_relay as rqhr


V1_ARTIFACT = Path(
    "results/radial_quote_handoff_relay_preregistration_2026-07-23.json"
)
V1_ARTIFACT_SHA256 = (
    "cecc1e7d77913e5cb0c0e2f6d16775289b18e9f31e6dbfddcb24a605c522da77"
)
V1_MANIFEST_HASH = (
    "d31a726bcbd1721e26a14ee65140b1e32f2a335236ab85a23c8109f6f2449398"
)
V1_POLICY_HASH = (
    "c20c4b9c78ae51b6d3ff4d1fb83c970f2a57b7ba1e0a9d477154cf0ace65d671"
)
V2_ARTIFACT = Path(
    "results/radial_quote_handoff_relay_preregistration_v2_2026-07-23.json"
)
V2_ARTIFACT_SHA256 = (
    "402c0b02cfea5932766f1892520860acd6beaeafb9e6948024d71e761c2d5f70"
)
V2_MANIFEST_HASH = (
    "888a2ee7eabf081004b43b861f5ba2000497a8d3320a929ff337d73187b34f89"
)
V2_POLICY_HASH = (
    "1ef1674c97df6e6b8cfce948f76c8afebe1f6f168a7796b69dad799f98288cb8"
)
V2_SOURCE_SHA256 = (
    "409a8c326682ed687d584635ff0d27e588e8b74129eb7b367d04ff81c5307a7b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v1_payload() -> dict:
    return json.loads(V1_ARTIFACT.read_text(encoding="utf-8"))


def v2_payload() -> dict:
    return json.loads(V2_ARTIFACT.read_text(encoding="utf-8"))


def test_v1_artifact_remains_immutable_and_self_consistent() -> None:
    report = v1_payload()
    assert sha256(V1_ARTIFACT) == V1_ARTIFACT_SHA256
    assert report["manifest_hash"] == V1_MANIFEST_HASH
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == rqhr.canonical_hash(core)
    assert report["policy_hash"] == V1_POLICY_HASH
    assert report["policy_hash"] == rqhr.canonical_hash(report["policy"])


def test_v1_artifact_opened_no_candidate_comparator_or_outcome_values() -> None:
    report = v1_payload()
    assert report["rqhr_net_path_efficiency_values_opened"] is False
    assert report["rqhr_features_arms_confirmations_or_events_opened"] is False
    assert report["synthetic_nulls_run"] is False
    assert report["comparator_rows_opened_during_preregistration"] is False
    assert report["outcomes_opened"] is False
    assert report["performance_values_opened"] is False
    boundary = report["outcome_boundary"]
    assert boundary["rqhr_columns_read"] == 0
    assert boundary["rqhr_events_derived"] == 0
    assert boundary["comparator_value_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False


def test_current_v2_builder_explicitly_supersedes_v1_before_incidence() -> None:
    payload = rqhr.build_preregistration(verify_sources=False)
    assert payload["protocol_version"].endswith("_v2")
    assert payload["supersedes"] == {
        "path": str(V1_ARTIFACT),
        "sha256": V1_ARTIFACT_SHA256,
        "before_rqhr_incidence": True,
        "reason": (
            "PDF-10 canonical clock is six fields, not the v1 two-field "
            "description"
        ),
    }
    assert rqhr.DEFAULT_OUTPUT != V1_ARTIFACT
    assert payload["rqhr_net_path_efficiency_values_opened"] is False
    assert payload["rqhr_features_arms_confirmations_or_events_opened"] is False
    assert payload["outcomes_opened"] is False


def test_v2_artifact_is_hash_bound_replayable_and_authoritative() -> None:
    report = v2_payload()
    assert sha256(V2_ARTIFACT) == V2_ARTIFACT_SHA256
    assert report["manifest_hash"] == V2_MANIFEST_HASH
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == rqhr.canonical_hash(core)
    assert report["policy_hash"] == V2_POLICY_HASH
    assert report["policy_hash"] == rqhr.canonical_hash(report["policy"])
    assert report["preregistration_source"]["sha256"] == V2_SOURCE_SHA256
    assert report["mechanism_decision"]["sha256"] == (
        rqhr.MECHANISM_DECISION_SHA256
    )
    rqhr.validate_preregistration(report, verify_sources=True)


def test_v2_artifact_corrects_pdf_projection_and_keeps_boundaries_closed() -> None:
    report = v2_payload()
    pdf = next(
        row
        for row in report["comparator_bindings"]
        if row["group"] == "pdf10:primary"
    )
    assert pdf["canonical_projection"] == (
        "ordered signal_position, entry_position, exit_position, numeric "
        "side, branch, and hold_bars"
    )
    assert pdf["producer"] == (
        "training/preregister_radial_liquidity_wavefront_cascade.py"
    )
    assert pdf["replay_source"] == (
        "training/preregister_cross_collateral_liquidity_credibility_fracture.py"
    )
    assert report["supersedes"]["sha256"] == V1_ARTIFACT_SHA256
    assert report["rqhr_net_path_efficiency_values_opened"] is False
    assert report["rqhr_features_arms_confirmations_or_events_opened"] is False
    assert report["synthetic_nulls_run"] is False
    assert report["comparator_rows_opened_during_preregistration"] is False
    assert report["outcomes_opened"] is False
