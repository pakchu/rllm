from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from training import cross_venue_volatility_shape_handoff as mechanism
from training import evaluate_cross_venue_volatility_shape_handoff_source_support as s
from training import preregister_cross_venue_volatility_shape_handoff as prereg


ROOT = Path(__file__).resolve().parents[1]
CLAIM = (
    ROOT
    / "results"
    / (
        "cross_venue_volatility_shape_handoff_source_support_"
        "attempt_claim_2026-07-30.json"
    )
)
BUNDLE = (
    ROOT
    / "results"
    / "cross_venue_volatility_shape_handoff_source_support_2026-07-30"
)
FAILURE = (
    ROOT
    / "results"
    / (
        "cross_venue_volatility_shape_handoff_source_support_"
        "failure_2026-07-30.json"
    )
)

CLAIM_SHA256 = (
    "5b9a6c1a8aa29172485b95f8c3717138a6926de5061a38f2ae3aa0da438dde08"
)
CLAIM_HASH = (
    "dc5b6e10bdeab08048004608ed968b4d71e98e9dc32d1033c3fe9a3f03c5aefb"
)
SUPPORT_MANIFEST_HASH = (
    "2e32d1a75dd249827c86f5445091f63f5fca23a95c3ccdcf922aa3bd0b2a2c16"
)
BUNDLE_MANIFEST_HASH = (
    "1e8096f6a3cc733d29bc4785169ed8017f71c4fbcf1e62611c4fa8a96f3ddb1e"
)
EVALUATOR_COMMIT = "c48daaa052c873743491d3e273bc4d9f3a90d45a"
EVALUATOR_SEAL_HASH = (
    "cc06902f6a0b647fe45243ac8c9c867acf0fc3b4655999c9d398e895dda44951"
)


def _load(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _fraction(raw: dict[str, Any]) -> Fraction:
    return Fraction(int(raw["numerator"]), int(raw["denominator"]))


def test_terminal_claim_is_hash_valid_and_precedes_protected_reads() -> None:
    assert CLAIM.is_file()
    assert not FAILURE.exists()
    assert hashlib.sha256(CLAIM.read_bytes()).hexdigest() == CLAIM_SHA256
    claim = _load(CLAIM)
    core = {key: value for key, value in claim.items() if key != "claim_hash"}
    assert claim["claim_hash"] == CLAIM_HASH
    assert claim["claim_hash"] == prereg.canonical_hash(core)
    assert claim["repository"]["commit"] == EVALUATOR_COMMIT
    assert claim["repository"]["canonical_remote_commit"] == EVALUATOR_COMMIT
    assert claim["evaluator_seal_hash"] == EVALUATOR_SEAL_HASH
    assert claim["authoritative_attempts_allowed"] == 1
    assert claim["retry_resume_fallback_or_repair_after_claim"] is False
    assert claim["verification_replay"] is False
    assert claim["protected_reads_at_claim"] == {
        "btc_execution_rows_opened": 0,
        "bvol_rows_decoded": 0,
        "candidate_incidence_opened": False,
        "comparator_rows_opened": 0,
        "dvol_rows_decoded": 0,
        "funding_rows_opened": 0,
        "gross9_rows_opened": 0,
        "outcomes_opened": False,
    }
    assert claim["source_transport"] == {
        "compressed_snapshots_retained_before_claim": True,
        "hash_header_and_snapshot_use_same_compressed_bytes": True,
        "source_paths_reopened_after_claim": False,
        "value_rows_decompressed_only_after_claim": True,
    }


def test_bundle_and_every_clock_artifact_are_byte_valid() -> None:
    report_path = BUNDLE / "report.json"
    assert report_path.is_file()
    report = _load(report_path)
    bundle_core = {
        key: value
        for key, value in report.items()
        if key != "bundle_manifest_hash"
    }
    assert report["bundle_manifest_hash"] == BUNDLE_MANIFEST_HASH
    assert report["bundle_manifest_hash"] == prereg.canonical_hash(bundle_core)
    assert report["attempt_claim"] == {
        "claim_hash": CLAIM_HASH,
        "path": str(s.ATTEMPT_CLAIM),
        "sha256": CLAIM_SHA256,
    }

    expected_controls = {
        *mechanism.OWN_CLOCKS,
        *mechanism.PARENT_SET_CONTROLS,
    }
    assert set(report["clock_artifacts"]) == expected_controls
    for control, artifact in report["clock_artifacts"].items():
        path = BUNDLE / artifact["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
        assert len(raw) == artifact["bytes"]
        assert int.from_bytes(raw[4:8], "little") == 0
        assert raw[3] & 0x08 == 0
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
            text = handle.read().decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == artifact["rows"]
        assert artifact["header"] == s.CLOCK_HEADER
        assert all(row["policy_id"] == mechanism.CANDIDATE_ID for row in rows)
        assert all(row["control"] == control for row in rows)


def test_only_preregistered_structural_distinctness_gate_failed() -> None:
    report = _load(BUNDLE / "report.json")
    assert report["support_manifest_hash"] == SUPPORT_MANIFEST_HASH
    later_binding_keys = {
        "attempt_claim",
        "authoritative_attempt",
        "bundle_manifest_hash",
        "clock_artifacts",
        "evaluator_seal_hash",
        "preregistered_protocol_seal_hash",
        "preregistration",
        "repository_commit",
        "retry_resume_fallback_or_repair_used",
        "support_manifest_hash",
    }
    support_core = {
        key: value
        for key, value in report.items()
        if key not in later_binding_keys
    }
    assert prereg.canonical_hash(support_core) == SUPPORT_MANIFEST_HASH
    assert report["passed"] is False
    assert report["failure_action"] == (
        "retire exact CVVH-432 unchanged before novelty"
    )
    failed = {
        key for key, passed in report["checks"].items() if passed is not True
    }
    assert failed == {"all_four_independent_controls_distinct"}
    assert report["support_statistics"]["selection"]["total"] == 249
    assert report["support_statistics"]["future25"]["total"] == 125
    assert report["support_statistics"]["future26"]["total"] == 59
    assert report["maximum_accepted_entry_gap_seconds"] == 23 * 86_400 + 64_800
    assert report["maximum_same_side_run"] == 10
    assert report["selection_prefix_append_invariance"]["byte_identical"] is True

    controls = report["structural_control_distinctness"]
    assert controls[mechanism.DERIBIT_LED]["passed"] is True
    for control in (
        mechanism.BODY_LEAD_ONLY,
        mechanism.RANGE_LEAD_ONLY,
        mechanism.STALE_DERIBIT,
    ):
        assert controls[control]["passed"] is False
        assert (
            _fraction(
                controls[control]["one_to_one_24h"][
                    "maximum_matched_share"
                ]
            )
            >= Fraction(19, 20)
        )
        assert (
            _fraction(controls[control]["exact_entry_jaccard"])
            < Fraction(9, 10)
        )


def test_no_novelty_gross9_market_funding_or_outcome_rows_opened() -> None:
    report = _load(BUNDLE / "report.json")
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["comparator_rows_opened"] == 0
    assert report["gross9_rows_opened"] == 0
    assert report["btc_execution_rows_opened"] == 0
    assert report["funding_rows_opened"] == 0
    assert report["authoritative_attempt"] == 1
    assert report["retry_resume_fallback_or_repair_used"] is False
    diagnostics = report["source_diagnostics"]
    assert diagnostics["decoded_from_preclaim_compressed_snapshots"] is True
    assert diagnostics["source_paths_reopened_after_claim"] == 0
    assert diagnostics["fills_imputations_tolerance_or_nearest"] == 0
    assert diagnostics["join_missing_dvol_rows"] == 0
    assert diagnostics["bvol"] == {
        "first_close_time_utc": "2023-06-20T01:00:00Z",
        "invalid_rows": 2797,
        "last_close_time_utc": "2026-07-01T00:00:00Z",
        "rows": 26568,
        "valid_rows": 23771,
    }
    assert diagnostics["dvol"]["rows"] == 26569
