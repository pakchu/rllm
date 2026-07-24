from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import verify_aave_v3_borrow_rate_ledger as verifier


REPORT = Path(
    "results/aave_v3_borrow_rate_ledger_source_parity_2026-07-24.json"
)
SENTINEL = Path(
    "results/.aave_v3_borrow_rate_ledger_source_parity_2026-07-24.started"
)
REPORT_SHA256 = (
    "d1df809e9fad9b24ebc4db96c7dfabbae3d502a7e398d779b5fbfb6e893aec11"
)
SENTINEL_SHA256 = (
    "dea61474a3c3a2cc22c08ad6efc5d296f69e7cb3b66fe5652b87ac451aba6081"
)
VERIFIER_COMMIT = "5fad22834e605b95dd752815a95ec853a41dcd7d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_a_rejection_artifacts_are_immutable_and_hash_bound() -> None:
    assert _sha256(REPORT) == REPORT_SHA256
    assert _sha256(SENTINEL) == SENTINEL_SHA256

    report = json.loads(REPORT.read_bytes())
    sentinel = json.loads(SENTINEL.read_bytes())

    assert sentinel == {
        "protocol_version": verifier.PROTOCOL_VERSION,
        "source_boundary_sha256": verifier.BOUNDARY_SHA256,
        "started_at_utc": "2026-07-24T11:49:45Z",
        "verifier_commit": VERIFIER_COMMIT,
    }
    assert report["protocol_version"] == verifier.PROTOCOL_VERSION
    assert report["source_boundary_sha256"] == verifier.BOUNDARY_SHA256
    assert report["verifier_commit"] == VERIFIER_COMMIT
    assert report["started_at_utc"] == sentinel["started_at_utc"]
    assert report["terminal_at_utc"] == "2026-07-24T11:49:50Z"
    assert report["status"] == "REJECT"
    assert report["state"] == "TERMINAL_REJECT"
    assert report["failure"] == {
        "stage": "source_read",
        "exception_type": "TransportError",
        "message": "transport failed",
    }

    without_manifest = dict(report)
    manifest = without_manifest.pop("manifest_sha256")
    expected_manifest = hashlib.sha256(
        b"AV3BRL-v1\0terminal-report\0"
        + verifier.canonical_json_bytes(without_manifest)
    ).hexdigest()
    assert manifest == expected_manifest


def test_stage_a_rejection_publishes_no_source_or_outcome_values() -> None:
    report = json.loads(REPORT.read_bytes())
    assert report["window_streams"] is None
    assert report["full_history_sha256"] is None
    assert report["ledger_gzip_sha256"] is None
    assert report["ledger_gzip_bytes"] is None
    assert report["providers"]["canonical_streams_equal"] is False
    assert report["forbidden_access"] == verifier.FORBIDDEN_ACCESS
    assert all(value is False for value in report["forbidden_access"].values())
    assert report["gates"] == {
        "committed_protocol": True,
        "clean_head": True,
        "disk": True,
        "pinned_sources": False,
        "chain_identity": False,
        "boundary_headers": False,
        "subdivision_redundancy": False,
        "historical_window": False,
        "recent_window": False,
        "event_header_audit": False,
        "forbidden_access": True,
    }
    assert not Path(
        "results/.aave_v3_borrow_rate_ledger_source_parity_2026-07-24.json.tmp"
    ).exists()
