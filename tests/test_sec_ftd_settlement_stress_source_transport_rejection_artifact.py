from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPORT = Path(
    "results/"
    "sec_ftd_settlement_stress_source_transport_rejection_2026-07-24.json"
)
BOUNDARY = Path(
    "docs/sec-ftd-settlement-stress-source-axis-decision-2026-07-24.md"
)
REPORT_SHA256 = (
    "de962adb7996a0c2bb11b5d65dd52cce4f9508c0ae76efe81abf6ef79a4f73b2"
)
BOUNDARY_SHA256 = (
    "475e3616848a3e6c8914a8ed55ed71d99efdbec1b678e402f8633f565fd6cdc4"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sftd_transport_rejection_is_hash_bound_and_terminal() -> None:
    assert _sha256(BOUNDARY) == BOUNDARY_SHA256
    assert _sha256(REPORT) == REPORT_SHA256

    report = json.loads(REPORT.read_bytes())
    assert report["protocol_version"] == "sftd_transport_preflight_v1"
    assert report["source_id"] == "SFTD-v1"
    assert report["source_boundary"] == {
        "path": str(BOUNDARY),
        "sha256": BOUNDARY_SHA256,
        "commit": "6ac40914002955ef7d5323ae75636cfee0d97e53",
    }
    assert report["status"] == "REJECT"
    assert report["state"] == "TERMINAL_REJECT"
    assert report["failure"] == {
        "stage": "index_transport_preflight",
        "exception_type": "HttpStatusError",
        "message": "transport failed",
    }
    assert report["response"] == {
        "http_status": 403,
        "content_type": "text/html",
        "content_length_header": 1925,
        "bytes_read": 1925,
    }

    without_manifest = dict(report)
    manifest = without_manifest.pop("manifest_sha256")
    canonical = json.dumps(
        without_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    expected = hashlib.sha256(
        b"SFTD-v1\0transport-rejection\0" + canonical
    ).hexdigest()
    assert manifest == expected


def test_sftd_rejection_exposes_no_source_or_outcome_values() -> None:
    report = json.loads(REPORT.read_bytes())
    assert report["request"]["redirect_count"] == 0
    assert report["gates"] == {
        "committed_source_boundary": True,
        "direct_index_transport": False,
        "archive_transport": False,
        "source_schema": False,
        "causal_availability": False,
        "three_year_support": False,
        "forbidden_access": True,
    }
    assert all(value is False for value in report["forbidden_access"].values())
    assert report["bindings"] == {
        "builder_path": None,
        "mechanism_document_path": None,
        "preregistration_document_path": None,
        "source_audit_report_path": None,
    }
    assert report["next_action"] == "select_new_source_axis"
