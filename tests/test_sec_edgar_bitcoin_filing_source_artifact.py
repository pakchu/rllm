from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from training import audit_sec_edgar_bitcoin_filing_source as audit


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz"
REPORT = ROOT / "results/sec_edgar_bitcoin_8k_6k_source_audit_2026-07-21.json"
SOURCE_SHA256 = "c8489dfe9b4ac25da8bea7653115e5b58a44fa897f2815eaf68bad354e10c6ce"
REPORT_SHA256 = "c1e11d1f5089378ac787fdb2a80474f0feec33d5fb2296fb0c3014d6f1fafec1"
MANIFEST_HASH = "b4234f71b559a6b98e4056491f3b726191e9a89c2c0bec1e549249d93840f575"
AUDITOR_SHA256 = "79e741cf3711f9ab1e041806de8cbf8daa3e909a321e5bff1e439d39fd6fe7b5"


def _report() -> dict[str, Any]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_source_and_report_are_hash_bound() -> None:
    report = _report()
    assert audit.sha256_file(SOURCE) == SOURCE_SHA256
    assert audit.sha256_file(REPORT) == REPORT_SHA256
    assert audit.sha256_file(audit.AUDITOR_SOURCE) == AUDITOR_SHA256
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["auditor"]["sha256"] == AUDITOR_SHA256
    assert report["source_artifact"]["sha256"] == SOURCE_SHA256


def test_source_clock_is_sorted_bounded_and_complete() -> None:
    report = _report()
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    keys = [
        (row["acceptance_datetime"], row["accession"], row["document"])
        for row in rows
    ]
    assert keys == sorted(keys)
    assert len(rows) == report["source_artifact"]["rows"] == 3_496
    assert audit.canonical_hash(rows) == report["source_artifact"][
        "canonical_rows_sha256"
    ]
    assert max(row["acceptance_datetime"] for row in rows) < "2024-01-01"
    assert all(row["form"] in audit.ALLOWED_FORMS for row in rows)


def test_source_support_passes_without_authorizing_outcomes() -> None:
    report = _report()
    assert report["metrics"]["accessions"] == 2_543
    assert report["metrics"]["emittable_accessions"] == 2_493
    assert report["metrics"]["ciks"] == 308
    assert report["metrics"]["event_days"] == 992
    assert all(report["official_documentation"]["checks"].values())
    assert all(report["transport"]["checks"].values())
    assert all(report["support_gates"].values())
    assert report["decision"]["source_contract_passed"] is True
    assert report["decision"]["candidate_preregistration_authorized"] is True
    assert report["decision"]["semantic_model_execution_authorized"] is False
    assert report["decision"]["economic_evaluation_authorized"] is False
    assert report["decision"]["2024_or_later_source_authorized"] is False


def test_source_audit_preserves_outcome_boundary() -> None:
    boundary = _report()["outcome_boundary"]
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["candidate_signal_rows_created"] == 0
    assert boundary["economic_outcomes_opened"] is False
    assert boundary["clean_room_claimed"] is False
