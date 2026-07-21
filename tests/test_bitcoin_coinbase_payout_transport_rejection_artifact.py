from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training import audit_bitcoin_coinbase_payout_transport as audit


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/bitcoin_coinbase_payout_transport_rejection_2026-07-21.json"
ARTIFACT_SHA256 = "c00bf8193a012b01a7843d27afc636cd9a205ddcecc6717367a700c7d8b8f99e"
MANIFEST_HASH = "6c749c30996fbc8a3793b4abbba9d1e806665d6ca23c5b2d699e520dd67f5491"
AUDITOR_SHA256 = "a47c5a3eff3413fbbe52e4c6af7e9b7ac90dea8cdb0703dc65bf831b79396029"


def _report() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_rejection_artifact_is_hash_bound() -> None:
    report = _report()
    assert audit.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    assert audit.sha256_file(audit.AUDITOR_SOURCE) == AUDITOR_SHA256
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["auditor"]["sha256"] == AUDITOR_SHA256


def test_missing_summary_field_retires_source_before_candidate() -> None:
    report = _report()
    assert report["probe"]["unique_blocks"] == 20
    assert report["probe"]["summary_complete_blocks"] == 19
    assert report["probe"]["missing_heights"] == [800_015]
    assert report["checks"]["missing_summary_is_transport_specific"] is True
    assert report["decision"]["status"] == "retired_before_full_source_build"
    assert report["decision"]["full_source_build_authorized"] is False
    assert report["decision"]["fallback_repair_authorized"] is False
    assert report["decision"]["candidate_authorized"] is False


def test_rejection_artifact_preserves_the_outcome_boundary() -> None:
    boundary = _report()["outcome_boundary"]
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_source_rows_read"] == 0
    assert boundary["candidate_signal_rows_created"] == 0
    assert boundary["economic_outcomes_opened"] is False
