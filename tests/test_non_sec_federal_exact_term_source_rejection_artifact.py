from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training import preregister_non_sec_federal_exact_term_source as nfet


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/non_sec_federal_exact_term_source_rejection_2026-07-22.json"
ARCHIVE = ROOT / "data/non_sec_federal_exact_term_membership_2020_2023"
EXPECTED_RESULT_HASH = (
    "65b388afd59a2fd42ff1dcbad3a29badcd46d147591e8f01e4f3c5b20e21aa75"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return _sha256(raw)


def _load_receipt_object(receipt: dict[str, Any]) -> bytes:
    path = ROOT / receipt["object_path"]
    encoded = path.read_bytes()
    assert len(encoded) == receipt["gzip_bytes"]
    assert _sha256(encoded) == receipt["gzip_sha256"]
    raw = gzip.decompress(encoded)
    assert len(raw) == receipt["raw_bytes"]
    assert _sha256(raw) == receipt["raw_sha256"]
    return raw


def test_nfet_rejection_hash_and_frozen_evidence_chain() -> None:
    payload = json.loads(RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    assert payload["result_hash"] == EXPECTED_RESULT_HASH
    assert _canonical_hash(core) == EXPECTED_RESULT_HASH
    assert payload["decision"] == "REJECT_NO_REPAIR"
    evidence = payload["frozen_evidence"]
    for record in (
        evidence["source_protocol"],
        evidence["source_parser"],
        evidence["candidate_access_seal"],
        evidence["candidate_index"],
        evidence["membership_builder"],
    ):
        assert _sha256((ROOT / record["path"]).read_bytes()) == record["sha256"]


def test_nfet_rejection_replays_exact_positive_and_agency_failure() -> None:
    payload = json.loads(RESULT.read_text())
    failure = payload["frozen_failure"]
    receipts = failure["official_source_receipts"]
    raw = {name: _load_receipt_object(receipt) for name, receipt in receipts.items()}
    canonical = nfet.official_html_membership_view(raw["html"])
    assert len(canonical.encode()) == failure["canonical_text_bytes"] == 1_303_217
    assert _sha256(canonical.encode()) == failure["canonical_text_sha256"]
    assert (
        nfet.exact_term_matches(canonical)
        == failure["exact_matches"]
        == [
            {
                "pattern_id": "blockchain",
                "substring": "blockchain",
                "span_start": 524_717,
                "span_end_exclusive": 524_727,
            }
        ]
    )
    candidate = failure["candidate"]
    mods = nfet.parse_govinfo_mods(raw["mods"])
    assert list(mods["agency_names"]) == failure["govinfo_agency_names"]
    detail = json.loads(raw["detail"])
    assert detail["agencies"][1] == {"raw_name": "Office of the Secretary"}
    assert "slug" not in detail["agencies"][1]
    assert raw["pdf"].startswith(b"%PDF-")
    with pytest.raises(ValueError, match="detail JSON has a malformed agency slug"):
        nfet.reconcile_positive_identity(
            candidate["document_number"],
            candidate["publication_date"],
            mods,
            detail,
        )


def test_nfet_rejection_keeps_only_failure_evidence_and_outcomes_closed() -> None:
    payload = json.loads(RESULT.read_text())
    attempt = payload["source_attempt"]
    assert attempt["candidate_envelope_rows"] == 486
    assert attempt["completed_candidates_before_failure"] == 355
    assert attempt["failing_candidate_position_one_based"] == 356
    assert attempt["partial_network_responses"] == {
        "count": 1267,
        "by_kind": {
            "detail_json": 356,
            "html_raw": 356,
            "mods_xml": 356,
            "pdf": 199,
        },
    }
    assert attempt["full_build_completed"] is False
    assert attempt["selected_source_written"] is False
    assert attempt["source_manifest_written"] is False
    assert attempt["support_result_written"] is False
    boundary = payload["opened_boundaries"]
    assert boundary["full_exact_member_incidence_opened"] is False
    assert boundary["source_support_calculated"] is False
    assert boundary["novelty_calculated"] is False
    assert boundary["market_clocks_opened"] is False
    assert boundary["outcomes_opened"] is False
    assert boundary["returns_or_pnl_calculated"] is False
    assert boundary["semantic_model_opened"] is False
    retained = {
        path.relative_to(ROOT).as_posix()
        for path in ARCHIVE.glob("objects/*/*.gz")
        if path.is_file()
    }
    expected = {
        receipt["object_path"]
        for receipt in payload["frozen_failure"]["official_source_receipts"].values()
    }
    assert retained == expected
    assert not (ARCHIVE / "resume_state.json").exists()
    assert not (
        ROOT / "data/non_sec_federal_exact_term_source_2020_2023.jsonl.gz"
    ).exists()
    assert not (
        ROOT / "results/non_sec_federal_exact_term_source_manifest_2026-07-20.json"
    ).exists()
    assert not (
        ROOT / "results/non_sec_federal_exact_term_source_support_2026-07-20.json"
    ).exists()
