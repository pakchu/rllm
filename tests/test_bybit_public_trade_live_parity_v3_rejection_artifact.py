from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.probe_bybit_public_trade_sequence_source import canonical_hash


RESULT = Path(
    "results/bybit_public_trade_live_parity_capture_v3_reject_2026-07-23.json"
)
DOCUMENT = Path("docs/bybit-public-trade-live-parity-v3-rejection-2026-07-23.md")
RESULT_FILE_SHA256 = (
    "493cde97193c3c837cda4ca2101c7d2068cab8972836fdb511a68b2b7b9fc5d5"
)
RESULT_MANIFEST_HASH = (
    "cb0cade10439c4ba083db528a65b4ffc04a55b46a6ffe7e29928f4a9642c8661"
)
DOCUMENT_SHA256 = (
    "f894abd2fc55c02a75e4e82c076b3ffc67ced0e63f1f774e1ce6be0671ed0ed6"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact() -> dict[str, Any]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_v3_rejection_is_file_manifest_and_document_hash_bound() -> None:
    assert _sha256(RESULT) == RESULT_FILE_SHA256
    assert _sha256(DOCUMENT) == DOCUMENT_SHA256
    payload = _artifact()
    observed = payload.pop("manifest_hash_without_self")
    assert observed == RESULT_MANIFEST_HASH
    assert canonical_hash(payload) == RESULT_MANIFEST_HASH


def test_v3_rejects_source_window_overflow_after_clock_pass() -> None:
    payload = _artifact()
    assert payload["decision"] == "SOURCE_PARITY_REJECT_REST_WINDOW_OVERFLOW"
    assert payload["authoritative_for_source_parity"] is True
    assert payload["bsea_24_authorized"] is False
    clock = payload["clock_validation"]
    assert clock["preflight_passed"] is True
    assert clock["capture_clock_contract_passed"] is True
    assert clock["capture_utc_reversals"] == 0
    parity = payload["parity"]
    assert parity["common_field_mismatches"] == 0
    assert parity["missing_eligible_ws_ids_in_rest"] == 5_816
    assert parity["missing_interior_rest_ids_in_ws"] == 0
    assert parity["adjacent_rest_zero_overlap_pairs"] == 2


def test_v3_burst_diagnosis_is_source_only_and_closes_outcomes() -> None:
    payload = _artifact()
    diagnosis = payload["source_only_burst_diagnosis"]
    assert diagnosis["rest_recent_trade_limit"] == 1_000
    assert diagnosis["maximum_ws_trades_per_server_match_second"] == 5_451
    assert diagnosis["server_match_seconds_above_rest_limit"] == 2
    assert diagnosis["missing_ids_per_affected_second_descending"] == [4_659, 1_157]
    assert payload["outcome_boundary"] == {
        "archive_rows_opened": False,
        "bsea_clock_built": False,
        "candidate_incidence_opened": False,
        "binance_comparator_opened": False,
        "market_outcomes_opened": False,
        "returns_or_pnl_opened": False,
    }
