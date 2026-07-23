from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.capture_bybit_public_trade_live_parity import canonical_hash


ARTIFACT = Path(
    "results/bybit_public_trade_live_parity_capture_v1_invalid_2026-07-23.json"
)
FILE_SHA256 = (
    "d7eb515b4571a767d5efebc18ea179ebdcdd0f345c425bf2840755807f32dcbf"
)
MANIFEST_HASH = (
    "78c0b14d552b090f923e2ca445e31184fed60f46a203f6fca4873871c741c4cb"
)


def test_v1_capture_invalidation_is_hash_bound_and_non_authoritative() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == FILE_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    observed = payload.pop("manifest_hash_without_self")
    assert observed == MANIFEST_HASH
    assert canonical_hash(payload) == MANIFEST_HASH
    assert payload["decision"] == "CAPTURE_INVALID_IMPLEMENTATION_FAIL_OPEN"
    assert payload["authoritative_for_source_parity"] is False
    assert payload["original_capture"]["reported_decision_authoritative"] is False
    assert payload["v1_live_rows_reusable_for_parity"] is False


def test_v1_invalidation_records_clock_failure_without_opening_outcomes() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    audit = payload["independent_clock_audit"]
    assert audit["utc_reversal_count"] == 11
    assert audit["clock_contract_passed"] is False
    assert audit["monotonic_elapsed_seconds"] > audit["utc_elapsed_seconds"]
    assert payload["outcome_boundary"] == {
        "bsea_clock_built": False,
        "candidate_incidence_opened": False,
        "binance_comparator_opened": False,
        "market_outcomes_opened": False,
        "returns_or_pnl_opened": False,
    }
