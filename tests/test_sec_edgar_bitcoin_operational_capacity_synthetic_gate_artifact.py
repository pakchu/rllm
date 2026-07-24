from __future__ import annotations

import json
from pathlib import Path

from training.preregister_sec_edgar_bitcoin_operational_capacity import (
    canonical_hash,
    sha256_file,
)


ARTIFACT = Path(
    "results/sec_edgar_bitcoin_operational_capacity_"
    "synthetic_gate_2026-07-24.json"
)
ARTIFACT_SHA256 = "12cc1bf61d1dc422a1f4cf7500cd4f53f3e765bd00d65dafc54fd630026fa8d9"
MANIFEST_HASH = "7e9d769347b7c5b1dccc7550164141df21b759f23931989d6b95cc7a9a99d605"


def _payload() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_committed_synthetic_result_is_exact_and_self_consistent() -> None:
    assert sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = _payload()
    assert payload["manifest_hash"] == MANIFEST_HASH
    manifest = dict(payload)
    observed = manifest.pop("manifest_hash")
    assert observed == canonical_hash(manifest)
    assert sha256_file(payload["runner"]["path"]) == payload["runner"]["sha256"]
    assert (
        payload["preregistration"]["sha256"]
        == "5ce80db5875e282a1d173489bb910026b53f073f9c783c5140ee99fe3d72605d"
    )


def test_exact_adapter_is_immutably_retired_on_frozen_gate_failure() -> None:
    payload = _payload()
    assert payload["training"]["rows"] == 512
    assert payload["training"]["optimizer_steps"] == 64
    assert payload["calibration"]["selected_step"] == 32
    gate = payload["final"]["gate"]
    assert gate["passed"] is False
    assert gate["adversarial"]["exact_count"] == 190
    assert gate["adversarial"]["exact_share"] == 190 / 192
    assert gate["adversarial"]["per_class"]["MIXED"]["exact_count"] == 46
    assert gate["swap_gate"]["both_exact_pairs"] == 63
    assert gate["swap_gate"]["invariant_pairs"] == 64
    assert gate["checks"]["adversarial_mixed_exact"] is False
    assert gate["checks"]["swap_exact"] is False
    assert gate["checks"]["inference_peak_allocated"] is False
    assert gate["checks"]["inference_peak_reserved"] is False
    assert payload["final"]["selected_adapter"] is None
    assert payload["decision"] == {
        "2024_or_later_authorized": False,
        "economic_evaluation_authorized": False,
        "historical_body_transport_authorized": False,
        "historical_semantic_execution_authorized": False,
        "next_step": (
            "retire EBOC-72 unchanged before all historical bodies and outcomes"
        ),
        "novelty_evaluation_authorized": False,
        "repair_authorized": False,
        "status": "retired_synthetic_failure",
        "synthetic_gate_passed": False,
    }


def test_failure_opened_no_historical_source_market_or_outcome() -> None:
    payload = _payload()
    assert payload["outcome_boundary"] == {
        "2024_or_later_source_rows_read": 0,
        "btc_market_rows_read": 0,
        "comparator_clock_fields_read": 0,
        "comparator_rows_parsed": 0,
        "filing_bodies_opened": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "historical_semantic_model_calls": 0,
        "historical_windows_created": 0,
        "return_or_pnl_fields_read": 0,
    }
