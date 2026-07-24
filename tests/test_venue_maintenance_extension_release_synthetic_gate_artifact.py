from __future__ import annotations

import json
from pathlib import Path

from training.preregister_venue_maintenance_extension_release import (
    canonical_hash,
    sha256_file,
)


ARTIFACT = Path(
    "results/venue_maintenance_extension_release_"
    "synthetic_gate_2026-07-24.json"
)
ARTIFACT_SHA256 = "a6c94ae35400e20043a877459381321d26b9b30c8532e8f62a4528676505427b"
MANIFEST_HASH = "f36195ada72d50aee82333ad52b142a33f7a4a90060d94604c6077831987b5a5"


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
        == "de5cc97ddd7b1c3bfb155a1d3e3cd11e501e43148047c6e8fd9b4d48100e5809"
    )


def test_exact_adapter_is_immutably_retired_on_frozen_gate_failure() -> None:
    payload = _payload()
    assert payload["training"]["rows"] == 384
    assert payload["training"]["optimizer_steps"] == 48
    assert payload["calibration"]["selected_step"] == 36
    gate = payload["final"]["gate"]
    assert gate["passed"] is False
    assert gate["adversarial"]["exact_count"] == 133
    assert gate["adversarial"]["per_class"]["CONTRADICTORY"]["exact_count"] == 44
    assert gate["adversarial"]["per_class"]["UNSUPPORTED"]["exact_count"] == 41
    assert gate["swap_gate"]["invariant_pairs"] == 48
    assert gate["swap_gate"]["both_exact_pairs"] == 46
    assert gate["guarded"] == {
        "by_split": {
            "adversarial": {"exact_count": 7, "model_calls": 0, "rows": 7},
            "swaps": {"exact_count": 6, "model_calls": 0, "rows": 6},
        },
        "exact_count": 13,
        "model_calls": 0,
        "rows": 13,
    }
    assert gate["checks"]["adversarial_contradictory_exact"] is False
    assert gate["checks"]["adversarial_unsupported_exact"] is False
    assert gate["checks"]["swaps_unsupported_exact"] is False
    assert gate["checks"]["swap_exact"] is False
    assert gate["checks"]["training_peak_reserved"] is False
    assert payload["final"]["selected_adapter"] is None
    assert payload["decision"] == {
        "2024_or_later_authorized": False,
        "economic_evaluation_authorized": False,
        "historical_semantic_execution_authorized": False,
        "historical_source_transport_authorized": False,
        "next_step": "retire VMER-2 unchanged before all historical source and outcomes",
        "novelty_evaluation_authorized": False,
        "repair_authorized": False,
        "status": "retired_synthetic_failure",
        "synthetic_gate_passed": False,
    }


def test_failure_opened_no_historical_source_market_or_outcome() -> None:
    payload = _payload()
    assert payload["outcome_boundary"] == {
        "2024_or_later_candidate_rows_read": 0,
        "btc_market_rows_read": 0,
        "candidate_detail_objects_opened": 0,
        "candidate_history_rows_opened": 0,
        "candidate_update_bodies_opened": 0,
        "comparator_clock_fields_read": 0,
        "comparator_rows_parsed": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "historical_semantic_model_calls": 0,
        "return_or_pnl_fields_read": 0,
    }


def test_rejected_checkpoint_manifests_are_preserved_without_selected_copy() -> None:
    payload = _payload()
    checkpoints = payload["checkpoints"]
    assert set(checkpoints["manifests"]) == {"12", "24", "36", "48"}
    assert checkpoints["retained_bytes_before_result_commit"] == 43_004_032
    assert all(
        manifest["bytes"] == 10_751_008
        for manifest in checkpoints["manifests"].values()
    )
    assert payload["final"]["selected_adapter"] is None
