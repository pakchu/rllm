from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/lamb21_source_support_rejection_2026-07-25.json"
DOCUMENT = ROOT / "docs/lamb21-source-support-retirement-2026-07-25.md"
ARTIFACT_SHA256 = (
    "394922d8cc737ce2b6c415b687b85b1a3a1ebcdfbde6c08ea1bb96ffa47fa9cd"
)
MANIFEST_HASH = (
    "67a03c8d36ccf9b57459c975606cca936525565a88d8c5ebdc20636bd821e40e"
)


def _payload() -> dict[str, object]:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == ARTIFACT_SHA256
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_lamb21_rejection_artifact_is_canonical_and_terminal() -> None:
    payload = _payload()
    manifest_hash = payload.pop("manifest_hash")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert manifest_hash == MANIFEST_HASH
    assert hashlib.sha256(encoded).hexdigest() == MANIFEST_HASH
    assert payload["decision"] == "fail"
    assert payload["failure_action"] == "retire_lamb21_unchanged_before_rewards"
    assert payload["authorized_next_stage"] is None
    assert payload["profitability_result_exists"] is False
    assert payload["repair_under_same_identity_allowed"] is False


def test_lamb21_rejection_is_gate_01_clock_evidence_only() -> None:
    payload = _payload()
    failure = payload["failure"]
    assert isinstance(failure, dict)

    assert failure["gate"] == "gate_01"
    assert failure["check"] == "cascade_transaction_clock"
    assert failure["violating_rows"] == 15_295
    assert failure["last_timestamp_exactly_at_exclusive_bar_end"] == 15_295
    assert failure["last_timestamp_after_bar_end"] == 0
    assert sum(failure["violating_rows_by_utc_year"].values()) == 15_295

    counters = payload["evidence_counters"]
    assert isinstance(counters, dict)
    assert counters["source_value_rows_decoded"] == 843_347
    assert counters["joint_state_rows_built"] == 0
    assert all(
        value == 0
        for name, value in counters.items()
        if name not in {"source_value_rows_decoded", "joint_state_rows_built"}
    )


def test_lamb21_retirement_document_forbids_posthoc_clock_repair() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")

    assert "RETIRE LAMB-21 unchanged" in text
    assert "15,295" in text
    assert "Do not change `< bar_end` to `<= bar_end`" in text
    assert "no profitability" in text
