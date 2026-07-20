from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT = Path(
    "results/stablecoin_denominator_dislocation_preregistration_2026-07-20.json"
)
EXPECTED_FILE_SHA256 = (
    "0db69de6f278d37e8bff09b5843127dc326b75eec2722d9b46d3882434c19280"
)
EXPECTED_MANIFEST_HASH = (
    "88974fadce5f1094d503674b7b0ee9e2dd74d8cf3808b72b71bbbebebc1fbcfa"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_sddr_preregistration_is_frozen_and_outcome_blind() -> None:
    assert _sha256(ARTIFACT) == EXPECTED_FILE_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["manifest_hash"] == _canonical_hash(core)
    assert payload["candidate"] == "SDDR-12"
    assert payload["outcomes_opened"] is False
    assert payload["outcome_sources_opened"] is False
    assert payload["post_2023_source_rows_opened"] is False


def test_sddr_source_and_comparators_are_hash_bound() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    source = payload["source"]
    assert _sha256(Path(source["panel"])) == source["panel_sha256"]
    assert _sha256(Path(source["manifest"])) == source["manifest_sha256"]
    comparators = payload["support_comparators"]
    assert _sha256(Path(comparators["clock"])) == comparators["clock_sha256"]
    assert _sha256(Path(comparators["support"])) == comparators["support_sha256"]
    assert _sha256(Path(payload["preregistration_source"])) == payload[
        "preregistration_source_sha256"
    ]
    assert _sha256(Path(payload["preregistration_document"])) == payload[
        "preregistration_document_sha256"
    ]


def test_sddr_clock_and_stopping_contract_are_unambiguous() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["policy"]["config"]["lookback_hours"] == 720
    assert payload["policy"]["config"]["minimum_history_hours"] == 672
    assert payload["policy"]["entry"] == "BTCUSDT USD-M perpetual at h+1h+5m open"
    assert payload["policy"]["exit"] == (
        "scheduled open after exactly 12 five-minute bars"
    )
    assert payload["support_gate"]["stop_if_failed"] is True
    assert payload["later_outcome_contract"]["stop_on_first_failure"] is True
    assert (
        payload["later_outcome_contract"]["evaluator_must_be_committed_before_outcome"]
        is True
    )
