from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT = Path(
    "results/usdt_collateral_breadth_relay_preregistration_2026-07-20.json"
)
EXPECTED_FILE_SHA256 = (
    "19758c9093261c4f0e3e226546fc5541a7ef89d832202162e95f54e4c28bb9cb"
)
EXPECTED_MANIFEST_HASH = (
    "483884a349219ba3888253b39c33eae34c0a500400beef59bc0b8fe56ad55a3d"
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


def test_ucbr_preregistration_is_frozen_before_incidence_or_outcomes() -> None:
    assert _sha256(ARTIFACT) == EXPECTED_FILE_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["manifest_hash"] == _canonical_hash(core)
    assert payload["candidate"] == "UCBR-12"
    assert payload["outcomes_opened"] is False
    assert payload["outcome_sources_opened"] is False
    assert payload["post_2023_source_rows_opened"] is False
    assert payload["real_event_incidence_opened"] is False


def test_ucbr_source_and_novelty_comparators_are_hash_bound() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    source = payload["source"]
    assert _sha256(Path(source["panel"])) == source["panel_sha256"]
    assert _sha256(Path(source["manifest"])) == source["manifest_sha256"]
    assert _sha256(Path(payload["preregistration_source"])) == payload[
        "preregistration_source_sha256"
    ]
    assert _sha256(Path(payload["preregistration_document"])) == payload[
        "preregistration_document_sha256"
    ]
    assert {item["candidate"] for item in payload["support_comparators"]} == {
        "SDDR-12",
        "SQFD-6",
    }
    for item in payload["support_comparators"]:
        assert _sha256(Path(item["clock"])) == item["clock_sha256"]
        assert _sha256(Path(item["support"])) == item["support_sha256"]


def test_ucbr_singleton_and_stop_contract_are_exact() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    config = payload["policy"]["config"]
    assert config["lookback_hours"] == 720
    assert config["minimum_history_hours"] == 672
    assert config["z_threshold"] == 1.25
    assert config["minimum_agreeing_issuers"] == 3
    assert config["hold_bars"] == 144
    assert payload["support_gate"]["stop_if_failed"] is True
    assert payload["later_outcome_contract"]["stop_on_first_failure"] is True
    assert (
        payload["later_outcome_contract"]["evaluator_must_be_committed_before_outcome"]
        is True
    )
