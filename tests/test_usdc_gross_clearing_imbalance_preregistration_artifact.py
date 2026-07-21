from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT = Path("results/usdc_gross_clearing_imbalance_preregistration_2026-07-22.json")
EXPECTED_FILE_SHA256 = (
    "7056eadfd5b347b8b9afbe06cbc2a33f832a2913dc3227891a2a8d211aaa454a"
)
EXPECTED_MANIFEST_HASH = (
    "61b6d60f8c2ef21b94b3343bc3cf2a5fd82366679ae9d768d846831b12829722"
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


def test_ugci_preregistration_is_byte_frozen_and_outcome_blind() -> None:
    assert _sha256(ARTIFACT) == EXPECTED_FILE_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["manifest_hash"] == _canonical_hash(core)
    assert payload["candidate"] == "UGCI-288"
    assert payload["source"]["rows_parsed_during_preregistration"] == 0
    assert not any(
        payload["outcome_boundary"][key]
        for key in (
            "outcomes_opened",
            "outcome_sources_opened",
            "post_2023_source_rows_opened",
            "btc_market_rows_read",
            "funding_rows_read",
            "future_return_rows_read",
            "return_or_pnl_fields_read",
            "comparator_rows_read",
            "network_calls",
            "subprocess_calls",
        )
    )


def test_ugci_preregistration_binds_every_source_and_comparator() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    source = payload["source"]
    assert _sha256(Path(source["csv"])) == source["csv_sha256"]
    assert _sha256(Path(source["manifest"])) == source["manifest_sha256"]
    assert (
        _sha256(Path(payload["preregistration_source"]))
        == payload["preregistration_source_sha256"]
    )
    assert (
        _sha256(Path(payload["preregistration_document"]))
        == payload["preregistration_document_sha256"]
    )
    for comparator in payload["novelty_comparators"]:
        assert _sha256(Path(comparator["path"])) == comparator["sha256"]


def test_ugci_policy_and_stopping_contract_are_unambiguous() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    config = payload["policy"]["config"]
    assert config["packet_hours"] == 6
    assert config["include_zero_event_packets"] is True
    assert config["gross_tail_quantile"] == 0.95
    assert config["minimum_imbalance_ratio"] == 0.60
    assert config["entry_delay_minutes"] == 10
    assert config["hold_bars"] * config["bar_minutes"] == 24 * 60
    assert payload["support_gate"]["stop_if_failed"] is True
    assert payload["one_way_sequence"]["post_2023_remains_sealed"] is True
    assert payload["one_way_sequence"]["failure_action"] == (
        "retire UGCI-288 without repair"
    )
