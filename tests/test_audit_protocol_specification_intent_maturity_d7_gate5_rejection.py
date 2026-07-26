from __future__ import annotations

import hashlib
import json

import pytest

from training import (
    audit_protocol_specification_intent_maturity_d7_gate5_rejection as audit,
)
from training import (
    build_protocol_specification_intent_maturity_d7_source_support as runner,
)


def test_relation_unit_count_matches_frozen_card_construction() -> None:
    assert audit.relation_unit_count(0, 0) == 1
    assert audit.relation_unit_count(5, 0) == 5
    assert audit.relation_unit_count(0, 14) == 14
    assert audit.relation_unit_count(5, 14) == 70
    assert audit.relation_unit_count(407, 3) == 1_221
    with pytest.raises(ValueError, match="negative"):
        audit.relation_unit_count(-1, 0)


def test_forensic_result_is_canonical_and_binds_terminal_authority() -> None:
    path = runner.REPO_ROOT / audit.DEFAULT_OUTPUT
    raw = path.read_bytes()
    payload = json.loads(raw)
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert raw == runner.canonical_json_bytes(payload)
    assert hashlib.sha256(raw).hexdigest() == (
        "35f961d2bde8a71045209698eee1c5508108218726b73fd2d3ceff35de85ab9b"
    )
    assert payload["result_hash"] == runner.canonical_hash(core)
    assert payload["result_hash"] == (
        "620d81baadafaa9d5cee1e5c38883846d1ac2df60acd00b67117241d87184144"
    )
    assert payload["authority"]["terminal_result_hash"] == (
        audit.TERMINAL_RESULT_HASH
    )
    assert payload["authority"]["terminal_sha256"] == audit.TERMINAL_SHA256


def test_forensic_result_identifies_exact_cardinality_failure() -> None:
    payload = json.loads(
        (runner.REPO_ROOT / audit.DEFAULT_OUTPUT).read_text(encoding="utf-8")
    )

    assert payload["failure"] == audit.EXPECTED_EXCEPTION
    assert payload["counts"] == {
        "events": 5_356,
        "model_visible_events": 4_261,
        "administrative_events": 1_095,
        "daily_cards_completed": 0,
    }
    assert payload["cardinality"]["maximum_model_events_per_card"] == 64
    assert payload["cardinality"]["overflow_card_cells"] == 24
    assert payload["cardinality"]["overflow_card_cells_by_schedule"] == {
        "ARCHIVE_D2": 6,
        "ARCHIVE_D7": 6,
        "ARCHIVE_D30": 6,
        "ARCHIVE_D90": 6,
    }
    assert payload["cardinality"]["first_overflow"] == {
        "schedule": "ARCHIVE_D2",
        "decision_day": "2020-10-02",
        "ethereum_events": 143,
        "bitcoin_events": 0,
        "event_count": 143,
        "relation_units": 143,
    }
    assert payload["cardinality"]["maximum_cardinality"] == {
        "schedule": "ARCHIVE_D2",
        "decision_day": "2022-05-08",
        "ethereum_events": 407,
        "bitcoin_events": 3,
        "event_count": 410,
        "relation_units": 1_221,
    }


def test_forensic_result_proves_read_only_pre_economic_boundary() -> None:
    payload = json.loads(
        (runner.REPO_ROOT / audit.DEFAULT_OUTPUT).read_text(encoding="utf-8")
    )
    boundary = payload["boundary"]
    integrity = payload["integrity"]

    assert boundary == {
        "official_run_reexecuted": False,
        "market_model_or_outcomes_accessed": False,
        "source_root_repaired_or_reused_for_candidate": False,
        "network_commands": 0,
    }
    assert payload["source_replay"]["network_commands"] == 0
    assert integrity["source_tree_unchanged"] is True
    assert integrity["terminal_artifact_unchanged"] is True
    assert (
        integrity["source_tree_manifest_before"]
        == integrity["source_tree_manifest_after"]
    )
    assert (
        integrity["terminal_sha256_before"]
        == integrity["terminal_sha256_after"]
        == audit.TERMINAL_SHA256
    )
