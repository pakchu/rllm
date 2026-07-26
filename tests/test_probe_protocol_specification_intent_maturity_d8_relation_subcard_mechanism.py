from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training import (
    probe_protocol_specification_intent_maturity_d8_relation_subcard_mechanism
    as probe,
)


PROBE_SHA256 = (
    "9c926f1fc44e60e4fcf92679dfd36db8d410220dcbbecec8c71e05bba0076d76"
)
RESULT_HASH = (
    "3b690e6e11399a12aca41a2ba79f74f5d8642f029dc5241d72d342a6f3706672"
)
SCENARIO_ROSTER_HASH = (
    "9a718845c1af15904a9d263511c601432d1ae3e2ddd17bad9e9bfb2fbefcc00c"
)


def probe_bytes() -> bytes:
    return probe.repository_path(probe.DEFAULT_OUTPUT).read_bytes()


def probe_payload() -> dict[str, Any]:
    return json.loads(probe_bytes())


def test_probe_is_canonical_and_hash_bound() -> None:
    raw = probe_bytes()
    payload = probe_payload()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert hashlib.sha256(raw).hexdigest() == PROBE_SHA256
    assert raw == probe.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == probe.canonical_hash(core)
    assert probe.canonical_json_bytes(probe.build_probe()) == raw
    assert payload["protocol_version"] == probe.PROTOCOL_VERSION
    assert payload["mechanism_version"] == probe.MECHANISM_VERSION
    assert payload["synthetic_only"] is True


def test_probe_binds_exact_d7_terminal_and_forensic_authority() -> None:
    authority = probe_payload()["d7_authority"]

    assert authority["terminal"] == {
        "commit": probe.D7_TERMINAL_COMMIT,
        "path": probe.D7_TERMINAL_PATH.as_posix(),
        "sha256": probe.D7_TERMINAL_SHA256,
        "result_hash": probe.D7_TERMINAL_RESULT_HASH,
    }
    assert authority["forensic"]["commit"] == probe.D7_FORENSIC_COMMIT
    assert authority["forensic"]["sha256"] == probe.D7_FORENSIC_SHA256
    assert authority["forensic"]["result_hash"] == (
        probe.D7_FORENSIC_RESULT_HASH
    )
    assert authority["observed_source_cardinality"] == {
        "overflow_card_cells": 24,
        "first_overflow_relation_units": 143,
        "maximum_relation_units": 1_221,
    }


def test_selected_mechanism_is_lossless_and_keeps_logical_day_identity() -> None:
    contract = probe_payload()["mechanism_contract"]

    assert contract["logical_daily_card_count"] == (
        "EXACTLY_ONE_PER_SCHEDULE_AND_DECISION_DAY"
    )
    assert contract["logical_daily_relation_roster"] == (
        "EXACT_D7_ORDERED_COMPLETE_RELATION_UNITS"
    )
    assert (
        contract["maximum_model_relation_units_per_subcard"]
        == probe.MAX_RELATION_UNITS_PER_SUBCARD
        == 64
    )
    assert contract["subcard_coverage"] == (
        "COMPLETE_NONOVERLAPPING_NO_GAP_NO_DUPLICATION"
    )
    assert contract["dropping_sampling_summarization_allowed"] is False
    assert contract["cap_raise_allowed"] is False
    assert contract["market_or_outcome_dependent_partition_allowed"] is False
    assert contract["control_denominator"] == (
        "UNIQUE_LOGICAL_DECISION_DAYS"
    )


@pytest.mark.parametrize(
    ("count", "expected_subcards"),
    ((1, 1), (64, 1), (65, 2), (70, 2), (143, 3), (1_221, 20)),
)
def test_subcard_partition_is_contiguous_complete_and_deterministic(
    count: int,
    expected_subcards: int,
) -> None:
    units = probe._synthetic_units(count)
    first = probe.build_relation_subcard_manifest(
        units,
        schedule="ARCHIVE_D90",
        decision_at="2023-01-01T12:05:00Z",
    )
    second = probe.build_relation_subcard_manifest(
        units,
        schedule="ARCHIVE_D90",
        decision_at="2023-01-01T12:05:00Z",
    )

    probe.validate_relation_subcard_manifest(units, first)
    assert first == second
    assert first["subcard_count"] == expected_subcards
    assert [row["start"] for row in first["subcards"]] == [
        index * 64 for index in range(expected_subcards)
    ]
    assert first["subcards"][-1]["end_exclusive"] == count
    assert sum(
        row["relation_unit_count"] for row in first["subcards"]
    ) == count
    assert all(
        1 <= row["relation_unit_count"] <= 64
        for row in first["subcards"]
    )


def test_manifest_rejects_empty_and_tampered_rosters() -> None:
    with pytest.raises(
        probe.D8SubcardError,
        match="ERROR_EMPTY_RELATION_ROSTER",
    ):
        probe.build_relation_subcard_manifest(
            [],
            schedule="ARCHIVE_D90",
            decision_at="2023-01-01T12:05:00Z",
        )

    units = probe._synthetic_units(70)
    manifest = probe.build_relation_subcard_manifest(
        units,
        schedule="ARCHIVE_D90",
        decision_at="2023-01-01T12:05:00Z",
    )
    tampered = json.loads(json.dumps(manifest))
    tampered["subcards"][1]["start"] += 1
    with pytest.raises(
        probe.D8SubcardError,
        match="ERROR_SUBCARD_MANIFEST_MISMATCH",
    ):
        probe.validate_relation_subcard_manifest(units, tampered)


def test_single_logical_card_hash_binds_completed_subcard_manifest() -> None:
    units = probe._synthetic_units(70)
    card = probe.build_logical_daily_card_envelope(
        schedule="ARCHIVE_D90",
        decision_at="2023-01-01T12:05:00Z",
        split="eval",
        prior_card_hash="1" * 64,
        protocol_state={
            "ethereum": "NEW_EVENT",
            "bitcoin": "NO_NEW_EVENT",
        },
        new_events=[{"synthetic": "event"}],
        relation_units=units,
    )

    probe.validate_logical_daily_card_envelope(card)
    assert card["schedule"] == "ARCHIVE_D90"
    assert card["decision_at"] == "2023-01-01T12:05:00Z"
    assert card["local_payload"]["relation_subcard_manifest"][
        "subcard_count"
    ] == 2
    assert card["local_payload_sha256"] == probe.canonical_hash(
        card["local_payload"]
    )
    assert card["card_hash"] == probe.canonical_hash(
        {
            "schedule": card["schedule"],
            "decision_at": card["decision_at"],
            "prior_card_hash": card["prior_card_hash"],
            "local_payload_sha256": card["local_payload_sha256"],
        }
    )

    tampered = deepcopy(card)
    tampered["local_payload"]["relation_subcard_manifest"]["subcards"][0][
        "end_exclusive"
    ] -= 1
    with pytest.raises(probe.D8SubcardError):
        probe.validate_logical_daily_card_envelope(tampered)

    cross_day = deepcopy(card)
    cross_day["decision_at"] = "2023-01-02T12:05:00Z"
    cross_day["card_hash"] = probe.canonical_hash(
        {
            "schedule": cross_day["schedule"],
            "decision_at": cross_day["decision_at"],
            "prior_card_hash": cross_day["prior_card_hash"],
            "local_payload_sha256": cross_day["local_payload_sha256"],
        }
    )
    with pytest.raises(
        probe.D8SubcardError,
        match="ERROR_LOGICAL_CARD_IDENTITY_MISMATCH",
    ):
        probe.validate_logical_daily_card_envelope(cross_day)


def test_synthetic_battery_is_complete() -> None:
    battery = probe_payload()["synthetic_battery"]

    assert battery["all_passed"] is True
    assert battery["scenario_count"] == 12
    assert len(battery["scenarios"]) == 12
    assert all(row["passed"] is True for row in battery["scenarios"])
    assert battery["scenario_roster_hash"] == SCENARIO_ROSTER_HASH


def test_probe_is_source_model_market_and_outcome_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = probe_payload()
    assert payload["access_boundary"] == {
        "d7_terminal_artifact_read": True,
        "d7_forensic_artifact_read": True,
        "d7_forensic_source_root_accessed": False,
        "d7_run_invoked": False,
        "external_network_accessed": False,
        "historical_proposal_text_accessed": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "outcomes_accessed": False,
        "reward_trade_pnl_accessed": False,
    }

    source = Path(probe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names: set[str] = set()
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
        elif isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert called_names.isdisjoint(
        {
            "run_audit",
            "run_official",
            "collect_commit_chain",
            "collect_proposal_groups",
            "materialize_events",
            "urlopen",
        }
    )
    assert imported_roots.isdisjoint(
        {
            "aiohttp",
            "datasets",
            "httpx",
            "pandas",
            "peft",
            "requests",
            "socket",
            "subprocess",
            "torch",
            "transformers",
            "urllib",
        }
    )
    assert "/tmp/psim-d7-source" not in source

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("forbidden source/audit function was called")

    monkeypatch.setattr(probe.d7_audit, "run_audit", forbidden)
    monkeypatch.setattr(
        probe.d7_audit.runner,
        "run_official",
        forbidden,
    )
    monkeypatch.setattr(
        probe.d7_audit.runner,
        "collect_commit_chain",
        forbidden,
    )
    monkeypatch.setattr(
        probe.d7_audit.runner,
        "collect_proposal_groups",
        forbidden,
    )
    assert probe.build_probe()["synthetic_battery"]["all_passed"] is True
