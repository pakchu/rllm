from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from training import build_ethereum_stablecoin_issuance_redemption as ethereum
from training import build_governance_intent_payload_relation_source_support as gipr
from training import preregister_governance_intent_payload_relation as prereg


UTC = timezone.utc


def _unchecked_proposal(
    *,
    proposal_id: int = 1,
    targets: tuple[str, ...] = (gipr._synthetic_address(1),),
    values: tuple[int, ...] = (0,),
    signatures: tuple[bytes, ...] = (b"setValue(uint256)",),
    calldatas: tuple[bytes, ...] = (b"\x01\x02\x03\x04",),
    description: bytes = b"# valid",
    start_block: int = 100,
    end_block: int = 200,
) -> str:
    tails = (
        gipr._abi_address_array(targets),
        gipr._abi_uint_array(values),
        gipr._abi_dynamic_bytes_array(signatures),
        gipr._abi_dynamic_bytes_array(calldatas),
        gipr._abi_bytes(description),
    )
    offsets: list[int] = []
    cursor = 9 * 32
    for tail in tails:
        offsets.append(cursor)
        cursor += len(tail)
    head = b"".join(
        (
            gipr._abi_uint(proposal_id),
            gipr._abi_address(gipr._synthetic_address(2)),
            gipr._abi_uint(offsets[0]),
            gipr._abi_uint(offsets[1]),
            gipr._abi_uint(offsets[2]),
            gipr._abi_uint(offsets[3]),
            gipr._abi_uint(start_block),
            gipr._abi_uint(end_block),
            gipr._abi_uint(offsets[4]),
        )
    )
    return "0x" + (head + b"".join(tails)).hex()


def _mutate_word(data_hex: str, word_index: int, value: int) -> str:
    raw = bytearray.fromhex(data_hex[2:])
    raw[word_index * 32 : (word_index + 1) * 32] = gipr._abi_uint(value)
    return "0x" + raw.hex()


def _event(
    event: str,
    proposal_id: int = 1,
    *,
    block: int,
    available: datetime,
) -> gipr.NormalizedEvent:
    if event == "proposal_created":
        return gipr._synthetic_event(
            protocol="compound",
            event=event,
            proposal_id=proposal_id,
            available_at=available,
            block_number=block,
            log_index=0,
            targets=(gipr._synthetic_address(3),),
            values=(0,),
            signatures=("setValue(uint256)",),
            calldatas=("0x01020304",),
            description=f"proposal {proposal_id}",
        )
    return gipr._synthetic_event(
        protocol="compound",
        event=event,
        proposal_id=proposal_id,
        available_at=available,
        block_number=block,
        log_index=0,
    )


def _passing_support_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for split in prereg.SPLITS:
        name = str(split["name"])
        proposals = max(100, int(split["minimum_proposals"]))
        days = max(100, int(split["minimum_daily_decisions_after_warmup"]))
        metrics[name] = {
            "proposals": proposals,
            "actions": int(split["minimum_actions"]),
            "active_calendar_months": int(
                split["minimum_active_calendar_months"]
            ),
            "protocol_proposals": {
                "compound": int(split["minimum_proposals_per_protocol"]),
                "uniswap": int(split["minimum_proposals_per_protocol"]),
            },
            "unique_targets": int(split["minimum_unique_targets"]),
            "unique_selectors": int(split["minimum_unique_selectors"]),
            "nonempty_descriptions": 95,
            "unique_descriptions": 90,
            "action_count_buckets": ["ONE", "TWO"],
            "target_role_mix_tokens": ["KNOWN_ONLY", "MIXED"],
            "selector_count_buckets": ["ONE", "TWO"],
            "lifecycle_event_types": [
                "proposal_created",
                "proposal_queued",
                "proposal_executed",
            ],
            "daily_decisions": days,
            "unique_complete_daily_cards": 30,
            "maximum_complete_daily_card_count": 35,
            "compound_stale_days": 75,
            "uniswap_stale_days": 75,
            "both_stale_days": 60,
        }
    return metrics


def _passing_control_metrics() -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for control in gipr.CONTROL_NAMES:
        result[control] = {
            str(split["name"]): {
                "eligible": (
                    10
                    if control
                    in {
                        "within_day_event_order_reverse",
                        "lifecycle_event_rotation",
                    }
                    else 10
                ),
                "changed": 1,
            }
            for split in prereg.SPLITS
        }
    return result


def test_canonical_log_normalizes_case_quantity_width_and_order() -> None:
    first = gipr.normalize_log(gipr._synthetic_raw_log())
    second = gipr.normalize_log(
        gipr._synthetic_raw_log(
            uppercase=True,
            padded_quantities=True,
        )
    )
    assert first == second
    later = replace(
        first,
        transaction_hash=gipr._synthetic_hash(999),
        log_index=first.log_index + 1,
    )
    assert gipr.normalize_logs(
        [
            {
                **gipr._synthetic_raw_log(),
                "transactionHash": later.transaction_hash,
                "logIndex": hex(later.log_index),
            },
            gipr._synthetic_raw_log(),
        ]
    ) == [first, later]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("removed", True, "removed"),
        ("blockNumber", hex(gipr.FIRST_LOG_BLOCK - 1), "outside"),
        ("topics", ["0x" + "ff" * 32], "topic"),
    ),
)
def test_canonical_log_rejects_invalid_transport_rows(
    field: str,
    value: Any,
    message: str,
) -> None:
    raw = gipr._synthetic_raw_log()
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        gipr.normalize_log(raw)


def test_canonical_log_rejects_duplicate_identity() -> None:
    raw = gipr._synthetic_raw_log()
    with pytest.raises(ValueError, match="duplicate"):
        gipr.normalize_logs([raw, dict(raw)])


def test_canonical_log_rejects_event_before_specific_governor_deployment() -> None:
    raw = gipr._synthetic_raw_log()
    governor = next(
        row
        for row in prereg.GOVERNORS
        if row.first_code_block > gipr.FIRST_LOG_BLOCK
    )
    raw["address"] = governor.address
    raw["blockNumber"] = hex(governor.first_code_block - 1)
    with pytest.raises(ValueError, match="precedes"):
        gipr.normalize_log(raw)


def test_proposal_abi_round_trip_preserves_action_order() -> None:
    encoded = gipr.encode_proposal_created(
        proposal_id=9,
        proposer=gipr._synthetic_address(20),
        targets=(gipr._synthetic_address(21), gipr._synthetic_address(22)),
        values=(0, 5),
        signatures=("first()", ""),
        calldatas=("0x01020304", "0xaabbccdd00"),
        start_block=1_000,
        end_block=2_000,
        description="# Description\nhttps://example.invalid",
    )
    decoded = gipr.decode_proposal_created(encoded)
    assert decoded["targets"] == (
        gipr._synthetic_address(21),
        gipr._synthetic_address(22),
    )
    assert decoded["values"] == (0, 5)
    assert decoded["signatures"] == ("first()", "")
    assert decoded["calldatas"] == ("0x01020304", "0xaabbccdd00")


@pytest.mark.parametrize(
    "payload",
    (
        _unchecked_proposal(
            targets=(gipr._synthetic_address(1),),
            values=(0, 1),
        ),
        _unchecked_proposal(description=b"\xff"),
        _unchecked_proposal(description=b"a\x00b"),
        _unchecked_proposal(
            targets=tuple(
                gipr._synthetic_address(index + 1) for index in range(11)
            ),
            values=tuple(0 for _ in range(11)),
            signatures=tuple(b"x()" for _ in range(11)),
            calldatas=tuple(b"" for _ in range(11)),
        ),
        _unchecked_proposal(end_block=100),
    ),
)
def test_proposal_abi_rejects_length_text_bounds_and_chronology(
    payload: str,
) -> None:
    with pytest.raises(ValueError):
        gipr.decode_proposal_created(payload)


def test_proposal_abi_rejects_misalignment_alias_padding_and_trailing() -> None:
    encoded = _unchecked_proposal(description=b"not-aligned")
    targets_offset = int.from_bytes(
        bytes.fromhex(encoded[2:])[64:96], byteorder="big"
    )
    with pytest.raises(ValueError, match="aligned"):
        gipr.decode_proposal_created(_mutate_word(encoded, 2, targets_offset + 1))
    with pytest.raises(ValueError):
        gipr.decode_proposal_created(_mutate_word(encoded, 3, targets_offset))
    padded = bytearray.fromhex(encoded[2:])
    padded[-1] = 1
    with pytest.raises(ValueError, match="padding"):
        gipr.decode_proposal_created("0x" + padded.hex())
    with pytest.raises(ValueError, match="trailing"):
        gipr.decode_proposal_created(encoded + "00" * 32)


def test_proposal_abi_rejects_gap_and_parser_byte_bounds() -> None:
    encoded = _unchecked_proposal()
    values_offset = int.from_bytes(
        bytes.fromhex(encoded[2:])[96:128], byteorder="big"
    )
    with pytest.raises(ValueError):
        gipr.decode_proposal_created(_mutate_word(encoded, 3, values_offset + 32))
    oversized = _unchecked_proposal(
        description=b"x" * (prereg.MAX_DESCRIPTION_BYTES + 1)
    )
    with pytest.raises(ValueError, match="bound"):
        gipr.decode_proposal_created(oversized)


def test_lifecycle_fixed_heads_reject_extra_words() -> None:
    assert gipr.decode_lifecycle_event(
        "proposal_queued",
        "0x" + (gipr._abi_uint(1) + gipr._abi_uint(2)).hex(),
    ) == {"proposal_id": 1, "queue_eta": 2}
    with pytest.raises(ValueError, match="length"):
        gipr.decode_lifecycle_event(
            "proposal_executed",
            "0x" + (gipr._abi_uint(1) + gipr._abi_uint(0)).hex(),
        )


@pytest.mark.parametrize(
    "sequence",
    (
        ("proposal_created", "proposal_queued", "proposal_executed"),
        ("proposal_created", "proposal_canceled"),
        ("proposal_created", "proposal_queued", "proposal_canceled"),
    ),
)
def test_lifecycle_accepts_only_frozen_valid_transitions(
    sequence: tuple[str, ...],
) -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    events = [
        _event(
            event,
            block=10_000_000 + index,
            available=start + timedelta(hours=index),
        )
        for index, event in enumerate(sequence)
    ]
    assert gipr.validate_lifecycle(events)


@pytest.mark.parametrize(
    "sequence",
    (
        ("proposal_queued",),
        ("proposal_created", "proposal_executed"),
        ("proposal_created", "proposal_created"),
        ("proposal_created", "proposal_canceled", "proposal_queued"),
        (
            "proposal_created",
            "proposal_queued",
            "proposal_executed",
            "proposal_canceled",
        ),
    ),
)
def test_lifecycle_rejects_invalid_transitions(
    sequence: tuple[str, ...],
) -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    events = [
        _event(
            event,
            block=10_100_000 + index,
            available=start + timedelta(hours=index),
        )
        for index, event in enumerate(sequence)
    ]
    with pytest.raises(ValueError):
        gipr.validate_lifecycle(events)


def test_n_plus_64_availability_uses_confirmation_header_and_utc_boundary() -> None:
    source_time = int(
        datetime(2022, 1, 1, 23, 59, tzinfo=UTC).timestamp()
    )
    confirmation_time = int(
        datetime(2022, 1, 2, 0, 1, tzinfo=UTC).timestamp()
    )
    event = gipr.parse_log(gipr.normalize_log(gipr._synthetic_raw_log()))
    confirmation = event.block_number + prereg.CONFIRMATION_BLOCKS
    source_header = ethereum.BlockHeader(
        event.block_number,
        event.block_hash,
        gipr._synthetic_hash(601),
        source_time,
    )
    confirmation_header = ethereum.BlockHeader(
        confirmation,
        gipr._synthetic_hash(602),
        gipr._synthetic_hash(603),
        confirmation_time,
    )
    headers = {
        event.block_number: source_header,
        confirmation: confirmation_header,
    }
    materialized = gipr.materialize_events([event], headers, dict(headers))[0]
    assert materialized.available_at == datetime(
        2022, 1, 2, 0, 1, tzinfo=UTC
    )
    assert gipr._event_day(materialized) == datetime(
        2022, 1, 3, tzinfo=UTC
    )
    assert gipr.first_daily_boundary(
        datetime(2022, 1, 2, 0, 0, tzinfo=UTC)
    ) == datetime(2022, 1, 2, 0, 0, tzinfo=UTC)
    bad_headers = dict(headers)
    bad_headers[event.block_number] = replace(
        source_header,
        block_hash=gipr._synthetic_hash(999),
    )
    with pytest.raises(RuntimeError, match="transport"):
        gipr.materialize_events([event], headers, bad_headers)


@pytest.mark.parametrize(
    ("age", "expected"),
    (
        (timedelta(days=2) - timedelta(seconds=1), "D0_1"),
        (timedelta(days=2), "D2_7"),
        (timedelta(days=8), "D8_28"),
        (timedelta(days=29), "D29_90"),
        (timedelta(days=90), "D29_90"),
        (timedelta(days=90, seconds=1), "STALE_OR_NONE"),
    ),
)
def test_age_bucket_exact_boundaries(
    age: timedelta,
    expected: str,
) -> None:
    assert gipr.age_bucket(age) == expected


def test_daily_card_stops_forward_fill_after_90_elapsed_days() -> None:
    created = _event(
        "proposal_created",
        block=11_000_000,
        available=datetime(2022, 1, 1, 0, 0, tzinfo=UTC),
    )
    cards = gipr.build_daily_cards([created])
    by_time = {row["decision_at"]: row for row in cards}
    assert (
        by_time["2022-04-01T00:00:00Z"]["model_card"]["compound"][
            "lifecycle_state"
        ]
        == "PROPOSAL_CREATED"
    )
    assert (
        by_time["2022-04-02T00:00:00Z"]["model_card"]["compound"][
            "lifecycle_state"
        ]
        == "STALE_OR_NO_PROPOSAL"
    )


def test_daily_card_output_has_no_raw_forbidden_fields_or_timestamp() -> None:
    cards = gipr.build_daily_cards(gipr._synthetic_source_events())
    output = gipr.card_output_rows(cards)
    assert not (
        gipr._nested_keys(output) & gipr.RAW_MODEL_CARD_FORBIDDEN_KEYS
    )
    assert "decision_at" not in output[0]


def test_split_support_exact_threshold_and_minus_one() -> None:
    metrics = _passing_support_metrics()
    assert gipr.gate_split_proposal_action_support(metrics).passed
    metrics["test"]["proposals"] = (
        prereg.SPLITS[1]["minimum_proposals"] - 1
    )
    gate = gipr.gate_split_proposal_action_support(metrics)
    assert not gate.passed
    assert "test:proposals" in gate.failure


def test_structural_fraction_exact_threshold_and_minus_one() -> None:
    metrics = _passing_support_metrics()
    assert gipr.gate_structural_vocabulary(metrics).passed
    metrics["eval"]["nonempty_descriptions"] = 94
    gate = gipr.gate_structural_vocabulary(metrics)
    assert not gate.passed
    assert "eval:nonempty_description_fraction" in gate.failure


def test_structural_gate_rejects_collapsed_vocabulary() -> None:
    metrics = _passing_support_metrics()
    metrics["train"]["action_count_buckets"] = ["ONE"]
    metrics["train"]["target_role_mix_tokens"] = ["ALL_UNKNOWN"]
    metrics["train"]["selector_count_buckets"] = ["ONE"]
    metrics["train"]["lifecycle_event_types"] = ["proposal_created"]
    gate = gipr.gate_structural_vocabulary(metrics)
    assert not gate.passed
    assert "train:action_count_bucket_diversity" in gate.failure
    assert "train:lifecycle_vocabulary" in gate.failure


def test_daily_schedule_fraction_exact_threshold_and_minus_one() -> None:
    metrics = _passing_support_metrics()
    metrics["test"]["maximum_complete_daily_card_count"] = 122
    assert gipr.gate_daily_schedule(metrics).passed
    metrics["test"]["maximum_complete_daily_card_count"] = 123
    gate = gipr.gate_daily_schedule(metrics)
    assert not gate.passed
    assert "test:maximum_complete_card_fraction" in gate.failure


def test_relation_controls_exact_threshold_and_minimum_eligible() -> None:
    metrics = _passing_control_metrics()
    assert gipr.gate_controls(metrics).passed
    metrics["protocol_label_swap"]["eval"]["changed"] = 0
    assert not gipr.gate_controls(metrics).passed
    metrics = _passing_control_metrics()
    metrics["within_day_event_order_reverse"]["test"] = {
        "eligible": 1,
        "changed": 1,
    }
    assert not gipr.gate_controls(metrics).passed


def test_synthetic_events_pass_all_six_relation_controls() -> None:
    events = gipr._synthetic_source_events()
    cards = gipr.build_daily_cards(events)
    metrics = gipr.control_metrics(events, cards)
    gate = gipr.gate_controls(metrics)
    assert tuple(metrics) == gipr.CONTROL_NAMES
    assert gate.passed, gate.failure


def test_future_append_is_schema_valid_and_byte_invariant() -> None:
    events = gipr._synthetic_source_events()
    cards = gipr.build_daily_cards(events)
    gate = gipr.future_append_gate(events, cards)
    assert gate.passed
    assert gate.metrics["sentinel_schema_valid"] is True
    assert all(gate.metrics["byte_identical"].values())


def test_forbidden_counter_failure_and_ordered_first_failure() -> None:
    ledger = gipr.AccessLedger.zero()
    ledger.counters["pnl_rows_built"] = 1
    assert not gipr.forbidden_gate(ledger).passed
    gates = [
        gipr.GateResult(gipr.GATE_NAMES[0], True, {}),
        gipr.GateResult(gipr.GATE_NAMES[1], False, {}, "synthetic"),
    ]
    report = gipr.build_result_report(
        decision="reject",
        authority={},
        gates=gates,
        source_audit={},
        event_count=0,
        proposal_count=0,
        card_count=0,
        artifacts=None,
        ledger=gipr.AccessLedger.zero(),
    )
    assert report["first_failure"] == {
        "gate_id": 2,
        "name": gipr.GATE_NAMES[1],
    }
    assert len(report["gates"]) == 2


def test_forbidden_gate_failure_publishes_terminal_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "EVENT_OUTPUT",
        "PROPOSAL_OUTPUT",
        "CARD_OUTPUT",
        "CONTROL_OUTPUT",
        "PASS_REPORT",
        "REJECTION_REPORT",
    ):
        monkeypatch.setattr(gipr, name, tmp_path / name.lower())
    ledger = gipr.AccessLedger.zero()
    ledger.counters["future_return_rows_read"] = 1
    gates = [
        *[
            gipr.GateResult(name, True, {})
            for name in gipr.GATE_NAMES[:-1]
        ],
        gipr.forbidden_gate(ledger),
    ]
    report = gipr.build_result_report(
        decision="reject",
        authority={},
        gates=gates,
        source_audit={"event_logs_opened": True},
        event_count=1,
        proposal_count=1,
        card_count=1,
        artifacts=None,
        ledger=ledger,
    )
    gipr._publish_rejection(report)
    stored = gipr._read_canonical_json(gipr.REJECTION_REPORT)
    gipr._validate_result_report(stored, decision="reject")
    assert stored["first_failure"] == {
        "gate_id": 12,
        "name": "forbidden_access_zero",
    }
    assert stored["forbidden_access"]["future_return_rows_read"] == 1


def test_checkpoint_is_protocol_bound_and_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint = gipr.LogCheckpoint(tmp_path / "checkpoint.sqlite3")
    row = gipr.normalize_log(gipr._synthetic_raw_log())
    rows = [row]
    checkpoint.put(
        "primary",
        row.block_number,
        row.block_number,
        rows,
    )
    checkpoint.put(
        "primary",
        row.block_number,
        row.block_number,
        rows,
    )
    assert checkpoint.get(
        "primary",
        row.block_number,
        row.block_number,
    ) == rows
    with pytest.raises(RuntimeError, match="conflicts"):
        checkpoint.put(
            "primary",
            row.block_number,
            row.block_number,
            [replace(row, log_index=row.log_index + 1)],
        )
    checkpoint.close()


def test_checkpoint_persists_only_canonical_allowed_log_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        **gipr._synthetic_raw_log(),
        "providerHost": "must-not-persist.invalid",
        "receiptTime": "future-provider-time",
    }
    block = int(raw["blockNumber"], 16)
    monkeypatch.setattr(gipr, "FIRST_LOG_BLOCK", block)
    monkeypatch.setattr(gipr, "LAST_LOG_BLOCK", block)

    class Rpc:
        def call(self, method: str, params: list[Any]) -> Any:
            assert method == "eth_getLogs"
            return [raw]

    checkpoint = gipr.LogCheckpoint(tmp_path / "canonical.sqlite3")
    rows = gipr.fetch_logs(
        Rpc(),
        role="primary",
        checkpoint=checkpoint,
        max_block_range=1,
    )
    stored = checkpoint.connection.execute(
        "SELECT payload_json FROM log_chunks"
    ).fetchone()[0]
    checkpoint.close()
    assert rows == [gipr.normalize_log(raw)]
    assert "providerHost" not in stored
    assert "receiptTime" not in stored
    payload = json.loads(stored)
    assert set(payload["rows"][0]) == set(
        rows[0].canonical_dict()
    )


def test_write_once_rejects_symlink_target(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.write_bytes(b"safe")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(RuntimeError, match="conflicting"):
        gipr._write_once_bytes(link, b"unsafe")
    assert real.read_bytes() == b"safe"


def test_write_once_rejects_symlink_parent(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "redirect"
    parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent"):
        gipr._write_once_bytes(parent / "artifact", b"unsafe")
    assert not (outside / "artifact").exists()


def test_canonical_publication_is_idempotent_and_conflict_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "EVENT_OUTPUT": tmp_path / "events.gz",
        "PROPOSAL_OUTPUT": tmp_path / "proposals.gz",
        "CARD_OUTPUT": tmp_path / "cards.gz",
        "CONTROL_OUTPUT": tmp_path / "controls.json",
        "PASS_REPORT": tmp_path / "pass.json",
        "REJECTION_REPORT": tmp_path / "reject.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(gipr, name, path)
    raw_by_path = {
        paths["EVENT_OUTPUT"]: b"events",
        paths["PROPOSAL_OUTPUT"]: b"proposals",
        paths["CARD_OUTPUT"]: b"cards",
        paths["CONTROL_OUTPUT"]: b"controls",
    }
    gates = [
        gipr.GateResult(name, True, {}) for name in gipr.GATE_NAMES
    ]
    artifacts = {
        "events": gipr._artifact_entry(
            paths["EVENT_OUTPUT"],
            raw_by_path[paths["EVENT_OUTPUT"]],
            rows=1,
            row_hash=gipr.canonical_hash(["events"]),
        ),
        "proposals": gipr._artifact_entry(
            paths["PROPOSAL_OUTPUT"],
            raw_by_path[paths["PROPOSAL_OUTPUT"]],
            rows=1,
            row_hash=gipr.canonical_hash(["proposals"]),
        ),
        "daily_cards": gipr._artifact_entry(
            paths["CARD_OUTPUT"],
            raw_by_path[paths["CARD_OUTPUT"]],
            rows=1,
            row_hash=gipr.canonical_hash(["cards"]),
        ),
        "controls": gipr._artifact_entry(
            paths["CONTROL_OUTPUT"],
            raw_by_path[paths["CONTROL_OUTPUT"]],
            rows=1,
            row_hash=gipr.canonical_hash(["controls"]),
        ),
    }
    report = gipr.build_result_report(
        decision="pass",
        authority={},
        gates=gates,
        source_audit={"event_logs_opened": True},
        event_count=1,
        proposal_count=1,
        card_count=1,
        artifacts=artifacts,
        ledger=gipr.AccessLedger.zero(),
    )
    gipr._publish_pass_group(raw_by_path, report)
    gipr._publish_pass_group(raw_by_path, report)
    paths["EVENT_OUTPUT"].write_bytes(b"conflict")
    with pytest.raises(RuntimeError, match="conflicts"):
        gipr._publish_pass_group(raw_by_path, report)


def test_self_check_is_canonical_network_free_and_source_clean() -> None:
    raw = gipr.self_check_bytes()
    payload = json.loads(raw)
    assert raw == gipr.canonical_json_bytes(payload)
    assert payload["network_calls"] == 0
    assert payload["source_event_rows_opened"] == 0
    assert payload["outcomes_opened"] is False
    assert payload["forbidden_access"] == gipr.AccessLedger.zero().snapshot()
    assert all(payload["checks"].values())


def test_source_configuration_requires_two_distinct_bounded_transports() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        gipr._source_configuration(gipr.Config())
    with pytest.raises(ValueError, match="distinct"):
        gipr._source_configuration(
            gipr.Config(
                primary_rpc_url="https://one.invalid",
                verification_rpc_url="https://one.invalid/",
            )
        )
    with pytest.raises(ValueError, match="bounds"):
        gipr._source_configuration(
            gipr.Config(
                primary_rpc_url="https://one.invalid",
                verification_rpc_url="https://two.invalid",
                max_block_range=prereg.MAX_BLOCK_RANGE + 1,
            )
        )
