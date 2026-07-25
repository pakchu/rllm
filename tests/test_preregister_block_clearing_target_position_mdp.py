from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random

import pytest

from training import preregister_block_clearing_relational_topology as bcrt
from training import preregister_block_clearing_target_position_mdp as p


UTC = timezone.utc


def _tokens(offset: int = 0) -> OrderedDict[str, str]:
    return OrderedDict(
        (
            name,
            vocabulary[(index + offset) % len(vocabulary)],
        )
        for index, (name, vocabulary) in enumerate(bcrt.TOKEN_SCHEMA)
    )


def _row(
    index: int,
    *,
    entry: datetime | None = None,
    bucket: datetime | None = None,
    confirmation_height: int | None = None,
    token_offset: int | None = None,
) -> dict[str, object]:
    entry_time = entry or datetime(2020, 4, 1, tzinfo=UTC) + timedelta(
        hours=12 * index
    )
    bucket_start = bucket or entry_time - timedelta(days=4)
    return {
        "signal_id": f"source-{index:04d}",
        "bucket_start": bucket_start.isoformat(),
        "confirmation_height": (
            confirmation_height
            if confirmation_height is not None
            else 700_000 + index
        ),
        "entry_time": entry_time.isoformat(),
        **_tokens(index if token_offset is None else token_offset),
    }


def test_manifest_is_source_only_and_2023_cannot_gate() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)

    assert payload["policy"]["policy_id"] == "BCTP-12H"
    assert payload["stage_authority"]["authorized"] == [
        "raw_source_integrity_replay",
        "bcrt_state_replay",
        "same_release_batching",
        "three_state_source_sequence",
        "source_sequence_support",
    ]
    assert payload["report_only_2023"] == {
        "incidence_emitted": True,
        "may_change_support_boolean": False,
        "may_select_or_repair": False,
        "unknown_vocabulary_action": "TARGET_FLAT",
    }
    assert payload["temporal_roles"]["2023_may_select"] is False
    assert "maximum_entry_gap" not in json.dumps(
        payload["source_support_gates"],
        sort_keys=True,
    )
    assert all(
        value in (0, False)
        for value in payload["outcome_boundary"].values()
    )


def test_manifest_binds_terminal_bcrt_without_repair() -> None:
    frozen = p.build_manifest()["immutable_bcrt_representation"]

    assert frozen["policy_id"] == "BCRT-72"
    assert frozen["retirement"]["remains_terminal"] is True
    assert frozen["retirement"]["failed_gap_gate_not_changed"] is True
    assert frozen["support_artifact"]["sha256"] == (
        p.BCRT_SUPPORT_ARTIFACT_SHA256
    )
    assert frozen["support_artifact"]["manifest_hash"] == (
        p.BCRT_SUPPORT_MANIFEST_HASH
    )
    assert frozen["expected_replay_counts"] == {
        "formed_buckets": 2_918,
        "rank_complete_states": 2_792,
        "token_ready_states": 2_791,
    }
    assert frozen["raw_source"]["allowlist"] == list(bcrt.SOURCE_ALLOWLIST)
    assert frozen["reference"]["allowlist"] == list(bcrt.REFERENCE_ALLOWLIST)


def test_snapshot_reuses_exact_bcrt_schema_and_vocabulary() -> None:
    tokens = _tokens()
    validated = p.validate_snapshot(tokens)
    assert tuple(validated) == bcrt.TOKEN_COLUMNS
    assert p.canonical_snapshot(tokens).count(" | ") == (
        len(bcrt.TOKEN_COLUMNS) - 1
    )

    reversed_tokens = OrderedDict(reversed(tuple(tokens.items())))
    with pytest.raises(ValueError, match="order or schema"):
        p.validate_snapshot(reversed_tokens)

    invalid = _tokens()
    invalid[bcrt.TOKEN_COLUMNS[0]] = "UNKNOWN"
    with pytest.raises(ValueError, match="invalid"):
        p.validate_snapshot(invalid)


def test_source_signature_is_oldest_first_and_contains_no_time() -> None:
    times = [
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=12 * index)
        for index in range(3)
    ]
    snapshots = [_tokens(index) for index in range(3)]
    signature = p.source_sequence_signature(times, snapshots)

    shifted = [value + timedelta(days=100) for value in times]
    assert signature == p.source_sequence_signature(shifted, snapshots)
    assert signature != p.source_sequence_signature(
        times,
        list(reversed(snapshots)),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        p.source_sequence_signature(
            [times[0], times[0], times[2]],
            snapshots,
        )
    with pytest.raises(ValueError, match="exactly three"):
        p.source_sequence_signature(times[:2], snapshots[:2])


def test_same_release_batch_uses_latest_bucket_then_confirmation() -> None:
    entry = datetime(2021, 5, 1, 12, tzinfo=UTC)
    older_bucket = entry - timedelta(days=5)
    latest_bucket = entry - timedelta(days=4)
    rows = [
        _row(
            1,
            entry=entry,
            bucket=older_bucket,
            confirmation_height=900_000,
        ),
        _row(
            2,
            entry=entry,
            bucket=latest_bucket,
            confirmation_height=800_000,
        ),
        _row(
            3,
            entry=entry,
            bucket=latest_bucket,
            confirmation_height=800_001,
        ),
        _row(4, entry=entry + timedelta(hours=12)),
    ]

    selected = p.batch_actionable_releases(rows)
    assert [row["signal_id"] for row in selected] == [
        "source-0003",
        "source-0004",
    ]
    assert selected[0]["confirmation_height"] == 800_001


def test_same_release_batch_rejects_missing_height_and_bad_clock() -> None:
    missing = _row(0)
    del missing["confirmation_height"]
    with pytest.raises(ValueError, match="confirmation_height"):
        p.batch_actionable_releases([missing])

    bad_clock = _row(0)
    bad_clock["bucket_start"] = bad_clock["entry_time"]
    with pytest.raises(ValueError, match="must precede"):
        p.batch_actionable_releases([bad_clock])

    first = _row(1)
    duplicate = dict(first)
    duplicate[bcrt.TOKEN_COLUMNS[0]] = _tokens(1)[
        bcrt.TOKEN_COLUMNS[0]
    ]
    with pytest.raises(ValueError, match="duplicate same-release"):
        p.batch_actionable_releases([first, duplicate])


def test_three_release_warmup_and_sequence_order() -> None:
    rows = [_row(index) for index in range(4)]
    sequences = p.build_source_sequences(rows)

    assert len(sequences) == 2
    first = sequences[0]
    second = sequences[1]
    assert first["source_signal_id_m2"] == "source-0000"
    assert first["source_signal_id_m1"] == "source-0001"
    assert first["source_signal_id_s0"] == "source-0002"
    assert second["source_signal_id_m2"] == "source-0001"
    assert second["source_signal_id_m1"] == "source-0002"
    assert second["source_signal_id_s0"] == "source-0003"
    assert tuple(first) == p.SOURCE_SEQUENCE_COLUMNS


def test_suppressed_same_release_does_not_consume_warmup_slot() -> None:
    entry = datetime(2020, 4, 1, tzinfo=UTC)
    rows = [
        _row(0, entry=entry, confirmation_height=700_000),
        _row(
            1,
            entry=entry,
            bucket=entry - timedelta(days=3),
            confirmation_height=700_001,
        ),
        _row(2, entry=entry + timedelta(hours=12)),
        _row(3, entry=entry + timedelta(hours=24)),
    ]

    sequences = p.build_source_sequences(rows)
    assert len(sequences) == 1
    assert sequences[0]["source_signal_id_m2"] == "source-0001"
    assert sequences[0]["source_signal_id_m1"] == "source-0002"
    assert sequences[0]["source_signal_id_s0"] == "source-0003"


def test_future_append_does_not_change_existing_sequences() -> None:
    prefix = [_row(index) for index in range(5)]
    baseline = p.build_source_sequences(prefix)
    extended = p.build_source_sequences(
        [*prefix, _row(5), _row(6)]
    )
    assert extended[: len(baseline)] == baseline


def test_input_order_does_not_change_sequence_bytes() -> None:
    rows = [_row(index) for index in range(8)]
    expected = p.build_source_sequences(rows)
    shuffled = rows.copy()
    random.Random(20260725).shuffle(shuffled)
    assert p.build_source_sequences(shuffled) == expected


def test_source_sequence_schema_contains_no_policy_or_outcome_field() -> None:
    columns = set(p.SOURCE_SEQUENCE_COLUMNS)
    assert "position" not in columns
    assert "action" not in columns
    assert "side" not in columns
    assert "confirmation_height" not in columns
    assert all(
        field not in columns for field in p.FORBIDDEN_SOURCE_OUTPUT_FIELDS
    )
    assert len(p.SOURCE_TOKEN_COLUMNS) == 3 * len(bcrt.TOKEN_COLUMNS)


def test_write_once_is_deterministic_and_rejects_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bctp-prereg.json"
    payload = p.build_manifest()
    first = p.write_once(output, payload)
    second = p.write_once(output, payload)
    assert first == second
    assert json.loads(output.read_text()) == payload

    changed = dict(payload)
    changed["protocol_version"] = "drift"
    with pytest.raises(RuntimeError, match="write-once"):
        p.write_once(output, changed)


def test_boundary_and_current_bcrt_dependency_hashes() -> None:
    assert p.sha256_file(p.BOUNDARY_DOCUMENT) == p.BOUNDARY_DOCUMENT_SHA256
    frozen = p.build_manifest()["immutable_bcrt_representation"]
    for contract in (
        frozen["preregistration_source"],
        frozen["support_source"],
        frozen["preregistration_artifact"],
        frozen["support_artifact"],
        frozen["retirement"],
        frozen["raw_source"],
        frozen["source_manifest"],
        frozen["reference"],
    ):
        assert p.sha256_file(contract["path"]) == contract["sha256"]


def test_frozen_preregistration_artifact_matches_builder() -> None:
    path = Path(p.DEFAULT_OUTPUT)
    payload = json.loads(path.read_text())
    assert payload == p.build_manifest()
    p.validate_manifest(payload)
    assert payload["manifest_hash"] == (
        "3c84d896c0d5e5c2917d06c9e34e786f6b0f8e396798971e1da35087f5d40635"
    )
