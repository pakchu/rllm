from __future__ import annotations

import gzip
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from training import preregister_london_cash_derivative_path as p


SYNTHETIC_PRODUCER = {
    "path": p.PRODUCER_SCRIPT,
    "commit": "synthetic-preregistration-test",
    "sha256": "0" * 64,
}


def manifest() -> dict:
    return p.build_manifest(
        producer_binding_override=SYNTHETIC_PRODUCER,
        validate_dependencies=False,
    )


def ready_tokens() -> dict[str, str]:
    return {field: values[0] for field, values in p.TOKEN_SCHEMA}


def test_policy_and_calendar_are_frozen() -> None:
    policy = p.Policy()
    assert p.expected_calendar_lines(policy) == 1096
    assert p.calendar_slot_counts(policy) == {
        "276": 3,
        "288": 1090,
        "300": 3,
    }
    with pytest.raises(ValueError, match="policy is frozen"):
        p.build_manifest(
            policy=replace(policy, sequence_lines=20),
            producer_binding_override=SYNTHETIC_PRODUCER,
            validate_dependencies=False,
        )


def test_london_dst_slot_counts_are_explicit() -> None:
    assert p.expected_source_slots(date(2020, 2, 1)) == 288
    assert p.expected_source_slots(date(2020, 3, 29)) == 276
    assert p.expected_source_slots(date(2020, 10, 25)) == 300


def test_manifest_is_source_token_only_and_outcome_blind() -> None:
    payload = manifest()
    p.validate_manifest(payload)
    assert payload["policy_id"] == "LCDP-D1"
    assert payload["decision"] == "freeze_source_token_support_only"
    assert payload["contingent_economic_chronology"]["authorized_now"] is False
    assert payload["research_history"]["candidate_specific_joint_state_opened"] is False
    assert all(value == 0 for value in payload["forbidden_access"].values())


def test_manifest_binds_boundary_and_raw_source_containers() -> None:
    payload = manifest()
    authority = payload["authority"]
    sources = payload["sources"]
    assert authority["boundary_commit"] == p.BOUNDARY_COMMIT
    assert authority["boundary_document_sha256"] == p.BOUNDARY_DOCUMENT_SHA256
    assert sources["coinbase"]["projected_columns"] == list(p.COINBASE_HEADER)
    assert sources["binance"]["projected_columns"] == list(p.BINANCE_HEADER)
    assert sources["manifest"]["manifest_hash"] == p.SOURCE_MANIFEST_HASH
    assert sources["manifest"]["historical_snapshot_is_point_in_time"] is False


def test_actual_hashes_and_headers_validate_without_source_row_parsing() -> None:
    p.assert_committed(p.BOUNDARY_DOCUMENT, expected_commit=p.BOUNDARY_COMMIT)
    p._validate_source_anchors()
    assert p.csv_header(p.COINBASE_SOURCE) == p.COINBASE_HEADER
    assert p.csv_header(p.BINANCE_SOURCE) == p.BINANCE_HEADER
    assert p.sha256_csv_header(p.COINBASE_SOURCE) == p.COINBASE_HEADER_SHA256
    assert p.sha256_csv_header(p.BINANCE_SOURCE) == p.BINANCE_HEADER_SHA256


def test_header_reader_does_not_decode_invalid_tail(tmp_path: Path) -> None:
    probe = tmp_path / "probe.csv.gz"
    with gzip.GzipFile(filename=probe, mode="wb", mtime=0) as handle:
        handle.write(b"a,b\n\xff\xfe\x00not-a-row")
    assert p.csv_header(probe) == ("a", "b")


def test_token_schema_and_action_space_are_exact() -> None:
    payload = manifest()
    language = payload["token_language"]
    assert tuple(language["ordered_fields"]) == p.TOKEN_COLUMNS
    assert tuple(language["action_space"]) == p.ACTION_SPACE
    assert len(p.TOKEN_COLUMNS) == 11
    assert language["sequence"].startswith("21 emitted calendar-day lines")
    assert language["safety_action"] == "TARGET_FLAT without model invocation"


def test_ready_line_serialization_is_ordered_and_exact() -> None:
    tokens = ready_tokens()
    serialized = p.serialize_line(tokens)
    assert serialized.split("|") == [
        f"{field}={tokens[field]}" for field in p.TOKEN_COLUMNS
    ]
    reordered = dict(reversed(list(tokens.items())))
    with pytest.raises(ValueError, match="fields or order changed"):
        p.serialize_line(reordered)


@pytest.mark.parametrize(
    ("calendar_context", "token"),
    [
        ("SATURDAY", "SOURCE_INVALID"),
        ("WEEKDAY", "SOURCE_INVALID_START"),
        ("SUNDAY", "RANK_UNREADY"),
    ],
)
def test_safety_lines_preserve_calendar_day(
    calendar_context: str,
    token: str,
) -> None:
    tokens = p.safety_line(calendar_context, token)
    assert tokens["calendar_context"] == calendar_context
    assert set(tokens[field] for field in p.TOKEN_COLUMNS[1:]) == {token}
    assert token in p.serialize_line(
        tokens,
        allow_source_invalid_start=token == "SOURCE_INVALID_START",
    )


def test_mixed_primary_safety_lines_are_rejected() -> None:
    tokens = ready_tokens()
    tokens["daily_alignment"] = "SOURCE_INVALID"
    with pytest.raises(ValueError, match="safety line must be uniform"):
        p.serialize_line(tokens)
    tokens = p.safety_line("WEEKDAY", "SOURCE_INVALID")
    tokens["daily_leader"] = "RANK_UNREADY"
    with pytest.raises(ValueError, match="safety line must be uniform"):
        p.serialize_line(tokens)


def test_source_invalid_start_requires_explicit_first_line_context() -> None:
    tokens = p.safety_line("WEEKDAY", "SOURCE_INVALID_START")
    with pytest.raises(ValueError, match="explicit first-line context"):
        p.serialize_line(tokens)
    assert "SOURCE_INVALID_START" in p.serialize_line(
        tokens,
        allow_source_invalid_start=True,
    )
    with pytest.raises(ValueError, match="calendar must be WEEKDAY"):
        p.safety_line("SATURDAY", "SOURCE_INVALID_START")


def test_primary_rejects_control_only_tokens() -> None:
    tokens = ready_tokens()
    tokens["calendar_context"] = "CALENDAR_MASKED"
    with pytest.raises(ValueError, match="invalid LCDP token"):
        p.serialize_line(tokens)
    assert "CALENDAR_MASKED" in p.serialize_line(tokens, control=True)


def test_control_vocabulary_is_field_scoped() -> None:
    vocabulary = p.control_vocabulary()
    assert "CALENDAR_MASKED" in vocabulary["calendar_context"]
    assert "CALENDAR_MASKED" not in vocabulary["daily_alignment"]
    assert "CASH_ONLY_RISE" in vocabulary["daily_alignment"]
    assert "CASH_ONLY_RISE" not in vocabulary["daily_leader"]
    assert "ABLATION_MASKED" in vocabulary["daily_leader"]
    assert "CONTROL_UNREADY" in vocabulary["participation_state"]


def test_control_unready_line_must_be_uniform() -> None:
    tokens = ready_tokens()
    for field in p.TOKEN_COLUMNS[1:]:
        tokens[field] = "CONTROL_UNREADY"
    assert "CONTROL_UNREADY" in p.serialize_line(tokens, control=True)
    tokens["daily_leader"] = "CASH_LEADS_RISE"
    with pytest.raises(ValueError, match="CONTROL_UNREADY line must be uniform"):
        p.serialize_line(tokens, control=True)


def test_controls_and_lclr_mapping_are_frozen_without_filtering_primary() -> None:
    controls = manifest()["controls"]
    assert tuple(controls["source_token_control_ids"]) == p.CONTROL_IDS
    lclr = controls["later_lclr"]
    assert "never filters primary" in lclr["lclr_mask_daily_target"]
    assert lclr["preregistration_sha256"] == p.LCLR_PREREGISTRATION_SHA256
    assert lclr["rejection_sha256"] == p.LCLR_REJECTION_SHA256


def test_clock_prevents_cross_year_outcome_access() -> None:
    clock = manifest()["clock"]
    assert "December 31 deterministic TARGET_FLAT" in clock["year_terminal"]
    assert "no next-year outcome opened" in clock["year_terminal"]
    assert "SOURCE_INVALID_START" in clock["first_line"]


def test_support_gates_are_conjunctive_and_stop_before_outcomes() -> None:
    gates = manifest()["support_gates"]
    assert gates["calendar_lines_exact"] == 1096
    assert gates["source_valid_share_min_year"] == 0.97
    assert gates["source_valid_share_min_quarter"] == 0.95
    assert gates["ready_min_2020"] == 280
    assert gates["ready_min_2021"] == 350
    assert gates["ready_min_2022"] == 350
    assert gates["category_share_min"] == 0.03
    assert gates["category_share_max"] == 0.94
    assert gates["control_difference_share_min"] == 0.05
    assert gates["failure_action"] == (
        "retire_lcdp_d1_unchanged_before_outcomes"
    )


def test_manifest_hash_rejects_mutation() -> None:
    payload = json.loads(json.dumps(manifest()))
    payload["policy"]["sequence_lines"] = 20
    with pytest.raises(ValueError, match="differs from frozen code"):
        p.validate_manifest(payload)


def test_forbidden_key_rejected_even_with_recomputed_hash() -> None:
    payload = json.loads(json.dumps(manifest()))
    payload["future_return"] = 0
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(ValueError, match="forbidden keys"):
        p.validate_manifest(payload)


def test_write_once_is_idempotent_and_refuses_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = manifest()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    output = Path("preregistration.json")
    first_sha = p.write_once(
        output,
        payload,
        enforce_producer_head=False,
    )
    assert p.write_once(
        output,
        payload,
        enforce_producer_head=False,
    ) == first_sha
    (tmp_path / output).write_text("{}\n")
    with pytest.raises(RuntimeError, match="artifact drift"):
        p.write_once(output, payload, enforce_producer_head=False)


def test_json_round_trip_validates() -> None:
    payload = json.loads(json.dumps(manifest()))
    p.validate_manifest(payload)
