from __future__ import annotations

import gzip
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from training import preregister_cboe_edge_flip_sequence_policy as p


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


def states(level: str = "LOWER") -> dict[str, dict[str, str]]:
    return {
        state: {edge: level for edge in p.EDGE_NAMES}
        for state in p.STATE_LABELS
    }


def test_policy_is_frozen() -> None:
    assert len(p.STATE_LABELS) == 5
    assert len(p.EDGE_NAMES) == 12
    with pytest.raises(ValueError, match="policy is frozen"):
        p.build_manifest(
            policy=replace(p.Policy(), sequence_states=4),
            producer_binding_override=SYNTHETIC_PRODUCER,
            validate_dependencies=False,
        )


def test_fixed_clock_is_future_row_independent_and_dst_aware() -> None:
    before = p.fixed_clock(date(2023, 3, 10))
    transition = p.fixed_clock(date(2023, 3, 11))
    after = p.fixed_clock(date(2023, 3, 12))
    assert before["entry_utc"].date().isoformat() == "2023-03-11"
    assert transition["entry_utc"].date().isoformat() == "2023-03-12"
    assert before["exit_utc"] - before["entry_utc"] == timedelta(days=1)
    assert transition["exit_utc"] - transition["entry_utc"] == timedelta(days=1)
    assert after["exit_utc"] - after["entry_utc"] == timedelta(days=1)
    assert transition["entry_utc"] - transition["available_utc"] == timedelta(
        minutes=5
    )


def test_manifest_is_outcome_blind_and_model_unfrozen() -> None:
    payload = manifest()
    p.validate_manifest(payload)
    assert payload["policy_id"] == "CEFS-D1"
    assert payload["decision"] == "freeze_source_language_support_only"
    assert payload["contingent_economic_chronology"]["authorized_now"] is False
    assert payload["contingent_economic_chronology"][
        "model_family_frozen_now"
    ] is False
    assert all(value == 0 for value in payload["forbidden_access"].values())


def test_actual_hashes_headers_and_manifests_validate_without_rows() -> None:
    p.assert_committed(p.BOUNDARY_DOCUMENT, expected_commit=p.BOUNDARY_COMMIT)
    p._validate_source_anchors()
    assert p.csv_header(p.TERM_SOURCE) == p.TERM_HEADER
    assert p.csv_header(p.TAIL_SOURCE) == p.TAIL_HEADER
    assert p.csv_header(p.FLOW_SOURCE) == p.FLOW_HEADER


def test_header_reader_does_not_decode_invalid_tail(tmp_path: Path) -> None:
    probe = tmp_path / "probe.csv.gz"
    with gzip.GzipFile(filename=probe, mode="wb", mtime=0) as handle:
        handle.write(b"a,b\n\xff\xfe\x00not-a-row")
    assert p.csv_header(probe) == ("a", "b")


def test_flow_projection_does_not_authorize_hidden_numeric_columns() -> None:
    flow = manifest()["sources"]["flow"]
    assert flow["relation_columns"] == list(p.FLOW_RELATION_COLUMNS)
    assert flow["integrity_text_columns"] == ["response_sha256"]
    assert "forbidden from relations" in flow["integrity_text_rules"][
        "response_sha256"
    ]
    assert "total_volume" in flow["forbidden_numeric_columns"]
    assert "response_sha256" not in flow["forbidden_numeric_columns"]


def test_edge_formula_order_and_no_aggregation_are_frozen() -> None:
    language = manifest()["relation_language"]
    assert tuple(language["ordered_edges"]) == p.EDGE_NAMES
    assert tuple(language["edge_formulas"]) == p.EDGE_NAMES
    assert language["aggregation_forbidden"] is True
    assert language["source_owned_side_forbidden"] is True


def test_primary_prompt_serialization_is_exact() -> None:
    prompt = p.serialize_prompt(states(), "TARGET_FLAT")
    lines = prompt.splitlines()
    assert len(lines) == 61
    assert lines[0] == "EARLIEST.TERM_FRONT_LEVEL=LOWER"
    assert lines[-2] == "CURRENT.FLOW_VIX_SHARE_CHANGE=LOWER"
    assert lines[-1] == "POSITION=TARGET_FLAT"
    assert prompt.endswith("\n")


def test_primary_rejects_mask_and_reordered_fields() -> None:
    masked = states()
    masked["EARLIEST"]["TERM_FRONT_LEVEL"] = "MASKED"
    with pytest.raises(ValueError, match="invalid CEFS edge token"):
        p.serialize_prompt(masked, "TARGET_FLAT")
    reordered = states()
    reordered["CURRENT"] = dict(reversed(list(reordered["CURRENT"].items())))
    with pytest.raises(ValueError, match="fields or order changed"):
        p.serialize_prompt(reordered, "TARGET_FLAT")


def test_control_accepts_mask_but_position_context_is_fixed() -> None:
    masked = states("MASKED")
    assert "MASKED" in p.serialize_prompt(masked, "TARGET_SHORT", control=True)
    with pytest.raises(ValueError, match="position context changed"):
        p.serialize_prompt(masked, "SHORT", control=True)


def test_state_order_is_exact() -> None:
    reordered = dict(reversed(list(states().items())))
    with pytest.raises(ValueError, match="sequence labels or order changed"):
        p.serialize_prompt(reordered, "TARGET_LONG")


def test_support_gates_and_controls_match_boundary() -> None:
    payload = manifest()
    gates = payload["support_gates"]
    assert gates["common_dates_exact"] == 1006
    assert gates["minimum_total_intervals"] == 920
    assert gates["minimum_year_intervals"] == 230
    assert gates["minimum_quarter_intervals"] == 50
    assert gates["minimum_change_direction_share"] == 0.10
    assert gates["maximum_role_level_share_drift"] == 0.25
    assert tuple(payload["controls"]["ordered_ids"]) == p.CONTROL_IDS
    assert gates["failure_action"] == (
        "retire_cefs_d1_unchanged_before_outcomes"
    )


def test_manifest_hash_rejects_mutation() -> None:
    payload = json.loads(json.dumps(manifest()))
    payload["policy"]["sequence_states"] = 4
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
    monkeypatch.setattr(p, "_validate_sealed_producer", lambda _: None)
    monkeypatch.setattr(p, "_assert_producer_head", lambda _: None)
    output = Path("preregistration.json")
    first_sha = p.write_once(output, payload)
    assert p.write_once(output, payload) == first_sha
    (tmp_path / output).write_text("{}\n")
    with pytest.raises(RuntimeError, match="artifact drift"):
        p.write_once(output, payload)


def test_existing_artifact_cannot_bypass_sealed_producer_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = manifest()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "_validate_sealed_producer", lambda _: None)
    monkeypatch.setattr(p, "_assert_producer_head", lambda _: None)
    output = Path("preregistration.json")
    p.write_once(output, payload)

    def reject(_: dict) -> None:
        raise RuntimeError("sealed producer checked")

    monkeypatch.setattr(p, "_validate_sealed_producer", reject)
    with pytest.raises(RuntimeError, match="sealed producer checked"):
        p.write_once(output, payload)


def test_json_round_trip_validates() -> None:
    payload = json.loads(json.dumps(manifest()))
    p.validate_manifest(payload)
