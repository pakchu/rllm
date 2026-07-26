from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    preregister_protocol_specification_intent_maturity as d1,
)
from training import (
    probe_protocol_specification_intent_maturity_d4_parser as probe,
)


RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_d4_parser_probe_"
    "2026-07-26.json"
)
DECISION_PATH = Path(
    "docs/post-psim-d3-alpha-mechanism-audit-2026-07-26.md"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    return probe.build_probe()


def test_d3_terminal_boundary_is_exactly_bound(
    payload: dict[str, object],
) -> None:
    assert probe.sha256_file(probe.D3_TERMINAL_PATH) == (
        probe.D3_TERMINAL_SHA256
    )
    binding = payload["d3_terminal_binding"]
    assert isinstance(binding, dict)
    assert binding == {
        "commit": "f9089a300d4ba97722ecc1b59f8f8260eff8851b",
        "decision": "reject",
        "first_failure_gate_id": 4,
        "path": (
            "results/protocol_specification_intent_maturity_d3_"
            "source_rejection_2026-07-25.json"
        ),
        "result_hash": (
            "b00b54b70720d42d213b315e82e7ff3ad0df03909b92aaa514299e750fa1ba2c"
        ),
        "sha256": (
            "a9be5b5990ad79b7da7d72a22968f4f62a2700877b198606565cc70206fe9802"
        ),
    }


def test_synthetic_d3_failure_shape_is_accepted_without_blob_reuse() -> None:
    with pytest.raises(ValueError, match="blank line inside header"):
        d1.parse_eip_preamble(probe.SYNTHETIC_D3_FAILURE_SHAPE)
    candidate = probe.parse_eip_preamble_d4(
        probe.SYNTHETIC_D3_FAILURE_SHAPE
    )
    control = d1.parse_eip_preamble(probe.SYNTHETIC_D3_FAILURE_CONTROL)
    assert candidate == control
    assert candidate["eip"] == "2378"


@pytest.mark.parametrize(
    ("with_empty", "control"),
    probe.NORMALIZED_EMPTY_ACCEPTANCE_PAIRS,
)
def test_normalized_empty_lines_are_nonsemantic_only_for_eip(
    with_empty: bytes,
    control: bytes,
) -> None:
    assert probe.parse_eip_preamble_d4(with_empty) == (
        probe.parse_eip_preamble_d4(control)
    )
    assert probe.parse_eip_preamble_d4(with_empty) == (
        d1.parse_eip_preamble(control)
    )


@pytest.mark.parametrize("raw", probe.D1_ACCEPTED_EIP_FIXTURES)
def test_all_synthetic_d1_accepted_eip_outputs_are_unchanged(
    raw: bytes,
) -> None:
    assert probe.parse_eip_preamble_d4(raw) == d1.parse_eip_preamble(raw)


@pytest.mark.parametrize("raw", probe.D1_NONBLANK_REJECTION_FIXTURES)
def test_every_nonblank_d1_eip_rejection_remains_rejected(
    raw: bytes,
) -> None:
    with pytest.raises((UnicodeDecodeError, ValueError)):
        probe.parse_eip_preamble_d4(raw)


def test_empty_lines_do_not_terminate_header_or_bypass_line_limit() -> None:
    raw = b"---\neip: 7\n\ntitle: Still header\n---\n"
    assert probe.parse_eip_preamble_d4(raw)["title"] == "Still header"

    within_limit = (
        b"---\n" + b"eip: 91\n" + (b"\n" * 255) + b"---\n"
    )
    beyond_limit = (
        b"---\n" + b"eip: 92\n" + (b"\n" * 256) + b"---\n"
    )
    assert probe.parse_eip_preamble_d4(within_limit)["eip"] == "91"
    with pytest.raises(ValueError, match="header line count"):
        probe.parse_eip_preamble_d4(beyond_limit)


def test_empty_lines_do_not_bypass_header_byte_limit() -> None:
    oversized_comment = b"#" + (b"x" * 50_000)
    raw = (
        b"---\n"
        b"eip: 93\n"
        b"\n"
        + oversized_comment
        + b"\n"
        + oversized_comment
        + b"\n"
        + oversized_comment
        + b"\n---\n"
    )
    with pytest.raises(ValueError, match="header exceeds maximum bytes"):
        probe.parse_eip_preamble_d4(raw)


def test_separator_does_not_rescue_empty_value() -> None:
    with pytest.raises(ValueError, match="empty header value"):
        probe.parse_eip_preamble_d4(
            b"---\neip: 94\ndescription:\n\n---\n"
        )


def test_empty_only_header_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="header line count"):
        probe.parse_eip_preamble_d4(b"---\n\n\n---\n")
    with pytest.raises(ValueError, match="header contains no fields"):
        probe.parse_eip_preamble_d4(b"---\n# comment\n\n---\n")


@pytest.mark.parametrize("raw", probe.BIP_ACCEPTED_FIXTURES)
def test_bip_acceptance_is_the_identical_d1_function(raw: bytes) -> None:
    assert probe.parse_bip_preamble_d4 is d1.parse_bip_preamble
    assert probe.parse_bip_preamble_d4(raw) == d1.parse_bip_preamble(raw)


@pytest.mark.parametrize("raw", probe.BIP_REJECTION_FIXTURES)
def test_bip_rejections_are_unchanged(raw: bytes) -> None:
    assert probe.parse_bip_preamble_d4 is d1.parse_bip_preamble
    with pytest.raises((UnicodeDecodeError, ValueError)):
        probe.parse_bip_preamble_d4(raw)


def test_probe_is_synthetic_only_and_outcome_blind(
    payload: dict[str, object],
) -> None:
    assert payload["synthetic_only"] is True
    assert payload["selection_scope"] == (
        "AUTHORIZE_D4_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
    )
    assert payload["access_boundary"] == {
        "d3_forensic_root_accessed": False,
        "d3_terminal_artifact_read": True,
        "market_data_accessed": False,
        "model_accessed": False,
        "official_historical_proposal_source_accessed": False,
        "outcomes_accessed": False,
    }
    provenance = payload["known_failure_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["fixture"] == (
        "SYNTHETIC_STRUCTURE_ONLY_NOT_HISTORICAL_BLOB_BYTES"
    )
    assert provenance["historical_blob_opened_by_probe"] is False


def test_probe_hash_is_complete(payload: dict[str, object]) -> None:
    unhashed = dict(payload)
    result_hash = unhashed.pop("result_hash")
    assert result_hash == probe.canonical_hash(unhashed)


def test_written_probe_is_canonical_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    target = tmp_path / "probe.json"
    monkeypatch.setattr(probe, "build_probe", lambda: payload)
    first = probe.write_probe(target)
    second = probe.write_probe(target)
    assert first == second == payload
    assert target.read_bytes() == probe.canonical_json_bytes(payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_existing_probe_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    target = tmp_path / "probe.json"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(probe, "build_probe", lambda: payload)
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D4 parser probe differs",
    ):
        probe.write_probe(target)


def test_decision_document_binds_evidence_and_version_mismatch() -> None:
    text = DECISION_PATH.read_text(encoding="utf-8")
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert "PSIM-D4" in text
    assert "PSIM-D3 is terminally rejected" in text
    assert "normalized-empty" in text
    assert "historical compatibility parser" in text
    assert "eipw-preamble 0.4.0" in text
    assert "current-validator compatibility" in text
    assert "https://eips.ethereum.org/EIPS/eip-1" in text
    assert "https://yaml.org/spec/1.2.2/#66-comments" in text
    assert (
        "https://github.com/ethereum/eipw/blob/"
        "5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa/"
        "eipw-preamble/src/lib.rs#L103-L155"
    ) in text
    assert result["result_hash"] in text
    assert probe.sha256_file(RESULT_PATH) in text
