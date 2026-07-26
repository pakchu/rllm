"""Probe the synthetic-only PSIM-D4 historical EIP preamble grammar.

PSIM-D3 is terminally rejected and is never repaired or rerun.  This module
does not read any proposal repository or forensic source root.  It binds the
published D3 rejection and evaluates one narrowly scoped successor rule on
synthetic bytes only: normalized-empty lines inside EIP front matter are
non-semantic separators.  The BIP parser and every other D1 rule remain
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import (
    preregister_protocol_specification_intent_maturity as d1,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d4_parser_probe_"
    "2026-07-26.json"
)
PROTOCOL_VERSION = "psim_d4_historical_eip_preamble_probe_v1"
PARSER_VERSION = "PSIM_PREAMBLE_STATE_MACHINE_V2_EIP_EMPTY_SEPARATORS"

D3_TERMINAL_COMMIT = "f9089a300d4ba97722ecc1b59f8f8260eff8851b"
D3_TERMINAL_PATH = Path(
    "results/protocol_specification_intent_maturity_d3_source_rejection_"
    "2026-07-25.json"
)
D3_TERMINAL_SHA256 = (
    "a9be5b5990ad79b7da7d72a22968f4f62a2700877b198606565cc70206fe9802"
)
D3_TERMINAL_RESULT_HASH = (
    "b00b54b70720d42d213b315e82e7ff3ad0df03909b92aaa514299e750fa1ba2c"
)

D1_PARSER_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity.py"
)

OFFICIAL_REFERENCE_NOTES = (
    {
        "claim": (
            "EIP-1 names the fenced RFC-822-style preamble Jekyll front "
            "matter; it does not explicitly define blank-line rejection."
        ),
        "url": "https://eips.ethereum.org/EIPS/eip-1",
        "version": "canonical page observed 2026-07-26 KST",
    },
    {
        "claim": (
            "The current EIPs repository uses eipw to enforce EIP-1 rules."
        ),
        "url": "https://github.com/ethereum/EIPs",
        "version": "repository README observed 2026-07-26 KST",
    },
    {
        "claim": (
            "eipw-preamble 0.4.0 parses every extracted line as a colon-"
            "delimited field, so an empty line is rejected by that current "
            "validator implementation."
        ),
        "url": (
            "https://github.com/ethereum/eipw/blob/"
            "5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa/"
            "eipw-preamble/src/lib.rs#L103-L155"
        ),
        "version": (
            "eipw-preamble 0.4.0 source VCS "
            "5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa"
        ),
    },
    {
        "claim": (
            "YAML 1.2.2 treats whitespace-only lines outside scalar content "
            "as comment lines and permits multi-line, possibly empty, "
            "separation comments."
        ),
        "url": "https://yaml.org/spec/1.2.2/#66-comments",
        "version": "YAML 1.2.2",
    },
)

KNOWN_D3_FAILURE_EVIDENCE = {
    "blob_oid": "ac34c07b91d6dffa14922951473f50dd587eb900",
    "commit_oid": "b788f38a216ca4cfea9d9de8ccfcf4cf658c8950",
    "effective_day": "2020-01-29",
    "error": "PSIM blank line inside header",
    "path": "EIPS/eip-2378.md",
    "proposal": 2378,
    "raw_sha256": (
        "a2fd3d87db7861f2b50739bf6c9015b968abc6fb6ffee7629492626034f41bb1"
    ),
    "side": "new",
}

# This is deliberately not a copy of the historical blob.  It represents only
# the already published failure shape and therefore cannot become a hidden
# source-data reuse path.
SYNTHETIC_D3_FAILURE_SHAPE = (
    b"---\n"
    b"eip: 2378\n"
    b"title: Synthetic historical compatibility fixture\n"
    b"author: Example Author\n"
    b"status: Draft\n"
    b"type: Standards Track\n"
    b"category: Core\n"
    b"created: 2000-01-01\n"
    b"\n"
    b"---\n"
    b"# Abstract\n"
)
SYNTHETIC_D3_FAILURE_CONTROL = SYNTHETIC_D3_FAILURE_SHAPE.replace(
    b"created: 2000-01-01\n\n---",
    b"created: 2000-01-01\n---",
)

D1_ACCEPTED_EIP_FIXTURES = (
    b"---\neip: 1\ntitle: Minimal\n---\n",
    (
        b"---\r\n"
        b"# comment\r\n"
        b"eip: 123\r\n"
        b"title: Example: with colon\r\n"
        b"description: \"opaque quoted value\"\r\n"
        b"requires: 1, 002\r\n"
        b"tags:\r\n"
        b"  - one\r\n"
        b"---\r\n"
    ),
    (
        "---\neip: 44\ntitle: Cafe\u0301\n"
        "author: Example\n---\n"
    ).encode("utf-8"),
    b"--- \t\neip: 55 \t\ntitle: Trailing space normalization \t\n--- \t",
    (
        b"---\n"
        b"eip: 89\n"
        b"title: Continuation\n"
        b"description: first\n"
        b"\tsecond\n"
        b"---\n"
    ),
)

D1_NONBLANK_REJECTION_FIXTURES = (
    b"\xef\xbb\xbf---\neip: 1\n---\n",
    b" ---\neip: 1\n---\n",
    b"---\neip: 1\n",
    b"---\neip: 1\nEIP: 2\n---\n",
    b"---\n  orphan\n---\n",
    b"---\neip: 1\ntitle:\n---\n",
    b"---\neip: \"1\"\n---\n",
    b"---\neip: 1\x00\n---\n",
    b"---\neip: \xff\n---\n",
    b"---\neip: 1\nmalformed\n---\n",
    b"---\ntitle: Missing number\n---\n",
    b"---\neip: 0\n---\n",
    b"---\n# comment only\n---\n",
)

NORMALIZED_EMPTY_ACCEPTANCE_PAIRS = (
    (SYNTHETIC_D3_FAILURE_SHAPE, SYNTHETIC_D3_FAILURE_CONTROL),
    (
        b"---\neip: 2\n\ntitle: Middle separator\n---\n",
        b"---\neip: 2\ntitle: Middle separator\n---\n",
    ),
    (
        b"---\n\n\neip: 3\n\n\ntitle: Multiple separators\n\n---\n",
        b"---\neip: 3\ntitle: Multiple separators\n---\n",
    ),
    (
        b"---\neip: 4\n \t \ntitle: Whitespace separator\n---\n",
        b"---\neip: 4\ntitle: Whitespace separator\n---\n",
    ),
    (
        b"---\r\neip: 5\r\n\r\ntitle: CRLF separator\r\n---\r\n",
        b"---\r\neip: 5\r\ntitle: CRLF separator\r\n---\r\n",
    ),
    (
        (
            b"---\n"
            b"eip: 6\n"
            b"description:\n"
            b"\n"
            b"  continuation value\n"
            b"---\n"
        ),
        (
            b"---\n"
            b"eip: 6\n"
            b"description:\n"
            b"  continuation value\n"
            b"---\n"
        ),
    ),
)

BIP_ACCEPTED_FIXTURES = (
    (
        b"<pre>\n"
        b"  BIP: 0003\n"
        b"  Layer:\n"
        b"  Title: Example: with colon\n"
        b"  Authors: A <a@example.test>\n"
        b"           B <b@example.test>\n"
        b"  Requires: 1, 2\n"
        b"</pre>\n"
    ),
    (
        b"\n"
        b"  BIP: 4\n"
        b"  Title: Markdown example\n"
        b"  Status: Draft\n"
        b"\n"
        b"# Abstract\n"
    ),
)

BIP_REJECTION_FIXTURES = (
    b"\n\n\n\n  BIP: 1\n\n",
    b"<pre>\n  BIP: 1\n",
    b"  Title: Missing number\n\n",
    b"  BIP: zero\n\n",
    b"  BIP: 1\n  bip: 2\n\n",
    b"  BIP: 1\x00\n\n",
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(repository_path(path).read_bytes())


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(
        canonical_json_bytes(payload, pretty=False).rstrip(b"\n")
    )


def _validate_header_bounds(lines: Sequence[str]) -> None:
    if not lines or len(lines) > d1.MAX_HEADER_LINES:
        raise ValueError("PSIM header line count is invalid")
    normalized_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    if len(normalized_bytes) > d1.MAX_HEADER_BYTES:
        raise ValueError("PSIM header exceeds maximum bytes")


def parse_eip_preamble_d4(raw: bytes) -> dict[str, str]:
    """Parse historical EIP front matter with one explicit D4 delta.

    D1 normalization happens first.  Lines that are empty after that frozen
    normalization are ignored as non-semantic separators, but they still
    count against D1's header line and byte limits.  Every nonempty line is
    parsed by the unchanged D1 header state machine.
    """

    lines = d1.normalize_blob_bytes(raw)
    if not lines or lines[0] != "---":
        raise ValueError("PSIM EIP opening fence is not exact")
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line == "---"
        ),
        None,
    )
    if closing_index is None:
        raise ValueError("PSIM EIP closing fence is missing")
    header_lines = lines[1:closing_index]
    _validate_header_bounds(header_lines)
    nonempty_lines = [line for line in header_lines if line]
    fields = d1._parse_header_lines(
        nonempty_lines,
        field_lines_may_be_indented=False,
        allow_empty_values=False,
        comment_styles=("hash",),
    )
    if "eip" not in fields:
        raise ValueError("PSIM EIP number field is missing")
    d1.parse_positive_proposal_number(fields["eip"])
    return fields


# The BIP function is intentionally an identity alias, not a fork.
parse_bip_preamble_d4 = d1.parse_bip_preamble


def _raises_parser_error(function: Any, raw: bytes) -> bool:
    try:
        function(raw)
    except (UnicodeDecodeError, ValueError):
        return True
    return False


def _load_d3_terminal_binding() -> dict[str, Any]:
    if sha256_file(D3_TERMINAL_PATH) != D3_TERMINAL_SHA256:
        raise RuntimeError("PSIM-D3 terminal artifact hash changed")
    payload = json.loads(repository_path(D3_TERMINAL_PATH).read_text("utf-8"))
    if (
        payload.get("decision") != "reject"
        or payload.get("result_hash") != D3_TERMINAL_RESULT_HASH
        or payload.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
    ):
        raise RuntimeError("PSIM-D3 terminal boundary changed")
    ledger = payload.get("access_ledger")
    if not isinstance(ledger, Mapping):
        raise RuntimeError("PSIM-D3 access ledger is missing")
    forbidden = (
        "btc_market_rows_read",
        "cagr_values_built",
        "funding_rows_read",
        "future_return_rows_read",
        "model_outputs_built",
        "models_loaded",
        "pnl_rows_built",
        "reward_rows_built",
        "strict_mdd_values_built",
        "trade_rows_built",
    )
    if any(ledger.get(key) != 0 for key in forbidden):
        raise RuntimeError("PSIM-D3 terminal artifact crossed outcome boundary")
    return {
        "commit": D3_TERMINAL_COMMIT,
        "decision": "reject",
        "first_failure_gate_id": 4,
        "path": str(D3_TERMINAL_PATH),
        "result_hash": D3_TERMINAL_RESULT_HASH,
        "sha256": D3_TERMINAL_SHA256,
    }


def _run_parser_battery() -> dict[str, Any]:
    d1_eip_equal = 0
    for raw in D1_ACCEPTED_EIP_FIXTURES:
        if parse_eip_preamble_d4(raw) != d1.parse_eip_preamble(raw):
            raise RuntimeError("D4 changed a D1-accepted EIP parse result")
        d1_eip_equal += 1

    normalized_empty_equal = 0
    for with_empty, control in NORMALIZED_EMPTY_ACCEPTANCE_PAIRS:
        candidate = parse_eip_preamble_d4(with_empty)
        if candidate != parse_eip_preamble_d4(control):
            raise RuntimeError("D4 normalized-empty separator changed fields")
        if candidate != d1.parse_eip_preamble(control):
            raise RuntimeError("D4 separator control differs from D1")
        normalized_empty_equal += 1

    d1_nonblank_rejections_preserved = sum(
        _raises_parser_error(parse_eip_preamble_d4, raw)
        for raw in D1_NONBLANK_REJECTION_FIXTURES
    )
    if d1_nonblank_rejections_preserved != len(
        D1_NONBLANK_REJECTION_FIXTURES
    ):
        raise RuntimeError("D4 relaxed a nonblank D1 rejection")

    bip_acceptance_equal = 0
    for raw in BIP_ACCEPTED_FIXTURES:
        if parse_bip_preamble_d4(raw) != d1.parse_bip_preamble(raw):
            raise RuntimeError("D4 changed an accepted BIP parse result")
        bip_acceptance_equal += 1
    bip_rejections_preserved = sum(
        _raises_parser_error(parse_bip_preamble_d4, raw)
        for raw in BIP_REJECTION_FIXTURES
    )
    if bip_rejections_preserved != len(BIP_REJECTION_FIXTURES):
        raise RuntimeError("D4 changed a BIP rejection")

    within_limit = (
        b"---\n" + b"eip: 91\n" + (b"\n" * 255) + b"---\n"
    )
    beyond_limit = (
        b"---\n" + b"eip: 92\n" + (b"\n" * 256) + b"---\n"
    )
    if parse_eip_preamble_d4(within_limit)["eip"] != "91":
        raise RuntimeError("D4 exact header-line limit did not parse")
    if not _raises_parser_error(parse_eip_preamble_d4, beyond_limit):
        raise RuntimeError("D4 empty lines bypassed the header-line limit")

    oversized_comment = b"#" + (b"x" * 50_000)
    beyond_byte_limit = (
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
    if not _raises_parser_error(parse_eip_preamble_d4, beyond_byte_limit):
        raise RuntimeError("D4 empty lines bypassed the header-byte limit")

    empty_value_after_separator = (
        b"---\neip: 94\ndescription:\n\n---\n"
    )
    if not _raises_parser_error(
        parse_eip_preamble_d4,
        empty_value_after_separator,
    ):
        raise RuntimeError("D4 separator incorrectly rescued an empty value")

    return {
        "bip_acceptance_outputs_unchanged": bip_acceptance_equal,
        "bip_parser_identity_alias": parse_bip_preamble_d4
        is d1.parse_bip_preamble,
        "bip_rejections_preserved": bip_rejections_preserved,
        "d1_accepted_eip_outputs_unchanged": d1_eip_equal,
        "d1_nonblank_eip_rejections_preserved": (
            d1_nonblank_rejections_preserved
        ),
        "header_byte_limit_counts_normalized_empty_lines": True,
        "header_line_limit_counts_normalized_empty_lines": True,
        "normalized_empty_acceptance_pairs_equal": normalized_empty_equal,
        "normalized_empty_separator_does_not_rescue_empty_value": True,
        "synthetic_d3_failure_shape_accepted": (
            parse_eip_preamble_d4(SYNTHETIC_D3_FAILURE_SHAPE)
            == d1.parse_eip_preamble(SYNTHETIC_D3_FAILURE_CONTROL)
        ),
    }


def build_probe() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "access_boundary": {
            "d3_forensic_root_accessed": False,
            "d3_terminal_artifact_read": True,
            "market_data_accessed": False,
            "model_accessed": False,
            "official_historical_proposal_source_accessed": False,
            "outcomes_accessed": False,
        },
        "candidate": {
            "id": "PSIM-D4",
            "name": (
                "Protocol Specification Intent-Maturity relation RLLM, "
                "historical EIP normalized-empty separator grammar"
            ),
            "parser_only_successor": True,
        },
        "d1_parser_binding": {
            "path": str(D1_PARSER_PATH),
            "sha256": sha256_file(D1_PARSER_PATH),
            "version": "PSIM_PREAMBLE_STATE_MACHINE_V1",
        },
        "d3_terminal_binding": _load_d3_terminal_binding(),
        "delta_contract": {
            "bip_parser": "IDENTICAL_FUNCTION_OBJECT_TO_D1",
            "blank_line_scope": (
                "EIP_FRONT_MATTER_ONLY_AFTER_D1_NORMALIZATION"
            ),
            "empty_line_semantics": "IGNORE_AS_NONSEMANTIC_SEPARATOR",
            "header_bounds_applied_before_empty_line_filter": True,
            "maximum_header_bytes_unchanged": d1.MAX_HEADER_BYTES,
            "maximum_header_lines_unchanged": d1.MAX_HEADER_LINES,
            "nonempty_line_parser": "UNCHANGED_D1_HEADER_STATE_MACHINE",
            "normalization": "UNCHANGED_D1_NORMALIZE_BLOB_BYTES",
            "physical_empty_or_ascii_horizontal_whitespace_only": (
                "BOTH_NORMALIZE_TO_EMPTY_UNDER_FROZEN_D1_NORMALIZER"
            ),
        },
        "known_failure_provenance": {
            "evidence": KNOWN_D3_FAILURE_EVIDENCE,
            "fixture": (
                "SYNTHETIC_STRUCTURE_ONLY_NOT_HISTORICAL_BLOB_BYTES"
            ),
            "historical_blob_opened_by_probe": False,
            "synthetic_fixture_sha256": sha256_bytes(
                SYNTHETIC_D3_FAILURE_SHAPE
            ),
        },
        "official_reference_notes": list(OFFICIAL_REFERENCE_NOTES),
        "parser_battery": _run_parser_battery(),
        "parser_version": PARSER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "selection_scope": (
            "AUTHORIZE_D4_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
        ),
        "synthetic_only": True,
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def write_probe(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_probe()
    target = repository_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("existing PSIM-D4 parser probe differs")
    target.write_bytes(raw)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()
    payload = write_probe(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "result_hash": payload["result_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
