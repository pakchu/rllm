"""Select the PSIM-D7 Bitcoin grammar mechanism with synthetic data only.

The probe binds the terminal PSIM-D6 Bitcoin grammar census but never opens a
forensic Git root, historical proposal text, market data, a model, rewards,
trades, PnL, CAGR, strict-MDD, or outcomes. It tests grammar-level rules only:

* a unique parseable later ``<pre>`` BIP header may become the metadata anchor
  when the frozen initial parser fails with the exact historical error; and
* Bitcoin dependency fields may contain bare decimal or exact uppercase
  ``BIP-``-prefixed decimal tokens while preserving original source text.

The exact PSIM-D6 ERC restoration and lossless chunk transport remain frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training import (
    audit_protocol_specification_intent_maturity_d6_bitcoin_grammar_census
    as d6_census,
)
from training import (
    preregister_protocol_specification_intent_maturity_d6 as d6_prereg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d7_mechanism_probe_"
    "2026-07-26.json"
)
PROTOCOL_VERSION = "psim_d7_bitcoin_grammar_mechanism_probe_v1"
MECHANISM_VERSION = (
    "PSIM_D7_UNIQUE_LATER_BIP_HEADER_PLUS_PREFIXED_DEPENDENCY_V1"
)

D6_CENSUS_COMMIT = "bfa35dc12f1c5a2cf0cee0be4bbd85347f8ab7c3"
D6_CENSUS_PATH = d6_census.DEFAULT_OUTPUT
D6_CENSUS_SHA256 = (
    "8bfe4a6c44a4c5381bb98caf2ffea57b42f2b3d77caec9e656895336b72d0217"
)
D6_CENSUS_RESULT_HASH = (
    "7ef74a017f8c0c1eb416608dcf59c2ce74af6587f5a71203b53e846d31c039ed"
)
D6_CENSUS_SCRIPT_PATH = Path(
    "training/"
    "audit_protocol_specification_intent_maturity_d6_bitcoin_grammar_"
    "census.py"
)
D6_CENSUS_SCRIPT_SHA256 = (
    "2e2ad700241b5a3dbe62ce67feb0dacf67c6acfb3fbe21943d48ff2fcedae01a"
)
D6_CENSUS_TEST_PATH = Path(
    "tests/"
    "test_audit_protocol_specification_intent_maturity_d6_bitcoin_grammar_"
    "census.py"
)
D6_CENSUS_TEST_SHA256 = (
    "9b37235f1c077f14d036d3eac173de1f4e2d1bf1e794457d2030d20192457cc6"
)
D6_CENSUS_DOCUMENT_PATH = Path(
    "docs/"
    "post-psim-d6-bitcoin-grammar-census-and-d7-requirements-2026-07-26.md"
)
D6_CENSUS_DOCUMENT_SHA256 = (
    "c87f497f90f7e8338379bc56f9dcda3cb31d6594d538afce1ab3fa435d545bb3"
)

D6_MECHANISM_COMMIT = d6_prereg.MECHANISM_PROBE_COMMIT
D6_MECHANISM_PATH = d6_prereg.MECHANISM_PROBE_PATH
D6_MECHANISM_SHA256 = d6_prereg.MECHANISM_PROBE_SHA256
D6_MECHANISM_RESULT_HASH = d6_prereg.MECHANISM_PROBE_RESULT_HASH
D6_MECHANISM_SCRIPT_PATH = d6_prereg.MECHANISM_PROBE_SCRIPT_PATH
D6_MECHANISM_SCRIPT_SHA256 = d6_prereg.MECHANISM_PROBE_SCRIPT_SHA256
D6_MECHANISM_TEST_PATH = d6_prereg.MECHANISM_PROBE_TEST_PATH
D6_MECHANISM_TEST_SHA256 = d6_prereg.MECHANISM_PROBE_TEST_SHA256
D6_MECHANISM_DOCUMENT_PATH = d6_prereg.MECHANISM_PROBE_DOCUMENT_PATH
D6_MECHANISM_DOCUMENT_SHA256 = (
    d6_prereg.MECHANISM_PROBE_DOCUMENT_SHA256
)
D6_MECHANISM_VERSION = d6_prereg.MECHANISM_VERSION
D6_SOURCE_MECHANISM_CONTRACT_HASH = (
    d6_prereg.SOURCE_MECHANISM_CONTRACT_HASH
)

INITIAL_ANCHOR = "INITIAL_FROZEN_BIP_PARSER"
LATER_PRE_ANCHOR = "UNIQUE_LATER_EXACT_PRE_BIP_HEADER"
EXACT_INITIAL_ERROR = "ValueError: PSIM malformed header line"


class D7MechanismError(ValueError):
    """A typed synthetic mechanism rejection."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class HeaderCandidate:
    opening_line_index: int
    closing_line_index: int
    header: Mapping[str, str]
    proposal_number: int


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return d6_census.d6.sha256_bytes(raw)


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(repository_path(path).read_bytes())


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return d6_census.d6.canonical_json_bytes(payload, pretty=pretty)


def canonical_hash(payload: Any) -> str:
    return d6_census.d6.canonical_hash(payload)


def _safe_output_path(path: str | Path) -> Path:
    requested = Path(path)
    results_root = REPO_ROOT.resolve() / "results"
    target = results_root / requested.name
    if (
        requested.is_absolute()
        or requested.parent != Path("results")
        or requested.suffix != ".json"
        or results_root.is_symlink()
        or not results_root.is_dir()
        or target.is_symlink()
        or (target.exists() and not target.is_file())
    ):
        raise RuntimeError(
            "PSIM-D7 mechanism output must be a safe flat JSON result"
        )
    return target


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D7 authority is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"PSIM-D7 authority is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D7 authority is noncanonical: {path}")
    return payload


def _load_d6_census_binding() -> dict[str, Any]:
    payload = _read_canonical_json(D6_CENSUS_PATH)
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    observed = payload.get("census")
    access = payload.get("access_boundary")
    if (
        sha256_file(D6_CENSUS_PATH) != D6_CENSUS_SHA256
        or payload.get("result_hash") != D6_CENSUS_RESULT_HASH
        or payload.get("result_hash") != canonical_hash(core)
        or payload.get("protocol_version")
        != d6_census.PROTOCOL_VERSION
        or payload.get("policy_id") != d6_census.POLICY_ID
        or not isinstance(observed, Mapping)
        or observed.get("grammar_class_counts")
        != d6_census.EXPECTED_GRAMMAR_CLASS_COUNTS
        or observed.get("unknown_grammar_count") != 0
        or observed.get("error_event_count") != 7
        or observed.get("error_proposal_count") != 2
        or not isinstance(access, Mapping)
        or access.get("network_commands") != 0
        or access.get("market_data_accessed") is not False
        or access.get("model_accessed") is not False
        or access.get("outcomes_accessed") is not False
        or access.get("raw_or_normalized_text_published") is not False
        or access.get("source_objects_mutated") is not False
        or payload.get("candidate_selection", {}).get(
            "d7_candidate_authorized"
        )
        is not False
    ):
        raise RuntimeError("PSIM-D6 Bitcoin census authority changed")
    if (
        sha256_file(D6_CENSUS_SCRIPT_PATH)
        != D6_CENSUS_SCRIPT_SHA256
        or sha256_file(D6_CENSUS_TEST_PATH) != D6_CENSUS_TEST_SHA256
        or sha256_file(D6_CENSUS_DOCUMENT_PATH)
        != D6_CENSUS_DOCUMENT_SHA256
    ):
        raise RuntimeError("PSIM-D6 Bitcoin census producer changed")
    return {
        "commit": D6_CENSUS_COMMIT,
        "document": {
            "path": D6_CENSUS_DOCUMENT_PATH.as_posix(),
            "sha256": D6_CENSUS_DOCUMENT_SHA256,
        },
        "grammar_class_counts": dict(
            observed["grammar_class_counts"]
        ),
        "path": D6_CENSUS_PATH.as_posix(),
        "result_hash": D6_CENSUS_RESULT_HASH,
        "script": {
            "path": D6_CENSUS_SCRIPT_PATH.as_posix(),
            "sha256": D6_CENSUS_SCRIPT_SHA256,
        },
        "sha256": D6_CENSUS_SHA256,
        "test": {
            "path": D6_CENSUS_TEST_PATH.as_posix(),
            "sha256": D6_CENSUS_TEST_SHA256,
        },
        "unknown_grammar_count": 0,
    }


def _load_d6_mechanism_binding() -> dict[str, Any]:
    payload = _read_canonical_json(D6_MECHANISM_PATH)
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    access = payload.get("access_boundary")
    contract = payload.get("mechanism_contract")
    if (
        sha256_file(D6_MECHANISM_PATH) != D6_MECHANISM_SHA256
        or payload.get("result_hash") != D6_MECHANISM_RESULT_HASH
        or payload.get("result_hash") != canonical_hash(core)
        or payload.get("protocol_version")
        != d6_prereg.MECHANISM_PROBE_PROTOCOL_VERSION
        or payload.get("mechanism_version") != D6_MECHANISM_VERSION
        or payload.get("synthetic_only") is not True
        or payload.get("selection_scope")
        != "AUTHORIZE_D6_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
        or not isinstance(access, Mapping)
        or any(
            access.get(name) is not False
            for name in (
                "d5_forensic_root_accessed",
                "d5_run_invoked",
                "external_network_accessed_by_probe",
                "historical_proposal_text_accessed",
                "market_data_accessed",
                "model_accessed",
                "official_historical_proposal_source_accessed",
                "outcomes_accessed",
                "raw_official_text_published",
            )
        )
        or not isinstance(contract, Mapping)
        or contract.get("full_text_reconstruction")
        != "BYTE_FOR_BYTE_REQUIRED"
        or contract.get("max_bytes_per_chunk") != 8_192
        or contract.get("max_chunks_per_event") != 8
        or contract.get("ninth_chunk")
        != "FAIL_CLOSED_NO_TRUNCATION_OR_SUMMARIZATION"
        or contract.get("administrative_restoration")
        != (
            "EXACT_THREE_STEP_CAUSAL_EPISODE_PLUS_PER_PROPOSAL_"
            "FROZEN_RECEIPT_HASH"
        )
    ):
        raise RuntimeError("PSIM-D6 mechanism authority changed")
    if (
        sha256_file(D6_MECHANISM_SCRIPT_PATH)
        != D6_MECHANISM_SCRIPT_SHA256
        or sha256_file(D6_MECHANISM_TEST_PATH)
        != D6_MECHANISM_TEST_SHA256
        or sha256_file(D6_MECHANISM_DOCUMENT_PATH)
        != D6_MECHANISM_DOCUMENT_SHA256
    ):
        raise RuntimeError("PSIM-D6 mechanism producer changed")
    return {
        "commit": D6_MECHANISM_COMMIT,
        "document": {
            "path": D6_MECHANISM_DOCUMENT_PATH.as_posix(),
            "sha256": D6_MECHANISM_DOCUMENT_SHA256,
        },
        "mechanism_version": D6_MECHANISM_VERSION,
        "path": D6_MECHANISM_PATH.as_posix(),
        "result_hash": D6_MECHANISM_RESULT_HASH,
        "script": {
            "path": D6_MECHANISM_SCRIPT_PATH.as_posix(),
            "sha256": D6_MECHANISM_SCRIPT_SHA256,
        },
        "sha256": D6_MECHANISM_SHA256,
        "source_mechanism_contract_hash": (
            D6_SOURCE_MECHANISM_CONTRACT_HASH
        ),
        "test": {
            "path": D6_MECHANISM_TEST_PATH.as_posix(),
            "sha256": D6_MECHANISM_TEST_SHA256,
        },
    }


def _normalize(raw: bytes) -> tuple[str, ...]:
    try:
        return tuple(d6_census.d6.core.prereg.normalize_blob_bytes(raw))
    except (UnicodeDecodeError, ValueError) as error:
        raise D7MechanismError("ERROR_STRICT_SOURCE_NORMALIZATION") from error


def _path_proposal(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise D7MechanismError("ERROR_PATH_PROPOSAL_IDENTITY")
    return value


def _exact_pre_blocks(
    lines: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    blocks: list[tuple[int, int]] = []
    opening: int | None = None
    for index, line in enumerate(lines):
        if line == "<pre>":
            if opening is not None:
                raise D7MechanismError("ERROR_PRE_FENCE_GRAMMAR")
            opening = index
        elif line == "</pre>":
            if opening is None:
                raise D7MechanismError("ERROR_PRE_FENCE_GRAMMAR")
            blocks.append((opening, index))
            opening = None
    if opening is not None:
        raise D7MechanismError("ERROR_PRE_FENCE_GRAMMAR")
    return tuple(blocks)


def _parseable_pre_headers(
    lines: Sequence[str],
    blocks: Sequence[tuple[int, int]],
) -> tuple[HeaderCandidate, ...]:
    candidates: list[HeaderCandidate] = []
    for opening, closing in blocks:
        isolated = (
            "\n".join(lines[opening : closing + 1]) + "\n"
        ).encode("utf-8")
        try:
            header = d6_census.d6.parse_bip_preamble(isolated)
            proposal = (
                d6_census.d6.core.prereg.parse_positive_proposal_number(
                    header["bip"]
                )
            )
        except (KeyError, ValueError):
            continue
        candidates.append(
            HeaderCandidate(
                opening_line_index=opening,
                closing_line_index=closing,
                header=dict(sorted(header.items())),
                proposal_number=proposal,
            )
        )
    return tuple(candidates)


def select_bip_header_d7(
    path_proposal_number: int,
    raw: bytes,
) -> dict[str, Any]:
    """Select only the frozen initial header or one unique later header."""

    proposal = _path_proposal(path_proposal_number)
    lines = _normalize(raw)
    try:
        initial = d6_census.d6.parse_bip_preamble(raw)
    except ValueError as error:
        initial_error = f"{type(error).__name__}: {error}"
    else:
        try:
            parsed = (
                d6_census.d6.core.prereg.parse_positive_proposal_number(
                    initial["bip"]
                )
            )
        except (KeyError, ValueError) as error:
            raise D7MechanismError(
                "ERROR_INITIAL_HEADER_PROPOSAL"
            ) from error
        if parsed != proposal:
            raise D7MechanismError("ERROR_PATH_HEADER_MISMATCH")
        header = dict(sorted(initial.items()))
        receipt_core = {
            "anchor": INITIAL_ANCHOR,
            "header_hash": canonical_hash(header),
            "normalized_line_count": len(lines),
            "normalized_text_hash": canonical_hash(list(lines)),
            "path_proposal_matches": True,
            "raw_sha256": sha256_bytes(raw),
        }
        return {
            "audit_receipt": {
                **receipt_core,
                "receipt_hash": canonical_hash(receipt_core),
            },
            "header": header,
            "normalized_lines": lines,
        }

    if initial_error != EXACT_INITIAL_ERROR:
        raise D7MechanismError(
            "ERROR_INITIAL_FAILURE_NOT_AUTHORIZED_FOR_FALLBACK"
        )
    blocks = _exact_pre_blocks(lines)
    candidates = _parseable_pre_headers(lines, blocks)
    if len(candidates) != 1:
        raise D7MechanismError("ERROR_LATER_HEADER_NOT_UNIQUE")
    candidate = candidates[0]
    if (
        candidate.proposal_number != proposal
        or candidate.opening_line_index <= 0
        or not any(lines[: candidate.opening_line_index])
    ):
        raise D7MechanismError("ERROR_LATER_HEADER_PATH_OR_PREFIX")
    if any(
        opening < candidate.opening_line_index
        for opening, _closing in blocks
    ):
        raise D7MechanismError("ERROR_LATER_HEADER_PREFIX_FENCE")
    header = dict(candidate.header)
    receipt_core = {
        "anchor": LATER_PRE_ANCHOR,
        "closing_line_index": candidate.closing_line_index,
        "header_hash": canonical_hash(header),
        "normalized_line_count": len(lines),
        "normalized_text_hash": canonical_hash(list(lines)),
        "opening_line_index": candidate.opening_line_index,
        "parseable_later_header_count": 1,
        "path_proposal_matches": True,
        "prefix_nonblank_line_count": sum(
            1
            for line in lines[: candidate.opening_line_index]
            if line
        ),
        "raw_sha256": sha256_bytes(raw),
    }
    return {
        "audit_receipt": {
            **receipt_core,
            "receipt_hash": canonical_hash(receipt_core),
        },
        "header": header,
        "normalized_lines": lines,
    }


def parse_dependency_ids_d7(
    value: str,
    *,
    self_id: int,
) -> tuple[int, ...]:
    """Parse Bitcoin dependency IDs without changing their source text."""

    _path_proposal(self_id)
    if not isinstance(value, str):
        raise D7MechanismError("ERROR_DEPENDENCY_VALUE_TYPE")
    if not value:
        return ()
    if "\n" in value or "\r" in value:
        raise D7MechanismError("ERROR_DEPENDENCY_MULTILINE")
    tokens = value.split(",")
    if len(tokens) > d6_census.d6.core.prereg.MAX_DEPENDENCIES:
        raise D7MechanismError("ERROR_DEPENDENCY_COUNT")
    parsed: list[int] = []
    for token in tokens:
        stripped = token.strip(" \t")
        if re.fullmatch(r"[0-9]+", stripped, re.ASCII):
            normalized = stripped
        elif re.fullmatch(r"BIP-[0-9]+", stripped, re.ASCII):
            normalized = stripped.removeprefix("BIP-")
        else:
            raise D7MechanismError("ERROR_DEPENDENCY_TOKEN_GRAMMAR")
        try:
            dependency = (
                d6_census.d6.core.prereg.parse_positive_proposal_number(
                    normalized
                )
            )
        except ValueError as error:
            raise D7MechanismError(
                "ERROR_DEPENDENCY_POSITIVE_DECIMAL"
            ) from error
        if dependency == self_id:
            raise D7MechanismError("ERROR_DEPENDENCY_SELF")
        parsed.append(dependency)
    if len(parsed) != len(set(parsed)):
        raise D7MechanismError("ERROR_DEPENDENCY_DUPLICATE")
    return tuple(sorted(parsed))


def parse_bip_dependencies_d7(
    header: Mapping[str, str],
    *,
    self_id: int,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(header, Mapping):
        raise D7MechanismError("ERROR_HEADER_TYPE")
    return {
        field: parse_dependency_ids_d7(
            header.get(field, ""),
            self_id=self_id,
        )
        for field in d6_census.d6.core.prereg.BIP_DEPENDENCY_FIELDS
    }


def decode_bitcoin_semantics_d7(
    path_proposal_number: int,
    raw: bytes,
) -> dict[str, Any]:
    """Build a synthetic audit receipt without exposing normalized text."""

    selection = select_bip_header_d7(path_proposal_number, raw)
    dependencies = parse_bip_dependencies_d7(
        selection["header"],
        self_id=path_proposal_number,
    )
    dependency_rows = [
        {
            "field": field,
            "proposal": proposal,
        }
        for field, values in sorted(dependencies.items())
        for proposal in values
    ]
    receipt_core = {
        "dependency_edge_count": len(dependency_rows),
        "dependency_edge_hash": canonical_hash(dependency_rows),
        "header_anchor": selection["audit_receipt"]["anchor"],
        "header_receipt_hash": selection["audit_receipt"][
            "receipt_hash"
        ],
        "normalized_line_count": len(selection["normalized_lines"]),
        "normalized_text_hash": canonical_hash(
            list(selection["normalized_lines"])
        ),
        "path_proposal_matches": True,
        "raw_sha256": sha256_bytes(raw),
    }
    return {
        **receipt_core,
        "receipt_hash": canonical_hash(receipt_core),
    }


def _synthetic_bip(
    proposal: int,
    *,
    prefix: str = "",
    dependency: str | None = None,
    second_header_proposal: int | None = None,
) -> bytes:
    dependency_line = (
        ""
        if dependency is None
        else f"  Requires: {dependency}\n"
    )
    first = (
        "<pre>\n"
        f"  BIP: {proposal}\n"
        "  Title: Synthetic mechanism fixture\n"
        f"{dependency_line}"
        "</pre>\n"
    )
    second = ""
    if second_header_proposal is not None:
        second = (
            "<pre>\n"
            f"  BIP: {second_header_proposal}\n"
            "  Title: Independent synthetic fixture\n"
            "</pre>\n"
        )
    return f"{prefix}{first}{second}Body\n".encode("utf-8")


def _run_scenario(
    scenario_id: str,
    function: Any,
    arguments: tuple[Any, ...],
    *,
    expected_anchor: str | None = None,
    expected_error: str | None = None,
) -> dict[str, Any]:
    try:
        observed = function(*arguments)
    except D7MechanismError as error:
        passed = expected_error == error.code
        receipt_hash = canonical_hash(
            {
                "error_code": error.code,
                "scenario_id": scenario_id,
            }
        )
        outcome = error.code
    else:
        anchor = observed.get("header_anchor")
        if anchor is None and isinstance(
            observed.get("audit_receipt"),
            Mapping,
        ):
            anchor = observed["audit_receipt"].get("anchor")
        passed = expected_error is None and anchor == expected_anchor
        receipt_hash = str(observed["receipt_hash"])
        outcome = str(anchor)
    if not passed:
        raise RuntimeError(
            f"PSIM-D7 synthetic scenario failed: {scenario_id}"
        )
    return {
        "expected": (
            expected_error
            if expected_error is not None
            else expected_anchor
        ),
        "observed": outcome,
        "passed": True,
        "receipt_hash": receipt_hash,
        "scenario_id": scenario_id,
    }


def _synthetic_battery() -> dict[str, Any]:
    proposal = 900
    lower_dependency = proposal - 1
    upper_dependency = proposal + 1
    maximum = d6_census.d6.core.prereg.MAX_DEPENDENCIES
    rows = [
        _run_scenario(
            "normal_initial_header",
            decode_bitcoin_semantics_d7,
            (proposal, _synthetic_bip(proposal)),
            expected_anchor=INITIAL_ANCHOR,
        ),
        _run_scenario(
            "unique_later_header_after_generic_prefix",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    prefix="Synthetic preface\nSecond synthetic line\n",
                ),
            ),
            expected_anchor=LATER_PRE_ANCHOR,
        ),
        _run_scenario(
            "later_header_path_mismatch",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    upper_dependency,
                    prefix="Synthetic preface\n",
                ),
            ),
            expected_error="ERROR_LATER_HEADER_PATH_OR_PREFIX",
        ),
        _run_scenario(
            "multiple_parseable_later_headers",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    prefix="Synthetic preface\n",
                    second_header_proposal=proposal,
                ),
            ),
            expected_error="ERROR_LATER_HEADER_NOT_UNIQUE",
        ),
        _run_scenario(
            "matching_and_mismatching_later_headers",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    prefix="Synthetic preface\n",
                    second_header_proposal=upper_dependency,
                ),
            ),
            expected_error="ERROR_LATER_HEADER_NOT_UNIQUE",
        ),
        _run_scenario(
            "unauthorized_initial_error",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                ("\n" * 4 + _synthetic_bip(proposal).decode()).encode(),
            ),
            expected_error=(
                "ERROR_INITIAL_FAILURE_NOT_AUTHORIZED_FOR_FALLBACK"
            ),
        ),
        _run_scenario(
            "strict_nul_rejection",
            decode_bitcoin_semantics_d7,
            (proposal, _synthetic_bip(proposal) + b"\x00"),
            expected_error="ERROR_STRICT_SOURCE_NORMALIZATION",
        ),
        _run_scenario(
            "strict_utf8_rejection",
            decode_bitcoin_semantics_d7,
            (proposal, _synthetic_bip(proposal) + b"\xff"),
            expected_error="ERROR_STRICT_SOURCE_NORMALIZATION",
        ),
        _run_scenario(
            "unclosed_opening_before_later_header",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                (
                    "Synthetic preface\n<pre>\nUnclosed block\n"
                ).encode()
                + _synthetic_bip(proposal),
            ),
            expected_error="ERROR_PRE_FENCE_GRAMMAR",
        ),
        _run_scenario(
            "unclosed_opening_after_later_header",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    prefix="Synthetic preface\n",
                )
                + b"<pre>\nUnclosed tail\n",
            ),
            expected_error="ERROR_PRE_FENCE_GRAMMAR",
        ),
        _run_scenario(
            "stray_closing_fence",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    prefix="Synthetic preface\n</pre>\n",
                ),
            ),
            expected_error="ERROR_PRE_FENCE_GRAMMAR",
        ),
        _run_scenario(
            "nested_fences",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                (
                    "Synthetic preface\n<pre>\n<pre>\n"
                    "Nested synthetic block\n</pre>\n</pre>\n"
                ).encode()
                + _synthetic_bip(proposal),
            ),
            expected_error="ERROR_PRE_FENCE_GRAMMAR",
        ),
        _run_scenario(
            "malformed_header_block_before_later_header",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                (
                    "Synthetic preface\n<pre>\nMalformed field\n</pre>\n"
                ).encode()
                + _synthetic_bip(proposal),
            ),
            expected_error="ERROR_LATER_HEADER_PREFIX_FENCE",
        ),
        _run_scenario(
            "duplicate_header_field_before_later_header",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                (
                    "Synthetic preface\n"
                    "<pre>\n"
                    f"  BIP: {proposal}\n"
                    f"  BIP: {proposal}\n"
                    "</pre>\n"
                ).encode()
                + _synthetic_bip(proposal),
            ),
            expected_error="ERROR_LATER_HEADER_PREFIX_FENCE",
        ),
        _run_scenario(
            "exact_prefixed_dependency",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=f"BIP-{lower_dependency}",
                ),
            ),
            expected_anchor=INITIAL_ANCHOR,
        ),
        _run_scenario(
            "mixed_bare_and_prefixed_dependencies",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=(
                        f"{lower_dependency}, BIP-{upper_dependency}"
                    ),
                ),
            ),
            expected_anchor=INITIAL_ANCHOR,
        ),
        _run_scenario(
            "lowercase_prefix_rejected",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=f"bip-{lower_dependency}",
                ),
            ),
            expected_error="ERROR_DEPENDENCY_TOKEN_GRAMMAR",
        ),
        _run_scenario(
            "spaced_prefix_rejected",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=f"BIP -{lower_dependency}",
                ),
            ),
            expected_error="ERROR_DEPENDENCY_TOKEN_GRAMMAR",
        ),
        _run_scenario(
            "range_token_rejected",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=(
                        f"BIP-{lower_dependency}-{upper_dependency}"
                    ),
                ),
            ),
            expected_error="ERROR_DEPENDENCY_TOKEN_GRAMMAR",
        ),
        _run_scenario(
            "self_dependency_rejected",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=f"BIP-{proposal}",
                ),
            ),
            expected_error="ERROR_DEPENDENCY_SELF",
        ),
        _run_scenario(
            "cross_style_duplicate_rejected",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=(
                        f"{lower_dependency}, BIP-{lower_dependency}"
                    ),
                ),
            ),
            expected_error="ERROR_DEPENDENCY_DUPLICATE",
        ),
        _run_scenario(
            "maximum_dependency_count",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=", ".join(
                        str(index)
                        for index in range(1, maximum + 1)
                    ),
                ),
            ),
            expected_anchor=INITIAL_ANCHOR,
        ),
        _run_scenario(
            "dependency_count_overflow",
            decode_bitcoin_semantics_d7,
            (
                proposal,
                _synthetic_bip(
                    proposal,
                    dependency=", ".join(
                        str(index)
                        for index in range(1, maximum + 2)
                    ),
                ),
            ),
            expected_error="ERROR_DEPENDENCY_COUNT",
        ),
    ]
    outcome_counts = Counter(row["observed"] for row in rows)
    return {
        "all_passed": all(row["passed"] for row in rows),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "scenario_count": len(rows),
        "scenario_roster_hash": canonical_hash(rows),
        "scenarios": rows,
    }


def build_probe() -> dict[str, Any]:
    census_binding = _load_d6_census_binding()
    d6_mechanism_binding = _load_d6_mechanism_binding()
    battery = _synthetic_battery()
    if not battery["all_passed"] or battery["scenario_count"] != 23:
        raise RuntimeError("PSIM-D7 synthetic battery is incomplete")
    payload: dict[str, Any] = {
        "access_boundary": {
            "d6_census_artifact_read": True,
            "d6_forensic_root_accessed": False,
            "d6_run_invoked": False,
            "external_network_accessed_by_probe": False,
            "historical_proposal_text_accessed": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "official_historical_proposal_source_accessed": False,
            "outcomes_accessed": False,
            "raw_official_text_published": False,
        },
        "candidate": {
            "id": "PSIM-D7",
            "name": (
                "UNIQUE_LATER_BIP_HEADER_AND_PREFIXED_DEPENDENCY_GRAMMAR"
            ),
            "source_representation_successor": True,
        },
        "d6_census_binding": census_binding,
        "d6_mechanism_binding": d6_mechanism_binding,
        "mechanism_contract": {
            "dependency_allowed_bare_token": "[0-9]+",
            "dependency_allowed_prefixed_token": "BIP-[0-9]+",
            "dependency_prefix_case": "EXACT_UPPERCASE",
            "dependency_prefix_effect": (
                "INTEGER_EDGE_VALIDATION_ONLY_SOURCE_TEXT_UNCHANGED"
            ),
            "dependency_token_outer_whitespace": (
                "INHERIT_D6_STRIP_SURROUNDING_SP_HTAB"
            ),
            "dependency_unknown_token": "FAIL_CLOSED",
            "fallback_pre_fence_grammar": (
                "BALANCED_NON_NESTED_EXACT_PAIRS"
            ),
            "fallback_pre_header_blocks_before_selected": "NONE",
            "header_candidate_parser": "UNCHANGED_FROZEN_BIP_PARSER",
            "header_candidate_selection": (
                "EXACTLY_ONE_PARSEABLE_LATER_PRE_HEADER_TOTAL"
            ),
            "header_path_binding": "BIP_FIELD_EQUALS_PATH_PROPOSAL",
            "header_prefix_and_body_retention": "FULL_NORMALIZED_TEXT_RETAINED",
            "identity_conditioned_allowlist": False,
            "initial_parser_first": True,
            "later_header_authorized_initial_error": EXACT_INITIAL_ERROR,
            "multiple_parseable_headers": "FAIL_CLOSED",
            "unknown_grammar": "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES",
        },
        "mechanism_version": MECHANISM_VERSION,
        "policy_id": "PSIM-D7-SYNTHETIC-GRAMMAR-MECHANISM-PROBE",
        "protocol_version": PROTOCOL_VERSION,
        "selection_scope": (
            "AUTHORIZE_D7_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
        ),
        "synthetic_battery": battery,
        "synthetic_only": True,
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def write_probe(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    target = _safe_output_path(path)
    payload = build_probe()
    raw = canonical_json_bytes(payload)
    if os.path.lexists(target):
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise RuntimeError(
                "existing PSIM-D7 mechanism artifact differs"
            )
        return payload
    temporary = target.with_name(target.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError("unsafe PSIM-D7 mechanism temporary path")
    temporary.write_bytes(raw)
    os.replace(temporary, target)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the synthetic PSIM-D7 Bitcoin grammar mechanism",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    payload = write_probe(arguments.output)
    print(
        json.dumps(
            {
                "mechanism_version": payload["mechanism_version"],
                "result_hash": payload["result_hash"],
                "scenario_count": payload["synthetic_battery"][
                    "scenario_count"
                ],
                "selection_scope": payload["selection_scope"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
