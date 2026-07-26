"""Probe synthetic-only PSIM-D5 path/text-delta event semantics.

PSIM-D4 is terminally rejected and is never repaired or rerun. This module
binds its terminal rejection and the complete post-terminal grammar census,
then evaluates one successor source representation using synthetic bytes
only. It never reads a proposal repository, forensic source root, market,
model, reward, trading, PnL, CAGR, strict-MDD, or outcome data.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training import (
    audit_protocol_specification_intent_maturity_d4_grammar_census as census,
)
from training import (
    build_protocol_specification_intent_maturity_d4_source_support as d4,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d5_event_semantics_"
    "probe_2026-07-26.json"
)
PROTOCOL_VERSION = "psim_d5_path_text_event_semantics_probe_v1"
SEMANTICS_VERSION = (
    "PSIM_PATH_IDENTITY_NORMALIZED_TEXT_DELTA_V1_EXACT_ERC_QUARANTINE"
)

D4_TERMINAL_COMMIT = "e406778f76f252d3b0cddb33242edcb51e984c80"
D4_TERMINAL_PATH = d4.DEFAULT_REJECTION_PATH
D4_TERMINAL_SHA256 = (
    "4d947075c0f54c5cd09c732710da0502c87d89fa52029fe81367dd3f27ab2aaf"
)
D4_TERMINAL_RESULT_HASH = (
    "8563ef3ace444896295d7076cd0f839e8f62f89899e312d711f9768f5cbf84aa"
)

CENSUS_COMMIT = "6be3e767d4320da0ce9aa34d2cfabbf4ac0fb3ef"
CENSUS_PATH = census.DEFAULT_OUTPUT
CENSUS_SHA256 = (
    "eaa5946844bb218b1ae211c84d509c49482111af6ee165bbb54fbad26ff3b77f"
)
CENSUS_RESULT_HASH = (
    "85ff0c04a1fe06b34b7f214f5fd7b9a1191a4ef0dd990a7e7d002f72efe9428d"
)
CENSUS_SCRIPT_PATH = Path(
    "training/"
    "audit_protocol_specification_intent_maturity_d4_grammar_census.py"
)
CENSUS_SCRIPT_SHA256 = (
    "677562847475c953d1d7462a3b6a5bb68b3faada43254a8aaadec34e0d822d2d"
)
CENSUS_TEST_PATH = Path(
    "tests/test_audit_protocol_specification_intent_maturity_d4_"
    "grammar_census.py"
)
CENSUS_TEST_SHA256 = (
    "1fa579a58c1c556303da8aaa2a348cbeb1a093a1017b4882699bf0455f309bf9"
)
DECISION_PATH = Path(
    "docs/post-psim-d4-grammar-census-and-d5-design-2026-07-26.md"
)
DECISION_SHA256 = (
    "0eda4fb027f0ea85844deb800682504e8c5a5a41fbebd16cbca7039b5adefb98"
)

KNOWN_INVALID_STATES = {
    "D4_DUPLICATE_IDENTICAL_HEADER": "INVALID_DUPLICATE_IDENTICAL",
    "D4_DUPLICATE_CONFLICTING_HEADER": "INVALID_DUPLICATE_CONFLICTING",
    "D4_MALFORMED_HEADER_LINE": "INVALID_MALFORMED_HEADER",
    "D4_SELF_DEPENDENCY": "INVALID_SELF_DEPENDENCY",
}
ADMINISTRATIVE_CLASSES = {
    "ERC_MIGRATION_REDIRECT_LOWER_PATH",
    "ERC_MIGRATION_REDIRECT_UPPER_PATH",
}
UNKNOWN_STATE = "INVALID_UNKNOWN"
ABSENT_STATE = "ABSENT"

OFFICIAL_REFERENCE_NOTES = (
    {
        "claim": (
            "Ethereum moved ERC documents out of the EIPs repository on "
            "2023-10-25 and replaced their prior contents with move stubs."
        ),
        "url": (
            "https://github.com/ethereum/EIPs/commit/"
            "0f44e2b94df4e504bb7b912f56ebd712db2ad396"
        ),
        "version": "official EIPs commit dated 2023-10-25",
    },
    {
        "claim": (
            "A same-day follow-up changed the move-stub target path casing "
            "from ercs to ERCS."
        ),
        "url": (
            "https://github.com/ethereum/EIPs/commit/"
            "47ce70257fae525a427780630bd8d1903cc96e75"
        ),
        "version": "official EIPs commit dated 2023-10-25",
    },
    {
        "claim": (
            "EIP-1 defines the front-matter field contract but does not "
            "define recovery semantics for malformed historical metadata."
        ),
        "url": "https://eips.ethereum.org/EIPS/eip-1",
        "version": "canonical page observed 2026-07-26 KST",
    },
    {
        "claim": "YAML 1.2.2 requires mapping keys to be unique.",
        "url": "https://yaml.org/spec/1.2.2/",
        "version": "YAML 1.2.2, 2021-10-01",
    },
)


@dataclass(frozen=True)
class BlobSemantics:
    """Path-bound source state for one proposal blob."""

    protocol: str
    proposal_number: int
    blob_oid: str
    blob_sha256: str
    normalized_lines: tuple[str, ...]
    line_sections: tuple[str, ...]
    section_presence: tuple[str, ...]
    metadata_state: str
    metadata_header: tuple[tuple[str, str], ...]
    dependency_edges: tuple[tuple[str, tuple[int, ...]], ...]
    dependency_availability: str
    administrative_class: str
    model_visible: bool
    classification_detail_hash: str


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


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D5 authority is absent or unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"PSIM-D5 authority is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D5 authority is noncanonical: {path}")
    return payload


def _load_d4_terminal_binding() -> dict[str, Any]:
    payload = _read_canonical_json(D4_TERMINAL_PATH)
    ledger = payload.get("access_ledger")
    if (
        sha256_file(D4_TERMINAL_PATH) != D4_TERMINAL_SHA256
        or payload.get("result_hash") != D4_TERMINAL_RESULT_HASH
        or payload.get("decision") != "reject"
        or payload.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or payload.get("outcomes_opened") is not False
        or not isinstance(ledger, Mapping)
        or any(ledger.get(name) != 0 for name in d4.FORBIDDEN_ACCESS_FIELDS)
    ):
        raise RuntimeError("PSIM-D4 terminal boundary changed")
    return {
        "commit": D4_TERMINAL_COMMIT,
        "decision": "reject",
        "first_failure_gate_id": 4,
        "path": D4_TERMINAL_PATH.as_posix(),
        "result_hash": D4_TERMINAL_RESULT_HASH,
        "sha256": D4_TERMINAL_SHA256,
    }


def _load_census_binding() -> dict[str, Any]:
    payload = _read_canonical_json(CENSUS_PATH)
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    if (
        sha256_file(CENSUS_PATH) != CENSUS_SHA256
        or payload.get("result_hash") != CENSUS_RESULT_HASH
        or payload.get("result_hash") != canonical_hash(core)
        or payload.get("terminal_binding", {}).get("result_hash")
        != D4_TERMINAL_RESULT_HASH
        or payload.get("access_boundary")
        != {
            "d4_forensic_root_read": True,
            "d4_run_invoked": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "network_commands": 0,
            "outcomes_accessed": False,
            "source_objects_mutated": False,
        }
        or payload.get("census", {}).get("class_counts")
        != census.EXPECTED_CLASS_COUNTS
        or payload.get("census", {}).get("unique_blob_contexts")
        != census.REQUESTED_BLOBS
    ):
        raise RuntimeError("PSIM-D4 grammar census boundary changed")
    if (
        sha256_file(CENSUS_SCRIPT_PATH) != CENSUS_SCRIPT_SHA256
        or sha256_file(CENSUS_TEST_PATH) != CENSUS_TEST_SHA256
        or sha256_file(DECISION_PATH) != DECISION_SHA256
    ):
        raise RuntimeError("PSIM-D5 census decision authority changed")
    return {
        "commit": CENSUS_COMMIT,
        "decision_document": {
            "path": DECISION_PATH.as_posix(),
            "sha256": DECISION_SHA256,
        },
        "path": CENSUS_PATH.as_posix(),
        "result_hash": CENSUS_RESULT_HASH,
        "script": {
            "path": CENSUS_SCRIPT_PATH.as_posix(),
            "sha256": CENSUS_SCRIPT_SHA256,
        },
        "sha256": CENSUS_SHA256,
        "test": {
            "path": CENSUS_TEST_PATH.as_posix(),
            "sha256": CENSUS_TEST_SHA256,
        },
    }


def _valid_blob_semantics(
    protocol: str,
    proposal_number: int,
    blob_oid: str,
    raw: bytes,
) -> BlobSemantics:
    features = d4.parse_blob_features(
        protocol,
        proposal_number,
        blob_oid,
        raw,
    )
    return BlobSemantics(
        protocol=protocol,
        proposal_number=proposal_number,
        blob_oid=blob_oid,
        blob_sha256=features.blob_sha256,
        normalized_lines=features.normalized_lines,
        line_sections=features.line_sections,
        section_presence=features.section_presence,
        metadata_state="VALID",
        metadata_header=tuple(sorted(features.header.items())),
        dependency_edges=tuple(sorted(features.dependency_edges.items())),
        dependency_availability="KNOWN",
        administrative_class="NONE",
        model_visible=True,
        classification_detail_hash=canonical_hash({}),
    )


def decode_blob_d5(
    protocol: str,
    proposal_number: int,
    blob_oid: str,
    raw: bytes,
) -> BlobSemantics:
    """Decode one blob without synthesizing or repairing metadata."""

    if protocol not in {"ethereum", "bitcoin"}:
        raise ValueError("PSIM-D5 protocol must be ethereum or bitcoin")
    if d4.core.git_object_sha1("blob", raw) != blob_oid:
        raise ValueError("PSIM-D5 blob object SHA-1 mismatch")
    if protocol == "bitcoin":
        return _valid_blob_semantics(
            protocol,
            proposal_number,
            blob_oid,
            raw,
        )

    classification, detail = census.classify_blob(
        proposal_number,
        blob_oid,
        raw,
    )
    if classification == "D4_VALID":
        return _valid_blob_semantics(
            protocol,
            proposal_number,
            blob_oid,
            raw,
        )

    lines = tuple(d4.core.prereg.normalize_blob_bytes(raw))
    sections = d4.core._line_sections(protocol, lines)
    detail_hash = canonical_hash(
        {
            "classification": classification,
            "detail": detail,
        }
    )
    if classification in ADMINISTRATIVE_CLASSES:
        return BlobSemantics(
            protocol=protocol,
            proposal_number=proposal_number,
            blob_oid=blob_oid,
            blob_sha256=sha256_bytes(raw),
            normalized_lines=lines,
            line_sections=sections,
            section_presence=tuple(sorted(set(sections))),
            metadata_state="ADMINISTRATIVE_REDIRECT",
            metadata_header=(),
            dependency_edges=(),
            dependency_availability="NOT_APPLICABLE_ADMINISTRATIVE",
            administrative_class=classification,
            model_visible=False,
            classification_detail_hash=detail_hash,
        )

    metadata_state = KNOWN_INVALID_STATES.get(
        classification,
        UNKNOWN_STATE,
    )
    return BlobSemantics(
        protocol=protocol,
        proposal_number=proposal_number,
        blob_oid=blob_oid,
        blob_sha256=sha256_bytes(raw),
        normalized_lines=lines,
        line_sections=sections,
        section_presence=tuple(sorted(set(sections))),
        metadata_state=metadata_state,
        metadata_header=(),
        dependency_edges=(),
        dependency_availability=(
            "UNKNOWN_INVALID_METADATA"
            if metadata_state != UNKNOWN_STATE
            else "UNKNOWN_UNCLASSIFIED"
        ),
        administrative_class="NONE",
        model_visible=metadata_state != UNKNOWN_STATE,
        classification_detail_hash=detail_hash,
    )


def _flatten_dependency_edges(
    blob: BlobSemantics,
) -> set[tuple[str, int]]:
    return {
        (field_name, proposal)
        for field_name, proposals in blob.dependency_edges
        for proposal in proposals
    }


def _known_dependency_delta(
    old: BlobSemantics | None,
    new: BlobSemantics | None,
) -> tuple[str, int]:
    if old is None and new is not None:
        return "NO_PRIOR", len(_flatten_dependency_edges(new))
    if old is not None and new is None:
        return "DELETED", len(_flatten_dependency_edges(old))
    if old is None or new is None:
        raise ValueError("PSIM-D5 dependency delta received empty pair")
    old_edges = _flatten_dependency_edges(old)
    new_edges = _flatten_dependency_edges(new)
    added = new_edges - old_edges
    removed = old_edges - new_edges
    if not added and not removed:
        return "STABLE", 0
    if added and not removed:
        return "ADDED", len(added)
    if removed and not added:
        return "REMOVED", len(removed)
    return "MIXED", len(added) + len(removed)


def _normalized_changed_rows(
    old: BlobSemantics | None,
    new: BlobSemantics | None,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    old_lines = () if old is None else old.normalized_lines
    new_lines = () if new is None else new.normalized_lines
    old_sections = () if old is None else old.line_sections
    new_sections = () if new is None else new.line_sections
    matcher = difflib.SequenceMatcher(
        a=old_lines,
        b=new_lines,
        autojunk=False,
    )
    rows: list[dict[str, str]] = []
    changed_sections: set[str] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_sections.update(old_sections[i1:i2])
        changed_sections.update(new_sections[j1:j2])
        rows.extend(
            {
                "direction": "REMOVE",
                "line": old_lines[index],
                "section": old_sections[index],
            }
            for index in range(i1, i2)
        )
        rows.extend(
            {
                "direction": "ADD",
                "line": new_lines[index],
                "section": new_sections[index],
            }
            for index in range(j1, j2)
        )
    return rows, tuple(sorted(changed_sections))


def _model_text_delta(rows: Sequence[Mapping[str, str]]) -> str:
    text = "\n".join(
        f"{row['section']}|{row['direction']}|{row['line']}"
        for row in rows
    )
    if (
        len(text.encode("utf-8"))
        > d4.core.prereg.MAX_MODEL_TEXT_BYTES_PER_EVENT
    ):
        raise ValueError(
            "PSIM-D5 model-visible normalized text delta exceeds frozen bound"
        )
    return text


def _model_visible_delta_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    allowed = frozenset(d4.core.MODEL_SECTION_ORDER)
    return [row for row in rows if row["section"] in allowed]


def _event_path_identity(
    protocol: str,
    proposal_number: int,
    *,
    old_path: str | None,
    new_path: str | None,
    old: BlobSemantics | None,
    new: BlobSemantics | None,
) -> dict[str, Any]:
    if protocol not in {"ethereum", "bitcoin"}:
        raise ValueError("PSIM-D5 event protocol is invalid")
    if (old is None) != (old_path is None) or (new is None) != (
        new_path is None
    ):
        raise ValueError("PSIM-D5 event path/blob side shape differs")
    for path, blob in ((old_path, old), (new_path, new)):
        if path is None or blob is None:
            continue
        identity = d4.core._path_identity(protocol, path)
        if (
            identity is None
            or identity[0] != proposal_number
            or blob.protocol != protocol
            or blob.proposal_number != proposal_number
        ):
            raise ValueError("PSIM-D5 event path identity changed")
    core = {
        "new_path": new_path,
        "old_path": old_path,
        "proposal_number": proposal_number,
        "protocol": protocol,
    }
    return {**core, "identity_hash": canonical_hash(core)}


def build_event_semantics_d5(
    protocol: str,
    proposal_number: int,
    *,
    old_path: str | None,
    new_path: str | None,
    old: BlobSemantics | None,
    new: BlobSemantics | None,
) -> dict[str, Any]:
    """Build one path-bound event, quarantining only exact migration stubs."""

    if old is None and new is None:
        raise ValueError("PSIM-D5 event has no blob sides")
    path_identity = _event_path_identity(
        protocol,
        proposal_number,
        old_path=old_path,
        new_path=new_path,
        old=old,
        new=new,
    )
    sides = [blob for blob in (old, new) if blob is not None]
    if any(blob.metadata_state == UNKNOWN_STATE for blob in sides):
        raise ValueError("PSIM-D5 unclassified metadata fails closed")

    rows, changed_sections = _normalized_changed_rows(old, new)
    audit_diff_hash = canonical_hash(rows)
    old_admin = old is not None and old.administrative_class != "NONE"
    new_admin = new is not None and new.administrative_class != "NONE"
    invalid_metadata_states = sorted(
        {
            blob.metadata_state
            for blob in sides
            if blob.metadata_state
            not in {"VALID", "ADMINISTRATIVE_REDIRECT"}
        }
    )
    invalid_metadata = bool(invalid_metadata_states)
    if old_admin and new is not None and not new_admin:
        raise ValueError(
            "PSIM-D5 reverse administrative migration is ambiguous"
        )
    if old_admin or new_admin:
        return {
            "audit_diff_hash": audit_diff_hash,
            "audit_line_change_count": len(rows),
            "changed_sections": [],
            "dependency_delta_state": "ADMINISTRATIVE_QUARANTINE",
            "dependency_edge_delta_count": None,
            "normalized_text_delta": "",
            "invalid_metadata_present": invalid_metadata,
            "invalid_metadata_states": invalid_metadata_states,
            "model_line_change_count": 0,
            "model_visibility": "ADMINISTRATIVE_QUARANTINE",
            "new_metadata_state": (
                ABSENT_STATE if new is None else new.metadata_state
            ),
            "old_metadata_state": (
                ABSENT_STATE if old is None else old.metadata_state
            ),
            "path_identity": path_identity,
            "path_identity_source": "PROPOSAL_GROUP_EXACT_SIDE_PATHS",
            "quarantine_reason": (
                "EXACT_2023_ETHEREUM_ERC_REPOSITORY_MIGRATION_STUB"
            ),
        }

    if invalid_metadata:
        dependency_state = "UNKNOWN_INVALID_METADATA"
        dependency_count: int | None = None
    else:
        dependency_state, dependency_count = _known_dependency_delta(
            old,
            new,
        )
    model_rows = _model_visible_delta_rows(rows)
    text = _model_text_delta(model_rows)
    return {
        "audit_diff_hash": audit_diff_hash,
        "audit_line_change_count": len(rows),
        "changed_sections": list(changed_sections),
        "dependency_delta_state": dependency_state,
        "dependency_edge_delta_count": dependency_count,
        "normalized_text_delta": text,
        "invalid_metadata_present": invalid_metadata,
        "invalid_metadata_states": invalid_metadata_states,
        "model_line_change_count": len(model_rows),
        "model_visibility": "MODEL_VISIBLE",
        "new_metadata_state": (
            ABSENT_STATE if new is None else new.metadata_state
        ),
        "old_metadata_state": (
            ABSENT_STATE if old is None else old.metadata_state
        ),
        "path_identity": path_identity,
        "path_identity_source": "PROPOSAL_GROUP_EXACT_SIDE_PATHS",
        "quarantine_reason": None,
    }


def _synthetic_blob(
    raw: bytes,
    *,
    protocol: str = "ethereum",
    proposal_number: int = 20,
) -> BlobSemantics:
    oid = d4.core.git_object_sha1("blob", raw)
    return decode_blob_d5(protocol, proposal_number, oid, raw)


def _synthetic_event(
    old: BlobSemantics | None,
    new: BlobSemantics | None,
    *,
    proposal_number: int = 20,
) -> dict[str, Any]:
    return build_event_semantics_d5(
        "ethereum",
        proposal_number,
        old_path=(
            None if old is None else f"EIPS/eip-{proposal_number}.md"
        ),
        new_path=(
            None if new is None else f"EIPS/eip-{proposal_number}.md"
        ),
        old=old,
        new=new,
    )


def _raises_event_error(
    proposal_number: int,
    old: BlobSemantics | None,
    new: BlobSemantics | None,
) -> bool:
    try:
        _synthetic_event(
            old,
            new,
            proposal_number=proposal_number,
        )
    except ValueError:
        return True
    return False


def _run_synthetic_battery() -> dict[str, Any]:
    valid_old = _synthetic_blob(
        b"---\n"
        b"eip: 20\n"
        b"title: Synthetic old\n"
        b"status: Draft\n"
        b"requires: 1\n"
        b"---\n"
        b"# Abstract\n"
        b"Old synthetic intent.\n"
    )
    valid_new = _synthetic_blob(
        b"---\n"
        b"eip: 20\n"
        b"title: Synthetic new\n"
        b"status: Review\n"
        b"requires: 1, 2\n"
        b"---\n"
        b"# Abstract\n"
        b"New synthetic intent.\n"
    )
    valid_event = _synthetic_event(valid_old, valid_new)
    required_fragments = {
        "ABSTRACT|REMOVE|Old synthetic intent.",
        "ABSTRACT|ADD|New synthetic intent.",
    }
    if not required_fragments.issubset(
        set(valid_event["normalized_text_delta"].splitlines())
    ):
        raise RuntimeError(
            "PSIM-D5 normalized text delta lost synthetic lines"
        )
    forbidden_fragments = {
        "OTHER|REMOVE|status: Draft",
        "OTHER|ADD|status: Review",
        "OTHER|REMOVE|requires: 1",
        "OTHER|ADD|requires: 1, 2",
    }
    if forbidden_fragments.intersection(
        valid_event["normalized_text_delta"].splitlines()
    ):
        raise RuntimeError(
            "PSIM-D5 model text exposed proposal metadata lines"
        )

    invalid_fixtures = {
        "INVALID_DUPLICATE_IDENTICAL": (
            b"---\neip: 20\nstatus: Draft\nstatus: Draft\n---\n"
            b"# Abstract\nSynthetic duplicate.\n"
        ),
        "INVALID_DUPLICATE_CONFLICTING": (
            b"---\neip: 20\nstatus: Draft\nstatus: Final\n---\n"
            b"# Abstract\nSynthetic conflict.\n"
        ),
        "INVALID_MALFORMED_HEADER": (
            b"---\neip: 20\nrequires (*optional): 1\n---\n"
            b"# Abstract\nSynthetic malformed field.\n"
        ),
        "INVALID_SELF_DEPENDENCY": (
            b"---\neip: 20\nrequires: 1, 20\n---\n"
            b"# Abstract\nSynthetic self dependency.\n"
        ),
    }
    observed_invalid: dict[str, bool] = {}
    invalid_blobs: dict[str, BlobSemantics] = {}
    for expected, raw in invalid_fixtures.items():
        blob = _synthetic_blob(raw)
        invalid_blobs[expected] = blob
        event = _synthetic_event(valid_old, blob)
        observed_invalid[expected] = (
            blob.metadata_state == expected
            and blob.metadata_header == ()
            and blob.dependency_edges == ()
            and blob.dependency_availability
            == "UNKNOWN_INVALID_METADATA"
            and event["model_visibility"] == "MODEL_VISIBLE"
            and event["dependency_delta_state"]
            == "UNKNOWN_INVALID_METADATA"
            and event["dependency_edge_delta_count"] is None
            and bool(event["normalized_text_delta"])
        )
    if not all(observed_invalid.values()):
        raise RuntimeError(
            "PSIM-D5 known invalid metadata state battery failed"
        )

    lower_redirect = _synthetic_blob(
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
    )
    upper_redirect = _synthetic_blob(
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ERCS/erc-20.md\n"
    )
    migration = _synthetic_event(valid_old, lower_redirect)
    casing_fix = _synthetic_event(
        lower_redirect,
        upper_redirect,
    )
    redirect_deleted = _synthetic_event(
        upper_redirect,
        None,
    )
    invalid_to_redirect = _synthetic_event(
        invalid_blobs["INVALID_DUPLICATE_CONFLICTING"],
        lower_redirect,
    )
    quarantines = (
        migration,
        casing_fix,
        redirect_deleted,
        invalid_to_redirect,
    )
    if not all(
        event["model_visibility"] == "ADMINISTRATIVE_QUARANTINE"
        and event["normalized_text_delta"] == ""
        and event["model_line_change_count"] == 0
        and event["audit_line_change_count"] > 0
        for event in quarantines
    ):
        raise RuntimeError("PSIM-D5 administrative quarantine battery failed")
    if (
        invalid_to_redirect["invalid_metadata_present"] is not True
        or invalid_to_redirect["invalid_metadata_states"]
        != ["INVALID_DUPLICATE_CONFLICTING"]
    ):
        raise RuntimeError(
            "PSIM-D5 quarantine lost invalid metadata audit state"
        )

    unknown_blobs = (
        _synthetic_blob(
            b"This file was moved to "
            b"https://github.com/ethereum/ercs/blob/master/ercs/erc-21.md\n"
        ),
        _synthetic_blob(
            b"This file was moved to "
            b"https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
            b"synthetic extra text\n"
        ),
        _synthetic_blob(
            b"---\neip: 21\ntitle: Synthetic number mismatch\n---\n"
        ),
    )
    if not all(
        blob.metadata_state == UNKNOWN_STATE
        and blob.model_visible is False
        and _raises_event_error(20, valid_old, blob)
        for blob in unknown_blobs
    ):
        raise RuntimeError("PSIM-D5 unknown grammar did not fail closed")
    if not _raises_event_error(20, lower_redirect, valid_new):
        raise RuntimeError("PSIM-D5 reverse migration did not fail closed")

    repeated_or_moved_pairs = (
        (
            _synthetic_blob(
                b"---\neip: 20\n---\n# Abstract\nrepeat\nanchor\nrepeat\n"
            ),
            _synthetic_blob(
                b"---\neip: 20\n---\n# Abstract\nrepeat\nrepeat\nanchor\n"
            ),
        ),
        (
            _synthetic_blob(
                b"---\neip: 20\n---\n# Abstract\nfirst\nsecond\nthird\n"
            ),
            _synthetic_blob(
                b"---\neip: 20\n---\n# Abstract\nthird\nfirst\nsecond\n"
            ),
        ),
    )
    for old, new in repeated_or_moved_pairs:
        first = _normalized_changed_rows(old, new)
        second = _normalized_changed_rows(old, new)
        if first != second or not first[0]:
            raise RuntimeError(
                "PSIM-D5 normalized algorithmic delta is not deterministic"
            )

    bip_raw = (
        b"<pre>\n"
        b"  BIP: 0020\n"
        b"  Title: Synthetic BIP control\n"
        b"  Requires: 1, 2\n"
        b"</pre>\n"
    )
    bip_oid = d4.core.git_object_sha1("blob", bip_raw)
    bip = decode_blob_d5("bitcoin", 20, bip_oid, bip_raw)
    bip_control = d4.parse_blob_features(
        "bitcoin",
        20,
        bip_oid,
        bip_raw,
    )
    if (
        bip.metadata_header != tuple(sorted(bip_control.header.items()))
        or bip.dependency_edges
        != tuple(sorted(bip_control.dependency_edges.items()))
    ):
        raise RuntimeError("PSIM-D5 changed the D4 BIP parse result")

    return {
        "administrative_quarantine_transitions": len(quarantines),
        "administrative_quarantine_preserves_invalid_metadata_audit": True,
        "algorithmic_delta_repeated_or_moved_line_cases": len(
            repeated_or_moved_pairs
        ),
        "bip_d4_parse_outputs_unchanged": 1,
        "normalized_text_delta_excludes_metadata_lines": True,
        "normalized_text_delta_includes_approved_body_lines": True,
        "dependency_unknown_without_repair_for_known_invalid": len(
            observed_invalid
        ),
        "exact_redirect_path_case_variants": 2,
        "known_invalid_metadata_states_model_visible": observed_invalid,
        "path_identity_binds_protocol_number_and_exact_side_paths": (
            valid_event["path_identity_source"]
            == "PROPOSAL_GROUP_EXACT_SIDE_PATHS"
            and valid_event["path_identity"]["protocol"] == "ethereum"
            and valid_event["path_identity"]["proposal_number"] == 20
            and valid_event["path_identity"]["old_path"]
            == "EIPS/eip-20.md"
            and valid_event["path_identity"]["new_path"]
            == "EIPS/eip-20.md"
            and valid_event["path_identity"]["identity_hash"]
            == canonical_hash(
                {
                    "new_path": "EIPS/eip-20.md",
                    "old_path": "EIPS/eip-20.md",
                    "proposal_number": 20,
                    "protocol": "ethereum",
                }
            )
        ),
        "raw_audit_diff_preserved_when_quarantined": True,
        "reverse_migration_fails_closed": True,
        "unknown_grammar_cases_fail_closed": len(unknown_blobs),
    }


def build_probe() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "access_boundary": {
            "d4_census_artifact_read": True,
            "d4_forensic_root_accessed": False,
            "d4_terminal_artifact_read": True,
            "external_network_accessed_by_probe": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "official_reference_research_preexisted_probe": True,
            "official_historical_proposal_source_accessed": False,
            "outcomes_accessed": False,
        },
        "candidate": {
            "id": "PSIM-D5",
            "name": (
                "Protocol Specification Intent-Maturity relation RLLM, "
                "exact path identity plus normalized text-delta semantics"
            ),
            "source_representation_successor": True,
        },
        "d4_census_binding": _load_census_binding(),
        "d4_terminal_binding": _load_d4_terminal_binding(),
        "official_reference_notes": list(OFFICIAL_REFERENCE_NOTES),
        "official_reference_provenance": {
            "accessed_by_probe": False,
            "model_visible": False,
            "origin": (
                "PRIOR_OFFICIAL_SOURCE_RESEARCH_RECORDED_IN_D4_CENSUS_"
                "DECISION"
            ),
            "selection_evidence_only": True,
        },
        "selection_scope": (
            "AUTHORIZE_D5_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
        ),
        "semantics_contract": {
            "administrative_quarantine": (
                "EXACT_ONE_LINE_ETHEREUM_ERC_MOVE_STUB_WITH_TARGET_NUMBER_"
                "EQUAL_TO_PROPOSAL_GROUP_PATH_NUMBER"
            ),
            "administrative_quarantine_invalid_metadata_audit": (
                "PRESERVE_EXPLICIT_INVALID_STATES_WHILE_MODEL_TEXT_IS_EMPTY"
            ),
            "administrative_text_model_visible": False,
            "bip_parser": "UNCHANGED_D4_STRICT_PARSER",
            "normalized_text_delta_is_causal_semantics_claim": False,
            "normalized_text_delta_order": (
                "SEQUENCE_MATCHER_OPCODE_ORDER_REMOVE_THEN_ADD_SOURCE_ORDER"
            ),
            "dependency_when_metadata_invalid": (
                "UNKNOWN_WITH_NULL_COUNT_NO_REPAIR"
            ),
            "invalid_metadata_states": sorted(
                {
                    *KNOWN_INVALID_STATES.values(),
                    UNKNOWN_STATE,
                }
            ),
            "known_invalid_metadata_text_model_visible": (
                "TRUE_FOR_NONADMINISTRATIVE_EVENTS"
            ),
            "metadata_resolution": (
                "NONE_NO_FIRST_LAST_MERGE_DEDUP_RENAME_OR_SELF_EDGE_DROP"
            ),
            "model_metadata_lines_visible": False,
            "model_text_field": "normalized_text_delta",
            "model_text_sections": list(d4.core.MODEL_SECTION_ORDER),
            "path_identity": (
                "PROTOCOL_PLUS_EXACT_OLD_NEW_GROUP_PATHS_PLUS_NUMBER_"
                "CANONICAL_HASH_BOUND"
            ),
            "raw_normalization": "UNCHANGED_D1_NORMALIZE_BLOB_BYTES",
            "reverse_administrative_transition": "FAIL_CLOSED",
            "unknown_grammar": "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES",
        },
        "semantics_version": SEMANTICS_VERSION,
        "synthetic_battery": _run_synthetic_battery(),
        "synthetic_only": True,
        "protocol_version": PROTOCOL_VERSION,
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def write_probe(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_probe()
    target = repository_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("existing PSIM-D5 event semantics probe differs")
    target.write_bytes(raw)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
