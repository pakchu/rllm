"""Select PSIM-D6 source mechanisms with synthetic controls only.

The probe binds the terminal PSIM-D5 event census, but it never opens the
forensic Git root, historical proposal text, market data, a model, rewards,
trades, PnL, CAGR, strict-MDD, or outcomes. It selects two source-only
mechanisms for later preregistration:

* exact receipt-bound ERC migration restoration quarantine; and
* lossless deterministic UTF-8 model-text chunks.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import (
    audit_protocol_specification_intent_maturity_d5_event_semantics_census
    as d5_census,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d6_mechanism_probe_"
    "2026-07-26.json"
)
PROTOCOL_VERSION = "psim_d6_source_mechanism_probe_v1"
MECHANISM_VERSION = (
    "PSIM_EXACT_MIGRATION_RECEIPT_PLUS_UTF8_CHUNK_TRANSPORT_V1"
)

D5_CENSUS_COMMIT = "4c4e3eb49962597eac7f63a0a3c1bf58f1fe73e4"
D5_CENSUS_PATH = d5_census.DEFAULT_OUTPUT
D5_CENSUS_SHA256 = (
    "df2bcbef28b22d6daeb258d5c0f36b918d833b9c5fb0e5c9229a44edce4c2d59"
)
D5_CENSUS_RESULT_HASH = (
    "0ca0a11f6693543dafbcf29052f2e963bf721c6e12f71f6fc9fbb1856e2dfe4a"
)
D5_CENSUS_SCRIPT_PATH = Path(
    "training/"
    "audit_protocol_specification_intent_maturity_d5_event_semantics_"
    "census.py"
)
D5_CENSUS_SCRIPT_SHA256 = (
    "eab91b9fc65392ec5e2b2e187324498f937555cdd0099880517e7efc98c28b0c"
)
D5_CENSUS_TEST_PATH = Path(
    "tests/"
    "test_audit_protocol_specification_intent_maturity_d5_event_semantics_"
    "census.py"
)
D5_CENSUS_TEST_SHA256 = (
    "c858f5802a906fdd65c35f7f443062e576eb86fd42d7274fe4b40121ed3cd7a4"
)
D5_CENSUS_DOCUMENT_PATH = Path(
    "docs/"
    "post-psim-d5-event-semantics-census-and-d6-requirements-2026-07-26.md"
)
D5_CENSUS_DOCUMENT_SHA256 = (
    "fcfa6f8e734856dc9e6316c1243a5f9f3a87df52f7bd3be90422350f6334f4b2"
)
D5_EPISODE_ROSTER_HASH = (
    "7065c33783f1ea54af1522da7e442ec05507c38355bb98ed90daf3f87e89b0bd"
)
D5_EPISODE_RECEIPT_MANIFEST_HASH = (
    "abf21a4691e4407158efc61a267cc6eaec8522751c25fa531aed6f782accdc07"
)
D5_MIGRATION_PROPOSAL_ROSTER_HASH = (
    "c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd"
)
D5_TEXT_BOUND_EVENT_ROSTER_HASH = (
    "0f299221248e66ca1eddc9cdd839cab504755537e47464c697c481544d169fd4"
)

MAX_MODEL_TEXT_BYTES_PER_CHUNK = 8_192
MAX_MODEL_TEXT_CHUNKS_PER_EVENT = 8
MAX_MODEL_TEXT_BYTES_PER_EVENT = (
    MAX_MODEL_TEXT_BYTES_PER_CHUNK
    * MAX_MODEL_TEXT_CHUNKS_PER_EVENT
)

MIGRATION_COMMIT_SEQUENCE = (
    "0f44e2b94df4e504bb7b912f56ebd712db2ad396",
    "47ce70257fae525a427780630bd8d1903cc96e75",
    "25cdf1d059778236e28bf22d752ca48a35af91f6",
)
MIGRATION_DAY_SEQUENCE = (
    "2023-10-25",
    "2023-10-25",
    "2023-10-26",
)
MIGRATION_CLASS_SEQUENCE = (
    ("D4_VALID", "ERC_MIGRATION_REDIRECT_LOWER_PATH"),
    (
        "ERC_MIGRATION_REDIRECT_LOWER_PATH",
        "ERC_MIGRATION_REDIRECT_UPPER_PATH",
    ),
    ("ERC_MIGRATION_REDIRECT_UPPER_PATH", "D4_VALID"),
)
D5_OUTCOME_SEQUENCE = (
    "PASS_ADMINISTRATIVE_QUARANTINE",
    "PASS_ADMINISTRATIVE_QUARANTINE",
    "ERROR_REVERSE_ADMINISTRATIVE_MIGRATION",
)

HEX40 = re.compile(r"[0-9a-f]{40}", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
EPISODE_FIELDS = frozenset(
    {
        "lower_redirect",
        "path",
        "proposal",
        "steps",
        "upper_redirect",
    }
)
EPISODE_STEP_FIELDS = frozenset(
    {
        "commit_oid",
        "effective_day",
        "event_id",
        "event_type",
        "new_blob_class",
        "new_blob_oid",
        "new_blob_sha256",
        "new_path",
        "old_blob_class",
        "old_blob_oid",
        "old_blob_sha256",
        "old_path",
        "outcome_id",
    }
)
REDIRECT_FIELDS = frozenset(
    {
        "classification",
        "classification_detail_hash",
        "path_case",
        "target_matches_path_proposal",
        "target_proposal",
    }
)
MODEL_CHUNK_FIELDS = frozenset(
    {
        "chunk_count",
        "chunk_index",
        "normalized_text_delta_chunk",
    }
)
MODEL_DELTA_ROW_FIELDS = frozenset({"direction", "line", "section"})


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return d5_census.d5.sha256_bytes(raw)


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


def _safe_output_path(path: str | Path) -> Path:
    requested = Path(path)
    results_root = REPO_ROOT.resolve() / "results"
    target = results_root / requested.name
    unsafe_existing_target = (
        target.exists() and not target.is_file()
    )
    if (
        requested.is_absolute()
        or requested.parent != Path("results")
        or results_root.is_symlink()
        or not results_root.is_dir()
        or requested.suffix != ".json"
        or target.is_symlink()
        or unsafe_existing_target
    ):
        raise RuntimeError(
            "PSIM-D6 probe output must be a safe repo-local result"
        )
    return target


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D6 authority is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PSIM-D6 authority is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"PSIM-D6 authority is noncanonical: {path}")
    return payload


def _load_d5_census_binding() -> tuple[
    dict[str, Any],
    dict[int, str],
]:
    payload = _read_canonical_json(D5_CENSUS_PATH)
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    census = payload.get("census")
    episode = (
        census.get("administrative_episode_census")
        if isinstance(census, Mapping)
        else None
    )
    failure_profiles = (
        census.get("failure_profiles")
        if isinstance(census, Mapping)
        else None
    )
    text_bound_profile = (
        failure_profiles.get(d5_census.MODEL_TEXT_BOUND_ERROR)
        if isinstance(failure_profiles, Mapping)
        else None
    )
    receipt_rows = (
        episode.get("per_proposal_receipt_hashes")
        if isinstance(episode, Mapping)
        else None
    )
    if (
        sha256_file(D5_CENSUS_PATH) != D5_CENSUS_SHA256
        or payload.get("result_hash") != D5_CENSUS_RESULT_HASH
        or payload.get("result_hash") != canonical_hash(core)
        or payload.get("protocol_version")
        != d5_census.PROTOCOL_VERSION
        or payload.get("policy_id")
        != "PSIM-D5-POST-TERMINAL-EVENT-SEMANTICS-CENSUS"
        or payload.get("access_boundary")
        != {
            "d5_forensic_root_read": True,
            "d5_run_invoked": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "network_commands": 0,
            "outcomes_accessed": False,
            "raw_text_published": False,
            "source_objects_mutated": False,
        }
        or payload.get("candidate_selection", {}).get("authorized")
        is not False
        or not isinstance(census, Mapping)
        or census.get("decoded_blob_contexts") != 5_206
        or census.get("groups_evaluated") != 4_985
        or census.get("event_outcome_counts")
        != d5_census.EXPECTED_EVENT_OUTCOME_COUNTS
        or not isinstance(text_bound_profile, Mapping)
        or text_bound_profile.get("event_roster_hash")
        != D5_TEXT_BOUND_EVENT_ROSTER_HASH
        or not isinstance(episode, Mapping)
        or episode.get("episode_count") != 365
        or episode.get("covered_administrative_quarantine_events") != 730
        or episode.get("covered_reverse_error_events") != 365
        or episode.get("episode_roster_hash")
        != D5_EPISODE_ROSTER_HASH
        or episode.get("proposal_roster_hash")
        != D5_MIGRATION_PROPOSAL_ROSTER_HASH
        or episode.get("all_redirect_targets_match_path_proposal")
        is not True
        or not isinstance(receipt_rows, list)
        or len(receipt_rows) != 365
        or canonical_hash(receipt_rows)
        != D5_EPISODE_RECEIPT_MANIFEST_HASH
        or payload.get("forensic_source", {}).get(
            "object_store_unchanged"
        )
        is not True
    ):
        raise RuntimeError("PSIM-D5 event census authority changed")
    if (
        sha256_file(D5_CENSUS_SCRIPT_PATH)
        != D5_CENSUS_SCRIPT_SHA256
        or sha256_file(D5_CENSUS_TEST_PATH) != D5_CENSUS_TEST_SHA256
        or sha256_file(D5_CENSUS_DOCUMENT_PATH)
        != D5_CENSUS_DOCUMENT_SHA256
    ):
        raise RuntimeError("PSIM-D5 event census producer changed")

    receipt_map: dict[int, str] = {}
    for row in receipt_rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"proposal", "receipt_hash"}
            or type(row["proposal"]) is not int
            or row["proposal"] <= 0
            or HEX64.fullmatch(str(row["receipt_hash"])) is None
            or row["proposal"] in receipt_map
        ):
            raise RuntimeError(
                "PSIM-D5 migration receipt authority is malformed"
            )
        receipt_map[row["proposal"]] = str(row["receipt_hash"])
    if canonical_hash(sorted(receipt_map)) != (
        D5_MIGRATION_PROPOSAL_ROSTER_HASH
    ):
        raise RuntimeError(
            "PSIM-D5 migration proposal roster authority changed"
        )

    return (
        {
            "commit": D5_CENSUS_COMMIT,
            "document": {
                "path": D5_CENSUS_DOCUMENT_PATH.as_posix(),
                "sha256": D5_CENSUS_DOCUMENT_SHA256,
            },
            "episode_receipt_count": len(receipt_map),
            "episode_receipt_manifest_hash": (
                D5_EPISODE_RECEIPT_MANIFEST_HASH
            ),
            "episode_roster_hash": D5_EPISODE_ROSTER_HASH,
            "path": D5_CENSUS_PATH.as_posix(),
            "proposal_roster_hash": D5_MIGRATION_PROPOSAL_ROSTER_HASH,
            "result_hash": D5_CENSUS_RESULT_HASH,
            "script": {
                "path": D5_CENSUS_SCRIPT_PATH.as_posix(),
                "sha256": D5_CENSUS_SCRIPT_SHA256,
            },
            "sha256": D5_CENSUS_SHA256,
            "test": {
                "path": D5_CENSUS_TEST_PATH.as_posix(),
                "sha256": D5_CENSUS_TEST_SHA256,
            },
            "text_bound_event_roster_hash": (
                D5_TEXT_BOUND_EVENT_ROSTER_HASH
            ),
        },
        receipt_map,
    )


def split_utf8_model_text_d6(full_text: str) -> tuple[str, ...]:
    """Split text losslessly at deterministic valid UTF-8 boundaries."""

    if not isinstance(full_text, str):
        raise TypeError("PSIM-D6 model text must be str")
    try:
        raw = full_text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("PSIM-D6 model text is not strict UTF-8") from error
    if not raw:
        return ()

    chunks: list[str] = []
    offset = 0
    while offset < len(raw):
        end = min(offset + MAX_MODEL_TEXT_BYTES_PER_CHUNK, len(raw))
        if end < len(raw):
            while end > offset and raw[end] & 0b1100_0000 == 0b1000_0000:
                end -= 1
        if end <= offset:
            raise ValueError("PSIM-D6 could not find a UTF-8 chunk boundary")
        try:
            chunk = raw[offset:end].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(
                "PSIM-D6 chunk boundary is not strict UTF-8"
            ) from error
        chunks.append(chunk)
        if len(chunks) > MAX_MODEL_TEXT_CHUNKS_PER_EVENT:
            raise ValueError(
                "PSIM-D6 model text requires more than eight chunks"
            )
        offset = end
    if b"".join(chunk.encode("utf-8") for chunk in chunks) != raw:
        raise RuntimeError("PSIM-D6 chunk split is not lossless")
    return tuple(chunks)


def serialize_model_delta_rows_d6(
    rows: Sequence[Mapping[str, str]],
) -> str:
    serialized: list[str] = []
    allowed_sections = frozenset(
        d5_census.d5.core.MODEL_SECTION_ORDER
    )
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != MODEL_DELTA_ROW_FIELDS:
            raise ValueError("PSIM-D6 model delta row fields changed")
        section = row["section"]
        direction = row["direction"]
        line = row["line"]
        if (
            not isinstance(section, str)
            or section not in allowed_sections
            or not isinstance(direction, str)
            or direction not in {"ADD", "REMOVE"}
            or not isinstance(line, str)
            or "\n" in line
            or "\r" in line
        ):
            raise ValueError("PSIM-D6 model delta row is malformed")
        serialized.append(f"{section}|{direction}|{line}")
    return "\n".join(serialized)


def build_model_chunk_payloads_d6(
    full_text: str,
) -> tuple[dict[str, Any], ...]:
    chunks = split_utf8_model_text_d6(full_text)
    count = len(chunks)
    return tuple(
        {
            "chunk_count": count,
            "chunk_index": index,
            "normalized_text_delta_chunk": chunk,
        }
        for index, chunk in enumerate(chunks)
    )


def validate_model_chunk_payloads_d6(
    full_text: str,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(full_text, str):
        raise TypeError("PSIM-D6 model text must be str")
    try:
        expected_raw = full_text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("PSIM-D6 model text is not strict UTF-8") from error
    rows = list(payloads)
    if len(rows) > MAX_MODEL_TEXT_CHUNKS_PER_EVENT:
        raise ValueError("PSIM-D6 chunk payload fanout exceeds eight")
    if not expected_raw and rows:
        raise ValueError("PSIM-D6 empty model text must have zero chunks")
    if expected_raw and not rows:
        raise ValueError("PSIM-D6 nonempty model text lost all chunks")

    reconstructed: list[bytes] = []
    receipt_rows: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != MODEL_CHUNK_FIELDS:
            raise ValueError("PSIM-D6 chunk payload fields changed")
        if (
            type(row["chunk_index"]) is not int
            or row["chunk_index"] != expected_index
            or type(row["chunk_count"]) is not int
            or row["chunk_count"] != len(rows)
            or not isinstance(row["normalized_text_delta_chunk"], str)
            or not row["normalized_text_delta_chunk"]
        ):
            raise ValueError("PSIM-D6 chunk index or count changed")
        try:
            raw = row["normalized_text_delta_chunk"].encode(
                "utf-8",
                errors="strict",
            )
        except UnicodeEncodeError as error:
            raise ValueError("PSIM-D6 chunk is not strict UTF-8") from error
        if not 1 <= len(raw) <= MAX_MODEL_TEXT_BYTES_PER_CHUNK:
            raise ValueError("PSIM-D6 chunk byte bound changed")
        reconstructed.append(raw)
        receipt_rows.append(
            {
                "chunk_index": expected_index,
                "sha256": sha256_bytes(raw),
                "utf8_bytes": len(raw),
            }
        )

    reconstructed_raw = b"".join(reconstructed)
    if reconstructed_raw != expected_raw:
        raise ValueError("PSIM-D6 chunk reconstruction differs")
    expected_payloads = list(build_model_chunk_payloads_d6(full_text))
    if rows != expected_payloads:
        raise ValueError("PSIM-D6 chunk partition is not canonical greedy")
    core = {
        "chunk_count": len(rows),
        "chunks": receipt_rows,
        "full_text_line_count": (
            0 if not expected_raw else expected_raw.count(b"\n") + 1
        ),
        "full_text_sha256": sha256_bytes(expected_raw),
        "full_text_utf8_bytes": len(expected_raw),
        "max_bytes_per_chunk": MAX_MODEL_TEXT_BYTES_PER_CHUNK,
        "max_chunks_per_event": MAX_MODEL_TEXT_CHUNKS_PER_EVENT,
        "protocol_version": "psim_d6_model_text_chunk_receipt_v1",
        "reconstructed_sha256": sha256_bytes(reconstructed_raw),
        "reconstruction_matches": True,
    }
    return {
        **core,
        "receipt_hash": canonical_hash(core),
    }


def build_model_text_transport_d6(
    full_text: str,
) -> dict[str, Any]:
    payloads = build_model_chunk_payloads_d6(full_text)
    return {
        "audit_receipt": validate_model_chunk_payloads_d6(
            full_text,
            payloads,
        ),
        "model_chunk_payloads": list(payloads),
    }


def _validate_hex(value: Any, pattern: re.Pattern[str], name: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"PSIM-D6 migration {name} is malformed")


def _validate_redirect(
    value: Any,
    *,
    proposal: int,
    classification: str,
    path_case: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != REDIRECT_FIELDS:
        raise ValueError("PSIM-D6 migration redirect fields changed")
    if (
        value["classification"] != classification
        or value["path_case"] != path_case
        or value["target_matches_path_proposal"] is not True
        or value["target_proposal"] != proposal
    ):
        raise ValueError("PSIM-D6 migration redirect identity changed")
    _validate_hex(
        value["classification_detail_hash"],
        HEX64,
        "redirect detail hash",
    )


def validate_migration_episode_d6(
    episode: Mapping[str, Any],
    expected_receipt_hash: str,
) -> str:
    """Validate one exact causal episode against its frozen D5 receipt."""

    if not isinstance(episode, Mapping) or set(episode) != EPISODE_FIELDS:
        raise ValueError("PSIM-D6 migration episode fields changed")
    proposal = episode["proposal"]
    if type(proposal) is not int or proposal <= 0:
        raise ValueError("PSIM-D6 migration proposal is malformed")
    expected_path = f"EIPS/eip-{proposal}.md"
    if episode["path"] != expected_path:
        raise ValueError("PSIM-D6 migration episode path changed")
    steps = episode["steps"]
    if not isinstance(steps, list) or len(steps) != 3:
        raise ValueError("PSIM-D6 migration episode must have three steps")

    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or set(step) != EPISODE_STEP_FIELDS:
            raise ValueError("PSIM-D6 migration step fields changed")
        if (
            step["commit_oid"] != MIGRATION_COMMIT_SEQUENCE[index]
            or step["effective_day"] != MIGRATION_DAY_SEQUENCE[index]
            or step["event_type"] != "UPDATE"
            or step["old_path"] != expected_path
            or step["new_path"] != expected_path
            or (
                step["old_blob_class"],
                step["new_blob_class"],
            )
            != MIGRATION_CLASS_SEQUENCE[index]
            or step["outcome_id"] != D5_OUTCOME_SEQUENCE[index]
        ):
            raise ValueError("PSIM-D6 migration causal sequence changed")
        _validate_hex(step["event_id"], HEX64, "event id")
        _validate_hex(step["old_blob_oid"], HEX40, "old blob OID")
        _validate_hex(step["new_blob_oid"], HEX40, "new blob OID")
        _validate_hex(
            step["old_blob_sha256"],
            HEX64,
            "old blob SHA-256",
        )
        _validate_hex(
            step["new_blob_sha256"],
            HEX64,
            "new blob SHA-256",
        )

    for prior, following in zip(steps, steps[1:]):
        if (
            prior["new_blob_oid"] != following["old_blob_oid"]
            or prior["new_blob_sha256"]
            != following["old_blob_sha256"]
        ):
            raise ValueError("PSIM-D6 migration blob continuity changed")

    _validate_redirect(
        episode["lower_redirect"],
        proposal=proposal,
        classification="ERC_MIGRATION_REDIRECT_LOWER_PATH",
        path_case="ercs",
    )
    _validate_redirect(
        episode["upper_redirect"],
        proposal=proposal,
        classification="ERC_MIGRATION_REDIRECT_UPPER_PATH",
        path_case="ERCS",
    )
    _validate_hex(
        expected_receipt_hash,
        HEX64,
        "expected receipt hash",
    )
    observed_hash = canonical_hash(episode)
    if observed_hash != expected_receipt_hash:
        raise ValueError("PSIM-D6 migration episode receipt differs")
    return observed_hash


def migration_receipt_manifest_d6(
    authorized_receipts: Mapping[int, str],
) -> list[dict[str, Any]]:
    items = list(authorized_receipts.items())
    for proposal, receipt_hash in items:
        if (
            type(proposal) is not int
            or proposal <= 0
            or not isinstance(receipt_hash, str)
            or HEX64.fullmatch(receipt_hash) is None
        ):
            raise ValueError(
                "PSIM-D6 migration receipt authority is malformed"
            )
    rows: list[dict[str, Any]] = []
    for proposal, receipt_hash in sorted(items):
        rows.append(
            {
                "proposal": proposal,
                "receipt_hash": receipt_hash,
            }
        )
    if len(rows) != len(authorized_receipts):
        raise ValueError("PSIM-D6 migration receipt authority repeats keys")
    return rows


def _authorize_migration_restoration_with_authority_d6(
    episode: Mapping[str, Any],
    authorized_receipts: Mapping[int, str],
    expected_manifest_hash: str,
) -> dict[str, Any]:
    _validate_hex(
        expected_manifest_hash,
        HEX64,
        "receipt manifest hash",
    )
    observed_manifest_hash = canonical_hash(
        migration_receipt_manifest_d6(authorized_receipts)
    )
    if observed_manifest_hash != expected_manifest_hash:
        raise ValueError("PSIM-D6 migration receipt manifest differs")
    proposal = episode.get("proposal") if isinstance(episode, Mapping) else None
    if type(proposal) is not int or proposal not in authorized_receipts:
        raise ValueError("PSIM-D6 migration proposal is not authorized")
    receipt_hash = validate_migration_episode_d6(
        episode,
        authorized_receipts[proposal],
    )
    return {
        "audit": {
            "authority_receipt_hash": receipt_hash,
            "authority_receipt_manifest_hash": observed_manifest_hash,
            "causal_episode_steps": 3,
            "protocol_version": (
                "psim_d6_exact_migration_restoration_receipt_v1"
            ),
            "quarantine_reason": (
                "EXACT_2023_ETHEREUM_ERC_MIGRATION_EPISODE_RESTORATION"
            ),
        },
        "model": {
            "administrative_quarantined": True,
            "model_visibility": "ADMINISTRATIVE_QUARANTINE",
            "normalized_text_delta_chunks": [],
        },
    }


def authorize_migration_restoration_d6(
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize only an episode in the frozen 365-receipt D5 authority."""

    _binding, authorized_receipts = _load_d5_census_binding()
    return _authorize_migration_restoration_with_authority_d6(
        episode,
        authorized_receipts,
        D5_EPISODE_RECEIPT_MANIFEST_HASH,
    )


def _synthetic_episode(proposal: int = 20) -> dict[str, Any]:
    path = f"EIPS/eip-{proposal}.md"
    blob_oids = ("1" * 40, "2" * 40, "3" * 40, "4" * 40)
    blob_hashes = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    steps = []
    for index in range(3):
        steps.append(
            {
                "commit_oid": MIGRATION_COMMIT_SEQUENCE[index],
                "effective_day": MIGRATION_DAY_SEQUENCE[index],
                "event_id": str(index + 5) * 64,
                "event_type": "UPDATE",
                "new_blob_class": MIGRATION_CLASS_SEQUENCE[index][1],
                "new_blob_oid": blob_oids[index + 1],
                "new_blob_sha256": blob_hashes[index + 1],
                "new_path": path,
                "old_blob_class": MIGRATION_CLASS_SEQUENCE[index][0],
                "old_blob_oid": blob_oids[index],
                "old_blob_sha256": blob_hashes[index],
                "old_path": path,
                "outcome_id": D5_OUTCOME_SEQUENCE[index],
            }
        )
    return {
        "lower_redirect": {
            "classification": "ERC_MIGRATION_REDIRECT_LOWER_PATH",
            "classification_detail_hash": "e" * 64,
            "path_case": "ercs",
            "target_matches_path_proposal": True,
            "target_proposal": proposal,
        },
        "path": path,
        "proposal": proposal,
        "steps": steps,
        "upper_redirect": {
            "classification": "ERC_MIGRATION_REDIRECT_UPPER_PATH",
            "classification_detail_hash": "f" * 64,
            "path_case": "ERCS",
            "target_matches_path_proposal": True,
            "target_proposal": proposal,
        },
    }


def _raises_value_error(function: Any, *arguments: Any) -> bool:
    try:
        function(*arguments)
    except ValueError:
        return True
    return False


def _chunk_synthetic_battery() -> dict[str, Any]:
    serialized_control = serialize_model_delta_rows_d6(
        [
            {
                "direction": "REMOVE",
                "line": "old",
                "section": "ABSTRACT",
            },
            {
                "direction": "ADD",
                "line": "new",
                "section": "ABSTRACT",
            },
        ]
    )
    if serialized_control != (
        "ABSTRACT|REMOVE|old\nABSTRACT|ADD|new"
    ):
        raise RuntimeError("PSIM-D6 model row serialization changed")
    fixtures = {
        "empty": "",
        "exact_8192": "a" * 8_192,
        "over_8192": "b" * 8_193,
        "utf8_boundary": "c" * 8_191 + "한" + "d",
        "lf_boundary": "e" * 8_191 + "\n" + "f",
        "single_oversized_row": "g" * 20_000,
        "historical_max_bytes": "h" * 58_416,
        "exact_65536": "i" * 65_536,
    }
    observed: dict[str, dict[str, Any]] = {}
    transports: dict[str, dict[str, Any]] = {}
    for name, text in fixtures.items():
        transport = build_model_text_transport_d6(text)
        transports[name] = transport
        receipt = transport["audit_receipt"]
        observed[name] = {
            "chunk_count": receipt["chunk_count"],
            "full_text_sha256": receipt["full_text_sha256"],
            "full_text_utf8_bytes": receipt["full_text_utf8_bytes"],
            "receipt_hash": receipt["receipt_hash"],
            "reconstruction_matches": receipt["reconstruction_matches"],
        }

    if (
        observed["empty"]["chunk_count"] != 0
        or observed["exact_8192"]["chunk_count"] != 1
        or observed["over_8192"]["chunk_count"] != 2
        or observed["utf8_boundary"]["chunk_count"] != 2
        or observed["lf_boundary"]["chunk_count"] != 2
        or observed["single_oversized_row"]["chunk_count"] != 3
        or observed["historical_max_bytes"]["chunk_count"] != 8
        or observed["exact_65536"]["chunk_count"] != 8
        or not all(
            row["reconstruction_matches"] is True
            for row in observed.values()
        )
    ):
        raise RuntimeError("PSIM-D6 chunk positive battery failed")
    if not _raises_value_error(
        split_utf8_model_text_d6,
        "j" * 65_537,
    ):
        raise RuntimeError("PSIM-D6 ninth chunk did not fail closed")

    original = transports["single_oversized_row"]["model_chunk_payloads"]
    tampered: list[list[dict[str, Any]]] = []
    tampered.append(copy.deepcopy(original[:-1]))
    tampered.append(copy.deepcopy([*original, original[-1]]))
    swapped = copy.deepcopy(original)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    tampered.append(swapped)
    changed = copy.deepcopy(original)
    changed[0]["normalized_text_delta_chunk"] += "x"
    tampered.append(changed)
    bad_index = copy.deepcopy(original)
    bad_index[0]["chunk_index"] = 1
    tampered.append(bad_index)
    bad_count = copy.deepcopy(original)
    bad_count[0]["chunk_count"] = len(bad_count) + 1
    tampered.append(bad_count)
    extra_field = copy.deepcopy(original)
    extra_field[0]["event_id"] = "0" * 64
    tampered.append(extra_field)
    repartitioned = copy.deepcopy(original)
    moved = repartitioned[1]["normalized_text_delta_chunk"][-1]
    repartitioned[1]["normalized_text_delta_chunk"] = (
        repartitioned[1]["normalized_text_delta_chunk"][:-1]
    )
    repartitioned[2]["normalized_text_delta_chunk"] = (
        moved + repartitioned[2]["normalized_text_delta_chunk"]
    )
    tampered.append(repartitioned)
    if not all(
        _raises_value_error(
            validate_model_chunk_payloads_d6,
            fixtures["single_oversized_row"],
            rows,
        )
        for rows in tampered
    ):
        raise RuntimeError("PSIM-D6 chunk tamper battery failed")
    if not _raises_value_error(
        validate_model_chunk_payloads_d6,
        fixtures["single_oversized_row"] + "x",
        original,
    ):
        raise RuntimeError("PSIM-D6 full-text hash control failed")

    utf8_chunks = transports["utf8_boundary"]["model_chunk_payloads"]
    if (
        len(utf8_chunks[0]["normalized_text_delta_chunk"].encode("utf-8"))
        != 8_191
        or "".join(
            row["normalized_text_delta_chunk"]
            for row in utf8_chunks
        )
        != fixtures["utf8_boundary"]
    ):
        raise RuntimeError("PSIM-D6 UTF-8 boundary battery failed")
    if any(
        "normalized_text_delta_chunk" in row
        for row in observed.values()
    ):
        raise RuntimeError("PSIM-D6 probe result exposed synthetic text")

    return {
        "max_bytes_per_chunk": MAX_MODEL_TEXT_BYTES_PER_CHUNK,
        "max_chunks_per_event": MAX_MODEL_TEXT_CHUNKS_PER_EVENT,
        "model_row_serialization_exact": True,
        "ninth_chunk_fails_closed": True,
        "positive_cases": observed,
        "tamper_cases_rejected": len(tampered) + 1,
        "utf8_boundary_is_lossless": True,
    }


def _migration_synthetic_battery() -> dict[str, Any]:
    episode = _synthetic_episode()
    receipt_hash = canonical_hash(episode)
    authorized = {episode["proposal"]: receipt_hash}
    manifest_hash = canonical_hash(
        migration_receipt_manifest_d6(authorized)
    )
    decision = _authorize_migration_restoration_with_authority_d6(
        episode,
        authorized,
        manifest_hash,
    )
    if (
        decision["model"]["administrative_quarantined"] is not True
        or decision["model"]["normalized_text_delta_chunks"] != []
        or decision["audit"]["authority_receipt_hash"] != receipt_hash
    ):
        raise RuntimeError("PSIM-D6 migration positive battery failed")

    mutations: list[dict[str, Any]] = []
    path = copy.deepcopy(episode)
    path["path"] = "EIPS/eip-21.md"
    mutations.append(path)
    target = copy.deepcopy(episode)
    target["upper_redirect"]["target_proposal"] = 21
    mutations.append(target)
    commit = copy.deepcopy(episode)
    commit["steps"][1]["commit_oid"] = "0" * 40
    mutations.append(commit)
    day = copy.deepcopy(episode)
    day["steps"][2]["effective_day"] = "2023-10-27"
    mutations.append(day)
    klass = copy.deepcopy(episode)
    klass["steps"][2]["old_blob_class"] = "D4_VALID"
    mutations.append(klass)
    continuity = copy.deepcopy(episode)
    continuity["steps"][1]["old_blob_oid"] = "9" * 40
    mutations.append(continuity)
    order = copy.deepcopy(episode)
    order["steps"][0], order["steps"][1] = (
        order["steps"][1],
        order["steps"][0],
    )
    mutations.append(order)
    generic_reverse = copy.deepcopy(episode)
    generic_reverse["steps"] = [generic_reverse["steps"][2]]
    mutations.append(generic_reverse)
    extra = copy.deepcopy(episode)
    extra["future_return"] = 1.0
    mutations.append(extra)

    if not all(
        _raises_value_error(
            _authorize_migration_restoration_with_authority_d6,
            mutation,
            authorized,
            manifest_hash,
        )
        for mutation in mutations
    ):
        raise RuntimeError("PSIM-D6 migration mutation battery failed")
    if not _raises_value_error(
        _authorize_migration_restoration_with_authority_d6,
        episode,
        {},
        manifest_hash,
    ):
        raise RuntimeError("PSIM-D6 unauthorized migration passed")
    extra_authority = {
        **authorized,
        21: "9" * 64,
    }
    if not _raises_value_error(
        _authorize_migration_restoration_with_authority_d6,
        episode,
        extra_authority,
        manifest_hash,
    ):
        raise RuntimeError("PSIM-D6 expanded migration roster passed")
    if not _raises_value_error(
        validate_migration_episode_d6,
        episode,
        "0" * 64,
    ):
        raise RuntimeError("PSIM-D6 altered receipt authority passed")

    return {
        "exact_three_step_episode_authorized": True,
        "generic_reverse_transition_authorized": False,
        "model_text_chunks_for_restoration": 0,
        "negative_cases_rejected": len(mutations) + 3,
        "synthetic_receipt_manifest_hash": manifest_hash,
        "synthetic_receipt_hash": receipt_hash,
    }


def build_probe() -> dict[str, Any]:
    census_binding, receipt_map = _load_d5_census_binding()
    payload: dict[str, Any] = {
        "access_boundary": {
            "d5_census_artifact_read": True,
            "d5_forensic_root_accessed": False,
            "d5_run_invoked": False,
            "external_network_accessed_by_probe": False,
            "historical_proposal_text_accessed": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "official_historical_proposal_source_accessed": False,
            "outcomes_accessed": False,
            "raw_official_text_published": False,
        },
        "candidate": {
            "id": "PSIM-D6",
            "name": (
                "Protocol Specification Intent-Maturity source support, "
                "receipt-bound migration lifecycle plus lossless chunks"
            ),
            "source_representation_successor": True,
        },
        "d5_census_binding": census_binding,
        "mechanism_contract": {
            "administrative_restoration": (
                "EXACT_THREE_STEP_CAUSAL_EPISODE_PLUS_PER_PROPOSAL_"
                "FROZEN_RECEIPT_HASH"
            ),
            "administrative_restoration_model_text_visible": False,
            "chunk_full_text_serialization": (
                "D5_CAUSAL_MODEL_ROWS_SECTION_DIRECTION_LINE_JOINED_BY_LF"
            ),
            "chunk_split": (
                "GREEDY_CONTIGUOUS_UTF8_BYTES_BACKTRACK_CONTINUATION_"
                "BOUNDARY"
            ),
            "chunk_transport_fields": sorted(MODEL_CHUNK_FIELDS),
            "chunk_transport_order": "ZERO_BASED_ASCENDING_CHUNK_INDEX",
            "full_text_reconstruction": "BYTE_FOR_BYTE_REQUIRED",
            "max_bytes_per_chunk": MAX_MODEL_TEXT_BYTES_PER_CHUNK,
            "max_chunks_per_event": MAX_MODEL_TEXT_CHUNKS_PER_EVENT,
            "ninth_chunk": "FAIL_CLOSED_NO_TRUNCATION_OR_SUMMARIZATION",
            "receipt_authority_count": len(receipt_map),
            "restoration_requires_prior_events_not_future_events": True,
            "unknown_episode": "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES",
        },
        "mechanism_version": MECHANISM_VERSION,
        "policy_id": "PSIM-D6-SYNTHETIC-MECHANISM-PROBE",
        "protocol_version": PROTOCOL_VERSION,
        "selection_scope": (
            "AUTHORIZE_D6_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_"
            "EXECUTION"
        ),
        "synthetic_battery": {
            "model_text_chunks": _chunk_synthetic_battery(),
            "migration_restoration": _migration_synthetic_battery(),
        },
        "synthetic_only": True,
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def write_probe(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    target = _safe_output_path(output)
    payload = build_probe()
    raw = canonical_json_bytes(payload)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("existing PSIM-D6 mechanism probe differs")
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
