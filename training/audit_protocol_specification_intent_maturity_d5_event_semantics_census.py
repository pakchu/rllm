"""Audit every PSIM-D5 Ethereum event after terminal rejection.

This is a terminal, source-only forensic census. It never runs PSIM-D5,
fetches objects, mutates the forensic root, or accesses market, model,
reward, trade, PnL, CAGR, strict-MDD, or outcome data. The census decodes
all already hydrated proposal blobs and evaluates all proposal groups so
that a successor is not designed by patching only the first exception.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import (
    build_protocol_specification_intent_maturity_d5_source_support as d5,
)
from training import (
    probe_protocol_specification_intent_maturity_d5_event_semantics
    as semantics,
)


PROTOCOL_VERSION = "psim_d5_post_terminal_event_semantics_census_v1"
SOURCE_ROOT = Path("/tmp/psim-d5-source")
ETHEREUM_ROOT = SOURCE_ROOT / "ethereum-a.git"
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d5_event_semantics_"
    "census_2026-07-26.json"
)

TERMINAL_COMMIT = "0f69f7472d89474052186bbb2b13fa8d6bf5d77f"
TERMINAL_SHA256 = (
    "ffdebf2e5107f08345f16e21adc895d3bfc2f236d6b231322d03c372d4764ca1"
)
TERMINAL_RESULT_HASH = (
    "0a23218e8784599f09e092d4f93942a48111c0af4f8e3ff85e2183eb84f56c56"
)
RUNNER_COMMIT = "90e7740edcd68a3b4c3acf8e9fe9a14f9e4eb8e1"
RUNNER_SHA256 = (
    "744959177c1f18d62cb920f5bd9c1068eb5415c07d4f7d5719af5b37542e0dba"
)
SEMANTICS_COMMIT = "0e62ec05e6861b2619e6737dd594e7306ad7c93a"
SEMANTICS_SHA256 = (
    "d1aaf55effec3df8f38854992b4c60bd39d612e4bd6cd00fe705f60b5cac9d85"
)

FORENSIC_OBJECT_STORE_HASH = (
    "d449f80fd3a6d2f1993e01c6418294d5385381084a9c2b893179bca368bae34a"
)
ETHEREUM_CHAIN_ROWS = 6_958
ETHEREUM_CHAIN_HASH = (
    "c022f028dfe9df0a9d36aeec173f227604d51243c0671a8cf090f687182b88d9"
)
ETHEREUM_GROUP_ROWS = 4_985
ETHEREUM_GROUP_HASH = (
    "a3eea9350bc5d0e1b6131515200cb771338063b7f673c971d67fa1684cda821c"
)
REQUESTED_BLOBS = 5_206
OID_MANIFEST_SHA256 = (
    "8aa47dbe594df92a42ce87f6872f2bb3548f5370371f7668b26c80a47c53c944"
)

EXPECTED_BLOB_CLASS_COUNTS = {
    "D4_DUPLICATE_IDENTICAL_HEADER": 7,
    "D4_MALFORMED_HEADER_LINE": 20,
    "D4_SELF_DEPENDENCY": 9,
    "D4_VALID": 4_440,
    "ERC_MIGRATION_REDIRECT_LOWER_PATH": 365,
    "ERC_MIGRATION_REDIRECT_UPPER_PATH": 365,
}
MODEL_TEXT_BOUND_ERROR = (
    "ValueError: PSIM-D5 model-visible normalized text delta exceeds frozen "
    "bound"
)
REVERSE_ADMIN_ERROR = (
    "ValueError: PSIM-D5 reverse administrative migration is ambiguous"
)
ERROR_IDS = {
    MODEL_TEXT_BOUND_ERROR: "ERROR_MODEL_TEXT_BOUND",
    REVERSE_ADMIN_ERROR: "ERROR_REVERSE_ADMINISTRATIVE_MIGRATION",
}
EXPECTED_EVENT_OUTCOME_COUNTS = {
    "ERROR_MODEL_TEXT_BOUND": 190,
    "ERROR_REVERSE_ADMINISTRATIVE_MIGRATION": 365,
    "PASS_ADMINISTRATIVE_QUARANTINE": 730,
    "PASS_MODEL_VISIBLE": 3_700,
}

LOCAL_GIT_SUBCOMMANDS = frozenset(
    {
        "cat-file",
        "diff-tree",
        "for-each-ref",
        "ls-tree",
        "rev-list",
        "rev-parse",
        "symbolic-ref",
        "verify-pack",
    }
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else d5.REPO_ROOT / candidate


def _safe_output_path(path: str | Path) -> Path:
    requested = Path(path)
    results_root = d5.REPO_ROOT.resolve() / "results"
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
            "PSIM-D5 event census output must be a safe repo-local result"
        )
    return target


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
    return d5.sha256_bytes(
        canonical_json_bytes(payload, pretty=False).rstrip(b"\n")
    )


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D5 forensic authority is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"PSIM-D5 forensic authority is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(
            f"PSIM-D5 forensic authority is noncanonical: {path}"
        )
    return payload


def _assert_terminal_boundary() -> Mapping[str, Any]:
    payload = _read_canonical_json(d5.DEFAULT_REJECTION_PATH)
    ledger = payload.get("access_ledger")
    audit = payload.get("source_audit")
    receipts = (
        audit.get("batch_hydration_receipts")
        if isinstance(audit, Mapping)
        else None
    )
    authority = payload.get("authority")
    if (
        d5.sha256_file(d5.DEFAULT_REJECTION_PATH) != TERMINAL_SHA256
        or payload.get("result_hash") != TERMINAL_RESULT_HASH
        or payload.get("decision") != "reject"
        or payload.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or payload.get("error") != {"type": "ValueError"}
        or payload.get("outcomes_opened") is not False
        or payload.get("profitability_result") is not False
        or payload.get("terminal_action")
        != "REJECT_PSIM_D5_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
        or not isinstance(ledger, Mapping)
        or any(ledger.get(name) != 0 for name in d5.FORBIDDEN_ACCESS_FIELDS)
        or not isinstance(authority, Mapping)
        or authority.get("execution_seal", {}).get("runner")
        != {
            "commit": RUNNER_COMMIT,
            "path": d5.RUNNER_PATH.as_posix(),
            "sha256": RUNNER_SHA256,
        }
        or authority.get("semantics_probe_producer")
        != {
            "commit": SEMANTICS_COMMIT,
            "path": d5.SEMANTICS_PROBE_SCRIPT_PATH.as_posix(),
            "sha256": SEMANTICS_SHA256,
        }
        or not isinstance(receipts, list)
        or len(receipts) != 1
        or receipts[0].get("repository_root_name") != "ethereum-a.git"
        or receipts[0].get("requested_blob_count") != REQUESTED_BLOBS
        or receipts[0].get("oid_manifest_sha256")
        != OID_MANIFEST_SHA256
        or receipts[0].get("hydrated_snapshot_hash")
        != FORENSIC_OBJECT_STORE_HASH
        or receipts[0].get("post_read_snapshot_hash")
        != FORENSIC_OBJECT_STORE_HASH
        or receipts[0].get("post_read_object_store_unchanged") is not True
        or audit.get("blob_semantics_receipts") != {}
        or audit.get("events_sha256") is not None
        or audit.get("cards_sha256") is not None
        or audit.get("controls_sha256") is not None
    ):
        raise RuntimeError("PSIM-D5 terminal boundary changed")
    if (
        d5.sha256_file(d5.RUNNER_PATH) != RUNNER_SHA256
        or d5.sha256_file(d5.SEMANTICS_PROBE_SCRIPT_PATH)
        != SEMANTICS_SHA256
    ):
        raise RuntimeError("PSIM-D5 executable semantics authority changed")
    return payload


def _assert_forensic_root() -> Path:
    if (
        ETHEREUM_ROOT.is_symlink()
        or not ETHEREUM_ROOT.is_dir()
        or ETHEREUM_ROOT.resolve() != ETHEREUM_ROOT
    ):
        raise RuntimeError("PSIM-D5 forensic root is absent or unsafe")
    return ETHEREUM_ROOT


def _assert_local_git_arguments(arguments: Sequence[str]) -> None:
    """Fail closed unless argv names one census-approved local Git reader."""

    values = tuple(arguments)
    if not values:
        raise RuntimeError(
            "PSIM-D5 event census forbids empty Git argv"
        )
    index = 0
    while index < len(values) and values[index] == "-C":
        if index + 1 >= len(values) or not values[index + 1]:
            raise RuntimeError(
                "PSIM-D5 event census forbids malformed Git -C"
            )
        index += 2
    if index >= len(values) or values[index].startswith("-"):
        raise RuntimeError(
            "PSIM-D5 event census forbids unrecognized Git global options"
        )
    if values[index] not in LOCAL_GIT_SUBCOMMANDS:
        raise RuntimeError(
            "PSIM-D5 event census forbids nonlocal Git subcommand"
        )


def _blob_class(blob: semantics.BlobSemantics | None) -> str | None:
    return None if blob is None else d5._blob_classification(blob)


def _event_metadata(
    group: d5.ProposalGroup,
    old: semantics.BlobSemantics | None,
    new: semantics.BlobSemantics | None,
) -> dict[str, Any]:
    return {
        "commit_oid": group.commit_oid,
        "effective_day": group.effective_day.isoformat(),
        "event_id": group.event_id,
        "event_type": group.event_type,
        "new_administrative_class": (
            None if new is None else new.administrative_class
        ),
        "new_blob_class": _blob_class(new),
        "new_blob_oid": group.new_blob_oid,
        "new_blob_sha256": None if new is None else new.blob_sha256,
        "new_metadata_state": None if new is None else new.metadata_state,
        "new_path": group.new_path,
        "old_administrative_class": (
            None if old is None else old.administrative_class
        ),
        "old_blob_class": _blob_class(old),
        "old_blob_oid": group.old_blob_oid,
        "old_blob_sha256": None if old is None else old.blob_sha256,
        "old_metadata_state": None if old is None else old.metadata_state,
        "old_path": group.old_path,
        "proposal": group.proposal_number,
    }


def _side_profile(
    group: d5.ProposalGroup,
    old: semantics.BlobSemantics | None,
    new: semantics.BlobSemantics | None,
) -> dict[str, Any]:
    return {
        "event_type": group.event_type,
        "new_administrative_class": (
            None if new is None else new.administrative_class
        ),
        "new_blob_class": _blob_class(new),
        "new_metadata_state": None if new is None else new.metadata_state,
        "old_administrative_class": (
            None if old is None else old.administrative_class
        ),
        "old_blob_class": _blob_class(old),
        "old_metadata_state": None if old is None else old.metadata_state,
    }


def _profile_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {
            **json.loads(serialized),
            "count": count,
        }
        for serialized, count in sorted(counter.items())
    ]


def _unbounded_model_text_measurement(
    old: semantics.BlobSemantics | None,
    new: semantics.BlobSemantics | None,
) -> tuple[int, int]:
    rows, _changed_sections = semantics._normalized_changed_rows(old, new)
    model_rows = semantics._model_visible_delta_rows(rows)
    text = "\n".join(
        f"{row['section']}|{row['direction']}|{row['line']}"
        for row in model_rows
    )
    return len(text.encode("utf-8")), len(model_rows)


def _group_sides(
    group: d5.ProposalGroup,
    features: Mapping[
        tuple[str, int, str],
        semantics.BlobSemantics,
    ],
) -> tuple[
    semantics.BlobSemantics | None,
    semantics.BlobSemantics | None,
]:
    old = (
        None
        if group.old_blob_oid is None
        else features[
            (
                group.protocol,
                group.proposal_number,
                group.old_blob_oid,
            )
        ]
    )
    new = (
        None
        if group.new_blob_oid is None
        else features[
            (
                group.protocol,
                group.proposal_number,
                group.new_blob_oid,
            )
        ]
    )
    return old, new


def _episode_step(
    group: d5.ProposalGroup,
    old: semantics.BlobSemantics | None,
    new: semantics.BlobSemantics | None,
    outcome_id: str,
) -> dict[str, Any]:
    return {
        "commit_oid": group.commit_oid,
        "effective_day": group.effective_day.isoformat(),
        "event_id": group.event_id,
        "event_type": group.event_type,
        "new_blob_class": _blob_class(new),
        "new_blob_oid": group.new_blob_oid,
        "new_blob_sha256": None if new is None else new.blob_sha256,
        "new_path": group.new_path,
        "old_blob_class": _blob_class(old),
        "old_blob_oid": group.old_blob_oid,
        "old_blob_sha256": None if old is None else old.blob_sha256,
        "old_path": group.old_path,
        "outcome_id": outcome_id,
    }


def _redirect_binding(
    blob: semantics.BlobSemantics,
    raw: bytes,
) -> dict[str, Any]:
    classification, detail = semantics.census.classify_blob(
        blob.proposal_number,
        blob.blob_oid,
        raw,
    )
    redirect = detail.get("redirect")
    if (
        classification not in semantics.ADMINISTRATIVE_CLASSES
        or not isinstance(redirect, Mapping)
        or redirect.get("target_matches_path_proposal") is not True
        or redirect.get("target_proposal") != blob.proposal_number
        or redirect.get("path_case") not in {"ercs", "ERCS"}
    ):
        raise RuntimeError(
            "PSIM-D5 administrative episode redirect binding changed"
        )
    return {
        "classification": classification,
        "classification_detail_hash": blob.classification_detail_hash,
        "path_case": redirect["path_case"],
        "target_matches_path_proposal": True,
        "target_proposal": redirect["target_proposal"],
    }


def _build_administrative_episode_census(
    groups: Sequence[d5.ProposalGroup],
    features: Mapping[
        tuple[str, int, str],
        semantics.BlobSemantics,
    ],
    raw_by_oid: Mapping[str, bytes],
    outcome_by_event_id: Mapping[str, str],
) -> dict[str, Any]:
    by_proposal: dict[int, list[d5.ProposalGroup]] = defaultdict(list)
    for group in groups:
        by_proposal[group.proposal_number].append(group)

    reverse_groups = [
        group
        for group in groups
        if outcome_by_event_id.get(group.event_id)
        == "ERROR_REVERSE_ADMINISTRATIVE_MIGRATION"
    ]
    episodes: list[dict[str, Any]] = []
    sequence_profiles: Counter[str] = Counter()
    consumed_admin_ids: set[str] = set()
    consumed_reverse_ids: set[str] = set()
    for reverse in reverse_groups:
        proposal_groups = by_proposal[reverse.proposal_number]
        index = next(
            (
                position
                for position, row in enumerate(proposal_groups)
                if row.event_id == reverse.event_id
            ),
            None,
        )
        if index is None or index < 2:
            raise RuntimeError(
                "PSIM-D5 administrative restoration lacks causal prefix"
            )
        triplet = proposal_groups[index - 2 : index + 1]
        outcomes = [
            outcome_by_event_id.get(group.event_id)
            for group in triplet
        ]
        if outcomes != [
            "PASS_ADMINISTRATIVE_QUARANTINE",
            "PASS_ADMINISTRATIVE_QUARANTINE",
            "ERROR_REVERSE_ADMINISTRATIVE_MIGRATION",
        ]:
            raise RuntimeError(
                "PSIM-D5 administrative episode outcome sequence changed"
            )
        side_rows = [
            _group_sides(group, features)
            for group in triplet
        ]
        class_sequence = [
            (_blob_class(old), _blob_class(new))
            for old, new in side_rows
        ]
        expected_classes = [
            ("D4_VALID", "ERC_MIGRATION_REDIRECT_LOWER_PATH"),
            (
                "ERC_MIGRATION_REDIRECT_LOWER_PATH",
                "ERC_MIGRATION_REDIRECT_UPPER_PATH",
            ),
            ("ERC_MIGRATION_REDIRECT_UPPER_PATH", "D4_VALID"),
        ]
        days = [group.effective_day.isoformat() for group in triplet]
        if (
            class_sequence != expected_classes
            or days != ["2023-10-25", "2023-10-25", "2023-10-26"]
            or any(group.event_type != "UPDATE" for group in triplet)
        ):
            raise RuntimeError(
                "PSIM-D5 administrative episode causal shape changed"
            )
        paths = {
            path
            for group in triplet
            for path in (group.old_path, group.new_path)
            if path is not None
        }
        expected_path = f"EIPS/eip-{reverse.proposal_number}.md"
        if paths != {expected_path}:
            raise RuntimeError(
                "PSIM-D5 administrative episode path identity changed"
            )

        lower = side_rows[0][1]
        upper = side_rows[1][1]
        if lower is None or upper is None:
            raise RuntimeError(
                "PSIM-D5 administrative episode redirect side is absent"
            )
        lower_binding = _redirect_binding(
            lower,
            raw_by_oid[lower.blob_oid],
        )
        upper_binding = _redirect_binding(
            upper,
            raw_by_oid[upper.blob_oid],
        )
        if (
            lower_binding["classification"]
            != "ERC_MIGRATION_REDIRECT_LOWER_PATH"
            or lower_binding["path_case"] != "ercs"
            or upper_binding["classification"]
            != "ERC_MIGRATION_REDIRECT_UPPER_PATH"
            or upper_binding["path_case"] != "ERCS"
        ):
            raise RuntimeError(
                "PSIM-D5 administrative episode redirect case changed"
            )

        steps = [
            _episode_step(group, old, new, outcome)
            for group, (old, new), outcome in zip(
                triplet,
                side_rows,
                outcomes,
                strict=True,
            )
        ]
        episode = {
            "lower_redirect": lower_binding,
            "path": expected_path,
            "proposal": reverse.proposal_number,
            "steps": steps,
            "upper_redirect": upper_binding,
        }
        episodes.append(episode)
        consumed_admin_ids.update(
            group.event_id for group in triplet[:2]
        )
        consumed_reverse_ids.add(triplet[2].event_id)
        sequence_profiles[
            json.dumps(
                {
                    "class_sequence": class_sequence,
                    "commit_sequence": [
                        group.commit_oid for group in triplet
                    ],
                    "day_sequence": days,
                    "outcome_sequence": outcomes,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ] += 1

    expected_admin_ids = {
        event_id
        for event_id, outcome_id in outcome_by_event_id.items()
        if outcome_id == "PASS_ADMINISTRATIVE_QUARANTINE"
    }
    expected_reverse_ids = {
        event_id
        for event_id, outcome_id in outcome_by_event_id.items()
        if outcome_id == "ERROR_REVERSE_ADMINISTRATIVE_MIGRATION"
    }
    if (
        len(episodes) != 365
        or consumed_admin_ids != expected_admin_ids
        or consumed_reverse_ids != expected_reverse_ids
    ):
        raise RuntimeError(
            "PSIM-D5 administrative episode coverage changed"
        )

    episodes.sort(key=lambda row: row["proposal"])
    receipt_rows = [
        {
            "proposal": episode["proposal"],
            "receipt_hash": canonical_hash(episode),
        }
        for episode in episodes
    ]
    return {
        "all_redirect_targets_match_path_proposal": True,
        "covered_administrative_quarantine_events": len(
            consumed_admin_ids
        ),
        "covered_reverse_error_events": len(consumed_reverse_ids),
        "episode_count": len(episodes),
        "episode_roster_hash": canonical_hash(episodes),
        "per_proposal_receipt_hashes": receipt_rows,
        "proposal_roster_hash": canonical_hash(
            [row["proposal"] for row in episodes]
        ),
        "representative": episodes[0],
        "sequence_profiles": _profile_rows(sequence_profiles),
    }


def build_census() -> dict[str, Any]:
    """Read the terminal D5 root locally and evaluate every event once."""

    terminal = _assert_terminal_boundary()
    repo = _assert_forensic_root()
    original_run_git = d5._run_git

    def no_network_git(
        arguments: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        _assert_local_git_arguments(arguments)
        if kwargs.get("network"):
            raise RuntimeError("PSIM-D5 event census forbids network Git")
        return original_run_git(arguments, **kwargs)

    d5._run_git = no_network_git
    ledger = d5.AccessLedger.zero()
    try:
        before = d5._object_store_snapshot(repo, ledger)
        if canonical_hash(before) != FORENSIC_OBJECT_STORE_HASH:
            raise RuntimeError("PSIM-D5 forensic object store changed")
        records = d5.collect_commit_chain(repo, "ethereum", ledger)
        groups, issues = d5.collect_proposal_groups(repo, records, ledger)
        if issues:
            raise RuntimeError("PSIM-D5 forensic proposal groups changed")
        if (
            len(records) != ETHEREUM_CHAIN_ROWS
            or d5.canonical_hash(d5._commit_rows(records))
            != ETHEREUM_CHAIN_HASH
            or len(groups) != ETHEREUM_GROUP_ROWS
            or d5.canonical_hash(d5._group_rows(groups))
            != ETHEREUM_GROUP_HASH
        ):
            raise RuntimeError("PSIM-D5 forensic chain or groups changed")
        object_ids = sorted(
            {
                oid
                for group in groups
                for oid in (group.old_blob_oid, group.new_blob_oid)
                if oid is not None
            }
        )
        manifest = ("\n".join(object_ids) + "\n").encode("ascii")
        if (
            len(object_ids) != REQUESTED_BLOBS
            or d5.sha256_bytes(manifest) != OID_MANIFEST_SHA256
        ):
            raise RuntimeError("PSIM-D5 forensic OID manifest changed")
        raw_by_oid = dict(
            d5._cat_file_batch_local(
                repo,
                object_ids,
                expected_type="blob",
                ledger=ledger,
            )
        )

        features: dict[
            tuple[str, int, str],
            semantics.BlobSemantics,
        ] = {}
        decode_errors: Counter[str] = Counter()
        decode_representatives: dict[str, dict[str, Any]] = {}
        for group in groups:
            for side, oid, path in (
                ("old", group.old_blob_oid, group.old_path),
                ("new", group.new_blob_oid, group.new_path),
            ):
                if oid is None:
                    continue
                key = (group.protocol, group.proposal_number, oid)
                if key in features:
                    continue
                try:
                    features[key] = d5.parse_blob_features(
                        group.protocol,
                        group.proposal_number,
                        oid,
                        raw_by_oid[oid],
                    )
                except Exception as error:
                    profile = f"{type(error).__name__}: {error}"
                    decode_errors[profile] += 1
                    decode_representatives.setdefault(
                        profile,
                        {
                            "blob_oid": oid,
                            "effective_day": group.effective_day.isoformat(),
                            "path": path,
                            "proposal": group.proposal_number,
                            "raw_sha256": d5.sha256_bytes(raw_by_oid[oid]),
                            "side": side,
                        },
                    )

        blob_class_counts = Counter(
            d5._blob_classification(blob)
            for blob in features.values()
        )
        outcome_counts: Counter[str] = Counter()
        outcome_rows: list[dict[str, str]] = []
        outcome_by_event_id: dict[str, str] = {}
        failure_counts: Counter[str] = Counter()
        failure_days: dict[str, Counter[str]] = defaultdict(Counter)
        failure_event_types: dict[str, Counter[str]] = defaultdict(Counter)
        failure_proposals: dict[str, set[int]] = defaultdict(set)
        failure_event_ids: dict[str, list[str]] = defaultdict(list)
        failure_side_profiles: dict[str, Counter[str]] = defaultdict(Counter)
        failure_representatives: dict[str, dict[str, Any]] = {}
        failure_type_counts: Counter[str] = Counter()
        overflow_rows: list[dict[str, Any]] = []
        max_successful_text: dict[str, Any] = {"bytes": -1}

        ordered_groups = sorted(
            groups,
            key=lambda row: (
                row.effective_day,
                row.first_parent_index,
                row.proposal_number,
                row.event_id,
            ),
        )
        for group in ordered_groups:
            old = (
                None
                if group.old_blob_oid is None
                else features.get(
                    (
                        group.protocol,
                        group.proposal_number,
                        group.old_blob_oid,
                    )
                )
            )
            new = (
                None
                if group.new_blob_oid is None
                else features.get(
                    (
                        group.protocol,
                        group.proposal_number,
                        group.new_blob_oid,
                    )
                )
            )
            old_decode_missing = (
                group.old_blob_oid is not None and old is None
            )
            new_decode_missing = (
                group.new_blob_oid is not None and new is None
            )
            if old_decode_missing or new_decode_missing:
                outcome_id = "ERROR_BLOB_DECODE_UNAVAILABLE"
                outcome_counts[outcome_id] += 1
                outcome_rows.append(
                    {"event_id": group.event_id, "outcome_id": outcome_id}
                )
                outcome_by_event_id[group.event_id] = outcome_id
                continue
            try:
                event = semantics.build_event_semantics_d5(
                    group.protocol,
                    group.proposal_number,
                    old_path=group.old_path,
                    new_path=group.new_path,
                    old=old,
                    new=new,
                )
            except Exception as error:
                profile = f"{type(error).__name__}: {error}"
                outcome_id = ERROR_IDS.get(profile, "ERROR_UNREGISTERED")
                outcome_counts[outcome_id] += 1
                outcome_rows.append(
                    {"event_id": group.event_id, "outcome_id": outcome_id}
                )
                outcome_by_event_id[group.event_id] = outcome_id
                failure_counts[profile] += 1
                failure_type_counts[type(error).__name__] += 1
                day = group.effective_day.isoformat()
                failure_days[profile][day] += 1
                failure_event_types[profile][group.event_type] += 1
                failure_proposals[profile].add(group.proposal_number)
                failure_event_ids[profile].append(group.event_id)
                failure_side_profiles[profile][
                    json.dumps(
                        _side_profile(group, old, new),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ] += 1
                failure_representatives.setdefault(
                    profile,
                    _event_metadata(group, old, new),
                )
                if profile == MODEL_TEXT_BOUND_ERROR:
                    byte_count, line_count = (
                        _unbounded_model_text_measurement(old, new)
                    )
                    overflow_rows.append(
                        {
                            "bytes": byte_count,
                            "event_id": group.event_id,
                            "model_line_change_count": line_count,
                        }
                    )
                continue

            visibility = event["model_visibility"]
            outcome_id = f"PASS_{visibility}"
            outcome_counts[outcome_id] += 1
            outcome_rows.append(
                {"event_id": group.event_id, "outcome_id": outcome_id}
            )
            outcome_by_event_id[group.event_id] = outcome_id
            text_bytes = len(
                event["normalized_text_delta"].encode("utf-8")
            )
            if text_bytes > max_successful_text["bytes"]:
                max_successful_text = {
                    "bytes": text_bytes,
                    **_event_metadata(group, old, new),
                }

        administrative_episodes = _build_administrative_episode_census(
            ordered_groups,
            features,
            raw_by_oid,
            outcome_by_event_id,
        )
        after = d5._object_store_snapshot(repo, ledger)
        if before != after:
            raise RuntimeError("PSIM-D5 event census mutated object store")
    finally:
        d5._run_git = original_run_git

    observed_blob_counts = dict(sorted(blob_class_counts.items()))
    observed_outcomes = dict(sorted(outcome_counts.items()))
    if decode_errors:
        raise RuntimeError("PSIM-D5 blob decoder census found exceptions")
    if len(features) != REQUESTED_BLOBS:
        raise RuntimeError("PSIM-D5 blob decoder census is incomplete")
    if observed_blob_counts != EXPECTED_BLOB_CLASS_COUNTS:
        raise RuntimeError("PSIM-D5 blob class census changed")
    if observed_outcomes != EXPECTED_EVENT_OUTCOME_COUNTS:
        raise RuntimeError("PSIM-D5 event exception census changed")
    if sum(observed_outcomes.values()) != ETHEREUM_GROUP_ROWS:
        raise RuntimeError("PSIM-D5 event exception census is incomplete")

    overflow_sizes = sorted(row["bytes"] for row in overflow_rows)
    model_text_bound = d5.core.prereg.MAX_MODEL_TEXT_BYTES_PER_EVENT
    if (
        len(overflow_rows)
        != EXPECTED_EVENT_OUTCOME_COUNTS["ERROR_MODEL_TEXT_BOUND"]
        or not overflow_sizes
        or overflow_sizes[0] <= model_text_bound
        or max_successful_text["bytes"] > model_text_bound
    ):
        raise RuntimeError("PSIM-D5 text-bound error profile changed")

    failure_profiles: dict[str, Any] = {}
    for profile, count in sorted(failure_counts.items()):
        failure_profiles[profile] = {
            "count": count,
            "effective_day_counts": dict(
                sorted(failure_days[profile].items())
            ),
            "error_type": profile.split(":", 1)[0],
            "event_roster_hash": canonical_hash(
                sorted(failure_event_ids[profile])
            ),
            "event_type_counts": dict(
                sorted(failure_event_types[profile].items())
            ),
            "message": profile.split(": ", 1)[1],
            "proposal_count": len(failure_proposals[profile]),
            "proposal_roster": sorted(failure_proposals[profile]),
            "proposal_roster_hash": canonical_hash(
                sorted(failure_proposals[profile])
            ),
            "representative": failure_representatives[profile],
            "side_profiles": _profile_rows(
                failure_side_profiles[profile]
            ),
        }
    failure_profiles[MODEL_TEXT_BOUND_ERROR]["overflow_measurement"] = {
        "allowed_bytes_per_event": model_text_bound,
        "event_size_roster_hash": canonical_hash(
            sorted(overflow_rows, key=lambda row: row["event_id"])
        ),
        "max_bytes": overflow_sizes[-1],
        "max_model_line_changes": max(
            row["model_line_change_count"] for row in overflow_rows
        ),
        "min_bytes": overflow_sizes[0],
        "min_model_line_changes": min(
            row["model_line_change_count"] for row in overflow_rows
        ),
        "observed_events": len(overflow_sizes),
        "sum_bytes": sum(overflow_sizes),
        "sum_model_line_changes": sum(
            row["model_line_change_count"] for row in overflow_rows
        ),
    }

    payload: dict[str, Any] = {
        "access_boundary": {
            "d5_forensic_root_read": True,
            "d5_run_invoked": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "network_commands": ledger.network_commands,
            "outcomes_accessed": False,
            "raw_text_published": False,
            "source_objects_mutated": False,
        },
        "candidate_selection": {
            "authorized": False,
            "d5_forensic_root_candidate_authorized": False,
            "d5_rerun_authorized": False,
            "next_step": (
                "OUTCOME_BLIND_D6_MECHANISM_DESIGN_AND_SYNTHETIC_PROBE_"
                "BEFORE_PREREGISTRATION"
            ),
            "sequential_first_exception_patching_rejected": True,
        },
        "census": {
            "blob_class_counts": observed_blob_counts,
            "blob_decode_error_counts": dict(sorted(decode_errors.items())),
            "blob_decode_representatives": dict(
                sorted(decode_representatives.items())
            ),
            "decoded_blob_contexts": len(features),
            "administrative_episode_census": administrative_episodes,
            "event_error_count": sum(failure_counts.values()),
            "event_outcome_counts": observed_outcomes,
            "event_outcome_roster_hash": canonical_hash(
                sorted(outcome_rows, key=lambda row: row["event_id"])
            ),
            "event_success_count": (
                observed_outcomes["PASS_ADMINISTRATIVE_QUARANTINE"]
                + observed_outcomes["PASS_MODEL_VISIBLE"]
            ),
            "failure_profiles": failure_profiles,
            "failure_type_counts": dict(sorted(failure_type_counts.items())),
            "groups_evaluated": len(ordered_groups),
            "max_successful_model_text": max_successful_text,
        },
        "forensic_source": {
            "chain_hash": ETHEREUM_CHAIN_HASH,
            "commit_rows": len(records),
            "git_commands": ledger.git_commands,
            "group_hash": ETHEREUM_GROUP_HASH,
            "groups": len(groups),
            "object_store_after_hash": canonical_hash(after),
            "object_store_before_hash": canonical_hash(before),
            "object_store_unchanged": before == after,
            "oid_manifest_sha256": d5.sha256_bytes(manifest),
            "requested_blobs": len(object_ids),
            "root": str(repo),
            "source_path_rows_opened": ledger.source_path_rows_opened,
        },
        "policy_id": "PSIM-D5-POST-TERMINAL-EVENT-SEMANTICS-CENSUS",
        "protocol_version": PROTOCOL_VERSION,
        "research_implication": {
            "administrative_lifecycle_contract_is_not_total": True,
            "fixed_whole_event_model_text_bound_is_not_total": True,
            "required_administrative_successor_property": (
                "EXACT_CAUSAL_MIGRATION_EPISODE_QUARANTINE_INCLUDING_"
                "RESTORATION_WITHOUT_GENERAL_EXCEPTION"
            ),
            "required_text_successor_property": (
                "LOSSLESS_AUDIT_DIFF_WITH_DETERMINISTIC_CAUSAL_BOUNDED_"
                "MODEL_CHUNKS_NO_TRUNCATION_OR_SUMMARIZATION"
            ),
            "source_decoder_is_total_for_all_hydrated_blobs": True,
        },
        "semantics_binding": {
            "commit": SEMANTICS_COMMIT,
            "path": d5.SEMANTICS_PROBE_SCRIPT_PATH.as_posix(),
            "sha256": SEMANTICS_SHA256,
        },
        "terminal_binding": {
            "commit": TERMINAL_COMMIT,
            "path": d5.DEFAULT_REJECTION_PATH.as_posix(),
            "result_hash": terminal["result_hash"],
            "runner": {
                "commit": RUNNER_COMMIT,
                "path": d5.RUNNER_PATH.as_posix(),
                "sha256": RUNNER_SHA256,
            },
            "sha256": TERMINAL_SHA256,
        },
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def write_census(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    target = _safe_output_path(output)
    payload = build_census()
    raw = canonical_json_bytes(payload)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("existing PSIM-D5 event census differs")
    target.write_bytes(raw)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()
    payload = write_census(arguments.output)
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
