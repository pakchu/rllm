"""Audit the terminal PSIM-D6 Bitcoin grammar failures.

This is a source-only, post-terminal forensic census. It never invokes the
PSIM-D6 evaluator, fetches objects, mutates the forensic source root, or
accesses market, model, reward, trade, PnL, CAGR, strict-MDD, or outcome data.
It classifies every already hydrated Bitcoin proposal blob under grammar-level
rules so a successor cannot be designed around observed proposal identities.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import (
    build_protocol_specification_intent_maturity_d6_source_support as d6,
)


PROTOCOL_VERSION = "psim_d6_post_terminal_bitcoin_grammar_census_v1"
POLICY_ID = "PSIM-D6-POST-TERMINAL-BITCOIN-GRAMMAR-CENSUS"
SOURCE_ROOT = Path("/tmp/psim-d6-source")
QUARANTINED_BITCOIN_ROOT = SOURCE_ROOT / "bitcoin-a.git"
FORENSIC_BITCOIN_ROOT = SOURCE_ROOT / "bitcoin-b.git"
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d6_bitcoin_grammar_"
    "census_2026-07-26.json"
)

TERMINAL_COMMIT = "aef35e00f3ddcb91f6f4b6a37ff40d9d9f67a7a4"
TERMINAL_SHA256 = (
    "f3e69893270be0d37299e78b651daa9208e1d05f07f24b39f6d1cf9a71c5d49f"
)
TERMINAL_RESULT_HASH = (
    "052c8a0c5f3584a3c9a970f1fcfc434ebfd59a6aa25d5e087c6554aa3f2c31da"
)
RUNNER_COMMIT = "5c3f3f6d26046a8bc7b2f7ad09178d944d61e17b"
RUNNER_SHA256 = (
    "bc78fb2ff6ac0b4f0cebaedd01d03a75830f97be81cd9a736e47e6ead46a9f8f"
)
SEAL_COMMIT = "8185e14b2e98fef6a4f8545828dc48b7d98417f2"
SEAL_SHA256 = (
    "cf9bdbea467a499c6075059ef9275f00699fb0431fa27643751539ffdea64e1d"
)

BITCOIN_CHAIN_ROWS = 1_482
BITCOIN_CHAIN_HASH = (
    "7e60f24b78aa863a2b317a7dc3a32b2af8e367c3d25f4a97012f4ddfd28d89d2"
)
BITCOIN_GROUP_ROWS = 371
BITCOIN_GROUP_HASH = (
    "3f7a8e10bb5f9ba57bb0231b5cd54a613fb81e67830c1ec1d9781fe0d22b6a8b"
)
REQUESTED_BLOBS = 434
OID_MANIFEST_SHA256 = (
    "33b974cdc205d35aa6436ea38424b81a272735cae8042679422906d72affc332"
)
FORENSIC_OBJECT_STORE_HASH = (
    "cbbcfe08eb5e20cb4fe67d28ef482a35c520dd820a30c1be0daa1dc2a5e1c756"
)
QUARANTINED_OBJECT_STORE_HASH = (
    "95ab65ba62182122cba39948a936f32a652201ae89684566c69a7e105f2650ec"
)
QUARANTINED_OBJECT_COUNT = 10_354
PRISTINE_OBJECT_COUNT = 9_890
TERMINAL_ERROR_EVENT_COUNT = 7
TERMINAL_ERROR_BLOB_COUNT = 8
TERMINAL_SEMANTIC_ERROR_ROSTER_HASH = (
    "5c58263e37c5970babccdb55f8622effd710b2e82a7368d1ccf424f7e0af7c9f"
)
TERMINAL_ERROR_PROFILE_HASH = (
    "cc140ad529a920164711396861dee05abbf551c3d7acda9765bac4f5c7906150"
)

VALID_CLASS = "D4_VALID"
LATER_PRE_CLASS = "BIP_LATER_EXACT_PRE_HEADER_AFTER_NONHEADER_PREFIX"
PREFIXED_DEPENDENCY_CLASS = "BIP_PREFIXED_DECIMAL_DEPENDENCY_TOKEN"
UNKNOWN_CLASS = "UNKNOWN_BITCOIN_GRAMMAR"

EXPECTED_GRAMMAR_CLASS_COUNTS = {
    VALID_CLASS: 426,
    LATER_PRE_CLASS: 7,
    PREFIXED_DEPENDENCY_CLASS: 1,
}
EXPECTED_BASELINE_ERROR_COUNTS = {
    "ValueError: PSIM malformed header line": 7,
    "ValueError: PSIM proposal number is not ASCII decimal": 1,
}
EXPECTED_ERROR_EVENT_PROFILE_COUNTS = {
    (
        "UPDATE",
        LATER_PRE_CLASS,
        LATER_PRE_CLASS,
    ): 6,
    (
        "CREATE",
        None,
        PREFIXED_DEPENDENCY_CLASS,
    ): 1,
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
    return candidate if candidate.is_absolute() else d6.REPO_ROOT / candidate


def _safe_output_path(path: str | Path) -> Path:
    requested = Path(path)
    results_root = d6.REPO_ROOT.resolve() / "results"
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
            "PSIM-D6 Bitcoin census output must be a safe flat result"
        )
    return target


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D6 census authority is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"PSIM-D6 census authority is unreadable: {path}"
        ) from error
    if (
        not isinstance(payload, dict)
        or raw != d6.canonical_json_bytes(payload)
    ):
        raise RuntimeError(
            f"PSIM-D6 census authority is noncanonical: {path}"
        )
    return payload


def _terminal_bitcoin_receipts(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    audit = payload.get("source_audit")
    receipts = (
        audit.get("blob_semantics_receipts")
        if isinstance(audit, Mapping)
        else None
    )
    if not isinstance(receipts, Mapping):
        raise RuntimeError("PSIM-D6 terminal semantics receipts are absent")
    left = receipts.get("bitcoin:a")
    right = receipts.get("bitcoin:b")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise RuntimeError("PSIM-D6 terminal Bitcoin replicas are absent")
    return left, right


def _assert_terminal_boundary() -> Mapping[str, Any]:
    payload = _read_canonical_json(d6.DEFAULT_REJECTION_PATH)
    ledger = payload.get("access_ledger")
    audit = payload.get("source_audit")
    left, right = _terminal_bitcoin_receipts(payload)
    gates = payload.get("gates")
    expected_gate_states = [
        ("sealed_git_identity_and_object_integrity", True),
        ("first_parent_traversal_and_causal_clock", True),
        ("path_object_grammar_and_unique_proposal_tree", True),
        ("historical_blob_preamble_dependency_integrity", False),
    ]
    if (
        d6.sha256_file(d6.DEFAULT_REJECTION_PATH) != TERMINAL_SHA256
        or payload.get("result_hash") != TERMINAL_RESULT_HASH
        or payload.get("decision") != "reject"
        or payload.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or payload.get("outcomes_opened") is not False
        or payload.get("profitability_result") is not False
        or payload.get("terminal_action") != d6.FAILURE_ACTION
        or not isinstance(ledger, Mapping)
        or any(ledger.get(name) != 0 for name in d6.FORBIDDEN_ACCESS_FIELDS)
        or not isinstance(audit, Mapping)
        or audit.get("source_run_attempt") != 1
        or audit.get("repair_or_provider_swap_used") is not False
        or audit.get("checkout_created") is not False
        or not isinstance(gates, list)
        or [
            (gate.get("name"), gate.get("passed"))
            for gate in gates
        ]
        != expected_gate_states
        or gates[3].get("failure") != "bitcoin"
        or left != right
        or left.get("requested_blob_count") != REQUESTED_BLOBS
        or left.get("decode_error_blob_count")
        != TERMINAL_ERROR_BLOB_COUNT
        or left.get("semantic_error_count")
        != TERMINAL_ERROR_EVENT_COUNT
        or left.get("semantic_error_roster_hash")
        != TERMINAL_SEMANTIC_ERROR_ROSTER_HASH
        or left.get("raw_or_normalized_text_published") is not False
    ):
        raise RuntimeError("PSIM-D6 terminal boundary changed")
    if (
        d6.sha256_file(d6.RUNNER_PATH) != RUNNER_SHA256
        or d6.sha256_file(d6.EXECUTION_SEAL_PATH) != SEAL_SHA256
    ):
        raise RuntimeError("PSIM-D6 executable authority changed")
    return payload


def _assert_forensic_root() -> Path:
    root = FORENSIC_BITCOIN_ROOT
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve() != root
    ):
        raise RuntimeError(
            "PSIM-D6 pristine Bitcoin forensic root is absent or unsafe"
        )
    if (
        QUARANTINED_BITCOIN_ROOT.is_symlink()
        or not QUARANTINED_BITCOIN_ROOT.is_dir()
    ):
        raise RuntimeError(
            "PSIM-D6 quarantined Bitcoin root marker is absent or unsafe"
        )
    return root


def _assert_local_git_arguments(arguments: Sequence[str]) -> None:
    values = tuple(arguments)
    if not values:
        raise RuntimeError("PSIM-D6 Bitcoin census forbids empty Git argv")
    if values[0] == "verify-pack":
        if (
            len(values) != 3
            or values[1] != "-v"
            or Path(values[2]).parent
            != FORENSIC_BITCOIN_ROOT / "objects" / "pack"
            or Path(values[2]).suffix != ".pack"
        ):
            raise RuntimeError(
                "PSIM-D6 Bitcoin census forbids unsafe verify-pack argv"
            )
        return
    if (
        len(values) < 4
        or values[0] != "-C"
        or values[1] != str(FORENSIC_BITCOIN_ROOT)
    ):
        raise RuntimeError(
            "PSIM-D6 Bitcoin census requires the pristine exact root"
        )
    command = values[2]
    if command not in LOCAL_GIT_SUBCOMMANDS - {"verify-pack"}:
        raise RuntimeError(
            "PSIM-D6 Bitcoin census forbids nonlocal Git subcommand"
        )
    tail = values[3:]
    allowed = False
    if command == "cat-file":
        allowed = tail in {
            ("--batch",),
            (
                "--batch-all-objects",
                "--batch-check=%(objectname) %(objecttype)",
            ),
        }
    elif command == "for-each-ref":
        allowed = tail == ("--format=%(refname) %(objectname)",)
    elif command == "symbolic-ref":
        allowed = tail == ("HEAD",)
    elif command == "rev-parse":
        allowed = tail in {
            ("HEAD^{commit}",),
            (f"{d6.SEALED_REF}^{{commit}}",),
        }
    elif command == "rev-list":
        allowed = tail == (
            "--first-parent",
            "--reverse",
            d6.SEALED_REF,
        )
    elif command == "ls-tree":
        allowed = (
            len(tail) == 4
            and tail[:3] == ("-r", "-z", "--name-only")
            and d6.HEX40.fullmatch(tail[3]) is not None
        )
    elif command == "diff-tree":
        root_prefix = (
            "--root",
            "--no-commit-id",
            "-r",
            "--raw",
            "-z",
            "--no-renames",
        )
        update_prefix = (
            "--no-commit-id",
            "-r",
            "--raw",
            "-z",
            "--no-renames",
        )
        allowed = (
            len(tail) == len(root_prefix) + 1
            and tail[: len(root_prefix)] == root_prefix
            and d6.HEX40.fullmatch(tail[-1]) is not None
        ) or (
            len(tail) == len(update_prefix) + 2
            and tail[: len(update_prefix)] == update_prefix
            and all(
                d6.HEX40.fullmatch(value) is not None
                for value in tail[-2:]
            )
        )
    if not allowed:
        raise RuntimeError(
            "PSIM-D6 Bitcoin census forbids unrecognized Git argv shape"
        )


def _error_profile(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _matching_later_pre_headers(
    lines: Sequence[str],
    proposal_number: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for opening in (
        index for index, line in enumerate(lines) if line == "<pre>"
    ):
        closing = next(
            (
                index
                for index, line in enumerate(
                    lines[opening + 1 :],
                    start=opening + 1,
                )
                if line == "</pre>"
            ),
            None,
        )
        if closing is None:
            continue
        candidate = (
            "\n".join(lines[opening : closing + 1]) + "\n"
        ).encode("utf-8")
        try:
            header = d6.parse_bip_preamble(candidate)
            parsed = d6.core.prereg.parse_positive_proposal_number(
                header["bip"]
            )
            dependencies = d6.core._dependency_edges(
                "bitcoin",
                header,
                proposal_number,
            )
        except (KeyError, ValueError):
            continue
        if parsed != proposal_number:
            continue
        prefix = tuple(lines[:opening])
        if not any(line for line in prefix):
            continue
        matches.append(
            {
                "candidate_header_field_count": len(header),
                "candidate_dependency_edge_count": sum(
                    len(values) for values in dependencies.values()
                ),
                "candidate_path_number_matches": True,
                "opening_line_index": opening,
                "prefix_bytes": len(
                    "\n".join(prefix).encode("utf-8")
                ),
                "prefix_nonblank_lines": sum(
                    1 for line in prefix if line
                ),
            }
        )
    return matches


def _parse_prefixed_dependencies(
    header: Mapping[str, str],
    proposal_number: int,
) -> dict[str, Any] | None:
    parsed: list[int] = []
    prefixed = 0
    bare = 0
    fields: list[str] = []
    for field_name in d6.core.prereg.BIP_DEPENDENCY_FIELDS:
        value = header.get(field_name)
        if value is None:
            continue
        fields.append(field_name)
        if "\n" in value:
            return None
        tokens = value.split(",") if value else []
        if len(tokens) > d6.core.prereg.MAX_DEPENDENCIES:
            return None
        for token in tokens:
            stripped = token.strip(" \t")
            if re.fullmatch(r"BIP-[0-9]+", stripped, re.ASCII):
                normalized = stripped.removeprefix("BIP-")
                prefixed += 1
            elif re.fullmatch(r"[0-9]+", stripped, re.ASCII):
                normalized = stripped
                bare += 1
            else:
                return None
            try:
                dependency = (
                    d6.core.prereg.parse_positive_proposal_number(
                        normalized
                    )
                )
            except ValueError:
                return None
            if dependency == proposal_number:
                return None
            parsed.append(dependency)
    if prefixed == 0 or len(parsed) != len(set(parsed)):
        return None
    return {
        "bare_decimal_token_count": bare,
        "dependency_field_count": len(fields),
        "dependency_fields": sorted(fields),
        "normalized_dependency_count": len(parsed),
        "prefixed_decimal_token_count": prefixed,
    }


def classify_bitcoin_blob(
    proposal_number: int,
    blob_oid: str,
    raw: bytes,
) -> tuple[str, dict[str, Any], str | None]:
    """Classify one Bitcoin blob without repairing or returning text."""

    try:
        d6.parse_blob_features(
            "bitcoin",
            proposal_number,
            blob_oid,
            raw,
        )
    except Exception as error:
        baseline_error = _error_profile(error)
    else:
        return VALID_CLASS, {}, None

    try:
        lines = tuple(d6.core.prereg.normalize_blob_bytes(raw))
    except Exception:
        return UNKNOWN_CLASS, {}, baseline_error

    try:
        header = d6.parse_bip_preamble(raw)
    except Exception:
        matches = _matching_later_pre_headers(lines, proposal_number)
        if (
            baseline_error == "ValueError: PSIM malformed header line"
            and len(matches) == 1
        ):
            return (
                LATER_PRE_CLASS,
                {
                    "exact_later_pre_candidate_count": 1,
                    **matches[0],
                },
                baseline_error,
            )
        return UNKNOWN_CLASS, {}, baseline_error

    try:
        parsed_number = d6.core.prereg.parse_positive_proposal_number(
            header["bip"]
        )
    except (KeyError, ValueError):
        return UNKNOWN_CLASS, {}, baseline_error
    dependency = _parse_prefixed_dependencies(
        header,
        proposal_number,
    )
    if (
        baseline_error
        == "ValueError: PSIM proposal number is not ASCII decimal"
        and parsed_number == proposal_number
        and dependency is not None
    ):
        return (
            PREFIXED_DEPENDENCY_CLASS,
            {
                "candidate_path_number_matches": True,
                **dependency,
            },
            baseline_error,
        )
    return UNKNOWN_CLASS, {}, baseline_error


def _profile_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {
            **json.loads(serialized),
            "count": count,
        }
        for serialized, count in sorted(counter.items())
    ]


def _measurement_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    )
    return {
        "all_path_numbers_match": all(
            row.get("candidate_path_number_matches") is True
            for row in rows
        ),
        "measurement_roster_hash": d6.canonical_hash(list(rows)),
        "numeric_ranges": {
            key: {
                "max": max(int(row[key]) for row in rows if key in row),
                "min": min(int(row[key]) for row in rows if key in row),
                "sum": sum(int(row[key]) for row in rows if key in row),
            }
            for key in numeric_keys
        },
    }


def build_census() -> dict[str, Any]:
    """Read both terminal roots locally and classify all Bitcoin blobs."""

    terminal = _assert_terminal_boundary()
    forensic_root = _assert_forensic_root()
    original_run_git = d6._run_git

    def no_network_git(
        arguments: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        _assert_local_git_arguments(arguments)
        if kwargs.get("network"):
            raise RuntimeError(
                "PSIM-D6 Bitcoin census forbids network Git"
            )
        return original_run_git(arguments, **kwargs)

    d6._run_git = no_network_git
    ledger = d6.AccessLedger.zero()
    try:
        before = d6._object_store_snapshot(forensic_root, ledger)
        before_hash = d6.canonical_hash(before)
        if (
            before_hash != FORENSIC_OBJECT_STORE_HASH
            or len(before["objects"]) != PRISTINE_OBJECT_COUNT
        ):
            raise RuntimeError(
                "PSIM-D6 Bitcoin forensic object store changed"
            )

        records = d6.collect_commit_chain(
            forensic_root,
            "bitcoin",
            ledger,
        )
        groups, issues = d6.collect_proposal_groups(
            forensic_root,
            records,
            ledger,
        )
        if (
            issues
            or len(records) != BITCOIN_CHAIN_ROWS
            or d6.canonical_hash(d6._commit_rows(records))
            != BITCOIN_CHAIN_HASH
            or len(groups) != BITCOIN_GROUP_ROWS
            or d6.canonical_hash(d6._group_rows(groups))
            != BITCOIN_GROUP_HASH
        ):
            raise RuntimeError(
                "PSIM-D6 Bitcoin forensic chain or groups changed"
            )

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
            or d6.sha256_bytes(manifest) != OID_MANIFEST_SHA256
        ):
            raise RuntimeError(
                "PSIM-D6 Bitcoin forensic OID manifest changed"
            )
        raw_by_oid = dict(
            d6._cat_file_batch_local(
                forensic_root,
                object_ids,
                expected_type="blob",
                ledger=ledger,
            )
        )

        contexts: dict[str, set[int]] = {
            oid: set() for oid in object_ids
        }
        for group in groups:
            for oid in (group.old_blob_oid, group.new_blob_oid):
                if oid is not None:
                    contexts[oid].add(group.proposal_number)
        if any(len(values) != 1 for values in contexts.values()):
            raise RuntimeError(
                "PSIM-D6 Bitcoin blob proposal context is ambiguous"
            )

        class_by_oid: dict[str, str] = {}
        detail_by_oid: dict[str, dict[str, Any]] = {}
        baseline_error_by_oid: dict[str, str] = {}
        grammar_counts: Counter[str] = Counter()
        baseline_errors: Counter[str] = Counter()
        class_rosters: dict[str, list[dict[str, Any]]] = {}
        for oid in object_ids:
            proposal_number = next(iter(contexts[oid]))
            raw = raw_by_oid[oid]
            classification, detail, baseline_error = (
                classify_bitcoin_blob(
                    proposal_number,
                    oid,
                    raw,
                )
            )
            class_by_oid[oid] = classification
            detail_by_oid[oid] = detail
            grammar_counts[classification] += 1
            if baseline_error is not None:
                baseline_error_by_oid[oid] = baseline_error
                baseline_errors[baseline_error] += 1
            class_rosters.setdefault(classification, []).append(
                {
                    "blob_oid": oid,
                    "blob_sha256": d6.sha256_bytes(raw),
                    "proposal_number": proposal_number,
                }
            )

        if (
            dict(sorted(grammar_counts.items()))
            != EXPECTED_GRAMMAR_CLASS_COUNTS
            or dict(sorted(baseline_errors.items()))
            != EXPECTED_BASELINE_ERROR_COUNTS
            or UNKNOWN_CLASS in grammar_counts
        ):
            raise RuntimeError(
                "PSIM-D6 Bitcoin grammar census changed"
            )

        error_event_profiles: Counter[str] = Counter()
        error_event_rows: list[dict[str, Any]] = []
        error_proposals: set[int] = set()
        for group in groups:
            old_class = (
                None
                if group.old_blob_oid is None
                else class_by_oid[group.old_blob_oid]
            )
            new_class = (
                None
                if group.new_blob_oid is None
                else class_by_oid[group.new_blob_oid]
            )
            if old_class == VALID_CLASS and new_class in {
                VALID_CLASS,
                None,
            }:
                continue
            if old_class is None and new_class == VALID_CLASS:
                continue
            if old_class == VALID_CLASS and new_class == VALID_CLASS:
                continue
            profile = {
                "event_type": group.event_type,
                "new_grammar_class": new_class,
                "old_grammar_class": old_class,
            }
            error_event_profiles[
                json.dumps(
                    profile,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ] += 1
            error_event_rows.append(
                {
                    "commit_oid": group.commit_oid,
                    "event_id": group.event_id,
                    "event_type": group.event_type,
                    "new_blob_oid": group.new_blob_oid,
                    "new_grammar_class": new_class,
                    "old_blob_oid": group.old_blob_oid,
                    "old_grammar_class": old_class,
                    "proposal_number": group.proposal_number,
                }
            )
            error_proposals.add(group.proposal_number)

        observed_profiles = {
            (
                row["event_type"],
                row["old_grammar_class"],
                row["new_grammar_class"],
            ): row["count"]
            for row in _profile_rows(error_event_profiles)
        }
        terminal_failures = [
            row
            for row in _terminal_bitcoin_receipts(terminal)[0][
                "event_outcomes"
            ]
            if row.get("passed") is False
        ]
        if (
            observed_profiles != EXPECTED_ERROR_EVENT_PROFILE_COUNTS
            or len(error_event_rows) != TERMINAL_ERROR_EVENT_COUNT
            or {
                row["event_id"] for row in error_event_rows
            }
            != {
                row["event_id"] for row in terminal_failures
            }
            or any(
                row.get("outcome_id")
                != "ERROR_BLOB_DECODE_UNAVAILABLE"
                or row.get("error_profile_hash")
                != TERMINAL_ERROR_PROFILE_HASH
                for row in terminal_failures
            )
        ):
            raise RuntimeError(
                "PSIM-D6 Bitcoin terminal error roster changed"
            )

        after = d6._object_store_snapshot(forensic_root, ledger)
        after_hash = d6.canonical_hash(after)
        if (
            before != after
            or after_hash != FORENSIC_OBJECT_STORE_HASH
            or len(after["objects"]) != PRISTINE_OBJECT_COUNT
        ):
            raise RuntimeError(
                "PSIM-D6 Bitcoin forensic object store mutated"
            )
    finally:
        d6._run_git = original_run_git

    non_valid_classes = (
        LATER_PRE_CLASS,
        PREFIXED_DEPENDENCY_CLASS,
    )
    class_roster_hashes = {
        name: d6.canonical_hash(
            sorted(
                class_rosters[name],
                key=lambda row: (
                    row["proposal_number"],
                    row["blob_oid"],
                ),
            )
        )
        for name in sorted(class_rosters)
    }
    measurements = {
        name: _measurement_summary(
            [
                detail_by_oid[oid]
                for oid in object_ids
                if class_by_oid[oid] == name
            ]
        )
        for name in non_valid_classes
    }
    baseline_error_roster_hash = d6.canonical_hash(
        [
            {
                "blob_oid": oid,
                "error_profile": baseline_error_by_oid[oid],
                "raw_sha256": d6.sha256_bytes(raw_by_oid[oid]),
            }
            for oid in sorted(baseline_error_by_oid)
        ]
    )
    error_event_rows.sort(
        key=lambda row: (
            row["proposal_number"],
            row["commit_oid"],
            row["event_id"],
        )
    )
    proposal_roster_hash = d6.canonical_hash(sorted(error_proposals))

    payload: dict[str, Any] = {
        "access_boundary": {
            "d6_forensic_root_read": True,
            "d6_run_invoked": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "network_commands": ledger.network_commands,
            "outcomes_accessed": False,
            "post_terminal_replica_a_drift_detected": True,
            "quarantined_object_store_read_by_final_census": False,
            "raw_or_normalized_text_published": False,
            "source_objects_mutated": False,
        },
        "candidate_selection": {
            "d6_forensic_root_candidate_authorized": False,
            "d6_rerun_authorized": False,
            "d7_candidate_authorized": False,
            "identity_conditioned_exceptions_authorized": False,
            "next_step": (
                "OUTCOME_BLIND_D7_GRAMMAR_MECHANISM_SYNTHETIC_PROBE_"
                "BEFORE_PREREGISTRATION"
            ),
        },
        "census": {
            "baseline_error_blob_count": len(baseline_error_by_oid),
            "baseline_error_counts": dict(
                sorted(baseline_errors.items())
            ),
            "baseline_error_roster_hash": baseline_error_roster_hash,
            "class_roster_hashes": class_roster_hashes,
            "error_event_count": len(error_event_rows),
            "error_event_profile_counts": _profile_rows(
                error_event_profiles
            ),
            "error_event_roster_hash": d6.canonical_hash(
                error_event_rows
            ),
            "error_proposal_count": len(error_proposals),
            "error_proposal_roster_hash": proposal_roster_hash,
            "grammar_class_counts": dict(sorted(grammar_counts.items())),
            "grammar_class_definitions": {
                LATER_PRE_CLASS: {
                    "authorization_basis": (
                        "FIRST_EXACT_LATER_PRE_BLOCK_WITH_MATCHING_PATH_"
                        "PROPOSAL_AND_VALID_UNREPAIRED_HEADER"
                    ),
                    "identity_fields_used": [],
                    "metadata_repair": False,
                },
                PREFIXED_DEPENDENCY_CLASS: {
                    "authorization_basis": (
                        "EXACT_UPPERCASE_BIP_HYPHEN_PREFIX_BEFORE_"
                        "POSITIVE_DECIMAL_IN_BITCOIN_DEPENDENCY_FIELD"
                    ),
                    "identity_fields_used": [],
                    "metadata_repair": False,
                },
            },
            "grammar_measurements": measurements,
            "terminal_error_event_set_matches": True,
            "terminal_semantic_error_roster_hash": (
                TERMINAL_SEMANTIC_ERROR_ROSTER_HASH
            ),
            "unique_blob_contexts": len(object_ids),
            "unknown_grammar_count": grammar_counts.get(
                UNKNOWN_CLASS,
                0,
            ),
        },
        "forensic_source": {
            "chain_hash": BITCOIN_CHAIN_HASH,
            "commit_rows": len(records),
            "git_commands": ledger.git_commands,
            "group_hash": BITCOIN_GROUP_HASH,
            "groups": len(groups),
            "census_replica_root_name": FORENSIC_BITCOIN_ROOT.name,
            "object_store_after_hash": after_hash,
            "object_store_before_hash": before_hash,
            "object_store_unchanged": True,
            "oid_manifest_sha256": OID_MANIFEST_SHA256,
            "post_terminal_quarantine": {
                "cause_attribution": "UNKNOWN_NOT_INFERRED",
                "census_use_authorized": False,
                "expected_terminal_snapshot_hash": (
                    FORENSIC_OBJECT_STORE_HASH
                ),
                "observed_object_count": QUARANTINED_OBJECT_COUNT,
                "observed_snapshot_hash": QUARANTINED_OBJECT_STORE_HASH,
                "root_name": QUARANTINED_BITCOIN_ROOT.name,
            },
            "requested_blobs": len(object_ids),
            "source_path_rows_opened": ledger.source_path_rows_opened,
            "terminal_replica_receipts_equal": True,
        },
        "policy_id": POLICY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "terminal_binding": {
            "commit": TERMINAL_COMMIT,
            "path": d6.DEFAULT_REJECTION_PATH.as_posix(),
            "result_hash": TERMINAL_RESULT_HASH,
            "runner": {
                "commit": RUNNER_COMMIT,
                "path": d6.RUNNER_PATH.as_posix(),
                "sha256": RUNNER_SHA256,
            },
            "seal": {
                "commit": SEAL_COMMIT,
                "path": d6.EXECUTION_SEAL_PATH.as_posix(),
                "sha256": SEAL_SHA256,
            },
            "sha256": TERMINAL_SHA256,
        },
    }
    payload["result_hash"] = d6.canonical_hash(payload)
    return payload


def write_census(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    target = _safe_output_path(path)
    payload = build_census()
    raw = d6.canonical_json_bytes(payload)
    if os.path.lexists(target):
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise RuntimeError(
                "existing PSIM-D6 Bitcoin census artifact differs"
            )
        return payload
    temporary = target.with_name(target.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(
            "unsafe PSIM-D6 Bitcoin census temporary path"
        )
    temporary.write_bytes(raw)
    os.replace(temporary, target)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit terminal PSIM-D6 Bitcoin grammar failures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    payload = write_census(arguments.output)
    print(
        json.dumps(
            {
                "decision": "FORENSIC_CENSUS_ONLY",
                "grammar_class_counts": payload["census"][
                    "grammar_class_counts"
                ],
                "result_hash": payload["result_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
