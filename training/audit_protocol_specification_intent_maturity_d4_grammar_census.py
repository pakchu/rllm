"""Audit every hydrated PSIM-D4 EIP blob after terminal rejection.

This is a terminal, source-only forensic census. It never runs PSIM-D4,
fetches objects, mutates the forensic root, or accesses market, model,
reward, trade, PnL, CAGR, strict-MDD, or outcome data. The census classifies
all already hydrated Ethereum proposal blobs so that a later candidate is
not designed by repeatedly patching only the first observed parser error.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import (
    build_protocol_specification_intent_maturity_d4_source_support as d4,
)


PROTOCOL_VERSION = "psim_d4_post_terminal_grammar_census_v1"
SOURCE_ROOT = Path("/tmp/psim-d4-source")
ETHEREUM_ROOT = SOURCE_ROOT / "ethereum-a.git"
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d4_grammar_census_"
    "2026-07-26.json"
)

TERMINAL_COMMIT = "e406778f76f252d3b0cddb33242edcb51e984c80"
TERMINAL_SHA256 = (
    "4d947075c0f54c5cd09c732710da0502c87d89fa52029fe81367dd3f27ab2aaf"
)
TERMINAL_RESULT_HASH = (
    "8563ef3ace444896295d7076cd0f839e8f62f89899e312d711f9768f5cbf84aa"
)
FORENSIC_OBJECT_STORE_HASH = (
    "fd0ac6636ab7a954e46deb82188f9963f135b1b92152785c6f50205995766a2a"
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

FIELD_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9 -]*):[ \t]*(.*)",
    re.ASCII,
)
REDIRECT_PATTERN = re.compile(
    r"This file was moved to "
    r"https://github\.com/ethereum/ercs/blob/master/"
    r"(ercs|ERCS)/erc-([0-9]+)\.md",
    re.ASCII,
)

EXPECTED_CLASS_COUNTS = {
    "D4_DUPLICATE_IDENTICAL_HEADER": 7,
    "D4_MALFORMED_HEADER_LINE": 20,
    "D4_SELF_DEPENDENCY": 9,
    "D4_VALID": 4_440,
    "ERC_MIGRATION_REDIRECT_LOWER_PATH": 365,
    "ERC_MIGRATION_REDIRECT_UPPER_PATH": 365,
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
    return candidate if candidate.is_absolute() else d4.REPO_ROOT / candidate


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
    return d4.sha256_bytes(
        canonical_json_bytes(payload, pretty=False).rstrip(b"\n")
    )


def _exact_header_lines(raw: bytes) -> tuple[str, ...]:
    lines = d4.core.prereg.normalize_blob_bytes(raw)
    if not lines or lines[0] != "---":
        return ()
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line == "---"
        ),
        None,
    )
    if closing_index is None:
        return ()
    return tuple(lines[1:closing_index])


def _field_occurrences(
    header_lines: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = defaultdict(list)
    for line in header_lines:
        if not line or line.lstrip(" \t").startswith("#"):
            continue
        match = FIELD_PATTERN.fullmatch(line)
        if match is None:
            continue
        key = d4.core.prereg.normalize_header_key(match.group(1))
        values[key].append(match.group(2))
    return {
        key: tuple(rows)
        for key, rows in sorted(values.items())
    }


def _malformed_header_lines(
    header_lines: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        line
        for line in header_lines
        if line
        and not line.lstrip(" \t").startswith("#")
        and not line.startswith((" ", "\t"))
        and FIELD_PATTERN.fullmatch(line) is None
    )


def _redirect_detail(
    raw: bytes,
    proposal_number: int,
) -> dict[str, Any] | None:
    lines = d4.core.prereg.normalize_blob_bytes(raw)
    if len(lines) not in {1, 2} or (len(lines) == 2 and lines[1] != ""):
        return None
    match = REDIRECT_PATTERN.fullmatch(lines[0])
    if match is None:
        return None
    target = int(match.group(2), 10)
    return {
        "path_case": match.group(1),
        "target_proposal": target,
        "target_matches_path_proposal": target == proposal_number,
    }


def classify_blob(
    proposal_number: int,
    blob_oid: str,
    raw: bytes,
) -> tuple[str, dict[str, Any]]:
    """Classify one already hydrated EIP blob without repairing it."""

    try:
        d4.parse_blob_features(
            "ethereum",
            proposal_number,
            blob_oid,
            raw,
        )
    except (UnicodeDecodeError, ValueError) as error:
        message = str(error)
        detail: dict[str, Any] = {
            "d4_error": message,
            "error_type": type(error).__name__,
        }
        if message == "PSIM EIP opening fence is not exact":
            redirect = _redirect_detail(raw, proposal_number)
            if redirect is None:
                return "D4_OPENING_FENCE_FAILURE_OTHER", detail
            detail["redirect"] = redirect
            if redirect["target_matches_path_proposal"] is not True:
                return "D4_REDIRECT_TARGET_MISMATCH", detail
            path_case = redirect["path_case"]
            return (
                (
                    "ERC_MIGRATION_REDIRECT_LOWER_PATH"
                    if path_case == "ercs"
                    else "ERC_MIGRATION_REDIRECT_UPPER_PATH"
                ),
                detail,
            )
        header_lines = _exact_header_lines(raw)
        if message == "PSIM duplicate normalized header key":
            duplicates = {
                key: list(values)
                for key, values in _field_occurrences(
                    header_lines
                ).items()
                if len(values) > 1
            }
            detail["duplicate_fields"] = duplicates
            if duplicates and all(
                len(set(values)) == 1
                for values in duplicates.values()
            ):
                return "D4_DUPLICATE_IDENTICAL_HEADER", detail
            return "D4_DUPLICATE_CONFLICTING_HEADER", detail
        if message == "PSIM malformed header line":
            detail["malformed_lines"] = list(
                _malformed_header_lines(header_lines)
            )
            return "D4_MALFORMED_HEADER_LINE", detail
        if message == "PSIM self dependency":
            fields = d4.parse_eip_preamble(raw)
            detail["requires"] = fields.get("requires")
            return "D4_SELF_DEPENDENCY", detail
        return "D4_OTHER_FAILURE", detail
    return "D4_VALID", {}


def _metadata(
    group: d4.ProposalGroup,
    *,
    side: str,
    path: str | None,
    blob_oid: str,
    raw: bytes,
) -> dict[str, Any]:
    return {
        "blob_oid": blob_oid,
        "commit_oid": group.commit_oid,
        "effective_day": group.effective_day.isoformat(),
        "path": path,
        "proposal": group.proposal_number,
        "protocol": group.protocol,
        "raw_sha256": d4.sha256_bytes(raw),
        "side": side,
    }


def _assert_terminal_boundary() -> Mapping[str, Any]:
    terminal_path = repository_path(d4.DEFAULT_REJECTION_PATH)
    if (
        terminal_path.is_symlink()
        or not terminal_path.is_file()
        or d4.sha256_file(terminal_path) != TERMINAL_SHA256
    ):
        raise RuntimeError("PSIM-D4 terminal artifact changed")
    payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        payload.get("result_hash") != TERMINAL_RESULT_HASH
        or payload.get("decision") != "reject"
        or payload.get("first_failure")
        != {
            "gate_id": 4,
            "name": "historical_blob_preamble_dependency_integrity",
        }
        or payload.get("outcomes_opened") is not False
        or any(
            payload.get("access_ledger", {}).get(name) != 0
            for name in d4.FORBIDDEN_ACCESS_FIELDS
        )
    ):
        raise RuntimeError("PSIM-D4 terminal boundary changed")
    return payload


def _assert_forensic_root() -> Path:
    if (
        ETHEREUM_ROOT.is_symlink()
        or not ETHEREUM_ROOT.is_dir()
        or ETHEREUM_ROOT.resolve() != ETHEREUM_ROOT
    ):
        raise RuntimeError("PSIM-D4 forensic root is absent or unsafe")
    return ETHEREUM_ROOT


def _assert_local_git_arguments(arguments: Sequence[str]) -> None:
    """Fail closed unless argv names one census-approved local Git reader."""

    values = tuple(arguments)
    if not values:
        raise RuntimeError("PSIM-D4 grammar census forbids empty Git argv")
    index = 0
    while index < len(values) and values[index] == "-C":
        if index + 1 >= len(values) or not values[index + 1]:
            raise RuntimeError(
                "PSIM-D4 grammar census forbids malformed Git -C"
            )
        index += 2
    if index >= len(values) or values[index].startswith("-"):
        raise RuntimeError(
            "PSIM-D4 grammar census forbids unrecognized Git global options"
        )
    if values[index] not in LOCAL_GIT_SUBCOMMANDS:
        raise RuntimeError(
            "PSIM-D4 grammar census forbids nonlocal Git subcommand"
        )


def build_census() -> dict[str, Any]:
    """Read the terminal D4 root once, locally, and classify every EIP blob."""

    terminal = _assert_terminal_boundary()
    repo = _assert_forensic_root()
    original_run_git = d4._run_git

    def no_network_git(
        arguments: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        _assert_local_git_arguments(arguments)
        if kwargs.get("network"):
            raise RuntimeError("PSIM-D4 grammar census forbids network Git")
        return original_run_git(arguments, **kwargs)

    d4._run_git = no_network_git
    ledger = d4.AccessLedger.zero()
    try:
        before = d4._object_store_snapshot(repo, ledger)
        if canonical_hash(before) != FORENSIC_OBJECT_STORE_HASH:
            raise RuntimeError("PSIM-D4 forensic object store changed")
        records = d4.collect_commit_chain(
            repo,
            "ethereum",
            ledger,
        )
        groups, issues = d4.collect_proposal_groups(
            repo,
            records,
            ledger,
        )
        if issues:
            raise RuntimeError("PSIM-D4 forensic proposal groups changed")
        if (
            len(records) != ETHEREUM_CHAIN_ROWS
            or d4.canonical_hash(d4._commit_rows(records))
            != ETHEREUM_CHAIN_HASH
            or len(groups) != ETHEREUM_GROUP_ROWS
            or d4.canonical_hash(d4._group_rows(groups))
            != ETHEREUM_GROUP_HASH
        ):
            raise RuntimeError("PSIM-D4 forensic chain or groups changed")
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
            or d4.sha256_bytes(manifest) != OID_MANIFEST_SHA256
        ):
            raise RuntimeError("PSIM-D4 forensic OID manifest changed")
        raw_by_oid = dict(
            d4._cat_file_batch_local(
                repo,
                object_ids,
                expected_type="blob",
                ledger=ledger,
            )
        )
        after = d4._object_store_snapshot(repo, ledger)
        if before != after:
            raise RuntimeError("PSIM-D4 census mutated the object store")
    finally:
        d4._run_git = original_run_git

    class_counts: Counter[str] = Counter()
    class_days: dict[str, Counter[str]] = defaultdict(Counter)
    class_proposals: dict[str, set[int]] = defaultdict(set)
    representatives: dict[str, dict[str, Any]] = {}
    detail_profiles: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, int, str]] = set()
    for group in groups:
        for side, blob_oid, path in (
            ("old", group.old_blob_oid, group.old_path),
            ("new", group.new_blob_oid, group.new_path),
        ):
            if blob_oid is None:
                continue
            key = (group.protocol, group.proposal_number, blob_oid)
            if key in seen:
                continue
            seen.add(key)
            raw = raw_by_oid[blob_oid]
            classification, detail = classify_blob(
                group.proposal_number,
                blob_oid,
                raw,
            )
            class_counts[classification] += 1
            class_days[classification][group.effective_day.isoformat()] += 1
            class_proposals[classification].add(group.proposal_number)
            representatives.setdefault(
                classification,
                _metadata(
                    group,
                    side=side,
                    path=path,
                    blob_oid=blob_oid,
                    raw=raw,
                ),
            )
            if classification == "D4_DUPLICATE_IDENTICAL_HEADER":
                profile = json.dumps(
                    detail["duplicate_fields"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                detail_profiles[classification][profile] += 1
            elif classification == "D4_MALFORMED_HEADER_LINE":
                profile = json.dumps(
                    detail["malformed_lines"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                detail_profiles[classification][profile] += 1
            elif classification == "D4_SELF_DEPENDENCY":
                detail_profiles[classification][str(detail["requires"])] += 1

    observed_counts = dict(sorted(class_counts.items()))
    if observed_counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError("PSIM-D4 grammar census class roster changed")
    migration_count = sum(
        observed_counts[name]
        for name in (
            "ERC_MIGRATION_REDIRECT_LOWER_PATH",
            "ERC_MIGRATION_REDIRECT_UPPER_PATH",
        )
    )
    nonmigration_failures = (
        REQUESTED_BLOBS
        - observed_counts["D4_VALID"]
        - migration_count
    )
    migration_classes = (
        "ERC_MIGRATION_REDIRECT_LOWER_PATH",
        "ERC_MIGRATION_REDIRECT_UPPER_PATH",
    )
    migration_proposals = sorted(
        set().union(
            *(class_proposals[name] for name in migration_classes)
        )
    )
    anomaly_classes = (
        "D4_DUPLICATE_IDENTICAL_HEADER",
        "D4_MALFORMED_HEADER_LINE",
        "D4_SELF_DEPENDENCY",
    )
    payload: dict[str, Any] = {
        "access_boundary": {
            "d4_forensic_root_read": True,
            "d4_run_invoked": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "network_commands": ledger.network_commands,
            "outcomes_accessed": False,
            "source_objects_mutated": False,
        },
        "candidate_selection": {
            "authorized": False,
            "next_step": (
                "OUTCOME_BLIND_D5_DESIGN_REVIEW_BEFORE_PREREGISTRATION"
            ),
            "sequential_first_exception_patching_rejected": True,
        },
        "census": {
            "class_counts": observed_counts,
            "class_effective_day_counts": {
                name: dict(sorted(class_days[name].items()))
                for name in (
                    *anomaly_classes,
                    *migration_classes,
                )
            },
            "class_proposal_counts": {
                name: len(values)
                for name, values in sorted(class_proposals.items())
            },
            "d4_strict_success_fraction": (
                observed_counts["D4_VALID"] / REQUESTED_BLOBS
            ),
            "detail_profiles": {
                name: dict(sorted(values.items()))
                for name, values in sorted(detail_profiles.items())
            },
            "migration_redirect_blobs": migration_count,
            "migration_redirect_fraction_of_failures": (
                migration_count
                / (REQUESTED_BLOBS - observed_counts["D4_VALID"])
            ),
            "migration_redirect_proposal_roster_hash": canonical_hash(
                migration_proposals
            ),
            "nonmigration_failure_proposals": {
                name: sorted(class_proposals[name])
                for name in anomaly_classes
            },
            "nonmigration_failure_blobs": nonmigration_failures,
            "representatives": dict(sorted(representatives.items())),
            "unique_blob_contexts": len(seen),
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
            "oid_manifest_sha256": d4.sha256_bytes(manifest),
            "requested_blobs": len(object_ids),
            "root": str(repo),
        },
        "policy_id": "PSIM-D4-POST-TERMINAL-GRAMMAR-CENSUS",
        "protocol_version": PROTOCOL_VERSION,
        "research_implication": {
            "administrative_redirects_are_not_specification_intent": True,
            "dependency_header_semantics_are_not_total_over_history": True,
            "preferred_next_mechanism": (
                "PATH_IDENTITY_TEXT_DIFF_WITH_EXACT_ADMINISTRATIVE_"
                "MIGRATION_QUARANTINE_AND_EXPLICIT_INVALID_METADATA_STATE"
            ),
            "strict_d4_header_parser_is_not_a_total_historical_decoder": True,
        },
        "terminal_binding": {
            "commit": TERMINAL_COMMIT,
            "path": d4.DEFAULT_REJECTION_PATH.as_posix(),
            "result_hash": terminal["result_hash"],
            "sha256": TERMINAL_SHA256,
        },
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def write_census(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_census()
    target = repository_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("existing PSIM-D4 grammar census differs")
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
