"""Freeze BCIMS source support before 2020-2023 incidence or outcomes open."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/bitcoin_core_immutable_merge_surface_source_protocol_2026-07-22.json"
)
DECISION_PATH = Path(
    "docs/bitcoin-core-immutable-merge-surface-source-axis-decision-2026-07-22.md"
)
DECISION_SHA256 = "af18cdb1c7184d3f603efbf62e6242212db08f97a41faebde7ba733ecc3761fa"
SCRIPT_PATH = Path("training/preregister_bitcoin_core_immutable_merge_surface.py")
OFFICIAL_REMOTE = "https://github.com/bitcoin/bitcoin.git"
OFFICIAL_BRANCH = "master"
PROBE_SEALED_TIP = "bc49bd154a31b285c0f89be51767a424ac380924"
SOURCE_START = "2020-01-01"
SOURCE_END_EXCLUSIVE = "2024-01-01"
DISK_LIMIT_GIB = 300
UNICODE_DATABASE_VERSION = unicodedata.unidata_version

MERGE_SUBJECT_PATTERN = re.compile(
    r"\AMerge (?P<repo>bitcoin/bitcoin|bitcoin-core/gui)"
    r"#(?P<pr_number>[1-9][0-9]*): (?P<title>\S(?:.*\S)?)\Z",
    re.ASCII,
)
CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return sha256_bytes(candidate.read_bytes())


def canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def parse_merge_subject(subject: str) -> dict[str, Any] | None:
    """Route an exact immutable merge subject without semantic inference."""

    if "\n" in subject or "\r" in subject or CONTROL_PATTERN.search(subject):
        raise ValueError("BCIMS merge subject contains a control character")
    match = MERGE_SUBJECT_PATTERN.fullmatch(subject)
    if match is None:
        return None
    repository = match.group("repo")
    return {
        "repository": repository,
        "pr_number": int(match.group("pr_number")),
        "title": match.group("title"),
        "stratum": (
            "primary_core"
            if repository == "bitcoin/bitcoin"
            else "gui_comparator"
        ),
    }


def path_surface(path: str) -> str:
    """Return the exact top-level Git path surface under the frozen contract."""

    path.encode("utf-8", errors="strict")
    if not path or path.startswith("/") or path.endswith("/"):
        raise ValueError("BCIMS path is empty or absolute/directory-shaped")
    if "\\" in path or CONTROL_PATTERN.search(path):
        raise ValueError("BCIMS path contains a forbidden byte shape")
    components = path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("BCIMS path contains an unsafe component")
    return components[0] if len(components) > 1 else "__root__"


def causal_availability_floors(committer_times: Sequence[str]) -> list[str]:
    """Map oldest-first committer times to conservative monotone UTC floors."""

    running_day: date | None = None
    floors: list[str] = []
    for value in committer_times:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("BCIMS committer time is malformed") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("BCIMS committer time must be timezone-aware")
        commit_day = parsed.astimezone(timezone.utc).date()
        running_day = commit_day if running_day is None else max(running_day, commit_day)
        floor_day = running_day + timedelta(days=2)
        floor = datetime.combine(floor_day, time(12), tzinfo=timezone.utc)
        floors.append(floor.isoformat().replace("+00:00", "Z"))
    return floors


def _source_contract() -> dict[str, Any]:
    return {
        "authority": {
            "remote": OFFICIAL_REMOTE,
            "branch": OFFICIAL_BRANCH,
            "sealed_tip": PROBE_SEALED_TIP,
            "default_branch_symref_must_equal": "refs/heads/master",
            "object_format": "sha1",
            "clone_arguments": [
                "--single-branch",
                "--branch=master",
                "--filter=blob:none",
                "--no-checkout",
            ],
            "mutable_github_pr_metadata_used": False,
            "blob_contents_opened_in_source_support": False,
        },
        "interval": {
            "start_inclusive": SOURCE_START,
            "end_exclusive": SOURCE_END_EXCLUSIVE,
            "membership_clock": "causal_availability_utc",
            "probe_interval_excluded": "[2024-01-01, sealed probe tip]",
        },
        "traversal": {
            "command_semantics": "sealed tip first-parent chain, reverse chronological order reversed to oldest-first",
            "every_first_parent_commit_retained_for_audit": True,
            "primary_parent_count": 2,
            "path_delta_parent": 1,
            "rename_detection": False,
            "path_encoding": "UTF-8 strict",
            "path_surface": "first component; root file maps to __root__",
        },
        "membership": {
            "subject_fullmatch": MERGE_SUBJECT_PATTERN.pattern,
            "primary_repository": "bitcoin/bitcoin",
            "comparator_repository": "bitcoin-core/gui",
            "other_form": "audit_only",
            "llm_can_change_membership": False,
            "duplicate_pr_number_within_stratum": "fatal",
        },
        "retained_fields": [
            "raw commit object bytes",
            "commit/tree/parent hashes",
            "author and committer timestamps normalized to UTC",
            "full immutable commit message",
            "exact merge-subject captures",
            "NUL-safe no-renames path delta against parent one",
            "raw extraction SHA-256 values",
        ],
        "forbidden_fields": [
            "current PR title/body/labels/milestone/state",
            "reactions/reviews/contributor profile",
            "market bars/returns/funding/PnL",
            "future repository state outside the sealed tip",
        ],
    }


def _quality_contract() -> dict[str, Any]:
    return {
        "integrity": {
            "git_fsck_connectivity_required": True,
            "sealed_tip_must_be_reachable_from_fetched_master": True,
            "first_parent_object_reconciliation_fraction": 1.0,
            "raw_commit_and_path_delta_hash_fraction": 1.0,
            "primary_and_comparator_parent_count": 2,
            "primary_nonempty_path_delta_fraction": 1.0,
            "unknown_first_parent_fraction_max": 0.05,
            "unknown_first_parent_fraction_max_each_year": 0.10,
            "quarantine_or_imputation_allowed": False,
        },
        "primary_core": {
            "minimum_events": 2400,
            "minimum_events_each_year": 500,
            "minimum_events_each_quarter": 100,
            "minimum_unique_availability_days_each_year": 180,
            "maximum_calendar_month_share": 0.12,
            "minimum_distinct_top_level_surfaces_each_year": 6,
            "maximum_fractional_top_level_surface_share": 0.70,
        },
        "gui_comparator": {
            "minimum_events": 30,
            "minimum_events_each_year": 5,
            "minimum_unique_availability_days": 25,
        },
        "fractional_surface_attribution": (
            "an event touching k unique top-level surfaces contributes 1/k to each"
        ),
        "failure_effect": "REJECT_NO_REPAIR",
    }


def build_manifest() -> dict[str, Any]:
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("BCIMS source-axis decision hash differs from the freeze")
    core: dict[str, Any] = {
        "protocol_version": "bitcoin_core_immutable_merge_surface_source_v1",
        "source_id": "BCIMS",
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "historical_source_incidence_opened": False,
        "semantic_model_opened": False,
        "source_only_probe_opened": True,
        "source_only_probe": {
            "interval": "[2024-01-01, sealed probe tip]",
            "excluded_from_source_support": True,
            "sealed_tip": PROBE_SEALED_TIP,
            "git_version": "2.43.0",
            "filtered_clone_size_mib": 74,
            "first_parent_commits": 3016,
            "two_parent_merges": 3016,
            "bitcoin_subjects": 2950,
            "gui_subjects": 66,
            "other_two_parent_subjects": 0,
            "non_two_parent_commits": 0,
            "unique_utc_committer_days": 697,
            "descending_committer_time_violations": 0,
            "market_or_outcomes_opened": False,
        },
        "decision_binding": {
            "path": str(DECISION_PATH),
            "sha256": DECISION_SHA256,
        },
        "implementation_binding": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "unicode_database_version": UNICODE_DATABASE_VERSION,
        "source_contract": _source_contract(),
        "availability_contract": {
            "historical": (
                "12:00 UTC on second calendar day after running-max UTC committer day"
            ),
            "git_committer_time_claimed_as_server_receipt": False,
            "live": (
                "max(historical floor, durable fetch + object verification + "
                "extraction + hash + manifest commit)"
            ),
            "force_push_or_unreachable_sealed_ancestor": "halt",
        },
        "source_quality_gates": _quality_contract(),
        "disk_contract": {
            "abort_before_fetch_when_used_gib_at_least": DISK_LIMIT_GIB,
            "raw_blob_download_authorized": False,
            "deterministic_gzip_mtime": 0,
        },
        "later_semantic_boundary": {
            "authorized_now": False,
            "eligible_inputs_after_source_pass": [
                "full immutable merge message",
                "exact changed-path list",
                "separately frozen immutable diff text",
            ],
            "mandatory_baselines": [
                "path-surface-only",
                "component-prefix-keyword-only",
                "cadence-only",
            ],
            "requirements": [
                "single local LLM",
                "evidence-grounded label with abstention",
                "model and labels frozen before returns open",
                "train-only adapter or RLLM selection",
            ],
            "forbidden": [
                "LLM event creation/deletion/retiming",
                "mutable GitHub metadata as historical model input",
                "eval reward or checkpoint selection",
                "analyzer/trader two-model split",
            ],
        },
        "later_clock_novelty_boundary": {
            "authorized_now": False,
            "applies_to": "later frozen semantic event clock, not dense source rows",
            "exact_entry_jaccard_max": 0.20,
            "tolerant_window_hours": 24,
            "tolerant_one_to_one_jaccard_max": 0.35,
            "primary_containment_max": 0.50,
            "all_prior_live_and_rejected_family_comparators_required": True,
        },
        "next_stage_artifacts": {
            "source_rows": (
                "data/bitcoin_core_immutable_merge_surface_2020_2023.jsonl.gz"
            ),
            "source_manifest": (
                "results/bitcoin_core_immutable_merge_surface_source_manifest_2026-07-22.json"
            ),
            "source_support_result": (
                "results/bitcoin_core_immutable_merge_surface_source_support_2026-07-22.json"
            ),
            "write_once": True,
        },
        "rejection_contract": (
            "any remote, object, traversal, parser, timestamp, path, disk, annual, "
            "quarterly, concentration, comparator, replay, or live-parity failure "
            "retires BCIMS without changing the interval, delay, membership regex, "
            "thresholds, or strata"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("BCIMS source protocol hash mismatch")
    for field in (
        "outcomes_opened",
        "market_clocks_opened",
        "historical_source_incidence_opened",
        "semantic_model_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"BCIMS source protocol must keep {field}=false")
    if payload.get("source_only_probe_opened") is not True:
        raise RuntimeError("BCIMS source protocol must disclose its parser probe")
    expected = build_manifest()
    expected_core = {
        key: value for key, value in expected.items() if key != "manifest_hash"
    }
    if core != expected_core:
        raise RuntimeError("BCIMS frozen source contract differs from code")


def write_manifest_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    validate_manifest(payload)
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise RuntimeError("BCIMS existing source protocol is not an object")
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen BCIMS source protocol")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = build_manifest()
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "source_id": payload["source_id"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": False,
                "historical_source_incidence_opened": False,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
