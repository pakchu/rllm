"""Freeze PSIM-D1 source support before 2020-2023 proposal incidence.

This module is deliberately source-blind.  It contains only sealed official
Git identities, immutable path/object grammar, causal archive clocks,
historical-blob parser rules, deterministic relation-card construction,
support gates, controls, and forbidden-access counters.  It must not clone a
repository, read a proposal blob, call a network, open market/funding data,
load a model, build a reward, or construct a trade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_protocol_specification_intent_maturity.py"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_preregistration_"
    "2026-07-25.json"
)
DECISION_PATH = Path(
    "docs/post-gipr-d1-alpha-mechanism-audit-2026-07-25.md"
)
DECISION_SHA256 = (
    "816fbb19c4ff9a841f75f75555e568f401e804b4aded258779ef4bce14ebaf04"
)
SELECTION_COMMIT = "6ebb43406f7197e2afb2e2fa5cb39b0a2cba2826"

SOURCE_START = "2020-01-01T00:00:00Z"
SOURCE_END_EXCLUSIVE = "2024-01-01T00:00:00Z"
CARD_END_EXCLUSIVE = "2024-04-01T00:00:00Z"
DECISION_FREQUENCY = "1D"
DECISION_UTC_TIME = "12:05:00"
COUNTERPART_LOOKBACK_DAYS = 90
DISK_LIMIT_GIB = 300

MAX_BLOB_BYTES = 2_097_152
MAX_HEADER_BYTES = 131_072
MAX_HEADER_LINES = 256
MAX_LINE_BYTES = 65_536
MAX_SECTIONS = 128
MAX_DEPENDENCIES = 128
MAX_MODEL_TEXT_BYTES_PER_EVENT = 8_192
MAX_MODEL_EVENTS_PER_CARD = 64

SHA1_PATTERN = re.compile(r"[0-9a-f]{40}", re.ASCII)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
HEADER_KEY_PATTERN = r"^[A-Za-z][A-Za-z0-9 -]*$"
EIP_PATH_PATTERN = r"^EIPS/eip-([1-9][0-9]*)\.md$"
BIP_PATH_PATTERN = r"^bip-([0-9]{4})\.(mediawiki|md)$"

OFFICIAL_SOURCES = (
    "https://eips.ethereum.org/EIPS/eip-1",
    "https://github.com/ethereum/EIPs",
    "https://github.com/ethereum/EIPs.git",
    "https://github.com/bitcoin/bips",
    "https://github.com/bitcoin/bips.git",
    "https://github.com/bitcoin/bips/blob/master/bip-0003.md",
    "https://git-scm.com/docs/git-log",
    "https://git-scm.com/book/en/v2/Git-Internals-Git-Objects",
)


@dataclass(frozen=True)
class RepositorySpec:
    protocol: str
    remote: str
    branch: str
    remote_head_symref: str
    sealed_tip: str
    object_format: str
    path_pattern: str
    document_formats: tuple[str, ...]


REPOSITORIES = (
    RepositorySpec(
        protocol="ethereum",
        remote="https://github.com/ethereum/EIPs.git",
        branch="master",
        remote_head_symref="refs/remotes/origin/master",
        sealed_tip="5e82ef62895121027a6c5f0c23276e1b2bed3071",
        object_format="sha1",
        path_pattern=EIP_PATH_PATTERN,
        document_formats=("markdown_frontmatter",),
    ),
    RepositorySpec(
        protocol="bitcoin",
        remote="https://github.com/bitcoin/bips.git",
        branch="master",
        remote_head_symref="refs/remotes/origin/master",
        sealed_tip="b289d016b99c81527623c10e995e0318f744ebf3",
        object_format="sha1",
        path_pattern=BIP_PATH_PATTERN,
        document_formats=("mediawiki_rfc822", "markdown_rfc822"),
    ),
)


@dataclass(frozen=True)
class ArchiveSchedule:
    name: str
    delay_calendar_days: int
    primary_economic_clock: bool
    profitability_claim_allowed: bool


ARCHIVE_SCHEDULES = (
    ArchiveSchedule("ARCHIVE_D2", 2, False, False),
    ArchiveSchedule("ARCHIVE_D7", 7, False, False),
    ArchiveSchedule("ARCHIVE_D30", 30, False, False),
    ArchiveSchedule("ARCHIVE_D90", 90, True, True),
)


SPLITS = (
    {
        "name": "train",
        "decision_start": "2020-01-01T00:00:00Z",
        "decision_end_exclusive": "2022-01-01T00:00:00Z",
        "minimum_events_total": 80,
        "minimum_events_per_protocol": 20,
        "minimum_events_per_protocol_per_source_year": 6,
        "minimum_unique_proposals_total": 24,
        "minimum_unique_proposals_per_protocol": 6,
        "minimum_unique_event_days_per_protocol": 12,
        "minimum_active_months_per_protocol": 8,
        "minimum_active_quarters_per_protocol": 6,
        "minimum_relation_units_nonexcluded": 40,
        "minimum_counterpart_fraction": "0.35",
        "maximum_top_proposal_event_share": "0.35",
        "maximum_top_event_day_share": "0.25",
    },
    {
        "name": "test",
        "decision_start": "2022-01-01T00:00:00Z",
        "decision_end_exclusive": "2023-01-01T00:00:00Z",
        "minimum_events_total": 30,
        "minimum_events_per_protocol": 8,
        "minimum_events_per_protocol_per_source_year": 6,
        "minimum_unique_proposals_total": 10,
        "minimum_unique_proposals_per_protocol": 3,
        "minimum_unique_event_days_per_protocol": 6,
        "minimum_active_months_per_protocol": 4,
        "minimum_active_quarters_per_protocol": 3,
        "minimum_relation_units_nonexcluded": 12,
        "minimum_counterpart_fraction": "0.25",
        "maximum_top_proposal_event_share": "0.50",
        "maximum_top_event_day_share": "0.35",
    },
    {
        "name": "eval",
        "decision_start": "2023-01-01T00:00:00Z",
        "decision_end_exclusive": "2024-01-01T00:00:00Z",
        "minimum_events_total": 30,
        "minimum_events_per_protocol": 8,
        "minimum_events_per_protocol_per_source_year": 6,
        "minimum_unique_proposals_total": 10,
        "minimum_unique_proposals_per_protocol": 3,
        "minimum_unique_event_days_per_protocol": 6,
        "minimum_active_months_per_protocol": 4,
        "minimum_active_quarters_per_protocol": 3,
        "minimum_relation_units_nonexcluded": 12,
        "minimum_counterpart_fraction": "0.25",
        "maximum_top_proposal_event_share": "0.50",
        "maximum_top_event_day_share": "0.35",
    },
)

BUCKET_EDGES = {
    "window_revision_count": (0, 1, 2, 4, 8, 16, 32),
    "window_age_days": (0, 7, 30, 90, 180, 365, 730),
    "update_gap_days": (0, 2, 7, 30, 90, 180, 365),
    "stale_age_days": (0, 2, 7, 30, 90, 180),
    "line_change_count": (0, 1, 3, 8, 20, 50, 100, 250, 500),
    "changed_section_count": (0, 1, 2, 4, 8, 16),
    "dependency_edge_delta_count": (0, 1, 2, 4, 8, 16),
}

SECTION_ALIASES = {
    "ABSTRACT": ("abstract",),
    "MOTIVATION": ("motivation",),
    "SPECIFICATION": ("spec", "specification", "specifications"),
    "RATIONALE": ("rationale",),
    "BACKWARD_COMPATIBILITY": (
        "backward compatibility",
        "backwards compatibility",
    ),
    "SECURITY": ("security consideration", "security considerations"),
    "TESTS": ("test case", "test cases", "test vector", "test vectors"),
    "IMPLEMENTATION": (
        "implementation",
        "reference implementation",
        "reference implementations",
    ),
    "COPYRIGHT": ("copyright", "copyright waiver"),
    "OTHER": (),
}

EIP_DEPENDENCY_FIELDS = ("requires",)
BIP_DEPENDENCY_FIELDS = ("requires", "replaces", "proposed-replacement")
EVENT_TYPES = ("CREATE", "UPDATE", "DELETE")
BOUNDARY_STATES = (
    "PRE_WINDOW",
    "PRE_WINDOW_BASELINE",
    "PRE_WINDOW_UNKNOWN",
    "NO_EVENT_YET",
)
DEPENDENCY_DELTA_STATES = (
    "NO_PRIOR",
    "STABLE",
    "ADDED",
    "REMOVED",
    "MIXED",
    "DELETED",
)

RELATION_TOKENS = (
    "CONVERGENT_INTENT",
    "COMPLEMENTARY_INTENT",
    "TECHNICAL_TENSION",
    "INDEPENDENT_INTENT",
    "INSUFFICIENT_EVIDENCE",
    "ABSTAIN",
)
MODEL_ACTIONS = ("TARGET_LONG", "TARGET_FLAT", "TARGET_SHORT")

MEMORIZATION_QUARANTINE = {
    "ethereum": (20, 721, 1559, 3675, 4337, 4844, 4895),
    "bitcoin": (32, 39, 44, 141, 340, 341, 342),
}

SOURCE_ONLY_GATES = (
    "sealed_git_identity_and_object_integrity",
    "first_parent_traversal_and_causal_clock",
    "path_object_grammar_and_unique_proposal_tree",
    "historical_blob_preamble_dependency_integrity",
    "split_annual_quarterly_unique_day_support",
    "event_section_dependency_revision_vocabulary_diversity",
    "daily_card_coverage_and_explicit_staleness",
    "independent_replay_and_canonical_manifest_identity",
    "future_append_invariance",
    "relation_control_sensitivity",
    "pairing_reset_quarantine_and_four_schedule_identity",
    "forbidden_access_zero",
    "terminal_publication",
)

RELATION_CONTROLS = (
    "protocol_label_swap",
    "within_day_event_order_reverse",
    "proposal_version_pair_cyclic_permutation",
    "old_new_direction_reverse",
    "section_label_cyclic_rotation",
    "dependency_edge_direction_reverse",
    "availability_plus_seven_days",
)

CONTROL_TRANSFORMS = {
    "protocol_label_swap": (
        "swap ethereum<->bitcoin labels after extraction; preserve raw event "
        "identity and text; rebuild relation cards"
    ),
    "within_day_event_order_reverse": (
        "reverse each protocol/day canonical event sequence before pairing; "
        "preserve event payloads"
    ),
    "proposal_version_pair_cyclic_permutation": (
        "within each protocol/source_year, sort eligible UPDATE events by "
        "event_id and rotate old-blob-derived fields forward by one while "
        "holding new-blob-derived fields fixed"
    ),
    "old_new_direction_reverse": (
        "swap all old/new derived fields; map CREATE<->DELETE and retain "
        "UPDATE; rebuild dependency and section deltas"
    ),
    "section_label_cyclic_rotation": (
        "rotate changed-section labels over ABSTRACT,MOTIVATION,SPECIFICATION,"
        "RATIONALE,BACKWARD_COMPATIBILITY,SECURITY,TESTS,IMPLEMENTATION,"
        "COPYRIGHT; keep OTHER fixed"
    ),
    "dependency_edge_direction_reverse": (
        "swap ADDED<->REMOVED; keep NO_PRIOR,STABLE,MIXED,DELETED fixed"
    ),
    "availability_plus_seven_days": (
        "add exactly seven calendar days to every available_at under each "
        "archive schedule and rebuild every card"
    ),
}

CONTROL_ELIGIBILITY = {
    "protocol_label_swap": (
        "decision day has at least one non-sentinel event or counterpart"
    ),
    "within_day_event_order_reverse": (
        "either protocol has at least two newly available events that day"
    ),
    "proposal_version_pair_cyclic_permutation": (
        "card contains an UPDATE from a protocol/source_year stratum with at "
        "least two eligible UPDATE events"
    ),
    "old_new_direction_reverse": "card contains at least one source event",
    "section_label_cyclic_rotation": (
        "card contains at least one changed section other than OTHER"
    ),
    "dependency_edge_direction_reverse": (
        "card contains at least one ADDED or REMOVED dependency delta"
    ),
    "availability_plus_seven_days": (
        "baseline or shifted card day contains at least one newly available "
        "event"
    ),
}

FORBIDDEN_COUNTERS = (
    "pre_2020_proposal_blobs_read",
    "post_2023_proposal_blobs_read",
    "source_incidence_rows_read",
    "proposal_text_rows_read",
    "dependency_rows_read",
    "daily_cards_built",
    "btc_market_rows_read",
    "funding_rows_read",
    "future_return_rows_read",
    "reward_rows_built",
    "model_rows_built",
    "model_outputs_built",
    "trade_rows_built",
    "pnl_rows_built",
    "cagr_values_built",
    "strict_mdd_values_built",
)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(_repo_path(path).read_bytes())


def canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def normalize_blob_bytes(raw: bytes) -> list[str]:
    if len(raw) > MAX_BLOB_BYTES:
        raise ValueError("PSIM blob exceeds maximum bytes")
    text = raw.decode("utf-8", errors="strict")
    if "\x00" in text:
        raise ValueError("PSIM blob contains NUL")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    for line in lines:
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise ValueError("PSIM line exceeds maximum bytes")
    return lines


def normalize_header_key(value: str) -> str:
    key = " ".join(value.strip().split()).casefold()
    if not key or re.fullmatch(HEADER_KEY_PATTERN, key, re.ASCII) is None:
        raise ValueError("PSIM header key is malformed")
    return key


def _parse_header_lines(
    lines: Sequence[str],
    *,
    field_lines_may_be_indented: bool,
    allow_empty_values: bool,
    comment_styles: tuple[str, ...],
) -> dict[str, str]:
    if not lines or len(lines) > MAX_HEADER_LINES:
        raise ValueError("PSIM header line count is invalid")
    if len(("\n".join(lines) + "\n").encode("utf-8")) > MAX_HEADER_BYTES:
        raise ValueError("PSIM header exceeds maximum bytes")
    fields: dict[str, str] = {}
    current_key: str | None = None
    for line in lines:
        if not line:
            raise ValueError("PSIM blank line inside header")
        stripped = line.lstrip(" \t")
        if "hash" in comment_styles and stripped.startswith("#"):
            continue
        if (
            "html" in comment_styles
            and stripped.startswith("<!--")
            and stripped.endswith("-->")
        ):
            continue
        candidate = stripped if field_lines_may_be_indented else line
        field_match = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9 -]*):[ \t]*(.*)",
            candidate,
            re.ASCII,
        )
        if field_match is not None:
            key = normalize_header_key(field_match.group(1))
            if key in fields:
                raise ValueError("PSIM duplicate normalized header key")
            fields[key] = field_match.group(2)
            current_key = key
            continue
        if line.startswith((" ", "\t")):
            if current_key is None or not stripped:
                raise ValueError("PSIM orphan or empty header continuation")
            prior = fields[current_key]
            fields[current_key] = f"{prior}\n{stripped}" if prior else stripped
            continue
        raise ValueError("PSIM malformed header line")
    if not fields:
        raise ValueError("PSIM header contains no fields")
    if not allow_empty_values and any(not value for value in fields.values()):
        raise ValueError("PSIM empty header value")
    return fields


def parse_positive_proposal_number(value: str) -> int:
    if re.fullmatch(r"[0-9]+", value, re.ASCII) is None:
        raise ValueError("PSIM proposal number is not ASCII decimal")
    parsed = int(value, 10)
    if parsed <= 0:
        raise ValueError("PSIM proposal number is not positive")
    return parsed


def parse_dependency_ids(value: str, *, self_id: int) -> tuple[int, ...]:
    if not value:
        return ()
    if "\n" in value:
        raise ValueError("PSIM dependency field must be one line")
    tokens = value.split(",")
    if len(tokens) > MAX_DEPENDENCIES:
        raise ValueError("PSIM dependency count exceeds maximum")
    parsed: list[int] = []
    for token in tokens:
        stripped = token.strip(" \t")
        dependency = parse_positive_proposal_number(stripped)
        if dependency == self_id:
            raise ValueError("PSIM self dependency")
        parsed.append(dependency)
    if len(parsed) != len(set(parsed)):
        raise ValueError("PSIM duplicate dependency")
    return tuple(sorted(parsed))


def parse_eip_preamble(raw: bytes) -> dict[str, str]:
    lines = normalize_blob_bytes(raw)
    if not lines or lines[0] != "---":
        raise ValueError("PSIM EIP opening fence is not exact")
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError("PSIM EIP closing fence is missing")
    fields = _parse_header_lines(
        lines[1:closing_index],
        field_lines_may_be_indented=False,
        allow_empty_values=False,
        comment_styles=("hash",),
    )
    if "eip" not in fields:
        raise ValueError("PSIM EIP number field is missing")
    parse_positive_proposal_number(fields["eip"])
    return fields


def parse_bip_preamble(raw: bytes) -> dict[str, str]:
    lines = normalize_blob_bytes(raw)
    leading_blank_count = 0
    while leading_blank_count < len(lines) and not lines[leading_blank_count]:
        leading_blank_count += 1
    if leading_blank_count > 3 or leading_blank_count == len(lines):
        raise ValueError("PSIM BIP leading blank prefix is invalid")
    start = leading_blank_count
    if lines[start] == "<pre>":
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[start + 1 :], start=start + 1)
                if line == "</pre>"
            ),
            None,
        )
        if closing_index is None:
            raise ValueError("PSIM BIP mediawiki closing tag is missing")
        header_lines = lines[start + 1 : closing_index]
    else:
        end = next(
            (
                index
                for index, line in enumerate(lines[start:], start=start)
                if not line
            ),
            len(lines),
        )
        header_lines = lines[start:end]
    fields = _parse_header_lines(
        header_lines,
        field_lines_may_be_indented=True,
        allow_empty_values=True,
        comment_styles=("hash", "html"),
    )
    if "bip" not in fields:
        raise ValueError("PSIM BIP number field is missing")
    parse_positive_proposal_number(fields["bip"])
    return fields


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"PSIM timestamp is not UTC: {value}")
    return parsed


def _validate_decision_binding() -> None:
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("PSIM-D1 decision document hash changed")
    if SHA1_PATTERN.fullmatch(SELECTION_COMMIT) is None:
        raise RuntimeError("PSIM-D1 selection commit is malformed")


def _validate_repositories() -> None:
    if len(REPOSITORIES) != 2:
        raise RuntimeError("PSIM repository roster must contain exactly two rows")
    if {row.protocol for row in REPOSITORIES} != {"ethereum", "bitcoin"}:
        raise RuntimeError("PSIM protocol roster changed")
    if len({row.remote for row in REPOSITORIES}) != 2:
        raise RuntimeError("PSIM remotes are not unique")
    for row in REPOSITORIES:
        if row.branch != "master":
            raise RuntimeError("PSIM branch identity changed")
        if row.remote_head_symref != "refs/remotes/origin/master":
            raise RuntimeError("PSIM remote HEAD symref changed")
        if row.object_format != "sha1":
            raise RuntimeError("PSIM object format changed")
        if SHA1_PATTERN.fullmatch(row.sealed_tip) is None:
            raise RuntimeError("PSIM sealed tip is malformed")
        re.compile(row.path_pattern, re.ASCII)
    eip = re.compile(EIP_PATH_PATTERN, re.ASCII)
    bip = re.compile(BIP_PATH_PATTERN, re.ASCII)
    if eip.fullmatch("EIPS/eip-1559.md") is None:
        raise RuntimeError("PSIM EIP path grammar rejects canonical example")
    if bip.fullmatch("bip-0341.mediawiki") is None:
        raise RuntimeError("PSIM BIP path grammar rejects mediawiki example")
    if bip.fullmatch("bip-0003.md") is None:
        raise RuntimeError("PSIM BIP path grammar rejects markdown example")
    forbidden_examples = (
        "EIPS/eip-0001.md",
        "eips/eip-1.md",
        "bip-341.mediawiki",
        "bip-0341.txt",
        "README.md",
    )
    if any(eip.fullmatch(x) or bip.fullmatch(x) for x in forbidden_examples):
        raise RuntimeError("PSIM path grammar accepts a forbidden example")


def _validate_time_contract() -> None:
    start = _parse_utc(SOURCE_START)
    end = _parse_utc(SOURCE_END_EXCLUSIVE)
    card_end = _parse_utc(CARD_END_EXCLUSIVE)
    if not start < end < card_end:
        raise RuntimeError("PSIM source/card envelope is malformed")
    if [row.delay_calendar_days for row in ARCHIVE_SCHEDULES] != [2, 7, 30, 90]:
        raise RuntimeError("PSIM archive-delay roster changed")
    primaries = [row for row in ARCHIVE_SCHEDULES if row.primary_economic_clock]
    if len(primaries) != 1 or primaries[0].name != "ARCHIVE_D90":
        raise RuntimeError("PSIM primary archive clock is not D90")
    for row in ARCHIVE_SCHEDULES:
        if row.profitability_claim_allowed != row.primary_economic_clock:
            raise RuntimeError("PSIM profitability clock permissions changed")
    if DECISION_FREQUENCY != "1D" or DECISION_UTC_TIME != "12:05:00":
        raise RuntimeError("PSIM daily decision schedule changed")
    if COUNTERPART_LOOKBACK_DAYS != 90:
        raise RuntimeError("PSIM counterpart lookback changed")


def _validate_splits() -> None:
    if [row["name"] for row in SPLITS] != ["train", "test", "eval"]:
        raise RuntimeError("PSIM split order changed")
    if SPLITS[0]["decision_start"] != SOURCE_START:
        raise RuntimeError("PSIM TRAIN does not start at source start")
    if SPLITS[-1]["decision_end_exclusive"] != SOURCE_END_EXCLUSIVE:
        raise RuntimeError("PSIM EVAL does not end at source end")
    for left, right in zip(SPLITS, SPLITS[1:]):
        if left["decision_end_exclusive"] != right["decision_start"]:
            raise RuntimeError("PSIM splits are not contiguous")
    for row in SPLITS:
        if _parse_utc(row["decision_start"]) >= _parse_utc(
            row["decision_end_exclusive"]
        ):
            raise RuntimeError("PSIM split interval is malformed")
        for key, value in row.items():
            if key.startswith("minimum_") and key != "minimum_counterpart_fraction":
                if not isinstance(value, int) or value <= 0:
                    raise RuntimeError("PSIM split minima must be positive integers")
            if key.startswith(("minimum_", "maximum_")) and isinstance(value, str):
                decimal = Decimal(value)
                if not Decimal("0") < decimal <= Decimal("1"):
                    raise RuntimeError("PSIM split share bound is outside (0,1]")


def _validate_parser_and_representation() -> None:
    bounds = (
        MAX_BLOB_BYTES,
        MAX_HEADER_BYTES,
        MAX_HEADER_LINES,
        MAX_LINE_BYTES,
        MAX_SECTIONS,
        MAX_DEPENDENCIES,
        MAX_MODEL_TEXT_BYTES_PER_EVENT,
        MAX_MODEL_EVENTS_PER_CARD,
    )
    if any(not isinstance(value, int) or value <= 0 for value in bounds):
        raise RuntimeError("PSIM parser bounds must be positive integers")
    if MAX_HEADER_BYTES > MAX_BLOB_BYTES or MAX_LINE_BYTES > MAX_BLOB_BYTES:
        raise RuntimeError("PSIM parser sub-bound exceeds blob bound")
    re.compile(HEADER_KEY_PATTERN, re.ASCII)
    if set(SECTION_ALIASES) != {
        "ABSTRACT",
        "MOTIVATION",
        "SPECIFICATION",
        "RATIONALE",
        "BACKWARD_COMPATIBILITY",
        "SECURITY",
        "TESTS",
        "IMPLEMENTATION",
        "COPYRIGHT",
        "OTHER",
    }:
        raise RuntimeError("PSIM section alias vocabulary changed")
    aliases = [
        alias
        for canonical, values in SECTION_ALIASES.items()
        if canonical != "OTHER"
        for alias in values
    ]
    if len(aliases) != len(set(aliases)):
        raise RuntimeError("PSIM section aliases are not unique")
    for name, edges in BUCKET_EDGES.items():
        if not edges or edges[0] != 0:
            raise RuntimeError(f"PSIM bucket {name} does not start at zero")
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise RuntimeError(f"PSIM bucket {name} is not strictly increasing")
    if EVENT_TYPES != ("CREATE", "UPDATE", "DELETE"):
        raise RuntimeError("PSIM event type vocabulary changed")
    if len(RELATION_TOKENS) != 6 or len(set(RELATION_TOKENS)) != 6:
        raise RuntimeError("PSIM relation token vocabulary changed")
    if MODEL_ACTIONS != ("TARGET_LONG", "TARGET_FLAT", "TARGET_SHORT"):
        raise RuntimeError("PSIM target action vocabulary changed")
    if len(SOURCE_ONLY_GATES) != 13 or len(set(SOURCE_ONLY_GATES)) != 13:
        raise RuntimeError("PSIM source gate roster changed")
    if len(RELATION_CONTROLS) != 7 or len(set(RELATION_CONTROLS)) != 7:
        raise RuntimeError("PSIM relation control roster changed")
    if set(CONTROL_TRANSFORMS) != set(RELATION_CONTROLS):
        raise RuntimeError("PSIM relation control transforms changed")
    if set(CONTROL_ELIGIBILITY) != set(RELATION_CONTROLS):
        raise RuntimeError("PSIM relation control eligibility changed")
    if any(not value.strip() for value in CONTROL_TRANSFORMS.values()):
        raise RuntimeError("PSIM relation control transform is empty")
    if any(not value.strip() for value in CONTROL_ELIGIBILITY.values()):
        raise RuntimeError("PSIM relation control eligibility is empty")
    for protocol, proposal_ids in MEMORIZATION_QUARANTINE.items():
        if protocol not in {"ethereum", "bitcoin"}:
            raise RuntimeError("PSIM quarantine protocol changed")
        if (
            not proposal_ids
            or len(proposal_ids) != len(set(proposal_ids))
            or any(value <= 0 for value in proposal_ids)
        ):
            raise RuntimeError("PSIM quarantine proposal IDs are malformed")


def build_preregistration() -> dict[str, Any]:
    _validate_decision_binding()
    _validate_repositories()
    _validate_time_contract()
    _validate_splits()
    _validate_parser_and_representation()
    core: dict[str, Any] = {
        "protocol_version": "psim_d1_source_preregistration_v1",
        "candidate": {
            "id": "PSIM-D1",
            "name": "Protocol Specification Intent-Maturity relation RLLM",
            "source_axis": "official_eip_bip_specification_revision_relation",
            "selection_commit": SELECTION_COMMIT,
            "stage": "source_support_only",
        },
        "decision_binding": {
            "path": str(DECISION_PATH),
            "sha256": DECISION_SHA256,
        },
        "official_sources": list(OFFICIAL_SOURCES),
        "source_contract": {
            "start": SOURCE_START,
            "end_exclusive": SOURCE_END_EXCLUSIVE,
            "card_end_exclusive": CARD_END_EXCLUSIVE,
            "repositories": [
                {
                    **asdict(row),
                    "document_formats": list(row.document_formats),
                }
                for row in REPOSITORIES
            ],
            "clone_arguments": [
                "--filter=blob:none",
                "--no-checkout",
                "--single-branch",
                "--branch",
                "master",
            ],
            "traversal": "sealed_tip_complete_first_parent_oldest_to_newest",
            "tree_delta_parent": "parent_one_or_empty_tree_for_root",
            "rename_detection": False,
            "path_handling": "NUL_SAFE_BYTE_EXACT_CASE_SENSITIVE",
            "commit_subject_used_for_inclusion_or_classification": False,
            "current_checkout_used_as_historical_truth": False,
            "github_pr_issue_label_review_metadata_allowed": False,
            "git_fsck_no_dangling_required": True,
            "git_version_and_object_format_bound_at_execution_seal": True,
            "disk_limit_gib": DISK_LIMIT_GIB,
        },
        "excluded_feasibility_probe": {
            "probe_start_inclusive": "2024-01-01T00:00:00Z",
            "source_interval_incidence_opened": False,
            "ethereum": {
                "commits": 2_280,
                "commit_days": 692,
                "proposal_path_touches": 2_358,
                "unique_proposal_paths": 413,
                "strict_utf8_newest_blob": True,
            },
            "bitcoin": {
                "commits": 1_146,
                "commit_days": 404,
                "proposal_path_touches": 1_664,
                "unique_proposal_paths": 210,
                "mediawiki_path_touches": 1_524,
                "markdown_path_touches": 140,
                "strict_utf8_newest_blob": True,
            },
            "probe_may_set_economic_direction_or_threshold": False,
        },
        "availability_contract": {
            "committer_time_is_publication_receipt": False,
            "effective_day": "running_max_utc_committer_calendar_day",
            "schedules": [asdict(row) for row in ARCHIVE_SCHEDULES],
            "daily_decision_frequency": DECISION_FREQUENCY,
            "daily_decision_utc_time": DECISION_UTC_TIME,
            "primary_historical_economic_schedule": "ARCHIVE_D90",
            "shorter_schedules_are_diagnostics_only": True,
            "counterpart_lookback_days": COUNTERPART_LOOKBACK_DAYS,
            "live_available_at": (
                "max(historical_floor,durable_fetch_object_verify_extract_"
                "hash_manifest_commit_time)"
            ),
            "force_push_or_unreachable_ancestor_action": "HALT_NO_REBUILD",
        },
        "event_contract": {
            "event_types": list(EVENT_TYPES),
            "create": "zero_old_blob_and_exactly_one_new_blob",
            "update": "exactly_one_old_blob_and_exactly_one_new_blob",
            "delete": "exactly_one_old_blob_and_zero_new_blob",
            "format_migration_same_number_is_update": True,
            "multiple_old_or_new_blobs_same_number": "REJECT",
            "duplicate_number_paths_in_one_tree": "REJECT",
            "path_number_preamble_number_must_match": True,
            "event_id_formula": (
                "SHA256(protocol||NUL||commit_oid||NUL||canonical_decimal_"
                "proposal_number||NUL||old_blob_oid_or_NULL||NUL||"
                "new_blob_oid_or_NULL)"
            ),
            "all_matching_path_events_retained": True,
        },
        "boundary_reset_contract": {
            "pre_2020_blob_warmup_allowed": False,
            "window_revision_count_before_first_event": 0,
            "window_age_before_first_event": "PRE_WINDOW",
            "stale_age_before_first_protocol_event": "NO_EVENT_YET",
            "first_update_old_blob_role": "PRE_WINDOW_BASELINE",
            "pre_first_event_dependency_state": "PRE_WINDOW_UNKNOWN",
            "old_blob_creates_synthetic_prior_event": False,
            "boundary_states": list(BOUNDARY_STATES),
        },
        "parser_contract": {
            "strict_utf8": True,
            "nul_allowed": False,
            "newline_normalization": "CRLF_AND_CR_TO_LF",
            "unicode_normalization": "NFC",
            "strip_trailing_horizontal_whitespace": True,
            "maximum_blob_bytes": MAX_BLOB_BYTES,
            "maximum_header_bytes": MAX_HEADER_BYTES,
            "maximum_header_lines": MAX_HEADER_LINES,
            "maximum_line_bytes": MAX_LINE_BYTES,
            "maximum_sections": MAX_SECTIONS,
            "maximum_dependencies": MAX_DEPENDENCIES,
            "header_key_pattern": HEADER_KEY_PATTERN,
            "header_key_normalization": (
                "NFC_ASCII_CASEFOLD_STRIP_COLLAPSE_INTERNAL_SPACE"
            ),
            "reference_parser": {
                "version": "PSIM_PREAMBLE_STATE_MACHINE_V1",
                "normalizer_function": "normalize_blob_bytes",
                "eip_function": "parse_eip_preamble",
                "bip_function": "parse_bip_preamble",
                "dependency_function": "parse_dependency_ids",
                "utf8_bom_allowed": False,
                "full_line_comments": {
                    "eip": ["ASCII optional whitespace then #"],
                    "bip": [
                        "ASCII optional whitespace then #",
                        "single-line exact <!--...--> after ASCII strip",
                    ],
                },
                "inline_comments_stripped": False,
                "field_separator": "first colon after normalized key",
                "quoted_or_bracket_values": "retained_as_opaque_text",
                "duplicate_keys_after_casefold": "REJECT",
                "eip_empty_value": (
                    "REJECT unless populated by one or more continuations"
                ),
                "bip_empty_value": "ALLOW",
                "continuation": (
                    "line begins ASCII space/tab, does not parse as a field "
                    "under format mode, and has nonempty stripped content"
                ),
                "eip_opening_and_first_closing_fence": "exact --- line",
                "bip_mediawiki_tags": "exact <pre> and </pre> lines",
                "bip_markdown_end": "first blank line after first field or EOF",
                "maximum_bip_leading_blank_lines": 3,
            },
            "eip_frontmatter": {
                "opening_line": "---",
                "closing_line": "---",
                "required_number_field": "eip",
                "dependency_fields": list(EIP_DEPENDENCY_FIELDS),
            },
            "bip_preamble": {
                "optional_mediawiki_opening_line": "<pre>",
                "optional_mediawiki_closing_line": "</pre>",
                "markdown_termination": "first_blank_line_after_first_field",
                "required_number_field_casefold": "bip",
                "dependency_fields_casefold": list(BIP_DEPENDENCY_FIELDS),
            },
            "field_continuation": "leading_ascii_space_or_tab",
            "duplicate_header_key_allowed": False,
            "proposal_number_leading_zero_normalization": "ASCII_DECIMAL_TO_INT",
            "primary_proposal_number_must_be_positive": True,
            "dependency_token_grammar": "comma_separated_positive_ascii_decimal",
            "duplicate_or_self_dependency_allowed": False,
            "current_process_vocabulary_validation_allowed": False,
            "declared_status_is_model_visible": False,
            "metadata_parse_failure_action": "REJECT_NOT_AUDIT_ONLY",
            "markdown_heading_pattern": r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$",
            "mediawiki_heading_pattern": r"^(={2,6})[ \t]*(.*?)[ \t]*\1$",
            "heading_alias_normalization": (
                "NFC_CASEFOLD_STRIP_COLLAPSE_ASCII_WHITESPACE_"
                "REMOVE_TRAILING_COLON"
            ),
            "section_aliases": {
                key: list(values) for key, values in SECTION_ALIASES.items()
            },
            "line_diff_algorithm": (
                "python_difflib_SequenceMatcher_autojunk_false_over_"
                "normalized_lines"
            ),
            "python_version_bound_at_execution_seal": True,
        },
        "bucket_contract": {
            "interval_rule": (
                "NEGATIVE_OR_SENTINEL_SEPARATE; OTHERWISE RIGHT_CLOSED_"
                "AT_LAST_EDGE_WITH_OVERFLOW"
            ),
            "edges": {key: list(values) for key, values in BUCKET_EDGES.items()},
            "raw_numeric_values_model_visible": False,
        },
        "daily_relation_contract": {
            "canonical_event_order": [
                "availability_utc",
                "first_parent_index",
                "proposal_number",
                "event_id",
            ],
            "both_protocol_sets_nonempty": "complete_cartesian_product",
            "exactly_one_protocol_set_nonempty": (
                "each_anchor_with_most_recent_opposite_event_in_90_days"
            ),
            "opposite_tie_break": "maximum_event_id",
            "missing_opposite_sentinel": "NO_COUNTERPART",
            "both_sets_empty_sentinel": "NO_ANCHOR",
            "semantic_or_market_pair_selection_allowed": False,
            "maximum_model_text_bytes_per_event": (
                MAX_MODEL_TEXT_BYTES_PER_EVENT
            ),
            "maximum_model_events_per_card": MAX_MODEL_EVENTS_PER_CARD,
            "over_limit_card_action": "SOURCE_GATE_REJECT_NO_TRUNCATION",
            "prior_card_hash_required": True,
            "explicit_no_new_event_state_required": True,
        },
        "representation_contract": {
            "deterministic_source_tokens": [
                "protocol",
                "event_type",
                "window_revision_count_bucket",
                "window_age_bucket",
                "update_gap_bucket",
                "stale_age_bucket",
                "old_section_presence",
                "new_section_presence",
                "changed_section_bitset",
                "dependency_delta_state",
                "dependency_edge_delta_count_bucket",
                "line_change_count_bucket",
                "changed_section_count_bucket",
                "counterpart_state",
                "prior_card_hash",
            ],
            "dependency_delta_states": list(DEPENDENCY_DELTA_STATES),
            "later_relation_tokens": list(RELATION_TOKENS),
            "later_model_actions": list(MODEL_ACTIONS),
            "single_model_single_call_per_card": True,
            "analyzer_trader_pair_allowed": False,
            "free_form_rationale_allowed": False,
            "abstain_forces_target_flat": True,
            "insufficient_evidence_forces_current_target": True,
            "split_boundary_without_position_forces_target_flat": True,
            "raw_proposal_number_hash_path_timestamp_date_author_url_allowed": False,
            "raw_price_return_rank_pnl_model_inputs_allowed": False,
            "model_controls_leverage_size_stop_cost_or_reward": False,
        },
        "memorization_contract": {
            "quarantine": {
                key: list(values)
                for key, values in MEMORIZATION_QUARANTINE.items()
            },
            "quarantined_events_retained_in_source_support": True,
            "quarantined_relation_units_model_or_economics_allowed": False,
            "challenge_selection_hash_suffix": "PSIM_MEMORIZATION_V1",
            "maximum_events_per_protocol_source_year": 16,
            "candidate_ids_per_challenge": 8,
            "decoys_same_protocol_and_source_year": True,
            "forced_choice_no_abstention": True,
            "chance_probability": "0.125",
            "tests": ["ethereum", "bitcoin", "combined"],
            "bonferroni_family_alpha": "0.01",
            "per_test_alpha": str(Decimal("0.01") / Decimal(3)),
            "minimum_eligible_events_per_protocol": 32,
            "base_and_final_models_must_pass_before_market": True,
            "first_failure_action": (
                "REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_OR_OUTCOMES"
            ),
            "repair_resample_or_model_swap_after_result_allowed": False,
        },
        "split_contract": {
            "splits": list(SPLITS),
            "source_event_support_assignment_field": "event_effective_day",
            "daily_card_and_relation_support_assignment_field": (
                "decision_timestamp_under_ARCHIVE_D90"
            ),
            "shorter_schedule_support_role": (
                "replay_and_control_only_not_threshold_rescue"
            ),
            "later_economic_split_assignment_field": (
                "decision_timestamp_under_ARCHIVE_D90"
            ),
            "post_2023_proposal_content_policy": "SEALED",
            "market_outcome_policy": "FORBIDDEN until separate evaluator freeze",
            "later_test_eval_minimum_cagr_strict_mdd": "3.0",
            "later_test_eval_positive_absolute_return_required": True,
        },
        "source_support_contract": {
            "gates_in_order": list(SOURCE_ONLY_GATES),
            "all_gates_required": True,
            "relation_controls": list(RELATION_CONTROLS),
            "relation_control_transforms": dict(CONTROL_TRANSFORMS),
            "relation_control_eligibility": dict(CONTROL_ELIGIBILITY),
            "control_sensitivity_metric": {
                "comparison_unit": (
                    "canonical local daily relation-card payload for one "
                    "decision day"
                ),
                "excluded_from_comparison_payload": [
                    "card_hash",
                    "prior_card_hash",
                    "raw_commit_or_blob_ids",
                    "control_name",
                    "audit_counters",
                ],
                "changed_definition": (
                    "SHA256(canonical_json(transformed_local_payload)) != "
                    "SHA256(canonical_json(baseline_local_payload))"
                ),
                "denominator": (
                    "unique eligible decision days for one exact "
                    "control,schedule,split cell"
                ),
                "numerator": "eligible denominator days whose payload changed",
                "minimum_eligible_days_per_cell": 4,
                "minimum_changed_fraction_per_cell": "0.10",
                "aggregation": (
                    "require every control x archive schedule x split cell; "
                    "no pooling or weighting"
                ),
                "zero_eligible_cell_action": "REJECT",
                "quarantined_events_in_source_control_payload": True,
                "first_failure_action": (
                    "REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
                ),
            },
            "minimum_event_types_overall": 2,
            "minimum_section_categories_overall": 4,
            "minimum_dependency_delta_categories_overall": 2,
            "minimum_revision_buckets_overall": 3,
            "maximum_top_changed_section_share": "0.80",
            "minimum_non_quarantined_events_per_protocol": 32,
            "parser_success_fraction_required": "1.0",
            "independent_replay_roots": 2,
            "shared_git_object_alternates_allowed": False,
            "shared_worktree_or_cache_allowed": False,
            "future_append_cutoff": SOURCE_END_EXCLUSIVE,
            "first_failure_action": (
                "REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
            ),
            "source_drop_or_repair_allowed": False,
            "gate_relaxation_after_incidence_allowed": False,
        },
        "forbidden_access_contract": {
            "counters": {name: 0 for name in FORBIDDEN_COUNTERS},
            "network_calls_during_preregistration": 0,
            "git_commands_during_preregistration": 0,
            "source_incidence_opened": False,
            "proposal_blobs_opened": False,
            "btc_or_funding_outcomes_opened": False,
            "models_loaded": 0,
        },
        "next_authorized_step": (
            "implement and seal synthetic-only PSIM-D1 source-support evaluator"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(
    output: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, dict[str, Any]]:
    payload = build_preregistration()
    destination = _repo_path(output)
    serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        existing = destination.read_bytes()
        if existing != serialized:
            raise RuntimeError("existing PSIM preregistration differs")
        return destination, payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialized)
    return destination, payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    destination, payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"]["id"],
                "output": str(destination),
                "manifest_hash": payload["manifest_hash"],
                "source_incidence_opened": payload[
                    "forbidden_access_contract"
                ]["source_incidence_opened"],
                "next_authorized_step": payload["next_authorized_step"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
