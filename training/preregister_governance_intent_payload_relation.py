"""Freeze the GIPR-D1 Ethereum source contract before event incidence.

This module is deliberately source-blind.  It contains only official contract
identities, bounded no-log transport probes, causal block boundaries, grammar,
support gates, controls, and forbidden-access counters.  It must not fetch or
decode a governance event, market row, funding row, outcome, model output, or
trade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Crypto.Hash import keccak


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_governance_intent_payload_relation.py")
DEFAULT_OUTPUT = Path(
    "results/governance_intent_payload_relation_preregistration_2026-07-25.json"
)
DECISION_PATH = Path(
    "docs/post-clor-d1-alpha-mechanism-audit-2026-07-25.md"
)
DECISION_SHA256 = (
    "fd52594fcdfbd85d9a8385ae57513f98ebca169da43f31cafbe058b7519e7c43"
)

CHAIN_ID = 1
SOURCE_START = "2020-01-01T00:00:00Z"
SOURCE_END_EXCLUSIVE = "2024-01-01T00:00:00Z"
START_BOUNDARY_BLOCK = 9_193_266
START_BOUNDARY_HASH = (
    "0xfa39cb98792d28b79138fc0a2ab7c08e59c00b43bd0dcd0b7e4bd49684f7216c"
)
START_BOUNDARY_TIMESTAMP = "2020-01-01T00:00:11Z"
END_BOUNDARY_BLOCK = 18_908_895
END_BOUNDARY_HASH = (
    "0x08f760c77e7a5843464404bf59d1f042a8b2ec18ae9345a16940546216069eec"
)
END_BOUNDARY_TIMESTAMP = "2024-01-01T00:00:11Z"
LAST_SOURCE_BLOCK = END_BOUNDARY_BLOCK - 1

CONFIRMATION_BLOCKS = 64
MAX_BLOCK_RANGE = 5_000
HEADER_BATCH_SIZE = 100
DISK_LIMIT_GIB = 300

DECISION_FREQUENCY = "1D"
DECISION_UTC_TIME = "00:00:00"
HISTORY_DAYS = 28
MAX_PROPOSAL_AGE_DAYS = 90

MAX_ACTIONS_PER_PROPOSAL = 10
MAX_DESCRIPTION_BYTES = 131_072
MAX_SIGNATURE_BYTES = 512
MAX_CALLDATA_BYTES_PER_ACTION = 65_536
MAX_CALLDATA_BYTES_PER_PROPOSAL = 262_144

OFFICIAL_SOURCES = (
    "https://docs.compound.finance/v2/",
    "https://docs.compound.finance/v2/governance/",
    (
        "https://github.com/compound-finance/compound-protocol/blob/"
        "a3214f67b73310d547e00fc578e8355911c9d376/contracts/Governance/"
        "GovernorAlpha.sol"
    ),
    (
        "https://github.com/compound-finance/compound-protocol/blob/"
        "a3214f67b73310d547e00fc578e8355911c9d376/contracts/Governance/"
        "GovernorBravoInterfaces.sol"
    ),
    (
        "https://github.com/compound-finance/compound-protocol/blob/"
        "a3214f67b73310d547e00fc578e8355911c9d376/networks/mainnet.json"
    ),
    (
        "https://developers.uniswap.org/docs/ecosystem/governance/"
        "technical-reference"
    ),
    (
        "https://github.com/Uniswap/governance/blob/"
        "eabd8c71ad01f61fb54ed6945162021ee419998e/contracts/"
        "GovernorAlpha.sol"
    ),
    "https://docs.openzeppelin.com/contracts/4.x/governance",
    "https://docs.openzeppelin.com/contracts/4.x/api/governance",
    "https://ethereum.org/developers/docs/apis/json-rpc/",
    "https://ethereum.github.io/execution-apis/api/methods/eth_getLogs/",
)

ADDRESS_PATTERN = re.compile(r"0x[0-9a-f]{40}", re.ASCII)
HASH_PATTERN = re.compile(r"0x[0-9a-f]{64}", re.ASCII)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)


def _event_topic(signature: str) -> str:
    digest = keccak.new(digest_bits=256)
    digest.update(signature.encode("ascii"))
    return "0x" + digest.hexdigest()


@dataclass(frozen=True)
class EventSpec:
    event: str
    signature: str
    topic: str
    topic_count: int
    lifecycle_rank: int


EVENT_SPECS = (
    EventSpec(
        event="proposal_created",
        signature=(
            "ProposalCreated(uint256,address,address[],uint256[],string[],"
            "bytes[],uint256,uint256,string)"
        ),
        topic=(
            "0x7d84a6263ae0d98d3329bd7b46bb4e8d6f98cd35a7adb45c274c8b7fd5ebd5e0"
        ),
        topic_count=1,
        lifecycle_rank=0,
    ),
    EventSpec(
        event="proposal_canceled",
        signature="ProposalCanceled(uint256)",
        topic=(
            "0x789cf55be980739dad1d0699b93b58e806b51c9d96619bfa8fe0a28abaa7b30c"
        ),
        topic_count=1,
        lifecycle_rank=3,
    ),
    EventSpec(
        event="proposal_queued",
        signature="ProposalQueued(uint256,uint256)",
        topic=(
            "0x9a2e42fd6722813d69113e7d0079d3d940171428df7373df9c7f7617cfda2892"
        ),
        topic_count=1,
        lifecycle_rank=1,
    ),
    EventSpec(
        event="proposal_executed",
        signature="ProposalExecuted(uint256)",
        topic=(
            "0x712ae1383f79ac853f8d882153778e0260ef8f03b504e2866e0593e04d2b291f"
        ),
        topic_count=1,
        lifecycle_rank=2,
    ),
)


@dataclass(frozen=True)
class GovernorSpec:
    protocol: str
    generation: str
    address: str
    first_code_block: int
    first_code_block_hash: str
    first_code_timestamp: str
    runtime_code_bytes: int
    runtime_code_sha256: str
    last_source_code_bytes: int
    last_source_code_sha256: str


GOVERNORS = (
    GovernorSpec(
        protocol="compound",
        generation="alpha",
        address="0xc0da01a04c3f3e0be433606045bb7017a7323e38",
        first_code_block=9_601_459,
        first_code_block_hash=(
            "0xbddb934b440954acb2dc68405b5d1dddef95d6243b64940bd244ce20102d7e5b"
        ),
        first_code_timestamp="2020-03-04T00:33:10Z",
        runtime_code_bytes=14_339,
        runtime_code_sha256=(
            "2bbca6c569d84c5d4ec74ae6d5e0ad4c5c6f2f72faa73c737ae7b13e92b5dd70"
        ),
        last_source_code_bytes=14_339,
        last_source_code_sha256=(
            "2bbca6c569d84c5d4ec74ae6d5e0ad4c5c6f2f72faa73c737ae7b13e92b5dd70"
        ),
    ),
    GovernorSpec(
        protocol="compound",
        generation="bravo",
        address="0xc0da02939e1441f497fd74f78ce7decb17b66529",
        first_code_block=12_006_099,
        first_code_block_hash=(
            "0x00454a9add83814d32e58011466c6f63c0a0ea855324a5eabc7aa38c04d9ef5d"
        ),
        first_code_timestamp="2021-03-09T19:07:59Z",
        runtime_code_bytes=1_065,
        runtime_code_sha256=(
            "092bfd06201b9c2b39eec9047663a84a3faa9dcf806266c8ea8c711022f93ea5"
        ),
        last_source_code_bytes=1_065,
        last_source_code_sha256=(
            "092bfd06201b9c2b39eec9047663a84a3faa9dcf806266c8ea8c711022f93ea5"
        ),
    ),
    GovernorSpec(
        protocol="uniswap",
        generation="alpha_v0",
        address="0x5e4be8bc9637f0eaa1a755019e06a68ce081d58f",
        first_code_block=10_861_678,
        first_code_block_hash=(
            "0x3e8f26ac4837d27fb9162ab0c5a7effacb77020c6b9a0685487658ca816d10fd"
        ),
        first_code_timestamp="2020-09-14T18:11:45Z",
        runtime_code_bytes=14_528,
        runtime_code_sha256=(
            "52522136d37cd631fec7217c1eaa33a51c36a0ab8d39dacf78ca81f96917bf73"
        ),
        last_source_code_bytes=14_528,
        last_source_code_sha256=(
            "52522136d37cd631fec7217c1eaa33a51c36a0ab8d39dacf78ca81f96917bf73"
        ),
    ),
    GovernorSpec(
        protocol="uniswap",
        generation="alpha_v2",
        address="0xc4e172459f1e7939d522503b81afaac1014ce6f6",
        first_code_block=12_543_659,
        first_code_block_hash=(
            "0xa24265bbf5b075a09603597df2062d5b1023688e34836f2f60870f2d786b373e"
        ),
        first_code_timestamp="2021-05-31T18:14:24Z",
        runtime_code_bytes=14_676,
        runtime_code_sha256=(
            "e3017ff6353b11ba6edaa6146214131a5d6eb495a1ed1077c06f30f499d6352f"
        ),
        last_source_code_bytes=14_676,
        last_source_code_sha256=(
            "e3017ff6353b11ba6edaa6146214131a5d6eb495a1ed1077c06f30f499d6352f"
        ),
    ),
    GovernorSpec(
        protocol="uniswap",
        generation="bravo",
        address="0x408ed6354d4973f66138c91495f2f2fcbd8724c3",
        first_code_block=13_059_157,
        first_code_block_hash=(
            "0x8dd8250f60a6fb2b2b79c9eb933942d0414fb10320680d6596d0d17170ae443e"
        ),
        first_code_timestamp="2021-08-20T01:12:02Z",
        runtime_code_bytes=1_065,
        runtime_code_sha256=(
            "e8105a936619cc787de45ab68bef2e92329eee665bf0cdb4924c0d46c0ba343e"
        ),
        last_source_code_bytes=1_065,
        last_source_code_sha256=(
            "e8105a936619cc787de45ab68bef2e92329eee665bf0cdb4924c0d46c0ba343e"
        ),
    ),
)


@dataclass(frozen=True)
class TargetRoleSpec:
    protocol: str
    role: str
    address: str


KNOWN_TARGET_ROLES = (
    TargetRoleSpec(
        "compound",
        "TIMELOCK",
        "0x6d903f6003cca6255d85cca4d3b5e5146dc33925",
    ),
    TargetRoleSpec(
        "compound",
        "GOVERNANCE_TOKEN",
        "0xc00e94cb662c3520282e6f5717214004a7f26888",
    ),
    TargetRoleSpec(
        "compound",
        "RISK_CONTROLLER",
        "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",
    ),
    TargetRoleSpec(
        "uniswap",
        "TIMELOCK",
        "0x1a9c8182c09f50c8318d769245bea52c32be35bc",
    ),
    TargetRoleSpec(
        "uniswap",
        "GOVERNANCE_TOKEN",
        "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
    ),
    TargetRoleSpec(
        "uniswap",
        "V2_FACTORY",
        "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
    ),
    TargetRoleSpec(
        "uniswap",
        "V3_FACTORY",
        "0x1f98431c8ad98523631ae4a59f267346ea31f984",
    ),
)

LIFECYCLE_TRANSITIONS = {
    "proposal_created": ["proposal_queued", "proposal_canceled"],
    "proposal_queued": ["proposal_executed", "proposal_canceled"],
    "proposal_executed": [],
    "proposal_canceled": [],
}


@dataclass(frozen=True)
class BoundarySpec:
    name: str
    timestamp: str
    first_block: int
    block_hash: str
    observed_block_timestamp: str


YEAR_BOUNDARIES = (
    BoundarySpec(
        "source_start",
        "2020-01-01T00:00:00Z",
        9_193_266,
        "0xfa39cb98792d28b79138fc0a2ab7c08e59c00b43bd0dcd0b7e4bd49684f7216c",
        "2020-01-01T00:00:11Z",
    ),
    BoundarySpec(
        "train_year_2",
        "2021-01-01T00:00:00Z",
        11_565_019,
        "0x12620d72a9306bc7c5ed1e2ba1ac75115197020d5101925305c714e8e1d174d7",
        "2021-01-01T00:00:00Z",
    ),
    BoundarySpec(
        "test_start",
        "2022-01-01T00:00:00Z",
        13_916_166,
        "0x2f03b3220805ee951f2d4c250621e9c2175439cef524e976aed707a2d595729e",
        "2022-01-01T00:00:03Z",
    ),
    BoundarySpec(
        "eval_start",
        "2023-01-01T00:00:00Z",
        16_308_190,
        "0x53dd35d982c984441b3b613919d64dbbf131063d0f85804d77f93f190fa5e106",
        "2023-01-01T00:00:11Z",
    ),
    BoundarySpec(
        "source_end_exclusive",
        "2024-01-01T00:00:00Z",
        18_908_895,
        "0x08f760c77e7a5843464404bf59d1f042a8b2ec18ae9345a16940546216069eec",
        "2024-01-01T00:00:11Z",
    ),
)


SPLITS = (
    {
        "name": "train",
        "start": "2020-01-01T00:00:00Z",
        "end_exclusive": "2022-01-01T00:00:00Z",
        "minimum_proposals": 24,
        "minimum_proposals_per_protocol": 4,
        "minimum_actions": 24,
        "minimum_active_calendar_months": 8,
        "minimum_daily_decisions_after_warmup": 700,
        "minimum_unique_targets": 8,
        "minimum_unique_selectors": 8,
    },
    {
        "name": "test",
        "start": "2022-01-01T00:00:00Z",
        "end_exclusive": "2023-01-01T00:00:00Z",
        "minimum_proposals": 12,
        "minimum_proposals_per_protocol": 4,
        "minimum_actions": 12,
        "minimum_active_calendar_months": 4,
        "minimum_daily_decisions_after_warmup": 350,
        "minimum_unique_targets": 4,
        "minimum_unique_selectors": 4,
    },
    {
        "name": "eval",
        "start": "2023-01-01T00:00:00Z",
        "end_exclusive": "2024-01-01T00:00:00Z",
        "minimum_proposals": 12,
        "minimum_proposals_per_protocol": 4,
        "minimum_actions": 12,
        "minimum_active_calendar_months": 4,
        "minimum_daily_decisions_after_warmup": 350,
        "minimum_unique_targets": 4,
        "minimum_unique_selectors": 4,
    },
)

SOURCE_ONLY_GATES = (
    "source_schema_chronology_reconciliation",
    "dual_transport_canonical_replay",
    "dynamic_abi_and_payload_integrity",
    "canonical_log_and_header_identity",
    "lifecycle_ordering",
    "split_proposal_action_support",
    "cross_protocol_support",
    "structural_vocabulary_diversity",
    "daily_schedule_coverage_and_staleness",
    "future_append_invariance",
    "relation_control_sensitivity",
    "forbidden_access_zero",
)

RELATION_CONTROLS = (
    "protocol_label_swap",
    "within_day_event_order_reverse",
    "text_payload_pair_permutation",
    "ordered_action_permutation",
    "lifecycle_event_rotation",
    "availability_plus_seven_days",
)

FORBIDDEN_COUNTERS = (
    "post_2023_source_rows_read",
    "btc_market_rows_read",
    "funding_rows_read",
    "future_return_rows_read",
    "reward_rows_built",
    "model_rows_built",
    "model_actions_built",
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


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"GIPR timestamp is not UTC: {value}")
    return parsed


def _validate_decision_binding() -> None:
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("GIPR-D1 decision document hash changed")


def _validate_events() -> None:
    expected_names = {
        "proposal_created",
        "proposal_canceled",
        "proposal_queued",
        "proposal_executed",
    }
    if {spec.event for spec in EVENT_SPECS} != expected_names:
        raise RuntimeError("GIPR event vocabulary changed")
    topics = [spec.topic for spec in EVENT_SPECS]
    if len(topics) != len(set(topics)):
        raise RuntimeError("GIPR event topics are not unique")
    for spec in EVENT_SPECS:
        if HASH_PATTERN.fullmatch(spec.topic) is None:
            raise RuntimeError("GIPR event topic is malformed")
        if spec.topic != _event_topic(spec.signature):
            raise RuntimeError("GIPR event topic differs from exact ABI signature")
        if spec.topic_count != 1:
            raise RuntimeError("GIPR events must have only the signature topic")


def _validate_governors() -> None:
    if len(GOVERNORS) != 5:
        raise RuntimeError("GIPR governor roster must contain exactly five contracts")
    identities = {(spec.protocol, spec.generation) for spec in GOVERNORS}
    addresses = [spec.address for spec in GOVERNORS]
    if len(identities) != len(GOVERNORS) or len(set(addresses)) != len(addresses):
        raise RuntimeError("GIPR governor identities are not unique")
    if {spec.protocol for spec in GOVERNORS} != {"compound", "uniswap"}:
        raise RuntimeError("GIPR protocol roster changed")
    for spec in GOVERNORS:
        if ADDRESS_PATTERN.fullmatch(spec.address) is None:
            raise RuntimeError("GIPR governor address is malformed")
        if HASH_PATTERN.fullmatch(spec.first_code_block_hash) is None:
            raise RuntimeError("GIPR first-code block hash is malformed")
        if not START_BOUNDARY_BLOCK <= spec.first_code_block <= LAST_SOURCE_BLOCK:
            raise RuntimeError("GIPR first-code block is outside source envelope")
        _parse_utc(spec.first_code_timestamp)
        if spec.runtime_code_bytes <= 0 or spec.last_source_code_bytes <= 0:
            raise RuntimeError("GIPR runtime bytecode must be non-empty")
        if SHA256_PATTERN.fullmatch(spec.runtime_code_sha256) is None:
            raise RuntimeError("GIPR runtime bytecode SHA is malformed")
        if SHA256_PATTERN.fullmatch(spec.last_source_code_sha256) is None:
            raise RuntimeError("GIPR last-source bytecode SHA is malformed")
    role_addresses = [spec.address for spec in KNOWN_TARGET_ROLES]
    if len(KNOWN_TARGET_ROLES) != 7 or len(set(role_addresses)) != 7:
        raise RuntimeError("GIPR known target-role registry changed")
    if {spec.protocol for spec in KNOWN_TARGET_ROLES} != {"compound", "uniswap"}:
        raise RuntimeError("GIPR known target-role protocols changed")
    for spec in KNOWN_TARGET_ROLES:
        if ADDRESS_PATTERN.fullmatch(spec.address) is None:
            raise RuntimeError("GIPR known target address is malformed")


def _validate_boundaries() -> None:
    start = _parse_utc(SOURCE_START)
    end = _parse_utc(SOURCE_END_EXCLUSIVE)
    if start >= end or START_BOUNDARY_BLOCK >= END_BOUNDARY_BLOCK:
        raise RuntimeError("GIPR source envelope is malformed")
    if LAST_SOURCE_BLOCK != END_BOUNDARY_BLOCK - 1:
        raise RuntimeError("GIPR last source block is not end boundary minus one")
    if YEAR_BOUNDARIES[0].first_block != START_BOUNDARY_BLOCK:
        raise RuntimeError("GIPR start boundary drifted")
    if YEAR_BOUNDARIES[-1].first_block != END_BOUNDARY_BLOCK:
        raise RuntimeError("GIPR end boundary drifted")
    previous_time = None
    previous_block = None
    for boundary in YEAR_BOUNDARIES:
        timestamp = _parse_utc(boundary.timestamp)
        observed = _parse_utc(boundary.observed_block_timestamp)
        if observed < timestamp:
            raise RuntimeError("GIPR observed boundary precedes requested UTC boundary")
        if HASH_PATTERN.fullmatch(boundary.block_hash) is None:
            raise RuntimeError("GIPR year boundary hash is malformed")
        if previous_time is not None and timestamp <= previous_time:
            raise RuntimeError("GIPR year boundaries are not chronological")
        if previous_block is not None and boundary.first_block <= previous_block:
            raise RuntimeError("GIPR year boundary blocks are not increasing")
        previous_time = timestamp
        previous_block = boundary.first_block
    if CONFIRMATION_BLOCKS != 64 or MAX_BLOCK_RANGE > 5_000:
        raise RuntimeError("GIPR finality or replay range guard drifted")


def _validate_splits() -> None:
    if [split["name"] for split in SPLITS] != ["train", "test", "eval"]:
        raise RuntimeError("GIPR split order changed")
    if SPLITS[0]["start"] != SOURCE_START:
        raise RuntimeError("GIPR TRAIN does not start at source start")
    if SPLITS[-1]["end_exclusive"] != SOURCE_END_EXCLUSIVE:
        raise RuntimeError("GIPR EVAL does not end at source end")
    for left, right in zip(SPLITS, SPLITS[1:]):
        if left["end_exclusive"] != right["start"]:
            raise RuntimeError("GIPR splits are not contiguous")
    for split in SPLITS:
        if _parse_utc(split["start"]) >= _parse_utc(split["end_exclusive"]):
            raise RuntimeError("GIPR split interval is malformed")
        integer_fields = [
            value
            for key, value in split.items()
            if key.startswith("minimum_")
        ]
        if not integer_fields or any(
            not isinstance(value, int) or value <= 0 for value in integer_fields
        ):
            raise RuntimeError("GIPR split minimums must be positive integers")


def _validate_grammar() -> None:
    if DECISION_FREQUENCY != "1D" or DECISION_UTC_TIME != "00:00:00":
        raise RuntimeError("GIPR decision cadence drifted")
    if HISTORY_DAYS != 28 or MAX_PROPOSAL_AGE_DAYS != 90:
        raise RuntimeError("GIPR history or stale boundary drifted")
    if MAX_ACTIONS_PER_PROPOSAL != 10:
        raise RuntimeError("GIPR maximum proposal actions drifted")
    limits = (
        MAX_DESCRIPTION_BYTES,
        MAX_SIGNATURE_BYTES,
        MAX_CALLDATA_BYTES_PER_ACTION,
        MAX_CALLDATA_BYTES_PER_PROPOSAL,
    )
    if any(value <= 0 for value in limits):
        raise RuntimeError("GIPR parser bounds must be positive")
    if MAX_CALLDATA_BYTES_PER_PROPOSAL < MAX_CALLDATA_BYTES_PER_ACTION:
        raise RuntimeError("GIPR total calldata bound is below per-action bound")
    if len(SOURCE_ONLY_GATES) != 12 or len(set(SOURCE_ONLY_GATES)) != 12:
        raise RuntimeError("GIPR source gate roster changed")
    if len(RELATION_CONTROLS) != 6 or len(set(RELATION_CONTROLS)) != 6:
        raise RuntimeError("GIPR control roster changed")
    if set(LIFECYCLE_TRANSITIONS) != {
        "proposal_created",
        "proposal_queued",
        "proposal_executed",
        "proposal_canceled",
    }:
        raise RuntimeError("GIPR lifecycle transition states changed")
    permitted = {
        ("proposal_created", "proposal_queued"),
        ("proposal_created", "proposal_canceled"),
        ("proposal_queued", "proposal_executed"),
        ("proposal_queued", "proposal_canceled"),
    }
    observed = {
        (origin, destination)
        for origin, destinations in LIFECYCLE_TRANSITIONS.items()
        for destination in destinations
    }
    if observed != permitted:
        raise RuntimeError("GIPR lifecycle transition matrix changed")


def build_preregistration() -> dict[str, Any]:
    _validate_decision_binding()
    _validate_events()
    _validate_governors()
    _validate_boundaries()
    _validate_splits()
    _validate_grammar()
    core: dict[str, Any] = {
        "protocol_version": "gipr_d1_source_preregistration_v1",
        "candidate": {
            "id": "GIPR-D1",
            "name": "Governance Intent-Payload Relation daily target-position RLLM",
            "source_axis": "ethereum_governance_intent_payload_relation",
            "selection_commit": "cdb8907002736d5c7703232faee3073dd96f2596",
            "correction_commit": "0016296bad08b8101944a6b87fbad184e00d6a59",
        },
        "decision_binding": {
            "path": str(DECISION_PATH),
            "sha256": DECISION_SHA256,
        },
        "official_sources": list(OFFICIAL_SOURCES),
        "source_contract": {
            "chain_id": CHAIN_ID,
            "start": SOURCE_START,
            "end_exclusive": SOURCE_END_EXCLUSIVE,
            "start_boundary": {
                "block_number": START_BOUNDARY_BLOCK,
                "block_hash": START_BOUNDARY_HASH,
                "block_timestamp": START_BOUNDARY_TIMESTAMP,
            },
            "end_boundary_exclusive": {
                "block_number": END_BOUNDARY_BLOCK,
                "block_hash": END_BOUNDARY_HASH,
                "block_timestamp": END_BOUNDARY_TIMESTAMP,
            },
            "last_source_block": LAST_SOURCE_BLOCK,
            "year_boundaries": [asdict(boundary) for boundary in YEAR_BOUNDARIES],
            "confirmation_blocks": CONFIRMATION_BLOCKS,
            "max_block_range": MAX_BLOCK_RANGE,
            "header_batch_size": HEADER_BATCH_SIZE,
            "disk_limit_gib": DISK_LIMIT_GIB,
            "governors": [asdict(governor) for governor in GOVERNORS],
            "known_target_roles": [
                asdict(target) for target in KNOWN_TARGET_ROLES
            ],
            "events": [asdict(event) for event in EVENT_SPECS],
        },
        "bounded_no_log_probe": {
            "transports": 2,
            "provider_hosts_part_of_artifact_identity": False,
            "eth_get_logs_called": False,
            "event_incidence_opened": False,
            "all_year_boundary_headers_equal": True,
            "all_first_code_boundaries_equal": True,
            "all_first_code_runtime_hashes_equal": True,
            "all_last_source_runtime_hashes_equal": True,
            "all_prior_block_code_empty": True,
            "compound_repository_alpha_marker_block": 9_601_447,
            "compound_alpha_observed_first_code_block": 9_601_459,
        },
        "availability_contract": {
            "event_time_field": "canonical event block timestamp",
            "effective_time_field": "canonical block N+64 timestamp",
            "effective_output_field": "available_at",
            "removed_logs_allowed": False,
            "provider_receipt_time_allowed": False,
            "live_finalized_head_required": True,
            "daily_decision_frequency": DECISION_FREQUENCY,
            "daily_decision_utc_time": DECISION_UTC_TIME,
            "decision_rule": "first 00:00 UTC boundary at or after available_at",
            "history_days": HISTORY_DAYS,
            "maximum_proposal_age_days": MAX_PROPOSAL_AGE_DAYS,
            "silent_forward_fill_allowed": False,
            "stale_state": "STALE_OR_NO_PROPOSAL",
            "later_market_rule": "one complete five-minute bar after decision timestamp",
        },
        "parser_contract": {
            "proposal_created_fields": [
                "proposal_id",
                "proposer",
                "targets",
                "values",
                "signatures",
                "calldatas",
                "start_block",
                "end_block",
                "description",
            ],
            "proposal_identity_fields": [
                "governor_address",
                "proposal_id",
            ],
            "lifecycle_transition_matrix": LIFECYCLE_TRANSITIONS,
            "duplicate_lifecycle_events_allowed": False,
            "execute_without_queue_allowed": False,
            "array_lengths_must_match": True,
            "minimum_actions_per_proposal": 1,
            "maximum_actions_per_proposal": MAX_ACTIONS_PER_PROPOSAL,
            "maximum_description_bytes": MAX_DESCRIPTION_BYTES,
            "maximum_signature_bytes": MAX_SIGNATURE_BYTES,
            "maximum_calldata_bytes_per_action": MAX_CALLDATA_BYTES_PER_ACTION,
            "maximum_calldata_bytes_per_proposal": (
                MAX_CALLDATA_BYTES_PER_PROPOSAL
            ),
            "strict_utf8_description_required": True,
            "nul_in_description_allowed": False,
            "empty_signature_allowed": True,
            "empty_calldata_allowed": True,
            "uint256_canonical_bounds_required": True,
            "dynamic_offset_alignment_required": True,
            "dynamic_tail_nonoverlap_required": True,
            "trailing_unparsed_bytes_allowed": False,
        },
        "representation_contract": {
            "deterministic_source_tokens": [
                "protocol",
                "governor_generation",
                "event_type",
                "proposal_age_bucket",
                "lifecycle_age_bucket",
                "action_count_bucket",
                "target_role_or_unknown",
                "signature_or_selector_or_unknown",
                "calldata_validity",
                "abi_shape_class",
                "native_value_presence",
                "description_structure",
                "cross_protocol_relation_persistence_or_flip",
                "current_target_position",
            ],
            "later_model_relation_tokens": [
                "AGREE",
                "PARTIAL",
                "CONFLICT",
                "UNRESOLVED",
            ],
            "raw_numeric_model_inputs_allowed": False,
            "raw_unknown_addresses_model_inputs_allowed": False,
            "raw_block_or_timestamp_model_inputs_allowed": False,
            "raw_price_return_rank_pnl_model_inputs_allowed": False,
            "hidden_reasoning_persisted": False,
            "single_model_only": True,
            "analyzer_trader_pair_allowed": False,
            "model_actions": [
                "TARGET_LONG",
                "TARGET_FLAT",
                "TARGET_SHORT",
            ],
            "model_controls_leverage_size_stop_cost_or_reward": False,
        },
        "split_contract": {
            "splits": list(SPLITS),
            "post_2023_source_policy": "SEALED",
            "market_outcome_policy": "FORBIDDEN until separate evaluator freeze",
        },
        "source_support_contract": {
            "gates_in_order": list(SOURCE_ONLY_GATES),
            "all_gates_required": True,
            "relation_controls": list(RELATION_CONTROLS),
            "minimum_control_changed_fraction": "0.10",
            "minimum_nonempty_description_fraction": "0.95",
            "minimum_unique_description_fraction": "0.90",
            "required_lifecycle_event": "proposal_created",
            "minimum_additional_lifecycle_event_types_per_split": 2,
            "future_append_cutoff": SOURCE_END_EXCLUSIVE,
            "first_failure_action": (
                "REJECT_GIPR_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
            ),
            "source_drop_or_repair_allowed": False,
            "gate_relaxation_after_incidence_allowed": False,
        },
        "forbidden_access_contract": {
            "counters": {name: 0 for name in FORBIDDEN_COUNTERS},
            "network_calls_during_preregistration": 0,
            "governance_event_rows_read": 0,
            "governance_descriptions_read": 0,
            "governance_payloads_read": 0,
            "lifecycle_rows_read": 0,
            "event_incidence_opened": False,
            "btc_or_funding_outcomes_opened": False,
        },
        "next_authorized_step": (
            "implement and seal synthetic-only GIPR-D1 source-support evaluator"
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
            raise RuntimeError("existing GIPR preregistration differs")
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
                "event_incidence_opened": payload[
                    "forbidden_access_contract"
                ]["event_incidence_opened"],
                "next_authorized_step": payload["next_authorized_step"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
