"""Freeze the WCBF Ethereum source contract before full incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/wbtc_custody_bridge_flow_source_protocol_2026-07-23.json"
)
DECISION_PATH = Path(
    "docs/wbtc-custody-bridge-flow-source-axis-decision-2026-07-23.md"
)
DECISION_SHA256 = "b0c1f5017ce6c1327becc9b01e57138f54b7832f369f957b6dcd2676423bc184"
SCRIPT_PATH = Path("training/preregister_wbtc_custody_bridge_flow_source.py")

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
MAX_BLOCK_RANGE = 10_000
HEADER_BATCH_SIZE = 100
RECEIPT_BATCH_SIZE = 25
DISK_LIMIT_GIB = 300

WBTC_ADDRESS = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
WBTC_DECIMALS = 8
MINT_TOPIC = "0x0f6798a560793a54c3bcfe86a93cde1e73087d944c0ea20544137d4121396885"
BURN_TOPIC = "0xcc16f5dbb4873280815c1ee09dbd06736cffcc184412cf7a71a0fdb75d397ca5"
TRANSFER_TOPIC = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)
ZERO_TOPIC = "0x" + "00" * 32
BOUNDARY_CODE_SHA256 = (
    "1377b93c57d7373bf87a742b4783deff88599e7a1bcca58f0a5d6a93e8a2973b"
)

OFFICIAL_SOURCES = (
    "https://docs.wbtc.network/resources/contract-addresses",
    "https://docs.wbtc.network/how-wbtc-works/mint-burn-mechanism",
    "https://docs.wbtc.network/overview/proof-of-reserve-and-transparency-dashboard",
    "https://docs.wbtc.network/resources/api-reference",
    "https://ethereum.org/developers/docs/apis/json-rpc/",
    "https://ethereum.github.io/execution-apis/api/methods/eth_getLogs/",
)

HEX_32_PATTERN = re.compile(r"0x[0-9a-f]{64}", re.ASCII)
ADDRESS_PATTERN = re.compile(r"0x[0-9a-f]{40}", re.ASCII)


@dataclass(frozen=True)
class EventSpec:
    event: str
    event_sign: int
    topic: str
    actor_role: str


EVENT_SPECS = (
    EventSpec("mint", 1, MINT_TOPIC, "recipient"),
    EventSpec("burn", -1, BURN_TOPIC, "burner"),
)

BOUNDED_PROBE = {
    "transports": 2,
    "chain_id": 1,
    "contract_code_bytes": 4582,
    "boundary_code_sha256": BOUNDARY_CODE_SHA256,
    "ranges": [
        {
            "start_block": 11_500_000,
            "end_block": 11_509_999,
            "rows": 1,
            "canonical_sha256": (
                "ca21e69844aaee0726eadd52afdf77339d222e7da975ec7ba86ee8631436ec71"
            ),
        },
        {
            "start_block": 17_020_000,
            "end_block": 17_029_999,
            "rows": 1,
            "canonical_sha256": (
                "176457c19ec92bf7497dcd029cb2ccfddb56b9d1f907a4a3f3e9a65f487c4785"
            ),
        },
        {
            "start_block": 17_030_000,
            "end_block": 17_039_999,
            "rows": 1,
            "canonical_sha256": (
                "ebec75f72a0fa6ef01884eeb1d347d7743257c72a8132ed5c5da6e4be70449d3"
            ),
        },
        {
            "start_block": 17_040_000,
            "end_block": 17_049_999,
            "rows": 1,
            "canonical_sha256": (
                "97da208ecebd983fb2fb5511761ab395725e7c241a944cabcdb62a274531e740"
            ),
        },
    ],
    "receipt_zero_transfer_pairs_verified": 4,
    "btc_market_or_outcomes_opened": False,
}


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


def _validate_contract() -> None:
    if ADDRESS_PATTERN.fullmatch(WBTC_ADDRESS) is None:
        raise RuntimeError("WCBF contract address is malformed")
    topics = [spec.topic for spec in EVENT_SPECS]
    if len(topics) != len(set(topics)):
        raise RuntimeError("WCBF semantic topics are not unique")
    for topic in (*topics, TRANSFER_TOPIC, ZERO_TOPIC):
        if HEX_32_PATTERN.fullmatch(topic) is None:
            raise RuntimeError("WCBF event topic is malformed")
    if {spec.event for spec in EVENT_SPECS} != {"mint", "burn"}:
        raise RuntimeError("WCBF semantic event set is not exactly mint/burn")
    if {spec.event_sign for spec in EVENT_SPECS} != {-1, 1}:
        raise RuntimeError("WCBF semantic signs are not exactly +/-1")
    if WBTC_DECIMALS != 8 or CONFIRMATION_BLOCKS != 64:
        raise RuntimeError("WCBF decimals or confirmation delay drifted")


def _validate_boundaries() -> None:
    start = datetime.fromisoformat(SOURCE_START.replace("Z", "+00:00"))
    end = datetime.fromisoformat(SOURCE_END_EXCLUSIVE.replace("Z", "+00:00"))
    if start.tzinfo != timezone.utc or end.tzinfo != timezone.utc or start >= end:
        raise RuntimeError("WCBF source interval is malformed")
    if START_BOUNDARY_BLOCK >= END_BOUNDARY_BLOCK:
        raise RuntimeError("WCBF block envelope is malformed")
    if LAST_SOURCE_BLOCK != END_BOUNDARY_BLOCK - 1:
        raise RuntimeError("WCBF last source block is not end boundary minus one")
    for value in (START_BOUNDARY_HASH, END_BOUNDARY_HASH):
        if HEX_32_PATTERN.fullmatch(value) is None:
            raise RuntimeError("WCBF boundary block hash is malformed")
    if re.fullmatch(r"[0-9a-f]{64}", BOUNDARY_CODE_SHA256, re.ASCII) is None:
        raise RuntimeError("WCBF boundary bytecode hash is malformed")


def build_protocol() -> dict[str, Any]:
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("WCBF source-axis decision hash changed")
    _validate_contract()
    _validate_boundaries()
    core: dict[str, Any] = {
        "protocol_version": "wbtc_custody_bridge_flow_source_v1",
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
            "confirmation_blocks": CONFIRMATION_BLOCKS,
            "max_block_range": MAX_BLOCK_RANGE,
            "header_batch_size": HEADER_BATCH_SIZE,
            "receipt_batch_size": RECEIPT_BATCH_SIZE,
            "contract_address": WBTC_ADDRESS,
            "decimals": WBTC_DECIMALS,
            "boundary_code_sha256": BOUNDARY_CODE_SHA256,
            "events": [asdict(spec) for spec in EVENT_SPECS],
            "transfer_topic": TRANSFER_TOPIC,
            "zero_topic": ZERO_TOPIC,
        },
        "availability_contract": {
            "event_time_field": "canonical event block timestamp",
            "effective_time_field": "canonical block N+64 timestamp",
            "effective_output_field": "available_at",
            "provider_receipt_time_allowed": False,
            "wbtc_api_date_allowed": False,
            "live_finalized_head_required": True,
            "market_execution_rule": "one complete five-minute bar after available_at",
        },
        "integrity_contract": {
            "independent_log_replays": 2,
            "provider_hosts_part_of_artifact_identity": False,
            "receipt_success_required": True,
            "receipt_block_identity_required": True,
            "semantic_event_zero_transfer_pair_required": True,
            "semantic_event_and_transfer_double_count_allowed": False,
            "boundary_code_must_match": True,
            "removed_logs_allowed": False,
            "duplicate_canonical_identities_allowed": False,
            "zero_amounts_allowed": False,
            "disk_limit_gib": DISK_LIMIT_GIB,
        },
        "source_promotion_gates": {
            "dual_replay_canonical_hash_equal": True,
            "mint_and_burn_present_in_each_calendar_year": True,
            "every_receipt_pair_valid": True,
            "boundary_code_equal_and_frozen": True,
            "every_confirmation_header_canonical": True,
            "finalized_head_covers_last_confirmation": True,
            "any_failure_effect": "REJECT_NO_REPAIR before mechanism or BTC data",
        },
        "bounded_probe": BOUNDED_PROBE,
        "outcome_boundary": {
            "source_only": True,
            "full_source_incidence_opened": False,
            "mechanism_features_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "labels_opened": False,
            "pnl_cagr_mdd_opened": False,
            "post_2023_contract_event_rows_read": 0,
            "post_2023_confirmation_headers_may_be_read": True,
        },
        "later_mechanism_boundary": {
            "standalone_mint_long_burn_short_authorized": False,
            "standalone_reversed_direction_authorized": False,
            "weak_factor_combinations_must_be_preregistered": True,
            "multiplicity_control_required": True,
            "llm_direction_selection_authorized": False,
        },
    }
    return core


def run(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    core = build_protocol()
    core["implementation_binding"] = {
        "path": str(SCRIPT_PATH),
        "sha256": sha256_file(SCRIPT_PATH),
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
