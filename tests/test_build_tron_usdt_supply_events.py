from __future__ import annotations

import csv
import gzip
import io
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence, TypedDict

import pytest
from typing_extensions import Unpack

from training import build_tron_usdt_supply_events as builder


TX_HASH = "0x" + "11" * 32
BLOCK_HASH = (
    "0x0000000002d1f1ce5e430281e5308004cf19dd6e31afd4402b670fc05da5b340"
)
PREVIOUS_HASH = (
    "0x0000000002d1f1cdd4da795414f43e7959c52951203ff7538d43a33db43675b6"
)
CONFIRMATION_PREVIOUS_HASH = "0x" + "31" * 32
CONFIRMATION_HASH = "0x" + "32" * 32
END_PREVIOUS_HASH = (
    "0x0000000004f58c1f11b270096780f20b6de4b249ea8e3f8d3d95a5886ab9be22"
)
END_HASH = (
    "0x0000000004f58c20deab323895309dd25eecc6bbbe4cd6c940713da2d78ca67a"
)
GENERIC_PARENT_HASH = "0x" + "61" * 32
RECIPIENT_TOPIC = "0x" + "0" * 24 + "12" * 20
SENDER_TOPIC = "0x" + "0" * 24 + "34" * 20
AMOUNT = 1_250_000
PRIMARY_URL = f"https://{builder.PRIMARY_HOST}/jsonrpc"
VERIFY_SECRET = "synthetic-chainstack-secret"
VERIFY_URL = f"https://{builder.VERIFY_HOST}/{VERIFY_SECRET}/jsonrpc"


class RawLogOverrides(TypedDict, total=False):
    topic0: str
    topics: list[str] | None
    data: str | None
    block: int
    block_hash: str
    transaction_hash: str
    transaction_index: int
    log_index: int
    address: str
    removed: bool


class ManifestKwargs(TypedDict):
    category_logs: Mapping[str, Sequence[builder.CanonicalLog]]
    replay_chunks: Sequence[tuple[int, int]]
    finalized_head: builder.Header
    boundary_evidence: Mapping[str, Any]
    source_integrity: Mapping[str, int]
    headers: Mapping[int, builder.Header]
    receipts: Mapping[str, builder.Receipt]


def word(value: int) -> str:
    return f"0x{value:064x}"


def raw_log(
    *,
    topic0: str = builder.ISSUE_TOPIC,
    topics: list[str] | None = None,
    data: str | None = None,
    block: int = builder.SOURCE_START_BLOCK,
    block_hash: str = BLOCK_HASH,
    transaction_hash: str = TX_HASH,
    transaction_index: int = 2,
    log_index: int = 3,
    address: str = builder.USDT_CONTRACT,
    removed: bool = False,
) -> dict[str, Any]:
    return {
        "address": address,
        "topics": topics if topics is not None else [topic0],
        "data": word(AMOUNT) if data is None else data,
        "blockNumber": hex(block),
        "transactionHash": transaction_hash,
        "transactionIndex": hex(transaction_index),
        "blockHash": block_hash,
        "logIndex": hex(log_index),
        "removed": removed,
    }


def issue_log(**overrides: Unpack[RawLogOverrides]) -> dict[str, Any]:
    return raw_log(**overrides)


def mint_log(**overrides: Unpack[RawLogOverrides]) -> dict[str, Any]:
    values: RawLogOverrides = {
        "topic0": builder.TRANSFER_TOPIC,
        "topics": [
            builder.TRANSFER_TOPIC,
            builder.ZERO_ADDRESS_TOPIC,
            RECIPIENT_TOPIC,
        ],
        "log_index": 4,
    }
    values.update(overrides)
    return raw_log(**values)


def redeem_log(**overrides: Unpack[RawLogOverrides]) -> dict[str, Any]:
    values: RawLogOverrides = {
        "topic0": builder.REDEEM_TOPIC,
        "log_index": 5,
    }
    values.update(overrides)
    return raw_log(**values)


def burn_log(**overrides: Unpack[RawLogOverrides]) -> dict[str, Any]:
    values: RawLogOverrides = {
        "topic0": builder.TRANSFER_TOPIC,
        "topics": [
            builder.TRANSFER_TOPIC,
            SENDER_TOPIC,
            builder.ZERO_ADDRESS_TOPIC,
        ],
        "log_index": 6,
    }
    values.update(overrides)
    return raw_log(**values)


def header(number: int) -> dict[str, str]:
    start = builder.SOURCE_START_BLOCK
    confirmation = start + builder.CONFIRMATION_BLOCKS
    frozen = {
        item["number"]: item for item in builder.frozen_boundary_header_payload()
    }
    if number in frozen:
        item = frozen[number]
        return {
            "number": hex(item["number"]),
            "hash": item["hash"],
            "parentHash": item["parentHash"],
            "timestamp": hex(item["timestamp"]),
        }
    if number == start - 1:
        block_hash, parent_hash, timestamp = (
            PREVIOUS_HASH,
            builder.FROZEN_BOUNDARIES[0]["previous"]["parentHash"],
            1_672_531_197,
        )
    elif number == start:
        block_hash, parent_hash, timestamp = (
            BLOCK_HASH,
            PREVIOUS_HASH,
            1_672_531_200,
        )
    elif number == confirmation - 1:
        block_hash, parent_hash, timestamp = (
            CONFIRMATION_PREVIOUS_HASH,
            GENERIC_PARENT_HASH,
            1_672_531_390,
        )
    elif number == confirmation:
        block_hash, parent_hash, timestamp = (
            CONFIRMATION_HASH,
            CONFIRMATION_PREVIOUS_HASH,
            1_672_531_393,
        )
    else:
        block_hash, parent_hash, timestamp = (
            "0x" + f"{number:064x}"[-64:],
            GENERIC_PARENT_HASH,
            1_672_531_200 + number - start,
        )
    return {
        "number": hex(number),
        "hash": block_hash,
        "parentHash": parent_hash,
        "timestamp": hex(timestamp),
    }


def finalized_header() -> dict[str, str]:
    return header(builder.LAST_CONFIRMATION_BLOCK)


def receipt(logs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "transactionHash": TX_HASH,
        "transactionIndex": "0x2",
        "blockHash": BLOCK_HASH,
        "blockNumber": hex(builder.SOURCE_START_BLOCK),
        "status": "0x1",
        "logs": [issue_log(), mint_log()] if logs is None else logs,
    }


class FakeRpc:
    def __init__(
        self,
        url: str,
        *,
        semantic: list[dict[str, Any]] | None = None,
        mint: list[dict[str, Any]] | None = None,
        burn: list[dict[str, Any]] | None = None,
        receipt_payload: dict[str, Any] | None = None,
        header_mutation: tuple[int, str, Any] | None = None,
        finalized_payload: dict[str, Any] | None = None,
        refetched_finalized_payload: dict[str, Any] | None = None,
    ) -> None:
        self.url = url
        self.role = builder.sanitize_transport(url)["role"]
        self.semantic = [issue_log()] if semantic is None else semantic
        self.mint = [mint_log()] if mint is None else mint
        self.burn = [] if burn is None else burn
        self.receipt_payload = receipt() if receipt_payload is None else receipt_payload
        self.header_mutation = header_mutation
        self.finalized_payload = finalized_payload
        self.refetched_finalized_payload = refetched_finalized_payload
        self.explicit_header_requests: dict[int, int] = {}
        self.requests: list[tuple[str, tuple[Any, ...]]] = []

    def call(self, method: str, params: Sequence[Any]) -> Any:
        return self.batch(((method, params),))[0]

    def batch(
        self, requests: Sequence[tuple[str, Sequence[Any]]]
    ) -> Sequence[Any]:
        assert len(requests) <= builder.TRANSPORT_MAX_BATCH[self.role]
        output = []
        for method, params in requests:
            frozen_params = tuple(params)
            self.requests.append((method, frozen_params))
            if method == "eth_chainId":
                output.append(builder.CHAIN_ID_HEX)
            elif method == "eth_getLogs":
                query = params[0]
                topics = query["topics"]
                if topics[0] == list(builder.SEMANTIC_TOPICS):
                    output.append(self.semantic)
                elif topics[1] == builder.ZERO_ADDRESS_TOPIC:
                    output.append(self.mint)
                else:
                    output.append(self.burn)
            elif method == "eth_getTransactionReceipt":
                output.append(self.receipt_payload)
            elif method == "eth_getBlockByNumber":
                selector = params[0]
                if selector == "finalized":
                    payload = (
                        finalized_header()
                        if self.finalized_payload is None
                        else dict(self.finalized_payload)
                    )
                else:
                    number = int(selector, 16)
                    payload = header(number)
                    finalized_number = int(
                        (
                            finalized_header()
                            if self.finalized_payload is None
                            else self.finalized_payload
                        )["number"],
                        16,
                    )
                    if number == finalized_number:
                        request_count = (
                            self.explicit_header_requests.get(number, 0) + 1
                        )
                        self.explicit_header_requests[number] = request_count
                        if (
                            request_count > 1
                            and self.refetched_finalized_payload is not None
                        ):
                            payload = dict(self.refetched_finalized_payload)
                if (
                    self.header_mutation is not None
                    and selector != "finalized"
                    and int(selector, 16) == self.header_mutation[0]
                ):
                    payload = dict(payload)
                    payload[self.header_mutation[1]] = self.header_mutation[2]
                output.append(payload)
            else:
                raise AssertionError(f"unexpected method {method}")
        return output


def clients(**right_overrides: Any) -> tuple[FakeRpc, FakeRpc]:
    return (
        FakeRpc(PRIMARY_URL),
        FakeRpc(VERIFY_URL, **right_overrides),
    )


def normalized(raw: dict[str, Any]) -> builder.CanonicalLog:
    return builder.normalize_log(raw)


def sample_rows() -> tuple[dict[str, Any], ...]:
    return (
        {
            column: value
            for column, value in zip(
                builder.CSV_COLUMNS,
                (
                    "Issue",
                    1,
                    "0x" + "12" * 20,
                    str(AMOUNT),
                    builder.SOURCE_START_BLOCK,
                    BLOCK_HASH,
                    TX_HASH,
                    2,
                    3,
                    4,
                    "2023-01-01T00:00:00Z",
                    builder.SOURCE_START_BLOCK + 64,
                    CONFIRMATION_HASH,
                    "2023-01-01T00:03:13Z",
                ),
                strict=True,
            )
        },
    )


def sample_logs() -> dict[str, tuple[builder.CanonicalLog, ...]]:
    return {
        builder.CATEGORY_SEMANTIC: (normalized(issue_log()),),
        builder.CATEGORY_MINT: (normalized(mint_log()),),
        builder.CATEGORY_BURN: (),
    }


def sample_claim_binding() -> dict[str, str]:
    return {
        "claim_commit": "b" * 40,
        "protocol_parent_commit": "a" * 40,
        "sha256": "c" * 64,
    }


def production_generation_bytes() -> tuple[
    bytes, bytes, dict[str, Any], dict[str, str]
]:
    rows = sample_rows()
    csv_bytes = builder.serialize_csv(rows)
    claim_binding = sample_claim_binding()
    manifest = builder.build_manifest(
        rows,
        csv_bytes,
        category_logs=sample_logs(),
        replay_chunks=builder.frozen_chunks(),
        finalized_head=builder.normalize_header(finalized_header()),
        boundary_evidence={
            "outside_before_count": 0,
            "outside_after_maximum_admissible_count": 0,
            "header_count": 12,
            "canonical_header_set_sha256": builder.BOUNDARY_HEADER_SET_SHA256,
            "frozen_header_set_exact": True,
        },
        source_integrity=builder.ZERO_SOURCE_INTEGRITY,
        headers={},
        receipts={},
        protocol_seal={
            "git_head": claim_binding["protocol_parent_commit"],
        },
        claim_binding=claim_binding,
        production=True,
    )
    return (
        csv_bytes,
        builder.serialize_manifest(manifest),
        manifest,
        claim_binding,
    )


def stub_committed_claim(
    monkeypatch: pytest.MonkeyPatch,
    claim_binding: dict[str, str],
    calls: list[str] | None = None,
) -> None:
    def validate(
        transport_identities: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, str], dict[str, str]]:
        assert builder.validate_transport_identities(
            transport_identities
        ) == tuple(dict(item) for item in builder.SANITIZED_TRANSPORTS)
        if calls is not None:
            calls.append("claim")
        return (
            {"git_head": claim_binding["protocol_parent_commit"]},
            dict(claim_binding),
        )

    monkeypatch.setattr(builder, "validate_production_claim", validate)


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def test_frozen_constants_and_chunks_are_exact() -> None:
    assert builder.CHAIN_ID_HEX == "0x2b6653dc"
    assert builder.USDT_CONTRACT_BASE58 == "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
    assert builder.USDT_CONTRACT == "0xa614f803b6fd780986a42c78ec9c7f77e6ded13c"
    chunks = builder.frozen_chunks()
    assert len(chunks) == 7_178
    assert chunks[0] == (47_313_358, 47_318_357)
    assert chunks[-1] == (83_198_358, 83_200_991)
    assert len(chunks[:-1]) == 7_177
    assert all(last - first + 1 == 5_000 for first, last in chunks[:-1])
    assert chunks[-1][1] - chunks[-1][0] + 1 == 2_634
    assert builder.SOURCE_END_BLOCK_EXCLUSIVE == 83_200_992
    assert builder.SOURCE_BLOCK_COUNT == 35_887_634
    assert all(
        left[1] + 1 == right[0] for left, right in zip(chunks, chunks[1:])
    )
    assert builder.LAST_EVENT_BLOCK + 64 == 83_201_055
    assert builder.PRODUCTION_THROTTLE_SECONDS == 0.25


def test_frozen_boundary_header_serialization_is_exact() -> None:
    payload = builder.frozen_boundary_header_payload()
    assert len(payload) == 12
    assert builder._sha256_bytes(builder._canonical_json_bytes(payload)) == (
        "d2513baf86cab444b034ef19079e18515ee8da3d756f4dc041fb1e889707927b"
    )
    assert [payload[index]["number"] for index in range(1, 12, 2)] == [
        47_313_358,
        51_652_374,
        57_811_194,
        68_346_198,
        78_854_231,
        83_201_056,
    ]


def test_protocol_closure_and_decision_hashes_are_frozen() -> None:
    required = {
        builder.SOURCE_DECISION_PATH,
        builder.MECHANISM_DECISION_PATH,
        builder.PREREGISTER_PATH,
        builder.PREREGISTER_TEST_PATH,
        builder.PREREGISTRATION_ARTIFACT_PATH,
        builder.PREREGISTRATION_ARTIFACT_TEST_PATH,
        builder.BUILDER_PATH,
        builder.TEST_PATH,
        builder.SOURCE_SUPPORT_PATH,
        builder.SOURCE_SUPPORT_TEST_PATH,
        builder.NOVELTY_PATH,
        builder.NOVELTY_TEST_PATH,
        builder.ECONOMICS_PATH,
        builder.ECONOMICS_TEST_PATH,
        builder.ESDI_PREREGISTER_HELPER_PATH,
        builder.ESDI_NOVELTY_HELPER_PATH,
        builder.ESDI_SOURCE_BUILDER_HELPER_PATH,
        builder.ESDI_SOURCE_SUPPORT_HELPER_PATH,
        builder.ESDI_ECONOMICS_HELPER_PATH,
    }
    assert required <= set(builder.PROTOCOL_PATHS)
    assert builder._sha256_bytes(
        (builder.REPOSITORY_ROOT / builder.SOURCE_DECISION_PATH).read_bytes()
    ) == builder.SOURCE_DECISION_SHA256
    assert builder._sha256_bytes(
        (builder.REPOSITORY_ROOT / builder.MECHANISM_DECISION_PATH).read_bytes()
    ) == builder.MECHANISM_DECISION_SHA256
    assert builder._sha256_bytes(
        (
            builder.REPOSITORY_ROOT
            / builder.ESDI_ECONOMICS_HELPER_PATH
        ).read_bytes()
    ) == builder.ESDI_ECONOMICS_HELPER_SHA256
    preregistration_bytes = (
        builder.REPOSITORY_ROOT / builder.PREREGISTRATION_ARTIFACT_PATH
    ).read_bytes()
    assert builder._sha256_bytes(preregistration_bytes) == (
        builder.PREREGISTRATION_ARTIFACT_SHA256
    )
    assert json.loads(preregistration_bytes)["manifest_hash"] == (
        builder.PREREGISTRATION_MANIFEST_HASH
    )


def test_synthetic_protocol_identity_is_explicit_and_production_forbidden() -> None:
    paths = (Path("synthetic/a.py"), Path("synthetic/test_a.py"))
    identity = {
        "git_head": "a" * 40,
        "files": {
            path.as_posix(): {
                "git_blob": None,
                "sha256": "b" * 64,
            }
            for path in paths
        },
    }
    seal = builder.current_protocol_seal(
        require_committed=False,
        protocol_paths=paths,
        synthetic_identity=identity,
    )
    assert seal["protocol_paths"] == [path.as_posix() for path in paths]
    assert seal["source_replay_schedule"][
        "inter_batch_throttle_seconds"
    ] == 0.25
    assert seal["source_replay_schedule"]["maximum_batch_by_role"] == {
        "primary": 100,
        "verification": 30,
    }
    assert seal["source_replay_schedule"]["rpc_methods"] == sorted(
        builder.RPC_METHODS
    )
    with pytest.raises(builder.TerminalSourceFailure, match="forbidden"):
        builder.current_protocol_seal(
            require_committed=True,
            protocol_paths=paths,
            synthetic_identity=identity,
        )


def test_category_filters_are_frozen() -> None:
    first, last = builder.frozen_chunks()[0]
    semantic = builder.category_filter(builder.CATEGORY_SEMANTIC, first, last)
    mint = builder.category_filter(builder.CATEGORY_MINT, first, last)
    burn = builder.category_filter(builder.CATEGORY_BURN, first, last)
    assert semantic["topics"] == [list(builder.SEMANTIC_TOPICS)]
    assert mint["topics"] == [
        builder.TRANSFER_TOPIC,
        builder.ZERO_ADDRESS_TOPIC,
    ]
    assert burn["topics"] == [
        builder.TRANSFER_TOPIC,
        None,
        builder.ZERO_ADDRESS_TOPIC,
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("blockNumber", "0x00"),
        ("transactionIndex", 2),
        ("logIndex", "2"),
        ("blockHash", "0x12"),
        ("transactionHash", "0X" + "11" * 32),
        ("address", builder.USDT_CONTRACT.upper()),
        ("data", "0x0"),
        ("topics", [builder.ISSUE_TOPIC[:-1]]),
        ("removed", True),
    ],
)
def test_normalize_log_rejects_malformed_values(field: str, value: Any) -> None:
    payload = issue_log()
    payload[field] = value
    with pytest.raises(builder.TerminalSourceFailure):
        builder.normalize_log(payload)


def test_normalize_log_rejects_extra_field_and_out_of_range() -> None:
    payload = issue_log()
    payload["extra"] = None
    with pytest.raises(builder.TerminalSourceFailure, match="fields"):
        builder.normalize_log(payload)
    with pytest.raises(builder.TerminalSourceFailure, match="precedes"):
        builder.normalize_log(
            issue_log(block=9), first_block=10, last_block=20
        )


def test_parse_quantity_is_canonical() -> None:
    assert builder.parse_quantity("0x0") == 0
    assert builder.parse_quantity("0x2b6653dc") == builder.CHAIN_ID
    for malformed in ("0x", "0x00", "0X1", "0xA", 1, True):
        with pytest.raises(builder.TerminalSourceFailure):
            builder.parse_quantity(malformed)


def test_pair_success_and_blacklist_destruction_stays_separate() -> None:
    destroyed = normalized(
        raw_log(
            topic0=builder.DESTROYED_BLACK_FUNDS_TOPIC,
            topics=[builder.DESTROYED_BLACK_FUNDS_TOPIC],
            data="0x" + "0" * 24 + "34" * 20 + f"{AMOUNT:064x}",
            log_index=7,
        )
    )
    rows = builder.pair_supply_events(
        [normalized(issue_log()), destroyed],
        [normalized(mint_log())],
        [],
    )
    assert [row["event_type"] for row in rows] == [
        "Issue",
        "DestroyedBlackFunds",
    ]
    assert rows[0]["paired_transfer_log_index"] == 4
    assert rows[1]["paired_transfer_log_index"] == ""
    assert rows[0]["supply_direction"] == 1
    assert rows[1]["supply_direction"] == -1
    assert rows[1]["actor_address"] == "0x" + "34" * 20


def test_redeem_pairs_only_with_burn() -> None:
    rows = builder.pair_supply_events(
        [normalized(redeem_log())],
        [],
        [normalized(burn_log())],
    )
    assert rows[0]["event_type"] == "Redeem"
    assert rows[0]["supply_direction"] == -1
    assert rows[0]["paired_transfer_log_index"] == 6
    assert rows[0]["actor_address"] == "0x" + "34" * 20


def test_pairing_rejects_missing_unmatched_and_duplicate_pair() -> None:
    with pytest.raises(builder.TerminalSourceFailure, match="no unique"):
        builder.pair_supply_events([normalized(issue_log())], [], [])
    with pytest.raises(builder.TerminalSourceFailure, match="unmatched"):
        builder.pair_supply_events([], [normalized(mint_log())], [])
    duplicate = normalized(mint_log(log_index=8))
    with pytest.raises(builder.TerminalSourceFailure, match="duplicate candidate"):
        builder.pair_supply_events(
            [normalized(issue_log())],
            [normalized(mint_log()), duplicate],
            [],
        )


def test_deprecate_is_terminal() -> None:
    deprecate = normalized(
        raw_log(topic0=builder.DEPRECATE_TOPIC, data=word(0), log_index=9)
    )
    with pytest.raises(builder.TerminalSourceFailure, match="Deprecate"):
        builder.pair_supply_events([deprecate], [], [])


def test_semantic_abi_data_and_last_event_boundary_are_strict() -> None:
    malformed_data = normalized(issue_log(data="0x"))
    with pytest.raises(builder.TerminalSourceFailure, match="ABI word"):
        builder.pair_supply_events([malformed_data], [], [])
    too_late = normalized(
        issue_log(
            block=builder.LAST_EVENT_BLOCK + 1,
            block_hash="0x" + "ab" * 32,
        )
    )
    with pytest.raises(builder.TerminalSourceFailure, match="last event"):
        builder.pair_supply_events([too_late], [], [])
    with pytest.raises(builder.TerminalSourceFailure, match="positive"):
        builder.pair_supply_events(
            [normalized(issue_log(data=word(0)))],
            [normalized(mint_log(data=word(0)))],
            [],
        )
    malformed_actor = mint_log()
    malformed_actor["topics"] = [
        builder.TRANSFER_TOPIC,
        builder.ZERO_ADDRESS_TOPIC,
        "0x" + "01" + "0" * 62,
    ]
    with pytest.raises(builder.TerminalSourceFailure, match="left-padded"):
        builder.pair_supply_events(
            [normalized(issue_log())], [normalized(malformed_actor)], []
        )


def test_replay_rejects_chunk_gap_and_provider_disagreement() -> None:
    pair = clients()
    with pytest.raises(builder.TerminalSourceFailure, match="gap"):
        builder.replay_log_categories(pair, chunks=((10, 10), (12, 12)))
    disagreeing = clients(semantic=[])
    with pytest.raises(builder.TerminalSourceFailure, match="disagree"):
        builder.replay_log_categories(
            disagreeing,
            chunks=((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),),
        )

    class ShortRpc(FakeRpc):
        def batch(
            self, requests: Sequence[tuple[str, Sequence[Any]]]
        ) -> Sequence[Any]:
            return tuple(super().batch(requests))[:-1]

    short = (ShortRpc(PRIMARY_URL), FakeRpc(VERIFY_URL))
    with pytest.raises(builder.TerminalSourceFailure, match="missing response"):
        builder.replay_log_categories(
            short,
            chunks=((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),),
        )


def test_integrity_evidence_requires_every_executable_stage() -> None:
    audit = builder.SourceIntegrityAudit()
    with pytest.raises(builder.TerminalSourceFailure, match="incomplete"):
        audit.zero_evidence()


def test_receipt_normalization_and_mismatch_fail_closed() -> None:
    normalized_receipt = builder.normalize_receipt(receipt())
    assert normalized_receipt.status == 1
    bad = receipt()
    bad["blockHash"] = "0x" + "99" * 32
    with pytest.raises(builder.TerminalSourceFailure, match="location"):
        builder.normalize_receipt(bad)
    failed = receipt()
    failed["status"] = "0x0"
    with pytest.raises(builder.TerminalSourceFailure, match="successful"):
        builder.normalize_receipt(failed)


def test_materialization_requires_exact_receipt_logs_and_parent_headers() -> None:
    paired = builder.pair_supply_events(
        [normalized(issue_log())], [normalized(mint_log())], []
    )
    number = builder.SOURCE_START_BLOCK
    headers = {
        number - 1: builder.normalize_header(header(number - 1)),
        number: builder.normalize_header(header(number)),
        number + 63: builder.normalize_header(header(number + 63)),
        number + 64: builder.normalize_header(header(number + 64)),
    }
    missing_transfer_receipt = builder.normalize_receipt(receipt([issue_log()]))
    with pytest.raises(builder.TerminalSourceFailure, match="exact pairing"):
        builder.materialize_events(
            paired, headers, {TX_HASH: missing_transfer_receipt}
        )
    orphan_transfer_receipt = builder.normalize_receipt(
        receipt([issue_log(), mint_log(), mint_log(log_index=8)])
    )
    with pytest.raises(builder.TerminalSourceFailure, match="exact pairing"):
        builder.materialize_events(
            paired, headers, {TX_HASH: orphan_transfer_receipt}
        )
    broken = dict(headers)
    confirmation = broken[number + 64]
    broken[number + 64] = builder.Header(
        confirmation.number,
        confirmation.block_hash,
        "0x" + "aa" * 32,
        confirmation.timestamp,
    )
    with pytest.raises(builder.TerminalSourceFailure, match="parent"):
        builder.materialize_events(
            paired, broken, {TX_HASH: builder.normalize_receipt(receipt())}
        )


def test_retrieve_evidence_rejects_header_and_receipt_provider_mismatch() -> None:
    paired = builder.pair_supply_events(
        [normalized(issue_log())], [normalized(mint_log())], []
    )
    header_clients = clients(
        header_mutation=(builder.SOURCE_START_BLOCK, "hash", "0x" + "aa" * 32)
    )
    with pytest.raises(builder.TerminalSourceFailure, match="headers disagree"):
        builder.retrieve_evidence(header_clients, paired)

    changed_receipt = receipt()
    changed_receipt["logs"] = [issue_log(), mint_log(data=word(AMOUNT + 1))]
    receipt_clients = clients(receipt_payload=changed_receipt)
    with pytest.raises(builder.TerminalSourceFailure, match="receipts disagree"):
        builder.retrieve_evidence(receipt_clients, paired)


def test_all_six_frozen_boundaries_are_refetched_and_exact() -> None:
    paired = builder.pair_supply_events(
        [normalized(issue_log())], [normalized(mint_log())], []
    )
    pair = clients()
    _, _, _, evidence = builder.retrieve_evidence(
        pair,
        paired,
        enforce_frozen_boundaries=True,
    )
    assert evidence["header_count"] == 12
    assert evidence["canonical_header_set_sha256"] == (
        builder.BOUNDARY_HEADER_SET_SHA256
    )
    assert evidence["frozen_header_set_exact"] is True
    requested = {
        int(params[0], 16)
        for method, params in pair[0].requests
        if method == "eth_getBlockByNumber" and params[0] != "finalized"
    }
    assert {
        number
        for boundary in builder.FROZEN_BOUNDARIES
        for number in (boundary["number"] - 1, boundary["number"])
    }.issubset(requested)
    finalized_selector = hex(builder.LAST_CONFIRMATION_BLOCK)
    assert sum(
        method == "eth_getBlockByNumber" and params[0] == finalized_selector
        for method, params in pair[0].requests
    ) == 3

    mutation = (
        builder.FROZEN_BOUNDARIES[2]["number"],
        "hash",
        "0x" + "aa" * 32,
    )
    agreeing_but_wrong = (
        FakeRpc(PRIMARY_URL, header_mutation=mutation),
        FakeRpc(VERIFY_URL, header_mutation=mutation),
    )
    with pytest.raises(builder.TerminalSourceFailure, match="frozen boundary"):
        builder.retrieve_evidence(
            agreeing_but_wrong,
            paired,
            enforce_frozen_boundaries=True,
        )

    contradictory_finalized = finalized_header()
    contradictory_finalized["timestamp"] = hex(
        int(contradictory_finalized["timestamp"], 16) + 3
    )
    contradictory = (
        FakeRpc(PRIMARY_URL, finalized_payload=contradictory_finalized),
        FakeRpc(VERIFY_URL, finalized_payload=contradictory_finalized),
    )
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="finalized tag differs",
    ):
        builder.retrieve_evidence(contradictory, paired)

    contradictory_existing = (
        FakeRpc(
            PRIMARY_URL,
            finalized_payload=contradictory_finalized,
            refetched_finalized_payload=contradictory_finalized,
        ),
        FakeRpc(
            VERIFY_URL,
            finalized_payload=contradictory_finalized,
            refetched_finalized_payload=contradictory_finalized,
        ),
    )
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="already fetched header",
    ):
        builder.retrieve_evidence(contradictory_existing, paired)

    later_finalized_number = builder.LAST_CONFIRMATION_BLOCK + 10
    later_finalized = header(later_finalized_number)
    later_pair = (
        FakeRpc(PRIMARY_URL, finalized_payload=later_finalized),
        FakeRpc(VERIFY_URL, finalized_payload=later_finalized),
    )
    later_headers, _, later_head, _ = builder.retrieve_evidence(
        later_pair,
        paired,
    )
    assert later_headers[later_finalized_number] == later_head
    assert sum(
        method == "eth_getBlockByNumber"
        and params[0] == hex(later_finalized_number)
        for method, params in later_pair[0].requests
    ) == 2

    changed_later_finalized = dict(later_finalized)
    changed_later_finalized["timestamp"] = hex(
        int(changed_later_finalized["timestamp"], 16) + 3
    )
    contradictory_later_pair = (
        FakeRpc(
            PRIMARY_URL,
            finalized_payload=later_finalized,
            refetched_finalized_payload=changed_later_finalized,
        ),
        FakeRpc(
            VERIFY_URL,
            finalized_payload=later_finalized,
            refetched_finalized_payload=changed_later_finalized,
        ),
    )
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="independent explicit",
    ):
        builder.retrieve_evidence(contradictory_later_pair, paired)


def test_exact_synthetic_build_is_deterministic_and_never_networks(
    tmp_path: Path,
) -> None:
    chunks = ((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),)
    config = builder.BuildConfig(non_production=True)
    first_clients = clients()
    second_clients = clients()
    computed_a = builder.build_from_clients(
        first_clients, config=config, chunks=chunks
    )
    computed_b = builder.build_from_clients(
        second_clients, config=config, chunks=chunks
    )

    csv_a = computed_a.csv_bytes
    csv_b = computed_b.csv_bytes
    manifest_a = computed_a.manifest
    manifest_b = computed_b.manifest
    assert csv_a == csv_b
    assert computed_a.manifest_bytes == computed_b.manifest_bytes
    assert not list(tmp_path.iterdir())
    assert csv_a[4:8] == b"\x00\x00\x00\x00"
    assert manifest_a == manifest_b
    assert manifest_a["generation_commit"] == (
        builder.SYNTHETIC_GENERATION_COMMIT
    )
    assert manifest_a["source_range"]["start_block_inclusive"] == (
        builder.SOURCE_START_BLOCK
    )
    assert manifest_a["source_range"]["end_block_exclusive"] == (
        builder.SOURCE_START_BLOCK + 1
    )
    assert manifest_a["source_range"]["block_count"] == 1
    assert manifest_a["source_range"]["chunk_count"] == 1
    assert manifest_a["transport_exact_set_equal"] is True
    assert manifest_a["event_counts"] == {
        "Issue": 1,
        "Redeem": 0,
        "DestroyedBlackFunds": 0,
    }
    assert manifest_a["boundary_evidence"]["outside_before_count"] == 0
    assert (
        manifest_a["boundary_evidence"][
            "outside_after_maximum_admissible_count"
        ]
        == 0
    )
    assert manifest_a["boundary_evidence"]["header_count"] == 12
    assert manifest_a["protocol_guards"]["market_policy_performance_opened"] is False
    assert manifest_a["source_replay_schedule"] == {
        "inter_batch_throttle_seconds": 0.25,
        "maximum_batch_by_role": {"primary": 100, "verification": 30},
        "rpc_methods": sorted(builder.RPC_METHODS),
    }
    assert manifest_a["source_integrity"] == builder.ZERO_SOURCE_INTEGRITY
    assert manifest_a["source_csv_sha256"] == builder._sha256_bytes(csv_a)
    assert manifest_a["year_counts"] == {
        "2023": 1,
        "2024": 0,
        "2025": 0,
        "2026": 0,
    }
    assert manifest_a["protocol_parent_commit"]
    assert manifest_a["replay_claim_commit"] is None
    assert manifest_a["replay_claim_sha256"] is None
    assert "csv_sha256" not in manifest_a
    assert "protocol_seal" not in manifest_a
    assert "replay_claim" not in manifest_a
    manifest_raw = computed_a.manifest_bytes.decode("ascii")
    assert VERIFY_SECRET not in manifest_raw
    assert VERIFY_URL not in manifest_raw
    assert manifest_a["transports"][1] == {
        "role": "verification",
        "scheme": "https",
        "hostname": builder.VERIFY_HOST,
        "port": 443,
    }

    with gzip.GzipFile(fileobj=io.BytesIO(csv_a), mode="rb") as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
    assert rows == [
        {
            "event_type": "Issue",
            "supply_direction": "1",
            "actor_address": "0x" + "12" * 20,
            "amount_raw": str(AMOUNT),
            "block_number": str(builder.SOURCE_START_BLOCK),
            "block_hash": BLOCK_HASH,
            "transaction_hash": TX_HASH,
            "transaction_index": "2",
            "log_index": "3",
            "paired_transfer_log_index": "4",
            "event_timestamp_utc": "2023-01-01T00:00:00Z",
            "confirmation_block": str(
                builder.SOURCE_START_BLOCK + builder.CONFIRMATION_BLOCKS
            ),
            "confirmation_block_hash": CONFIRMATION_HASH,
            "available_at_utc": "2023-01-01T00:03:13Z",
        }
    ]
    assert all(
        request[0] in builder.RPC_METHODS
        for client in first_clients
        for request in client.requests
    )


def test_empty_source_and_earlier_last_incidence_are_replay_valid(
) -> None:
    pair = (
        FakeRpc(PRIMARY_URL, semantic=[], mint=[], burn=[]),
        FakeRpc(VERIFY_URL, semantic=[], mint=[], burn=[]),
    )
    computed = builder.build_from_clients(
        pair,
        config=builder.BuildConfig(non_production=True),
        chunks=((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),),
    )
    manifest = computed.manifest
    assert manifest["event_count"] == 0
    assert manifest["event_counts"] == {
        "Issue": 0,
        "Redeem": 0,
        "DestroyedBlackFunds": 0,
    }
    assert not any(
        method == "eth_getTransactionReceipt"
        for client in pair
        for method, _ in client.requests
    )


def test_synthetic_build_requires_explicit_nonproduction_before_any_rpc() -> None:
    pair = clients()
    with pytest.raises(TypeError):
        builder.BuildConfig()  # pyright: ignore[reportCallIssue]
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="explicit non-production",
    ):
        builder.build_from_clients(
            pair,
            config=builder.BuildConfig(non_production=False),
            chunks=((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),),
        )
    assert not any(client.requests for client in pair)
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="committed-claim",
    ):
        builder.build_from_clients(
            pair,
            config=builder.BuildConfig(non_production=True),
            chunks=((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),),
            production=True,
        )
    assert not any(client.requests for client in pair)
    assert set(builder.BuildConfig.__dataclass_fields__) == {"non_production"}


def test_existing_output_and_symlink_are_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        builder.atomic_write_outputs(
            {existing: b"new"}, allowed_root=tmp_path
        )

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(builder.TerminalSourceFailure, match="symlink"):
        builder.atomic_write_outputs(
            {linked / "output": b"new"}, allowed_root=tmp_path
        )


def test_output_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "safe"
    root.mkdir()
    with pytest.raises(builder.TerminalSourceFailure, match="escapes"):
        builder.atomic_write_outputs(
            {tmp_path / "outside": b"x"}, allowed_root=root
        )


def test_atomic_failure_rolls_back_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link
    count = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("synthetic publication failure")
        real_link(source, destination)

    monkeypatch.setattr(builder.os, "link", fail_second)
    first = tmp_path / "a"
    second = tmp_path / "b"
    with pytest.raises(OSError, match="synthetic"):
        builder.atomic_write_outputs(
            {first: b"a", second: b"b"}, allowed_root=tmp_path
        )
    assert not first.exists()
    assert not second.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_generic_writer_cannot_publish_canonical_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="production replay",
    ):
        builder.atomic_write_outputs(
            {builder.DEFAULT_CSV_OUTPUT: b"forbidden"},
            allowed_root=tmp_path,
        )
    csv_bytes, manifest_bytes, _, _ = production_generation_bytes()
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="claim",
    ):
        builder._publish_production_generation(
            csv_bytes,
            manifest_bytes,
            repository_root=tmp_path,
        )
    assert not (tmp_path / builder.DEFAULT_CSV_OUTPUT).exists()


def test_generation_publication_rejects_symlinked_canonical_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_bytes, manifest_bytes, _, claim_binding = production_generation_bytes()
    stub_committed_claim(monkeypatch, claim_binding)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(builder.TerminalSourceFailure, match="symlink"):
        builder._publish_production_generation(
            csv_bytes,
            manifest_bytes,
            repository_root=tmp_path,
        )
    assert not list(outside.iterdir())


def test_manifest_last_publication_is_the_single_generation_commit_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_bytes, manifest_bytes, expected, claim_binding = (
        production_generation_bytes()
    )
    validation_calls: list[str] = []
    stub_committed_claim(monkeypatch, claim_binding, validation_calls)
    published = builder._publish_production_generation(
        csv_bytes,
        manifest_bytes,
        repository_root=tmp_path,
    )
    assert validation_calls == ["claim"]
    assert published == expected
    assert builder.read_committed_generation(repository_root=tmp_path) == expected
    assert (tmp_path / builder.DEFAULT_CSV_OUTPUT).read_bytes() == csv_bytes
    assert (tmp_path / builder.DEFAULT_MANIFEST_OUTPUT).read_bytes() == (
        manifest_bytes
    )
    assert not (tmp_path / builder.GENERATION_STAGE_DIRECTORY).exists()
    assert expected["generation_commit"] == {
        "protocol": "manifest_last_v1",
        "mode": "production",
        "full_envelope_integrity": True,
        "canonical_publication_eligible": True,
        "manifest_is_commit_marker": True,
        "posix_multi_path_atomic": False,
    }


def test_generation_reader_requires_marker_and_matching_csv_hash(
    tmp_path: Path,
) -> None:
    csv_bytes, manifest_bytes, manifest, _ = production_generation_bytes()
    csv_path = tmp_path / builder.DEFAULT_CSV_OUTPUT
    manifest_path = tmp_path / builder.DEFAULT_MANIFEST_OUTPUT
    csv_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    csv_path.write_bytes(csv_bytes)

    unmarked = dict(manifest)
    unmarked.pop("generation_commit")
    unmarked_core = {
        key: value for key, value in unmarked.items() if key != "manifest_hash"
    }
    unmarked["manifest_hash"] = builder._sha256_bytes(
        builder._canonical_json_bytes(unmarked_core)
    )
    manifest_path.write_bytes(builder.serialize_manifest(unmarked))
    assert builder.read_committed_generation(repository_root=tmp_path) is None

    manifest_path.write_bytes(manifest_bytes)
    csv_path.write_bytes(b"tampered")
    with pytest.raises(builder.TerminalSourceFailure, match="CSV hash"):
        builder.read_committed_generation(repository_root=tmp_path)


def test_generation_staging_fsyncs_files_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_bytes, manifest_bytes, _, claim_binding = production_generation_bytes()
    stub_committed_claim(monkeypatch, claim_binding)
    real_fsync = os.fsync
    fsync_kinds: list[str] = []

    def tracking_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        fsync_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(builder.os, "fsync", tracking_fsync)
    builder._publish_production_generation(
        csv_bytes,
        manifest_bytes,
        repository_root=tmp_path,
    )
    assert fsync_kinds.count("file") == 2
    assert fsync_kinds.count("directory") >= 6


def test_catchable_generation_publication_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_bytes, manifest_bytes, _, claim_binding = production_generation_bytes()
    stub_committed_claim(monkeypatch, claim_binding)
    real_link = os.link

    def fail_manifest(source: Path, destination: Path) -> None:
        if destination == tmp_path / builder.DEFAULT_MANIFEST_OUTPUT:
            raise OSError("synthetic manifest publication failure")
        real_link(source, destination)

    monkeypatch.setattr(builder.os, "link", fail_manifest)
    with pytest.raises(OSError, match="synthetic manifest"):
        builder._publish_production_generation(
            csv_bytes,
            manifest_bytes,
            repository_root=tmp_path,
        )
    assert builder.read_committed_generation(repository_root=tmp_path) is None
    assert not (tmp_path / builder.DEFAULT_CSV_OUTPUT).exists()
    assert not (tmp_path / builder.DEFAULT_MANIFEST_OUTPUT).exists()
    assert not (tmp_path / builder.GENERATION_STAGE_DIRECTORY).exists()


@pytest.mark.parametrize("recovery_action", ["finalize", "clean"])
def test_crash_orphan_is_unpublished_and_recovers_without_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_action: str,
) -> None:
    csv_bytes, manifest_bytes, expected, claim_binding = (
        production_generation_bytes()
    )
    stub_committed_claim(monkeypatch, claim_binding)
    real_link = os.link

    def crash_before_marker(source: Path, destination: Path) -> None:
        if destination == tmp_path / builder.DEFAULT_MANIFEST_OUTPUT:
            raise KeyboardInterrupt("synthetic crash")
        real_link(source, destination)

    monkeypatch.setattr(builder.os, "link", crash_before_marker)
    with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
        builder._publish_production_generation(
            csv_bytes,
            manifest_bytes,
            repository_root=tmp_path,
        )
    assert (tmp_path / builder.DEFAULT_CSV_OUTPUT).is_file()
    assert not (tmp_path / builder.DEFAULT_MANIFEST_OUTPUT).exists()
    assert (tmp_path / builder.GENERATION_STAGE_DIRECTORY).is_dir()
    assert builder.read_committed_generation(repository_root=tmp_path) is None

    monkeypatch.setattr(builder.os, "link", real_link)
    recovered = builder._recover_production_generation(
        action=recovery_action,
        repository_root=tmp_path,
    )
    if recovery_action == "finalize":
        assert recovered == expected
        assert (
            builder.read_committed_generation(repository_root=tmp_path)
            == expected
        )
    else:
        assert recovered is None
        assert builder.read_committed_generation(repository_root=tmp_path) is None
        assert not (tmp_path / builder.DEFAULT_CSV_OUTPUT).exists()
        assert not (tmp_path / builder.DEFAULT_MANIFEST_OUTPUT).exists()
    assert not (tmp_path / builder.GENERATION_STAGE_DIRECTORY).exists()


def test_fully_staged_generation_recovers_before_csv_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_bytes, manifest_bytes, expected, claim_binding = (
        production_generation_bytes()
    )
    stub_committed_claim(monkeypatch, claim_binding)
    real_link = os.link

    def crash_before_csv(source: Path, destination: Path) -> None:
        if destination == tmp_path / builder.DEFAULT_CSV_OUTPUT:
            raise KeyboardInterrupt
        real_link(source, destination)

    monkeypatch.setattr(builder.os, "link", crash_before_csv)
    with pytest.raises(KeyboardInterrupt):
        builder._publish_production_generation(
            csv_bytes,
            manifest_bytes,
            repository_root=tmp_path,
        )
    assert not (tmp_path / builder.DEFAULT_CSV_OUTPUT).exists()
    assert builder.read_committed_generation(repository_root=tmp_path) is None

    monkeypatch.setattr(builder.os, "link", real_link)
    assert (
        builder._recover_production_generation(
            action="finalize",
            repository_root=tmp_path,
        )
        == expected
    )
    assert builder.read_committed_generation(repository_root=tmp_path) == expected


def test_replay_finalizes_crash_stage_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_bytes, manifest_bytes, expected, claim_binding = (
        production_generation_bytes()
    )
    stub_committed_claim(monkeypatch, claim_binding)
    real_link = os.link

    def crash_before_marker(source: Path, destination: Path) -> None:
        if destination == tmp_path / builder.DEFAULT_MANIFEST_OUTPUT:
            raise KeyboardInterrupt
        real_link(source, destination)

    monkeypatch.setattr(builder.os, "link", crash_before_marker)
    with pytest.raises(KeyboardInterrupt):
        builder._publish_production_generation(
            csv_bytes,
            manifest_bytes,
            repository_root=tmp_path,
        )
    monkeypatch.setattr(builder.os, "link", real_link)
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv(builder.PRIMARY_RPC_ENV, PRIMARY_URL)
    monkeypatch.setenv(builder.VERIFY_RPC_ENV, VERIFY_URL)
    monkeypatch.setattr(
        builder,
        "HttpJsonRpcClient",
        lambda *args, **kwargs: pytest.fail(
            "recovery constructed a network client"
        ),
    )
    assert builder.replay_production() == expected
    assert builder.read_committed_generation(repository_root=tmp_path) == expected


def test_incomplete_prepublication_stage_can_be_cleaned_without_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_committed_claim(monkeypatch, sample_claim_binding())
    stage = tmp_path / builder.GENERATION_STAGE_DIRECTORY
    stage.mkdir(parents=True)
    (stage / builder.STAGED_CSV_NAME).write_bytes(b"incomplete")
    assert (
        builder._recover_production_generation(
            action="clean",
            repository_root=tmp_path,
        )
        is None
    )
    assert not stage.exists()
    assert builder.read_committed_generation(repository_root=tmp_path) is None


def test_clean_recovery_preserves_unstaged_existing_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_committed_claim(monkeypatch, sample_claim_binding())
    csv_path = tmp_path / builder.DEFAULT_CSV_OUTPUT
    csv_path.parent.mkdir(parents=True)
    csv_path.write_bytes(b"unverified-existing-bytes")
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="lacks a durable recovery stage",
    ):
        builder._recover_production_generation(
            action="clean",
            repository_root=tmp_path,
        )
    assert csv_path.read_bytes() == b"unverified-existing-bytes"
    assert not (tmp_path / builder.DEFAULT_MANIFEST_OUTPUT).exists()


def test_clean_preserves_csv_when_staged_hash_evidence_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_committed_claim(monkeypatch, sample_claim_binding())
    csv_path = tmp_path / builder.DEFAULT_CSV_OUTPUT
    csv_path.parent.mkdir(parents=True)
    csv_path.write_bytes(b"same-bytes")
    stage = tmp_path / builder.GENERATION_STAGE_DIRECTORY
    stage.mkdir(parents=True)
    (stage / builder.STAGED_CSV_NAME).write_bytes(b"same-bytes")
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="complete staged evidence",
    ):
        builder._recover_production_generation(
            action="clean",
            repository_root=tmp_path,
        )
    assert csv_path.read_bytes() == b"same-bytes"
    assert stage.is_dir()


def test_manifest_hash_and_serialization_are_deterministic() -> None:
    logs = sample_logs()
    rows = sample_rows()
    csv_bytes = builder.serialize_csv(rows)
    finalized = builder.normalize_header(finalized_header())
    kwargs: ManifestKwargs = {
        "category_logs": logs,
        "replay_chunks": (
            (builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),
        ),
        "finalized_head": finalized,
        "boundary_evidence": {"start": {}, "end": {}},
        "source_integrity": builder.ZERO_SOURCE_INTEGRITY,
        "headers": {},
        "receipts": {},
    }
    first = builder.build_manifest(rows, csv_bytes, **kwargs)
    second = builder.build_manifest(rows, csv_bytes, **kwargs)
    assert first == second
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == builder._sha256_bytes(
        builder._canonical_json_bytes(core)
    )
    assert builder.serialize_manifest(first) == builder.serialize_manifest(second)
    changed_integrity = dict(builder.ZERO_SOURCE_INTEGRITY)
    changed_integrity["header_differences"] = 1
    changed_kwargs = kwargs.copy()
    changed_kwargs["source_integrity"] = changed_integrity
    with pytest.raises(builder.TerminalSourceFailure, match="zero-difference"):
        builder.build_manifest(
            rows,
            csv_bytes,
            **changed_kwargs,
        )
    binding = sample_claim_binding()
    production_kwargs = kwargs.copy()
    production_kwargs["replay_chunks"] = builder.frozen_chunks()
    production_kwargs["boundary_evidence"] = {
        "outside_before_count": 0,
        "outside_after_maximum_admissible_count": 0,
        "header_count": 12,
        "canonical_header_set_sha256": builder.BOUNDARY_HEADER_SET_SHA256,
        "frozen_header_set_exact": True,
    }
    bound = builder.build_manifest(
        rows,
        csv_bytes,
        **production_kwargs,
        protocol_seal={"git_head": binding["protocol_parent_commit"]},
        claim_binding=binding,
        production=True,
    )
    assert bound["protocol_parent_commit"] == binding["protocol_parent_commit"]
    assert bound["replay_claim_commit"] == binding["claim_commit"]
    assert bound["replay_claim_sha256"] == "c" * 64
    assert bound["generation_commit"] == builder.PRODUCTION_GENERATION_COMMIT
    assert "protocol_seal" not in bound
    assert "replay_claim" not in bound
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="frozen envelope",
    ):
        incomplete_production_kwargs = kwargs.copy()
        incomplete_production_kwargs["boundary_evidence"] = {
            "outside_before_count": 0,
            "outside_after_maximum_admissible_count": 0,
            "header_count": 12,
            "canonical_header_set_sha256": builder.BOUNDARY_HEADER_SET_SHA256,
            "frozen_header_set_exact": True,
        }
        builder.build_manifest(
            rows,
            csv_bytes,
            **incomplete_production_kwargs,
            protocol_seal={"git_head": binding["protocol_parent_commit"]},
            claim_binding=binding,
            production=True,
        )


def test_claim_and_manifest_reject_noncanonical_transport_identities() -> None:
    secret = "must-never-be-serialized"
    valid = [dict(identity) for identity in builder.SANITIZED_TRANSPORTS]
    adversarial: list[list[dict[str, Any]]] = []
    for key in ("url", "path", "query", "userinfo"):
        candidate = [dict(identity) for identity in valid]
        candidate[1][key] = secret
        adversarial.append(candidate)
    adversarial.extend(
        [
            list(reversed([dict(identity) for identity in valid])),
            [
                dict(valid[0]),
                {**valid[1], "port": "443"},
            ],
        ]
    )

    rows = sample_rows()
    csv_bytes = builder.serialize_csv(rows)
    manifest_kwargs: ManifestKwargs = {
        "category_logs": sample_logs(),
        "replay_chunks": (
            (builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),
        ),
        "finalized_head": builder.normalize_header(finalized_header()),
        "boundary_evidence": {"synthetic": True},
        "source_integrity": builder.ZERO_SOURCE_INTEGRITY,
        "headers": {},
        "receipts": {},
    }
    for identities in adversarial:
        with pytest.raises(
            builder.TerminalSourceFailure,
            match="frozen schema",
        ) as claim_error:
            builder._claim_payload(
                {"git_head": "a" * 40},
                identities,
            )
        with pytest.raises(
            builder.TerminalSourceFailure,
            match="frozen schema",
        ) as manifest_error:
            builder.build_manifest(
                rows,
                csv_bytes,
                **manifest_kwargs,
                transport_identities=identities,
            )
        assert secret not in str(claim_error.value)
        assert secret not in str(manifest_error.value)


def test_production_claim_is_write_once_and_binds_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    seal = {"seal_hash": "ab" * 32}
    identities = (
        builder.sanitize_transport(PRIMARY_URL, expected_role="primary"),
        builder.sanitize_transport(VERIFY_URL, expected_role="verification"),
    )
    binding = builder.create_production_claim(
        seal, identities, repository_root=tmp_path
    )
    claim_path = tmp_path / builder.REPLAY_CLAIM_PATH
    claim_raw = claim_path.read_text()
    payload = json.loads(claim_raw)
    assert payload["status"] == "claim_created_for_committed_replay"
    assert payload["protocol_seal"] == seal
    assert payload["transports"] == list(builder.SANITIZED_TRANSPORTS)
    assert payload["source_replay_schedule"] == {
        "inter_batch_throttle_seconds": 0.25,
        "maximum_batch_by_role": {"primary": 100, "verification": 30},
    }
    assert payload["rpc_methods"] == sorted(builder.RPC_METHODS)
    assert payload["source_envelope"]["final_chunk"] == [
        83_198_358,
        83_200_991,
    ]
    assert payload["source_envelope"]["block_count"] == 35_887_634
    assert all(value == 0 for value in payload["access_at_claim_creation"].values())
    claim_core = {
        key: value for key, value in payload.items() if key != "claim_hash"
    }
    assert payload["claim_hash"] == builder._sha256_bytes(
        builder._canonical_json_bytes(claim_core)
    )
    assert VERIFY_SECRET not in claim_raw
    assert VERIFY_URL not in claim_raw
    assert binding["sha256"] == builder._sha256_bytes(claim_path.read_bytes())
    with pytest.raises(FileExistsError):
        builder.create_production_claim(
            seal, identities, repository_root=tmp_path
        )


def test_claim_only_commit_parent_upstream_and_protocol_blobs_validate(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    remote = tmp_path / "remote.git"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "Synthetic Test")
    git(repository, "config", "user.email", "synthetic@example.invalid")
    paths = (Path("protocol/builder.py"), Path("protocol/test_builder.py"))
    for index, relative in enumerate(paths):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"value = {index}\n")
    git(repository, "add", "--", *(path.as_posix() for path in paths))
    git(repository, "commit", "-m", "protocol parent")
    parent = git(repository, "rev-parse", "HEAD")
    git(repository, "remote", "add", "origin", str(remote))
    git(repository, "push", "-u", "origin", "main")

    seal = builder.current_protocol_seal(
        require_committed=True,
        repository_root=repository,
        protocol_paths=paths,
    )
    identities = (
        builder.sanitize_transport(PRIMARY_URL, expected_role="primary"),
        builder.sanitize_transport(VERIFY_URL, expected_role="verification"),
    )
    builder.create_production_claim(
        seal,
        identities,
        repository_root=repository,
    )
    git(repository, "add", "--", builder.REPLAY_CLAIM_PATH.as_posix())
    git(repository, "commit", "-m", "claim only")
    claim_commit = git(repository, "rev-parse", "HEAD")
    git(repository, "push", "origin", "main")

    validated_seal, binding = builder.validate_production_claim(
        identities,
        repository_root=repository,
        protocol_paths=paths,
    )
    assert validated_seal == seal
    assert binding["protocol_parent_commit"] == parent
    assert binding["claim_commit"] == claim_commit
    assert git(repository, "rev-list", "--parents", "-n", "1", "HEAD").split() == [
        claim_commit,
        parent,
    ]

    (repository / paths[0]).write_text("value = 99\n")
    with pytest.raises(builder.TerminalSourceFailure, match="HEAD-clean"):
        builder.validate_production_claim(
            identities,
            repository_root=repository,
            protocol_paths=paths,
        )


def test_create_claim_only_never_constructs_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    monkeypatch.setattr(
        builder,
        "current_protocol_seal",
        lambda require_committed=True: order.append("seal") or {"seal": True},
    )
    monkeypatch.setattr(
        builder,
        "create_production_claim",
        lambda seal, identities: order.append("claim") or {"claim": True},
    )
    monkeypatch.setenv(builder.PRIMARY_RPC_ENV, PRIMARY_URL)
    monkeypatch.setenv(builder.VERIFY_RPC_ENV, VERIFY_URL)

    monkeypatch.setattr(
        builder,
        "HttpJsonRpcClient",
        lambda *args, **kwargs: pytest.fail("claim path constructed an RPC client"),
    )
    assert builder.create_claim_only() == {"claim": True}
    assert order == ["seal", "claim"]


def test_replay_validation_finishes_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setenv(builder.PRIMARY_RPC_ENV, PRIMARY_URL)
    monkeypatch.setenv(builder.VERIFY_RPC_ENV, VERIFY_URL)

    def reject_claim(*args: Any, **kwargs: Any) -> None:
        order.append("validate")
        raise builder.TerminalSourceFailure("synthetic invalid committed claim")

    monkeypatch.setattr(builder, "validate_production_claim", reject_claim)
    monkeypatch.setattr(
        builder,
        "HttpJsonRpcClient",
        lambda *args, **kwargs: order.append("client"),
    )
    with pytest.raises(builder.TerminalSourceFailure, match="committed claim"):
        builder.replay_production()
    assert order == ["validate"]


def test_replay_constructs_clients_with_exact_frozen_throttle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []
    order: list[str] = []
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv(builder.PRIMARY_RPC_ENV, PRIMARY_URL)
    monkeypatch.setenv(builder.VERIFY_RPC_ENV, VERIFY_URL)
    monkeypatch.setattr(
        builder,
        "validate_production_claim",
        lambda identities, **kwargs: order.append("claim")
        or ({"seal": True}, {"claim": True}),
    )

    class StopAfterConstruction(Exception):
        pass

    def client_constructor(
        url: str, *, throttle_seconds: float
    ) -> None:
        order.append("client")
        observed.append(throttle_seconds)
        raise StopAfterConstruction

    monkeypatch.setattr(builder, "HttpJsonRpcClient", client_constructor)
    with pytest.raises(StopAfterConstruction):
        builder.replay_production()
    assert order == ["claim", "claim", "client"]
    assert observed == [builder.PRODUCTION_THROTTLE_SECONDS] == [0.25]


def test_replay_rejects_nonfrozen_schedule_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv(builder.PRIMARY_RPC_ENV, PRIMARY_URL)
    monkeypatch.setenv(builder.VERIFY_RPC_ENV, VERIFY_URL)
    monkeypatch.setattr(
        builder,
        "validate_production_claim",
        lambda identities, **kwargs: order.append("claim")
        or ({"seal": True}, {"claim": True}),
    )
    monkeypatch.setattr(
        builder,
        "frozen_chunks",
        lambda: ((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),),
    )
    monkeypatch.setattr(
        builder,
        "HttpJsonRpcClient",
        lambda *args, **kwargs: order.append("client"),
    )
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="frozen envelope",
    ):
        builder.replay_production()
    assert order == ["claim"]


def test_http_client_rejects_unfrozen_transport_and_oversized_batch() -> None:
    with pytest.raises(ValueError):
        builder.HttpJsonRpcClient("https://example.invalid")
    with pytest.raises(ValueError):
        builder.sanitize_transport(
            f"https://user:password@{builder.VERIFY_HOST}/secret/jsonrpc"
        )
    with pytest.raises(ValueError):
        builder.sanitize_transport(f"{VERIFY_URL}?token=leak")
    with pytest.raises(ValueError):
        builder.sanitize_transport(
            f"https://{builder.VERIFY_HOST}:444/secret/jsonrpc"
        )
    with pytest.raises(ValueError):
        builder.sanitize_transport(
            f"https://{builder.VERIFY_HOST.upper()}/secret/jsonrpc"
        )
    with pytest.raises(ValueError):
        builder.sanitize_transport(f"https://{builder.VERIFY_HOST}/")
    assert builder.sanitize_transport(VERIFY_URL) == {
        "role": "verification",
        "scheme": "https",
        "hostname": builder.VERIFY_HOST,
        "port": 443,
    }
    client = builder.HttpJsonRpcClient(VERIFY_URL)
    with pytest.raises(builder.TerminalSourceFailure, match="exceeds"):
        client.batch(
            [("eth_chainId", ())] * (builder.TRANSPORT_MAX_BATCH[client.role] + 1)
        )
    with pytest.raises(builder.TerminalSourceFailure, match="network-capable"):
        builder.build_from_clients(
            (
                builder.HttpJsonRpcClient(PRIMARY_URL),
                builder.HttpJsonRpcClient(VERIFY_URL),
            ),
            config=builder.BuildConfig(non_production=True),
            chunks=((builder.SOURCE_START_BLOCK, builder.SOURCE_START_BLOCK),),
        )
    with pytest.raises(SystemExit):
        builder.main(["--throttle-seconds", "0"])


def test_http_client_uses_exact_fixed_inter_batch_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    client = builder.HttpJsonRpcClient(
        PRIMARY_URL,
        throttle_seconds=builder.PRODUCTION_THROTTLE_SECONDS,
    )

    def fake_post(payload: bytes) -> bytes:
        requests = json.loads(payload)
        return builder._canonical_json_bytes(
            [
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": builder.CHAIN_ID_HEX,
                }
                for request in requests
            ]
        )

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(builder.time, "sleep", sleeps.append)
    assert client.call("eth_chainId", ()) == builder.CHAIN_ID_HEX
    assert client.call("eth_chainId", ()) == builder.CHAIN_ID_HEX
    assert sleeps == [0.25]


def test_transport_failure_does_not_leak_runtime_url_or_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = builder.HttpJsonRpcClient(VERIFY_URL)

    class FailingOpener:
        def open(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(VERIFY_URL)

    monkeypatch.setattr(client, "_opener", FailingOpener())
    with pytest.raises(builder.TerminalSourceFailure) as captured:
        client.call("eth_chainId", ())
    assert VERIFY_SECRET not in str(captured.value)
    assert VERIFY_URL not in str(captured.value)
    assert captured.value.__cause__ is None

    monkeypatch.setattr(
        client,
        "_post",
        lambda payload: VERIFY_URL.encode("utf-8"),
    )
    with pytest.raises(builder.TerminalSourceFailure) as malformed:
        client.call("eth_chainId", ())
    assert VERIFY_SECRET not in str(malformed.value)
    assert VERIFY_URL not in str(malformed.value)
    assert malformed.value.__cause__ is None
