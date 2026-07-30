from __future__ import annotations

from decimal import Decimal
import gzip
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

import pytest

from training import build_ethereum_settlement_demand_impulse_source as builder
from training import preregister_ethereum_settlement_demand_impulse as prereg


def _hash(seed: int) -> str:
    return "0x" + f"{seed:064x}"


def _header(number: int, timestamp: int) -> dict[str, Any]:
    return {
        "number": hex(number),
        "hash": _hash(number),
        "parentHash": _hash(number - 1),
        "timestamp": hex(timestamp),
    }


def _fee_history(
    start: int,
    count: int,
    *,
    fee: Callable[[int], int] = lambda number: number + 1,
    ratio: str = "0.5",
) -> dict[str, Any]:
    return {
        "oldestBlock": hex(start),
        "baseFeePerGas": [hex(fee(number)) for number in range(start, start + count + 1)],
        "gasUsedRatio": [Decimal(ratio) for _ in range(count)],
    }


class FakeClient:
    def __init__(
        self,
        url: str,
        *,
        mutate: Callable[[str, Sequence[Any], Any], Any] | None = None,
    ) -> None:
        self.url = url
        self.mutate = mutate
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def call(self, method: str, params: Sequence[Any]) -> Any:
        self.calls.append((method, tuple(params)))
        if method == "eth_chainId":
            result: Any = "0x1"
        elif method == "eth_getBlockByNumber":
            selector = params[0]
            if selector == "finalized":
                result = _header(builder.LAST_CONFIRMATION_BLOCK + 10, 2_000_000_000)
            else:
                number = int(selector, 16)
                result = _header(number, 1_600_000_000 + number)
        elif method == "eth_feeHistory":
            count = int(params[0], 16)
            end = int(params[1], 16)
            result = _fee_history(end - count + 1, count)
        else:
            raise AssertionError(f"unexpected method {method}")
        return self.mutate(method, params, result) if self.mutate else result


def _clients(
    mutate_second: Callable[[str, Sequence[Any], Any], Any] | None = None,
) -> tuple[FakeClient, FakeClient]:
    return (
        FakeClient(builder.TRANSPORTS[0]),
        FakeClient(builder.TRANSPORTS[1], mutate=mutate_second),
    )


def _init_protocol_repository(root: Path) -> None:
    for index, relative in enumerate(builder.PROTOCOL_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protocol file {index}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "protocol seal"],
        check=True,
    )


def test_frozen_constants_and_preregistration_binding() -> None:
    assert builder.TRANSPORTS == (
        "https://eth-mainnet.public.blastapi.io",
        "https://eth.merkle.io",
    )
    assert builder.RPC_METHODS == (
        "eth_chainId",
        "eth_getBlockByNumber",
        "eth_feeHistory",
    )
    assert builder.REQUEST_CHUNK_BLOCKS == 1_024
    assert builder.REQUEST_COUNT == 8_698
    assert builder.FIRST_REQUESTED_BLOCK == 16_311_600
    assert builder.LAST_RETAINED_BLOCK == 25_217_999
    assert builder.LAST_REQUESTED_BLOCK == 25_218_351
    assert builder.TERMINAL_PADDING_BLOCKS == 352
    assert (
        builder.FIRST_REQUESTED_BLOCK
        + builder.REQUEST_COUNT * builder.REQUEST_CHUNK_BLOCKS
        - 1
        == builder.LAST_REQUESTED_BLOCK
    )
    assert builder.LAST_REQUESTED_BLOCK < builder.PRE_2026_06_BOUNDARY_BLOCK
    assert builder.EPOCH_COUNT == 2_474
    assert builder.BOUNDARY_HEADER_REQUESTS == 10
    assert builder.EPOCH_HEADER_REQUESTS == 4_948
    assert builder.TOTAL_RPC_REQUESTS_PER_TRANSPORT == 13_658
    assert builder.PREREGISTRATION_SHA256 == (
        "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
    )
    assert builder.PREREGISTRATION_MANIFEST_HASH == (
        "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
    )
    payload = builder.validate_preregistration()
    assert payload["manifest_hash"] == builder.PREREGISTRATION_MANIFEST_HASH
    assert payload["frozen_preregistration"]["repository_identity"]["sha256"][
        builder.PREREGISTER_EXECUTABLE_PATH.as_posix()
    ] == builder._sha256_file(builder.PREREGISTER_EXECUTABLE_PATH)


@pytest.mark.parametrize(
    "value",
    ["0x00", "0x01", "0X1", "0xA", "1", "", None, True, 1],
)
def test_quantities_must_be_exact_canonical_hex(value: Any) -> None:
    with pytest.raises(builder.TerminalSourceFailure, match="canonical quantity"):
        builder._parse_quantity(value, "test")
    assert builder._parse_quantity("0x0", "test") == 0
    assert builder._parse_quantity("0xabcdef", "test") == 0xABCDEF


@pytest.mark.parametrize(
    "value",
    [0.5, Decimal("NaN"), Decimal("Infinity"), "-0.1", "1.1", ".5", "00", object()],
)
def test_gas_ratios_are_canonical_exact_decimal_and_bounded(value: Any) -> None:
    with pytest.raises(builder.TerminalSourceFailure):
        builder._parse_gas_ratio(value)
    assert builder._parse_gas_ratio(Decimal("0.5000")) == (Decimal("0.5000"), "0.5")
    assert builder._parse_gas_ratio("1") == (Decimal(1), "1")
    assert builder._parse_gas_ratio(Decimal("1.000")) == (Decimal("1.000"), "1")


def test_http_rpc_uses_one_attempt_and_parses_decimal_without_float() -> None:
    class Response(io.BytesIO):
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

    class Opener:
        def __init__(self, body: bytes | None = None) -> None:
            self.body = body
            self.calls = 0

        def open(self, request: Any, *, timeout: float) -> Response:
            self.calls += 1
            assert request.full_url == builder.TRANSPORTS[0]
            assert timeout == 3
            if self.body is None:
                raise OSError("synthetic transport failure")
            return Response(self.body)

    client = builder.HttpJsonRpcClient(
        builder.TRANSPORTS[0], timeout_seconds=3
    )
    success = Opener(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"ratio": 0.125},
            }
        ).encode()
    )
    client._opener = success
    result = client.call("eth_feeHistory", ("0x1", "0x1", []))
    assert result == {"ratio": Decimal("0.125")}
    assert success.calls == 1

    failed = Opener()
    client._opener = failed
    with pytest.raises(builder.TerminalSourceFailure, match="single RPC attempt"):
        client.call("eth_chainId", ())
    assert failed.calls == 1


def test_fee_history_requires_exact_shape_and_imported_vector_hash() -> None:
    result = _fee_history(100, 4)
    chunk = builder.normalize_fee_history(result, expected_start=100, block_count=4)
    assert chunk.oldest_block == 100
    assert chunk.base_fees == (101, 102, 103, 104, 105)
    assert all(isinstance(value, Decimal) for value in chunk.gas_ratios)

    for field, replacement, message in (
        ("oldestBlock", "0x65", "oldest"),
        ("baseFeePerGas", result["baseFeePerGas"][:-1], "shortened"),
        ("gasUsedRatio", result["gasUsedRatio"][:-1], "shortened"),
    ):
        with pytest.raises(builder.TerminalSourceFailure, match=message):
            builder.normalize_fee_history(
                {**result, field: replacement}, expected_start=100, block_count=4
            )

    values = list(range(1, prereg.EPOCH_SIZE_BLOCKS + 1))
    row = builder.build_epoch_row(
        epoch_id=builder.FIRST_EPOCH_ID,
        base_fees=values,
        gas_ratios=[Decimal("0.5")] * prereg.EPOCH_SIZE_BLOCKS,
        end_header=builder.Header(
            prereg.epoch_blocks(builder.FIRST_EPOCH_ID)[1],
            _hash(1),
            _hash(0),
            100,
        ),
        confirmation_header=builder.Header(
            prereg.epoch_blocks(builder.FIRST_EPOCH_ID)[2],
            _hash(2),
            _hash(1),
            200,
        ),
    )
    assert row["median_base_fee_wei_x2"] == 3_601
    assert row["base_fee_vector_sha256"] == prereg.base_fee_vector_sha256(values)
    assert row["mean_gas_used_ratio_decimal"] == "0.5"


def test_fee_history_agreement_is_exact_only_for_normalized_consumed_fields() -> None:
    def primary_optional(
        method: str, params: Sequence[Any], result: Any
    ) -> Any:
        if method != "eth_feeHistory":
            return result
        count = len(result["gasUsedRatio"])
        return {
            **result,
            "reward": [["0x1"] for _ in range(count)],
            "baseFeePerBlobGas": ["0x1"] * (count + 1),
            "blobGasUsedRatio": [Decimal("0.2500")] * count,
        }

    clients = (
        FakeClient(builder.TRANSPORTS[0], mutate=primary_optional),
        FakeClient(builder.TRANSPORTS[1]),
    )
    chunk = builder._fee_history_pair(clients, start=100, block_count=4)
    assert chunk.gas_ratios == (Decimal("0.5"),) * 4

    def decimal_variant(
        method: str, params: Sequence[Any], result: Any
    ) -> Any:
        if method == "eth_feeHistory":
            return {
                **result,
                "gasUsedRatio": [Decimal("0.5000")] * 4,
            }
        return result

    normalized = builder._fee_history_pair(
        (
            FakeClient(builder.TRANSPORTS[0]),
            FakeClient(builder.TRANSPORTS[1], mutate=decimal_variant),
        ),
        start=100,
        block_count=4,
    )
    assert normalized.gas_ratio_strings == ("0.5",) * 4

    def unknown(method: str, params: Sequence[Any], result: Any) -> Any:
        return {**result, "futureField": []} if method == "eth_feeHistory" else result

    forward_compatible = builder._fee_history_pair(
        (
            FakeClient(builder.TRANSPORTS[0], mutate=unknown),
            FakeClient(builder.TRANSPORTS[1]),
        ),
        start=100,
        block_count=4,
    )
    assert forward_compatible == builder.normalize_fee_history(
        _fee_history(100, 4), expected_start=100, block_count=4
    )

    def short_optional(
        method: str, params: Sequence[Any], result: Any
    ) -> Any:
        if method == "eth_feeHistory":
            return {**result, "baseFeePerBlobGas": ["0x1"]}
        return result

    optional_difference = builder._fee_history_pair(
        (
            FakeClient(builder.TRANSPORTS[0], mutate=short_optional),
            FakeClient(builder.TRANSPORTS[1]),
        ),
        start=100,
        block_count=4,
    )
    assert optional_difference.base_fees == (101, 102, 103, 104, 105)


def test_dual_fee_history_is_exact_and_one_call_per_transport() -> None:
    clients = _clients()
    chunk = builder._fee_history_pair(clients, start=100, block_count=4)
    assert chunk.oldest_block == 100
    assert [client.calls for client in clients] == [
        [("eth_feeHistory", ("0x4", "0x67", []))],
        [("eth_feeHistory", ("0x4", "0x67", []))],
    ]

    def disagree(method: str, params: Sequence[Any], result: Any) -> Any:
        if method == "eth_feeHistory":
            result = dict(result)
            result["baseFeePerGas"] = list(result["baseFeePerGas"])
            result["baseFeePerGas"][2] = "0x1"
        return result

    with pytest.raises(builder.TerminalSourceFailure, match="disagree"):
        builder._fee_history_pair(_clients(disagree), start=100, block_count=4)


def test_adjacent_chunk_next_base_fee_overlap_is_exact() -> None:
    first = builder.normalize_fee_history(
        _fee_history(100, 4), expected_start=100, block_count=4
    )
    second = builder.normalize_fee_history(
        _fee_history(104, 4), expected_start=104, block_count=4
    )
    previous = builder.validate_next_base_fee_overlap(None, first)
    assert previous == first.base_fees[-1] == second.base_fees[0]
    assert builder.validate_next_base_fee_overlap(previous, second) == (
        second.base_fees[-1]
    )
    bad = builder.FeeHistoryChunk(
        oldest_block=second.oldest_block,
        base_fees=(1, *second.base_fees[1:]),
        gas_ratios=second.gas_ratios,
        gas_ratio_strings=second.gas_ratio_strings,
    )
    with pytest.raises(builder.TerminalSourceFailure, match="overlap"):
        builder.validate_next_base_fee_overlap(previous, bad)


def test_chain_boundary_parent_timestamp_hash_and_finalized_checks() -> None:
    clients = _clients()
    builder.validate_chain_ids(clients)

    boundary = builder.FROZEN_BOUNDARIES[0]
    number = boundary["first_block_at_or_after"]
    threshold = int(
        builder.datetime.fromisoformat(
            boundary["utc"].replace("Z", "+00:00")
        ).timestamp()
    )

    def boundary_result(
        method: str, params: Sequence[Any], result: Any
    ) -> Any:
        if method != "eth_getBlockByNumber":
            return result
        if params[0] == hex(number - 1):
            return {
                **_header(number - 1, threshold - 1),
                "hash": _hash(999),
            }
        if params[0] == hex(number):
            return {
                **_header(number, threshold),
                "hash": boundary["hash"],
                "parentHash": _hash(999),
            }
        return result

    paired = (
        FakeClient(builder.TRANSPORTS[0], mutate=boundary_result),
        FakeClient(builder.TRANSPORTS[1], mutate=boundary_result),
    )
    # The remaining frozen boundaries intentionally fail with synthetic hashes;
    # isolate the exact relation checker to the first boundary.
    original = builder.FROZEN_BOUNDARIES
    try:
        builder.FROZEN_BOUNDARIES = (boundary,)
        audit = builder.validate_boundaries(paired)
    finally:
        builder.FROZEN_BOUNDARIES = original
    assert audit[0]["parent_relation_exact"] is True
    finalized = builder.validate_common_finalized_head(clients)
    assert finalized.number >= builder.LAST_CONFIRMATION_BLOCK

    def low_finalized(method: str, params: Sequence[Any], result: Any) -> Any:
        if method == "eth_getBlockByNumber" and params[0] == "finalized":
            return _header(builder.LAST_CONFIRMATION_BLOCK - 1, 2_000_000_000)
        return result

    with pytest.raises(builder.TerminalSourceFailure, match="below"):
        builder.validate_common_finalized_head(
            (
                FakeClient(builder.TRANSPORTS[0], mutate=low_finalized),
                FakeClient(builder.TRANSPORTS[1], mutate=low_finalized),
            )
        )


def test_provider_header_disagreement_and_boundary_relation_fail_closed() -> None:
    def disagree(method: str, params: Sequence[Any], result: Any) -> Any:
        if method == "eth_getBlockByNumber":
            return {**result, "timestamp": hex(int(result["timestamp"], 16) + 1)}
        return result

    with pytest.raises(builder.TerminalSourceFailure, match="headers disagree"):
        builder._header_pair(_clients(disagree), "0x64", expected_number=100)

    boundary = builder.FROZEN_BOUNDARIES[0]
    number = boundary["first_block_at_or_after"]
    threshold = int(
        builder.datetime.fromisoformat(
            boundary["utc"].replace("Z", "+00:00")
        ).timestamp()
    )

    def bad_relation(method: str, params: Sequence[Any], result: Any) -> Any:
        if method == "eth_getBlockByNumber" and params[0] == hex(number):
            return {
                **_header(number, threshold),
                "hash": boundary["hash"],
                "parentHash": _hash(123),
            }
        if method == "eth_getBlockByNumber" and params[0] == hex(number - 1):
            return _header(number - 1, threshold - 1)
        return result

    paired = (
        FakeClient(builder.TRANSPORTS[0], mutate=bad_relation),
        FakeClient(builder.TRANSPORTS[1], mutate=bad_relation),
    )
    original = builder.FROZEN_BOUNDARIES
    try:
        builder.FROZEN_BOUNDARIES = (boundary,)
        with pytest.raises(builder.TerminalSourceFailure, match="parent"):
            builder.validate_boundaries(paired)
    finally:
        builder.FROZEN_BOUNDARIES = original


def test_block_provider_extras_are_ignored_after_required_normalization() -> None:
    def primary_extra(method: str, params: Sequence[Any], result: Any) -> Any:
        if method == "eth_getBlockByNumber":
            return {**result, "transactions": [], "size": "0x1"}
        return result

    def verification_extra(method: str, params: Sequence[Any], result: Any) -> Any:
        if method == "eth_getBlockByNumber":
            return {**result, "transactions": ["ignored"], "withdrawals": []}
        return result

    clients = (
        FakeClient(builder.TRANSPORTS[0], mutate=primary_extra),
        FakeClient(builder.TRANSPORTS[1], mutate=verification_extra),
    )
    assert builder._header_pair(
        clients, "0x64", expected_number=100
    ) == builder.normalize_header(_header(100, 1_600_000_100))


def test_protocol_seal_tracks_complete_gross9_adapter_closure_and_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert builder.PROTOCOL_PATHS == (
        Path("training/build_ethereum_settlement_demand_impulse_source.py"),
        Path("training/preregister_ethereum_settlement_demand_impulse.py"),
        *builder.GROSS9_ADAPTER_EXTRA_CLOSURE_PATHS,
        Path("training/evaluate_ethereum_settlement_demand_impulse_economics.py"),
        Path("training/evaluate_ethereum_settlement_demand_impulse_novelty.py"),
        Path("training/evaluate_ethereum_settlement_demand_impulse_source_support.py"),
        Path("tests/test_build_ethereum_settlement_demand_impulse_source.py"),
        Path("tests/test_evaluate_ethereum_settlement_demand_impulse_economics.py"),
        Path("tests/test_evaluate_ethereum_settlement_demand_impulse_novelty.py"),
        Path("tests/test_evaluate_ethereum_settlement_demand_impulse_source_support.py"),
    )
    _init_protocol_repository(tmp_path)
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    recorded = builder.current_protocol_seal()
    assert len(recorded["files"]) == 37
    assert all(
        set(binding) == {"git_blob", "sha256"}
        for binding in recorded["files"].values()
    )
    assert builder.validate_protocol_seal(recorded)["files"] == recorded["files"]

    changed = tmp_path / builder.PROTOCOL_PATHS[0]
    changed.write_text("staged-only drift\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", str(changed)], check=True)
    changed.write_text("protocol file 0\n", encoding="utf-8")
    with pytest.raises(builder.TerminalSourceFailure, match="HEAD-clean"):
        builder.current_protocol_seal()
    subprocess.run(
        ["git", "-C", str(tmp_path), "restore", "--staged", str(changed)],
        check=True,
    )

    changed.write_text("working tree drift\n", encoding="utf-8")
    with pytest.raises(builder.TerminalSourceFailure, match="HEAD-clean"):
        builder.current_protocol_seal()
    changed.write_text("protocol file 0\n", encoding="utf-8")

    unrelated = tmp_path / "README.synthetic"
    unrelated.write_text("descendant commit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", str(unrelated)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "descendant"],
        check=True,
    )
    assert builder.validate_protocol_seal(recorded)["files"] == recorded["files"]

    changed.write_text("committed protocol drift\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", str(changed)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "protocol drift"],
        check=True,
    )
    with pytest.raises(builder.TerminalSourceFailure, match="unchanged"):
        builder.validate_protocol_seal(recorded)


def test_deterministic_gzip_csv_has_exact_epoch_schema() -> None:
    row = {column: f"value-{index}" for index, column in enumerate(builder.EPOCH_COLUMNS)}
    first = builder._csv_gzip([row])
    second = builder._csv_gzip([row])
    assert first == second
    assert first[4:8] == b"\x00\x00\x00\x00"
    lines = gzip.decompress(first).decode().splitlines()
    assert lines[0].split(",") == list(builder.EPOCH_COLUMNS)
    assert len(lines) == 2


def test_manifest_explicitly_discards_terminal_padding_and_is_deterministic(
    tmp_path: Path,
) -> None:
    cfg = builder._SyntheticConfig(
        raw_output=str(tmp_path / "raw.ndjson.gz"),
        epoch_output=str(tmp_path / "epochs.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
    )
    finalized = builder.Header(
        builder.LAST_CONFIRMATION_BLOCK,
        _hash(1),
        _hash(0),
        1,
    )
    first = builder._manifest(
        cfg=cfg,
        raw_size=3,
        raw_sha256=hashlib.sha256(b"raw").hexdigest(),
        epoch_bytes=b"epochs",
        finalized=finalized,
        boundary_audit=[],
        builder_sha256="a" * 64,
        claim_binding={"path": "claim.json", "sha256": "b" * 64},
        pre_replay_protocol_seal={"seal_hash": "c" * 64},
    )
    second = builder._manifest(
        cfg=cfg,
        raw_size=3,
        raw_sha256=hashlib.sha256(b"raw").hexdigest(),
        epoch_bytes=b"epochs",
        finalized=finalized,
        boundary_audit=[],
        builder_sha256="a" * 64,
        claim_binding={"path": "claim.json", "sha256": "b" * 64},
        pre_replay_protocol_seal={"seal_hash": "c" * 64},
    )
    assert first == second
    assert first["range"]["terminal_padding_blocks_requested"] == 352
    assert first["range"]["terminal_padding_disposition"] == (
        "discarded_before_epoch_normalization"
    )
    assert first["range"]["terminal_padding_entered_normalized_epochs"] == 0
    assert first["epochs"]["rows"] == 2_474
    assert first["claim"]["sha256"] == "b" * 64
    assert first["pre_replay_protocol_seal"]["seal_hash"] == "c" * 64
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == hashlib.sha256(
        builder._canonical_json_bytes(core)
    ).hexdigest()


def test_atomic_publish_is_write_once_and_rolls_back_partial_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary_a = tmp_path / "temporary-a"
    temporary_b = tmp_path / "temporary-b"
    temporary_a.write_bytes(b"a")
    temporary_b.write_bytes(b"b")
    final_a = tmp_path / "final-a"
    final_b = tmp_path / "final-b"

    real_link = os.link
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publish failure")
        real_link(source, target)

    monkeypatch.setattr(builder.os, "link", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        builder.atomic_publish(
            {final_a: temporary_a, final_b: temporary_b}
        )
    assert not final_a.exists()
    assert not final_b.exists()

    monkeypatch.setattr(builder.os, "link", real_link)
    final_a.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="write-once"):
        builder.atomic_publish(
            {final_a: temporary_a, final_b: temporary_b}
        )
    assert final_a.read_bytes() == b"existing"
    assert not final_b.exists()


def _complete_production_stage(
    root: Path,
    *,
    seal: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, str],
    tuple[Path, Path, Path],
]:
    claim_payload = builder._production_claim_payload(seal)
    claim_raw = builder._canonical_json_bytes(claim_payload, trailing_lf=True)
    claim_path = root / builder.REPLAY_CLAIM_PATH
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_bytes(claim_raw)
    claim_binding = {
        "path": builder.REPLAY_CLAIM_PATH.as_posix(),
        "sha256": hashlib.sha256(claim_raw).hexdigest(),
        "claim_hash": claim_payload["claim_hash"],
    }
    staged = builder._production_stage_paths(claim_payload["claim_hash"])
    staged[0].parent.mkdir(parents=True)
    raw_bytes = b"durable raw generation"
    epoch_bytes = b"durable epoch generation"
    staged[0].write_bytes(raw_bytes)
    staged[1].write_bytes(epoch_bytes)
    cfg = builder._SyntheticConfig(
        raw_output=builder.DEFAULT_RAW_OUTPUT.as_posix(),
        epoch_output=builder.DEFAULT_EPOCH_OUTPUT.as_posix(),
        manifest_output=builder.DEFAULT_MANIFEST_OUTPUT.as_posix(),
    )
    manifest = builder._manifest(
        cfg=cfg,
        raw_size=len(raw_bytes),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        epoch_bytes=epoch_bytes,
        finalized=builder.Header(
            builder.LAST_CONFIRMATION_BLOCK,
            _hash(1),
            _hash(0),
            1,
        ),
        boundary_audit=[],
        builder_sha256="a" * 64,
        claim_binding=claim_binding,
        pre_replay_protocol_seal=seal,
    )
    staged[2].write_bytes(
        builder._canonical_json_bytes(manifest, trailing_lf=True)
    )
    return manifest, claim_binding, staged


def test_production_publishes_one_complete_directory_and_verifies_without_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    seal = {"seal_hash": "b" * 64}
    expected, claim_binding, staged = _complete_production_stage(
        tmp_path, seal=seal
    )
    staged_bytes = tuple(path.read_bytes() for path in staged)
    stage_directory = staged[0].parent
    final_directory = builder._canonical_output_directory()
    canonical = builder._canonical_output_paths()
    real_rename = os.rename
    renamed: list[tuple[Path, Path]] = []

    def record_rename(source: Path, target: Path) -> None:
        assert source == stage_directory
        assert target == final_directory
        assert not target.exists()
        assert all(path.is_file() for path in staged)
        renamed.append((source, target))
        real_rename(source, target)
        assert all(path.is_file() for path in canonical)

    monkeypatch.setattr(builder.os, "rename", record_rename)
    assert builder._publish_staged_generation(
        staged_paths=staged,
        claim_binding=claim_binding,
        pre_replay_protocol_seal=seal,
    ) == expected
    assert renamed == [(stage_directory, final_directory)]
    assert all(
        path.read_bytes() == expected_bytes
        for path, expected_bytes in zip(canonical, staged_bytes)
    )
    assert not stage_directory.exists()

    monkeypatch.setattr(builder, "validate_preregistration", lambda: {})
    monkeypatch.setattr(builder, "validate_protocol_seal", lambda _: {})

    def forbidden(*_: object, **__: object) -> Any:
        raise AssertionError("published-generation verification must make zero RPC calls")

    monkeypatch.setattr(builder, "HttpJsonRpcClient", forbidden)
    assert builder.build_source() == expected
    assert renamed == [(stage_directory, final_directory)]


def test_failure_before_directory_rename_leaves_no_canonical_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    seal = {"seal_hash": "b" * 64}
    _, claim_binding, staged = _complete_production_stage(tmp_path, seal=seal)
    stage_directory = staged[0].parent
    final_directory = builder._canonical_output_directory()

    def fail_rename(_source: Path, _target: Path) -> None:
        raise OSError("simulated process death before publication")

    monkeypatch.setattr(builder.os, "rename", fail_rename)
    with pytest.raises(OSError, match="process death"):
        builder._publish_staged_generation(
            staged_paths=staged,
            claim_binding=claim_binding,
            pre_replay_protocol_seal=seal,
        )
    assert stage_directory.is_dir()
    assert not final_directory.exists()
    assert not any(path.exists() for path in builder._canonical_output_paths())

    monkeypatch.setattr(builder, "validate_preregistration", lambda: {})
    monkeypatch.setattr(builder, "validate_protocol_seal", lambda _: {})
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="unpublished durable stage",
    ):
        builder.build_source()


def test_failure_after_directory_rename_recovers_complete_generation_without_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    seal = {"seal_hash": "b" * 64}
    expected, claim_binding, staged = _complete_production_stage(
        tmp_path, seal=seal
    )
    stage_directory = staged[0].parent
    final_directory = builder._canonical_output_directory()
    final_parent = final_directory.parent
    real_fsync_directory = builder._fsync_directory

    def fail_post_rename_fsync(path: Path) -> None:
        if (
            path == final_parent
            and final_directory.is_dir()
            and not stage_directory.exists()
        ):
            raise OSError("simulated crash after atomic rename")
        real_fsync_directory(path)

    monkeypatch.setattr(
        builder,
        "_fsync_directory",
        fail_post_rename_fsync,
    )
    with pytest.raises(OSError, match="after atomic rename"):
        builder._publish_staged_generation(
            staged_paths=staged,
            claim_binding=claim_binding,
            pre_replay_protocol_seal=seal,
        )
    assert final_directory.is_dir()
    assert not stage_directory.exists()
    assert all(path.is_file() for path in builder._canonical_output_paths())

    monkeypatch.setattr(builder, "_fsync_directory", real_fsync_directory)
    monkeypatch.setattr(builder, "validate_preregistration", lambda: {})
    monkeypatch.setattr(builder, "validate_protocol_seal", lambda _: {})

    def forbidden(*_: object, **__: object) -> Any:
        raise AssertionError("post-rename recovery must make zero RPC calls")

    monkeypatch.setattr(builder, "HttpJsonRpcClient", forbidden)
    assert builder.build_source() == expected


@pytest.mark.parametrize("tamper_complete_stage", [False, True])
def test_production_never_recovers_unpublished_stage_without_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_complete_stage: bool,
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    seal = {"seal_hash": "b" * 64}
    if tamper_complete_stage:
        _, _, staged = _complete_production_stage(tmp_path, seal=seal)
        staged[1].write_bytes(b"cryptographically different")
    else:
        claim_payload = builder._production_claim_payload(seal)
        claim_raw = builder._canonical_json_bytes(
            claim_payload, trailing_lf=True
        )
        claim_path = tmp_path / builder.REPLAY_CLAIM_PATH
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.write_bytes(claim_raw)
        staged = builder._production_stage_paths(claim_payload["claim_hash"])
        staged[0].parent.mkdir(parents=True)
        staged[0].write_bytes(b"partial raw")

    monkeypatch.setattr(builder, "validate_preregistration", lambda: {})
    monkeypatch.setattr(builder, "validate_protocol_seal", lambda _: {})

    def forbidden(*_: object, **__: object) -> Any:
        raise AssertionError("terminal recovery must make zero RPC calls")

    monkeypatch.setattr(builder, "HttpJsonRpcClient", forbidden)
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="unpublished durable stage",
    ):
        builder.build_source()
    assert not any(path.exists() for path in builder._canonical_output_paths())


def test_claim_without_stage_or_atomic_generation_is_terminal_without_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    seal = {"seal_hash": "b" * 64}
    claim_payload = builder._production_claim_payload(seal)
    claim_path = tmp_path / builder.REPLAY_CLAIM_PATH
    claim_path.parent.mkdir(parents=True)
    claim_path.write_bytes(
        builder._canonical_json_bytes(claim_payload, trailing_lf=True)
    )
    monkeypatch.setattr(builder, "validate_preregistration", lambda: {})
    monkeypatch.setattr(builder, "validate_protocol_seal", lambda _: {})

    def forbidden(*_: object, **__: object) -> Any:
        raise AssertionError("terminal claimed replay must make zero RPC calls")

    monkeypatch.setattr(builder, "HttpJsonRpcClient", forbidden)
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="without an atomically published source generation",
    ):
        builder.build_source()


def test_output_preflight_happens_before_any_rpc_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw.gz"
    raw.write_bytes(b"existing")
    cfg = builder._SyntheticConfig(
        raw_output=str(raw),
        epoch_output=str(tmp_path / "epochs.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
    )
    clients = _clients()
    monkeypatch.setattr(builder, "validate_preregistration", lambda: {})
    with pytest.raises(FileExistsError, match="write-once"):
        builder._build_source_synthetic(cfg, clients=clients)
    assert all(not client.calls for client in clients)


def test_public_build_is_argument_free_and_cli_freezes_canonical_outputs() -> None:
    assert tuple(inspect.signature(builder.build_source).parameters) == ()
    assert builder.DEFAULT_RAW_OUTPUT == Path(
        "data/ethereum_settlement_demand_impulse/"
        "ethereum_fee_history_chunks_2023_2026.ndjson.gz"
    )
    assert builder.DEFAULT_EPOCH_OUTPUT == Path(
        "data/ethereum_settlement_demand_impulse/"
        "ethereum_settlement_demand_impulse_epochs_2023_2026.csv.gz"
    )
    assert builder.DEFAULT_MANIFEST_OUTPUT == Path(
        "data/ethereum_settlement_demand_impulse/"
        "ethereum_settlement_demand_impulse_source_manifest_2026-07-30.json"
    )
    assert builder.REPLAY_CLAIM_PATH == Path(
        "results/ethereum_settlement_demand_impulse_source_replay_claim_2026-07-30.json"
    )
    with pytest.raises(SystemExit):
        builder.parse_args(["--raw-output", "forbidden"])


def test_synthetic_helper_cannot_publish_any_canonical_path(
    tmp_path: Path,
) -> None:
    clients = _clients()
    for canonical in (
        builder.DEFAULT_RAW_OUTPUT,
        builder.DEFAULT_EPOCH_OUTPUT,
        builder.DEFAULT_MANIFEST_OUTPUT,
    ):
        cfg = builder._SyntheticConfig(
            raw_output=str(
                canonical
                if canonical == builder.DEFAULT_RAW_OUTPUT
                else tmp_path / "raw.gz"
            ),
            epoch_output=str(
                canonical
                if canonical == builder.DEFAULT_EPOCH_OUTPUT
                else tmp_path / "epochs.gz"
            ),
            manifest_output=str(
                canonical
                if canonical == builder.DEFAULT_MANIFEST_OUTPUT
                else tmp_path / "manifest.json"
            ),
        )
        with pytest.raises(builder.TerminalSourceFailure, match="synthetic"):
            builder._build_source_synthetic(cfg, clients=clients)
    assert all(not client.calls for client in clients)


def test_production_output_guard_prevents_seal_and_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    output = tmp_path / builder.DEFAULT_RAW_OUTPUT
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing canonical output")
    called = False

    def forbidden(*_: object, **__: object) -> Any:
        nonlocal called
        called = True
        raise AssertionError("must not run")

    monkeypatch.setattr(builder, "current_protocol_seal", forbidden)
    monkeypatch.setattr(builder, "HttpJsonRpcClient", forbidden)
    with pytest.raises(FileExistsError, match="forever"):
        builder.build_source()
    assert called is False


def test_production_claim_precedes_rpc_and_survives_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(builder, "validate_preregistration", lambda: {})
    seal = {
        "git_commit": "a" * 40,
        "files": {},
        "seal_hash": "b" * 64,
    }
    events: list[str] = []

    def sealed(*, require_committed: bool = True) -> dict[str, Any]:
        assert require_committed is True
        events.append("seal")
        return seal

    class FailingProductionClient:
        def __init__(self, url: str, *, timeout_seconds: float) -> None:
            self.url = url

        def call(self, method: str, params: Sequence[Any]) -> Any:
            claim = tmp_path / builder.REPLAY_CLAIM_PATH
            assert claim.is_file()
            events.append("rpc")
            raise builder.TerminalSourceFailure("synthetic terminal RPC failure")

    monkeypatch.setattr(builder, "current_protocol_seal", sealed)
    monkeypatch.setattr(builder, "HttpJsonRpcClient", FailingProductionClient)
    with pytest.raises(builder.TerminalSourceFailure, match="terminal RPC"):
        builder.build_source()
    claim = tmp_path / builder.REPLAY_CLAIM_PATH
    claim_bytes = claim.read_bytes()
    assert events == ["seal", "rpc"]
    assert not (tmp_path / builder.DEFAULT_RAW_OUTPUT).exists()
    assert not (tmp_path / builder.DEFAULT_EPOCH_OUTPUT).exists()
    assert not (tmp_path / builder.DEFAULT_MANIFEST_OUTPUT).exists()

    events.clear()
    monkeypatch.setattr(builder, "validate_protocol_seal", lambda _: {})
    with pytest.raises(
        builder.TerminalSourceFailure,
        match="unpublished durable stage",
    ):
        builder.build_source()
    assert events == []
    assert claim.read_bytes() == claim_bytes
