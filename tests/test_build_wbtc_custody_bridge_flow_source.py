from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any, Sequence

import pytest

from training import build_wbtc_custody_bridge_flow_source as builder
from training import preregister_wbtc_custody_bridge_flow_source as protocol


def _hash(number: int) -> str:
    return "0x" + f"{number:064x}"


def _actor(seed: int) -> str:
    return "0x" + f"{seed:040x}"


def _actor_topic(seed: int) -> str:
    return "0x" + "0" * 24 + f"{seed:040x}"


def _header(number: int, timestamp: int, block_hash: str | None = None) -> dict[str, Any]:
    return {
        "number": hex(number),
        "hash": block_hash or _hash(number),
        "parentHash": _hash(max(0, number - 1)),
        "timestamp": hex(timestamp),
        "transactions": [],
    }


def _semantic_log(
    *, event: str, block: int, transaction_index: int, log_index: int, amount: int
) -> dict[str, Any]:
    topic = protocol.MINT_TOPIC if event == "mint" else protocol.BURN_TOPIC
    seed = block % 10_000 + 1
    return {
        "address": protocol.WBTC_ADDRESS,
        "topics": [topic, _actor_topic(seed)],
        "data": "0x" + f"{amount:064x}",
        "blockNumber": hex(block),
        "blockHash": _hash(block),
        "transactionHash": _hash(20_000_000 + block + log_index),
        "transactionIndex": hex(transaction_index),
        "logIndex": hex(log_index),
        "removed": False,
    }


def _transfer_companion(row: dict[str, Any]) -> dict[str, Any]:
    zero = protocol.ZERO_TOPIC
    topics = (
        [protocol.TRANSFER_TOPIC, zero, row["topics"][1]]
        if row["topics"][0] == protocol.MINT_TOPIC
        else [protocol.TRANSFER_TOPIC, row["topics"][1], zero]
    )
    return {
        **row,
        "topics": topics,
        "logIndex": hex(int(row["logIndex"], 16) + 1),
    }


def _receipt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "transactionHash": row["transactionHash"],
        "blockNumber": row["blockNumber"],
        "blockHash": row["blockHash"],
        "transactionIndex": row["transactionIndex"],
        "status": "0x1",
        "logs": [row, _transfer_companion(row)],
    }


class FakeRpc:
    def __init__(
        self,
        logs: Sequence[dict[str, Any]],
        *,
        code: str = "0x6000",
        receipts: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.logs = list(logs)
        self.code = code
        self.receipts = receipts or {
            row["transactionHash"]: _receipt(row) for row in self.logs
        }
        self.timestamps: dict[int, int] = {
            protocol.START_BOUNDARY_BLOCK: 1_577_836_811,
            protocol.END_BOUNDARY_BLOCK: 1_704_067_211,
        }
        for offset, row in enumerate(self.logs):
            block = int(row["blockNumber"], 16)
            year = 2020 + offset // 2
            base = {
                2020: 1_578_000_000,
                2021: 1_609_545_600,
                2022: 1_640_995_200,
                2023: 1_672_531_200,
            }[year]
            self.timestamps[block] = base + offset * 600
            self.timestamps[block + protocol.CONFIRMATION_BLOCKS] = base + offset * 600 + 800

    def _block(self, number: int) -> dict[str, Any]:
        if number == protocol.START_BOUNDARY_BLOCK:
            return _header(number, self.timestamps[number], protocol.START_BOUNDARY_HASH)
        if number == protocol.END_BOUNDARY_BLOCK:
            return _header(number, self.timestamps[number], protocol.END_BOUNDARY_HASH)
        return _header(number, self.timestamps.get(number, 1_700_000_000 + number))

    def call(self, method: str, params: list[Any]) -> Any:
        if method == "eth_chainId":
            return "0x1"
        if method == "eth_getCode":
            return self.code
        if method == "eth_getBlockByNumber":
            value = params[0]
            if value in {"latest", "finalized"}:
                return self._block(protocol.LAST_SOURCE_BLOCK + 10_000)
            return self._block(int(value, 16))
        if method == "eth_getLogs":
            query = params[0]
            start = int(query["fromBlock"], 16)
            end = int(query["toBlock"], 16)
            topics = set(query["topics"][0])
            return [
                row
                for row in self.logs
                if start <= int(row["blockNumber"], 16) <= end
                and row["topics"][0] in topics
            ]
        if method == "eth_getTransactionReceipt":
            return self.receipts.get(params[0])
        raise AssertionError(f"unexpected RPC method {method}")

    def batch(self, requests: Sequence[tuple[str, list[Any]]]) -> list[Any]:
        return [self.call(method, params) for method, params in requests]


def _logs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(8):
        rows.append(
            _semantic_log(
                event="mint" if offset % 2 == 0 else "burn",
                block=protocol.START_BOUNDARY_BLOCK + 10 + offset * 100_000,
                transaction_index=offset,
                log_index=offset * 3,
                amount=(offset + 1) * 100_000_000,
            )
        )
    return rows


def _config(tmp_path: Path) -> builder.Config:
    return builder.Config(
        output_csv=str(tmp_path / "source.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        primary_rpc_url="https://primary.invalid",
        verification_rpc_url="https://verification.invalid",
        checkpoint_db=str(tmp_path / "checkpoint.sqlite3"),
        max_block_range=protocol.MAX_BLOCK_RANGE,
        header_batch_size=8,
        receipt_batch_size=4,
    )


def test_normalize_semantic_log_decodes_frozen_abi() -> None:
    raw = _semantic_log(
        event="mint",
        block=protocol.START_BOUNDARY_BLOCK + 1,
        transaction_index=2,
        log_index=3,
        amount=125_000_000,
    )
    row = builder.normalize_semantic_log(
        raw,
        start_block=protocol.START_BOUNDARY_BLOCK,
        end_block=protocol.LAST_SOURCE_BLOCK,
    )
    assert row["event"] == "mint"
    assert row["event_sign"] == 1
    assert row["amount_raw"] == "125000000"
    assert row["decimals"] == 8
    assert row["actor_address"] == _actor(int(raw["blockNumber"], 16) % 10_000 + 1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"address": _actor(999)}, "contract/topic"),
        ({"removed": True}, "removed"),
        ({"topics": [protocol.MINT_TOPIC]}, "topic count"),
        ({"data": "0x" + "0" * 64}, "positive"),
        ({"data": "0x00"}, "32 bytes"),
    ],
)
def test_normalize_semantic_log_fails_closed(
    change: dict[str, Any], message: str
) -> None:
    raw = _semantic_log(
        event="mint",
        block=protocol.START_BOUNDARY_BLOCK + 1,
        transaction_index=2,
        log_index=3,
        amount=125_000_000,
    ) | change
    with pytest.raises((ValueError, RuntimeError), match=message):
        builder.normalize_semantic_log(
            raw,
            start_block=protocol.START_BOUNDARY_BLOCK,
            end_block=protocol.LAST_SOURCE_BLOCK,
        )


def test_duplicate_semantic_identity_fails_closed() -> None:
    row = _logs()[0]
    with pytest.raises(RuntimeError, match="duplicate canonical log identity"):
        builder.normalize_semantic_logs(
            [row, dict(row)],
            start_block=protocol.START_BOUNDARY_BLOCK,
            end_block=protocol.LAST_SOURCE_BLOCK,
        )


def test_receipt_pair_requires_success_and_adjacent_zero_transfer() -> None:
    raw = _logs()[0]
    normalized = builder.normalize_semantic_logs(
        [raw],
        start_block=protocol.START_BOUNDARY_BLOCK,
        end_block=protocol.LAST_SOURCE_BLOCK,
    )
    paired, audit = builder.validate_receipt_pairs(
        FakeRpc([raw]), normalized, batch_size=1
    )
    assert paired[0]["companion_transfer_log_index"] == normalized[0]["log_index"] + 1
    assert audit["semantic_events_verified"] == 1
    assert audit["zero_transfer_pairs_verified"] == 1

    reverted = _receipt(raw) | {"status": "0x0"}
    with pytest.raises(RuntimeError, match="reverted"):
        builder.validate_receipt_pairs(
            FakeRpc([raw], receipts={raw["transactionHash"]: reverted}),
            normalized,
            batch_size=1,
        )

    missing_pair = _receipt(raw) | {"logs": [raw]}
    with pytest.raises(RuntimeError, match="zero-transfer pair"):
        builder.validate_receipt_pairs(
            FakeRpc([raw], receipts={raw["transactionHash"]: missing_pair}),
            normalized,
            batch_size=1,
        )


def test_build_outputs_is_dual_replayed_and_source_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = "0x6000"
    expected_code_hash = hashlib.sha256(bytes.fromhex(code[2:])).hexdigest()
    monkeypatch.setattr(protocol, "BOUNDARY_CODE_SHA256", expected_code_hash)
    logs = _logs()
    primary = FakeRpc(logs, code=code)
    verification = FakeRpc(logs, code=code)
    rows, core = builder.build_outputs(
        _config(tmp_path),
        primary_rpc=primary,
        verification_rpc=verification,
        header_rpc=primary,
        receipt_rpc=verification,
    )
    assert len(rows) == 8
    assert core["dual_replay"]["canonical_replay_equal"] is True
    assert len(set(core["dual_replay"]["canonical_log_hashes"])) == 1
    assert core["source_support"]["decision"] == "PASS_SOURCE"
    assert all(
        counts == {"mint": 1, "burn": 1}
        for counts in core["source_support"]["year_event_counts"].values()
    )
    assert core["outcome_boundary"] == {
        "source_only": True,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "labels_opened": False,
        "pnl_cagr_mdd_opened": False,
        "mechanism_features_opened": False,
        "post_2023_contract_event_rows_read": 0,
        "post_2023_confirmation_headers_may_be_read": True,
    }
    assert core["dual_replay"]["provider_urls_embedded"] is False


def test_dual_replay_disagreement_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = "0x6000"
    monkeypatch.setattr(
        protocol,
        "BOUNDARY_CODE_SHA256",
        hashlib.sha256(bytes.fromhex(code[2:])).hexdigest(),
    )
    primary_logs = _logs()
    changed_logs = _logs()
    changed_logs[0] = changed_logs[0] | {"data": "0x" + f"{999:064x}"}
    with pytest.raises(RuntimeError, match="replays disagree"):
        builder.build_outputs(
            _config(tmp_path),
            primary_rpc=FakeRpc(primary_logs, code=code),
            verification_rpc=FakeRpc(changed_logs, code=code),
        )


def test_deterministic_gzip_writer(tmp_path: Path) -> None:
    rows = [
        {
            column: index if column.endswith("index") or column.endswith("number") else str(index)
            for index, column in enumerate(builder.OUTPUT_COLUMNS)
        }
    ]
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    builder._write_csv_gz(first, rows)
    builder._write_csv_gz(second, rows)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert handle.readline().strip().split(",") == list(builder.OUTPUT_COLUMNS)


def test_independent_rpc_host_is_required() -> None:
    cfg = builder.Config(
        primary_rpc_url="https://same.invalid/a",
        verification_rpc_url="https://same.invalid/b",
    )
    with pytest.raises(ValueError, match="independent hostname"):
        builder._transport_hosts(cfg)


def test_log_checkpoint_resumes_completed_ranges(tmp_path: Path) -> None:
    class CountingRpc(FakeRpc):
        def __init__(self, logs: Sequence[dict[str, Any]]) -> None:
            super().__init__(logs)
            self.log_calls = 0

        def call(self, method: str, params: list[Any]) -> Any:
            if method == "eth_getLogs":
                self.log_calls += 1
            return super().call(method, params)

    rpc = CountingRpc(_logs()[:2])
    checkpoint_path = tmp_path / "checkpoint.sqlite3"
    checkpoint = builder.LogCheckpoint(checkpoint_path)
    first = builder.fetch_semantic_logs(
        rpc,
        protocol.START_BOUNDARY_BLOCK,
        protocol.START_BOUNDARY_BLOCK + 20_000,
        max_block_range=10_000,
        checkpoint=checkpoint,
        checkpoint_role="primary:test",
    )
    first_call_count = rpc.log_calls
    checkpoint.close()

    checkpoint = builder.LogCheckpoint(checkpoint_path)
    second = builder.fetch_semantic_logs(
        rpc,
        protocol.START_BOUNDARY_BLOCK,
        protocol.START_BOUNDARY_BLOCK + 20_000,
        max_block_range=10_000,
        checkpoint=checkpoint,
        checkpoint_role="primary:test",
    )
    checkpoint.close()
    assert first == second
    assert first_call_count == 3
    assert rpc.log_calls == first_call_count


def test_protocol_and_helper_bindings_are_current() -> None:
    payload = builder._load_protocol()
    assert payload["manifest_hash"] == builder.PROTOCOL_MANIFEST_HASH
    assert builder._sha256(builder.PROTOCOL_PATH) == builder.PROTOCOL_SHA256
    assert builder._sha256(builder.ETHEREUM_HELPER_PATH) == (
        builder.ETHEREUM_HELPER_SHA256
    )


def test_disk_limit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Usage:
        used = protocol.DISK_LIMIT_GIB * 1024**3

    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _: Usage())
    with pytest.raises(RuntimeError, match="used disk below"):
        builder._check_disk_limit()
