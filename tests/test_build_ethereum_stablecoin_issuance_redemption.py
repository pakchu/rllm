from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any, Sequence

import pytest

from training import build_ethereum_stablecoin_issuance_redemption as builder


def _hash(number: int) -> str:
    return "0x" + f"{number:064x}"


def _header(number: int, timestamp: int) -> dict[str, Any]:
    return {
        "number": hex(number),
        "hash": _hash(10_000 + number),
        "parentHash": _hash(10_000 + max(0, number - 1)),
        "timestamp": hex(timestamp),
        "transactions": [],
    }


def _log(
    *,
    address: str,
    topic: str,
    block: int,
    log_index: int,
    amount: int,
    extra_topics: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "address": address,
        "topics": [topic, *extra_topics],
        "data": "0x" + f"{amount:064x}",
        "blockNumber": hex(block),
        "blockHash": _hash(10_000 + block),
        "transactionHash": _hash(20_000 + log_index),
        "transactionIndex": "0x1",
        "logIndex": hex(log_index),
        "removed": False,
    }


class FakeRpc:
    def __init__(
        self,
        *,
        headers: dict[int, dict[str, Any]],
        logs: Sequence[dict[str, Any]] = (),
        finalized_number: int | None = None,
    ) -> None:
        self.headers = headers
        self.logs = list(logs)
        self.finalized_number = (
            max(headers) if finalized_number is None else finalized_number
        )

    def call(self, method: str, params: list[Any]) -> Any:
        if method == "eth_chainId":
            return "0x1"
        if method == "eth_getBlockByNumber":
            block = params[0]
            if block == "latest":
                number = max(self.headers)
            elif block == "finalized":
                number = self.finalized_number
            else:
                number = int(block, 16)
            return self.headers[number]
        if method == "eth_getCode":
            return "0x6000"
        if method == "eth_getLogs":
            query = params[0]
            start = int(query["fromBlock"], 16)
            end = int(query["toBlock"], 16)
            address = str(query["address"]).lower()
            allowed_topics = set(query["topics"][0])
            return [
                row
                for row in self.logs
                if start <= int(row["blockNumber"], 16) <= end
                and str(row["address"]).lower() == address
                and row["topics"][0] in allowed_topics
            ]
        raise AssertionError(f"unexpected RPC method: {method}")

    def batch(self, requests: Sequence[tuple[str, list[Any]]]) -> list[Any]:
        return [self.call(method, params) for method, params in requests]


class NoBatchRpc(FakeRpc):
    def batch(self, requests: Sequence[tuple[str, list[Any]]]) -> list[Any]:
        raise AssertionError("verification replay must not fetch confirmation headers")


def _headers() -> dict[int, dict[str, Any]]:
    return {number: _header(number, number * 100) for number in range(11)}


def _fixture_logs() -> list[dict[str, Any]]:
    return [
        _log(
            address=builder.USDT_ADDRESS,
            topic=builder.USDT_ISSUE_TOPIC,
            block=3,
            log_index=0,
            amount=25_000_000,
        ),
        _log(
            address=builder.USDC_ADDRESS,
            topic=builder.USDC_BURN_TOPIC,
            block=4,
            log_index=1,
            amount=10_000_000,
            extra_topics=(_hash(777),),
        ),
    ]


def _config(tmp_path: Path) -> builder.Config:
    return builder.Config(
        output_csv=str(tmp_path / "source.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        primary_rpc_url="https://primary.invalid",
        verification_rpc_url="https://secondary.invalid",
        start="1970-01-01T00:03:20Z",
        end_exclusive="1970-01-01T00:08:20Z",
        confirmation_blocks=2,
        max_block_range=2,
        header_batch_size=2,
    )


def test_frozen_topics_match_exact_keccak_signatures() -> None:
    from Crypto.Hash import keccak

    expected = {
        "Issue(uint256)": builder.USDT_ISSUE_TOPIC,
        "Redeem(uint256)": builder.USDT_REDEEM_TOPIC,
        "DestroyedBlackFunds(address,uint256)": builder.USDT_DESTROYED_BLACK_FUNDS_TOPIC,
        "Deprecate(address)": builder.USDT_DEPRECATE_TOPIC,
        "Mint(address,address,uint256)": builder.USDC_MINT_TOPIC,
        "Burn(address,uint256)": builder.USDC_BURN_TOPIC,
    }
    for signature, topic in expected.items():
        digest = keccak.new(digest_bits=256)
        digest.update(signature.encode("ascii"))
        assert "0x" + digest.hexdigest() == topic


def test_find_first_block_at_or_after_is_strict_boundary() -> None:
    rpc = FakeRpc(headers=_headers())
    assert builder.find_first_block_at_or_after(rpc, 550).number == 6
    assert builder.find_first_block_at_or_after(rpc, 600).number == 6


def test_materialize_rows_uses_confirmation_block_timestamp() -> None:
    rpc = FakeRpc(headers=_headers())
    rows = builder.materialize_rows(
        rpc,
        _fixture_logs(),
        start_block=2,
        end_block=7,
        confirmation_blocks=2,
        header_batch_size=2,
    )
    assert [(row["asset"], row["event"]) for row in rows] == [
        ("usdt_eth", "issue"),
        ("usdc_eth", "burn"),
    ]
    assert rows[0]["block_timestamp"] == "1970-01-01T00:05:00Z"
    assert rows[0]["available_at"] == "1970-01-01T00:08:20Z"
    assert rows[0]["confirmation_block_number"] == 5
    assert rows[1]["available_at"] == "1970-01-01T00:10:00Z"


def test_destroyed_black_funds_decodes_address_and_amount_separately() -> None:
    subject = "0x" + "11" * 20
    raw = _log(
        address=builder.USDT_ADDRESS,
        topic=builder.USDT_DESTROYED_BLACK_FUNDS_TOPIC,
        block=3,
        log_index=9,
        amount=1,
    )
    raw["data"] = "0x" + "0" * 24 + subject[2:] + f"{12_500_000:064x}"
    row = builder.normalize_log(raw, start_block=2, end_block=7)
    assert row["event"] == "destroyed_black_funds"
    assert row["event_sign"] == -1
    assert row["data_address"] == subject
    assert row["amount_raw"] == "12500000"


def test_deprecate_is_metadata_without_supply_amount() -> None:
    successor = "0x" + "22" * 20
    raw = _log(
        address=builder.USDT_ADDRESS,
        topic=builder.USDT_DEPRECATE_TOPIC,
        block=3,
        log_index=10,
        amount=1,
    )
    raw["data"] = "0x" + "0" * 24 + successor[2:]
    row = builder.normalize_log(raw, start_block=2, end_block=7)
    assert row["event"] == "deprecate"
    assert row["event_sign"] == 0
    assert row["amount_raw"] == ""
    assert row["data_address"] == successor


def test_redeem_and_mint_decode_frozen_abi_shapes() -> None:
    minter = "0x" + "33" * 20
    recipient = "0x" + "44" * 20
    redeem = _log(
        address=builder.USDT_ADDRESS,
        topic=builder.USDT_REDEEM_TOPIC,
        block=3,
        log_index=11,
        amount=3_000_000,
    )
    mint = _log(
        address=builder.USDC_ADDRESS,
        topic=builder.USDC_MINT_TOPIC,
        block=4,
        log_index=12,
        amount=7_000_000,
        extra_topics=(
            "0x" + "0" * 24 + minter[2:],
            "0x" + "0" * 24 + recipient[2:],
        ),
    )
    rows = builder.materialize_rows(
        FakeRpc(headers=_headers()),
        [redeem, mint],
        start_block=2,
        end_block=7,
        confirmation_blocks=2,
        header_batch_size=2,
    )
    assert rows[0]["event"] == "redeem"
    assert rows[0]["event_sign"] == -1
    assert rows[0]["amount_raw"] == "3000000"
    assert rows[1]["event"] == "mint"
    assert rows[1]["event_sign"] == 1
    assert rows[1]["amount_raw"] == "7000000"
    assert rows[1]["indexed_address_1"] == minter
    assert rows[1]["indexed_address_2"] == recipient


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"removed": True}, "removed"),
        ({"data": "0x00"}, "32 bytes"),
        ({"topics": [builder.USDC_BURN_TOPIC]}, "topic count"),
        ({"address": builder.USDT_ADDRESS}, "contract/topic"),
    ],
)
def test_normalize_log_fails_closed_on_malformed_event(
    change: dict[str, Any], message: str
) -> None:
    raw = _fixture_logs()[1] | change
    with pytest.raises((ValueError, RuntimeError), match=message):
        builder.normalize_log(raw, start_block=2, end_block=7)


def test_duplicate_canonical_identity_fails_closed() -> None:
    rpc = FakeRpc(headers=_headers())
    row = _fixture_logs()[0]
    with pytest.raises(RuntimeError, match="duplicate canonical log identity"):
        builder.materialize_rows(
            rpc,
            [row, dict(row)],
            start_block=2,
            end_block=7,
            confirmation_blocks=2,
            header_batch_size=2,
        )


def test_dual_replay_build_has_no_outcome_access(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    primary = FakeRpc(headers=_headers(), logs=_fixture_logs())
    verification = NoBatchRpc(headers=_headers(), logs=_fixture_logs())
    rows, core = builder.build_outputs(
        cfg, primary_rpc=primary, verification_rpc=verification
    )
    assert len(rows) == 2
    assert core["dual_replay"]["canonical_replay_equal"] is True
    assert len(core["dual_replay"]["canonical_log_hashes"]) == 2
    assert len(set(core["dual_replay"]["canonical_log_hashes"])) == 1
    assert core["dual_replay"]["provider_urls_embedded"] is False
    assert core["event_counts"] == {"usdc_eth:burn": 1, "usdt_eth:issue": 1}
    assert core["outcome_boundary"] == {
        "source_only": True,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "pnl_cagr_mdd_opened": False,
        "post_2023_contract_event_rows_read": 0,
        "post_2023_confirmation_headers_may_be_read": True,
    }
    assert not any("rpc" in key.lower() for key in core["source_contract"])


def test_dual_replay_mismatch_fails_closed(tmp_path: Path) -> None:
    changed = _fixture_logs()
    changed[0] = changed[0] | {"data": "0x" + f"{99_000_000:064x}"}
    with pytest.raises(RuntimeError, match="independent Ethereum RPC replays disagree"):
        builder.build_outputs(
            _config(tmp_path),
            primary_rpc=FakeRpc(headers=_headers(), logs=_fixture_logs()),
            verification_rpc=FakeRpc(headers=_headers(), logs=changed),
        )


def test_finalized_head_must_cover_confirmation_blocks(tmp_path: Path) -> None:
    primary = FakeRpc(headers=_headers(), logs=_fixture_logs(), finalized_number=5)
    verification = FakeRpc(headers=_headers(), logs=_fixture_logs())
    with pytest.raises(RuntimeError, match="finalized head does not cover"):
        builder.build_outputs(
            _config(tmp_path),
            primary_rpc=primary,
            verification_rpc=verification,
        )


def test_deterministic_gzip_writer(tmp_path: Path) -> None:
    rpc = FakeRpc(headers=_headers())
    rows = builder.materialize_rows(
        rpc,
        _fixture_logs(),
        start_block=2,
        end_block=7,
        confirmation_blocks=2,
        header_batch_size=2,
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    builder._write_csv_gz(first, rows)
    builder._write_csv_gz(second, rows)
    assert (
        hashlib.sha256(first.read_bytes()).digest()
        == hashlib.sha256(second.read_bytes()).digest()
    )
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert handle.readline().strip().split(",") == list(builder.OUTPUT_COLUMNS)


def test_independent_rpc_host_is_required() -> None:
    cfg = builder.Config(
        primary_rpc_url="https://same.example/a",
        verification_rpc_url="https://same.example/b",
    )
    with pytest.raises(ValueError, match="independent hostname"):
        builder._transport_hosts(cfg)


def test_json_rpc_client_retries_only_transient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_urlopen(request: Any, *, timeout: float) -> io.BytesIO:
        nonlocal attempts
        attempts += 1
        assert timeout == 1.0
        if attempts == 1:
            headers = Message()
            headers["Retry-After"] = "0"
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                headers,
                None,
            )
        return io.BytesIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": "0x1"}).encode()
        )

    monkeypatch.setattr(builder.urllib.request, "urlopen", fake_urlopen)
    client = builder.JsonRpcClient(
        "https://rpc.invalid", timeout_sec=1.0, max_retries=1
    )
    assert client.call("eth_chainId", []) == "0x1"
    assert attempts == 2


def test_json_rpc_client_does_not_retry_non_transient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_urlopen(request: Any, *, timeout: float) -> io.BytesIO:
        nonlocal attempts
        attempts += 1
        headers = Message()
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            headers,
            None,
        )

    monkeypatch.setattr(builder.urllib.request, "urlopen", fake_urlopen)
    client = builder.JsonRpcClient(
        "https://rpc.invalid", timeout_sec=1.0, max_retries=3
    )
    with pytest.raises(urllib.error.HTTPError):
        client.call("eth_chainId", [])
    assert attempts == 1
