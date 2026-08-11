from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_uniswap_wbtc_liquidity_shock_relay_support as b
from training import preregister_high_volatility_uniswap_wbtc_liquidity_shock_relay as p


def _hash(number: int) -> str:
    return "0x" + f"{number:064x}"


def _address_topic(address: str) -> str:
    return "0x" + "00" * 12 + address.lower()[2:]


def _signed(value: int, bits: int = 256) -> bytes:
    return (value % (1 << bits)).to_bytes(32, "big")


def _swap_data(
    amount0: int = -100_000_000,
    amount1: int = 60_000_000,
    sqrt_price_x96: int = 1 << 96,
    liquidity: int = 10**12,
    tick: int = -12,
) -> str:
    return "0x" + b"".join(
        (
            _signed(amount0),
            _signed(amount1),
            sqrt_price_x96.to_bytes(32, "big"),
            liquidity.to_bytes(32, "big"),
            _signed(tick),
        )
    ).hex()


def _log(**changes: Any) -> dict[str, Any]:
    row = {
        "address": p.POOL,
        "topics": [p.SWAP_TOPIC0, _address_topic(p.WBTC), _address_topic(p.USDC)],
        "data": _swap_data(),
        "blockNumber": "0x64",
        "blockHash": _hash(100),
        "transactionHash": _hash(200),
        "transactionIndex": "0x1",
        "logIndex": "0x2",
        "removed": False,
    }
    row.update(changes)
    return row


class PoolRpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any]]] = []

    def call(self, method: str, params: list[Any]) -> Any:
        self.calls.append((method, params))
        if method == "eth_chainId":
            return "0x1"
        selector = params[0]["data"][2:10]
        values = {
            b.GET_POOL_SELECTOR: p.POOL,
            b.TOKEN0_SELECTOR: p.WBTC,
            b.TOKEN1_SELECTOR: p.USDC,
        }
        if selector == b.FEE_SELECTOR:
            return "0x" + f"{3000:064x}"
        address = values[selector]
        return "0x" + "00" * 12 + address.lower()[2:]

    def batch(self, requests: Any) -> list[Any]:
        raise AssertionError("not used")


def test_frozen_preregistration_source_and_artifact_hashes_are_verified():
    registration = b.verify_frozen_preregistration()
    assert registration == p.build()
    assert b.sha256(b.PREREG_SOURCE) == b.PREREG_SOURCE_SHA256
    assert b.sha256(p.DEFAULT_OUTPUT) == b.PREREG_ARTIFACT_SHA256


def test_role_to_host_pool_identity_calls_are_exact():
    rpc = PoolRpc()
    result = b.verify_pool_identity(rpc, "primary")
    assert result["role"] == "primary"
    assert result["chain_id"] == 1
    assert result["pool"] == p.POOL.lower()
    calls = [params for method, params in rpc.calls if method == "eth_call"]
    assert calls[0][0]["to"] == p.FACTORY
    assert calls[0][0]["data"].startswith("0x" + b.GET_POOL_SELECTOR)
    assert calls[0][0]["data"].endswith(f"{3000:064x}")
    assert [call[0]["data"] for call in calls[1:]] == [
        "0x" + b.TOKEN0_SELECTOR,
        "0x" + b.TOKEN1_SELECTOR,
        "0x" + b.FEE_SELECTOR,
    ]
    assert all(call[1] == "latest" for call in calls)


def test_swap_normalization_binds_full_abi_and_identity():
    row = b.normalize_swap_log(_log(), 90, 110)
    assert row["amount0_raw"] == -100_000_000
    assert row["amount1_raw"] == 60_000_000
    assert row["tick"] == -12
    assert row["sender"] == p.WBTC.lower()
    assert row["recipient"] == p.USDC.lower()
    assert row["topics"][0] == p.SWAP_TOPIC0
    assert len(row["topics_sha256"]) == len(row["data_sha256"]) == 64
    assert (row["block_hash"], row["transaction_hash"], row["log_index"]) == (
        _hash(100),
        _hash(200),
        2,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"address": "0x" + "11" * 20},
        {"topics": [p.SWAP_TOPIC0]},
        {"topics": [_hash(9), _address_topic(p.WBTC), _address_topic(p.USDC)]},
        {"topics": [p.SWAP_TOPIC0, "0x01" + "00" * 31, _address_topic(p.USDC)]},
        {"data": "0x00"},
        {"data": _swap_data(amount0=1, amount1=2)},
        {"data": _swap_data(amount0=0)},
        {"data": _swap_data(sqrt_price_x96=1 << 160)},
        {"data": _swap_data(liquidity=1 << 128)},
        {"removed": True},
        {"blockNumber": "0x6e"},
    ],
)
def test_swap_normalization_fails_closed(change: dict[str, Any]):
    with pytest.raises((ValueError, RuntimeError)):
        b.normalize_swap_log(_log(**change), 90, 110)


def test_noncanonical_int24_sign_extension_is_rejected():
    data = bytearray.fromhex(_swap_data()[2:])
    data[-32:] = ((1 << 24) - 12).to_bytes(32, "big")
    with pytest.raises(ValueError, match="noncanonical"):
        b.normalize_swap_log(_log(data="0x" + data.hex()), 90, 110)


class LogRpc:
    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []

    def call(self, method: str, params: list[Any]) -> Any:
        assert method == "eth_getLogs"
        query = params[0]
        self.ranges.append((int(query["fromBlock"], 16), int(query["toBlock"], 16)))
        assert query["address"] == p.POOL
        assert query["topics"] == [p.SWAP_TOPIC0]
        return []

    def batch(self, requests: Any) -> list[Any]:
        raise AssertionError("not used")


def test_log_queries_never_exceed_2000_inclusive_blocks():
    rpc = LogRpc()
    assert b.fetch_logs(rpc, 10, 4511) == []
    assert rpc.ranges == [(10, 2009), (2010, 4009), (4010, 4510)]
    assert all(last - first + 1 <= 2000 for first, last in rpc.ranges)
    with pytest.raises(ValueError):
        b.fetch_logs(rpc, 0, 1, 2001)


def _boundary(day: str, first: int, last: int) -> dict[str, Any]:
    return {
        "source_day": day,
        "start": {"number": first, "hash": _hash(first), "parent_hash": _hash(first - 1), "timestamp": 1},
        "end_exclusive": {"number": last, "hash": _hash(last), "parent_hash": _hash(last - 1), "timestamp": 2},
        "finality": {
            "descendants": 64,
            "tip": {"number": last + 64, "hash": _hash(last + 64), "parent_hash": _hash(last + 63), "timestamp": 3},
        },
    }


def _normalized(amount0: int, block: int, tx_index: int, log_index: int) -> dict[str, Any]:
    return {
        "amount0_raw": amount0,
        "amount1_raw": -amount0 * 10,
        "block_number": block,
        "block_hash": _hash(block),
        "transaction_hash": _hash(block * 100 + tx_index),
        "transaction_index": tx_index,
        "log_index": log_index,
        "topics_sha256": "a" * 64,
        "data_sha256": "b" * 64,
    }


def test_daily_largest_wbtc_tie_break_and_zero_log_day_are_deterministic():
    boundaries = [_boundary("2023-01-01", 100, 200), _boundary("2023-01-02", 200, 300)]
    logs = [
        _normalized(500, 150, 2, 1),
        _normalized(-500, 150, 1, 9),
        _normalized(500, 149, 8, 8),
        _normalized(200, 151, 0, 0),
    ]
    panel = b.build_daily_panel(boundaries, logs)
    winner = panel.iloc[0]
    assert winner.amount0_raw == 500
    assert (winner.block_number, winner.transaction_index, winner.log_index) == (149, 8, 8)
    assert winner.result_side == -1
    assert winner.daily_net_amount0_raw == 700
    assert winner.daily_net_side == -1
    zero = panel.iloc[1]
    assert zero.source_valid
    assert zero.swap_count == 0
    assert np.isnan(zero.abs_wbtc_amount)
    assert zero.confirmation_tip_block == 364


def test_strict_prior_midrank_excludes_current_and_skips_missing_values():
    values = pd.Series([1.0, np.nan, 3.0, 3.0, 2.0])
    ranks = b.strict_prior_midrank(values, lookback=3, minimum=2)
    assert np.isnan(ranks.iloc[0]) and np.isnan(ranks.iloc[1])
    assert np.isnan(ranks.iloc[2])
    assert ranks.iloc[3] == 0.75
    assert ranks.iloc[4] == 1 / 3


def test_btc_variation_uses_exact_half_open_1440_open_close_bars():
    decision = pd.Timestamp("2024-01-02T12:00:00Z")
    timestamps = pd.date_range(decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left")
    bars = pd.DataFrame({"ts": timestamps, "open": 100.0, "close": 101.0})
    result = b.calculate_variation(pd.Series([decision]), bars)
    expected = np.sqrt(1440 * np.log(1.01) ** 2)
    assert result.iloc[0].btc_realized_variation == pytest.approx(expected)
    with pytest.raises(RuntimeError, match="invalid BTC 1m window"):
        b.calculate_variation(pd.Series([decision]), bars.iloc[:-1])
    with pytest.raises(ValueError, match="only ts/open/close"):
        b.calculate_variation(pd.Series([decision]), bars.assign(high=102.0))


def _clock_features() -> pd.DataFrame:
    decisions = pd.to_datetime(
        ["2023-07-01T12:00:00Z", "2023-07-02T12:00:00Z", "2023-07-03T11:59:00Z"]
    )
    return pd.DataFrame(
        {
            "source_day": decisions.floor("D") - pd.Timedelta(days=1),
            "decision_time": decisions,
            "source_valid": True,
            "amount0_raw": [-100, 200, -300],
            "abs_wbtc_amount": [1.0, 2.0, 3.0],
            "shock_magnitude_rank": [0.9, 0.9, 0.9],
            "result_side": [1, -1, 1],
            "daily_net_side": [-1, 1, -1],
            "btc_realized_variation": [0.1, 0.2, 0.3],
            "btc_variation_rank": [0.9, 0.9, 0.9],
            "block_number": [1, 2, 3],
            "transaction_index": [0, 0, 0],
            "log_index": [0, 1, 2],
        }
    )


def test_global_half_open_clock_allows_equal_boundary_and_reserves_overlap():
    clock = b.build_clock(_clock_features())
    assert len(clock) == 2
    assert clock.side.tolist() == [1, -1]
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert clock.candidate.unique().tolist() == ["HVUWLS-24"]
    assert clock.split.unique().tolist() == ["train"]


def test_five_frozen_diagnostics_change_only_the_preregistered_axis():
    frame = _clock_features().iloc[:2].copy()
    frame.loc[0, "btc_variation_rank"] = 0.1
    assert len(b.CONTROLS) == 5
    assert len(b.build_clock(frame, "no_btc_volatility_gate")) == 2
    primary = b.build_clock(frame)
    flipped = b.build_clock(frame, "liquidity_shock_direction_flip")
    forced = b.build_clock(frame, "same_clock_forced_long")
    net = b.build_clock(frame, "daily_net_wbtc_flow")
    assert flipped.side.tolist() == [-value for value in primary.side.tolist()]
    assert forced.side.tolist() == [1] * len(primary)
    assert net.side.tolist() == [1]
    stale = b.build_clock(_clock_features(), "one_day_stale_liquidity_shock")
    assert stale.amount0_raw.tolist() == [-100]


def test_support_gates_are_frozen():
    assert b.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
    rows = pd.DataFrame(
        {
            "split": ["train"] * 5,
            "side": [1, 1, 1, 1, -1],
            "entry_time": pd.to_datetime(
                ["2023-07-01", "2023-07-02", "2023-08-01", "2023-09-01", "2023-10-01"], utc=True
            ),
        }
    )
    values = b.stats(rows, "train")
    assert values["minority_side_share"] == 0.2
    assert values["max_month_share"] == 0.4
