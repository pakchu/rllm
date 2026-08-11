from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from training import build_high_volatility_ethereum_staking_deposit_pressure_relay_support as b
from training import preregister_high_volatility_ethereum_staking_deposit_pressure_relay as p


def _field(value: bytes) -> bytes:
    return len(value).to_bytes(32, "big") + value + bytes((-len(value)) % 32)


def _data() -> str:
    values = (bytes(48), bytes(32), (32_000_000_000).to_bytes(8, "little"), bytes(96), (7).to_bytes(8, "little"))
    offset = 32 * 5
    heads = []
    tails = []
    for value in values:
        encoded = _field(value)
        heads.append(offset.to_bytes(32, "big"))
        tails.append(encoded)
        offset += len(encoded)
    return "0x" + b"".join((*heads, *tails)).hex()


def _hash(number: int) -> str:
    return "0x" + f"{number:064x}"


def _log() -> dict[str, Any]:
    return {"address": p.DEPOSIT_CONTRACT, "topics": [p.DEPOSIT_EVENT_TOPIC], "data": _data(), "blockNumber": "0x64", "blockHash": _hash(100), "transactionHash": _hash(200), "transactionIndex": "0x1", "logIndex": "0x2", "removed": False}


def test_deposit_event_abi_and_identity_are_frozen():
    fields = b.decode_deposit_event_data(_data())
    assert tuple(map(len, fields)) == b.ABI_LENGTHS
    row = b.normalize_log(_log(), 90, 110)
    assert row["deposit_index_little_endian"] == 7
    assert row["block_number"] == 100
    assert len(row["data_sha256"]) == 64


@pytest.mark.parametrize("change", [{"topics": []}, {"address": "0x" + "11" * 20}, {"removed": True}, {"data": "0x00"}])
def test_deposit_event_fails_closed(change: dict[str, Any]):
    with pytest.raises((ValueError, RuntimeError)):
        b.normalize_log(_log() | change, 90, 110)


def test_daily_pressure_side_and_causal_clock():
    events = [{"source_day": "2022-01-01"}] * 2 + [{"source_day": "2022-01-08"}] * 5
    frame = b.build_daily_panel(events)
    row = frame.loc[frame.source_day.eq(pd.Timestamp("2022-01-08T00:00:00Z"))].iloc[0]
    assert row.pressure_change == 3
    assert row.result_side == 1
    assert row.decision_time == pd.Timestamp("2022-01-09T12:00:00Z")


def test_clock_reservation_and_split_containment():
    frame = pd.DataFrame({"decision_time": pd.to_datetime(["2023-07-01T12:00:00Z", "2023-07-02T12:00:00Z"]), "result_side": [1, -1], "raw_day_over_day_side": [1, -1], "btc_variation_rank": [0.9, 0.9], "daily_count": [10, 8], "pressure_change": [2, -2], "btc_realized_variation": [0.1, 0.2]})
    clock = b.build_clock(frame)
    assert len(clock) == 2
    assert clock.side.tolist() == [1, -1]
    assert clock.candidate.unique().tolist() == ["HVESDP-24"]
