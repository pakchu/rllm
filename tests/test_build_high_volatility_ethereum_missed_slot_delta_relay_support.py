import pandas as pd
import pytest

from training import build_high_volatility_ethereum_missed_slot_delta_relay_support as b


def raw_header(number: int, timestamp: int, parent: str | None = None) -> dict:
    block_hash = "0x" + f"{number:064x}"
    return {
        "number": hex(number),
        "hash": block_hash,
        "parentHash": parent or "0x" + f"{number - 1:064x}",
        "timestamp": hex(timestamp),
    }


def test_header_parser_and_first_block_search() -> None:
    headers = {number: b.parse_header(raw_header(number, 1_000 + 12 * number)) for number in range(1, 12)}
    found = b.first_block_at_or_after(headers.__getitem__, 1_061, 1, 11)
    assert found.number == 6
    assert found.timestamp == 1_072
    with pytest.raises(RuntimeError, match="bracket"):
        b.first_block_at_or_after(headers.__getitem__, 999, 1, 11)


def test_prior_rank_excludes_current() -> None:
    rank = b.prior_rank(pd.Series([1.0, 2.0, 3.0]), 2, 2)
    assert rank.iloc[:2].isna().all()
    assert rank.iloc[2] == 1.0


def states() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_day": pd.to_datetime(["2023-07-03", "2023-07-04", "2023-07-05"], utc=True),
            "decision_time": pd.to_datetime(["2023-07-04T00:20Z", "2023-07-05T00:20Z", "2023-07-06T00:20Z"]),
            "produced_blocks": [7198, 7194, 7199],
            "missed_slots": [2, 6, 1],
            "missed_change": [2.0, 4.0, -5.0],
            "missed_change_rank": [0.9, 0.9, 0.9],
            "btc_variation": [0.01, 0.01, 0.01],
            "btc_variation_rank": [0.9, 0.9, 0.9],
            "state_valid": [True, True, True],
        }
    )


def test_frozen_signal_sides_controls_and_clock() -> None:
    frame = states()
    active, side = b.signal(frame, "primary")
    assert active.all()
    assert side.tolist() == [-1, -1, 1]
    _, flipped = b.signal(frame, "direction_flip")
    assert flipped.tolist() == [1, 1, -1]
    _, forced = b.signal(frame, "same_clock_forced_long")
    assert forced.tolist() == [1, 1, 1]
    clock = b.build_clock(frame)
    assert clock.entry_time.dt.strftime("%Y-%m-%dT%H:%MZ").iloc[0] == "2023-07-04T00:25Z"
    assert len(b.CONTROLS) == 6


def test_support_fails_closed() -> None:
    assert b.stats(pd.DataFrame(columns=b.COLUMNS), "train")["events"] == 0
    assert b.MINIMUM == {"train": 8, "test": 12, "eval": 12, "final": 8}
