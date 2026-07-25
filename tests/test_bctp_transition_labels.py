from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import bctp_transition_labels as labels
from training import preregister_block_clearing_target_position_mdp as prereg


def _states() -> pd.DataFrame:
    records = []
    for index, entry in enumerate(
        [
            "2020-01-01T00:05:00Z",
            "2020-01-01T00:15:00Z",
            "2020-01-01T00:25:00Z",
        ]
    ):
        row = {
            "sequence_id": f"id-{index}",
            "entry_time": entry,
            "source_signal_id_m2": f"m2-{index}",
            "source_signal_id_m1": f"m1-{index}",
            "source_signal_id_s0": f"s0-{index}",
            "source_signature": f"signature-{index}",
        }
        for column in prereg.SOURCE_TOKEN_COLUMNS:
            row[column] = f"VALUE-{index}"
        records.append(row)
    frame = pd.DataFrame(
        records,
        columns=pd.Index(prereg.SOURCE_SEQUENCE_COLUMNS),
    )
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    return frame


def _market() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01T00:00:00Z",
        "2020-01-01T00:40:00Z",
        freq="5min",
        inclusive="left",
    )
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        }
    )


def _funding() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "funding_time",
            "symbol",
            "funding_rate",
            "settlement_mark_price",
        ]
    )


def test_reward_tensor_has_only_first_flat_reachable() -> None:
    result = labels.build_reward_tensor(
        _states(),
        _market(),
        _funding(),
        start=pd.Timestamp("2020-01-01T00:00:00Z"),
        end=pd.Timestamp("2020-01-01T00:40:00Z"),
    )
    flat = labels.POSITION_ORDER.index("POSITION_FLAT")
    assert result["reachable_mask"][0].sum() == 1
    assert result["reachable_mask"][0, flat]
    assert result["reachable_mask"][1:].all()
    assert result["terminal"].tolist() == [False, False, True]
    assert len(result["ledger"]) == 21
    assert result["action_order"] == (0.0, -0.5, 0.5)


def test_flat_market_makes_flat_action_best_for_flat_position() -> None:
    result = labels.build_reward_tensor(
        _states(),
        _market(),
        _funding(),
        start=pd.Timestamp("2020-01-01T00:00:00Z"),
        end=pd.Timestamp("2020-01-01T00:40:00Z"),
    )
    flat_position = labels.POSITION_ORDER.index("POSITION_FLAT")
    flat_action = labels.ACTION_ORDER.index(0.0)
    first_rewards = result["reward_tensor"][0, flat_position]
    assert first_rewards.argmax() == flat_action
    assert first_rewards[flat_action] == pytest.approx(0.0)


def test_transition_ledger_is_deterministic_and_write_once(tmp_path) -> None:
    result = labels.build_reward_tensor(
        _states(),
        _market(),
        _funding(),
        start=pd.Timestamp("2020-01-01T00:00:00Z"),
        end=pd.Timestamp("2020-01-01T00:40:00Z"),
    )
    frame = result["ledger"]
    left = labels.deterministic_gzip_csv_bytes(frame)
    right = labels.deterministic_gzip_csv_bytes(frame)
    assert left == right
    output = tmp_path / "ledger.csv.gz"
    first = labels.write_ledger_once(output, frame)
    second = labels.write_ledger_once(output, frame)
    assert first == second
    loaded = labels.read_ledger(output)
    assert len(loaded) == len(frame)
    assert labels.ledger_frame_hash(loaded) == first["frame_hash"]
    arrays = labels.arrays_from_ledger(result["states"], loaded)
    assert arrays["reachable_mask"].tolist() == result[
        "reachable_mask"
    ].tolist()
    assert arrays["terminal"].tolist() == result["terminal"].tolist()
    assert np.allclose(
        arrays["reward_tensor"][arrays["reachable_mask"]],
        result["reward_tensor"][result["reachable_mask"]],
    )
    changed = frame.copy()
    changed.loc[0, "reward"] = 1.0
    with pytest.raises(RuntimeError, match="drift"):
        labels.write_ledger_once(output, changed)
