from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.same_millisecond_cascade import BAR_COLUMNS, aggregate_same_millisecond_five_minute
from training.build_binance_aggtrade_microstructure import RAW_COLUMNS


def _row(
    agg_id: int,
    time_ms: int,
    *,
    price: float,
    quantity: float,
    maker: bool,
    first_trade_id: int | None = None,
    last_trade_id: int | None = None,
) -> list[object]:
    first = agg_id if first_trade_id is None else first_trade_id
    last = first if last_trade_id is None else last_trade_id
    return [agg_id, price, quantity, first, last, time_ms, maker]


def test_largest_same_millisecond_group_is_selected_and_scored() -> None:
    base = 1_609_459_200_000
    frame = pd.DataFrame(
        [
            _row(1, base, price=100.0, quantity=1.0, maker=False, first_trade_id=10),
            _row(2, base + 1, price=101.0, quantity=2.0, maker=False, first_trade_id=11),
            _row(3, base + 1, price=102.0, quantity=2.0, maker=False, first_trade_id=12),
            _row(4, base + 2, price=101.0, quantity=1.0, maker=True, first_trade_id=13),
        ],
        columns=RAW_COLUMNS,
    )
    output = aggregate_same_millisecond_five_minute(frame)
    row = output.iloc[0]
    assert row["millisecond_group_count"] == 3
    assert row["collision_group_count"] == 1
    assert row["max_ms_transact_time"] == base + 1
    assert row["max_ms_event_count"] == 2
    assert row["max_ms_side"] == 1
    assert row["max_ms_pre_group_price"] == 100.0
    assert row["max_ms_last_price"] == 102.0
    assert row["max_ms_sweep_bp"] > 0.0
    assert row["max_ms_score"] > 0.0
    assert tuple(output.columns) == BAR_COLUMNS


def test_max_notional_tie_uses_earliest_millisecond() -> None:
    base = 1_609_459_200_000
    frame = pd.DataFrame(
        [
            _row(1, base, price=100.0, quantity=1.0, maker=False),
            _row(2, base + 1, price=100.0, quantity=2.0, maker=False),
            _row(3, base + 2, price=100.0, quantity=2.0, maker=True),
        ],
        columns=RAW_COLUMNS,
    )
    output = aggregate_same_millisecond_five_minute(frame)
    assert output.loc[0, "max_ms_transact_time"] == base + 1
    assert output.loc[0, "max_ms_side"] == 1


def test_first_group_of_bar_has_zero_sweep_without_cross_bar_state() -> None:
    base = 1_609_459_200_000
    frame = pd.DataFrame(
        [
            _row(1, base, price=100.0, quantity=1.0, maker=False),
            _row(2, base + 300_000, price=110.0, quantity=2.0, maker=False),
        ],
        columns=RAW_COLUMNS,
    )
    output = aggregate_same_millisecond_five_minute(frame)
    assert output["max_ms_sweep_bp"].tolist() == [0.0, 0.0]
    assert output["max_ms_score"].tolist() == [0.0, 0.0]
    assert np.isfinite(output.drop(columns="date").to_numpy(float)).all()


def test_mixed_side_group_coherence_and_side_are_not_overstated() -> None:
    base = 1_609_459_200_000
    frame = pd.DataFrame(
        [
            _row(1, base, price=100.0, quantity=1.0, maker=False),
            _row(2, base + 1, price=101.0, quantity=1.0, maker=False),
            _row(3, base + 1, price=101.0, quantity=1.0, maker=True),
        ],
        columns=RAW_COLUMNS,
    )
    output = aggregate_same_millisecond_five_minute(frame)
    assert output.loc[0, "max_ms_event_count"] == 2
    assert output.loc[0, "max_ms_coherence"] == 0.0
    assert output.loc[0, "max_ms_side"] == 0
    assert output.loc[0, "max_ms_score"] == 0.0


def test_exact_midnight_event_starts_a_new_five_minute_bar() -> None:
    midnight = 1_609_545_600_000
    frame = pd.DataFrame(
        [
            _row(1, midnight - 1, price=100.0, quantity=1.0, maker=False),
            _row(2, midnight, price=101.0, quantity=1.0, maker=False),
        ],
        columns=RAW_COLUMNS,
    )
    output = aggregate_same_millisecond_five_minute(frame)
    assert output["date"].tolist() == [
        pd.Timestamp("2021-01-01 23:55:00"),
        pd.Timestamp("2021-01-02 00:00:00"),
    ]
    assert output["max_ms_score"].tolist() == [0.0, 0.0]
