from __future__ import annotations

import pandas as pd
import pytest

from preprocessing.quantity_lattice_cohort import (
    BAR_COLUMNS,
    aggregate_quantity_lattice_five_minute,
)


def _frame(quantities: list[float], makers: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "agg_trade_id": range(1, len(quantities) + 1),
            "quantity": quantities,
            "transact_time": [1_577_836_800_000 + i for i in range(len(quantities))],
            "is_buyer_maker": makers,
        }
    )


def test_cohorts_are_mutually_exclusive_and_score_coarse_against_fine() -> None:
    frame = _frame(
        [0.100, 0.200, 0.010, 0.020, 0.003, 0.007],
        [False, False, True, True, True, True],
    )
    output = aggregate_quantity_lattice_five_minute(frame)
    row = output.iloc[0]

    assert tuple(output.columns) == BAR_COLUMNS
    assert row["total_quantity_mbtc"] == 340
    assert row["coarse_event_count"] == 2
    assert row["coarse_quantity_mbtc"] == 300
    assert row["coarse_signed_quantity_mbtc"] == 300
    assert row["medium_event_count"] == 2
    assert row["medium_quantity_mbtc"] == 30
    assert row["fine_event_count"] == 2
    assert row["fine_quantity_mbtc"] == 10
    assert row["fine_signed_quantity_mbtc"] == -10
    assert row["coarse_side"] == 1
    assert row["coarse_coherence"] == pytest.approx(1.0)
    assert row["fine_signed_share"] == pytest.approx(-1.0)
    assert row["cohort_opposition"] == pytest.approx(1.0)
    assert row["qlcd_score"] == pytest.approx(300 / 340)


def test_aligned_fine_flow_has_zero_opposition_and_score() -> None:
    output = aggregate_quantity_lattice_five_minute(
        _frame([0.100, 0.003], [False, False])
    )
    assert output.loc[0, "coarse_side"] == 1
    assert output.loc[0, "fine_signed_share"] == pytest.approx(1.0)
    assert output.loc[0, "cohort_opposition"] == 0.0
    assert output.loc[0, "qlcd_score"] == 0.0


def test_sub_millibtc_quantity_fails_closed() -> None:
    with pytest.raises(ValueError, match="0.001-BTC lattice"):
        aggregate_quantity_lattice_five_minute(_frame([0.0005], [False]))


def test_string_maker_flags_are_parsed_and_unknown_values_fail_closed() -> None:
    frame = _frame([0.100, 0.003], [False, True])
    frame["is_buyer_maker"] = ["false", "true"]
    output = aggregate_quantity_lattice_five_minute(frame)
    assert output.loc[0, "coarse_signed_quantity_mbtc"] == 100
    assert output.loc[0, "fine_signed_quantity_mbtc"] == -3

    frame.loc[1, "is_buyer_maker"] = "unknown"
    with pytest.raises(ValueError, match="unknown value"):
        aggregate_quantity_lattice_five_minute(frame)


def test_bars_are_utc_floored_and_sorted() -> None:
    frame = _frame([0.100, 0.003], [False, True])
    frame.loc[0, "transact_time"] += 300_000
    output = aggregate_quantity_lattice_five_minute(frame)
    assert len(output) == 2
    assert output["date"].is_monotonic_increasing
    assert output["source_observed"].all()
