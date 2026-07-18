from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_build_binance_cross_venue_minute_leadership import _archive, _venue_rows
from training import build_binance_cross_venue_minute_dispersion as builder
from training.build_binance_cross_venue_minute_leadership import read_archive


def _frame(
    *,
    quote: tuple[float, ...],
    trades: tuple[int, ...],
    flows: tuple[float, ...],
    returns: tuple[float, ...],
    base_price: float = 100.0,
) -> pd.DataFrame:
    rows = _venue_rows(base_price=base_price, flows=flows, returns=returns)
    for row, q, n, flow in zip(rows, quote, trades, flows, strict=True):
        row[5] = q / float(row[1])
        row[7] = q
        row[8] = n
        row[9] = row[5] * (1.0 + flow) / 2.0
        row[10] = q * (1.0 + flow) / 2.0
    return read_archive(_archive(rows, header=False), venue="spot")


def test_minute_dispersion_descriptors_match_hand_calculation() -> None:
    spot = _frame(
        quote=(1.0, 1.0, 1.0, 1.0, 6.0),
        trades=(1, 1, 1, 1, 1),
        flows=(0.8, 0.8, -0.8, -0.8, 0.8),
        returns=(0.0, 0.001, 0.001, 0.001, 0.001),
    )
    um = _frame(
        quote=(2.0, 2.0, 2.0, 2.0, 2.0),
        trades=(2, 2, 2, 2, 2),
        flows=(0.2, 0.2, 0.2, 0.2, 0.2),
        returns=(0.0, 0.0005, 0.0005, 0.0005, 0.0005),
        base_price=101.0,
    )
    row = builder.aggregate_cross_venue_minute_dispersion(spot, um).iloc[0]

    assert row["minute_dispersion_feature_valid"] == np.bool_(True)
    assert row["spot_quote_time_hhi"] == pytest.approx(40.0 / 100.0)
    assert row["spot_trade_time_hhi"] == pytest.approx(0.2)
    assert row["spot_quote_minus_trade_time_hhi"] == pytest.approx(0.2)
    assert row["um_quote_time_hhi"] == pytest.approx(0.2)
    assert row["spot_flow_sign_switch_rate"] == pytest.approx(0.5)
    assert row["um_flow_sign_switch_rate"] == pytest.approx(0.0)
    assert row["net_flow_sign_agreement"] == pytest.approx(1.0)
    assert row["feature_available_time_utc"] == pd.Timestamp("2023-01-01 00:05:00")
    assert row["trade_earliest_time_utc"] == pd.Timestamp("2023-01-01 00:05:00")


def test_future_bar_cannot_change_completed_bar_features() -> None:
    first = _frame(
        quote=(1, 2, 3, 4, 5),
        trades=(1, 1, 1, 1, 1),
        flows=(0.8, 0.6, 0.4, 0.2, 0.1),
        returns=(0.0, 0.001, 0.001, 0.001, 0.001),
    )
    other = _frame(
        quote=(5, 4, 3, 2, 1),
        trades=(2, 2, 2, 2, 2),
        flows=(-0.8, -0.6, -0.4, -0.2, -0.1),
        returns=(0.0, -0.001, -0.001, -0.001, -0.001),
        base_price=101.0,
    )
    base = builder.aggregate_cross_venue_minute_dispersion(first, other).iloc[0]

    shifted_first = first.copy()
    shifted_other = other.copy()
    for frame in (shifted_first, shifted_other):
        frame["open_time"] += 5 * 60_000
        frame["close_time"] += 5 * 60_000
    extended = builder.aggregate_cross_venue_minute_dispersion(
        pd.concat([first, shifted_first], ignore_index=True),
        pd.concat([other, shifted_other], ignore_index=True),
    ).iloc[0]

    for column in builder.FEATURE_COLUMNS:
        assert extended[column] == pytest.approx(base[column]), column


def test_missing_minute_quarantines_all_features() -> None:
    spot = _frame(
        quote=(1, 1, 1, 1, 1),
        trades=(1, 1, 1, 1, 1),
        flows=(0.1, 0.1, 0.1, 0.1, 0.1),
        returns=(0.0, 0.0, 0.0, 0.0, 0.0),
    ).drop(index=2)
    um = _frame(
        quote=(1, 1, 1, 1, 1),
        trades=(1, 1, 1, 1, 1),
        flows=(0.1, 0.1, 0.1, 0.1, 0.1),
        returns=(0.0, 0.0, 0.0, 0.0, 0.0),
        base_price=101.0,
    )
    expected = pd.date_range("2023-01-01", periods=5, freq="1min")
    row = builder.aggregate_cross_venue_minute_dispersion(
        spot, um, expected_minutes=expected
    ).iloc[0]

    assert row["source_complete"] == np.bool_(False)
    assert row["minute_dispersion_feature_valid"] == np.bool_(False)
    assert row["feature_invalid_reason"].startswith("source_incomplete")
    assert row[list(builder.FEATURE_COLUMNS)].isna().all()


def test_post2023_build_is_fail_closed_without_explicit_unseal(tmp_path) -> None:
    cfg = builder.BuildConfig(
        start="2023-12-01",
        end="2024-02-01",
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="sealed"):
        builder.build(cfg)
