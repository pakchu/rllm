from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_liquidation_scar_field_alpha import (
    build_causal_inputs,
    fit_threshold,
    replay_scar_field,
    scar_signals,
)


def test_open_interest_is_delayed_one_complete_bar() -> None:
    rows = 2_400
    market = pd.DataFrame(
        {
            "close": np.exp(np.linspace(8.0, 8.1, rows)),
            "open_interest": 200.0 + 20.0 * np.sin(np.arange(rows, dtype=float) / 17.0),
            "open_interest_available": 1.0,
            "quote_asset_volume": 100.0,
            "taker_buy_quote": 50.0,
        }
    )
    baseline = build_causal_inputs(market)
    changed = market.copy()
    changed.loc[2_200:, "open_interest"] = 1.0
    replay = build_causal_inputs(changed)

    assert np.isfinite(baseline.loc[2_200, "contraction_z"])
    assert baseline.loc[2_200, "contraction_z"] == replay.loc[2_200, "contraction_z"]
    assert baseline.loc[2_201, "contraction_z"] != replay.loc[2_201, "contraction_z"]
    assert replay.loc[2_200, "oi_source_delay_bars"] == 1.0


def test_fit_threshold_ignores_2023_selection_values() -> None:
    fit_dates = pd.date_range("2020-06-01", periods=6_000, freq="3h")
    selection_dates = pd.date_range("2023-01-01", periods=200, freq="3h")
    dates = pd.Series(fit_dates.append(selection_dates))
    values = pd.Series(np.r_[np.arange(6_000, dtype=float), np.full(200, 1_000_000.0)])

    assert fit_threshold(values, dates, 0.9) == float(pd.Series(np.arange(6_000)).quantile(0.9))


def test_scar_query_occurs_before_current_deposit() -> None:
    inputs = pd.DataFrame(
        {
            "log_price": np.log([100.0, 100.0, 100.0]),
            "ret_12": [np.nan, np.nan, np.nan],
            "contraction_z": [2.0, np.nan, np.nan],
            "flow_z": [2.0, np.nan, np.nan],
        }
    )
    dates = pd.Series(pd.date_range("2022-01-01", periods=3, freq="5min"))

    field = replay_scar_field(
        inputs,
        dates,
        bin_width=0.001,
        half_life=288,
        contraction_threshold=1.0,
    )

    assert field.loc[0, "up_scar_ahead"] == 0.0
    assert field.loc[0, "deposit_mass"] > 0.0
    assert field.loc[1, "up_scar_ahead"] > 0.0


def test_scar_field_prefix_does_not_depend_on_future_suffix() -> None:
    prefix = pd.DataFrame(
        {
            "log_price": np.log(np.linspace(100.0, 101.0, 30)),
            "ret_12": np.r_[np.full(12, np.nan), np.ones(18) / 100.0],
            "contraction_z": np.where(np.arange(30) % 7 == 0, 2.0, np.nan),
            "flow_z": np.where(np.arange(30) % 7 == 0, -2.0, np.nan),
        }
    )
    full = pd.concat(
        [
            prefix,
            pd.DataFrame(
                {
                    "log_price": np.log([1_000_000.0] * 10),
                    "ret_12": 100.0,
                    "contraction_z": 100.0,
                    "flow_z": 100.0,
                }
            ),
        ],
        ignore_index=True,
    )
    prefix_dates = pd.Series(pd.date_range("2022-01-01", periods=len(prefix), freq="5min"))
    full_dates = pd.Series(pd.date_range("2022-01-01", periods=len(full), freq="5min"))

    expected = replay_scar_field(prefix, prefix_dates, bin_width=0.001, half_life=288, contraction_threshold=1.0)
    actual = replay_scar_field(full, full_dates, bin_width=0.001, half_life=288, contraction_threshold=1.0)
    pd.testing.assert_frame_equal(actual.iloc[: len(prefix)].reset_index(drop=True), expected.reset_index(drop=True))


def test_permeability_and_fade_are_exact_opposite_mappings() -> None:
    features = pd.DataFrame(
        {
            "up_scar_ahead": [0.0, 2.0, 2.0, 0.0],
            "down_scar_below": [0.0, 0.0, 0.0, 2.0],
            "scalar_up": 0.0,
            "scalar_down": 0.0,
            "ret_12": [1.0, 1.0, 1.0, -1.0],
        }
    )

    permeability_long, permeability_short = scar_signals(
        features,
        up_threshold=1.0,
        down_threshold=1.0,
        mapping="permeability",
    )
    fade_long, fade_short = scar_signals(
        features,
        up_threshold=1.0,
        down_threshold=1.0,
        mapping="fade",
    )
    np.testing.assert_array_equal(permeability_long, fade_short)
    np.testing.assert_array_equal(permeability_short, fade_long)
