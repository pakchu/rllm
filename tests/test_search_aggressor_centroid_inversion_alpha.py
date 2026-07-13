from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import search_aggressor_centroid_inversion_alpha as centroid


def _market(periods: int = 24) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.Series(pd.date_range("2023-01-01", periods=periods, freq="5min"))
    close = np.linspace(100.0, 101.0, periods)
    market = pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.full(periods, 10.0),
            "quote_asset_volume": np.full(periods, 1000.0),
            "taker_buy_base": np.full(periods, 4.0),
            "taker_buy_quote": np.full(periods, 396.0),
        }
    )
    return market, dates


def _state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision": [True] * 8,
            "buy_centroid": [99.0, 99.0, 101.0, 101.0, 99.0, 101.0, 99.0, np.nan],
            "sell_centroid": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "hourly_close": [101.0, 98.0, 102.0, 99.0, 99.5, 100.5, 100.5, 101.0],
            "market_vwap": [100.0] * 8,
            "hourly_open": [100.0] * 8,
            "taker_imbalance": [0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, 0.2],
        }
    )


def test_centroids_reconstruct_completed_hour_execution_prices() -> None:
    market, dates = _market(12)
    state = centroid.build_centroid_state(market, dates)
    row = state.loc[11]
    assert row["decision"]
    assert np.isclose(row["buy_centroid"], 99.0)
    assert np.isclose(row["sell_centroid"], 604.0 / 6.0)
    assert np.isclose(row["market_vwap"], 100.0)
    audit = centroid.validate_centroid_accounting(state)
    assert audit["valid_centroid_hours"] == 1
    assert audit["range_bound_violations"] == 0


def test_state_is_emitted_only_on_completed_minute_55_hours() -> None:
    market, dates = _market()
    state = centroid.build_centroid_state(market, dates)
    assert np.flatnonzero(state["decision"].to_numpy(bool)).tolist() == [11, 23]
    assert state.loc[~state["decision"], "buy_centroid"].isna().all()


def test_primary_topology_is_exact_inverted_terminal_ordering() -> None:
    long_active, short_active = centroid.topology_masks(_state())
    assert np.flatnonzero(long_active).tolist() == [0, 6]
    assert np.flatnonzero(short_active).tolist() == [1]


def test_ordinary_topology_is_the_buy_sell_label_swap_control() -> None:
    state = _state()
    ordinary = centroid.topology_masks(state, "ordinary_terminal")
    swapped = state.copy()
    swapped[["buy_centroid", "sell_centroid"]] = swapped[
        ["sell_centroid", "buy_centroid"]
    ].to_numpy()
    primary_after_swap = centroid.topology_masks(swapped)
    np.testing.assert_array_equal(ordinary[0], primary_after_swap[0])
    np.testing.assert_array_equal(ordinary[1], primary_after_swap[1])


def test_primary_is_subset_of_terminal_and_inverted_midpoint_controls() -> None:
    primary_long, primary_short = centroid.topology_masks(_state())
    terminal_long, terminal_short = centroid.topology_masks(
        _state(), "terminal_any_order"
    )
    midpoint_long, midpoint_short = centroid.topology_masks(
        _state(), "inverted_midpoint"
    )
    assert np.all(~primary_long | terminal_long)
    assert np.all(~primary_short | terminal_short)
    assert np.all(~primary_long | midpoint_long)
    assert np.all(~primary_short | midpoint_short)


def test_inversion_momentum_controls_keep_order_but_relax_terminal_settlement() -> None:
    state = _state()
    primary_long, primary_short = centroid.topology_masks(state)
    vwap_long, vwap_short = centroid.topology_masks(
        state, "inversion_plus_vwap_direction"
    )
    return_long, return_short = centroid.topology_masks(
        state, "inversion_plus_hourly_return"
    )
    assert np.all(~primary_long | vwap_long)
    assert np.all(~primary_short | vwap_short)
    assert np.all(~primary_long | return_long)
    assert np.all(~primary_short | return_short)


def test_direction_flip_is_exact_opposite() -> None:
    primary_long, primary_short = centroid.topology_masks(_state())
    flip_long, flip_short = centroid.topology_masks(_state(), flip=True)
    np.testing.assert_array_equal(flip_long, primary_short)
    np.testing.assert_array_equal(flip_short, primary_long)


def test_future_suffix_cannot_change_completed_prefix_centroids() -> None:
    market, dates = _market(24)
    first = centroid.build_centroid_state(market, dates)
    changed = market.copy()
    changed.loc[12:, [
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]] *= 1000.0
    second = centroid.build_centroid_state(changed, dates)
    columns = ["buy_centroid", "sell_centroid", "market_vwap"]
    np.testing.assert_allclose(
        first.loc[:11, columns], second.loc[:11, columns], equal_nan=True
    )


def test_missing_inputs_and_unknown_variants_are_rejected() -> None:
    market, dates = _market()
    with pytest.raises(KeyError):
        centroid.build_centroid_state(market.drop(columns="taker_buy_base"), dates)
    with pytest.raises(KeyError):
        centroid.topology_masks(_state(), "future")


def test_detailed_support_reports_executable_side_counts(monkeypatch) -> None:
    monkeypatch.setitem(
        centroid.WINDOWS, "sample", ("2023-01-01", "2023-01-01 00:40")
    )
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    monkeypatch.setattr(centroid, "HOLD_BARS", 2)
    long_active = np.array(
        [True, True, False, False, False, False, True, False, False, False]
    )
    short_active = np.array(
        [False, False, False, False, True, False, False, False, False, False]
    )
    counts = centroid.detailed_support_counts(
        dates, long_active, short_active, window="sample"
    )
    assert counts == {
        "raw": 4,
        "raw_long": 3,
        "raw_short": 1,
        "strict_executable": 2,
        "strict_executable_long": 1,
        "strict_executable_short": 1,
    }


def test_support_only_cannot_simulate_or_write_results(monkeypatch, tmp_path) -> None:
    market, dates = _market()
    result_path = tmp_path / "must_not_exist.json"
    monkeypatch.setattr(centroid, "load_pre2024", lambda: (market, dates))
    monkeypatch.setattr(centroid, "RESULT_PATH", result_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("support-only crossed the outcome boundary")

    monkeypatch.setattr(centroid, "_future_extreme", forbidden)
    monkeypatch.setattr(centroid, "simulate", forbidden)
    output = centroid.run(support_only=True)
    assert output["support_only"] is True
    assert not result_path.exists()


def test_loader_keeps_returned_frame_pre2024_and_complete() -> None:
    _, dates = centroid.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()
