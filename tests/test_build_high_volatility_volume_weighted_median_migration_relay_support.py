import math

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_volume_weighted_median_migration_relay_support as support


def bars_for_window(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="1min")
    minute = np.arange(periods)
    open_ = 100.0 + (minute % 11) * 0.01
    close = open_ * np.where(minute % 2, 1.0001, 0.9999)
    return pd.DataFrame({
        "ts": ts,
        "open": open_,
        "high": np.maximum(open_, close) + 0.02,
        "low": np.minimum(open_, close) - 0.02,
        "close": close,
        "volume": np.ones(periods),
        "quote_asset_volume": np.full(periods, 100.0),
    })


def minimal_states(decisions: pd.DatetimeIndex) -> pd.DataFrame:
    size = len(decisions)
    return pd.DataFrame({
        "decision_time": decisions,
        "block_valid": True,
        "variation_valid": True,
        "source_valid": True,
        "first_half_weighted_median": np.full(size, 100.0),
        "second_half_weighted_median": np.full(size, 110.0),
        "value_migration": np.full(size, math.log(1.1)),
        "first_half_arithmetic_vwap": np.full(size, 100.0),
        "second_half_arithmetic_vwap": np.full(size, 110.0),
        "arithmetic_vwap_migration": np.full(size, math.log(1.1)),
        "realized_variation": np.full(size, 0.2),
        "absolute_migration_rank": np.full(size, 0.9),
        "variation_rank": np.full(size, 0.9),
    })


def test_contract_is_frozen_and_query_is_source_only() -> None:
    assert support.PREREG_SHA256 == "14ec00d3055ce4d3ec2a78d4493b95d00d91c8d3190638542fb3a051956874c5"
    assert support.CONTROLS == (
        "no_migration_rank_gate",
        "no_volatility_gate",
        "arithmetic_vwap_migration",
        "one_boundary_stale_migration",
        "direction_flip",
    )
    normalized = " ".join(support.QUERY.split())
    assert normalized.startswith(
        "SELECT ts,open,high,low,close,volume,quote_asset_volume FROM bars_binance"
    )
    assert "symbol='BTCUSDT'" in normalized
    assert "interval='1m'" in normalized
    for forbidden in ("funding", "gross9", "return", "pnl", "bars_binance_premium"):
        assert forbidden not in support.QUERY.lower()


def test_quote_weighted_median_uses_lowest_half_weight_crossing() -> None:
    prices = np.array([30.0, 10.0, 20.0, 40.0])
    weights = np.array([1.0, 3.0, 2.0, 4.0])
    assert support.quote_weighted_median(prices, weights) == 20.0
    # Cumulative weight equals exactly half at 20, so the next price is not selected.
    assert support.quote_weighted_median(
        np.array([10.0, 20.0, 30.0]), np.array([1.0, 4.0, 5.0])
    ) == 20.0
    assert math.isnan(support.quote_weighted_median(prices, np.array([1.0, 0.0, 2.0, 5.0])))


def test_strict_prior_midrank_excludes_current_uses_ties_and_caps_lookback() -> None:
    ranked = support.strict_prior_midrank(
        pd.Series([1.0, 1.0, 2.0, 3.0]), lookback=2, minimum=2
    )
    assert np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 1.0
    # Only prior [1, 2] are retained; current 3 is excluded.
    assert ranked.iloc[3] == 1.0
    tied = support.strict_prior_midrank(pd.Series([2.0, 2.0, 2.0]), lookback=3, minimum=2)
    assert tied.iloc[2] == 0.5


def test_build_states_uses_exact_fixed_halves_and_exact_24h_variation() -> None:
    decision = pd.Timestamp("2024-02-01T00:00:00Z")
    bars = bars_for_window("2024-01-31T00:00:00Z", 1440)
    in_block = bars["ts"].ge(decision - pd.Timedelta(hours=8))
    second_half = bars["ts"].ge(decision - pd.Timedelta(hours=4))
    bars.loc[in_block & ~second_half, "quote_asset_volume"] = 100.0
    bars.loc[in_block & second_half, "quote_asset_volume"] = 120.0

    states = support.build_states(bars, start=decision, end=decision + pd.Timedelta(minutes=1))
    row = states.iloc[0]
    assert bool(row.source_valid)
    assert row.first_half_weighted_median == 100.0
    assert row.second_half_weighted_median == 120.0
    assert row.value_migration == pytest.approx(math.log(1.2))
    expected_returns = np.log(bars["close"].to_numpy() / bars["open"].to_numpy())
    assert row.realized_variation == pytest.approx(np.sqrt(np.square(expected_returns).sum()))


@pytest.mark.parametrize(
    ("mutation", "column"),
    [
        (lambda frame: frame.drop(frame.index[-1]), "source_valid"),
        (
            lambda frame: frame.assign(
                volume=lambda value: value.volume.mask(value.index == value.index[-1], 0.0)
            ),
            "block_valid",
        ),
        (
            lambda frame: frame.assign(
                quote_asset_volume=lambda value: value.quote_asset_volume.mask(
                    value.index == value.index[0], np.inf
                )
            ),
            "variation_valid",
        ),
        (
            lambda frame: frame.assign(
                high=lambda value: value.high.mask(
                    value.index == value.index[-1], value.close.iloc[-1] - 1.0
                )
            ),
            "block_valid",
        ),
    ],
)
def test_build_states_rejects_missing_nonpositive_nonfinite_or_incoherent_rows(mutation, column) -> None:
    decision = pd.Timestamp("2024-02-01T00:00:00Z")
    bars = mutation(bars_for_window("2024-01-31T00:00:00Z", 1440))
    states = support.build_states(bars, start=decision, end=decision + pd.Timedelta(minutes=1))
    assert not bool(states.iloc[0][column])
    assert not bool(states.iloc[0].source_valid)


def test_duplicate_timestamp_invalidates_every_window_containing_that_minute() -> None:
    decision = pd.Timestamp("2024-02-01T00:00:00Z")
    bars = bars_for_window("2024-01-31T00:00:00Z", 1440)
    bars = pd.concat([bars, bars.iloc[[-1]]], ignore_index=True)
    states = support.build_states(bars, start=decision, end=decision + pd.Timedelta(minutes=1))
    assert not bool(states.iloc[0].block_valid)


def test_arithmetic_vwap_control_uses_its_geometry_and_sign_but_primary_rank() -> None:
    states = minimal_states(pd.DatetimeIndex([pd.Timestamp("2024-03-01T00:00:00Z")]))
    states.loc[0, "arithmetic_vwap_migration"] = -0.05
    states.loc[0, "first_half_arithmetic_vwap"] = 105.0
    states.loc[0, "second_half_arithmetic_vwap"] = 100.0
    states.loc[0, "absolute_migration_rank"] = 0.8
    eligible, side = support.active(states, "arithmetic_vwap_migration")
    assert bool(eligible.iloc[0])
    assert side.iloc[0] == -1
    states.loc[0, "absolute_migration_rank"] = 0.74
    assert not bool(support.active(states, "arithmetic_vwap_migration")[0].iloc[0])


def test_one_boundary_stale_shifts_complete_migration_geometry_and_rank_only() -> None:
    decisions = pd.date_range("2024-03-01T00:00:00Z", periods=2, freq="8h")
    states = minimal_states(decisions)
    states.loc[0, [
        "first_half_weighted_median", "second_half_weighted_median", "value_migration",
        "absolute_migration_rank", "first_half_arithmetic_vwap",
        "second_half_arithmetic_vwap", "arithmetic_vwap_migration",
    ]] = [90.0, 81.0, math.log(0.9), 0.95, 92.0, 82.0, math.log(82.0 / 92.0)]
    states.loc[1, "absolute_migration_rank"] = 0.1
    states.loc[1, "variation_rank"] = 0.99
    states.loc[1, "realized_variation"] = 7.0

    frame = support.control_features(states, "one_boundary_stale_migration")
    assert frame.loc[1, "first_half_weighted_median"] == 90.0
    assert frame.loc[1, "second_half_weighted_median"] == 81.0
    assert frame.loc[1, "absolute_migration_rank"] == 0.95
    assert frame.loc[1, "variation_rank"] == 0.99
    assert frame.loc[1, "realized_variation"] == 7.0
    eligible, side = support.active(states, "one_boundary_stale_migration")
    assert not bool(eligible.iloc[0])
    assert bool(eligible.iloc[1])
    assert side.iloc[1] == -1


def test_no_gate_controls_and_direction_flip_are_exact() -> None:
    states = minimal_states(pd.DatetimeIndex([pd.Timestamp("2024-03-01T00:00:00Z")]))
    states.loc[0, ["absolute_migration_rank", "variation_rank"]] = [0.1, 0.1]
    assert not bool(support.active(states)[0].iloc[0])
    states.loc[0, "variation_rank"] = 0.9
    assert bool(support.active(states, "no_migration_rank_gate")[0].iloc[0])
    states.loc[0, ["absolute_migration_rank", "variation_rank"]] = [0.9, 0.1]
    assert bool(support.active(states, "no_volatility_gate")[0].iloc[0])
    assert support.active(states, "direction_flip")[1].iloc[0] == -1


def test_clock_allows_equal_exit_entry_and_enforces_split_half_open_gate() -> None:
    decisions = pd.DatetimeIndex([
        pd.Timestamp("2023-12-31T16:00:00Z"),  # exit crosses the train/test boundary
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T08:00:00Z"),  # entry equals the prior exit
    ])
    states = minimal_states(decisions)
    candidate = support.clock(states)
    assert list(candidate.decision_time) == list(decisions[1:])
    assert candidate.iloc[0].exit_time == candidate.iloc[1].entry_time
    assert list(candidate.split) == ["test", "test"]
    assert list(candidate.side) == [1, 1]
    assert (candidate.feature_available_time == candidate.decision_time).all()


def test_schema_must_match_frozen_query_columns_exactly() -> None:
    bars = bars_for_window("2024-01-31T00:00:00Z", 1440).assign(extra=1)
    with pytest.raises(ValueError, match="source schema"):
        support.build_states(
            bars,
            start=pd.Timestamp("2024-02-01T00:00:00Z"),
            end=pd.Timestamp("2024-02-01T00:01:00Z"),
        )
