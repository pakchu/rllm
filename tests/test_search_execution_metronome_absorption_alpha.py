from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import search_execution_metronome_absorption_alpha as metronome


def _valid_hour() -> tuple[np.ndarray, ...]:
    index = np.arange(metronome.HOUR_BARS, dtype=float)
    open_price = 100.0 + index * 0.1
    close = open_price * np.exp(0.0002 * np.sin(index))
    quote = 1_000_000.0 * (1.0 + 0.05 * np.sin(2.0 * np.pi * index / 3.0))
    trades = np.full(metronome.HOUR_BARS, 1_000.0)
    taker_buy = quote * 0.65
    return open_price, close, quote, trades, taker_buy


def test_spectral_regularity_detects_single_frequency() -> None:
    index = np.arange(metronome.HOUR_BARS, dtype=float)
    periodic = np.sin(2.0 * np.pi * index / 3.0)
    rng = np.random.default_rng(11)
    noise = rng.normal(size=metronome.HOUR_BARS)
    assert metronome.spectral_regularity(periodic) > metronome.spectral_regularity(noise)


def test_constant_or_linear_ticket_path_is_not_maximally_regular() -> None:
    constant = np.ones(metronome.HOUR_BARS)
    linear = np.arange(metronome.HOUR_BARS, dtype=float)
    assert np.isnan(metronome.spectral_regularity(constant))
    assert np.isnan(metronome.spectral_regularity(linear))


def test_hour_metrics_have_coherent_fade_inputs() -> None:
    metrics = metronome._hour_metrics(*_valid_hour())
    assert metrics["flow_direction"] == 1.0
    assert 0.0 < metrics["flow_coherence"] <= 1.0
    assert 0.0 <= metrics["price_nonacceptance"] <= 1.0
    assert metrics["hour_quote"] > 0.0
    assert metrics["hour_trades"] > 0.0


def test_hour_metrics_reject_degenerate_denominators() -> None:
    open_price, close, quote, trades, taker_buy = _valid_hour()
    bad_trades = trades.copy()
    bad_trades[3] = 0.0
    assert not metronome._hour_metrics(open_price, close, quote, bad_trades, taker_buy)
    flat = np.full(metronome.HOUR_BARS, 100.0)
    assert not metronome._hour_metrics(flat, flat, quote, trades, taker_buy)


def test_prior_zscore_prefix_is_unchanged_by_future_suffix() -> None:
    values = np.sin(np.arange(1000) / 19.0)
    first = metronome.prior_zscore(values, window=120, min_periods=60)
    changed = values.copy()
    changed[700:] = 1e9
    second = metronome.prior_zscore(changed, window=120, min_periods=60)
    np.testing.assert_allclose(first[:700], second[:700], equal_nan=True)


def test_policy_fades_flow_direction() -> None:
    state = pd.DataFrame(
        {
            "eligible": [True, True, True, False],
            "score": [2.0, 3.0, 0.5, 9.0],
            "flow_direction": [1.0, -1.0, 1.0, -1.0],
        }
    )
    long_active, short_active = metronome.policy_masks(
        state, "score", threshold=1.0
    )
    assert np.flatnonzero(long_active).tolist() == [1]
    assert np.flatnonzero(short_active).tolist() == [0]


def test_fit_threshold_ignores_selection_values(monkeypatch) -> None:
    monkeypatch.setitem(metronome.WINDOWS, "fit", ("2021-01-01", "2021-02-01"))
    dates = pd.Series(pd.date_range("2021-01-01", periods=20_000, freq="5min"))
    state = pd.DataFrame({"score": np.linspace(0.0, 1.0, len(dates))})
    first = metronome.fit_threshold(state, dates, "score")
    state.loc[dates >= pd.Timestamp("2021-02-01"), "score"] = 1e12
    second = metronome.fit_threshold(state, dates, "score")
    assert first == pytest.approx(second)


def test_support_counts_are_nonoverlapping_and_split_contained(monkeypatch) -> None:
    monkeypatch.setitem(
        metronome.WINDOWS, "sample", ("2023-01-01", "2023-01-01 00:45")
    )
    monkeypatch.setattr(metronome, "HOLD_BARS", 2)
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    long_active = np.array(
        [True, True, False, False, False, False, True, False, False, False]
    )
    short_active = np.array(
        [False, False, False, False, True, False, False, False, False, False]
    )
    assert metronome.support_counts(
        dates, long_active, short_active, window="sample"
    ) == {
        "raw": 4,
        "raw_long": 3,
        "raw_short": 1,
        "strict_executable": 2,
        "strict_executable_long": 1,
        "strict_executable_short": 1,
    }


def test_build_state_maps_completed_hour_to_minute05_entry(monkeypatch) -> None:
    hours = 1_200
    dates = pd.Series(pd.date_range("2021-01-01", periods=hours * 12 + 1, freq="5min"))
    index = np.arange(len(dates), dtype=float)
    quote = 1_000_000.0 * (1.0 + 0.05 * np.sin(2.0 * np.pi * index / 3.0))
    market = pd.DataFrame(
        {
            "open": 100.0 * np.exp(index * 1e-7),
            "close": 100.0 * np.exp(index * 1e-7 + 1e-4 * np.sin(index)),
            "quote_asset_volume": quote,
            "number_of_trades": 900.0 + 20.0 * np.cos(index / 5.0),
            "taker_buy_quote": quote * (0.60 + 0.02 * np.sin(index / 7.0)),
        }
    )
    state = metronome.build_state(market, dates)
    decision_positions = np.flatnonzero(state["decision"].to_numpy(bool))
    assert dates.iloc[decision_positions[0]].minute == 0
    assert pd.Timestamp(state.loc[decision_positions[0], "source_time"]) == dates.iloc[decision_positions[0] - 1]
    assert dates.iloc[decision_positions[0] + 1].minute == 5


def test_support_only_cannot_simulate_or_write(monkeypatch, tmp_path: Path) -> None:
    market = pd.DataFrame({"low": [1.0], "high": [1.0]})
    dates = pd.Series([pd.Timestamp("2023-01-01")])
    state = pd.DataFrame(
        {
            "decision": [True],
            "eligible": [True],
            "flow_direction": [1.0],
            **{column: [2.0] for column in metronome.SCORE_COLUMNS},
        }
    )
    state.attrs["denominator_floors"] = {
        "hour_quote": 1.0,
        "hour_trades": 1.0,
        "absolute_flow_fraction": 0.1,
        "price_path_length": 0.1,
    }
    monkeypatch.setattr(metronome, "load_pre2024", lambda *args: (market, dates))
    monkeypatch.setattr(metronome, "build_state", lambda *args: state)
    monkeypatch.setattr(metronome, "fit_threshold", lambda *args: 1.0)
    monkeypatch.setattr(
        metronome,
        "support_counts",
        lambda *args, window, **kwargs: {
            "raw": 100,
            "raw_long": 50,
            "raw_short": 50,
            "strict_executable": 100 if window == "fit" else 30,
            "strict_executable_long": 50 if window == "fit" else 15,
            "strict_executable_short": 50 if window == "fit" else 15,
        },
    )
    monkeypatch.setattr(metronome, "RESULT_PATH", tmp_path / "forbidden.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("support-only crossed the outcome boundary")

    monkeypatch.setattr(metronome, "_future_extreme", forbidden)
    monkeypatch.setattr(metronome, "simulate", forbidden)
    monkeypatch.setattr(metronome, "legacy_orderflow_event_masks", forbidden)
    output = metronome.run(support_only=True)
    assert output["support_only"] is True
    assert output["support_passed"] is True
    assert not metronome.RESULT_PATH.exists()
