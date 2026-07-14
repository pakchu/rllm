from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import search_hidden_safe_haven_cancellation_alpha as hidden_fx


def test_safe_haven_strength_has_positive_risk_stress() -> None:
    oriented = pd.DataFrame({ticker: [0.0] for ticker in hidden_fx.TICKERS})
    oriented.loc[0, ["USDJPY", "USDCHF"]] = -2.0
    assert hidden_fx.safe_haven_risk_stress(oriented).iloc[0] == pytest.approx(2.0)


def test_continuous_return_rejects_a_gap() -> None:
    values = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    valid = pd.Series([True, True, False, True, True])
    result = hidden_fx.continuous_log_return(values, valid, horizon=2)
    assert np.isnan(result.iloc[2])
    assert np.isnan(result.iloc[3])
    assert np.isnan(result.iloc[4])


def test_prior_zscore_prefix_does_not_change_from_future_suffix() -> None:
    values = pd.Series(np.sin(np.arange(1000) / 17.0))
    first = hidden_fx.prior_zscore(values, window=120, min_observations=60)
    changed = values.copy()
    changed.iloc[700:] = 1e9
    second = hidden_fx.prior_zscore(changed, window=120, min_observations=60)
    np.testing.assert_allclose(first.iloc[:700], second.iloc[:700], equal_nan=True)


def test_prior_scaled_horizon_return_uses_only_prior_hourly_volatility() -> None:
    rng = np.random.default_rng(7)
    one_hour = rng.normal(0.0, 0.01, size=1000)
    values = pd.Series(np.exp(np.cumsum(one_hour)))
    valid = pd.Series(True, index=values.index)
    _, first = hidden_fx.prior_scaled_horizon_return(
        values, valid, horizon=6, window=120, min_observations=60
    )
    changed = values.copy()
    changed.iloc[700:] *= 1e6
    _, second = hidden_fx.prior_scaled_horizon_return(
        changed, valid, horizon=6, window=120, min_observations=60
    )
    np.testing.assert_allclose(first.iloc[:700], second.iloc[:700], equal_nan=True)


def test_policy_fades_positive_unpriced_stress() -> None:
    state = pd.DataFrame(
        {
            "eligible": [True, True, True, False],
            "cancellation_score": [2.0, 3.0, 0.5, 9.0],
            "unpriced_stress": [1.0, -1.0, 1.0, -1.0],
        }
    )
    long_active, short_active = hidden_fx.policy_masks(state, threshold=1.0)
    assert np.flatnonzero(long_active).tolist() == [1]
    assert np.flatnonzero(short_active).tolist() == [0]


def test_fit_threshold_ignores_2023_values(monkeypatch) -> None:
    monkeypatch.setitem(hidden_fx.WINDOWS, "fit", ("2021-01-01", "2021-02-01"))
    dates = pd.Series(pd.date_range("2021-01-01", periods=20_000, freq="5min"))
    state = pd.DataFrame({"score": np.linspace(0.0, 1.0, len(dates))})
    first = hidden_fx.fit_threshold(state, dates, "score")
    state.loc[dates >= pd.Timestamp("2021-02-01"), "score"] = 1e12
    second = hidden_fx.fit_threshold(state, dates, "score")
    assert first == pytest.approx(second)


def test_support_counts_are_nonoverlapping_and_split_contained(monkeypatch) -> None:
    monkeypatch.setitem(
        hidden_fx.WINDOWS, "sample", ("2023-01-01", "2023-01-01 00:45")
    )
    monkeypatch.setattr(hidden_fx, "HOLD_BARS", 2)
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    long_active = np.array(
        [True, True, False, False, False, False, True, False, False, False]
    )
    short_active = np.array(
        [False, False, False, False, True, False, False, False, False, False]
    )
    assert hidden_fx.support_counts(
        dates, long_active, short_active, window="sample"
    ) == {
        "raw": 4,
        "raw_long": 3,
        "raw_short": 1,
        "strict_executable": 2,
        "strict_executable_long": 1,
        "strict_executable_short": 1,
    }


def test_fx_loader_requires_completed_hour_and_cutoff(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for hour in pd.date_range("2020-01-01", periods=2, freq="h"):
        for minute in range(60):
            timestamp = hour + pd.Timedelta(minutes=minute)
            for index, ticker in enumerate(hidden_fx.TICKERS):
                rows.append(
                    {
                        "date": timestamp.tz_localize("UTC"),
                        "tic": ticker,
                        "interval": "1m",
                        "close": 100.0 + index + minute / 100.0,
                    }
                )
    # A post-cutoff row may be physically read in the crossing chunk but must be discarded.
    rows.append(
        {
            "date": pd.Timestamp("2020-01-01 02:00", tz="UTC"),
            "tic": "EURUSD",
            "interval": "1m",
            "close": 999.0,
        }
    )
    path = tmp_path / "fx.csv.gz"
    pd.DataFrame(rows).sort_values("date").to_csv(path, index=False, compression="gzip")
    hourly = hidden_fx.read_completed_fx_hours_before(
        path, cutoff="2020-01-01 02:00", chunksize=137
    )
    assert hourly["effective_time"].tolist() == [pd.Timestamp("2020-01-01 01:00")]
    assert hourly["valid_hour"].tolist() == [True]
    assert hourly["source_rows_min"].tolist() == [60]
    assert hourly["source_time"].iloc[0] == pd.Timestamp("2020-01-01 00:59")


def test_fx_loader_rejects_unsorted_source(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01 00:01:00+00:00", "2020-01-01 00:00:00+00:00"],
            "tic": ["EURUSD", "EURUSD"],
            "interval": ["1m", "1m"],
            "close": [1.0, 1.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sorted"):
        hidden_fx.read_completed_fx_hours_before(path, cutoff="2020-01-02", chunksize=10)


def test_build_state_waits_until_minute05_entry_boundary(monkeypatch) -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=37, freq="5min"))
    market = pd.DataFrame({"close": np.linspace(100.0, 101.0, len(dates))})
    fx_features = pd.DataFrame(
        {
            "effective_time": [
                pd.Timestamp("2023-01-01 01:00"),
                pd.Timestamp("2023-01-01 02:00"),
                pd.Timestamp("2023-01-01 03:00"),
            ],
            "fx_source_time": [
                pd.Timestamp("2023-01-01 00:59"),
                pd.Timestamp("2023-01-01 01:59"),
                pd.Timestamp("2023-01-01 02:59"),
            ],
            "fx_valid": [True, True, True],
            "risk_stress": [1.0, -1.0, 1.0],
            "raw_risk_stress": [0.01, -0.01, 0.01],
            "broad_usd_strength": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(hidden_fx, "build_fx_features", lambda _: fx_features)
    monkeypatch.setattr(
        hidden_fx,
        "prior_scaled_horizon_return",
        lambda values, valid: (
            pd.Series(np.zeros(len(values))),
            pd.Series(np.zeros(len(values))),
        ),
    )
    state = hidden_fx.build_state(market, dates, pd.DataFrame())
    first_decision = int(np.flatnonzero(state["decision"].to_numpy(bool))[0])
    assert dates.iloc[first_decision] == pd.Timestamp("2023-01-01 01:00")
    assert dates.iloc[first_decision + 1] == pd.Timestamp("2023-01-01 01:05")
    assert not state.loc[dates.eq(pd.Timestamp("2023-01-01 00:55")), "decision"].any()


def test_support_only_cannot_open_outcomes_or_write(monkeypatch, tmp_path: Path) -> None:
    market = pd.DataFrame({"low": [1.0], "high": [1.0]})
    dates = pd.Series([pd.Timestamp("2023-01-01")])
    fx = pd.DataFrame(
        {
            "valid_hour": [True],
            "source_time": [pd.Timestamp("2022-12-31 23:59")],
        }
    )
    state = pd.DataFrame(
        {
            "decision": [True],
            "eligible": [True],
            "cancellation_score": [2.0],
            "unpriced_stress": [-1.0],
        }
    )
    monkeypatch.setattr(hidden_fx, "load_market_before", lambda *args: (market, dates))
    monkeypatch.setattr(hidden_fx, "read_completed_fx_hours_before", lambda *args: fx)
    monkeypatch.setattr(hidden_fx, "build_state", lambda *args: state)
    monkeypatch.setattr(hidden_fx, "fit_threshold", lambda *args: 1.0)
    monkeypatch.setattr(
        hidden_fx,
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
    monkeypatch.setattr(hidden_fx, "RESULT_PATH", tmp_path / "forbidden.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("support-only crossed the outcome boundary")

    monkeypatch.setattr(hidden_fx, "_future_extreme", forbidden)
    monkeypatch.setattr(hidden_fx, "simulate", forbidden)
    output = hidden_fx.run(support_only=True)
    assert output["support_only"] is True
    assert output["support_passed"] is True
    assert not hidden_fx.RESULT_PATH.exists()
