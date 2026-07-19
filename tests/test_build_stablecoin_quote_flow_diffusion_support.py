from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import build_stablecoin_quote_flow_diffusion_support as support


def _source(
    imbalances: dict[str, list[float]],
    *,
    start: str = "2023-08-04T08:00:00Z",
) -> pd.DataFrame:
    hours = len(next(iter(imbalances.values())))
    dates = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    rows: list[dict[str, object]] = []
    for symbol in support.SYMBOLS:
        for date, imbalance in zip(dates, imbalances[symbol], strict=True):
            volume = 10.0
            signed = volume * imbalance
            buy = (volume + signed) / 2.0
            close = date + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open_time_us": int(date.timestamp() * 1_000_000),
                    "close_time_us": int(close.timestamp() * 1_000_000),
                    "base_volume_btc": volume,
                    "trade_count": 100,
                    "taker_buy_base_btc": buy,
                    "taker_sell_base_btc": volume - buy,
                    "signed_taker_flow_btc": signed,
                    "source_complete": True,
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def _state(onset_times: list[str]) -> pd.DataFrame:
    times = pd.to_datetime(onset_times, utc=True)
    rows = len(times)
    frame = pd.DataFrame(
        {
            "source_hour_start": times,
            "decision_time": times + pd.Timedelta(hours=1),
            "feature_available_time": times + pd.Timedelta(hours=1),
            "source_valid": True,
            "z_usdt": 0.0,
            "z_usdc": 1.0,
            "z_fdusd": 1.0,
            "alt_share": 0.5,
            "prior_alt_share_q50": 0.4,
            "min_alt_abs_z": 1.0,
            "weighted_alt_z": 1.0,
            "primary_side": np.resize(np.array([1.0, -1.0]), rows),
            "primary_onset": True,
            "no_alt_breadth_side": 1.0,
            "no_alt_breadth_onset": False,
            "no_usdt_lag_side": 1.0,
            "no_usdt_lag_onset": False,
            "no_participation_side": 1.0,
            "no_participation_onset": False,
            "usdt_only_side": 1.0,
            "usdt_only_onset": False,
        }
    )
    return frame.loc[:, list(support.STATE_COLUMNS)]


def test_strict_prior_linear_quantile_excludes_current_value() -> None:
    policy = replace(support.Policy(), prior_window_hours=4, prior_min_periods_hours=3)
    values = pd.Series([0.0, 1.0, 2.0, 1_000.0, -1_000.0])
    median = support._strict_prior_quantile(values, quantile=0.5, policy=policy)
    assert np.isnan(median.iloc[2])
    assert median.iloc[3] == 1.0
    assert median.iloc[4] == 1.5


def test_state_prefix_is_invariant_to_appended_future_source() -> None:
    policy = replace(support.Policy(), prior_window_hours=4, prior_min_periods_hours=3)
    base = {
        "BTCUSDT": [-0.3, -0.1, 0.1, 0.0, 0.2, -0.2],
        "BTCUSDC": [-0.4, -0.2, 0.0, 0.8, 0.3, -0.3],
        "BTCFDUSD": [-0.5, -0.1, 0.1, 0.9, 0.4, -0.4],
    }
    prefix = support.derive_state(_source(base), policy)
    extended = {symbol: values + [0.99, -0.99] for symbol, values in base.items()}
    full = support.derive_state(_source(extended), policy).iloc[: len(prefix)]
    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True), full.reset_index(drop=True)
    )


def test_primary_requires_both_alternative_books_and_usdt_lag() -> None:
    policy = replace(support.Policy(), prior_window_hours=4, prior_min_periods_hours=3)
    agreeing = {
        "BTCUSDT": [-0.3, 0.0, 0.3, 0.0, 0.0],
        "BTCUSDC": [-0.4, 0.0, 0.4, 0.9, 0.0],
        "BTCFDUSD": [-0.5, 0.0, 0.5, 0.9, 0.0],
    }
    state = support.derive_state(_source(agreeing), policy)
    assert bool(state.iloc[3]["primary_onset"])
    assert state.iloc[3]["primary_side"] == 1.0

    disagreeing = {key: list(value) for key, value in agreeing.items()}
    disagreeing["BTCFDUSD"][3] = -0.9
    state = support.derive_state(_source(disagreeing), policy)
    assert not bool(state.iloc[3]["primary_onset"])

    followed = {key: list(value) for key, value in agreeing.items()}
    followed["BTCUSDT"][3] = 0.9
    state = support.derive_state(_source(followed), policy)
    assert not bool(state.iloc[3]["primary_onset"])


def test_global_reservation_happens_before_split_containment() -> None:
    state = _state(
        [
            "2023-12-31T20:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-01T04:00:00Z",
        ]
    )
    clocks = support.build_clocks(state)
    primary = clocks[clocks["control"].eq("primary")]
    assert len(primary) == 1
    assert primary.iloc[0]["source_hour_start"] == pd.Timestamp("2024-01-01T04:00:00Z")
    assert primary.iloc[0]["entry_time"] == pd.Timestamp("2024-01-01T05:05:00Z")


def test_exit_exactly_at_split_end_is_contained() -> None:
    policy = replace(support.Policy(), hold_hours=6)
    state = _state(["2023-12-31T16:55:00Z"])
    state["decision_time"] = pd.Timestamp("2023-12-31T17:55:00Z")
    state["feature_available_time"] = state["decision_time"]
    clocks = support.build_clocks(state, policy)
    primary = clocks[clocks["control"].eq("primary")]
    assert len(primary) == 1
    assert primary.iloc[0]["exit_time"] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert primary.iloc[0]["split"] == "train"


def test_random_side_uses_frozen_ascii_timestamp_contract() -> None:
    decision = support._utc_timestamp("2024-01-02T03:00:00Z")
    expected = -1  # SHA256("SQFD-6|2024-01-02T03:00:00Z") starts with odd nibble 7.
    assert support._random_side(decision) == expected


def test_novelty_uses_common_coverage_and_bidirectional_containment() -> None:
    primary = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-07-01T01:05:00Z",
                "2023-07-03T01:05:00Z",
                "2025-01-01T01:05:00Z",
            ],
            utc=True,
        )
    )
    comparator = pd.DatetimeIndex(
        pd.to_datetime(["2023-07-01T01:05:00Z", "2023-07-03T05:05:00Z"], utc=True)
    )
    result = support._novelty(
        primary,
        comparator,
        start=support._utc_timestamp("2023-07-01T00:00:00Z"),
        end=support._utc_timestamp("2024-01-01T00:00:00Z"),
        near_hours=6,
    )
    assert result["primary_events"] == 2
    assert result["exact_jaccard"] == 1 / 3
    assert result["primary_near_share"] == 1.0
    assert result["comparator_near_share"] == 1.0
    assert result["max_bidirectional_near_share"] == 1.0


def test_clock_schema_cannot_retain_outcomes() -> None:
    forbidden = {
        "price",
        "open",
        "high",
        "low",
        "close",
        "return",
        "pnl",
        "funding",
        "cagr",
        "drawdown",
    }
    assert forbidden.isdisjoint(support.CLOCK_COLUMNS)


def test_real_source_load_is_outcome_blind_and_hash_bound() -> None:
    prereg = support.load_preregistration()
    source = support.load_source(prereg)
    assert len(source) == 78_088
    assert tuple(source.columns) == support.SOURCE_COLUMNS
    assert not any(
        token in column.lower()
        for column in source.columns
        for token in ("price", "return", "funding", "pnl")
    )


def test_source_grid_rejects_a_missing_whole_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        support.SPLITS,
        "final",
        (
            support._utc_timestamp("2023-08-04T08:00:00Z"),
            support._utc_timestamp("2023-08-04T11:00:00Z"),
        ),
    )
    activation = {symbol: "2023-08-04T08:00:00Z" for symbol in support.SYMBOLS}
    imbalances = {symbol: [0.0, 0.0, 0.0] for symbol in support.SYMBOLS}
    source = _source(imbalances)
    source = source.loc[~source["date"].eq(pd.Timestamp("2023-08-04T09:00:00Z"))].copy()
    with pytest.raises(ValueError, match="hourly grid"):
        support._validate_source_grid(source, activation)
