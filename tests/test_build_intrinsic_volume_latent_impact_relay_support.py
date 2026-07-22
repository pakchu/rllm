from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from training import build_intrinsic_volume_latent_impact_relay_support as s
from training import preregister_intrinsic_volume_latent_impact_relay as p


def _synthetic(days: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days * 288, freq="5min", tz="UTC")
    day = np.repeat(np.arange(days), 288)
    local = np.tile(np.arange(288), days)
    side = np.where(day % 2 == 0, 1.0, -1.0)
    quote = np.full(len(dates), 1_000.0)
    # Alternate high and moderate flow days so a causal q60 has both passes/fails.
    strength = np.where(day % 4 < 2, 0.12, 0.04)
    imbalance = side * strength
    taker = quote * (1.0 + imbalance) / 2.0
    within_day_move = side * strength * np.minimum(local, 143) / 143 * 0.01
    base = 100.0 + day * 0.01
    close = base * np.exp(within_day_move)
    open_ = np.r_[close[0], close[:-1]]
    open_[local == 0] = base[local == 0]
    high = np.maximum(open_, close) * 1.0005
    low = np.minimum(open_, close) * 0.9995
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "quote_asset_volume": quote,
            "taker_buy_quote": taker,
        }
    )


def _policy() -> p.Policy:
    return replace(
        p.Policy(),
        utc_day_volume_lookback_days=7,
        utc_day_volume_min_days=5,
        event_reference_days=20,
        event_reference_min_days=10,
        rolling_extrema_bars=288,
    )


def test_anchor_is_first_intrinsic_volume_passage() -> None:
    anchors = s.build_anchor_features(_synthetic(40), _policy())
    assert len(anchors) > 20
    # Constant 1,000 quote volume and 288k prior-day total: q50 first crosses
    # on zero-based bar 143, i.e. completed 11:55 UTC.
    assert set(anchors["anchor_minute_utc"]) == {11 * 60 + 55}
    assert np.allclose(anchors["cumulative_quote_volume"], 144_000.0)


def test_reference_is_strictly_prior_and_prefix_invariant() -> None:
    policy = _policy()
    base = _synthetic(80)
    original = s.apply_causal_event_references(s.build_anchor_features(base, policy), policy)
    future = _synthetic(100)
    # Make the appended future source extreme. Earlier thresholds must not move.
    cutoff = base["date"].iloc[-1]
    future.loc[future["date"] > cutoff, "taker_buy_quote"] = 999.0
    extended = s.apply_causal_event_references(
        s.build_anchor_features(future, policy), policy
    )
    prefix = extended.loc[extended["anchor_time"] <= original["anchor_time"].max()]
    columns = ["anchor_time", "flow_threshold", "impact_threshold", "primary"]
    pd.testing.assert_frame_equal(
        original[columns].reset_index(drop=True),
        prefix[columns].reset_index(drop=True),
    )
    first_ready = original.index[original["reference_ready"]][0]
    assert original.loc[first_ready, "reference_count"] == policy.event_reference_min_days


def test_schedule_is_split_contained_and_nonoverlapping(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "source_day": pd.to_datetime(
                ["2022-12-31", "2023-01-01", "2023-01-02"], utc=True
            ),
            "anchor_time": pd.to_datetime(
                ["2022-12-31 17:55", "2023-01-01 11:55", "2023-01-02 11:55"],
                utc=True,
            ),
            "entry_time": pd.to_datetime(
                ["2022-12-31 18:00", "2023-01-01 12:00", "2023-01-02 12:00"],
                utc=True,
            ),
            "exit_time": pd.to_datetime(
                ["2023-01-01 00:00", "2023-01-01 18:00", "2023-01-02 18:00"],
                utc=True,
            ),
            "side": ["LONG", "SHORT", "LONG"],
        }
    )
    scheduled = s.schedule_across_splits(frame, "x")
    assert len(scheduled) == 3
    assert scheduled.iloc[0]["exit_time"] == s.TRAIN_END
    assert scheduled.iloc[1]["entry_time"] >= s.SELECTION_START
    assert (scheduled["entry_time"].iloc[1:].reset_index(drop=True) >= scheduled["exit_time"].iloc[:-1].reset_index(drop=True)).all()


def test_clock_schema_has_no_prices_or_outcomes(tmp_path) -> None:
    clock = pd.DataFrame(
        {
            "clock_name": ["primary"],
            "source_day": pd.to_datetime(["2023-01-01"], utc=True),
            "decision_time": pd.to_datetime(["2023-01-01 11:55"], utc=True),
            "entry_time": pd.to_datetime(["2023-01-01 12:00"], utc=True),
            "exit_time": pd.to_datetime(["2023-01-01 18:00"], utc=True),
            "side": ["LONG"],
        }
    )
    output = tmp_path / "clock.csv.gz"
    s.write_clock(output, {"primary": clock})
    stored = pd.read_csv(output)
    assert list(stored.columns) == s.CLOCK_COLUMNS
    assert not any(
        token in column.lower()
        for column in stored.columns
        for token in ("open", "high", "low", "close", "return", "pnl")
    )
