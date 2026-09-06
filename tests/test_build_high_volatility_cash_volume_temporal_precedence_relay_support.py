import gzip
import json
import math

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_cash_volume_temporal_precedence_relay_support as support
from training import preregister_high_volatility_cash_volume_temporal_precedence_relay as prereg


def bars(start: str, periods: int = 480, *, weight_minute: int = 100, up: bool = True) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    minute = np.arange(periods, dtype=float)
    open_ = 100.0 + minute * 0.001
    close = open_ * (1.0001 if up else 0.9999)
    weights = np.zeros(periods)
    if 0 <= weight_minute < periods:
        weights[weight_minute] = 10.0
    return pd.DataFrame(
        {
            "ts": timestamps,
            "open": open_,
            "high": np.maximum(open_, close) + 0.01,
            "low": np.minimum(open_, close) - 0.01,
            "close": close,
            "quote_asset_volume": weights,
        }
    )


def panel(decisions: pd.DatetimeIndex) -> pd.DataFrame:
    size = len(decisions)
    frame = pd.DataFrame(
        {
            "decision_time": decisions,
            "feature_available_time": decisions,
            "source_valid": True,
            "spot_minute_count": 480,
            "perpetual_minute_count": 480,
            "spot_weighted_median_minute": 100.0,
            "perpetual_weighted_median_minute": 200.0,
            "cash_precedence": 100.0,
            "precedence_rank": 0.9,
            "spot_weighted_mean_minute": 100.0,
            "perpetual_weighted_mean_minute": 200.0,
            "mean_arrival_precedence": 100.0,
            "spot_return": 0.01,
            "perpetual_return": 0.01,
            "spot_final_two_hour_return": 0.005,
            "perpetual_final_two_hour_return": 0.005,
            "direction_side": 1,
            "perpetual_variation": 0.1,
            "variation_rank": 0.9,
            "eligible": True,
            "onset": False,
        }
    )
    return frame.loc[:, support.PANEL_COLUMNS]


def test_frozen_preregistration_and_source_only_queries() -> None:
    assert support.PREREG_SHA == "573eaefe87cf1c69501264243f887032281955c8e4cbe65f22f8950d196836af"
    assert support.sha256(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == (
        "no_precedence_tail", "no_variation_gate", "mean_arrival_precedence",
        "one_decision_stale_precedence", "direction_flip", "forced_long",
    )
    normalized = {name: " ".join(query.lower().split()) for name, query in support.QUERIES.items()}
    assert "from bars_binance_spot" in normalized["spot"]
    assert "from bars_binance " in normalized["perpetual"]
    assert all("quote_asset_volume" in query for query in normalized.values())
    forbidden = ("funding", "gross9", "pnl", "execution", "outcome")
    assert all(term not in query for query in normalized.values() for term in forbidden)


def test_weighted_arrival_statistics_use_zero_based_minutes_and_lower_crossing() -> None:
    weights = np.zeros(480)
    weights[[2, 7]] = [5.0, 5.0]
    assert support.weighted_median_minute(weights) == 2.0
    assert support.weighted_mean_minute(weights) == 4.5
    weights[2] = 4.0
    assert support.weighted_median_minute(weights) == 7.0
    assert math.isnan(support.weighted_median_minute(np.zeros(480)))
    assert math.isnan(support.weighted_mean_minute(np.r_[np.ones(479), -1.0]))


def test_strict_prior_midrank_excludes_current_ties_and_caps_at_270() -> None:
    ranks = support.strict_prior_midrank(pd.Series([2.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == 1.0
    tied = support.strict_prior_midrank(pd.Series([2.0, 2.0, 2.0]), lookback=2, minimum=2)
    assert tied.iloc[2] == 0.5
    capped = support.strict_prior_midrank(pd.Series(np.arange(272, dtype=float)), lookback=270, minimum=180)
    assert capped.iloc[179] != capped.iloc[179]
    assert capped.iloc[180] == 1.0
    assert capped.iloc[271] == 1.0


def test_build_panel_uses_exact_0400_grid_aligned_480_rows_and_both_return_horizons(monkeypatch) -> None:
    monkeypatch.setattr(support, "START", pd.Timestamp("2024-01-01T00:00:00Z"))
    monkeypatch.setattr(support, "END", pd.Timestamp("2024-01-01T04:01:00Z"))
    spot = bars("2023-12-31T20:00:00Z", weight_minute=50)
    perpetual = bars("2023-12-31T20:00:00Z", weight_minute=300)
    built = support.build_panel(spot, perpetual)
    row = built.iloc[0]
    assert row.decision_time == pd.Timestamp("2024-01-01T04:00:00Z")
    assert bool(row.source_valid)
    assert row.spot_minute_count == row.perpetual_minute_count == 480
    assert row.spot_weighted_median_minute == row.spot_weighted_mean_minute == 50.0
    assert row.perpetual_weighted_median_minute == row.perpetual_weighted_mean_minute == 300.0
    assert row.cash_precedence == row.mean_arrival_precedence == 250.0
    assert row.direction_side == 1
    expected = math.sqrt(480 * math.log(1.0001) ** 2)
    assert row.perpetual_variation == pytest.approx(expected)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.drop(frame.index[-1]),
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
        lambda frame: frame.assign(high=lambda value: value["low"] - 1.0),
        lambda frame: frame.assign(quote_asset_volume=0.0),
    ],
)
def test_source_contract_fails_closed(monkeypatch, mutate) -> None:
    monkeypatch.setattr(support, "START", pd.Timestamp("2024-01-01T00:00:00Z"))
    monkeypatch.setattr(support, "END", pd.Timestamp("2024-01-01T04:01:00Z"))
    good = bars("2023-12-31T20:00:00Z", weight_minute=100)
    changed = mutate(good.copy())
    if changed["ts"].duplicated().any():
        with pytest.raises(RuntimeError, match="duplicate"):
            support.build_panel(changed, good)
    else:
        assert not bool(support.build_panel(changed, good).iloc[0].source_valid)


def test_onset_requires_immediately_previous_scheduled_valid_ineligible_decision() -> None:
    eligible = pd.Series([False, True, False, True, True, False, True])
    valid = pd.Series([True, True, False, True, True, True, True])
    assert support.immediate_prior_onset(eligible, valid).tolist() == [
        False, True, False, False, False, False, True
    ]


def test_primary_and_six_controls_preserve_frozen_geometry() -> None:
    decisions = pd.date_range("2024-07-01T04:00:00Z", periods=5, freq="8h")
    frame = panel(decisions)
    frame["precedence_rank"] = [0.5, 0.9, 0.9, 0.5, 0.9]
    frame.loc[0, "cash_precedence"] = -1.0
    active, side, _ = support.active(frame)
    assert active.tolist() == [False, True, False, False, True]
    assert side.tolist() == [1] * 5
    low_precedence = frame.copy()
    low_precedence.loc[1, "precedence_rank"] = 0.5
    assert support.active(low_precedence, "no_precedence_tail")[0].iloc[1]
    low_variation = frame.copy()
    low_variation.loc[1, "variation_rank"] = 0.5
    assert support.active(low_variation, "no_variation_gate")[0].iloc[1]
    mean = frame.copy()
    mean.loc[1, "cash_precedence"] = -1.0
    assert support.active(mean, "mean_arrival_precedence")[0].iloc[1]
    assert support.active(frame, "direction_flip")[1].tolist() == [-1] * 5
    short = frame.copy()
    short["direction_side"] = -1
    assert support.active(short, "forced_long")[1].tolist() == [1] * 5
    stale_active, _, stale = support.active(frame, "one_decision_stale_precedence")
    assert stale.loc[1, "cash_precedence"] == frame.loc[0, "cash_precedence"]
    assert stale.loc[1, "feature_available_time"] == frame.loc[0, "feature_available_time"]
    assert not stale_active.iloc[0]


def test_split_crossing_is_skipped_and_clock_schema_is_stable() -> None:
    decisions = pd.DatetimeIndex(
        [pd.Timestamp("2023-12-31T20:00:00Z"), pd.Timestamp("2024-01-01T04:00:00Z")]
    )
    frame = panel(decisions)
    frame["precedence_rank"] = [0.5, 0.9]
    clock = support.build_clock(frame)
    assert tuple(clock.columns) == support.CLOCK_COLUMNS
    assert len(clock) == 1
    assert clock.iloc[0].split == "test"
    crossing = panel(pd.DatetimeIndex([pd.Timestamp("2023-12-31T12:00:00Z"), pd.Timestamp("2023-12-31T20:00:00Z")]))
    crossing["precedence_rank"] = [0.5, 0.9]
    assert support.build_clock(crossing).empty


def test_support_gates_and_deterministic_immutable_artifacts(tmp_path) -> None:
    entries = pd.date_range("2024-01-01T04:05:00Z", periods=5, freq="31D")
    clock = pd.DataFrame({"split": "test", "side": [1, 1, 1, -1, -1], "entry_time": entries})
    stats = support.support_stats(clock, "test")
    assert stats == {
        "events": 5, "longs": 3, "shorts": 2,
        "minority_side_share": 0.4, "max_month_share": 0.2,
    }
    frame = pd.DataFrame({"text": ["현금"], "value": [1.25]})
    first = support.deterministic_gzip(frame)
    assert first == support.deterministic_gzip(frame)
    assert "현금" in gzip.decompress(first).decode()
    encoded = support.canonical_json_bytes({"label": "현금"})
    assert b"\\u" not in encoded
    artifact = tmp_path / "artifact.json"
    support.immutable_write(artifact, encoded)
    support.immutable_write(artifact, encoded)
    with pytest.raises(RuntimeError, match="immutable"):
        support.immutable_write(artifact, b"different")
