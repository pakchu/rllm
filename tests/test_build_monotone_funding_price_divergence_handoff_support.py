from pathlib import Path

import numpy as np
import pandas as pd

from training import build_monotone_funding_price_divergence_handoff_support as support


def minute_bars(start: str, periods: int, start_price: float = 100.0, end_price: float = 90.0):
    timestamps = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    opens = np.linspace(start_price, end_price, periods)
    closes = opens.copy()
    return pd.DataFrame(
        {
            "ts": timestamps,
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": closes,
        }
    )


def funding_rows(times, rates):
    return pd.DataFrame(
        {"funding_time": pd.to_datetime(times, utc=True), "funding_rate": rates, "mark_price": 100.0}
    )


def eligible_feature_rows(times):
    count = len(times)
    return pd.DataFrame(
        {
            "settlement_time": pd.to_datetime(times, utc=True),
            "decision_time": pd.to_datetime(times, utc=True),
            "feature_available_time": pd.to_datetime(times, utc=True),
            "funding_rate_f2": [0.0001] * count,
            "funding_rate_f1": [0.0002] * count,
            "funding_rate_f0": [0.0003] * count,
            "funding_event_valid": [True] * count,
            "three_settlement_path": [True] * count,
            "two_settlement_acceleration": [True] * count,
            "return_window_valid": [True] * count,
            "return_16h": [-0.02] * count,
            "absolute_return_prior_midrank": [0.8] * count,
            "rv20": [0.7] * count,
            "rv20_prior_q90": [0.6] * count,
            "rv20_q90_active": [True] * count,
        }
    )


def test_mfdh_builder_is_hash_bound_and_source_only():
    source = Path(support.__file__).read_text()
    assert support.PREREG_SHA == "5e5b1a98c63c8ea8680f61379d5943dace6ed4b5ec524f1e3471d6cf7447eab1"
    assert support.FUNDING_QUERY.startswith("SELECT funding_time,funding_rate,mark_price")
    assert support.BAR_QUERY.startswith("SELECT ts,open,high,low,close")
    assert "execution_prices_opened\": False" in source
    assert "postentry_return_or_pnl_opened\": False" in source
    assert "gross9_rows_opened\": False" in source


def test_strict_prior_midrank_excludes_current_and_caps_history():
    values = pd.Series([0.0] * 180 + [0.0, 1.0] + list(range(2, 272)), dtype=float)
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:180].isna().all()
    assert ranks.iloc[180] == 0.5
    assert ranks.iloc[181] == 1.0
    assert ranks.iloc[-1] == 1.0


def test_exact_960_minute_return_and_monotone_divergence_features():
    settlement = pd.Timestamp("2024-07-01T16:00:00Z")
    bars = minute_bars("2024-07-01T00:00:00Z", 960)
    funding = funding_rows(
        [settlement - pd.Timedelta(hours=16), settlement - pd.Timedelta(hours=8), settlement],
        [0.0001, 0.0002, 0.0003],
    )
    features = support.build_features(bars, funding)
    current = features.iloc[-1]
    assert current["return_window_valid"]
    assert current["three_settlement_path"]
    assert current["return_16h"] < 0

    missing = bars.drop(index=500)
    invalid = support.build_features(missing, funding).iloc[-1]
    assert not invalid["return_window_valid"]
    assert np.isnan(invalid["return_16h"])


def test_missing_intermediate_settlement_breaks_monotone_path():
    settlement = pd.Timestamp("2024-07-01T16:00:00Z")
    bars = minute_bars("2024-07-01T00:00:00Z", 960)
    funding = funding_rows(
        [settlement - pd.Timedelta(hours=24), settlement - pd.Timedelta(hours=8), settlement],
        [0.0001, 0.0002, 0.0003],
    )
    assert not support.build_features(bars, funding).iloc[-1]["three_settlement_path"]


def test_duplicate_or_incoherent_source_rows_make_settlement_ineligible():
    settlement = pd.Timestamp("2024-07-01T16:00:00Z")
    bars = minute_bars("2024-07-01T00:00:00Z", 960)
    times = [settlement - pd.Timedelta(hours=16), settlement - pd.Timedelta(hours=8), settlement]
    duplicated = pd.concat(
        [funding_rows(times, [0.0001, 0.0002, 0.0003]), funding_rows([settlement], [0.0003])],
        ignore_index=True,
    )
    features = support.build_features(bars, duplicated)
    assert not features.iloc[-1]["funding_event_valid"]
    assert not features.iloc[-1]["three_settlement_path"]

    incoherent = bars.copy()
    incoherent.loc[100, "high"] = incoherent.loc[100, "low"] - 1
    valid_funding = funding_rows(times, [0.0001, 0.0002, 0.0003])
    assert not support.build_features(incoherent, valid_funding).iloc[-1]["return_window_valid"]


def test_controls_have_frozen_eligibility_and_side_semantics():
    row = eligible_feature_rows(["2024-07-01T16:00:00Z"])
    assert support.signal(row, "primary").iloc[0] == -1
    assert support.signal(row, "funding_side_instead_of_price_side").iloc[0] == 1
    assert support.signal(row, "direction_flip").iloc[0] == 1

    below_rank = row.copy()
    below_rank["absolute_return_prior_midrank"] = 0.59
    assert support.signal(below_rank, "primary").iloc[0] == 0
    assert support.signal(below_rank, "no_return_rank").iloc[0] == -1

    only_two = row.copy()
    only_two["three_settlement_path"] = False
    assert support.signal(only_two, "primary").iloc[0] == 0
    assert support.signal(only_two, "two_settlement_acceleration").iloc[0] == -1
    assert support.CONTROLS == (
        "no_return_rank",
        "two_settlement_acceleration",
        "funding_side_instead_of_price_side",
        "direction_flip",
    )


def test_clock_is_s_plus_five_eight_hours_half_open_and_split_crossing_skips():
    features = eligible_feature_rows(
        [
            "2024-07-01T00:00:00Z",
            "2024-07-01T08:00:00Z",
            "2024-12-31T16:00:00Z",
            "2025-01-01T00:00:00Z",
        ]
    )
    clock = support.build_clock(features)
    assert len(clock) == 3
    assert clock.iloc[0]["entry_time"] == pd.Timestamp("2024-07-01T00:05:00Z")
    assert clock.iloc[0]["exit_time"] == clock.iloc[1]["entry_time"]
    assert (clock["exit_time"] - clock["entry_time"]).eq(pd.Timedelta(hours=8)).all()
    assert pd.Timestamp("2024-12-31T16:00:00Z") not in set(clock["decision_time"])
    assert pd.Timestamp("2025-01-01T00:00:00Z") in set(clock["decision_time"])


def test_rv20_threshold_is_causal_report_only():
    values = pd.Series(np.arange(757, dtype=float))
    threshold = support.strict_prior_quantile(values)
    assert threshold.iloc[:756].isna().all()
    assert threshold.iloc[756] == np.quantile(np.arange(756, dtype=float), 0.90, method="linear")

    feature = eligible_feature_rows(["2024-07-01T16:00:00Z"])
    feature["rv20"] = np.nan
    feature["rv20_prior_q90"] = np.nan
    feature["rv20_q90_active"] = False
    assert len(support.build_clock(feature)) == 1


def test_support_gates_are_frozen():
    assert support.MINIMUM_EVENTS == {"train": 8, "test": 12, "eval": 12, "final": 8}
    rows = eligible_feature_rows(pd.date_range("2024-01-01", periods=5, freq="8h", tz="UTC"))
    rows.loc[:3, "return_16h"] = -0.02
    rows.loc[4, "return_16h"] = 0.02
    rows.loc[4, ["funding_rate_f2", "funding_rate_f1", "funding_rate_f0"]] = [-0.0001, -0.0002, -0.0003]
    clock = support.build_clock(rows)
    stats = support.split_stats(clock, "test")
    assert stats["events"] == 5
    assert stats["minority_side_share"] == 0.2
