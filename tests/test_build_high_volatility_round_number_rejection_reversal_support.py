import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_round_number_rejection_reversal_support as support
from training import preregister_high_volatility_round_number_rejection_reversal as prereg


def _bars(start: pd.Timestamp, periods: int = 1440, price: float = 9500.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "duplicate_count": 1,
        }
    )


def _feature_frame(decisions: list[str] | None = None) -> pd.DataFrame:
    times = pd.to_datetime(
        decisions
        or [
            "2024-07-01T00:00:00Z",
            "2024-07-01T01:00:00Z",
            "2024-07-01T06:00:00Z",
            "2024-07-01T12:00:00Z",
        ]
    )
    count = len(times)
    return pd.DataFrame(
        {
            "decision_time": times,
            "feature_available_time": times,
            "source_valid": [True] * count,
            "hour_valid": [True] * count,
            "variation_valid": [True] * count,
            "hour_minute_count": [60] * count,
            "variation_minute_count": [1440] * count,
            "geometry_decision_time": times,
            "hour_first_open": [9600.0, 10400.0, 9600.0, 10400.0][:count],
            "nearest_level": [10000.0] * count,
            "hour_high": [10020.0, 10510.0, 10030.0, 10520.0][:count],
            "hour_low": [9590.0, 9980.0, 9580.0, 9970.0][:count],
            "hour_last_close": [9900.0, 10100.0, 9900.0, 10100.0][:count],
            "opening_side": [-1, 1, -1, 1][:count],
            "penetration": [0.002, 0.002, 0.003, 0.003][:count],
            "rejection": [True] * count,
            "rejection_side": [-1, 1, -1, 1][:count],
            "penetration_rank": [0.75] * count,
            "btc_realized_variation": [0.1] * count,
            "variation_rank": [0.65] * count,
        }
    ).loc[:, support.FEATURE_COLUMNS]


def test_preregistration_is_bound_to_exact_frozen_artifact() -> None:
    assert support.PREREG_SHA == (
        "ccdde7ff713e7a41932339970c1f94422eb99c904e175b7519d9bd5a544919d2"
    )
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])


def test_nearest_level_uses_lower_level_at_exact_half_ties() -> None:
    assert support.nearest_round_level(9499.999) == 9000.0
    assert support.nearest_round_level(9500.0) == 9000.0
    assert support.nearest_round_level(9500.001) == 10000.0
    assert support.nearest_round_level(10500.0) == 10000.0
    assert support.nearest_round_level(11500.0) == 11000.0
    assert math.isnan(support.nearest_round_level(0.0))


def test_exact_hour_rejection_penetration_close_back_side_and_exact_level_invalid() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    raw = _bars(start)
    # The completed hour opens below 10k, strictly penetrates, then closes below.
    raw.loc[1380:, ["open", "high", "low", "close"]] = [9600.0, 9900.0, 9590.0, 9700.0]
    raw.loc[1400, "high"] = 10020.0
    feature = support.boundary_features(support.prepare_source(raw), decision)
    assert feature["source_valid"] is True
    assert feature["hour_minute_count"] == 60
    assert feature["variation_minute_count"] == 1440
    assert feature["nearest_level"] == 10000.0
    assert feature["penetration"] == pytest.approx(0.002)
    assert feature["rejection"] is True
    assert feature["rejection_side"] == -1

    # Touching without strict penetration and closing exactly at the level both fail.
    touched = raw.copy()
    touched.loc[1400, "high"] = 10000.0
    assert support.boundary_features(support.prepare_source(touched), decision)["rejection"] is False
    exact_close = raw.copy()
    exact_close.loc[1439, ["open", "high", "low", "close"]] = [9700.0, 10000.0, 9600.0, 10000.0]
    assert support.boundary_features(support.prepare_source(exact_close), decision)["rejection"] is False

    exact_open = raw.copy()
    exact_open.loc[1380, ["open", "high", "low", "close"]] = [10000.0, 10020.0, 9900.0, 9900.0]
    feature = support.boundary_features(support.prepare_source(exact_open), decision)
    assert feature["opening_side"] == 0
    assert feature["rejection"] is False and feature["rejection_side"] == 0


def test_rejection_from_above_trades_long_away_from_level() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    raw = _bars(start, price=10500.0)
    raw.loc[1380:, ["open", "high", "low", "close"]] = [10500.0, 10510.0, 10100.0, 10200.0]
    raw.loc[1400, "low"] = 9980.0
    feature = support.boundary_features(support.prepare_source(raw), decision)
    assert feature["nearest_level"] == 10000.0
    assert feature["penetration"] == pytest.approx(0.002)
    assert feature["rejection"] is True
    assert feature["rejection_side"] == 1


def test_variation_is_exact_24h_minute_close_open_and_missing_rows_invalidate() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    returns = np.linspace(-0.001, 0.001, 1440)
    raw = _bars(start)
    raw["close"] = raw.open * np.exp(returns)
    raw["high"] = raw[["open", "close"]].max(axis=1) + 1
    raw["low"] = raw[["open", "close"]].min(axis=1) - 1
    feature = support.boundary_features(support.prepare_source(raw), decision)
    assert feature["btc_realized_variation"] == pytest.approx(
        math.sqrt(float(np.square(returns).sum()))
    )

    missing = raw.drop(index=10).reset_index(drop=True)
    invalid = support.boundary_features(support.prepare_source(missing), decision)
    assert invalid["source_valid"] is False
    assert invalid["variation_minute_count"] == 1439


def test_strict_prior_hourly_midranks_exclude_current_skip_invalid_and_cap_history() -> None:
    values = pd.Series([*map(float, range(2161)), np.nan, 2160.0])
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[:1440]).all()
    assert ranks.iloc[1440] == 1.0
    assert ranks.iloc[2160] == 1.0
    assert np.isnan(ranks.iloc[2161])
    assert ranks.iloc[2162] == pytest.approx((2159 + 0.5) / 2160)


def test_controls_are_isolated_and_stale_control_shifts_complete_adjacent_geometry() -> None:
    features = _feature_frame()
    features.loc[1, "penetration_rank"] = 0.749
    features.loc[2, "variation_rank"] = 0.649
    primary, side, _ = support.active_and_side(features)
    assert primary.tolist() == [True, False, False, True]
    assert side.tolist() == [-1, 1, -1, 1]
    assert support.active_and_side(features, "no_penetration_rank_gate")[0].tolist() == [True, True, False, True]
    assert support.active_and_side(features, "no_volatility_gate")[0].tolist() == [True, False, True, True]
    assert support.active_and_side(features, "direction_flip")[1].tolist() == [1, -1, 1, -1]

    stale_active, stale_side, stale = support.active_and_side(_feature_frame(), "one_hour_stale_rejection")
    assert stale_active.tolist() == [False, True, False, False]
    assert stale_side.tolist() == [0, -1, 1, -1]
    for column in support.GEOMETRY_COLUMNS:
        assert stale.loc[1, column] == _feature_frame().loc[0, column]


def test_clock_uses_five_minute_entry_six_hour_global_half_open_reservation_and_split_skip() -> None:
    clock = support.build_clock(_feature_frame())
    assert clock.decision_time.tolist() == pd.to_datetime(
        ["2024-07-01T00:00:00Z", "2024-07-01T06:00:00Z", "2024-07-01T12:00:00Z"]
    ).tolist()
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=6)).all()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert set(clock.split) == {"test"}

    crossing = _feature_frame(["2023-12-31T23:00:00Z", "2024-01-01T06:00:00Z"])
    assert support.build_clock(crossing).decision_time.tolist() == [
        pd.Timestamp("2024-01-01T06:00:00Z")
    ]


def test_source_query_and_report_boundary_are_source_only() -> None:
    assert "FROM bars_binance\n" in support.QUERY
    assert "symbol='BTCUSDT'" in support.QUERY and "interval='1m'" in support.QUERY
    assert "SELECT ts,open,high,low,close," in support.QUERY
    assert support.QUERY_START == pd.Timestamp("2023-04-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    lowered = support.QUERY.lower()
    for forbidden in ("funding", "execution", "gross9", "outcome", "pnl"):
        assert forbidden not in lowered


def test_run_writes_split_artifacts_controls_gates_and_sealed_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "sources"
    split_dir = tmp_path / "splits"
    control_dir = tmp_path / "controls"
    monkeypatch.setattr(support, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(support, "FEATURES", source_dir / "features.csv.gz")
    monkeypatch.setattr(support, "SOURCE_MANIFEST", source_dir / "manifest.json")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "clock.csv.gz")
    monkeypatch.setattr(support, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(support, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(support, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(support, "load_source", lambda: pd.DataFrame())
    monkeypatch.setattr(support, "build_features", lambda _bars: _feature_frame())

    result = support.run()
    manifest = json.loads((source_dir / "manifest.json").read_text())
    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert result["policy_id"] == "HVRNRR-6"
    assert result["ranking"] == {
        "lookback_hours": 2160, "minimum_prior_hours": 1440, "current_excluded": True,
    }
    assert result["reservation"] == {
        "scope": "global", "interval": "half_open",
        "equal_open_after_exit_allowed": True, "split_crossing_action": "skip",
    }
    assert set(result["split_artifacts"]) == set(support.SPLITS)
    assert set(result["support"]) == set(support.SPLITS)
    assert len(result["support_checks"]) == 3 * len(support.SPLITS)
    assert all((split_dir / f"{name}.csv.gz").is_file() for name in support.SPLITS)
    assert set(result["controls"]) == set(support.CONTROLS)
    assert all((control_dir / f"{name}.csv.gz").is_file() for name in support.CONTROLS)
    assert all(not item["promotion_authorized"] for item in result["controls"].values())
    assert manifest["table"] == "bars_binance"
    assert manifest["symbol"] == "BTCUSDT" and manifest["interval"] == "1m"
    assert manifest["columns"] == ["ts", "open", "high", "low", "close"]
    assert manifest["funding_values_opened"] is False
    assert manifest["gross9_rows_opened"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
