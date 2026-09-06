import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_interbar_gap_dominance_relay_support as support
from training import preregister_high_volatility_interbar_gap_dominance_relay as prereg


def _bars(
    start: pd.Timestamp,
    periods: int = 1440,
    *,
    gaps: dict[int, float] | None = None,
    candle_returns: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build coherent chained OHLC; gap keys are row positions after the first."""
    gaps = gaps or {}
    returns = np.zeros(periods) if candle_returns is None else np.asarray(candle_returns, float)
    assert len(returns) == periods
    opens = np.empty(periods)
    closes = np.empty(periods)
    opens[0] = 100.0
    closes[0] = opens[0] * math.exp(returns[0])
    for index in range(1, periods):
        opens[index] = closes[index - 1] * math.exp(gaps.get(index, 0.0))
        closes[index] = opens[index] * math.exp(returns[index])
    return pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999,
            "close": closes,
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
            "valid_minute_count": [1440] * count,
            "gap_count": [359] * count,
            "gap_energy": [0.0008] * count,
            "dominant_gap_index": [11] * count,
            "dominant_gap_time": times - pd.Timedelta(hours=6) + pd.Timedelta(minutes=11),
            "dominant_gap": [0.02] * count,
            "latest_dominant_gap_index": [21] * count,
            "latest_dominant_gap_time": times - pd.Timedelta(hours=6) + pd.Timedelta(minutes=21),
            "latest_dominant_gap": [-0.02] * count,
            "gap_dominance": [0.5] * count,
            "block_return": [0.03] * count,
            "direction_alignment": [True] * count,
            "latest_direction_alignment": [False] * count,
            "btc_realized_variation": [0.1] * count,
            "gap_dominance_rank": [0.80] * count,
            "block_return_rank": [0.60] * count,
            "variation_rank": [0.65] * count,
        }
    )


def test_preregistration_is_bound_to_exact_committed_artifact() -> None:
    assert support.PREREG_SHA == (
        "aa850bcee14c51a9defecb4e1f828660b226a01e5c84284914085c68e95183d0"
    )
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])


def test_source_query_is_exact_btcusdt_one_minute_ohlc_only() -> None:
    assert "FROM bars_binance\n" in support.QUERY
    assert "symbol='BTCUSDT'" in support.QUERY
    assert "interval='1m'" in support.QUERY
    assert "SELECT ts,open,high,low,close," in support.QUERY
    assert support.QUERY_START == pd.Timestamp("2023-04-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    lowered = support.QUERY.lower()
    for forbidden in ("funding", "execution", "gross9", "outcome", "pnl"):
        assert forbidden not in lowered


def test_prepare_source_requires_schema_unique_timestamps_and_coherent_positive_ohlc() -> None:
    raw = _bars(pd.Timestamp("2024-01-01T00:00:00Z"), periods=3)
    prepared = support.prepare_source(raw)
    assert prepared.source_valid.tolist() == [True, True, True]

    incoherent = raw.copy()
    incoherent.loc[1, "high"] = incoherent.loc[1, "low"] - 1
    assert support.prepare_source(incoherent).source_valid.tolist() == [True, False, True]

    duplicate = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate source timestamps"):
        support.prepare_source(duplicate)
    with pytest.raises(RuntimeError, match="schema drift"):
        support.prepare_source(raw.drop(columns="duplicate_count"))


def test_boundary_uses_360_rows_359_close_to_next_open_gaps_and_earliest_tie() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    returns = np.zeros(1440)
    returns[-360:] = 0.03 / 360
    # Final-block row positions 11 and 21 correspond to gap array indices 10 and 20.
    tied_gap = math.log(2.0)
    raw = _bars(
        start,
        gaps={1080 + 11: tied_gap, 1080 + 21: -tied_gap},
        candle_returns=returns,
    )
    feature = support.boundary_features(support.prepare_source(raw), decision)

    assert feature["source_valid"] is True
    assert feature["valid_minute_count"] == 1440
    assert feature["gap_count"] == 359
    assert feature["dominant_gap_index"] == 11
    assert feature["latest_dominant_gap_index"] == 21
    assert feature["dominant_gap_time"] == decision - pd.Timedelta(hours=6) + pd.Timedelta(minutes=11)
    assert feature["latest_dominant_gap_time"] == decision - pd.Timedelta(hours=6) + pd.Timedelta(minutes=21)
    assert feature["dominant_gap"] == pytest.approx(tied_gap)
    assert feature["latest_dominant_gap"] == pytest.approx(-tied_gap)
    assert feature["gap_energy"] == pytest.approx(2 * tied_gap**2)
    assert feature["gap_dominance"] == pytest.approx(0.5)
    assert feature["block_return"] == pytest.approx(0.03)
    assert feature["direction_alignment"] is True
    assert feature["latest_direction_alignment"] is False


def test_realized_variation_is_exact_24h_close_open_variation_not_gap_variation() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    returns = np.linspace(-0.001, 0.0012, 1440)
    raw = _bars(start, gaps={1200: 0.04}, candle_returns=returns)
    feature = support.boundary_features(support.prepare_source(raw), decision)
    assert feature["source_valid"] is True
    assert feature["btc_realized_variation"] == pytest.approx(
        math.sqrt(float(np.square(returns).sum()))
    )
    assert feature["btc_realized_variation"] != pytest.approx(
        math.sqrt(float(np.square(returns).sum()) + 0.04**2)
    )


def test_boundary_rejects_missing_incoherent_zero_energy_and_zero_block_return() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    missing = _bars(start).drop(index=400).reset_index(drop=True)
    result = support.boundary_features(support.prepare_source(missing), decision)
    assert result["source_valid"] is False
    assert result["valid_minute_count"] == 1439

    zero_energy = support.boundary_features(support.prepare_source(_bars(start)), decision)
    assert zero_energy["source_valid"] is False

    # Preserve nonzero gap energy while forcing last close == first block open exactly.
    zero_return = _bars(start, gaps={1200: 0.01})
    first_block_open = zero_return.loc[1080, "open"]
    zero_return.loc[1439, ["open", "high", "low", "close"]] = first_block_open
    result = support.boundary_features(support.prepare_source(zero_return), decision)
    assert result["source_valid"] is False


def test_strict_prior_midrank_excludes_current_skips_invalid_and_caps_at_2160() -> None:
    values = pd.Series([*map(float, range(2161)), np.nan, 2160.0])
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[:1440]).all()
    assert ranks.iloc[1440] == 1.0
    assert ranks.iloc[2160] == 1.0
    assert np.isnan(ranks.iloc[2161])
    # Last 2160 finite priors are 1..2160; the current tied maximum is excluded.
    assert ranks.iloc[2162] == pytest.approx((2159 + 0.5) / 2160)


def test_primary_thresholds_are_inclusive_and_each_frozen_gate_control_is_isolated() -> None:
    features = _feature_frame()
    features.loc[1, "gap_dominance_rank"] = 0.799
    features.loc[2, "block_return_rank"] = 0.599
    features.loc[3, "variation_rank"] = 0.649
    primary, side, _ = support.active_and_side(features)
    assert primary.tolist() == [True, False, False, False]
    assert side.tolist() == [1, 1, 1, 1]
    assert support.active_and_side(features, "no_gap_dominance_gate")[0].tolist() == [True, True, False, False]
    assert support.active_and_side(features, "no_block_return_tail")[0].tolist() == [True, False, True, False]
    assert support.active_and_side(features, "no_volatility_gate")[0].tolist() == [True, False, False, True]
    assert support.active_and_side(features, "direction_flip")[1].tolist() == [-1, -1, -1, -1]


def test_latest_tie_break_changes_only_selected_gap_and_alignment() -> None:
    features = _feature_frame().iloc[[0]].copy()
    primary_active, primary_side, primary_used = support.active_and_side(features)
    latest_active, latest_side, latest_used = support.active_and_side(
        features, "latest_dominant_gap_tie_break"
    )
    assert primary_active.tolist() == [True]
    assert latest_active.tolist() == [False]
    assert primary_side.tolist() == latest_side.tolist() == [1]
    assert primary_used.iloc[0].dominant_gap == pytest.approx(0.02)
    assert latest_used.iloc[0].dominant_gap == pytest.approx(-0.02)
    assert latest_used.iloc[0].gap_dominance == primary_used.iloc[0].gap_dominance
    assert latest_used.iloc[0].gap_dominance_rank == primary_used.iloc[0].gap_dominance_rank


def test_clock_has_exact_delay_hold_global_half_open_reservation_and_split_containment() -> None:
    clock = support.build_clock(_feature_frame())
    # 01:05 is suppressed by the globally reserved [00:05, 06:05) interval.
    assert clock.decision_time.tolist() == pd.to_datetime(
        ["2024-07-01T00:00:00Z", "2024-07-01T06:00:00Z", "2024-07-01T12:00:00Z"]
    ).tolist()
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=6)).all()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert set(clock.split) == {"test"}

    crossing = _feature_frame(["2023-12-31T23:00:00Z", "2024-01-01T06:00:00Z"])
    crossing_clock = support.build_clock(crossing)
    assert crossing_clock.decision_time.tolist() == [pd.Timestamp("2024-01-01T06:00:00Z")]


def test_support_stats_enforces_side_and_month_concentration_inputs() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "side": [1, 1, 1, -1],
            "entry_time": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-02-01", "2024-03-01"], utc=True
            ),
        }
    )
    assert support.support_stats(frame, "test") == {
        "events": 4,
        "longs": 3,
        "shorts": 1,
        "minority_side_share": 0.25,
        "max_month_share": 0.5,
    }


def test_run_writes_source_clock_controls_reservation_and_terminal_support_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "sources"
    control_dir = tmp_path / "controls"
    monkeypatch.setattr(support, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(support, "FEATURES", source_dir / "features.csv.gz")
    monkeypatch.setattr(support, "SOURCE_MANIFEST", source_dir / "manifest.json")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "clock.csv.gz")
    monkeypatch.setattr(support, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(support, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(support, "load_source", lambda: pd.DataFrame())
    monkeypatch.setattr(support, "build_features", lambda _bars: _feature_frame())

    result = support.run()
    manifest = json.loads((source_dir / "manifest.json").read_text())
    written = json.loads((tmp_path / "result.json").read_text())
    assert written == result
    assert result["policy_id"] == "HVIGDR-6"
    assert result["ranking"] == {
        "lookback_hours": 2160,
        "minimum_prior_hours": 1440,
        "current_excluded": True,
    }
    assert result["reservation"] == {
        "scope": "global",
        "interval": "half_open",
        "equal_open_after_exit_allowed": True,
        "split_crossing_action": "skip",
    }
    assert set(result["support"]) == set(support.SPLITS)
    assert set(result["controls"]) == set(support.CONTROLS)
    assert all(not item["promotion_authorized"] for item in result["controls"].values())
    assert all((control_dir / f"{name}.csv.gz").is_file() for name in support.CONTROLS)
    assert manifest["table"] == "bars_binance"
    assert manifest["symbol"] == "BTCUSDT" and manifest["interval"] == "1m"
    assert manifest["funding_values_opened"] is False
    assert manifest["gross9_rows_opened"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
