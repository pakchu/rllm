import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_directional_trade_breadth_asymmetry_relay_support as support
from training import preregister_high_volatility_directional_trade_breadth_asymmetry_relay as prereg


def _bars(start: pd.Timestamp, periods: int = 1440) -> pd.DataFrame:
    minute = np.arange(periods)
    opens = 100.0 + (minute % 17) * 0.01
    signs = np.where(minute % 2 == 0, 1.0, -1.0)
    closes = opens * np.exp(signs * 0.001)
    return pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999,
            "close": closes,
            "number_of_trades": np.where(signs > 0, 2, 1),
            "quote_asset_volume": np.where(signs > 0, 1.0, 2.0),
        }
    )


def _features(decisions: list[str] | None = None) -> pd.DataFrame:
    times = pd.to_datetime(
        decisions
        or [
            "2024-07-01T00:00:00Z",
            "2024-07-01T01:00:00Z",
            "2024-07-01T02:00:00Z",
            "2024-07-01T06:00:00Z",
            "2024-07-01T07:00:00Z",
            "2024-07-01T08:00:00Z",
        ]
    )
    count = len(times)
    frame = pd.DataFrame(
        {
            "decision_time": times,
            "feature_available_time": times,
            "source_valid": [True] * count,
            "breadth_valid": [True] * count,
            "variation_valid": [True] * count,
            "quote_breadth_valid": [True] * count,
            "breadth_minute_count": [360] * count,
            "variation_minute_count": [1440] * count,
            "positive_minute_count": [180] * count,
            "negative_minute_count": [180] * count,
            "zero_return_minute_count": [0] * count,
            "up_count": [360] * count,
            "down_count": [180] * count,
            "directional_trade_breadth": [1 / 3] * count,
            "up_quote_volume": [180.0] * count,
            "down_quote_volume": [360.0] * count,
            "quote_volume_directional_breadth": [-1 / 3] * count,
            "btc_realized_variation": [0.00144] * count,
            "absolute_breadth_rank": [0.90] * count,
            "absolute_quote_breadth_rank": [0.90] * count,
            "variation_rank": [0.70] * count,
        }
    )
    return frame.loc[:, support.FEATURE_COLUMNS]


def test_preregistration_is_bound_to_exact_committed_artifact() -> None:
    assert support.PREREG_SHA == (
        "21adc9b086b13be8ed798709f0e49ed2d062fb80722679b60a6f550ef8b0a5db"
    )
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])


def test_query_is_exact_source_only_btcusdt_one_minute_contract() -> None:
    normalized = " ".join(support.QUERY.split())
    assert normalized.startswith(
        "SELECT ts,open,high,low,close,number_of_trades,quote_asset_volume FROM bars_binance"
    )
    assert "symbol='BTCUSDT'" in normalized
    assert "interval='1m'" in normalized
    assert support.QUERY_START == pd.Timestamp("2023-04-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    for forbidden in ("funding", "execution_price", "gross9", "pnl", "return"):
        assert forbidden not in support.QUERY.lower()


def test_prepare_source_is_strict_about_schema_timestamp_ohlc_trade_count_and_quote() -> None:
    raw = _bars(pd.Timestamp("2024-01-01T00:00:00Z"), periods=3)
    prepared = support.prepare_source(raw)
    assert prepared.primary_row_valid.tolist() == [True, True, True]
    assert prepared.quote_row_valid.tolist() == [True, True, True]

    invalid = raw.astype({"number_of_trades": float})
    invalid.loc[0, "high"] = invalid.loc[0, "low"] - 1
    invalid.loc[1, "number_of_trades"] = 1.5
    invalid.loc[2, "quote_asset_volume"] = -1
    prepared = support.prepare_source(invalid)
    assert prepared.primary_row_valid.tolist() == [False, False, True]
    assert prepared.quote_row_valid.tolist() == [False, False, False]

    with pytest.raises(RuntimeError, match="schema drift"):
        support.prepare_source(raw[[column for column in raw if column != "number_of_trades"]])
    duplicate = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate source timestamps"):
        support.prepare_source(duplicate)


def test_boundary_pair_computes_exact_frozen_six_hour_counts_and_24h_variation() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    raw = _bars(start)
    pair = support.boundary_pair(support.prepare_source(raw), decision)

    assert pair["source_valid"] is True
    assert pair["breadth_minute_count"] == 360
    assert pair["variation_minute_count"] == 1440
    assert pair["positive_minute_count"] == pair["negative_minute_count"] == 180
    assert pair["zero_return_minute_count"] == 0
    assert pair["up_count"] == 360 and pair["down_count"] == 180
    assert pair["directional_trade_breadth"] == pytest.approx(1 / 3)
    assert pair["up_quote_volume"] == 180 and pair["down_quote_volume"] == 360
    assert pair["quote_volume_directional_breadth"] == pytest.approx(-1 / 3)
    # The frozen variation is the sum of squared close/open minute returns, not sqrt.
    expected = float(
        np.square(np.log(raw.close.to_numpy(float) / raw.open.to_numpy(float))).sum()
    )
    assert pair["btc_realized_variation"] == pytest.approx(expected)
    assert pair["btc_realized_variation"] != pytest.approx(math.sqrt(expected))


def test_boundary_excludes_zero_returns_and_requires_60_minutes_each_direction() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    raw = _bars(start)
    block = raw.ts.ge(decision - pd.Timedelta(hours=6))
    positions = raw.index[block]
    raw.loc[positions[:241], "close"] = raw.loc[positions[:241], "open"]
    raw.loc[positions[:241], "high"] = raw.loc[positions[:241], "open"] * 1.0001
    raw.loc[positions[:241], "low"] = raw.loc[positions[:241], "open"] * 0.9999
    pair = support.boundary_pair(support.prepare_source(raw), decision)
    assert pair["positive_minute_count"] < 60
    assert pair["zero_return_minute_count"] == 241
    assert pair["breadth_valid"] is False
    assert pair["source_valid"] is False


@pytest.mark.parametrize("mutation", ["missing", "incoherent", "fractional", "zero_denominator"])
def test_boundary_fails_closed_on_invalid_exact_grid_or_counts(mutation: str) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    raw = _bars(start)
    if mutation == "missing":
        raw = raw.drop(index=100).reset_index(drop=True)
    elif mutation == "incoherent":
        raw.loc[100, "low"] = raw.loc[100, "high"] + 1
    elif mutation == "fractional":
        raw = raw.astype({"number_of_trades": float})
        raw.loc[1300, "number_of_trades"] = 2.5
    else:
        raw["number_of_trades"] = 0
    pair = support.boundary_pair(support.prepare_source(raw), decision)
    assert pair["source_valid"] is False


def test_invalid_quote_volume_rejects_only_quote_control_geometry() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=24)
    raw = _bars(start)
    raw.loc[1300, "quote_asset_volume"] = np.inf
    pair = support.boundary_pair(support.prepare_source(raw), decision)
    assert pair["source_valid"] is True
    assert pair["quote_breadth_valid"] is False
    assert math.isnan(pair["quote_volume_directional_breadth"])


def test_strict_prior_midrank_excludes_current_skips_invalid_and_caps_valid_history() -> None:
    values = pd.Series([*map(float, range(2161)), np.nan, 2160.0])
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:1440].isna().all()
    assert ranks.iloc[1440] == 1.0
    assert ranks.iloc[2160] == 1.0
    assert np.isnan(ranks.iloc[2161])
    assert ranks.iloc[2162] == pytest.approx((2159 + 0.5) / 2160)


def test_feature_ranks_use_only_valid_decisions_and_separate_quote_population() -> None:
    rows = []
    start = pd.Timestamp("2023-04-02T00:00:00Z")
    for index in range(1442):
        row = _features([str(start + pd.Timedelta(hours=index))]).iloc[0].to_dict()
        row["directional_trade_breadth"] = 0.1 + index / 10000
        row["quote_volume_directional_breadth"] = -(0.2 + index / 10000)
        row["btc_realized_variation"] = 1.0 + index
        if index == 100:
            row["source_valid"] = False
        rows.append({column: row[column] for column in support.PAIR_COLUMNS})
    features = support.build_features(pd.DataFrame(rows, columns=support.PAIR_COLUMNS))
    # One invalid primary decision means position 1440 has only 1439 valid priors.
    assert np.isnan(features.absolute_breadth_rank.iloc[1440])
    assert features.absolute_breadth_rank.iloc[1441] == 1.0
    # Quote validity stayed complete and therefore reaches minimum one row earlier.
    assert features.absolute_quote_breadth_rank.iloc[1440] == 1.0


def test_source_valid_onset_requires_adjacent_valid_ineligible_prior_hour() -> None:
    features = _features()
    features["absolute_breadth_rank"] = [0.1, 0.9, 0.9, 0.1, 0.9, 0.9]
    eligible, onset, side, _ = support.active_and_side(features)
    assert eligible.tolist() == [False, True, True, False, True, True]
    # The 06->07 transition is adjacent; the 02->06 gap cannot manufacture an onset.
    assert onset.tolist() == [False, True, False, False, True, False]
    assert side.tolist() == [1] * len(features)
    features.loc[0, "source_valid"] = False
    assert not bool(support.active_and_side(features)[1].iloc[1])


def test_controls_are_isolated_and_use_frozen_quote_and_stale_geometry() -> None:
    features = _features(["2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z"])
    features.loc[0, ["absolute_breadth_rank", "variation_rank"]] = [0.1, 0.1]
    features.loc[1, ["absolute_breadth_rank", "variation_rank"]] = [0.9, 0.1]
    assert support.active_and_side(features)[0].tolist() == [False, False]
    assert support.active_and_side(features, "no_variation_gate")[0].tolist() == [False, True]
    features.loc[:, "variation_rank"] = 0.7
    features.loc[1, "absolute_breadth_rank"] = 0.1
    assert support.active_and_side(features, "no_breadth_magnitude_gate")[0].tolist() == [True, True]

    features.loc[:, ["absolute_breadth_rank", "variation_rank"]] = [0.9, 0.7]
    _, _, quote_side, _ = support.active_and_side(features, "quote_volume_directional_breadth")
    assert quote_side.tolist() == [-1, -1]
    assert support.active_and_side(features, "direction_flip")[2].tolist() == [-1, -1]
    assert support.active_and_side(features, "forced_long")[2].tolist() == [1, 1]

    features.loc[0, "directional_trade_breadth"] = -0.25
    _, _, stale_side, stale = support.active_and_side(features, "one_hour_stale_features")
    assert stale_side.iloc[1] == -1
    assert stale.loc[1, "directional_trade_breadth"] == -0.25
    assert stale.loc[1, "feature_available_time"] == features.loc[0, "decision_time"]


def test_clock_uses_onsets_global_half_open_reservation_and_split_skip() -> None:
    features = _features(
        [
            "2023-12-31T22:00:00Z", "2023-12-31T23:00:00Z",  # crossing onset: skip
            "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z",  # accepted onset
            "2024-01-01T06:00:00Z", "2024-01-01T07:00:00Z",  # suppressed
            "2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z",  # accepted at equal exit
        ]
    )
    features["absolute_breadth_rank"] = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    clock = support.build_clock(features)
    assert clock.decision_time.tolist() == pd.to_datetime(
        ["2024-01-01T01:00:00Z", "2024-01-01T07:00:00Z"]
    ).tolist()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=6)).all()
    assert set(clock.split) == {"test"}


def test_deterministic_immutable_artifact_writer_allows_identity_and_rejects_drift(
    tmp_path: Path,
) -> None:
    frame = _features().iloc[:2]
    first = support.deterministic_csv_gzip(frame)
    second = support.deterministic_csv_gzip(frame.copy())
    assert first == second
    assert gzip.decompress(first).startswith(b"decision_time,feature_available_time")
    path = tmp_path / "artifact.csv.gz"
    support.write_immutable(path, first)
    support.write_immutable(path, second)
    with pytest.raises(RuntimeError, match="immutable HVDTBA artifact"):
        support.write_immutable(path, first + b"drift")


def test_run_writes_pair_feature_clock_split_control_manifest_and_terminal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    split_dir = tmp_path / "splits"
    control_dir = tmp_path / "controls"
    monkeypatch.setattr(support, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(support, "PAIR_PANEL", source_dir / "pair.csv.gz")
    monkeypatch.setattr(support, "FEATURE_PANEL", source_dir / "features.csv.gz")
    monkeypatch.setattr(support, "SOURCE_MANIFEST", source_dir / "manifest.json")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "clock.csv.gz")
    monkeypatch.setattr(support, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(support, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(support, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(support, "load_source", lambda: pd.DataFrame())
    pair = _features().loc[:, support.PAIR_COLUMNS]
    features = _features()
    monkeypatch.setattr(support, "build_pair_panel", lambda _bars: pair)
    monkeypatch.setattr(support, "build_features", lambda _pair: features)

    result = support.run()
    manifest = json.loads((source_dir / "manifest.json").read_text())
    written = json.loads((tmp_path / "result.json").read_text())
    assert written == result
    assert result["policy_id"] == "HVDTBA-6"
    assert result["ranking"] == {
        "lookback_valid_decisions": 2160,
        "minimum_prior_valid_decisions": 1440,
        "current_excluded": True,
        "ties": "midrank",
    }
    assert result["reservation"] == {
        "scope": "global", "hours": 6, "interval": "half_open",
        "equal_open_after_exit_allowed": True, "split_crossing_action": "skip",
    }
    assert set(result["controls"]) == set(support.CONTROLS)
    assert all(not item["promotion_authorized"] for item in result["controls"].values())
    assert set(result["split_artifacts"]) == set(support.SPLITS)
    assert all((control_dir / f"{name}.csv.gz").is_file() for name in support.CONTROLS)
    assert manifest["pair_panel"]["path"] == str(source_dir / "pair.csv.gz")
    assert manifest["feature_panel"]["path"] == str(source_dir / "features.csv.gz")
    assert manifest["funding_values_opened"] is False
    assert manifest["gross9_rows_opened"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
