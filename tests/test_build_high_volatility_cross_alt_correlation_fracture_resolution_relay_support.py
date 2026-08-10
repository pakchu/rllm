import builtins
import gzip
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import (
    build_high_volatility_cross_alt_correlation_fracture_resolution_relay_support as support,
)
from training import (
    preregister_high_volatility_cross_alt_correlation_fracture_resolution_relay as prereg,
)


def _return_paths(periods: int = 720) -> dict[str, np.ndarray]:
    minute = np.arange(periods, dtype=float)
    alt = 0.0011 * np.sin(minute / 13.0) + 0.0005 * np.cos(minute / 29.0)
    alt[-120:] += 0.00003
    btc = np.empty(periods)
    btc[:360] = 0.75 * alt[:360] + 0.00035 * np.sin(minute[:360] / 5.0)
    btc[360:] = -0.35 * alt[360:] + 0.00045 * np.cos(minute[360:] / 7.0)
    paths = {"BTCUSDT": btc}
    for index, symbol in enumerate(support.ALTS):
        paths[symbol] = alt + (index - 2.5) * 0.00001
    return paths


def _bars(start: pd.Timestamp, periods: int = 720) -> pd.DataFrame:
    paths = _return_paths(periods)
    frames = []
    for symbol_index, symbol in enumerate(support.SYMBOLS):
        minute = np.arange(periods, dtype=float)
        opens = 100.0 + symbol_index + minute * 0.001
        closes = opens * np.exp(paths[symbol])
        frames.append(pd.DataFrame({
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "symbol": symbol,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999,
            "close": closes,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ts", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def _features(decisions: list[str] | None = None) -> pd.DataFrame:
    times = pd.to_datetime(decisions or [
        "2024-07-01T03:00:00Z", "2024-07-02T03:00:00Z",
        "2024-07-03T03:00:00Z", "2024-07-04T03:00:00Z",
    ])
    rows = []
    for decision in times:
        row = {column: 0.0 for column in support.FEATURE_COLUMNS}
        row.update({
            "decision_time": decision,
            "feature_available_time": decision,
            "source_valid": True,
            "minute_count": 720 * 7,
            "first_half_correlation": 0.8,
            "second_half_correlation": 0.1,
            "correlation_fracture": 0.7,
            "correlation_fracture_rank": 0.8,
            "full_window_correlation": 0.4,
            "full_window_correlation_rank": 0.8,
            "btc_variation": 0.01,
            "btc_variation_rank": 0.7,
            "final_120_alt_return": 0.02,
            "direction_side": 1,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=support.FEATURE_COLUMNS)


def test_preregistration_binding_daily_source_query_and_lazy_sqlalchemy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert support.PREREG_SHA == "65032a3cbc8c0ac43fe73ce283970f2166d604d718030de67c263a7628dd08bc"
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])
    assert support.POLICY["prior_valid_days"] == 180
    assert support.POLICY["minimum_prior_valid_days"] == 90
    normalized = " ".join(support.QUERY.split())
    assert normalized.startswith("SELECT ts,symbol,open,high,low,close FROM bars_binance")
    assert "interval='1m'" in normalized
    assert all(f"'{symbol}'" in normalized for symbol in support.SYMBOLS)
    assert support.QUERY_START == pd.Timestamp("2023-01-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    assert all(word not in support.QUERY.lower() for word in ("funding", "execution", "pnl", "gross9"))

    sys.modules.pop(support.__name__, None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sqlalchemy" or name.startswith("sqlalchemy."):
            raise AssertionError("SQLAlchemy imported at module load")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module(support.__name__)


def test_exact_median_population_correlations_fracture_variation_and_side() -> None:
    start = pd.Timestamp("2024-01-01T15:00:00Z")
    pair = support.boundary_pair(
        support.prepare_source(_bars(start)),
        pd.Timestamp("2024-01-02T03:00:00Z"),
    )
    paths = _return_paths()
    btc = paths["BTCUSDT"]
    stacked = np.column_stack([paths[symbol] for symbol in support.ALTS])
    sorted_alts = np.sort(stacked, axis=1)
    alt = (sorted_alts[:, 2] + sorted_alts[:, 3]) / 2.0
    first = support.population_pearson(btc[:360], alt[:360])
    second = support.population_pearson(btc[360:], alt[360:])
    assert pair["source_valid"] is True
    assert pair["minute_count"] == 720 * 7
    assert pair["first_half_correlation"] == pytest.approx(first)
    assert pair["second_half_correlation"] == pytest.approx(second)
    assert pair["correlation_fracture"] == pytest.approx(first - second)
    assert pair["full_window_correlation"] == pytest.approx(
        support.population_pearson(btc, alt)
    )
    assert pair["btc_variation"] == pytest.approx(np.square(btc).sum())
    assert pair["final_120_alt_return"] == pytest.approx(alt[-120:].sum())
    assert pair["direction_side"] == int(np.sign(alt[-120:].sum()))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "incoherent", "nonvariance"])
def test_missing_duplicate_incoherent_and_nonvariance_fail_closed(mutation: str) -> None:
    start = pd.Timestamp("2024-01-01T15:00:00Z")
    raw = _bars(start)
    if mutation == "missing":
        raw = raw.drop(raw[raw.symbol.eq("ADAUSDT")].index[10])
    elif mutation == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        with pytest.raises(RuntimeError, match="duplicate source key"):
            support.prepare_source(raw)
        return
    elif mutation == "incoherent":
        index = raw[raw.symbol.eq("BNBUSDT")].index[20]
        raw.loc[index, "high"] = raw.loc[index, "low"] - 1.0
    else:
        indexes = raw[raw.symbol.eq("BTCUSDT")].index
        raw.loc[indexes, "close"] = raw.loc[indexes, "open"]
        raw.loc[indexes, "high"] = raw.loc[indexes, "open"]
        raw.loc[indexes, "low"] = raw.loc[indexes, "open"]
    pair = support.boundary_pair(
        support.prepare_source(raw), pd.Timestamp("2024-01-02T03:00:00Z")
    )
    assert pair["source_valid"] is False
    assert np.isnan(pair["correlation_fracture"])


def test_pair_panel_uses_only_exact_daily_0300_decisions() -> None:
    bars = _bars(pd.Timestamp("2023-01-01T15:00:00Z"), periods=720)
    original_end = support.END
    try:
        support.END = pd.Timestamp("2023-01-03T00:00:00Z")
        pair = support.build_pair_panel(bars)
    finally:
        support.END = original_end
    assert pair.decision_time.tolist() == [pd.Timestamp("2023-01-02T03:00:00Z")]


def test_strict_prior_midrank_excludes_current_skips_invalid_and_caps_at_180() -> None:
    values = pd.Series([*map(float, range(181)), np.nan, 180.0])
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:90].isna().all()
    assert ranks.iloc[90] == 1.0
    assert ranks.iloc[180] == 1.0
    assert np.isnan(ranks.iloc[181])
    assert ranks.iloc[182] == pytest.approx((179 + 0.5) / 180)


def test_all_three_ranks_use_only_source_valid_daily_decisions() -> None:
    rows = []
    start = pd.Timestamp("2023-01-02T03:00:00Z")
    for index in range(92):
        row = _features([str(start + pd.Timedelta(days=index))]).iloc[0].to_dict()
        row["correlation_fracture"] = index + 1.0
        row["full_window_correlation"] = index + 2.0
        row["btc_variation"] = index + 3.0
        if index == 10:
            row["source_valid"] = False
        rows.append({column: row[column] for column in support.PAIR_COLUMNS})
    features = support.build_features(pd.DataFrame(rows, columns=support.PAIR_COLUMNS))
    for column in (
        "correlation_fracture_rank", "full_window_correlation_rank", "btc_variation_rank"
    ):
        assert np.isnan(features[column].iloc[90])
        assert features[column].iloc[91] == 1.0


def test_primary_strict_gates_and_previous_source_valid_onset() -> None:
    features = _features()
    features["correlation_fracture_rank"] = [0.7, 0.9, 0.9, 0.9]
    features.loc[2, "source_valid"] = False
    eligible, onset, side, _ = support.active_and_side(features)
    assert eligible.tolist() == [False, True, False, True]
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, 1, 1]

    features.loc[1, "direction_side"] = 0
    assert not bool(support.active_and_side(features)[0].iloc[1])
    features.loc[1, "direction_side"] = -1
    assert bool(support.active_and_side(features)[0].iloc[1])
    assert support.active_and_side(features)[2].iloc[1] == -1
    features.loc[0, "source_valid"] = False
    assert not bool(support.active_and_side(features)[1].iloc[1])


def test_all_six_controls_are_isolated_full_correlation_has_own_rank_and_stale_skips_invalid() -> None:
    features = _features([
        "2024-07-01T03:00:00Z", "2024-07-02T03:00:00Z", "2024-07-03T03:00:00Z"
    ])
    features["correlation_fracture_rank"] = 0.7
    assert not support.active_and_side(features)[0].any()
    assert support.active_and_side(features, "no_correlation_fracture_gate")[0].all()

    features["btc_variation_rank"] = 0.6
    assert not support.active_and_side(features, "no_correlation_fracture_gate")[0].any()
    features["correlation_fracture_rank"] = 0.9
    assert support.active_and_side(features, "no_btc_variation_gate")[0].all()

    features["btc_variation_rank"] = 0.7
    features["full_window_correlation_rank"] = [0.74, 0.75, 0.74]
    assert support.active_and_side(
        features, "contemporaneous_full_window_correlation"
    )[0].tolist() == [False, True, False]
    assert support.active_and_side(features, "direction_flip")[2].tolist() == [-1, -1, -1]
    assert support.active_and_side(features, "forced_long")[2].tolist() == [1, 1, 1]

    features.loc[0, "correlation_fracture"] = -0.4
    features.loc[1, "source_valid"] = False
    _, _, _, stale = support.active_and_side(features, "one_day_stale_features")
    assert stale.loc[2, "correlation_fracture"] == -0.4
    assert stale.loc[2, "feature_available_time"] == features.loc[0, "decision_time"]
    assert pd.isna(stale.loc[0, "source_valid"])


def test_clock_d_plus_5_hold_8h_and_split_assignment() -> None:
    features = _features([
        "2023-12-30T03:00:00Z", "2023-12-31T03:00:00Z",
        "2024-01-01T03:00:00Z", "2024-01-02T03:00:00Z",
    ])
    features["correlation_fracture_rank"] = [0.7, 0.8, 0.7, 0.8]
    clock = support.build_clock(features)
    assert clock.decision_time.tolist() == pd.to_datetime([
        "2023-12-31T03:00:00Z", "2024-01-02T03:00:00Z"
    ]).tolist()
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=8)).all()
    assert clock.split.tolist() == ["train", "test"]


def test_global_half_open_reservation_allows_equal_entry_and_split_crossing_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features([
        "2024-01-01T03:00:00Z", "2024-01-02T03:00:00Z", "2024-01-03T03:00:00Z"
    ])

    def every_onset(ordered: pd.DataFrame, control: str = "primary"):
        true = pd.Series(True, index=ordered.index)
        sides = pd.Series(1, index=ordered.index)
        return true, true, sides, ordered.copy()

    monkeypatch.setattr(support, "active_and_side", every_onset)
    monkeypatch.setitem(support.POLICY, "hold_hours", 24)
    clock = support.build_clock(features)
    assert len(clock) == 3
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]

    monkeypatch.setitem(support.POLICY, "hold_hours", 25)
    overlapping = support.build_clock(features)
    assert overlapping.decision_time.tolist() == pd.to_datetime([
        "2024-01-01T03:00:00Z", "2024-01-03T03:00:00Z"
    ]).tolist()

    monkeypatch.setitem(support.POLICY, "hold_hours", 24)
    monkeypatch.setattr(support, "SPLITS", {
        "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-02T12:00:00Z"))
    })
    crossing = support.build_clock(features.iloc[:2])
    assert crossing.decision_time.tolist() == [pd.Timestamp("2024-01-01T03:00:00Z")]


def test_deterministic_immutable_artifacts_support_gates_and_result_contract(tmp_path: Path) -> None:
    frame = _features().iloc[:2]
    first = support.deterministic_csv_gzip(frame)
    assert first == support.deterministic_csv_gzip(frame.copy())
    assert gzip.decompress(first).startswith(b"decision_time,feature_available_time")
    path = tmp_path / "artifact.csv.gz"
    support.write_immutable(path, first)
    support.write_immutable(path, first)
    with pytest.raises(RuntimeError, match="immutable HVCACFR artifact"):
        support.write_immutable(path, first + b"drift")

    clock = pd.DataFrame({
        "split": ["test", "test", "test"],
        "side": [1, 1, -1],
        "entry_time": pd.to_datetime([
            "2024-01-01T03:05:00Z", "2024-01-15T03:05:00Z", "2024-02-01T03:05:00Z"
        ]),
    })
    stats = support.support_stats(clock, "test")
    assert stats["events"] == 3
    assert stats["minority_side_share"] == pytest.approx(1 / 3)
    assert stats["max_month_share"] == pytest.approx(2 / 3)
    assert support.MINIMUM_EVENTS == prereg.build()["source_support_gates"]["minimum_events"]
    assert support.RESULT.name == (
        "high_volatility_cross_alt_correlation_fracture_resolution_relay_support_2026-08-10.json"
    )
