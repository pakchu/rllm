import builtins
import gzip
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_cross_alt_rank_persistence_relay_support as support
from training import preregister_high_volatility_cross_alt_rank_persistence_relay as prereg


FIRST = np.array([-0.030, -0.020, -0.010, 0.010, 0.020, 0.030])
SECOND = np.array([0.015, -0.025, 0.035, -0.005, 0.025, 0.005])


def _bars(
    start: pd.Timestamp,
    first: np.ndarray = FIRST,
    second: np.ndarray = SECOND,
    btc_minute_returns: np.ndarray | None = None,
) -> pd.DataFrame:
    btc_returns = (
        np.linspace(-0.001, 0.0015, 360)
        if btc_minute_returns is None
        else np.asarray(btc_minute_returns, dtype=float)
    )
    frames = []
    for symbol_index, symbol in enumerate(support.SYMBOLS):
        opens = np.full(360, 100.0 + symbol_index)
        minute_returns = np.zeros(360)
        if symbol == "BTCUSDT":
            minute_returns = btc_returns.copy()
        else:
            alt_index = support.ALTS.index(symbol)
            minute_returns[179] = first[alt_index]
            minute_returns[359] = second[alt_index]
        closes = opens * np.exp(minute_returns)
        frames.append(pd.DataFrame({
            "ts": pd.date_range(start, periods=360, freq="1min"),
            "symbol": symbol, "open": opens,
            "high": np.maximum(opens, closes), "low": np.minimum(opens, closes),
            "close": closes,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ts", "symbol"]
    ).reset_index(drop=True)


def _features(decisions: list[str] | None = None) -> pd.DataFrame:
    times = pd.to_datetime(decisions or [
        "2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z",
        "2024-07-01T02:00:00Z", "2024-07-01T03:00:00Z",
    ])
    rows = []
    for decision in times:
        row = {column: 0.0 for column in support.FEATURE_COLUMNS}
        row.update({
            "decision_time": decision, "feature_available_time": decision,
            "source_valid": True, "minute_count": 2520,
            "rank_persistence": 0.9, "rank_persistence_rank": 0.9,
            "btc_variation": 0.01, "btc_variation_rank": 0.7,
            "btc_second_half_return": 0.02,
            "median_alt_second_half_return": 0.01, "direction_side": 1,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=support.FEATURE_COLUMNS)


def test_preregistration_binding_source_only_query_and_lazy_sqlalchemy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert support.PREREG_SHA == "40f8ce8063e3ff52caa0185a45afc0b272150827dd81181f08ff8579a8cb588a"
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])
    assert support.POLICY["history_hours"] == 2160
    assert support.POLICY["minimum_history_hours"] == 1440
    normalized = " ".join(support.QUERY.split())
    assert normalized.startswith("SELECT ts,symbol,open,high,low,close FROM bars_binance")
    assert "interval='1m'" in normalized
    assert all(f"'{symbol}'" in normalized for symbol in support.SYMBOLS)
    assert support.QUERY_START == pd.Timestamp("2023-04-01T00:00:00Z")
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


def test_exact_half_returns_spearman_variation_median_and_direction_formulas() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    btc_minute = np.linspace(-0.001, 0.0015, 360)
    pair = support.boundary_pair(
        support.prepare_source(_bars(start, btc_minute_returns=btc_minute)),
        start + pd.Timedelta(hours=6),
    )
    first_ranks = np.argsort(np.argsort(FIRST)) + 1
    second_ranks = np.argsort(np.argsort(SECOND)) + 1
    expected_rho = 1 - 6 * np.square(first_ranks - second_ranks).sum() / (6 * (6**2 - 1))
    assert pair["source_valid"] is True
    assert pair["minute_count"] == 360 * 7
    assert pair["rank_persistence"] == pytest.approx(expected_rho)
    assert pair["btc_variation"] == pytest.approx(np.square(btc_minute).sum())
    assert pair["btc_second_half_return"] == pytest.approx(btc_minute[-1])
    assert pair["median_alt_second_half_return"] == pytest.approx(np.median(SECOND))
    assert pair["direction_side"] == 1
    assert support.spearman_unique(FIRST, np.sort(SECOND)) == 1.0
    assert support.spearman_unique(FIRST, np.sort(SECOND)[::-1]) == -1.0


@pytest.mark.parametrize("half", ["first", "second"])
def test_ties_in_either_alt_half_fail_source_geometry(half: str) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    first, second = FIRST.copy(), SECOND.copy()
    target = first if half == "first" else second
    target[1] = target[0]
    pair = support.boundary_pair(
        support.prepare_source(_bars(start, first, second)),
        start + pd.Timedelta(hours=6),
    )
    assert pair["source_valid"] is False
    assert np.isnan(pair["rank_persistence"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "incoherent"])
def test_exact_360_by_7_geometry_and_coherent_ohlc_fail_closed(mutation: str) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    raw = _bars(start)
    if mutation == "missing":
        raw = raw.drop(raw[raw.symbol.eq("ADAUSDT")].index[10])
    elif mutation == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        with pytest.raises(RuntimeError, match="duplicate source key"):
            support.prepare_source(raw)
        return
    else:
        index = raw[raw.symbol.eq("BNBUSDT")].index[20]
        raw.loc[index, "high"] = raw.loc[index, "low"] - 1
    pair = support.boundary_pair(support.prepare_source(raw), start + pd.Timedelta(hours=6))
    assert pair["source_valid"] is False


def test_strict_prior_midrank_excludes_current_skips_invalid_and_caps_at_2160() -> None:
    values = pd.Series([*map(float, range(2161)), np.nan, 2160.0])
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:1440].isna().all()
    assert ranks.iloc[1440] == 1.0
    assert ranks.iloc[2160] == 1.0
    assert np.isnan(ranks.iloc[2161])
    assert ranks.iloc[2162] == pytest.approx((2159 + 0.5) / 2160)


def test_both_feature_ranks_use_only_source_valid_decisions() -> None:
    rows = []
    start = pd.Timestamp("2023-04-02T00:00:00Z")
    for index in range(1442):
        row = _features([str(start + pd.Timedelta(hours=index))]).iloc[0].to_dict()
        row["rank_persistence"] = index + 1.0
        row["btc_variation"] = index + 2.0
        if index == 100:
            row["source_valid"] = False
        rows.append({column: row[column] for column in support.PAIR_COLUMNS})
    features = support.build_features(pd.DataFrame(rows, columns=support.PAIR_COLUMNS))
    for column in ("rank_persistence_rank", "btc_variation_rank"):
        assert np.isnan(features[column].iloc[1440])
        assert features[column].iloc[1441] == 1.0


def test_primary_gates_strict_sign_and_exact_source_valid_onset() -> None:
    features = _features()
    features["rank_persistence_rank"] = [0.7, 0.8, 0.9, 0.7]
    eligible, onset, side, _ = support.active_and_side(features)
    assert eligible.tolist() == [False, True, True, False]
    assert onset.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, 1, 1]

    features.loc[1, "direction_side"] = 0
    assert not bool(support.active_and_side(features)[0].iloc[1])
    features.loc[1, "direction_side"] = -1
    assert bool(support.active_and_side(features)[0].iloc[1])
    assert support.active_and_side(features)[2].iloc[1] == -1
    features.loc[0, "source_valid"] = False
    assert not bool(support.active_and_side(features)[1].iloc[1])
    features.loc[0, "source_valid"] = True
    features.loc[1, "decision_time"] += pd.Timedelta(minutes=1)
    assert not bool(support.active_and_side(features)[1].iloc[1])


def test_all_six_controls_are_isolated_and_reversal_uses_current_rho() -> None:
    features = _features(["2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z"])
    features["rank_persistence_rank"] = 0.7
    assert not support.active_and_side(features)[0].any()
    assert support.active_and_side(features, "no_rank_persistence_gate")[0].all()

    features["btc_variation_rank"] = 0.6
    assert not support.active_and_side(features, "no_rank_persistence_gate")[0].any()
    features["rank_persistence_rank"] = 0.9
    assert support.active_and_side(features, "no_btc_variation_gate")[0].all()

    features["btc_variation_rank"] = 0.7
    features["rank_persistence_rank"] = [0.99, 0.01]
    features["rank_persistence"] = [-0.79, -0.80]
    assert support.active_and_side(features, "second_half_rank_reversal")[0].tolist() == [False, True]
    assert support.active_and_side(features, "direction_flip")[2].tolist() == [-1, -1]
    assert support.active_and_side(features, "forced_long")[2].tolist() == [1, 1]

    features.loc[0, "rank_persistence"] = -1.0
    _, _, _, stale = support.active_and_side(features, "one_hour_stale_features")
    assert stale.loc[1, "rank_persistence"] == -1.0
    assert stale.loc[1, "feature_available_time"] == features.loc[0, "decision_time"]
    assert pd.isna(stale.loc[0, "source_valid"])


def test_clock_uses_d_plus_5_hold_8h_global_half_open_and_split_crossing_skip() -> None:
    times = pd.date_range("2023-12-31T14:00:00Z", "2024-01-01T09:00:00Z", freq="1h")
    features = _features([str(value) for value in times])
    features["rank_persistence_rank"] = 0.7
    onset_hours = {15, 17, 23, 1, 9}
    for index, decision in enumerate(times):
        if decision.hour in onset_hours:
            features.loc[index, "rank_persistence_rank"] = 0.8
    clock = support.build_clock(features)
    assert clock.decision_time.tolist() == pd.to_datetime([
        "2023-12-31T15:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T09:00:00Z"
    ]).tolist()
    assert clock.entry_time.iloc[2] == clock.exit_time.iloc[1]
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=8)).all()
    assert clock.split.tolist() == ["train", "test", "test"]
    assert pd.Timestamp("2023-12-31T23:00:00Z") not in set(clock.decision_time)


def test_deterministic_immutable_artifacts(tmp_path: Path) -> None:
    frame = _features().iloc[:2]
    first = support.deterministic_csv_gzip(frame)
    assert first == support.deterministic_csv_gzip(frame.copy())
    assert gzip.decompress(first).startswith(b"decision_time,feature_available_time")
    path = tmp_path / "artifact.csv.gz"
    support.write_immutable(path, first)
    support.write_immutable(path, first)
    with pytest.raises(RuntimeError, match="immutable HVCARP artifact"):
        support.write_immutable(path, first + b"drift")


def test_support_stats_and_result_contract_fields() -> None:
    clock = pd.DataFrame({
        "split": ["test", "test", "test"],
        "side": [1, 1, -1],
        "entry_time": pd.to_datetime([
            "2024-01-01T00:05:00Z", "2024-01-15T00:05:00Z", "2024-02-01T00:05:00Z"
        ]),
    })
    assert support.support_stats(clock, "test") == {
        "events": 3, "longs": 2, "shorts": 1,
        "minority_side_share": pytest.approx(1 / 3),
        "max_month_share": pytest.approx(2 / 3),
    }
    assert support.RESULT.name == "high_volatility_cross_alt_rank_persistence_relay_support_2026-08-10.json"
    assert support.CLOCK.name.endswith("clocks_2023_2026.csv.gz")
