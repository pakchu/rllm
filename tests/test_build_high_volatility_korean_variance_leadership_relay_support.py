import gzip
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_korean_variance_leadership_relay_support as support
from training import preregister_high_volatility_korean_variance_leadership_relay as prereg


def _bars(
    start: pd.Timestamp,
    periods: int = 360,
    *,
    minute_return: float = 0.001,
    final_direction: int = 1,
) -> pd.DataFrame:
    minute = np.arange(periods)
    opens = 100.0 + minute * 0.01
    signs = np.where(minute % 2 == 0, 1.0, -1.0)
    signs[-60:] = final_direction
    closes = opens * np.exp(signs * minute_return)
    return pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0001,
            "low": np.minimum(opens, closes) * 0.9999,
            "close": closes,
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
            "upbit_source_rows": [360] * count,
            "binance_source_rows": [360] * count,
            "upbit_variation": [0.00072] * count,
            "binance_variation": [0.00036] * count,
            "variance_leadership": [math.log(2)] * count,
            "upbit_final_hour_return": [0.02] * count,
            "binance_final_hour_return": [0.01] * count,
            "return_magnitude_leadership": [math.log(2)] * count,
            "variance_leadership_rank": [0.90] * count,
            "binance_variation_rank": [0.70] * count,
            "return_magnitude_leadership_rank": [0.90] * count,
        }
    )
    return frame.loc[:, support.FEATURE_COLUMNS]


def test_preregistration_is_bound_to_exact_committed_artifact() -> None:
    assert support.PREREG_SHA == (
        "adef4c9ba8f020c410692a55328b1f57b0a6f150b19bfa229818e8e613080d1e"
    )
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])


def test_queries_are_exact_source_only_ohlc_contracts() -> None:
    upbit = " ".join(support.UPBIT_QUERY.split())
    binance = " ".join(support.BINANCE_QUERY.split())
    assert upbit.startswith("SELECT ts,open,high,low,close FROM bars_upbit")
    assert "symbol='KRW-BTC'" in upbit and "interval='1m'" in upbit
    assert binance.startswith("SELECT ts,open,high,low,close FROM bars_binance")
    assert "symbol='BTCUSDT'" in binance and "interval='1m'" in binance
    assert support.START == pd.Timestamp("2023-04-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    for query in (support.UPBIT_QUERY, support.BINANCE_QUERY):
        for forbidden in (
            "volume", "trade", "funding", "execution_price", "gross9", "pnl"
        ):
            assert forbidden not in query.lower()


def test_load_sources_uses_one_connection_exact_queries_and_disposes(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Engine:
        disposed = False

        def connect(self):
            return Connection()

        def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(support, "postgres_engine", lambda: engine)
    monkeypatch.setitem(
        sys.modules, "sqlalchemy", types.SimpleNamespace(text=lambda query: query)
    )

    def fake_read(query, _connection, params):
        calls.append((query, params))
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])

    monkeypatch.setattr(pd, "read_sql_query", fake_read)
    upbit, binance = support.load_sources()
    assert upbit.empty and binance.empty
    assert [call[0] for call in calls] == [support.UPBIT_QUERY, support.BINANCE_QUERY]
    assert all(set(params) == {"start", "end"} for _, params in calls)
    assert engine.disposed is True


def test_prepare_source_is_strict_about_schema_timestamps_and_coherent_ohlc() -> None:
    raw = _bars(pd.Timestamp("2024-01-01T00:00:00Z"), periods=4)
    prepared = support.prepare_source(raw, "upbit")
    assert prepared.row_valid.tolist() == [True] * 4

    invalid = raw.copy()
    invalid.loc[0, "open"] = 0
    invalid.loc[1, "high"] = invalid.loc[1, ["open", "close"]].min() - 1
    invalid.loc[2, "low"] = invalid.loc[2, ["open", "close"]].max() + 1
    invalid.loc[3, "close"] = np.inf
    assert support.prepare_source(invalid, "upbit").row_valid.tolist() == [False] * 4

    with pytest.raises(RuntimeError, match="schema drift"):
        support.prepare_source(raw.drop(columns="high"), "upbit")
    duplicate = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate upbit source timestamps"):
        support.prepare_source(duplicate, "upbit")


def test_boundary_pair_uses_exact_360_minute_variations_ratio_and_final_hour() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=6)
    upbit_raw = _bars(start, minute_return=0.002, final_direction=1)
    binance_raw = _bars(start, minute_return=0.001, final_direction=1)
    pair = support.boundary_pair(
        support.prepare_source(upbit_raw, "upbit"),
        support.prepare_source(binance_raw, "binance"),
        decision,
    )

    upbit_expected = float(
        np.square(np.log(upbit_raw.close.to_numpy() / upbit_raw.open.to_numpy())).sum()
    )
    binance_expected = float(
        np.square(
            np.log(binance_raw.close.to_numpy() / binance_raw.open.to_numpy())
        ).sum()
    )
    assert pair["source_valid"] is True
    assert pair["upbit_source_rows"] == pair["binance_source_rows"] == 360
    assert pair["upbit_variation"] == pytest.approx(upbit_expected)
    assert pair["binance_variation"] == pytest.approx(binance_expected)
    assert pair["variance_leadership"] == pytest.approx(
        math.log(upbit_expected / binance_expected)
    )
    assert pair["upbit_final_hour_return"] == pytest.approx(
        math.log(upbit_raw.close.iloc[-1] / upbit_raw.open.iloc[300])
    )
    assert pair["binance_final_hour_return"] == pytest.approx(
        math.log(binance_raw.close.iloc[-1] / binance_raw.open.iloc[300])
    )
    assert pair["return_magnitude_leadership"] == pytest.approx(
        math.log(
            abs(pair["upbit_final_hour_return"])
            / abs(pair["binance_final_hour_return"])
        )
    )


@pytest.mark.parametrize(
    ("venue", "mutation"),
    [
        ("upbit", "missing"),
        ("binance", "missing"),
        ("upbit", "incoherent"),
        ("binance", "nan"),
        ("upbit", "zero_variation"),
        ("binance", "zero_variation"),
    ],
)
def test_boundary_pair_fails_closed_for_either_incoherent_aligned_source(
    venue: str, mutation: str
) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = start + pd.Timedelta(hours=6)
    frames = {"upbit": _bars(start), "binance": _bars(start)}
    target = frames[venue]
    if mutation == "missing":
        frames[venue] = target.drop(index=10).reset_index(drop=True)
    elif mutation == "incoherent":
        target.loc[10, "high"] = target.loc[10, "low"] - 1
    elif mutation == "nan":
        target.loc[10, "close"] = np.nan
    else:
        target["close"] = target["open"]
        target["high"] = target["open"]
        target["low"] = target["open"]
    pair = support.boundary_pair(
        support.prepare_source(frames["upbit"], "upbit"),
        support.prepare_source(frames["binance"], "binance"),
        decision,
    )
    assert pair["source_valid"] is False
    assert math.isnan(pair["variance_leadership"])


def test_pair_panel_builds_every_exact_hour_in_frozen_window(monkeypatch) -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    monkeypatch.setattr(support, "START", start)
    monkeypatch.setattr(support, "END", start + pd.Timedelta(hours=3))
    empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    panel = support.build_pair_panel(empty, empty)
    assert panel.decision_time.tolist() == pd.date_range(
        start, periods=3, freq="1h"
    ).tolist()
    assert panel.columns.tolist() == list(support.PAIR_COLUMNS)
    assert not panel.source_valid.any()


def test_strict_prior_midrank_excludes_current_skips_invalid_and_caps_history() -> None:
    values = pd.Series([*map(float, range(2161)), np.nan, 2160.0])
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:1440].isna().all()
    assert ranks.iloc[1440] == 1.0
    assert ranks.iloc[2160] == 1.0
    assert np.isnan(ranks.iloc[2161])
    assert ranks.iloc[2162] == pytest.approx((2159 + 0.5) / 2160)


def test_feature_ranks_use_only_strict_prior_source_valid_decisions() -> None:
    rows = []
    start = pd.Timestamp("2023-04-01T00:00:00Z")
    for index in range(1442):
        row = _features([str(start + pd.Timedelta(hours=index))]).iloc[0].to_dict()
        row["upbit_variation"] = 2.0 + index
        row["binance_variation"] = 1.0 + index
        row["variance_leadership"] = index / 1000
        row["return_magnitude_leadership"] = index / 2000
        if index == 100:
            row["source_valid"] = False
        rows.append({column: row[column] for column in support.PAIR_COLUMNS})
    features = support.build_features(pd.DataFrame(rows, columns=support.PAIR_COLUMNS))
    assert np.isnan(features.variance_leadership_rank.iloc[1440])
    assert features.variance_leadership_rank.iloc[1441] == 1.0
    assert features.binance_variation_rank.iloc[1441] == 1.0
    assert features.return_magnitude_leadership_rank.iloc[1441] == 1.0


def test_source_valid_onset_requires_adjacent_valid_ineligible_prior_hour() -> None:
    features = _features()
    features["variance_leadership_rank"] = [0.1, 0.9, 0.9, 0.1, 0.9, 0.9]
    eligible, onset, side, _ = support.active_and_side(features)
    assert eligible.tolist() == [False, True, True, False, True, True]
    assert onset.tolist() == [False, True, False, False, True, False]
    assert side.tolist() == [1] * len(features)
    features.loc[0, "source_valid"] = False
    assert not bool(support.active_and_side(features)[1].iloc[1])


def test_same_sign_confirmation_is_strict_and_controls_are_isolated() -> None:
    features = _features(["2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z"])
    features.loc[0, ["variance_leadership_rank", "binance_variation_rank"]] = [0.1, 0.1]
    features.loc[1, ["variance_leadership_rank", "binance_variation_rank"]] = [0.9, 0.1]
    assert support.active_and_side(features)[0].tolist() == [False, False]
    assert support.active_and_side(features, "no_binance_variation_gate")[0].tolist() == [False, True]
    features.loc[:, "binance_variation_rank"] = 0.7
    features.loc[1, "variance_leadership_rank"] = 0.1
    assert support.active_and_side(features, "no_variance_leadership_tail")[0].tolist() == [True, True]

    features.loc[:, ["variance_leadership_rank", "return_magnitude_leadership_rank"]] = [0.1, 0.9]
    assert support.active_and_side(features, "return_magnitude_leadership")[0].tolist() == [True, True]
    assert support.active_and_side(features, "direction_flip")[2].tolist() == [-1, -1]
    assert support.active_and_side(features, "forced_long")[2].tolist() == [1, 1]

    features.loc[0, "upbit_final_hour_return"] = -0.01
    features.loc[1, "upbit_final_hour_return"] = 0.0
    for control in ("no_binance_variation_gate", "no_variance_leadership_tail"):
        assert support.active_and_side(features, control)[0].tolist() == [False, False]


def test_stale_control_uses_exact_prior_hour_geometry_and_availability() -> None:
    features = _features(["2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z"])
    features.loc[0, "upbit_final_hour_return"] = -0.02
    features.loc[0, "binance_final_hour_return"] = -0.01
    _, _, stale_side, stale = support.active_and_side(
        features, "one_hour_stale_features"
    )
    assert stale_side.iloc[1] == -1
    assert stale.loc[1, "variance_leadership"] == features.loc[0, "variance_leadership"]
    assert stale.loc[1, "feature_available_time"] == features.loc[0, "decision_time"]


def test_clock_uses_onsets_global_half_open_reservation_and_split_skip() -> None:
    features = _features(
        [
            "2023-12-31T22:00:00Z", "2023-12-31T23:00:00Z",
            "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z",
            "2024-01-01T06:00:00Z", "2024-01-01T07:00:00Z",
            "2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z",
        ]
    )
    features["variance_leadership_rank"] = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    clock = support.build_clock(features)
    assert clock.decision_time.tolist() == pd.to_datetime(
        ["2024-01-01T01:00:00Z", "2024-01-01T07:00:00Z"]
    ).tolist()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert (clock.entry_time == clock.decision_time + pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time == clock.entry_time + pd.Timedelta(hours=6)).all()
    assert set(clock.split) == {"test"}


def test_support_stats_enforces_side_and_month_geometry() -> None:
    clock = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "side": [1, 1, 1, -1],
            "entry_time": pd.to_datetime(
                [
                    "2024-01-01T00:05:00Z", "2024-01-02T00:05:00Z",
                    "2024-02-01T00:05:00Z", "2024-03-01T00:05:00Z",
                ]
            ),
        }
    )
    stats = support.support_stats(clock, "test")
    assert stats == {
        "events": 4, "longs": 3, "shorts": 1,
        "minority_side_share": 0.25, "max_month_share": 0.5,
    }


def test_deterministic_immutable_writers_allow_identity_and_reject_drift(
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
    with pytest.raises(RuntimeError, match="immutable HVKVLR artifact"):
        support.write_immutable(path, first + b"drift")


def test_run_writes_only_source_support_artifacts_and_terminal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    control_dir = tmp_path / "controls"
    monkeypatch.setattr(support, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(support, "PAIR_PANEL", source_dir / "pair.csv.gz")
    monkeypatch.setattr(support, "FEATURE_PANEL", source_dir / "features.csv.gz")
    monkeypatch.setattr(support, "SOURCE_MANIFEST", source_dir / "manifest.json")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "clock.csv.gz")
    monkeypatch.setattr(support, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(support, "RESULT", tmp_path / "result.json")
    empty_source = pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    monkeypatch.setattr(support, "load_sources", lambda: (empty_source, empty_source))
    pair = _features().loc[:, support.PAIR_COLUMNS]
    features = _features()
    monkeypatch.setattr(support, "build_pair_panel", lambda *_sources: pair)
    monkeypatch.setattr(support, "build_features", lambda _pair: features)

    result = support.run()
    manifest = json.loads((source_dir / "manifest.json").read_text())
    written = json.loads((tmp_path / "result.json").read_text())
    assert written == result
    assert result["policy_id"] == "HVKVLR-6"
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
    assert all((control_dir / f"{name}.csv.gz").is_file() for name in support.CONTROLS)
    assert manifest["pair_panel"]["path"] == str(source_dir / "pair.csv.gz")
    assert manifest["feature_panel"]["path"] == str(source_dir / "features.csv.gz")
    assert manifest["sources"]["upbit"]["table"] == "bars_upbit"
    assert manifest["sources"]["binance"]["table"] == "bars_binance"
    assert manifest["funding_values_opened"] is False
    assert manifest["gross9_rows_opened"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
