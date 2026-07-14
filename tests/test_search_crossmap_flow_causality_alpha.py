from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import search_crossmap_flow_causality_alpha as crossmap


def test_delay_embedding_is_current_to_oldest_lag() -> None:
    values = np.arange(6, dtype=float)
    embedded = crossmap.delay_embedding(values, dimension=3)
    np.testing.assert_array_equal(
        embedded,
        np.array([[2, 1, 0], [3, 2, 1], [4, 3, 2], [5, 4, 3]], dtype=float),
    )


def test_theiler_matrix_excludes_self_and_adjacent_rows() -> None:
    embedded = crossmap.delay_embedding(np.arange(8, dtype=float), dimension=3)
    distance = crossmap.theiler_distance_matrix(embedded, radius=1)
    assert np.isinf(np.diag(distance)).all()
    assert np.isinf(distance[np.arange(len(distance) - 1), np.arange(1, len(distance))]).all()
    assert np.isfinite(distance[0, 2])


def test_cross_map_skill_is_finite_for_nonlinear_library() -> None:
    x = np.sin(np.arange(120) / 7.0)
    y = np.tanh(1.7 * x) + 0.05 * np.cos(np.arange(120) / 5.0)
    skill = crossmap.cross_map_skill(
        crossmap.delay_embedding(x),
        y[crossmap.EMBEDDING_DIMENSION - 1 :],
    )
    assert np.isfinite(skill)
    assert -1.0 <= skill <= 1.0


def _block_frame(rows: int = 300) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "position": np.arange(rows, dtype=int),
            "effective_time": pd.date_range("2020-01-01", periods=rows, freq="6h"),
            "source_time": pd.date_range("2019-12-31 23:55", periods=rows, freq="6h"),
            "price_return": 0.01 * np.sin(index / 9.0) + 0.003 * np.cos(index / 4.0),
            "flow_fraction": 0.2 * np.tanh(np.sin((index - 1.0) / 9.0)) + 0.02 * np.cos(index / 11.0),
            "quote_volume": np.full(rows, 1e9),
        }
    )


def test_crossmap_features_are_prefix_invariant() -> None:
    blocks = _block_frame()
    first = crossmap.build_crossmap_features(blocks)
    changed = blocks.copy()
    changed.loc[230:, "price_return"] = 1e6
    changed.loc[230:, "flow_fraction"] = -1e6
    second = crossmap.build_crossmap_features(changed)
    columns = [
        "flow_to_price_skill",
        "price_to_flow_skill",
        "crossmap_dominance",
        "dominance_threshold",
    ]
    np.testing.assert_allclose(
        first.loc[:229, columns],
        second.loc[:229, columns],
        equal_nan=True,
    )


def test_policy_follows_flow_when_dominance_positive_and_fades_when_negative() -> None:
    features = pd.DataFrame(
        {
            "position": [0, 1, 2, 3],
            "crossmap_dominance": [0.3, -0.4, 0.1, np.nan],
            "dominance_threshold": [0.2, 0.2, 0.2, 0.2],
            "flow_fraction": [1.0, 1.0, -1.0, 1.0],
        }
    )
    long_active, short_active = crossmap.policy_masks(features, rows=4)
    assert np.flatnonzero(long_active).tolist() == [0]
    assert np.flatnonzero(short_active).tolist() == [1]


def test_gate_equality_is_inactive() -> None:
    features = pd.DataFrame(
        {
            "position": [0],
            "crossmap_dominance": [0.2],
            "dominance_threshold": [0.2],
            "flow_fraction": [1.0],
        }
    )
    long_active, short_active = crossmap.policy_masks(features, rows=1)
    assert not long_active.any()
    assert not short_active.any()


def test_completed_block_waits_for_minute55_source() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=145, freq="5min"))
    market = pd.DataFrame(
        {
            "open": np.linspace(100.0, 101.0, len(dates)),
            "close": np.linspace(100.1, 101.1, len(dates)),
            "quote_asset_volume": np.full(len(dates), 1e6),
            "taker_buy_quote": np.full(len(dates), 5.1e5),
        }
    )
    blocks = crossmap.build_completed_blocks(market, dates)
    row = blocks.iloc[0]
    assert row["effective_time"] == pd.Timestamp("2023-01-01 06:00")
    assert row["source_time"] == pd.Timestamp("2023-01-01 05:55")
    assert int(row["position"]) == 72


def test_load_pre2024_returns_only_pre_cutoff_rows(tmp_path: Path) -> None:
    dates = pd.date_range("2023-12-31 23:50", periods=4, freq="5min", tz="UTC")
    source = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(4, 100.0),
            "high": np.full(4, 101.0),
            "low": np.full(4, 99.0),
            "close": np.full(4, 100.5),
            "quote_asset_volume": np.full(4, 1e6),
            "taker_buy_quote": np.full(4, 5e5),
        }
    )
    path = tmp_path / "crossing.csv.gz"
    source.to_csv(path, index=False, compression="gzip")
    market, returned_dates = crossmap.load_pre2024(path)
    assert len(market) == 2
    assert returned_dates.max() == pd.Timestamp("2023-12-31 23:55")
    assert (returned_dates < pd.Timestamp(crossmap.CUTOFF)).all()


def test_support_counts_are_nonoverlapping_and_split_contained(monkeypatch) -> None:
    monkeypatch.setitem(crossmap.WINDOWS, "sample", ("2023-01-01", "2023-01-01 01:00"))
    dates = pd.Series(pd.date_range("2023-01-01", periods=13, freq="5min"))
    long_active = np.array([True, True, False, False, False, False, True, False, False, False, False, False, False])
    short_active = np.array([False, False, False, False, True, False, False, False, False, False, True, False, False])
    counts = crossmap.support_counts(
        dates,
        long_active,
        short_active,
        window="sample",
        hold_bars=2,
    )
    assert counts["raw"] == 5
    assert counts["strict_executable"] == 2
    assert counts["strict_executable_long"] == 1
    assert counts["strict_executable_short"] == 1


def test_executable_contract_enters_next_bar_and_stays_inside_split() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=8, freq="5min"))
    long_active = np.array([True, False, False, True, False, True, False, False])
    short_active = np.zeros(8, dtype=bool)
    positions = crossmap.select_executable_positions(
        dates,
        long_active,
        short_active,
        start="2023-01-01",
        end="2023-01-01 00:35",
        hold_bars=2,
    )
    # Signal 00:00 enters 00:05 and exits 00:15. Signal 00:15 overlaps;
    # signal 00:25 would exit at the exclusive 00:35 split boundary.
    assert positions == [0]


def test_support_only_cannot_open_outcomes_or_write(monkeypatch, tmp_path: Path) -> None:
    market = pd.DataFrame({"low": [1.0], "high": [1.0]})
    dates = pd.Series([pd.Timestamp("2023-01-01")])
    blocks = pd.DataFrame(
        {
            "source_time": [pd.Timestamp("2022-12-31 23:55")],
            "position": [0],
        }
    )
    features = pd.DataFrame(
        {
            "position": [0],
            "crossmap_dominance": [1.0],
            "linear_leadlag_dominance": [0.0],
            "price_flow_correlation": [0.0],
            "dominance_threshold": [0.5],
            "linear_threshold": [0.5],
            "flow_fraction": [1.0],
            "price_return": [1.0],
        }
    )
    monkeypatch.setattr(crossmap, "load_pre2024", lambda *args: (market, dates))
    monkeypatch.setattr(crossmap, "build_completed_blocks", lambda *args: blocks)
    monkeypatch.setattr(crossmap, "build_crossmap_features", lambda *args: features)
    monkeypatch.setattr(crossmap, "finite_spearman", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(crossmap, "event_jaccard", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        crossmap,
        "support_counts",
        lambda *args, window, **kwargs: {
            "raw": 300,
            "raw_long": 150,
            "raw_short": 150,
            "strict_executable": 300 if window == "fit" else 80,
            "strict_executable_long": 150 if window == "fit" else 40,
            "strict_executable_short": 150 if window == "fit" else 40,
        },
    )
    monkeypatch.setattr(crossmap, "RESULT_PATH", tmp_path / "forbidden.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("support-only crossed the outcome boundary")

    monkeypatch.setattr(crossmap, "_future_extreme", forbidden)
    monkeypatch.setattr(crossmap, "simulate", forbidden)
    monkeypatch.setattr(crossmap, "_frame_sha256", forbidden)
    output = crossmap.run(support_only=True)
    assert output["preflight_passed"] is True
    assert output["outcomes_opened"] is False
    assert not crossmap.RESULT_PATH.exists()
