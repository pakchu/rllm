from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training import search_visibility_layer_irreversibility_alpha as visibility
from training.search_crossmap_flow_causality_alpha import select_executable_positions


def _brute_hvg_degrees(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    inbound = np.zeros(len(values), dtype=np.int16)
    outbound = np.zeros(len(values), dtype=np.int16)
    for left in range(len(values) - 1):
        for right in range(left + 1, len(values)):
            between = values[left + 1 : right]
            if not len(between) or np.all(between < min(values[left], values[right])):
                outbound[left] += 1
                inbound[right] += 1
    return inbound, outbound


def test_linear_hvg_matches_brute_force_with_ties() -> None:
    samples = (
        np.array([1.0, 1.0, 2.0, 0.0, 2.0]),
        np.array([3.0, 2.0, 1.0, 4.0, 0.0, 5.0]),
        np.random.default_rng(7).normal(size=50),
    )
    for values in samples:
        expected = _brute_hvg_degrees(values)
        actual = visibility.directed_hvg_degrees(values)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_irreversibility_is_reversal_and_monotone_invariant() -> None:
    values = np.random.default_rng(11).normal(size=168)
    original = visibility.degree_irreversibility(values)
    reversed_value = visibility.degree_irreversibility(values[::-1])
    transformed = visibility.degree_irreversibility(np.exp(values))
    assert np.isclose(original, reversed_value, atol=1e-15)
    assert np.isclose(original, transformed, atol=1e-15)


def test_permutation_entropy_is_monotone_invariant() -> None:
    values = np.random.default_rng(13).normal(size=168)
    assert np.isclose(
        visibility.permutation_entropy(values),
        visibility.permutation_entropy(np.exp(values)),
        atol=1e-15,
    )


def _block_frame(rows: int = 330) -> pd.DataFrame:
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


def test_visibility_features_are_prefix_invariant() -> None:
    blocks = _block_frame()
    first = visibility.build_visibility_features(blocks)
    changed = blocks.copy()
    changed.loc[260:, "price_return"] = 1e6
    changed.loc[260:, "flow_fraction"] = -1e6
    second = visibility.build_visibility_features(changed)
    columns = [
        "price_hvg_irreversibility",
        "flow_hvg_irreversibility",
        "hvg_layer_log_ratio",
        "hvg_layer_threshold",
    ]
    np.testing.assert_allclose(first.loc[:259, columns], second.loc[:259, columns], equal_nan=True)


def test_visibility_window_includes_current_block_but_gate_excludes_current_score(monkeypatch) -> None:
    rows = 240
    blocks = pd.DataFrame(
        {
            "position": np.arange(rows),
            "effective_time": pd.date_range("2020-01-01", periods=rows, freq="6h"),
            "source_time": pd.date_range("2019-12-31 23:55", periods=rows, freq="6h"),
            "price_return": np.arange(1, rows + 1, dtype=float),
            "flow_fraction": np.arange(1001, 1001 + rows, dtype=float),
            "quote_volume": np.ones(rows),
        }
    )

    def terminal_value(values: np.ndarray, **kwargs) -> float:
        return float(values[-1])

    monkeypatch.setattr(visibility, "degree_irreversibility", terminal_value)
    features = visibility.build_visibility_features(blocks)
    first = visibility.VISIBILITY_BLOCKS - 1
    assert np.isnan(features.loc[first - 1, "price_hvg_irreversibility"])
    assert features.loc[first, "price_hvg_irreversibility"] == blocks.loc[first, "price_return"]
    assert features.loc[first, "flow_hvg_irreversibility"] == blocks.loc[first, "flow_fraction"]
    first_threshold = first + visibility.GATE_MIN_OBSERVATIONS
    score = features["hvg_layer_score"].to_numpy(float)
    expected = np.quantile(score[first:first_threshold], visibility.GATE_QUANTILE)
    assert np.isclose(features.loc[first_threshold, "hvg_layer_threshold"], expected)


def test_policy_uses_direction_of_dominant_layer_and_strict_gate() -> None:
    features = pd.DataFrame(
        {
            "position": [0, 1, 2],
            "hvg_layer_log_ratio": [1.0, -1.0, 0.5],
            "hvg_layer_score": [1.0, 1.0, 0.5],
            "hvg_layer_threshold": [0.5, 0.5, 0.5],
            "price_return": [-1.0, 1.0, 1.0],
            "flow_fraction": [1.0, -1.0, -1.0],
        }
    )
    long_active, short_active = visibility.policy_masks(features, rows=3)
    assert np.flatnonzero(long_active).tolist() == [0, 1]
    assert not short_active.any()


def test_signal_at_boundary_enters_next_five_minute_open() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=8, freq="5min"))
    long_active = np.array([True, False, False, True, False, True, False, False])
    short_active = np.zeros(8, dtype=bool)
    positions = select_executable_positions(
        dates,
        long_active,
        short_active,
        start="2023-01-01",
        end="2023-01-01 00:35",
        hold_bars=2,
    )
    assert positions == [0]
    assert dates.iloc[positions[0] + 1] == pd.Timestamp("2023-01-01 00:05")


def test_support_only_cannot_open_outcomes_hash_or_write(monkeypatch, tmp_path: Path) -> None:
    market = pd.DataFrame({"low": [1.0], "high": [1.0]})
    dates = pd.Series([pd.Timestamp("2023-01-01")])
    blocks = pd.DataFrame({"position": [0], "source_time": [pd.Timestamp("2022-12-31 23:55")]})
    features = pd.DataFrame(
        {
            "position": [0],
            "hvg_layer_log_ratio": [1.0],
            "hvg_layer_score": [1.0],
            "hvg_layer_threshold": [0.5],
            "price_hvg_irreversibility": [1.0],
            "flow_hvg_irreversibility": [2.0],
            "price_ordinal_entropy_o3": [0.5],
            "price_hvg_threshold": [0.5],
            "flow_hvg_threshold": [0.5],
            "price_return": [1.0],
            "flow_fraction": [1.0],
            "price_realized_vol": [1.0],
            "mean_absolute_flow": [1.0],
            "price_trend": [1.0],
        }
    )
    crossmap = pd.DataFrame(
        {"crossmap_dominance": [0.0], "dominance_threshold": [0.5]}
    )
    monkeypatch.setattr(visibility, "load_pre2024", lambda *args: (market, dates))
    monkeypatch.setattr(visibility, "build_completed_blocks", lambda *args: blocks)
    monkeypatch.setattr(visibility, "build_visibility_features", lambda *args: features)
    monkeypatch.setattr(visibility, "build_crossmap_features", lambda *args: crossmap)
    monkeypatch.setattr(visibility, "finite_spearman", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(visibility, "event_jaccard", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        visibility,
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
    monkeypatch.setattr(visibility, "RESULT_PATH", tmp_path / "forbidden.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("support-only crossed the outcome boundary")

    monkeypatch.setattr(visibility, "_future_extreme", forbidden)
    monkeypatch.setattr(visibility, "simulate", forbidden)
    monkeypatch.setattr(visibility, "_frame_sha256", forbidden)
    output = visibility.run(support_only=True)
    assert output["preflight_passed"] is True
    assert output["outcomes_opened"] is False
    assert not visibility.RESULT_PATH.exists()
