from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import search_positioning_simplex_curvature_alpha as simplex


def _state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision": [True, True, True, False],
            "migration": [1.0, -1.0, 1.0, 1.0],
            "oi_curvature_speed": [2.0, 3.0, 0.5, 9.0],
        }
    )


def test_simplex_curvature_is_zero_for_collinear_migration() -> None:
    previous = np.array([[1.0, 2.0, 3.0]])
    current = np.array([[2.0, 4.0, 6.0]])
    assert np.isclose(simplex.simplex_curvature(previous, current)[0], 0.0)


def test_simplex_curvature_preserves_turn_orientation() -> None:
    previous = np.array([[1.0, 0.0, 0.0]])
    current = np.array([[0.0, 1.0, 0.0]])
    forward = simplex.simplex_curvature(previous, current)[0]
    reverse = simplex.simplex_curvature(current, previous)[0]
    assert forward > 0.0
    assert np.isclose(forward, -reverse)


def test_curvature_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        simplex.simplex_curvature(np.ones((3, 2)), np.ones((3, 2)))


def test_fade_and_continuation_are_exact_opposites() -> None:
    fade_long, fade_short = simplex.policy_masks(
        _state(), "oi_curvature_speed", 1.0, mapping="fade"
    )
    follow_long, follow_short = simplex.policy_masks(
        _state(), "oi_curvature_speed", 1.0, mapping="continuation"
    )
    np.testing.assert_array_equal(fade_long, follow_short)
    np.testing.assert_array_equal(fade_short, follow_long)
    assert np.flatnonzero(fade_long).tolist() == [1]
    assert np.flatnonzero(fade_short).tolist() == [0]


def test_fit_threshold_does_not_use_selection_values(monkeypatch) -> None:
    monkeypatch.setitem(simplex.WINDOWS, "fit", ("2021-01-01", "2021-01-05"))
    dates = pd.Series(pd.date_range("2021-01-01", periods=1400, freq="5min"))
    values = np.arange(1400, dtype=float)
    state = pd.DataFrame({"score": values})
    first = simplex.fit_threshold(state, dates, "score")
    state.loc[dates >= pd.Timestamp("2021-01-05"), "score"] = 1e12
    second = simplex.fit_threshold(state, dates, "score")
    assert np.isclose(first, second)


def test_support_counts_match_nonoverlap_and_side_counts(monkeypatch) -> None:
    monkeypatch.setitem(
        simplex.WINDOWS, "sample", ("2023-01-01", "2023-01-01 00:40")
    )
    monkeypatch.setattr(simplex, "HOLD_BARS", 2)
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    long_active = np.array(
        [True, True, False, False, False, False, True, False, False, False]
    )
    short_active = np.array(
        [False, False, False, False, True, False, False, False, False, False]
    )
    assert simplex.support_counts(
        dates, long_active, short_active, window="sample"
    ) == {
        "raw": 4,
        "raw_long": 3,
        "raw_short": 1,
        "strict_executable": 2,
        "strict_executable_long": 1,
        "strict_executable_short": 1,
    }


def test_future_suffix_cannot_change_simplex_prefix() -> None:
    periods = 36 * 12
    dates = pd.Series(pd.date_range("2023-01-01", periods=periods, freq="5min"))
    base = np.linspace(0.9, 1.1, periods)
    market = pd.DataFrame(
        {
            "sum_toptrader_long_short_ratio": base,
            "count_long_short_ratio": base[::-1],
            "sum_taker_long_short_vol_ratio": 1.0 + 0.05 * np.sin(np.arange(periods)),
            "sum_open_interest": np.linspace(1000.0, 1100.0, periods),
        }
    )
    first = simplex.build_simplex_state(market, dates)
    changed = market.copy()
    changed.loc[periods // 2 :, list(simplex.RATIO_COLUMNS)] *= 1000.0
    second = simplex.build_simplex_state(changed, dates)
    columns = ["migration", "curvature", "oi_curvature_speed"]
    np.testing.assert_allclose(
        first.loc[: periods // 2 - 1, columns],
        second.loc[: periods // 2 - 1, columns],
        equal_nan=True,
    )


def test_support_only_cannot_simulate_or_write(monkeypatch, tmp_path) -> None:
    market = pd.DataFrame({"low": [1.0], "high": [1.0]})
    dates = pd.Series([pd.Timestamp("2023-01-01")])
    state = pd.DataFrame(
        {
            "decision": [True],
            "migration": [1.0],
            **{name: [1.0] for name in simplex.SCORE_VARIANTS},
        }
    )
    monkeypatch.setattr(simplex, "load_pre2024", lambda: (market, dates))
    monkeypatch.setattr(simplex, "build_simplex_state", lambda *args: state)
    monkeypatch.setattr(simplex, "fit_threshold", lambda *args: 0.5)
    monkeypatch.setattr(simplex, "RESULT_PATH", tmp_path / "forbidden.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("support-only crossed outcome boundary")

    monkeypatch.setattr(simplex, "_future_extreme", forbidden)
    monkeypatch.setattr(simplex, "simulate", forbidden)
    output = simplex.run(support_only=True)
    assert output["support_only"] is True
    assert not simplex.RESULT_PATH.exists()


def test_loader_is_pre2024_and_source_delayed() -> None:
    market, dates = simplex.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    source = pd.to_datetime(market["positioning_source_time"], errors="coerce")
    valid = source.notna()
    assert (source[valid] <= dates[valid] - pd.Timedelta("5min")).all()
