from __future__ import annotations

import numpy as np
import pandas as pd

from training import search_orderflow_campaign_terminal_absorption_alpha as terminal


def test_terminal_scores_are_side_symmetric() -> None:
    features = pd.DataFrame(
        {
            "a_imbalance_z": [2.0],
            "a_return_z": [0.5],
            "a_impact_z": [-1.0],
            "a_clv": [0.25],
        }
    )
    long_score, short_score = terminal.terminal_absorption_scores(features)
    assert np.isclose(long_score[0], 2.25)
    assert np.isclose(short_score[0], -0.25)


def test_terminal_signal_waits_then_fades_campaign_side() -> None:
    campaign_long = np.array([False, True, False, False, False])
    campaign_short = np.zeros(5, dtype=bool)
    long_score = np.array([0.0, 5.0, 0.0, 2.1, 0.0])
    short_score = np.zeros(5)

    output_long, output_short, diagnostics = terminal.terminal_absorption_signals(
        campaign_long,
        campaign_short,
        long_score,
        short_score,
        threshold=2.0,
        max_wait_bars=3,
    )

    assert not output_long.any()
    np.testing.assert_array_equal(np.flatnonzero(output_short), [3])
    assert diagnostics["signal_age"][3] == 2
    assert diagnostics["started_campaigns"] == 1


def test_campaign_expires_before_late_absorption() -> None:
    campaign_long = np.array([True, False, False, False, False])
    score = np.array([0.0, 0.0, 0.0, 3.0, 3.0])
    output_long, output_short, diagnostics = terminal.terminal_absorption_signals(
        campaign_long,
        np.zeros(5, dtype=bool),
        score,
        np.zeros(5),
        threshold=2.0,
        max_wait_bars=2,
    )
    assert not output_long.any()
    assert not output_short.any()
    assert diagnostics["expired_campaigns"] == 1


def test_new_campaign_at_pending_endpoint_is_ignored() -> None:
    campaign_long = np.array([True, False, True, False, False, False])
    long_score = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 3.0])
    _, output_short, diagnostics = terminal.terminal_absorption_signals(
        campaign_long,
        np.zeros(6, dtype=bool),
        long_score,
        np.zeros(6),
        threshold=2.0,
        max_wait_bars=2,
    )
    assert not output_short.any()
    assert diagnostics["started_campaigns"] == 1
    assert diagnostics["expired_campaigns"] == 1


def test_terminal_prefix_is_future_suffix_independent_and_flip_is_exact() -> None:
    campaign_long = np.array([False, True, False, False, False, False])
    campaign_short = np.zeros(6, dtype=bool)
    long_score = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    short_score = np.zeros(6)
    kwargs = {"threshold": 2.0, "max_wait_bars": 4}
    expected_long, expected_short, _ = terminal.terminal_absorption_signals(
        campaign_long, campaign_short, long_score, short_score, **kwargs
    )
    suffix = np.array([False, True, False])
    actual_long, actual_short, _ = terminal.terminal_absorption_signals(
        np.r_[campaign_long, suffix],
        np.r_[campaign_short, np.zeros(3, dtype=bool)],
        np.r_[long_score, [100.0, 100.0, 100.0]],
        np.r_[short_score, [100.0, 100.0, 100.0]],
        **kwargs,
    )
    flip_long, flip_short, _ = terminal.terminal_absorption_signals(
        campaign_long, campaign_short, long_score, short_score, flip=True, **kwargs
    )
    np.testing.assert_array_equal(actual_long[:6], expected_long)
    np.testing.assert_array_equal(actual_short[:6], expected_short)
    np.testing.assert_array_equal(flip_long, expected_short)
    np.testing.assert_array_equal(flip_short, expected_long)


def test_standalone_control_uses_onsets_and_global_cooldown() -> None:
    long_score = np.array([0.0, 3.0, 3.0, 0.0, 4.0, 0.0, 0.0])
    short_score = np.zeros(7)
    output_long, output_short = terminal.standalone_absorption_signals(
        long_score,
        short_score,
        threshold=2.0,
        cooldown_bars=4,
    )
    assert not output_long.any()
    np.testing.assert_array_equal(np.flatnonzero(output_short), [1])


def test_lag_has_no_wraparound() -> None:
    values = np.array([True, False, True, False])
    assert terminal.lag_boolean(values, 2).tolist() == [False, False, True, False]


def test_support_counts_match_nonoverlap_and_split_containment(monkeypatch) -> None:
    monkeypatch.setitem(terminal.WINDOWS, "sample", ("2023-01-01", "2023-01-01 00:40"))
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    active = np.array([True, True, False, False, True, False, True, False, False, False])
    counts = terminal.support_counts(
        dates,
        active,
        np.zeros(10, dtype=bool),
        window="sample",
        hold_bars=2,
    )
    assert counts == {"raw": 4, "strict_executable": 2}


def test_loader_keeps_2024_sealed_and_grid_complete() -> None:
    _, dates = terminal.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()
