from __future__ import annotations

import numpy as np

from training.search_orderflow_trophic_campaign_alpha import campaign_signals


def _events(length: int, long_at: tuple[int, ...], short_at: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    long_events = np.zeros(length, dtype=bool)
    short_events = np.zeros(length, dtype=bool)
    long_events[list(long_at)] = True
    short_events[list(short_at)] = True
    return long_events, short_events


def test_campaign_confirms_only_on_kth_current_same_direction_event() -> None:
    long_events, short_events = _events(12, (1, 3, 4, 9, 10))

    confirmed_long, confirmed_short, diagnostics = campaign_signals(
        long_events,
        short_events,
        lookback_bars=5,
        min_same_events=2,
        max_opposite_events=0,
        cooldown_bars=5,
    )

    np.testing.assert_array_equal(np.flatnonzero(confirmed_long), [3, 10])
    assert not confirmed_short.any()
    assert diagnostics["long_count"][3] == 2
    assert diagnostics["long_count"][10] == 2


def test_opposite_event_veto_is_causal_and_explicit() -> None:
    long_events, short_events = _events(8, (1, 3), (2,))

    vetoed, _, _ = campaign_signals(
        long_events,
        short_events,
        lookback_bars=5,
        min_same_events=2,
        max_opposite_events=0,
    )
    allowed, _, _ = campaign_signals(
        long_events,
        short_events,
        lookback_bars=5,
        min_same_events=2,
        max_opposite_events=1,
    )

    assert not vetoed.any()
    np.testing.assert_array_equal(np.flatnonzero(allowed), [3])


def test_global_cooldown_consumes_repeated_confirmations() -> None:
    long_events, short_events = _events(10, (1, 2, 3, 4, 8, 9))

    confirmed_long, _, _ = campaign_signals(
        long_events,
        short_events,
        lookback_bars=5,
        min_same_events=2,
        max_opposite_events=0,
        cooldown_bars=5,
    )

    np.testing.assert_array_equal(np.flatnonzero(confirmed_long), [2, 8])


def test_direction_flip_is_exact_and_prefix_is_suffix_independent() -> None:
    prefix_long, prefix_short = _events(12, (1, 3, 9, 10), (5, 6))
    suffix_long, suffix_short = _events(5, (0, 1, 2), (3, 4))
    full_long = np.r_[prefix_long, suffix_long]
    full_short = np.r_[prefix_short, suffix_short]
    kwargs = {"lookback_bars": 5, "min_same_events": 2, "max_opposite_events": 1}

    expected_long, expected_short, _ = campaign_signals(prefix_long, prefix_short, **kwargs)
    actual_long, actual_short, _ = campaign_signals(full_long, full_short, **kwargs)
    flip_long, flip_short, _ = campaign_signals(prefix_long, prefix_short, flip=True, **kwargs)

    np.testing.assert_array_equal(actual_long[: len(prefix_long)], expected_long)
    np.testing.assert_array_equal(actual_short[: len(prefix_short)], expected_short)
    np.testing.assert_array_equal(flip_long, expected_short)
    np.testing.assert_array_equal(flip_short, expected_long)


def test_campaign_rejects_ambiguous_dual_direction_event() -> None:
    long_events, short_events = _events(5, (2,), (2,))

    with np.testing.assert_raises(ValueError):
        campaign_signals(long_events, short_events, lookback_bars=5, min_same_events=2)
