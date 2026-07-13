from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_market_braid_alpha import (
    WINDOWS,
    admission,
    build_bar_state,
    build_signals,
    load_pre2024,
    market_braid_events,
)


def synthetic_state(size: int = 12) -> dict[str, np.ndarray]:
    state = {
        "spot": np.zeros(size),
        "perp": np.zeros(size),
        "log_oi": np.zeros(size),
        "premium": np.zeros(size),
        "valid": np.ones(size, dtype=bool),
        "shock_z": np.zeros(size),
        "spot_unit": np.ones(size),
        "perp_unit": np.ones(size),
        "oi_unit": np.ones(size),
        "premium_unit": np.ones(size),
    }
    state["shock_z"][1] = 3.0
    return state


def test_organic_strict_chain_continues_after_completion() -> None:
    state = synthetic_state()
    state["spot"][2:] = 1.0
    state["perp"][3:] = 1.0
    state["log_oi"][4:] = 1.0
    state["premium"][4:] = 1.0
    events = market_braid_events(
        state, shock_z=2.0, passage_z=0.5, max_age=8, topology_mode="strict_chain"
    )
    assert np.flatnonzero(events["episode_age"].to_numpy(int) > 0).tolist() == [4]
    assert events.loc[4, "sequence"] == "spot>perp>leverage"
    assert events.loc[4, "signal_side"] == 1


def test_synthetic_strict_chain_fades_after_completion() -> None:
    state = synthetic_state()
    state["log_oi"][2:] = 1.0
    state["premium"][2:] = 1.0
    state["perp"][3:] = 1.0
    state["spot"][4:] = 1.0
    events = market_braid_events(
        state, shock_z=2.0, passage_z=0.5, max_age=8, topology_mode="strict_chain"
    )
    assert events.loc[4, "sequence"] == "leverage>perp>spot"
    assert events.loc[4, "signal_side"] == -1


def test_same_bar_passages_are_discarded_without_guessing_order() -> None:
    state = synthetic_state()
    state["spot"][2:] = 1.0
    state["perp"][2:] = 1.0
    state["log_oi"][3:] = 1.0
    state["premium"][3:] = 1.0
    events = market_braid_events(
        state, shock_z=2.0, passage_z=0.5, max_age=8, topology_mode="relative_order"
    )
    assert bool(events.loc[2, "tie_discarded"])
    assert int((events["episode_age"] > 0).sum()) == 0


def test_no_leverage_control_completes_only_two_strands() -> None:
    state = synthetic_state()
    state["spot"][2:] = 1.0
    state["perp"][3:] = 1.0
    events = market_braid_events(
        state,
        shock_z=2.0,
        passage_z=0.5,
        max_age=8,
        topology_mode="strict_chain",
        leverage_mode="none",
    )
    assert events.loc[3, "sequence"] == "spot>perp"
    assert events.loc[3, "signal_side"] == 1


def test_event_prefix_is_suffix_independent() -> None:
    state = synthetic_state()
    state["spot"][2:] = 1.0
    state["perp"][3:] = 1.0
    state["log_oi"][4:] = 1.0
    state["premium"][4:] = 1.0
    extended = {name: np.r_[value, value[-5:]] for name, value in state.items()}
    left = market_braid_events(
        state, shock_z=2.0, passage_z=0.5, max_age=8, topology_mode="relative_order"
    )
    right = market_braid_events(
        extended, shock_z=2.0, passage_z=0.5, max_age=8, topology_mode="relative_order"
    ).iloc[: len(left)]
    pd.testing.assert_frame_equal(left, right.reset_index(drop=True))


def test_signal_flip_and_order_blind_control() -> None:
    events = pd.DataFrame(
        {
            "signal_side": [0, 1, -1, 0],
            "impulse_side": [0, -1, -1, 1],
            "episode_age": [0, 2, 3, 0],
        }
    )
    long_signal, short_signal = build_signals(events)
    assert long_signal.tolist() == [False, True, False, False]
    assert short_signal.tolist() == [False, False, True, False]
    flip_long, flip_short = build_signals(events, flip=True)
    assert np.array_equal(flip_long, short_signal)
    assert np.array_equal(flip_short, long_signal)
    blind_long, blind_short = build_signals(events, order_blind=True)
    assert not blind_long.any()
    assert blind_short.tolist() == [False, True, True, False]


def test_build_bar_state_delays_oi_one_complete_bar_and_is_prefix_causal() -> None:
    size = 1300
    index = np.arange(size, dtype=float)
    frame = pd.DataFrame(
        {
            "spot_close": 20_000.0 * np.exp(0.0001 * index + 0.001 * np.sin(index / 17.0)),
            "close": 20_001.0 * np.exp(0.0001 * index + 0.001 * np.sin(index / 19.0)),
            "premium_index_1m_close": 0.0001 * np.sin(index / 23.0),
            "open_interest": 1_000.0 + index,
            "spot_rows": np.full(size, 5.0),
            "premium_rows": np.full(size, 5.0),
            "open_interest_available": np.ones(size),
        }
    )
    left = build_bar_state(frame)
    assert np.isnan(left["log_oi"][0])
    assert left["log_oi"][1] == np.log(frame.loc[0, "open_interest"])
    extension = frame.iloc[-20:].copy()
    extension["open_interest"] = 1_000_000.0
    right = build_bar_state(pd.concat([frame, extension], ignore_index=True))
    for name in left:
        np.testing.assert_allclose(left[name], right[name][:size], equal_nan=True)


def test_real_loader_physically_excludes_2024() -> None:
    _, dates = load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()


def test_admission_requires_fit_and_selection_ratio_three() -> None:
    def row(ratio: float, trades: int = 50) -> dict[str, float | int]:
        return {"return_pct": 1.0, "ratio": ratio, "trades": trades}

    stats = {name: row(3.1) for name in WINDOWS}
    assert admission(stats)
    stats["select_2023"] = row(2.99)
    assert not admission(stats)
