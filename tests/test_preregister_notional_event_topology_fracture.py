from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from training import preregister_notional_event_topology_fracture as netf


def _frame() -> pd.DataFrame:
    rows = 14
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "signed_quote_notional": np.ones(rows),
            "signed_event_imbalance": np.full(rows, 0.10),
            "micro_log_return": np.full(rows, 0.001),
            "flow_coherence": np.full(rows, 0.10),
            "buy_sell_event_size_log_ratio": np.full(rows, 0.10),
            "interarrival_burstiness": np.full(rows, 0.10),
            "event_notional_hhi": np.full(rows, 0.01),
            "underlying_trades_per_agg_event": np.full(rows, 1.5),
            "agg_trade_count": np.full(rows, 100),
            "close": np.full(rows, 100.0),
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )
    # At t=4, many small sell events move price down immediately, while larger
    # buy notional is the capital side.
    frame.loc[4, [
        "signed_quote_notional",
        "signed_event_imbalance",
        "micro_log_return",
        "flow_coherence",
        "buy_sell_event_size_log_ratio",
        "interarrival_burstiness",
        "close",
    ]] = [10.0, -0.50, -0.01, 0.60, 1.20, 0.90, 99.0]
    # During the next two completed bars, capital flow, event breadth, and
    # price all reveal in the buy-notional direction.
    frame.loc[5, ["signed_quote_notional", "signed_event_imbalance", "close"]] = [
        5.0,
        0.20,
        99.5,
    ]
    frame.loc[6, ["signed_quote_notional", "signed_event_imbalance", "close"]] = [
        5.0,
        0.20,
        100.5,
    ]
    return frame


def _cfg() -> netf.Config:
    return replace(
        netf.Config(),
        tension_quantile=0.50,
        structure_quantile=0.50,
        baseline_bars=4,
        baseline_min_periods=1,
        minimum_agg_trade_count=1,
    )


def test_lagged_quantile_excludes_current_observation() -> None:
    values = pd.Series([1.0, 1.0, 100.0])
    baseline = netf._lagged_clean_quantile(
        values,
        pd.Series([True, True, True]),
        quantile=0.50,
        window=3,
        minimum=1,
    )
    assert baseline.iloc[2] == 1.0


def test_candidate_requires_positive_confirmation_and_hold() -> None:
    with np.testing.assert_raises_regex(ValueError, "must be positive"):
        netf.compute_netf(
            _frame(),
            netf.Candidate("invalid", confirmation_bars=0, hold_bars=2),
            _cfg(),
        )


def test_support_stopping_rule_selects_strictest_passing_quantile() -> None:
    trials = [
        {"tension_quantile": 0.85, "all_candidates_pass_support": True},
        {"tension_quantile": 0.875, "all_candidates_pass_support": True},
        {"tension_quantile": 0.90, "all_candidates_pass_support": False},
    ]
    assert netf._selected_support_quantile(trials) == 0.875


def test_capital_revelation_fires_only_after_confirmation() -> None:
    frame = _frame()
    candidate = netf.Candidate("test", confirmation_bars=2, hold_bars=2)
    signal = netf.compute_netf(frame, candidate, _cfg())
    assert signal.loc[4, "setup"]
    assert signal.loc[4, "side"] == 0
    assert signal.loc[6, "revealed"]
    assert signal.loc[6, "side"] == 1
    assert signal.loc[6, "hold_bars"] == 2


def test_confirmation_requires_price_to_reveal_toward_capital() -> None:
    frame = _frame()
    frame.loc[6, "close"] = 98.5
    candidate = netf.Candidate("test", confirmation_bars=2, hold_bars=2)
    signal = netf.compute_netf(frame, candidate, _cfg())
    assert signal.loc[4, "setup"]
    assert not signal.loc[6, "revealed"]
    assert signal.loc[6, "side"] == 0


def test_signal_prefix_is_invariant_to_future_changes() -> None:
    frame = _frame()
    candidate = netf.Candidate("test", confirmation_bars=2, hold_bars=2)
    baseline = netf.compute_netf(frame, candidate, _cfg())
    changed = frame.copy()
    changed.loc[7:, "signed_quote_notional"] = -1_000_000.0
    changed.loc[7:, "signed_event_imbalance"] = -1.0
    changed.loc[7:, "close"] = 1.0
    replay = netf.compute_netf(changed, candidate, _cfg())
    pd.testing.assert_frame_equal(baseline.loc[:6], replay.loc[:6])


def test_quarantine_inside_setup_confirmation_path_blocks_signal() -> None:
    frame = _frame()
    frame.loc[5, "quarantined"] = True
    candidate = netf.Candidate("test", confirmation_bars=2, hold_bars=2)
    signal = netf.compute_netf(frame, candidate, _cfg())
    assert signal.loc[4, "setup"]
    assert not signal.loc[6, "revealed"]


def test_split_schedule_requires_setup_origin_inside_period() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-06-30 23:45", periods=8, freq="5min"),
            "quarantined": np.zeros(8, dtype=bool),
        }
    )
    signal = pd.DataFrame(
        {
            "side": [0, 0, 0, 1, 0, 1, 0, 0],
            "hold_bars": [0, 0, 0, 1, 0, 1, 0, 0],
            "branch": [
                "none",
                "none",
                "none",
                "capital_revelation",
                "none",
                "capital_revelation",
                "none",
                "none",
            ],
            # Signal 3 confirms a setup from before the H2 boundary. Signal 5
            # confirms a setup at the first H2 bar and is therefore eligible.
            "origin_position": [-1, -1, -1, 1, -1, 3, -1, -1],
        }
    )
    schedule = netf.nonoverlapping_netf_schedule(
        signal,
        frame,
        start="2023-07-01",
        end="2024-01-01",
    )
    assert schedule["signal_position"].tolist() == [5]
