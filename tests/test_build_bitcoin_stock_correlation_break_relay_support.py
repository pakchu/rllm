import numpy as np
import pandas as pd
import pytest

from training import build_bitcoin_stock_correlation_break_relay_support as support


def test_softmax_parameters_are_stationary() -> None:
    first, second = support._softmax_pair(20.0, 20.0)
    assert first >= 0.0
    assert second >= 0.0
    assert first + second < 0.999


def test_garch_filter_uses_current_residual_only_in_post_close_state() -> None:
    pre, post = support.garch_filter(
        np.asarray([2.0, 3.0]),
        (1.0, 0.25, 0.50),
        4.0,
    )
    assert pre.tolist() == pytest.approx([4.0, 4.0])
    assert post.tolist() == pytest.approx([4.0, 5.25])


def test_dcc_filter_reads_post_close_update_without_lookahead() -> None:
    standardized = np.asarray([[1.0, -1.0], [1.0, 1.0]])
    qbar = np.eye(2)
    pre, post = support.dcc_filter(standardized, qbar, 0.10, 0.80)
    assert pre[0] == pytest.approx(0.0)
    assert post[0] < 0.0
    assert pre[1] == pytest.approx(post[0])
    assert post[1] > post[0]


def test_session_schedule_handles_dst_and_frozen_early_close() -> None:
    closes = [
        support.cash_close_time(pd.Timestamp(date))[0]
        for date in ("2023-07-03", "2023-11-24", "2023-11-27")
    ]
    assert pd.DatetimeIndex(closes).strftime("%Y-%m-%dT%H:%MZ").tolist() == [
        "2023-07-03T17:00Z",
        "2023-11-24T18:00Z",
        "2023-11-27T21:00Z",
    ]


def test_session_schedule_rejects_missing_official_session() -> None:
    spy = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04"])})
    with pytest.raises(RuntimeError, match="calendar mismatch"):
        support.session_schedule(spy)


def test_paired_returns_adds_dividend_and_preserves_elapsed_gap() -> None:
    spy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-06", "2023-01-09"]),
            "close": [100.0, 101.0],
            "cash_dividend": [0.0, 1.0],
            "history_valid": [True, True],
        }
    )
    schedule = pd.DataFrame(
        {
            "session_date": spy["date"],
            "cash_close_time": pd.to_datetime(
                ["2023-01-06T21:00:00Z", "2023-01-09T21:00:00Z"]
            ),
            "close_local_time": ["16:00", "16:00"],
            "early_close": [False, False],
        }
    )
    btc = pd.DataFrame(
        {
            "decision_time": schedule["cash_close_time"],
            "hour_close": [20000.0, 22000.0],
            "source_valid": [True, True],
        }
    )
    paired = support.paired_returns(spy, schedule, btc)
    assert paired.loc[1, "spy_return"] == pytest.approx(np.log(1.02))
    assert paired.loc[1, "btc_return"] == pytest.approx(np.log(1.10))
    assert paired.loc[1, "elapsed_gap_hours"] == 72.0
    assert paired.loc[1, "pair_valid"]


def test_strict_prior_midrank_excludes_current_value() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    ranks = support.strict_prior_midrank(values, window=3, minimum=3)
    assert ranks.iloc[:3].isna().all()
    assert ranks.iloc[3] == 1.0


def test_clock_enforces_volatility_gate_direction_and_global_nonoverlap() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "cash_close_time": pd.to_datetime(
                ["2024-01-02T21:00Z", "2024-01-03T21:00Z", "2024-01-04T21:00Z"]
            ),
            "rho_post": [0.0, -0.03, 0.01],
            "delta_rho": [-0.03, -0.03, 0.04],
            "btc_sigma_post": [0.02, 0.03, 0.04],
            "btc_sigma_prior_midrank": [0.70, 0.80, 0.90],
            "elapsed_gap_hours": [24.0, 24.0, 24.0],
            "rolling_delta_rho": [-0.03, -0.03, 0.04],
            "spy_z": [1.0, 1.0, -1.0],
            "btc_z": [-1.0, -1.0, -1.0],
        }
    )
    clock = support.build_clock(frame)
    assert len(clock) == 3
    assert clock["side"].tolist() == [1, 1, -1]
    assert clock["entry_time"].dt.strftime("%H:%M").tolist() == ["21:10"] * 3


def test_support_stats_fail_closed_for_empty_split() -> None:
    empty = pd.DataFrame(columns=support.CLOCK_COLUMNS)
    assert support.support_stats(empty, "train") == {
        "events": 0,
        "longs": 0,
        "shorts": 0,
        "minority_side_share": 0.0,
        "max_month_share": 0.0,
    }


def test_optimizer_boundary_drift_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="boundary drift"):
        support.reject_optimizer_boundary(np.asarray([0.0, 20.0]), "test")
