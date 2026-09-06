import numpy as np
import pandas as pd
import pytest

from training import build_quarter_hour_opening_imbalance_relay_support as support


def _clock_frame(times, imbalances, ranks=None):
    times = pd.to_datetime(times, utc=True)
    count = len(times)
    return pd.DataFrame(
        {
            "decision_time": times,
            "is_quarter_hour": [timestamp.minute % 15 == 0 for timestamp in times],
            "source_valid": [True] * count,
            "opening_imbalance": imbalances,
            "shifted_phase_plus_2m_valid": [True] * count,
            "shifted_phase_plus_2m_imbalance": imbalances,
            "prior_quarter_valid": [True] * count,
            "prior_quarter_imbalance": imbalances,
            "realized_variation": [0.1] * count,
            "variation_rank": ranks or [0.8] * count,
        }
    )


def test_source_panel_uses_current_opening_bar_but_strictly_prior_variation():
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    times = pd.date_range(start, periods=1446, freq="1min")
    bars = pd.DataFrame(
        {
            "ts": times,
            "open": np.full(len(times), 100.0),
            "close": np.full(len(times), 101.0),
            "volume": np.full(len(times), 10.0),
            "taker_buy_base": np.full(len(times), 7.0),
        }
    )
    boundary = start + pd.Timedelta(days=1)
    bars.loc[bars.ts.eq(boundary), "close"] = 200.0
    panel = support.build_source_panel(bars, rank_lookback=2, rank_minimum=1)
    row = panel.loc[panel.decision_time.eq(boundary)].iloc[0]
    expected = np.sqrt(1440 * np.log(101.0 / 100.0) ** 2)
    assert row.opening_imbalance == pytest.approx(0.4)
    assert row.realized_variation == pytest.approx(expected)
    assert row.variation_rank != row.variation_rank


def test_source_panel_rejects_any_minute_grid_failure():
    times = pd.date_range("2024-01-01", periods=1442, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {"ts": times, "open": 100.0, "close": 101.0, "volume": 10.0, "taker_buy_base": 5.0}
    ).drop(index=500)
    with pytest.raises(RuntimeError, match="exact requested one-minute grid"):
        support.build_source_panel(
            bars,
            start=times[0],
            end=times[-1] + pd.Timedelta(minutes=1),
            rank_minimum=1,
        )


def test_midrank_excludes_current_and_uses_midrank_for_ties():
    ranks = support.strict_prior_midrank(pd.Series([1.0, 2.0, 2.0, 3.0]), lookback=3, minimum=2)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == pytest.approx(0.75)
    assert ranks.iloc[3] == pytest.approx(1.0)


def test_global_half_open_reservation_exits_before_equal_entry():
    frame = _clock_frame(
        [
            "2024-07-01T00:00:00Z",
            "2024-07-01T00:15:00Z",
            "2024-07-01T08:00:00Z",
        ],
        [0.2, -0.3, -0.4],
    )
    candidate = support.clock(frame)
    assert candidate.entry_time.tolist() == [
        pd.Timestamp("2024-07-01T00:05:00Z"),
        pd.Timestamp("2024-07-01T08:05:00Z"),
    ]
    assert candidate.side.tolist() == [1, -1]
    assert candidate.iloc[0].exit_time == candidate.iloc[1].entry_time


def test_primary_gates_and_all_six_controls_are_frozen():
    frame = _clock_frame(
        [
            "2024-07-01T00:00:00Z",
            "2024-07-01T00:05:00Z",
            "2024-07-01T00:15:00Z",
            "2024-07-01T08:00:00Z",
        ],
        [0.2, -0.3, 0.0, -0.4],
        [0.8, 0.8, 0.8, 0.64],
    )
    frame["shifted_phase_plus_2m_imbalance"] = [-0.2, 0.3, 0.5, 0.4]
    frame["prior_quarter_imbalance"] = [-0.1, -0.1, 0.6, 0.7]

    primary, primary_side = support.conditions(frame)
    assert primary.tolist() == [True, False, False, False]
    assert primary_side[primary].tolist() == [1]
    assert support.conditions(frame, "shifted_phase_plus_2m")[1].iloc[0] == -1
    assert support.conditions(frame, "five_minute_phase_only")[0].tolist() == [False, True, False, False]
    assert support.conditions(frame, "no_volatility_gate")[0].tolist() == [True, False, False, True]
    assert support.conditions(frame, "one_quarter_stale_imbalance")[0].tolist() == [True, False, True, False]
    assert support.conditions(frame, "direction_flip")[1].iloc[0] == -1
    assert support.conditions(frame, "exclude_funding_boundaries")[0].tolist() == [False, False, False, False]


def test_support_gates_match_preregistration_exactly():
    passing = {
        split: {
            "events": support.MINIMUM[split],
            "longs": 1,
            "shorts": 1,
            "minority_side_share": 0.2,
            "max_month_share": 0.45,
        }
        for split in support.SPLITS
    }
    assert all(support.support_checks(passing).values())
    passing["train"]["events"] = 7
    checks = support.support_checks(passing)
    assert checks["train_minimum_events"] is False


def test_preregistration_hash_is_pinned() -> None:
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA256
