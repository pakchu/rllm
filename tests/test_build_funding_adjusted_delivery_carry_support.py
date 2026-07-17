from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.build_funding_adjusted_delivery_carry_support import (
    Config,
    aggregate_perpetual_signal,
    build_events,
    event_counts,
    support_passes,
)


def test_perpetual_aggregation_fails_closed_on_missing_minute(tmp_path: Path) -> None:
    times = pd.date_range("2023-01-01", periods=10, freq="1min", tz="UTC")
    frame = pd.DataFrame({"date": times.delete(3), "close": range(9)})
    path = tmp_path / "perp.csv"
    frame.to_csv(path, index=False)
    five, diagnostics = aggregate_perpetual_signal(str(path))
    assert diagnostics["incomplete_five_minute_rows"] == 1
    assert five["perp_signal_complete"].tolist() == [False, True]


def _event_frame() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2022-01-01 00:00Z",
            "2022-01-02 00:00Z",
            "2022-01-03 00:00Z",
            "2022-01-04 00:00Z",
        ]
    )
    delivery = pd.Timestamp("2022-03-25 08:00Z")
    return pd.DataFrame(
        {
            "open_time": times,
            "signal_time": times + pd.Timedelta(minutes=5),
            "entry_time": times + pd.Timedelta(minutes=10),
            "mandatory_exit_time": [delivery - pd.Timedelta(days=1)] * 4,
            "delivery_time": [delivery] * 4,
            "contract_segment": ["20220325"] * 4,
            "dte_days": [83.0, 82.0, 81.0, 80.0],
            "annualized_basis": [0.10, 0.10, 0.10, 0.10],
            "annualized_funding": [0.02, 0.02, 0.02, 0.12],
            "carry_gap": [0.08, 0.08, 0.08, -0.02],
            "edge_to_scheduled_exit": [0.01, 0.01, 0.01, 0.002],
            "eligible": [True, True, True, False],
        }
    )


def test_event_state_uses_next_open_and_causal_sign_flip_exit() -> None:
    cfg = Config(minimum_hold_hours=24, cooldown_hours=24)
    events = build_events(_event_frame(), cfg)
    assert len(events) == 1
    event = events.iloc[0]
    assert event["entry_time"] == pd.Timestamp("2022-01-01 00:10Z")
    assert event["exit_time"] == pd.Timestamp("2022-01-04 00:10Z")
    assert event["perpetual_side"] == 1
    assert event["quarterly_side"] == -1
    assert event["exit_reason"] == "normalization"


def test_support_gate_requires_both_directions() -> None:
    train = {
        "events": 24,
        "by_year": {"2021": 12, "2022": 12},
        "by_half": {"2021H1": 6, "2021H2": 6, "2022H1": 6, "2022H2": 6},
        "by_direction": {"perp_long_quarter_short": 24},
        "maximum_month_share": 0.10,
    }
    holdout = {
        "events": 8,
        "by_half": {"2023H1": 4, "2023H2": 4},
        "by_direction": {
            "perp_long_quarter_short": 4,
            "perp_short_quarter_long": 4,
        },
        "maximum_month_share": 0.25,
    }
    passed, failures = support_passes(train, holdout)
    assert passed is False
    assert "pre2023_missing_carry_direction" in failures


def test_event_counts_reports_wall_clock_activity() -> None:
    events = build_events(_event_frame(), Config())
    counts = event_counts(events)
    assert counts["events"] == 1
    assert counts["active_days"] == 3.0
    assert counts["by_direction"] == {"perp_long_quarter_short": 1}
