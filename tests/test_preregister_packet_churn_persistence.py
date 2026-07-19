from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from training.preregister_minute_packet_topology_alpha import Candidate
from training.preregister_packet_churn_persistence import (
    CONFIRMATION_BARS,
    ENTRY_DELAY_BARS,
    HOLD_BARS,
    Config,
    _raw_return_bp,
    build_schedule,
    support_gates,
    support_summary,
    write_artifacts,
)


def _frame() -> pd.DataFrame:
    rows = 160
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "minute_dispersion_feature_valid": True,
            "um_net_flow_fraction": 0.10,
            "spot_net_flow_fraction": 0.10,
            "um_signed_impact_bp": 1.0,
            "spot_signed_impact_bp": 1.0,
            "um_flow_sign_switch_rate": 0.75,
        }
    )
    return frame


def test_raw_return_recovers_completed_bar_direction() -> None:
    frame = pd.DataFrame(
        {
            "um_net_flow_fraction": [0.2, -0.3, 0.0],
            "um_signed_impact_bp": [4.0, -5.0, 7.0],
        }
    )
    recovered = _raw_return_bp(frame, "um")
    assert recovered.iloc[:2].tolist() == [4.0, 5.0]
    assert np.isnan(recovered.iloc[2])


def test_confirmation_is_complete_before_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()
    onset = pd.Series(False, index=frame.index)
    onset.iloc[10] = True
    side = pd.Series(1, index=frame.index, dtype=np.int8)

    monkeypatch.setattr(
        "training.preregister_packet_churn_persistence.candidate_clock",
        lambda *_args, **_kwargs: (onset, side),
    )
    schedule, raw = build_schedule(
        frame,
        {},
        Candidate("cross_venue_churn_breakout", HOLD_BARS, 0.70, 0.35),
    )
    assert raw == 1
    row = schedule.iloc[0]
    assert row.confirmation_end_position == 10 + CONFIRMATION_BARS
    assert row.entry_position == row.confirmation_end_position + ENTRY_DELAY_BARS
    assert row.exit_position == row.entry_position + HOLD_BARS
    assert pd.Timestamp(row.signal_available_at) == pd.Timestamp(
        row.confirmation_end_bar_date
    ) + pd.Timedelta("5min")
    assert pd.Timestamp(row.entry_date) == pd.Timestamp(
        row.signal_available_at
    ) + pd.Timedelta("5min")


def test_failed_confirmation_does_not_reserve_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame.loc[11:16, "um_flow_sign_switch_rate"] = 0.0
    onset = pd.Series(False, index=frame.index)
    onset.iloc[[10, 20]] = True
    side = pd.Series(1, index=frame.index, dtype=np.int8)
    monkeypatch.setattr(
        "training.preregister_packet_churn_persistence.candidate_clock",
        lambda *_args, **_kwargs: (onset, side),
    )
    schedule, raw = build_schedule(
        frame,
        {},
        Candidate("cross_venue_churn_breakout", HOLD_BARS, 0.70, 0.35),
    )
    assert raw == 1
    assert schedule["setup_position"].tolist() == [20]


def test_zero_flow_confirmation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()
    frame.loc[12, "spot_net_flow_fraction"] = 0.0
    onset = pd.Series(False, index=frame.index)
    onset.iloc[10] = True
    side = pd.Series(1, index=frame.index, dtype=np.int8)
    monkeypatch.setattr(
        "training.preregister_packet_churn_persistence.candidate_clock",
        lambda *_args, **_kwargs: (onset, side),
    )
    schedule, raw = build_schedule(
        frame,
        {},
        Candidate("cross_venue_churn_breakout", HOLD_BARS, 0.70, 0.35),
    )
    assert raw == 0
    assert schedule.empty


def test_support_gate_requires_time_and_side_dispersion() -> None:
    rows = []
    for year in range(2020, 2024):
        for month in range(1, 11):
            for side in (-1, 1):
                rows.extend(
                    {
                        "entry_date": f"{year}-{month:02d}-{day:02d}",
                        "side": side,
                    }
                    for day in (1, 8, 15)
                )
    schedule = pd.DataFrame(rows)
    summary = support_summary(schedule)
    assert all(support_gates(summary).values())
    schedule["side"] = 1
    assert not support_gates(support_summary(schedule))[
        "each_train_side_at_least_25pct"
    ]


def test_2023_incidence_does_not_change_support_gate() -> None:
    rows = []
    for year in range(2020, 2024):
        for month in range(1, 11):
            for side in (-1, 1):
                rows.extend(
                    {
                        "entry_date": f"{year}-{month:02d}-{day:02d}",
                        "side": side,
                    }
                    for day in (1, 8, 15)
                )
    schedule = pd.DataFrame(rows)
    train_only = cast(
        pd.DataFrame,
        schedule.loc[pd.to_datetime(schedule["entry_date"]).dt.year < 2023].copy(),
    )
    assert support_gates(support_summary(schedule)) == support_gates(
        support_summary(train_only)
    )


def test_artifacts_are_write_once(tmp_path: Path) -> None:
    cfg = Config(
        input_csv="unused",
        output=str(tmp_path / "support.json"),
        clock_output=str(tmp_path / "clock.csv"),
    )
    report = {"clock": {}, "created_at": "fixed", "result_hash": "old"}
    clock = pd.DataFrame({"entry_date": ["2023-01-01"], "side": [1]})
    write_artifacts(cfg, report, clock)
    assert Path(cfg.output).exists()
    assert Path(cfg.clock_output).exists()
    with pytest.raises(FileExistsError):
        write_artifacts(cfg, report, clock)
