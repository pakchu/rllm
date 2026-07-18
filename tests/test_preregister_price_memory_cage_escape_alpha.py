from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import preregister_price_memory_cage_escape_alpha as cage


def test_volume_clock_uses_strictly_prior_target_and_is_prefix_independent() -> None:
    size = 700
    market = pd.DataFrame(
        {
            "quote_asset_volume": np.linspace(100.0, 300.0, size),
            "taker_buy_quote": np.linspace(40.0, 190.0, size),
        }
    )
    left = cage.volume_clock_flow_speed(market)
    changed = market.copy()
    changed.loc[600:, "quote_asset_volume"] *= 100.0
    changed.loc[600:, "taker_buy_quote"] = 0.0
    right = cage.volume_clock_flow_speed(changed)
    assert np.allclose(left[:600], right[:600], equal_nan=True)
    assert np.isnan(left[:288]).all()
    assert np.isfinite(left[288])


def test_combined_clock_requires_three_way_direction_agreement() -> None:
    dates = pd.Series(pd.date_range("2023-01-01 00:55", periods=6, freq="5min"))
    occupation = pd.DataFrame(
        {
            "side": [1, 0, 0, 0, -1, 0],
            "barrier_depth": [0.8, np.nan, np.nan, np.nan, 0.7, np.nan],
            "saddle_count": [2, 0, 0, 0, 2, 0],
        }
    )
    persistence = pd.DataFrame(
        {
            "side": [0, 1, 0, 0, 0, 1],
            "barrier_count": [0, 3, 0, 0, 0, 4],
        }
    )
    flow = np.array([np.nan, 0.01, np.nan, np.nan, np.nan, -0.01])
    active, side = cage.combine_event_clock(occupation, persistence, flow, dates)
    assert active.tolist() == [False, True, False, False, False, False]
    assert side.tolist() == [0, 1, 0, 0, 0, 0]


def test_combined_clock_uses_previous_minute55_occupation_row() -> None:
    dates = pd.Series(pd.date_range("2023-01-01 00:55", periods=3, freq="5min"))
    occupation = pd.DataFrame(
        {
            "side": [1, -1, 0],
            "barrier_depth": [0.5, 0.6, np.nan],
            "saddle_count": [2, 2, 0],
        }
    )
    persistence = pd.DataFrame(
        {"side": [0, 1, 0], "barrier_count": [0, 3, 0]}
    )
    active, side = cage.combine_event_clock(
        occupation, persistence, np.array([0.0, 0.01, 0.0]), dates
    )
    assert active.tolist() == [False, True, False]
    assert side.tolist() == [0, 1, 0]


def test_schedule_enters_after_completed_signal_bar_and_never_overlaps() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=300, freq="5min"))
    active = np.zeros(len(dates), dtype=bool)
    side = np.zeros(len(dates), dtype=np.int8)
    active[[0, 12, 145]] = True
    side[[0, 12, 145]] = [1, -1, -1]
    schedule = cage.nonoverlapping_schedule(dates, active, side, hold_hours=12)
    assert len(schedule) == 2
    assert pd.Timestamp(schedule.iloc[0]["feature_available"]) == dates.iloc[0] + pd.Timedelta("5min")
    assert schedule.iloc[0]["entry_time"] == schedule.iloc[0]["feature_available"]
    assert pd.Timestamp(schedule.iloc[1]["entry_time"]) >= pd.Timestamp(schedule.iloc[0]["exit_time"])


def test_schedule_requires_exit_inside_window() -> None:
    dates = pd.Series(pd.date_range("2022-12-31 20:00", periods=60, freq="5min"))
    active = np.zeros(len(dates), dtype=bool)
    side = np.zeros(len(dates), dtype=np.int8)
    active[0] = True
    side[0] = 1
    schedule = cage.nonoverlapping_schedule(
        dates,
        active,
        side,
        hold_hours=12,
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2023-01-01"),
    )
    assert schedule.empty


def test_support_gate_requires_both_sides_and_time_dispersion() -> None:
    rows = []
    for month in pd.period_range("2021-01", "2023-12", freq="M"):
        for day, side in ((5, 1), (20, -1)):
            entry = month.to_timestamp() + pd.Timedelta(days=day - 1)
            rows.append(
                {
                    "signal_bar_open": str(entry - pd.Timedelta("5min")),
                    "feature_available": str(entry),
                    "entry_time": str(entry),
                    "exit_time": str(entry + pd.Timedelta("12h")),
                    "side": side,
                }
            )
    summary = cage.support_summary(pd.DataFrame(rows))
    assert all(cage.support_gates(summary).values())
    one_sided = pd.DataFrame(rows).assign(side=1)
    assert not all(cage.support_gates(cage.support_summary(one_sided)).values())


def test_exclusive_writer_refuses_to_replace_freeze(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    cage.write_exclusive(output, {"value": 1})
    with pytest.raises(FileExistsError):
        cage.write_exclusive(output, {"value": 2})


def _minimal_market_csv(tmp_path, rows: int = 12):
    dates = pd.date_range("2020-01-01", periods=rows, freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(100.0, 101.0, rows),
            "quote_asset_volume": np.full(rows, 1_000.0),
            "taker_buy_quote": np.full(rows, 400.0),
        }
    )
    path = tmp_path / "support_only.csv"
    frame.to_csv(path, index=False)
    return path, dates


def test_loader_accepts_support_columns_without_open_high_low(tmp_path, monkeypatch) -> None:
    path, dates = _minimal_market_csv(tmp_path)
    monkeypatch.setattr(cage, "SELECTION_END", dates[-1] + pd.Timedelta("5min"))
    market, loaded_dates = cage.load_market(path)
    assert set(market) == {"date", "close", "quote_asset_volume", "taker_buy_quote"}
    assert loaded_dates.iloc[-1] == dates[-1]


def test_loader_rejects_truncated_sealed_interval(tmp_path, monkeypatch) -> None:
    path, dates = _minimal_market_csv(tmp_path)
    monkeypatch.setattr(cage, "SELECTION_END", dates[-1] + pd.Timedelta("10min"))
    with pytest.raises(ValueError, match="sealed 2020-2023 interval"):
        cage.load_market(path)


def test_implementation_path_is_not_cwd_sensitive(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cage.implementation_path() == "training/preregister_price_memory_cage_escape_alpha.py"


def test_real_loader_physically_seals_2024() -> None:
    _, dates = cage.load_market(cage.Config.input_csv)
    assert dates.min() == pd.Timestamp("2020-01-01")
    assert dates.max() == pd.Timestamp("2023-12-31 23:55")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()
