from __future__ import annotations

import numpy as np
import pandas as pd

from training import preregister_inverse_collateral_liquidation_absorption as icla


def _sources(rows: int = 4_500) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-06-25", periods=rows, freq="5min")
    liquidation = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time": dates + pd.Timedelta(minutes=5, seconds=1),
            "source_valid": np.ones(rows, dtype=bool),
            "event_count": np.ones(rows, dtype=int),
            "total_liquidation_usd": np.full(rows, 100.0),
            "signed_liquidation_usd": np.zeros(rows),
        }
    )
    activity = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time": dates + pd.Timedelta(minutes=5, seconds=1),
            "quote_asset_volume": np.full(rows, 1_000.0),
            "taker_buy_quote": np.full(rows, 500.0),
            "taker_sell_quote": np.full(rows, 500.0),
            "taker_imbalance": np.zeros(rows),
            "number_of_trades": np.full(rows, 100),
        }
    )
    return liquidation, activity


def _inject_absorbed_long_liquidation(
    liquidation: pd.DataFrame, activity: pd.DataFrame, end_index: int
) -> None:
    start = end_index - icla.WAVE_BARS + 1
    liquidation.loc[start:end_index, "total_liquidation_usd"] = 1_000.0
    liquidation.loc[start:end_index, "signed_liquidation_usd"] = -1_000.0
    activity.loc[start:end_index, "taker_buy_quote"] = 600.0
    activity.loc[start:end_index, "taker_sell_quote"] = 400.0
    activity.loc[start:end_index, "taker_imbalance"] = 0.2


def test_wave_threshold_is_strictly_prior() -> None:
    liquidation, activity = _sources()
    index = 4_200
    _inject_absorbed_long_liquidation(liquidation, activity, index)
    original = icla.derive_wave_state(liquidation, activity)

    changed = liquidation.copy()
    changed.loc[index, "total_liquidation_usd"] *= 1_000.0
    changed.loc[index, "signed_liquidation_usd"] *= 1_000.0
    mutated = icla.derive_wave_state(changed, activity)

    assert original.loc[index, "prior_wave_threshold_usd"] == mutated.loc[
        index, "prior_wave_threshold_usd"
    ]
    assert bool(original.loc[index, "is_candidate"])
    assert int(original.loc[index, "direction"]) == 1


def test_same_direction_usdm_flow_rejects_candidate() -> None:
    liquidation, activity = _sources()
    index = 4_200
    _inject_absorbed_long_liquidation(liquidation, activity, index)
    activity.loc[index - icla.WAVE_BARS + 1 : index, "taker_buy_quote"] = 400.0
    activity.loc[index - icla.WAVE_BARS + 1 : index, "taker_sell_quote"] = 600.0
    state = icla.derive_wave_state(liquidation, activity)
    assert not bool(state.loc[index, "is_candidate"])


def test_invalid_liquidation_bar_invalidates_wave() -> None:
    liquidation, activity = _sources()
    index = 4_200
    _inject_absorbed_long_liquidation(liquidation, activity, index)
    liquidation.loc[index - 3, "source_valid"] = False
    state = icla.derive_wave_state(liquidation, activity)
    assert not bool(state.loc[index, "wave_source_valid"])
    assert not bool(state.loc[index, "is_candidate"])


def test_build_clocks_enforces_latency_hold_and_nonoverlap(monkeypatch) -> None:
    monkeypatch.setattr(
        icla,
        "SPLITS",
        {"train": ("2023-06-25", "2023-06-27")},
    )
    dates = pd.date_range("2023-06-25", periods=576, freq="5min")
    state = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time": dates + pd.Timedelta(minutes=5, seconds=1),
            "is_candidate": np.zeros(len(dates), dtype=bool),
            "direction": np.ones(len(dates), dtype=int),
            "wave_event_count": np.ones(len(dates), dtype=int),
            "wave_total_liquidation_usd": np.full(len(dates), 1_000.0),
            "prior_wave_threshold_usd": np.full(len(dates), 500.0),
            "wave_liquidation_imbalance": np.full(len(dates), -1.0),
            "wave_quote_asset_volume": np.full(len(dates), 10_000.0),
            "wave_usdm_taker_imbalance": np.full(len(dates), 0.1),
            "absorption_alignment": np.full(len(dates), 0.1),
        }
    )
    state.loc[[20, 21, 40], "is_candidate"] = True
    clocks = icla.build_clocks(state)
    assert len(clocks) == 2
    assert clocks["wave_completed_time"].iloc[0] == dates[20] + pd.Timedelta(
        minutes=5
    )
    assert clocks["entry_time"].iloc[0] == dates[20] + pd.Timedelta(minutes=10)
    assert clocks["planned_exit_time"].iloc[0] - clocks["entry_time"].iloc[
        0
    ] == pd.Timedelta(hours=1)
    assert clocks["entry_time"].iloc[1] >= clocks["planned_exit_time"].iloc[0]
