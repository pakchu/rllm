from __future__ import annotations

import pandas as pd
import pytest

from training import build_post_funding_crowding_release_episode_v2_support as pfcr2


def parent_clock(hours: list[int]) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2023-01-01 00:00:00")
    for hour in hours:
        settlement = base + pd.Timedelta(hours=hour)
        rows.append(
            {
                "policy_id": pfcr2.PARENT_POLICY_ID,
                "settlement_time": settlement,
                "feature_available_time": settlement + pd.Timedelta(minutes=5),
                "entry_time": settlement + pd.Timedelta(minutes=10),
                "exit_time": settlement + pd.Timedelta(hours=4, minutes=10),
                "long_symbol": "ETHUSDT",
                "short_symbol": "SOLUSDT",
                "long_weight": 0.5,
                "short_weight_abs": 0.5,
                "long_beta": 1.0,
                "short_beta": 1.0,
                "current_funding_spread": 0.002,
                "prior_spread_q90": 0.001,
                "long_current_funding_rate": -0.001,
                "short_current_funding_rate": 0.001,
            }
        )
    return pd.DataFrame(rows, columns=pfcr2.CLOCK_COLUMNS)


def test_episode_onset_anchors_cooldown_to_prior_accepted_event() -> None:
    clock = pfcr2.episode_onset_clock(parent_clock([0, 8, 32, 36, 40, 72]))
    expected = pd.to_datetime(["2023-01-01 00:00", "2023-01-02 12:00", "2023-01-04 00:00"])
    assert clock["settlement_time"].tolist() == expected.tolist()
    assert clock["policy_id"].eq(pfcr2.POLICY_ID).all()
    pfcr2.assert_clock_contract(clock)


def test_event_exactly_at_36_hours_is_accepted() -> None:
    clock = pfcr2.episode_onset_clock(parent_clock([0, 35, 36]))
    assert len(clock) == 2
    assert clock["settlement_time"].diff().dropna().iloc[0] == pd.Timedelta(hours=36)


def test_clock_contract_rejects_shortened_cooldown() -> None:
    clock = pfcr2.episode_onset_clock(parent_clock([0, 36]))
    clock.loc[1, "settlement_time"] = clock.loc[0, "settlement_time"] + pd.Timedelta(hours=8)
    clock.loc[1, "feature_available_time"] = clock.loc[1, "settlement_time"] + pd.Timedelta(minutes=5)
    clock.loc[1, "entry_time"] = clock.loc[1, "settlement_time"] + pd.Timedelta(minutes=10)
    clock.loc[1, "exit_time"] = clock.loc[1, "entry_time"] + pd.Timedelta(hours=4)
    with pytest.raises(RuntimeError, match="episode cooldown"):
        pfcr2.assert_clock_contract(clock)


def test_empty_parent_clock_stays_schema_complete() -> None:
    clock = pfcr2.episode_onset_clock(pd.DataFrame(columns=pfcr2.CLOCK_COLUMNS))
    assert clock.empty
    assert tuple(clock.columns) == pfcr2.CLOCK_COLUMNS
