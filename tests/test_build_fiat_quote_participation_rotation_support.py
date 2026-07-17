from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_fiat_quote_participation_rotation_support as support


def test_prior_midrank_is_strictly_lagged_and_uses_mid_ties() -> None:
    series = pd.Series([1.0, 2.0, 2.0, 4.0, 2.0])
    ranked = support.prior_midrank(series, 4)
    assert ranked.iloc[:4].isna().all()
    assert ranked.iloc[4] == pytest.approx((1.0 + 0.5 * 2.0) / 4.0)
    changed_current = series.copy()
    changed_current.iloc[4] = 100.0
    reranked = support.prior_midrank(changed_current, 4)
    assert reranked.iloc[4] == 1.0
    assert ranked.iloc[:4].equals(reranked.iloc[:4])


def _feature_frame() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=8, freq="1D")
    frame = pd.DataFrame(index=dates)
    for label in ("eur", "try", "brl"):
        frame[f"volume_share_rank_{label}"] = 0.8
        frame[f"ticket_share_rank_{label}"] = 0.6
        frame[f"participation_score_{label}"] = 0.7
        frame[f"relative_taker_pressure_{label}"] = 0.1
        frame[f"absolute_participation_{label}"] = 0.7
    frame["median_relative_taker_pressure"] = 0.1
    frame["reference_raw_participation"] = 0.5
    frame["reference_buy_odds"] = 0.1
    return frame


def test_primary_flag_combines_weak_participation_and_median_flow() -> None:
    features = _feature_frame()
    flags = support.build_flags(features, 0.65)
    assert flags["primary"].all()
    features.loc[features.index[2], "participation_score_eur"] = 0.1
    features.loc[features.index[2], "participation_score_try"] = 0.1
    features.loc[features.index[3], "relative_taker_pressure_eur"] = -1.0
    features.loc[features.index[3], "relative_taker_pressure_try"] = -1.0
    features.loc[features.index[3], "median_relative_taker_pressure"] = -1.0
    flags = support.build_flags(features, 0.65)
    assert not flags["primary"].iloc[2]
    assert not flags["primary"].iloc[3]
    assert flags["no_taker"].iloc[3]


def test_false_to_true_and_clock_reservation_are_nonoverlapping_and_delayed() -> None:
    index = pd.date_range("2022-01-01", periods=10, freq="1D")
    flag = pd.Series(
        [False, True, True, False, True, False, True, False, False, True],
        index=index,
    )
    signal_days = support.false_to_true_days(flag)
    assert signal_days.tolist() == [index[1], index[4], index[6], index[9]]
    clock = support.reserve_signal_days(
        signal_days,
        clock_name="primary",
        q=0.65,
        hold_bars=864,
        execution_delay_bars=1,
    )
    assert clock["source_signal_day"].tolist() == [index[1], index[4], index[9]]
    assert clock.iloc[0]["entry_time"] == index[2] + pd.Timedelta(minutes=5)
    assert clock.iloc[0]["exit_time"] == index[5] + pd.Timedelta(minutes=5)
    assert (clock["entry_time"].iloc[1:].to_numpy() >= clock["exit_time"].iloc[:-1].to_numpy()).all()


def test_split_clock_allows_prior_history_but_contains_signal_entry_exit() -> None:
    clock = pd.DataFrame(
        {
            "signal_day": pd.to_datetime(["2022-12-31", "2023-01-01", "2023-12-30"]),
            "entry_time": pd.to_datetime(
                ["2023-01-01 00:05", "2023-01-02 00:05", "2023-12-31 00:05"]
            ),
            "exit_time": pd.to_datetime(
                ["2023-01-04 00:05", "2023-01-05 00:05", "2024-01-03 00:05"]
            ),
        }
    )
    selected = support.split_clock(clock, support.YEAR_2023)
    assert selected["signal_day"].tolist() == [pd.Timestamp("2023-01-01")]


def test_support_builder_never_verifies_execution_or_funding_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    real_sha = support._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    monkeypatch.setattr(support, "_sha256", tracking_sha)
    result = support.build(
        output_path=str(tmp_path / "support.json"),
        clocks_path=str(tmp_path / "clocks.csv"),
    )
    assert result["outcome_sources_opened"] == []
    assert not any("kline_reference" in path for path in seen)
    assert not any("funding_marks" in path for path in seen)


def test_real_source_selects_highest_train_q_and_2023_cannot_fallback(
    tmp_path: Path,
) -> None:
    result = support.build(
        output_path=str(tmp_path / "support.json"),
        clocks_path=str(tmp_path / "clocks.csv"),
    )
    grid = result["train_grid_descending_q"]
    assert [item["q"] for item in grid] == [0.70, 0.65, 0.60, 0.55, 0.50]
    assert grid[0]["passed"] is False
    assert grid[1]["passed"] is True
    assert result["selected_q"] == 0.65
    assert result["selected_2023"]["q"] == 0.65
    assert result["support_passed"] is True
    assert result["advance_to_stage1_outcomes"] is True
    assert result["selected_train"]["primary"]["entries"] >= 40
    assert result["selected_2023"]["primary"]["entries"] >= 20
    clocks = pd.read_csv(tmp_path / "clocks.csv", parse_dates=["entry_time", "exit_time"])
    assert clocks["entry_time"].ge("2021-01-01").all()
    assert clocks["exit_time"].le("2024-01-01").all()


def test_written_support_artifacts_are_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    clocks = tmp_path / "clocks.csv"
    first = support.build(output_path=str(output), clocks_path=str(clocks))
    first_json = output.read_bytes()
    first_clocks = clocks.read_bytes()
    second = support.build(output_path=str(output), clocks_path=str(clocks))
    assert output.read_bytes() == first_json
    assert clocks.read_bytes() == first_clocks
    assert first == second
    replay = json.loads(first_json)
    assert replay["outcomes_opened"] is False
    assert replay["outcome_sources_opened"] == []
    assert np.isfinite(replay["selected_q"])
