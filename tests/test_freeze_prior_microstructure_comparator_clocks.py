from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import freeze_prior_microstructure_comparator_clocks as freeze


def _clock(dates: list[str], sides: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"signal_date": dates, "side": sides})


def test_prior_schedule_matches_no_stop_boundary_reservation() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=12, freq="5min"))
    long_active = np.zeros(len(dates), dtype=bool)
    short_active = np.zeros(len(dates), dtype=bool)
    # hold=2: signal 0 enters 1/exits 3, then the prior simulator reserves
    # signal positions through 3 and accepts the first signal at 4.
    long_active[[0, 2, 3, 4]] = True
    short_active[8] = True
    clock = freeze.prior_no_stop_schedule(
        dates,
        long_active,
        short_active,
        hold_bars=2,
        start="2020-01-01",
        end="2020-01-01 01:00:00",
    )
    assert clock["signal_date"].tolist() == [dates.iloc[0], dates.iloc[4], dates.iloc[8]]
    assert clock["side"].tolist() == [1, 1, -1]


def test_union_clock_deduplicates_timestamps_and_reports_side_conflict() -> None:
    first = _clock(
        ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        [1, 1],
    )
    second = _clock(
        ["2023-01-01 00:00:00", "2023-01-01 02:00:00"],
        [-1, -1],
    )
    union, conflicts = freeze.union_clock([first, second])
    assert len(union) == 3
    assert conflicts == 1
    assert not union["signal_date"].duplicated().any()


def test_signal_market_loader_discards_2024_and_unused_columns(tmp_path: Path) -> None:
    path = tmp_path / "market.csv.gz"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-12-31 23:50:00", periods=5, freq="5min"),
            "close": np.arange(5) + 100.0,
            "future_return": np.arange(5),
        }
    )
    frame.to_csv(path, index=False, compression="gzip")
    loaded, metadata = freeze._load_signal_market(path, columns=("close",))
    assert loaded["date"].max() < pd.Timestamp("2024-01-01")
    assert list(loaded.columns) == ["date", "close"]
    assert metadata["columns_loaded"] == ["date", "close"]
    assert metadata["future_outcome_transform_applied"] is False


def test_frozen_artifact_hash_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="frozen value"):
        freeze._require_sha256(path, "0" * 64, label="fixture")
