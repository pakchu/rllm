from __future__ import annotations

from pathlib import Path

import pandas as pd

from training import export_wikimedia_selection_market_prefix as prefix


def test_prefix_frame_stops_at_first_cutoff_chunk(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "date": [
                "2022-12-31 23:50",
                "2022-12-31 23:55",
                "2023-01-01 00:00",
                "2023-01-01 00:05",
            ],
            "open": [1.0, 1.0, 2.0, 2.0],
        }
    ).to_csv(path, index=False)
    frame = prefix.prefix_frame(
        path,
        date_column="date",
        cutoff="2023-01-01",
        usecols=["date", "open"],
    )
    assert frame["date"].tolist() == [
        pd.Timestamp("2022-12-31 23:50"),
        pd.Timestamp("2022-12-31 23:55"),
    ]


def test_market_validation_requires_complete_grid() -> None:
    good = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=3, freq="5min"),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
        }
    )
    assert len(prefix.validate_market(good)) == 3
