from __future__ import annotations

import pandas as pd
import pytest

from training import export_tail_arrival_selection_prefix as source


def _feature_row(date: str, *, p99: float = 90.0, maximum: float = 100.0) -> dict:
    return {
        "date": pd.Timestamp(date),
        "agg_trade_count": 100,
        "event_notional_mean": 10.0,
        "event_notional_std": 3.0,
        "event_notional_p50": 5.0,
        "event_notional_p90": 50.0,
        "event_notional_p99": p99,
        "event_notional_max": maximum,
        "interarrival_mean_ms": 10.0,
        "interarrival_std_ms": 5.0,
        "buy_sell_event_size_log_ratio": 0.4,
        "micro_log_return": 0.001,
    }


def test_normalize_features_retains_missing_grid_rows_without_imputation() -> None:
    grid = pd.date_range("2020-01-01", periods=3, freq="5min")
    raw = pd.DataFrame([_feature_row(str(grid[0])), _feature_row(str(grid[2]))])
    frame = source.normalize_features(raw, grid, {"2020-01-01"})
    assert frame["source_complete"].tolist() == [1, 0, 1]
    assert frame["source_gap_day"].tolist() == [1, 1, 1]
    assert pd.isna(frame.loc[1, "event_notional_p99"])


def test_normalize_features_rejects_incoherent_tail_quantiles() -> None:
    grid = pd.date_range("2020-01-01", periods=1, freq="5min")
    raw = pd.DataFrame([_feature_row(str(grid[0]), p99=110.0, maximum=100.0)])
    with pytest.raises(ValueError, match="quantiles"):
        source.normalize_features(raw, grid, set())


def test_gap_day_parser_detects_internal_archive_id_loss() -> None:
    manifest = {
        "months": [
            {
                "archives": [
                    {
                        "date": "2020-01-01",
                        "first_agg_trade_id": 10,
                        "last_agg_trade_id": 20,
                        "agg_trade_rows": 10,
                    },
                    {
                        "date": "2020-01-02",
                        "first_agg_trade_id": 21,
                        "last_agg_trade_id": 30,
                        "agg_trade_rows": 10,
                    },
                ]
            }
        ]
    }
    assert source.source_gap_days(manifest, "2023-01-01") == {"2020-01-01"}


def test_expected_grid_is_half_open_before_2023() -> None:
    grid = source.expected_grid("2022-12-31", "2023-01-01")
    assert len(grid) == 288
    assert grid.max() == pd.Timestamp("2022-12-31 23:55:00")
