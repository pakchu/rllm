from __future__ import annotations

import io
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import build_binance_cross_collateral_book_credibility_2023 as builder
from training import build_binance_cross_collateral_book_depth_2023 as base


def _archive(frame: pd.DataFrame) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("depth.csv", frame.to_csv(index=False))
    return payload.getvalue()


def _raw_snapshots(count: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot in range(count):
        timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(seconds=30 * snapshot)
        for level in base.PERCENTAGES:
            distance = abs(level)
            rows.append(
                {
                    "timestamp": timestamp,
                    "percentage": level,
                    "depth": 100.0 * distance + snapshot,
                    "notional": 1_000.0 * distance + snapshot,
                }
            )
    return pd.DataFrame(rows)


def test_credibility_aggregation_preserves_depth_and_adds_causal_path_stats() -> None:
    raw = base.read_archive(_archive(_raw_snapshots()))
    output = builder.aggregate_credibility(raw, builder.Config())
    assert output["date"].tolist() == [pd.Timestamp("2023-01-01")]
    assert output.loc[0, "depth_m1"] == np.median(np.arange(100.0, 110.0))

    log_values = np.log(np.arange(100.0, 110.0))
    expected_mad = np.median(np.abs(log_values - np.median(log_values)))
    assert output.loc[0, "log_mad_m1"] == pytest.approx(expected_mad)
    assert output.loc[0, "log_net_m1"] == pytest.approx(np.log(109.0 / 100.0))
    assert output.loc[0, "log_step_m1"] == pytest.approx(
        np.diff(log_values).mean()
    )


def test_credibility_aggregation_rejects_insufficient_snapshot_coverage() -> None:
    raw = base.read_archive(_archive(_raw_snapshots(7)))
    output = builder.aggregate_credibility(raw, builder.Config())
    assert output.empty
    assert "log_mad_m1" in output.columns
    assert "log_net_p5" in output.columns
    assert "log_step_m3" in output.columns


def test_process_day_records_missing_archive_without_fabrication() -> None:
    def missing_fetcher(url: str, *, retries: int, timeout: int) -> bytes:
        del url, retries, timeout
        raise FileNotFoundError

    result = builder.process_day(
        "um",
        "BTCUSDT",
        pd.Timestamp("2023-02-08").date(),
        builder.Config(),
        fetcher=missing_fetcher,
    )
    assert result["available"] is False
    assert result["frame"].empty


def test_builder_rejects_any_request_outside_calendar_2023() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), end="2024-01-02"))
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), start="2022-12-31"))


def test_base_replay_allows_only_csv_roundtrip_noise() -> None:
    frozen = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01"]),
            "depth": [17_305.4065],
            "source_complete": [True],
        }
    )
    roundtripped = frozen.copy()
    roundtripped.loc[0, "depth"] += 4e-12
    builder._assert_base_frame_equal(roundtripped, frozen)

    changed = frozen.copy()
    changed.loc[0, "depth"] += 1e-6
    with pytest.raises(AssertionError):
        builder._assert_base_frame_equal(changed, frozen)
