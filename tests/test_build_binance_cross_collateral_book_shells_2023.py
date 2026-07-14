from __future__ import annotations

import io
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import build_binance_cross_collateral_book_depth_2023 as base
from training import build_binance_cross_collateral_book_shells_2023 as builder


def _raw_snapshots(count: int = 9) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2023-01-01 00:00:00")
    for snapshot in range(count):
        timestamp = start + pd.Timedelta(seconds=30 * snapshot)
        for level in base.PERCENTAGES:
            distance = abs(level)
            side_scale = 1.0 if level < 0 else 1.2
            cumulative = side_scale * (10.0 * distance + snapshot)
            rows.append(
                {
                    "timestamp": timestamp,
                    "percentage": level,
                    "depth": cumulative,
                    "notional": cumulative * 100.0,
                }
            )
    return pd.DataFrame(rows)


def _archive(frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("book.csv", frame.to_csv(index=False))
    return output.getvalue()


def test_snapshot_shells_are_nonoverlapping_and_sum_to_total() -> None:
    raw = _raw_snapshots(1)
    output = builder._snapshot_shells(raw, "p")
    assert output.loc[0, [f"mass_{shell}" for shell in range(1, 6)]].tolist() == [
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
    ]
    assert output.loc[0, "total"] == pytest.approx(60.0)
    assert output.loc[0, [f"share_{shell}" for shell in range(1, 6)]].sum() == (
        pytest.approx(1.0)
    )


def test_snapshot_shells_allow_a_zero_mass_annulus() -> None:
    raw = _raw_snapshots(1)
    for side_levels in ((-1, -2), (1, 2)):
        first = raw.loc[raw["percentage"].eq(side_levels[0]), "depth"].iloc[0]
        raw.loc[raw["percentage"].eq(side_levels[1]), "depth"] = first
    output = builder._snapshot_shells(raw, "m")
    assert output.loc[0, "mass_2"] == 0.0
    assert output.loc[0, "share_2"] == 0.0
    assert np.isfinite(output.to_numpy()[:, 2:].astype(float)).all()


def test_side_aggregation_has_dimensionless_exact_path_statistics() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01 00:00:00", "2023-01-01 00:00:30"]
            ),
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "total": [100.0, 100.0],
            "mass_1": [10.0, 20.0],
            "mass_2": [20.0, 20.0],
            "mass_3": [20.0, 20.0],
            "mass_4": [20.0, 20.0],
            "mass_5": [30.0, 20.0],
            "share_1": [0.1, 0.2],
            "share_2": [0.2, 0.2],
            "share_3": [0.2, 0.2],
            "share_4": [0.2, 0.2],
            "share_5": [0.3, 0.2],
        }
    )
    output = builder._aggregate_side_shells(snapshots, "m").iloc[0]
    assert output["shell_share_median_m1"] == pytest.approx(0.15)
    assert output["shell_flow_net_m1"] == pytest.approx(0.10)
    assert output["shell_flow_add_m1"] == pytest.approx(0.10)
    assert output["shell_flow_withdraw_m1"] == 0.0
    assert output["shell_flow_churn_m1"] == pytest.approx(0.10)
    assert output["shell_flow_efficiency_m1"] == pytest.approx(1.0)


def test_full_aggregation_reuses_frozen_bar_acceptance() -> None:
    raw = _raw_snapshots(9)
    output = builder.aggregate_shells(raw, builder.Config())
    assert len(output) == 1
    shell_columns = [column for column in output if "shell_" in column]
    assert len(shell_columns) == 60
    assert np.isfinite(output[shell_columns].to_numpy(float)).all()
    assert output["snapshot_count"].iloc[0] == 9


def test_shell_aggregation_rejects_insufficient_snapshot_coverage() -> None:
    output = builder.aggregate_shells(_raw_snapshots(7), builder.Config())
    assert output.empty
    assert "shell_share_median_m1" in output.columns
    assert "shell_flow_efficiency_p5" in output.columns


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


def test_archive_parser_and_shell_builder_preserve_monotonic_contract() -> None:
    raw = base.read_archive(_archive(_raw_snapshots(9)))
    output = builder.aggregate_shells(raw, builder.Config())
    assert len(output) == 1


def test_builder_rejects_any_request_outside_calendar_2023() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), end="2024-01-02"))
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), start="2022-12-31"))
    with pytest.raises(ValueError, match="minimum snapshots"):
        builder.build(replace(builder.Config(), minimum_snapshots_per_bar=0))


def test_unknown_shell_side_fails_closed() -> None:
    with pytest.raises(ValueError, match="side must be"):
        builder._snapshot_shells(_raw_snapshots(1), "x")
