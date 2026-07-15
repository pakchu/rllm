from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import build_binance_cross_collateral_book_depth_2023 as base
from training import build_binance_cross_collateral_near_pressure_2024_2026 as builder


def _raw_snapshots(count: int = 9) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot in range(count):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=30 * snapshot)
        for level in base.PERCENTAGES:
            distance = abs(level)
            if level < 0:
                cumulative = 10.0 * distance + snapshot * (3.0 if distance == 1 else 4.0)
            else:
                cumulative = 12.0 * distance + snapshot * (1.0 if distance == 1 else 2.0)
            rows.append(
                {
                    "timestamp": timestamp,
                    "percentage": level,
                    "depth": cumulative,
                    "notional": cumulative * 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_near_pressure_matches_shell_net_formula() -> None:
    raw = _raw_snapshots()
    shell_panel = builder.shells.aggregate_shells(raw, builder.Config())
    output = builder.aggregate_near_pressure(raw, builder.Config())
    expected = (
        shell_panel.loc[0, "shell_flow_net_m1"]
        + 0.5 * shell_panel.loc[0, "shell_flow_net_m2"]
        - shell_panel.loc[0, "shell_flow_net_p1"]
        - 0.5 * shell_panel.loc[0, "shell_flow_net_p2"]
    )
    assert len(output) == 1
    assert output.loc[0, "near_pressure"] == pytest.approx(expected)
    assert np.isfinite(output.loc[0, "near_pressure"])


def test_insufficient_snapshots_produce_no_fabricated_bar() -> None:
    output = builder.aggregate_near_pressure(_raw_snapshots(7), builder.Config())
    assert output.empty
    assert output.columns.tolist() == [
        "date",
        "near_pressure",
        "snapshot_count",
        "first_offset_seconds",
        "last_offset_seconds",
    ]


def test_missing_archive_is_recorded_without_fabrication() -> None:
    def missing(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError

    result = builder.process_day(
        "um", "BTCUSDT", pd.Timestamp("2024-01-01").date(), builder.Config(), fetcher=missing
    )
    assert result["available"] is False
    assert result["frame"].empty


def test_future_builder_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        builder.validate_config(replace(builder.Config(), start="2023-12-31"))
    with pytest.raises(ValueError, match="physically bounded"):
        builder.validate_config(replace(builder.Config(), end="2026-06-03"))
    with pytest.raises(ValueError, match="minimum snapshots"):
        builder.validate_config(replace(builder.Config(), minimum_snapshots_per_bar=0))
