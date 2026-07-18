from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_cross_venue_vol_disagreement_features as builder


def _write_sources(tmp_path: Path) -> builder.Config:
    hours = pd.date_range("2023-06-20", periods=8, freq="1h")
    bvol = pd.DataFrame(
        {
            "date": hours,
            "feature_available_time_utc": hours + pd.Timedelta("1h"),
            "trade_earliest_time_utc": hours + pd.Timedelta("1h"),
            "close": np.linspace(50.0, 57.0, len(hours)),
            "feature_valid": True,
            "feature_invalid_reason": "ok",
        }
    )
    dvol = pd.DataFrame(
        {
            "close_time": hours + pd.Timedelta("1h"),
            "close": np.linspace(45.0, 52.0, len(hours)),
        }
    )
    bars = pd.date_range("2023-06-19 20:00:00", "2023-06-20 08:00:00", freq="5min", inclusive="left")
    market = pd.DataFrame({"date": bars, "close": np.linspace(25_000.0, 26_000.0, len(bars))})
    paths = [tmp_path / name for name in ("bvol.csv", "dvol.csv", "market.csv")]
    bvol.to_csv(paths[0], index=False)
    dvol.to_csv(paths[1], index=False)
    market.to_csv(paths[2], index=False)
    return builder.Config(
        bvol_csv=str(paths[0]),
        dvol_csv=str(paths[1]),
        market_csv=str(paths[2]),
        output_dir=str(tmp_path / "out"),
    )


def test_exact_close_time_join_and_backward_four_hour_return(tmp_path: Path) -> None:
    cfg = _write_sources(tmp_path)
    output = builder.build_frame(cfg)

    row = output.loc[output["signal_time_utc"].eq(pd.Timestamp("2023-06-20 05:00:00"))].iloc[0]
    assert row["feature_available_time_utc"] == row["signal_time_utc"]
    assert row["trade_earliest_time_utc"] == row["signal_time_utc"] + pd.Timedelta("5min")
    assert row["log_bvol_dvol_ratio"] == pytest.approx(np.log(row["bvol_close"] / row["dvol_close"]))
    assert row["btc_return_4h"] > 0.0
    assert bool(row["feature_valid"])


def test_bvol_availability_mismatch_fails_closed(tmp_path: Path) -> None:
    cfg = _write_sources(tmp_path)
    bvol = pd.read_csv(cfg.bvol_csv)
    bvol.loc[0, "feature_available_time_utc"] = "2023-06-20 02:00:00"
    bvol.to_csv(cfg.bvol_csv, index=False)

    with pytest.raises(ValueError, match="availability"):
        builder.build_frame(cfg)


def test_builder_refuses_post2023_cutoff(tmp_path: Path) -> None:
    cfg = _write_sources(tmp_path)
    cfg = builder.Config(**{**cfg.__dict__, "cutoff": "2024-01-02"})
    with pytest.raises(ValueError, match="sealed"):
        builder.build_frame(cfg)
